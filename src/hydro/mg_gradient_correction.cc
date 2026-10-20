#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <climits>
#include <float.h>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <Kokkos_Core.hpp>
#include "../declarations/gpu_all_mirror.h"  /* MUST precede allvars.h: device-side All reads go to this file's mirror */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../system/gpu_particles_arena.h"
#include "../mesh/kernel.h"
#include "../mesh/neighbor_list.h"
#include "../mesh/sfc_tiles.h"
#include "../mesh/ghost_writeback.h"
#include "compute_finitevol_faces_functions.h"
#include "hydro_corridor.h"

/*! \file mg_gradient_correction.cc
 *  \brief Modified-gradient (MG) method for exact div(B)=0 correction.
 *
 *  Implements Tu, Wang, Gao & Tang (2026), arXiv:2603.04077. The discrete divergence
 *  of cell i is the face sum b_i = sum_j (1/2)(B'_i + B'_j).A_ij, with B' the linear
 *  extrapolation of the (once-limited) cell gradients to the face midpoint. Shifting
 *  both face states by (c_j - c_i) kappa_ij A_ij (along the face area vector only)
 *  changes that face's term by (c_j - c_i) kappa_ij |A_ij|^2, so b = 0 on every cell is
 *  the graph-Laplacian system  R c = b  over all gas cells. kappa = r^2/8 (Tu et al.), or
 *  under MHD_MODIFIED_GRADIENT_RELATIVE_FIELD_WEIGHT the normalised pair field scale
 *  squared plus a floor (mg_face_shift_factor in core/proto.h, the one definition every consumer uses).
 *
 *  This module:
 *    1) assembles R and b over every local gas cell (the graph must be complete: on
 *       an MG step the hydro corridor imports every gas cell's neighbourhood);
 *    2) checks R is symmetric, labels its connected pieces (each has one constant null
 *       mode), checks b sums to roundoff on every piece, and skips pieces whose b is
 *       already at roundoff relative to their face-flux scale;
 *    3) fixes one unknown per piece at zero and solves the now definite system. The
 *       fixed (anchor) row is the one with the largest diagonal: the coefficients can span
 *       tens of orders of magnitude, and an anchor among small-diagonal rows gives the
 *       large-diagonal rows a common offset in c far above their face differences, which
 *       then lose precision. c is used only through differences across faces, so the
 *       choice of anchor never enters the physics;
 *    4) checks the true residual of R c = b: a solve that misses the tolerance is
 *       still applied if it lowers the residual (less accuracy, not a broken state);
 *       otherwise the coefficients and pieces of the last accepted solve are kept and
 *       the step proceeds as one without the solve. Either way a warning is printed.
 *  c_i is stored in CellP[i].MG_cgcoeff (with its piece in CellP[i].MG_piece_token) for the
 *  face reconstruction in hydro and for the constrained-gradient correction on steps
 *  without the solve.
 *
 *  Where the work runs: everything that walks the faces (the coefficients, the symmetry
 *  check, the solver's matrix arrays, the true residual) runs in parallel on the device
 *  over the physical graph -- every kept face of every local row, sorted by global column.
 *  The piece labelling, the per-piece sums and the exchanges with other tasks run on the
 *  host, reading per-row values and the cross-task faces only. The face loops are one
 *  thread per row: an MG step spans every gas cell, so there are always as many rows as
 *  cells (about a million per task in a large run) against a few tens of faces per row,
 *  and a row walked in order by one thread sums exactly as the serial loop does.
 */
/*
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO,
 * based on the method described in Tu, Wang, Gao & Tang (2026), arXiv:2603.04077.
 */

#ifdef MHD_MODIFIED_GRADIENT

#include "../mesh/gpu_neighbor_list.h"   /* gpu_build_symmetric_neighbor_list, gizmo_sym_neighbor_list_device */

#if !defined(MHD_MODIFIED_GRADIENT_CG_ONLY) && !defined(MHD_MODIFIED_GRADIENT_USE_PARDISO)
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
#define MG_HYPRE_ON_DEVICE   /* a GPU-built HYPRE takes every array it is handed in device memory */
#endif
#endif

#ifdef MHD_MODIFIED_GRADIENT_USE_PARDISO
#include <mkl.h>
#include <mkl_pardiso.h>
#endif

#define MG_CG_MAX_ITER 500
#define MG_CG_TOL 1.0e-10  /* relative-residual tolerance of the solve: tighter values sit below the roundoff
                              floor of the divB-flux right-hand side, so converged solves would never meet them */
#define MG_NOISE_FLOOR 1.0e-10                /* a piece whose ||b|| is below this fraction of its face-flux scale ||s|| is already divergence-free to roundoff: c = 0 there, no solve */
#define MG_RHS_IMBALANCE_STRUCTURAL 1.0e-6    /* |sum b| / sum s on a piece beyond this is not roundoff: the graph or the face fluxes are not antisymmetric */
#define MG_SYMMETRY_TOL 1.0e-12               /* both rows of a face compute its coefficient from the same expression */
#define MG_REPLICATED_PIECE_GRAPH_MAX_BYTES (64LL << 20) /* above this the cross-rank piece graph is resolved by label propagation instead of being gathered everywhere */

/* ==================================================================================== */
/* Storage                                                                                */
/* ==================================================================================== */

struct mg_row_data {
    double Rdiag;        /* sum_j Qnorm2_ij over all faces */
    double rhs;          /* b_i: face sum of (1/2)(B'_i+B'_j).A_ij */
    double flux_scale;   /* s_i: face sum of (1/2)(|B'_i|+|B'_j|)|A_ij|, the scale b_i is roundoff against */
    long long piece;     /* connected piece label (identity, and the task that reduces the piece): its smallest global row id */
    long long anchor;    /* the piece's anchor row (global id), whose unknown is fixed at zero */
};

static struct mg_row_data *MG_Rows = NULL;
static std::vector<long long> MG_TaskRowOffsets;  /* global row id of task t's row 0 */
static std::vector<char> MG_RowSolved;            /* 1 = unknown solved for; 0 = c fixed at 0 (a piece's anchor, an isolated cell, or a piece below the noise floor) */
static std::vector<double> MG_RhsSolved;          /* b on solved rows, 0 elsewhere */
static std::vector<char> MG_RowInSolvedPiece;     /* 1 = the row's piece is above the noise floor (its residual is judged) */

/* The physical graph, on the device: for each local row i, its kept faces in
   [MGd_off[i], MGd_off[i+1]), as (global column, Qnorm2) sorted by column. Kept from
   assembly through the residual check; every solver derives its equations from it. */
static long long *MGd_off = NULL, *MGd_col = NULL;
static double *MGd_q = NULL;
static long long MG_GraphEdges = 0;
/* Each row's cross-task faces, on the host, in the physical graph's order: what the
   labelling, the symmetry check across tasks and the exchanges need. */
static std::vector<long long> MG_RemoteOff, MG_RemoteCol;
static std::vector<double> MG_RemoteQ;
#if defined(MHD_MODIFIED_GRADIENT_CG_ONLY) || defined(MHD_MODIFIED_GRADIENT_USE_PARDISO)
/* host copy of the physical graph for the solvers that run on the host */
static std::vector<long long> MG_HostOff, MG_HostCol;
static std::vector<double> MG_HostQ;
#endif

static inline long long mg_global_row(int task, int index) {return MG_TaskRowOffsets[task] + (long long)index;}
static inline int mg_row_owner(long long global_row)
{
    return (int)(std::upper_bound(MG_TaskRowOffsets.begin(), MG_TaskRowOffsets.end(), global_row) - MG_TaskRowOffsets.begin()) - 1;
}
/* the anchor unknown of a piece is fixed at zero, so its column drops out of the other rows */
static inline bool mg_column_kept(int i, long long global_col) {return global_col != MG_Rows[i].anchor;}

/* Device allocations report exhaustion by returning NULL; the flag is checked collectively. The
   bytes asked for in the current phase are kept for the warning when one fails. */
static int MG_DeviceAllocFailed = 0;
static double MG_DeviceBytesRequested = 0;
template<class T> static T *mg_device_alloc(size_t n, const char *label)
{
    MG_DeviceBytesRequested += (double) DMAX(n, (size_t)1) * sizeof(T);
    T *p = (T *) gizmo_gpu_alloc_device(DMAX(n, (size_t)1) * sizeof(T), label);
    if(!p) {MG_DeviceAllocFailed = 1;}
    return p;
}
template<class T> static void mg_device_free(T *&p) {gizmo_gpu_free_device((void *) p); p = NULL;}

/* first index in [a, b) whose column is not below `key` */
KOKKOS_INLINE_FUNCTION long long mg_lower_bound(const long long *col, long long a, long long b, long long key)
{
    while(a < b) {long long m = a + (b - a) / 2; if(col[m] < key) {a = m + 1;} else {b = m;}}
    return a;
}

/* Each row of the physical graph holds its faces to cells on this task first, then those to
   other tasks, each part ascending by global column: the one order every sum over a row's faces
   runs in. */
KOKKOS_INLINE_FUNCTION bool mg_column_is_local(long long c, long long first_local, long long end_local) {return (c >= first_local) && (c < end_local);}
KOKKOS_INLINE_FUNCTION bool mg_face_precedes(long long c1, long long c2, long long first_local, long long end_local)
{
    bool l1 = mg_column_is_local(c1, first_local, end_local), l2 = mg_column_is_local(c2, first_local, end_local);
    return (l1 != l2) ? l1 : (c1 < c2);
}
/* the end of a row's faces to cells on this task: its first other-task face, or b */
KOKKOS_INLINE_FUNCTION long long mg_local_end(const long long *col, long long a, long long b, long long first_local, long long end_local)
{
    while(a < b) {long long m = a + (b - a) / 2; if(mg_column_is_local(col[m], first_local, end_local)) {a = m + 1;} else {b = m;}}
    return a;
}
/* A row's faces put into summation order (this task's columns first, each part ascending), each coefficient moving with its
   column. Rows hold a few tens of faces, where insertion sort is fastest; a cell with thousands of faces would cost k^2 steps
   on its one thread and hold every task at the next collective, so a long row is heapsorted in place instead (k log k, no
   recursion, no scratch) and needs no second level of parallelism. Columns are unique within a row: both give the same order. */
#define MG_ROW_HEAPSORT_MIN_FACES 64
KOKKOS_INLINE_FUNCTION void mg_row_sift_down(long long *col, double *q, long long root, long long n, long long first_local, long long end_local)
{
    while(2 * root + 1 < n) {
        long long child = 2 * root + 1;
        if(child + 1 < n && mg_face_precedes(col[child], col[child + 1], first_local, end_local)) {child++;}
        if(!mg_face_precedes(col[root], col[child], first_local, end_local)) {return;}
        long long tc = col[root]; col[root] = col[child]; col[child] = tc;
        double tq = q[root]; q[root] = q[child]; q[child] = tq;
        root = child;
    }
}
KOKKOS_INLINE_FUNCTION void mg_sort_row(long long *col, double *q, long long n, long long first_local, long long end_local)
{
    if(n < MG_ROW_HEAPSORT_MIN_FACES) {
        for(long long a = 1; a < n; a++) {
            long long c = col[a]; double qa = q[a]; long long b = a - 1;
            while(b >= 0 && mg_face_precedes(c, col[b], first_local, end_local)) {col[b+1] = col[b]; q[b+1] = q[b]; b--;}
            col[b+1] = c; q[b+1] = qa;
        }
        return;
    }
    for(long long root = n / 2 - 1; root >= 0; root--) {mg_row_sift_down(col, q, root, n, first_local, end_local);}
    for(long long last = n - 1; last > 0; last--) {
        long long tc = col[0]; col[0] = col[last]; col[last] = tc;
        double tq = q[0]; q[0] = q[last]; q[last] = tq;
        mg_row_sift_down(col, q, 0, last, first_local, end_local);
    }
}


/* ==================================================================================== */
/* Values of remote neighbours: each task fetches, for every other-task row its faces    */
/* reach, a value its owner holds for that row (piece labels, solution components).      */
/* ==================================================================================== */

static int *mg_xch_send_counts = NULL, *mg_xch_recv_counts = NULL, *mg_xch_send_offsets = NULL, *mg_xch_recv_offsets = NULL;
static int *mg_xch_send_indices = NULL;   /* local rows other tasks asked for, grouped by task */
static int mg_xch_n_recv = 0, mg_xch_n_send = 0;
static std::vector<long long> MG_RequestCol;    /* every other-task row this task's faces reach, ascending (so grouped by task) */
static std::vector<double> MG_ExchangeValues;   /* received values, one per MG_RequestCol entry */
static std::vector<int> MG_RemoteSlot;          /* each cross-task face's place in MG_ExchangeValues */

static inline int mg_request_slot(long long global_col)
{
    return (int)(std::lower_bound(MG_RequestCol.begin(), MG_RequestCol.end(), global_col) - MG_RequestCol.begin());
}

static void mg_setup_exchange(void)
{
    MG_RequestCol.assign(MG_RemoteCol.begin(), MG_RemoteCol.end());
    std::sort(MG_RequestCol.begin(), MG_RequestCol.end());
    MG_RequestCol.erase(std::unique(MG_RequestCol.begin(), MG_RequestCol.end()), MG_RequestCol.end());
    mg_xch_send_counts = (int *) calloc(NTask, sizeof(int)); mg_xch_recv_counts = (int *) calloc(NTask, sizeof(int));
    mg_xch_send_offsets = (int *) calloc(NTask, sizeof(int)); mg_xch_recv_offsets = (int *) calloc(NTask, sizeof(int));
    std::vector<int> requested((size_t)DMAX(MG_RequestCol.size(), (size_t)1));
    for(size_t k = 0; k < MG_RequestCol.size(); k++) {
        int owner = mg_row_owner(MG_RequestCol[k]);
        mg_xch_recv_counts[owner]++;
        requested[k] = (int)(MG_RequestCol[k] - MG_TaskRowOffsets[owner]);
    }
    MPI_Alltoall(mg_xch_recv_counts, 1, MPI_INT, mg_xch_send_counts, 1, MPI_INT, MPI_COMM_WORLD);
    mg_xch_n_recv = mg_xch_n_send = 0;
    for(int t = 0; t < NTask; t++) {
        mg_xch_recv_offsets[t] = mg_xch_n_recv; mg_xch_send_offsets[t] = mg_xch_n_send;
        mg_xch_n_recv += mg_xch_recv_counts[t]; mg_xch_n_send += mg_xch_send_counts[t];
    }
    mg_xch_send_indices = (int *) malloc((size_t)DMAX(mg_xch_n_send, 1) * sizeof(int));
    MPI_Alltoallv(requested.data(), mg_xch_recv_counts, mg_xch_recv_offsets, MPI_INT,
                  mg_xch_send_indices, mg_xch_send_counts, mg_xch_send_offsets, MPI_INT, MPI_COMM_WORLD);
    MG_ExchangeValues.assign((size_t)DMAX(mg_xch_n_recv, 1), 0.0);
    MG_RemoteSlot.resize(MG_RemoteCol.size());
    for(size_t e = 0; e < MG_RemoteCol.size(); e++) {MG_RemoteSlot[e] = mg_request_slot(MG_RemoteCol[e]);}
}

static void mg_exchange_values(const double *local_values)
{
    std::vector<double> send((size_t)DMAX(mg_xch_n_send, 1));
    for(int k = 0; k < mg_xch_n_send; k++) {send[k] = local_values[mg_xch_send_indices[k]];}
    MPI_Alltoallv(send.data(), mg_xch_send_counts, mg_xch_send_offsets, MPI_DOUBLE,
                  MG_ExchangeValues.data(), mg_xch_recv_counts, mg_xch_recv_offsets, MPI_DOUBLE, MPI_COMM_WORLD);
}

static void mg_cleanup_exchange(void)
{
    free(mg_xch_send_indices); mg_xch_send_indices = NULL;
    free(mg_xch_recv_offsets); free(mg_xch_send_offsets); free(mg_xch_recv_counts); free(mg_xch_send_counts);
    mg_xch_recv_offsets = mg_xch_send_offsets = mg_xch_recv_counts = mg_xch_send_counts = NULL;
    MG_ExchangeValues.clear(); MG_ExchangeValues.shrink_to_fit();
    MG_RequestCol.clear(); MG_RequestCol.shrink_to_fit();
    MG_RemoteSlot.clear(); MG_RemoteSlot.shrink_to_fit();
}


/* ==================================================================================== */
/* Assembly over a symmetric CSR whose rows cover every local gas cell                   */
/* ==================================================================================== */

/* the CSR the graph is built from: host arrays (read by the labelling) and their device copy */
struct mg_csr {
    const int *rows; const int64_t *offsets; const int *neighbors; int n_rows;   /* host */
    const int *d_rows; const int64_t *d_offsets; const int *d_neighbors;          /* device */
};

/* the kernel values compute_finitevol_faces reads for one face */
struct mg_face_kernel { Vec3<double> dp; double r, wk_i, wk_j, dwk_i, dwk_j, h_i; };

/* Builds the physical graph (MGd_*) and the per-row values on the device, copies the per-row
   values into MG_Rows, the kept flag of every CSR entry into `kept` (for the labelling, which walks
   the host CSR), and each row's cross-task faces into MG_Remote*. Returns 0 if device memory
   could not be had on some task (collective). */
/* The face-weight normalisation this solve's coefficients use (MHD_MODIFIED_GRADIENT_RELATIVE_FIELD_WEIGHT): 1 / mean |B|^2 over the
   cells of the solve (0 if the field is zero everywhere). It becomes All.MG_RelativeFieldInvMeanB2 only when the solve is accepted
   (mg_commit_solution), together with the coefficients it belongs to; a rejected solve leaves the previous pair in place. */
static double MG_CandidateInvMeanB2 = 0;

/* whether a cell is a row of the MG graph: the coefficients, the field normalisation and the commit all use this one test */
GIZMO_GPU_FUNCTION static inline bool mg_cell_in_graph(const struct particle_data &p, const struct gas_cell_data &c)
{
    return (p.Type == 0) && (p.Mass > 0) && (c.Density > 0) && (p.KernelRadius > 0);
}

static int mg_build_graph(const struct mg_csr *csr, std::vector<unsigned char> &kept)
{
    const int n_rows = csr->n_rows, n_gas = N_gas;
    const long long n_cand = (n_rows > 0) ? (long long) csr->offsets[n_rows] : 0;
    const int num_local = ghost_get_num_local(), num_ghosts = (NTask > 1) ? ghost_get_num_ghosts() : 0;
    const long long first_local = MG_TaskRowOffsets[ThisTask], end_local = first_local + n_gas;
    MG_DeviceBytesRequested = 0;

    double *d_cand_q = mg_device_alloc<double>((size_t)n_cand, "mg_graph_candidate_q");
    unsigned char *d_cand_kept = mg_device_alloc<unsigned char>((size_t)n_cand, "mg_graph_candidate_kept");
    double *d_rdiag = mg_device_alloc<double>((size_t)n_gas, "mg_graph_rdiag");
    double *d_rhs = mg_device_alloc<double>((size_t)n_gas, "mg_graph_rhs");
    double *d_s = mg_device_alloc<double>((size_t)n_gas, "mg_graph_flux_scale");
    long long *d_count = mg_device_alloc<long long>((size_t)n_gas + 1, "mg_graph_row_count");
    long long *d_task_off = mg_device_alloc<long long>((size_t)NTask + 1, "mg_graph_task_offsets");
    int *d_home_rank = mg_device_alloc<int>((size_t)num_ghosts, "mg_graph_ghost_home_rank");
    int *d_home_index = mg_device_alloc<int>((size_t)num_ghosts, "mg_graph_ghost_home_index");
    MGd_off = mg_device_alloc<long long>((size_t)n_gas + 1, "mg_graph_offsets");
    int failed = MG_DeviceAllocFailed;
    MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if(failed) {
        mg_device_free(d_cand_q); mg_device_free(d_cand_kept); mg_device_free(d_rdiag); mg_device_free(d_rhs); mg_device_free(d_s);
        mg_device_free(d_count); mg_device_free(d_task_off); mg_device_free(d_home_rank); mg_device_free(d_home_index); mg_device_free(MGd_off);
        MG_DeviceAllocFailed = 0;
        return 0;
    }
    gizmo_gpu_copy_to_device(d_task_off, MG_TaskRowOffsets.data(), (size_t)(NTask + 1) * sizeof(long long));
    if(num_ghosts > 0) {
        gizmo_gpu_copy_to_device(d_home_rank, ghost_get_home_rank(), (size_t)num_ghosts * sizeof(int));
        gizmo_gpu_copy_to_device(d_home_index, ghost_get_home_index(), (size_t)num_ghosts * sizeof(int));
    }

    gpu_particles_arena_acquire(NumPart, P, CellP);
    struct particle_data *Pd = gpu_particles_arena_P();
    struct gas_cell_data *Cd = gpu_particles_arena_CellP();
    const int *rows = csr->d_rows; const int64_t *off = csr->d_offsets; const int *ngb = csr->d_neighbors;

    Kokkos::parallel_for("mg_row_zero", Kokkos::RangePolicy<>(0, n_gas + 1), KOKKOS_LAMBDA(const int i) {
        d_count[i] = 0;
        if(i < n_gas) {d_rdiag[i] = 0; d_rhs[i] = 0; d_s[i] = 0;}
    });

#ifdef MHD_MODIFIED_GRADIENT_RELATIVE_FIELD_WEIGHT
    {   /* this solve's mean |B|^2, over the same cells the coefficients below keep as rows */
        double sum_Bsq = 0, n_cells = 0;
        Kokkos::parallel_reduce("mg_field_mean_sum", Kokkos::RangePolicy<>(0, n_rows), KOKKOS_LAMBDA(const int aa, double &s) {
            const int i = rows[aa];
            if(mg_cell_in_graph(Pd[i], Cd[i])) {s += Cd[i].Bfield().norm_sq();}
        }, sum_Bsq);
        Kokkos::parallel_reduce("mg_field_mean_count", Kokkos::RangePolicy<>(0, n_rows), KOKKOS_LAMBDA(const int aa, double &c) {
            const int i = rows[aa];
            if(mg_cell_in_graph(Pd[i], Cd[i])) {c += 1;}
        }, n_cells);
        double totals[2] = {sum_Bsq, n_cells};
        MPI_Allreduce(MPI_IN_PLACE, totals, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MG_CandidateInvMeanB2 = (totals[0] > 0) ? totals[1] / totals[0] : 0.0;
    }
#endif
    const double inv_mean_Bsq = MG_CandidateInvMeanB2;

    /* Coefficients: one thread per row, faces in CSR order (each face appears in both of its rows'
       CSR; a row accumulates only itself, so rows are independent). Each face is evaluated in one
       orientation, with the cell of the smaller global row as "i", whichever of its two rows (on
       whichever task) is doing the work: whether the face is kept, its coefficient and its flux are then
       the same bits in both rows, and its right-hand-side term exactly opposite. Evaluated from each end,
       the two orientations round differently (fused multiply-adds), and a face whose area nearly cancels
       could be kept in one row and dropped in the other, leaving the matrix unsymmetric. */
    Kokkos::parallel_for("mg_coefficients", Kokkos::RangePolicy<>(0, n_rows), KOKKOS_LAMBDA(const int aa) {
        const int i = rows[aa];
        if(!mg_cell_in_graph(Pd[i], Cd[i])) {
            for(int64_t nn = off[aa]; nn < off[aa+1]; nn++) {d_cand_kept[nn] = 0;}
            return;
        }
        const long long row_i = first_local + i;
        double rdiag = 0, rhs = 0, flux_scale = 0;
        long long n_kept = 0;

        for(int64_t nn = off[aa]; nn < off[aa+1]; nn++) {
            d_cand_kept[nn] = 0;
            int j = ngb[nn];
            if(!mg_cell_in_graph(Pd[j], Cd[j])) continue;
            const long long row_j = (j < num_local) ? first_local + j : d_task_off[d_home_rank[j - num_local]] + d_home_index[j - num_local];
            const int a = (row_i < row_j) ? i : j, b = (row_i < row_j) ? j : i;   /* the face's orientation */
            const double sign = (a == i) ? 1.0 : -1.0;                             /* this row's side of it */

            struct mg_face_kernel kernel;
            kernel.dp = Pd[a].Pos - Pd[b].Pos;
            nearest_xyz(kernel.dp);
            double h_a = Pd[a].KernelRadius, h_b = Pd[b].KernelRadius, r2 = kernel.dp.norm_sq();
            if(r2 <= 0 || (r2 >= h_a * h_a && r2 >= h_b * h_b)) continue;
            kernel.r = sqrt(r2); kernel.h_i = h_a;
            double u, hinv, hinv3, hinv4;
            if(kernel.r < h_a) {kernel_hinv(h_a, &hinv, &hinv3, &hinv4); u = kernel.r * hinv; kernel_main(u, hinv3, hinv4, &kernel.wk_i, &kernel.dwk_i, 0);} else {kernel.wk_i = kernel.dwk_i = 0;}
            if(kernel.r < h_b) {kernel_hinv(h_b, &hinv, &hinv3, &hinv4); u = kernel.r * hinv; kernel_main(u, hinv3, hinv4, &kernel.wk_j, &kernel.dwk_j, 0);} else {kernel.wk_j = kernel.dwk_j = 0;}

            double V_a = Pd[a].Mass / Cd[a].Density, V_b = Pd[b].Mass / Cd[b].Density;
            double cnumcrit2 = ((double)CONDITION_NUMBER_DANGER)*((double)CONDITION_NUMBER_DANGER) - Cd[a].ConditionNumber*Cd[a].ConditionNumber;
            Vec3<double> Face_Area_Vec; double Face_Area_Norm, Vi_inv_corr_unused, Vj_inv_corr_unused;
            compute_finitevol_faces(Cd[a], Cd[b], kernel, 1.0 / (MIN_REAL_NUMBER + kernel.r), r2, V_a, V_b,
                                    pow(V_a, 1./NUMDIMS), Pd[b].Get_Particle_Size(), cnumcrit2,
                                    Face_Area_Vec, Face_Area_Norm, Vi_inv_corr_unused, Vj_inv_corr_unused);
            if(Face_Area_Norm <= 0) continue;

            double Qnorm2 = mg_face_shift_factor(r2, Cd[a].Bfield().norm_sq(), Cd[b].Bfield().norm_sq(), inv_mean_Bsq) * Face_Area_Vec.norm_sq() + 1.0e-60;
            rdiag += Qnorm2;

            /* face states as hydro reconstructs them at the face midpoint (x_ab - x_a = -dp/2, x_ab - x_b = +dp/2) */
            Vec3<double> Ba = Cd[a].Bfield(), Bb = Cd[b].Bfield();
            NGB_SHEARBOX_BOUNDARY_BCORR_(Pd[a].Pos, Pd[b].Pos, Bb, -1);
            Vec3<double> Bface_a = Ba, Bface_b = Bb;
            for(int k = 0; k < 3; k++) for(int k2 = 0; k2 < 3; k2++) {
                Bface_a[k] -= 0.5 * Cd[a].Gradients.B[k][k2] * kernel.dp[k2];
                Bface_b[k] += 0.5 * Cd[b].Gradients.B[k][k2] * kernel.dp[k2];
            }
            rhs += sign * (0.5 * dot(Bface_a + Bface_b, Face_Area_Vec));
            flux_scale += 0.5 * (sqrt(Bface_a.norm_sq()) + sqrt(Bface_b.norm_sq())) * Face_Area_Norm;

            d_cand_q[nn] = Qnorm2; d_cand_kept[nn] = 1; n_kept++;
        }
        d_rdiag[i] = rdiag; d_rhs[i] = rhs; d_s[i] = flux_scale; d_count[i] = n_kept;
    });

    /* each row's place in the physical graph */
    long long n_edges = 0;
    long long *graph_off = MGd_off;   /* lambdas capture locals, never file-scope variables */
    Kokkos::parallel_scan("mg_graph_offsets", Kokkos::RangePolicy<>(0, n_gas + 1), KOKKOS_LAMBDA(const int i, long long &partial, const bool final) {
        long long c = (i < n_gas) ? d_count[i] : 0;
        if(final) {graph_off[i] = partial;}
        partial += c;
    }, n_edges);
    MG_GraphEdges = n_edges;
    MGd_col = mg_device_alloc<long long>((size_t)n_edges, "mg_graph_columns");
    MGd_q = mg_device_alloc<double>((size_t)n_edges, "mg_graph_coefficients");
    failed = MG_DeviceAllocFailed;
    MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if(!failed) {
        long long *gcol = MGd_col, *goff = MGd_off; double *gq = MGd_q;
        /* compact each row's kept faces with their global columns (owner known per face: this task, or the
           ghost's home), then sort the row into its summation order */
        Kokkos::parallel_for("mg_graph_compact", Kokkos::RangePolicy<>(0, n_rows), KOKKOS_LAMBDA(const int aa) {
            const int i = rows[aa];
            long long base = goff[i], k = base;
            for(int64_t nn = off[aa]; nn < off[aa+1]; nn++) {
                if(!d_cand_kept[nn]) continue;
                int j = ngb[nn];
                gcol[k] = (j < num_local) ? first_local + j : d_task_off[d_home_rank[j - num_local]] + d_home_index[j - num_local];
                gq[k] = d_cand_q[nn];
                k++;
            }
            mg_sort_row(gcol + base, gq + base, k - base, first_local, end_local);
        });
        /* every row's cross-task faces: count, place, copy (in the row's order) */
        Kokkos::parallel_for("mg_graph_remote_count", Kokkos::RangePolicy<>(0, n_gas), KOKKOS_LAMBDA(const int i) {
            d_count[i] = goff[i+1] - mg_local_end(gcol, goff[i], goff[i+1], first_local, end_local);
        });
    }
    std::vector<long long> host_count((size_t)DMAX(n_gas, 1), 0);
    std::vector<double> host_rdiag((size_t)DMAX(n_gas, 1)), host_rhs((size_t)DMAX(n_gas, 1)), host_s((size_t)DMAX(n_gas, 1));
    kept.assign((size_t)DMAX(n_cand, 1LL), 0);
    gizmo_gpu_copy_to_host(host_rdiag.data(), d_rdiag, (size_t)n_gas * sizeof(double));
    gizmo_gpu_copy_to_host(host_rhs.data(), d_rhs, (size_t)n_gas * sizeof(double));
    gizmo_gpu_copy_to_host(host_s.data(), d_s, (size_t)n_gas * sizeof(double));
    gizmo_gpu_copy_to_host(kept.data(), d_cand_kept, (size_t)n_cand * sizeof(unsigned char));
    for(int i = 0; i < n_gas; i++) {MG_Rows[i].Rdiag = host_rdiag[i]; MG_Rows[i].rhs = host_rhs[i]; MG_Rows[i].flux_scale = host_s[i];}
    mg_device_free(d_cand_q); mg_device_free(d_cand_kept); mg_device_free(d_rdiag); mg_device_free(d_rhs); mg_device_free(d_s);
    mg_device_free(d_task_off); mg_device_free(d_home_rank); mg_device_free(d_home_index);

    if(!failed) {
        gizmo_gpu_copy_to_host(host_count.data(), d_count, (size_t)n_gas * sizeof(long long));
        MG_RemoteOff.assign((size_t)n_gas + 1, 0);
        for(int i = 0; i < n_gas; i++) {MG_RemoteOff[i+1] = MG_RemoteOff[i] + host_count[i];}
        const long long n_remote = MG_RemoteOff[n_gas];
        long long *d_roff = mg_device_alloc<long long>((size_t)n_gas + 1, "mg_graph_remote_offsets");
        long long *d_rcol = mg_device_alloc<long long>((size_t)n_remote, "mg_graph_remote_columns");
        double *d_rq = mg_device_alloc<double>((size_t)n_remote, "mg_graph_remote_coefficients");
        failed = MG_DeviceAllocFailed;
        MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if(!failed) {
            gizmo_gpu_copy_to_device(d_roff, MG_RemoteOff.data(), (size_t)(n_gas + 1) * sizeof(long long));
            long long *gcol = MGd_col, *goff = MGd_off; double *gq = MGd_q;
            Kokkos::parallel_for("mg_graph_remote_copy", Kokkos::RangePolicy<>(0, n_gas), KOKKOS_LAMBDA(const int i) {
                long long k = d_roff[i];
                for(long long e = mg_local_end(gcol, goff[i], goff[i+1], first_local, end_local); e < goff[i+1]; e++) {d_rcol[k] = gcol[e]; d_rq[k] = gq[e]; k++;}
            });
            MG_RemoteCol.resize((size_t)DMAX(n_remote, 1LL)); MG_RemoteQ.resize((size_t)DMAX(n_remote, 1LL));
            gizmo_gpu_copy_to_host(MG_RemoteCol.data(), d_rcol, (size_t)n_remote * sizeof(long long));
            gizmo_gpu_copy_to_host(MG_RemoteQ.data(), d_rq, (size_t)n_remote * sizeof(double));
            MG_RemoteCol.resize((size_t)n_remote); MG_RemoteQ.resize((size_t)n_remote);
        }
        mg_device_free(d_roff); mg_device_free(d_rcol); mg_device_free(d_rq);
    }
    mg_device_free(d_count);
    MG_DeviceAllocFailed = 0;
    if(failed) {mg_device_free(MGd_col); mg_device_free(MGd_q); mg_device_free(MGd_off); return 0;}

#if defined(MHD_MODIFIED_GRADIENT_CG_ONLY) || defined(MHD_MODIFIED_GRADIENT_USE_PARDISO)
    MG_HostOff.resize((size_t)n_gas + 1); MG_HostCol.resize((size_t)DMAX(n_edges, 1LL)); MG_HostQ.resize((size_t)DMAX(n_edges, 1LL));
    gizmo_gpu_copy_to_host(MG_HostOff.data(), MGd_off, (size_t)(n_gas + 1) * sizeof(long long));
    gizmo_gpu_copy_to_host(MG_HostCol.data(), MGd_col, (size_t)n_edges * sizeof(long long));
    gizmo_gpu_copy_to_host(MG_HostQ.data(), MGd_q, (size_t)n_edges * sizeof(double));
#endif
    return 1;
}

static void mg_free_graph(void)
{
    mg_device_free(MGd_off); mg_device_free(MGd_col); mg_device_free(MGd_q);
    MG_GraphEdges = 0;
    MG_RemoteOff.clear(); MG_RemoteCol.clear(); MG_RemoteQ.clear();
    MG_RemoteOff.shrink_to_fit(); MG_RemoteCol.shrink_to_fit(); MG_RemoteQ.shrink_to_fit();
#if defined(MHD_MODIFIED_GRADIENT_CG_ONLY) || defined(MHD_MODIFIED_GRADIENT_USE_PARDISO)
    MG_HostOff.clear(); MG_HostCol.clear(); MG_HostQ.clear();
    MG_HostOff.shrink_to_fit(); MG_HostCol.shrink_to_fit(); MG_HostQ.shrink_to_fit();
#endif
}


/* ==================================================================================== */
/* Symmetry: every face (i -> j, q) has its reverse (j -> i, q).                          */
/* ==================================================================================== */

static long long mg_count_asymmetric_entries(void)
{
    const int n_gas = N_gas;
    const long long first_local = MG_TaskRowOffsets[ThisTask], end_local = first_local + n_gas;
    const long long *gcol = MGd_col, *goff = MGd_off; const double *gq = MGd_q;
    /* faces to cells on this task: binary search of the reverse in the neighbour's sorted row */
    long long bad = 0;
    Kokkos::parallel_reduce("mg_symmetry_local", Kokkos::RangePolicy<>(0, n_gas), KOKKOS_LAMBDA(const int i, long long &nbad) {
        long long lend = mg_local_end(gcol, goff[i], goff[i+1], first_local, end_local);
        for(long long e = goff[i]; e < lend; e++) {
            long long j = gcol[e] - first_local, me = first_local + i;
            long long jend = mg_local_end(gcol, goff[j], goff[j+1], first_local, end_local);
            long long r = mg_lower_bound(gcol, goff[j], jend, me);
            double a = gq[e], b = (r < jend && gcol[r] == me) ? gq[r] : 0;
            if(r >= jend || gcol[r] != me || !(fabs(a - b) <= MG_SYMMETRY_TOL * DMAX(fabs(a), fabs(b)))) {nbad++;}
        }
    }, bad);
    /* cross-task faces: send (my row, its row, q) to the owner, which looks for the reverse */
    auto close = [](double a, double b) {return fabs(a - b) <= MG_SYMMETRY_TOL * DMAX(fabs(a), fabs(b));};
    std::vector<int> scount(NTask, 0), rcount(NTask, 0), soff(NTask, 0), roff(NTask, 0), fill(NTask, 0);
    for(size_t k = 0; k < MG_RemoteCol.size(); k++) {scount[mg_row_owner(MG_RemoteCol[k])] += 3;}
    MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int ns = 0, nr = 0; for(int t = 0; t < NTask; t++) {soff[t] = ns; roff[t] = nr; ns += scount[t]; nr += rcount[t];}
    std::vector<double> sbuf((size_t)DMAX(ns, 1)), rbuf((size_t)DMAX(nr, 1));
    for(int i = 0; i < n_gas; i++) for(long long e = MG_RemoteOff[i]; e < MG_RemoteOff[i+1]; e++) {
        int t = mg_row_owner(MG_RemoteCol[e]);
        size_t k = (size_t)(soff[t] + fill[t]); fill[t] += 3;
        sbuf[k] = i; sbuf[k+1] = (double)(MG_RemoteCol[e] - MG_TaskRowOffsets[t]); sbuf[k+2] = MG_RemoteQ[e];
    }
    MPI_Alltoallv(sbuf.data(), scount.data(), soff.data(), MPI_DOUBLE, rbuf.data(), rcount.data(), roff.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    for(int t = 0; t < NTask; t++) for(int k = roff[t]; k < roff[t] + rcount[t]; k += 3) {
        long long sender = mg_global_row(t, (int) rbuf[k]);
        int j = (int) rbuf[k+1];
        if(j < 0 || j >= n_gas) {bad++; continue;}
        const long long *first = MG_RemoteCol.data() + MG_RemoteOff[j], *last = MG_RemoteCol.data() + MG_RemoteOff[j+1];
        const long long *e = std::lower_bound(first, last, sender);
        if(e == last || *e != sender || !close(MG_RemoteQ[e - MG_RemoteCol.data()], rbuf[k+2])) {bad++;}
    }
    long long bad_global = 0;
    MPI_Allreduce(&bad, &bad_global, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return bad_global;
}


/* ==================================================================================== */
/* Connected pieces of the graph                                                          */
/* ==================================================================================== */

static int mg_find(std::vector<int> &parent, int a) {while(parent[a] != a) {parent[a] = parent[parent[a]]; a = parent[a];} return a;}

struct mg_piece_graph_report {long long pieces, cross_task_links; int rounds; int replicated;};

/* Label each row with its piece: the smallest global row id in the piece. Pieces are
 * found on each task first (from the host CSR and the kept flag of each of its entries);
 * the links between task-local pieces across tasks are then resolved either by gathering
 * the (deduplicated) link list everywhere, when it is small enough, or by propagating the
 * smallest label along the links until nothing changes. */
static struct mg_piece_graph_report mg_label_pieces(const struct mg_csr *csr, const std::vector<unsigned char> &kept)
{
    struct mg_piece_graph_report rep = {0, 0, 0, 1};
    const int num_local = ghost_get_num_local();
    std::vector<int> parent((size_t)DMAX(N_gas, 1));
    for(int i = 0; i < N_gas; i++) {parent[i] = i;}
    for(int aa = 0; aa < csr->n_rows; aa++) {
        int i = csr->rows[aa];
        for(int64_t nn = csr->offsets[aa]; nn < csr->offsets[aa+1]; nn++) {
            int j = csr->neighbors[nn];
            if(!kept[nn] || j >= num_local) continue;
            int a = mg_find(parent, i), b = mg_find(parent, j);
            if(a != b) {if(a < b) {parent[b] = a;} else {parent[a] = b;}}   /* the root stays the smallest row */
        }
    }
    std::vector<double> local_label((size_t)DMAX(N_gas, 1));
    for(int i = 0; i < N_gas; i++) {local_label[i] = (double) mg_global_row(ThisTask, mg_find(parent, i));} /* exact in a double below 2^53 */
    mg_exchange_values(local_label.data());

    struct link {long long mine, theirs; int task;};
    std::vector<link> links;
    for(int i = 0; i < N_gas; i++) for(long long e = MG_RemoteOff[i]; e < MG_RemoteOff[i+1]; e++) {
        long long c = MG_RemoteCol[e];
        links.push_back({(long long) local_label[i], (long long) MG_ExchangeValues[MG_RemoteSlot[e]], mg_row_owner(c)});
    }
    std::sort(links.begin(), links.end(), [](const link &a, const link &b) {return (a.mine != b.mine) ? (a.mine < b.mine) : ((a.theirs != b.theirs) ? (a.theirs < b.theirs) : (a.task < b.task));});
    links.erase(std::unique(links.begin(), links.end(), [](const link &a, const link &b) {return a.mine == b.mine && a.theirs == b.theirs;}), links.end());
    long long n_links = (long long) links.size();
    MPI_Allreduce(&n_links, &rep.cross_task_links, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    std::unordered_map<long long, long long> resolved;   /* task-local piece label -> global piece label, for pieces with cross-task links */
    for(const link &l : links) {resolved[l.mine] = l.mine;}
    if(rep.cross_task_links > 0) {
        const long long replicated_bytes = 2LL * rep.cross_task_links * (long long)sizeof(long long);
        rep.replicated = (replicated_bytes <= MG_REPLICATED_PIECE_GRAPH_MAX_BYTES) && (2LL * rep.cross_task_links < (long long)INT_MAX);
        if(rep.replicated) {
            int my_count = 2 * (int) links.size();
            std::vector<int> counts(NTask), displs(NTask);
            MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
            int total = 0; for(int t = 0; t < NTask; t++) {displs[t] = total; total += counts[t];}
            std::vector<long long> mine_flat((size_t)DMAX(my_count, 1)), all_flat((size_t)DMAX(total, 1));
            for(size_t k = 0; k < links.size(); k++) {mine_flat[2*k] = links[k].mine; mine_flat[2*k+1] = links[k].theirs;}
            MPI_Allgatherv(mine_flat.data(), my_count, MPI_LONG_LONG, all_flat.data(), counts.data(), displs.data(), MPI_LONG_LONG, MPI_COMM_WORLD);
            std::vector<long long> labels(all_flat.begin(), all_flat.begin() + total);
            std::sort(labels.begin(), labels.end()); labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
            std::vector<int> lparent(labels.size());
            for(size_t k = 0; k < labels.size(); k++) {lparent[k] = (int)k;}
            auto index_of = [&labels](long long v) {return (int)(std::lower_bound(labels.begin(), labels.end(), v) - labels.begin());};
            for(int k = 0; k < total; k += 2) {
                int a = mg_find(lparent, index_of(all_flat[k])), b = mg_find(lparent, index_of(all_flat[k+1]));
                if(a != b) {if(a < b) {lparent[b] = a;} else {lparent[a] = b;}}   /* sorted labels: the root is the smallest */
            }
            for(auto &kv : resolved) {kv.second = labels[mg_find(lparent, index_of(kv.first))];}
        } else {
            /* propagate the smallest label along the links (both directions are present in a symmetric graph) */
            std::vector<int> scount(NTask), rcount(NTask), soff(NTask), roff(NTask), fill(NTask);
            for(int changed_global = 1; changed_global; ) {
                rep.rounds++;
                std::fill(scount.begin(), scount.end(), 0); std::fill(fill.begin(), fill.end(), 0);
                for(const link &l : links) {scount[l.task] += 2;}
                MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, MPI_COMM_WORLD);
                int ns = 0, nr = 0; for(int t = 0; t < NTask; t++) {soff[t] = ns; roff[t] = nr; ns += scount[t]; nr += rcount[t];}
                std::vector<long long> sbuf((size_t)DMAX(ns, 1)), rbuf((size_t)DMAX(nr, 1));
                for(const link &l : links) {size_t k = (size_t)(soff[l.task] + fill[l.task]); fill[l.task] += 2; sbuf[k] = l.theirs; sbuf[k+1] = resolved[l.mine];}
                MPI_Alltoallv(sbuf.data(), scount.data(), soff.data(), MPI_LONG_LONG, rbuf.data(), rcount.data(), roff.data(), MPI_LONG_LONG, MPI_COMM_WORLD);
                int changed = 0;
                for(int k = 0; k < nr; k += 2) {
                    auto it = resolved.find(rbuf[k]);
                    if(it == resolved.end()) {resolved[rbuf[k]] = rbuf[k+1]; changed = 1;}
                    else if(rbuf[k+1] < it->second) {it->second = rbuf[k+1]; changed = 1;}
                }
                MPI_Allreduce(&changed, &changed_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            }
        }
    }
    long long pieces = 0;
    for(int i = 0; i < N_gas; i++) {
        long long l = (long long) local_label[i];
        auto it = resolved.find(l);
        MG_Rows[i].piece = (it == resolved.end()) ? l : it->second;
        if(MG_Rows[i].piece == mg_global_row(ThisTask, i)) {pieces++;}
    }
    MPI_Allreduce(&pieces, &rep.pieces, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return rep;
}

enum mg_piece_op {MG_PIECE_SUM, MG_PIECE_MAX, MG_PIECE_MIN};
static inline double mg_piece_identity(int op) {return (op == MG_PIECE_MAX) ? -HUGE_VAL : ((op == MG_PIECE_MIN) ? HUGE_VAL : 0.0);}
static inline void mg_piece_combine(int op, double *acc, double v) {if(op == MG_PIECE_MAX) {*acc = DMAX(*acc, v);} else if(op == MG_PIECE_MIN) {*acc = DMIN(*acc, v);} else {*acc += v;}}

/* Reduce nv per-row values over each piece, across tasks (sum, max or min): row_values and
 * row_totals are N_gas x nv. Each piece is reduced at the task owning its label, which
 * returns the result to every task holding part of the piece. Local partial sums are
 * compensated (the piece sum of b is compared against roundoff). */
static void mg_piece_reduce(int nv, const double *row_values, double *row_totals, int op)
{
    std::unordered_map<long long, int> slot;
    std::vector<long long> slot_label;
    std::vector<double> sum, comp;
    for(int i = 0; i < N_gas; i++) {
        auto it = slot.find(MG_Rows[i].piece);
        int s;
        if(it == slot.end()) {s = (int) slot_label.size(); slot[MG_Rows[i].piece] = s; slot_label.push_back(MG_Rows[i].piece); sum.resize(sum.size() + nv, mg_piece_identity(op)); comp.resize(comp.size() + nv, 0.0);}
        else {s = it->second;}
        for(int v = 0; v < nv; v++) {
            double x = row_values[(size_t)i*nv + v];
            if(op == MG_PIECE_SUM) {   /* Kahan */
                double y = x - comp[(size_t)s*nv + v], t = sum[(size_t)s*nv + v] + y;
                comp[(size_t)s*nv + v] = (t - sum[(size_t)s*nv + v]) - y; sum[(size_t)s*nv + v] = t;
            } else {mg_piece_combine(op, &sum[(size_t)s*nv + v], x);}
        }
    }
    /* partial results of pieces owned elsewhere go to the owner */
    const int rec = nv + 1;
    std::vector<int> scount(NTask, 0), rcount(NTask), soff(NTask), roff(NTask), fill(NTask, 0);
    for(size_t s = 0; s < slot_label.size(); s++) {int o = mg_row_owner(slot_label[s]); if(o != ThisTask) {scount[o] += rec;}}
    MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int ns = 0, nr = 0; for(int t = 0; t < NTask; t++) {soff[t] = ns; roff[t] = nr; ns += scount[t]; nr += rcount[t];}
    std::vector<double> sbuf((size_t)DMAX(ns, 1)), rbuf((size_t)DMAX(nr, 1));
    for(size_t s = 0; s < slot_label.size(); s++) {
        int o = mg_row_owner(slot_label[s]); if(o == ThisTask) continue;
        size_t k = (size_t)(soff[o] + fill[o]); fill[o] += rec;
        sbuf[k] = (double) slot_label[s]; for(int v = 0; v < nv; v++) {sbuf[k+1+v] = sum[s*nv + v];}
    }
    MPI_Alltoallv(sbuf.data(), scount.data(), soff.data(), MPI_DOUBLE, rbuf.data(), rcount.data(), roff.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    for(int k = 0; k < nr; k += rec) {
        long long l = (long long) rbuf[k];
        auto it = slot.find(l);
        int s;
        if(it == slot.end()) {s = (int) slot_label.size(); slot[l] = s; slot_label.push_back(l); sum.resize(sum.size() + nv, mg_piece_identity(op));}
        else {s = it->second;}
        for(int v = 0; v < nv; v++) {mg_piece_combine(op, &sum[(size_t)s*nv + v], rbuf[k+1+v]);}
    }
    /* the owner returns the result to every task that contributed (the reverse of the send pattern) */
    for(int k = 0; k < nr; k += rec) {int s = slot[(long long) rbuf[k]]; for(int v = 0; v < nv; v++) {rbuf[k+1+v] = sum[(size_t)s*nv + v];}}
    MPI_Alltoallv(rbuf.data(), rcount.data(), roff.data(), MPI_DOUBLE, sbuf.data(), scount.data(), soff.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    for(int k = 0; k < ns; k += rec) {int s = slot[(long long) sbuf[k]]; for(int v = 0; v < nv; v++) {sum[(size_t)s*nv + v] = sbuf[k+1+v];}}
    for(int i = 0; i < N_gas; i++) {int s = slot[MG_Rows[i].piece]; for(int v = 0; v < nv; v++) {row_totals[(size_t)i*nv + v] = sum[(size_t)s*nv + v];}}
}

/* Each piece's anchor: its row of largest Rdiag, the smallest global row id among equal ones
 * (row ids are exact in a double below 2^53). A row without faces never anchors a piece that has them. */
static void mg_select_anchors(void)
{
    std::vector<double> v((size_t)DMAX(N_gas, 1)), best((size_t)DMAX(N_gas, 1));
    for(int i = 0; i < N_gas; i++) {v[i] = (MG_Rows[i].Rdiag > 1.0e-60 && isfinite(MG_Rows[i].Rdiag)) ? MG_Rows[i].Rdiag : 0.0;}
    mg_piece_reduce(1, v.data(), best.data(), MG_PIECE_MAX);
    for(int i = 0; i < N_gas; i++) {v[i] = (v[i] == best[i]) ? (double) mg_global_row(ThisTask, i) : HUGE_VAL;}
    mg_piece_reduce(1, v.data(), best.data(), MG_PIECE_MIN);
    for(int i = 0; i < N_gas; i++) {MG_Rows[i].anchor = (long long) best[i];}
}


/* ==================================================================================== */
/* The true (unpinned) operator on a solution: residual terms of every row               */
/* ==================================================================================== */

/* For every row i, with x_j of other-task rows exchanged: the edge form v_i = sum_j q_ij (x_i - x_j)
 * (exact for Rdiag = sum_j q_ij, and free of the cancellation of Rdiag x_i - sum_j q_ij x_j when the
 * coefficients span tens of orders of magnitude), its rounding bound e_i = 16 eps sum_j q_ij (|x_i| +
 * |x_j|) and the flux the correction puts through its faces, corr_i = sum_j q_ij |x_j - x_i|, with its largest
 * single-face term, envelope_i = max_j q_ij |x_j - x_i| (q without the matrix floor), which bounds the correction
 * reused on steps without the solve. Returns 0 if device memory could not be had on some task (collective). */
static int mg_residual_terms(const double *x, double *v, double *e_bound, double *corr, double *envelope)
{
    const int n_gas = N_gas, n_req = (int) MG_RequestCol.size();
    const long long first_local = MG_TaskRowOffsets[ThisTask], end_local = first_local + n_gas;
    mg_exchange_values(x);
    MG_DeviceBytesRequested = 0;
    double *d_x = mg_device_alloc<double>((size_t)n_gas, "mg_graph_residual_x");
    double *d_xr = mg_device_alloc<double>((size_t)n_req, "mg_graph_residual_xremote");
    long long *d_req = mg_device_alloc<long long>((size_t)n_req, "mg_graph_residual_requests");
    double *d_v = mg_device_alloc<double>((size_t)n_gas, "mg_graph_residual_v");
    double *d_e = mg_device_alloc<double>((size_t)n_gas, "mg_graph_residual_bound");
    double *d_c = mg_device_alloc<double>((size_t)n_gas, "mg_graph_residual_corr");
    double *d_env = mg_device_alloc<double>((size_t)n_gas, "mg_graph_residual_envelope");
    int failed = MG_DeviceAllocFailed;
    MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MG_DeviceAllocFailed = 0;
    if(!failed) {
        gizmo_gpu_copy_to_device(d_x, x, (size_t)n_gas * sizeof(double));
        gizmo_gpu_copy_to_device(d_xr, MG_ExchangeValues.data(), (size_t)n_req * sizeof(double));
        gizmo_gpu_copy_to_device(d_req, MG_RequestCol.data(), (size_t)n_req * sizeof(long long));
        const long long *gcol = MGd_col, *goff = MGd_off; const double *gq = MGd_q;
        Kokkos::parallel_for("mg_residual", Kokkos::RangePolicy<>(0, n_gas), KOKKOS_LAMBDA(const int i) {
            long long lend = mg_local_end(gcol, goff[i], goff[i+1], first_local, end_local);
            double xi = d_x[i], vi = 0, ei = 0, ci = 0, envi = 0;
            for(long long k = goff[i]; k < goff[i+1]; k++) {
                double xj = (k < lend) ? d_x[gcol[k] - first_local] : d_xr[mg_lower_bound(d_req, 0, n_req, gcol[k])];
                vi += gq[k] * (xi - xj); ei += gq[k] * (fabs(xi) + fabs(xj)); ci += gq[k] * fabs(xj - xi);
                double face_flux = (gq[k] - 1.0e-60) * fabs(xj - xi);
                if(face_flux > envi) {envi = face_flux;}
            }
            d_v[i] = vi; d_e[i] = ei; d_c[i] = ci; d_env[i] = envi;
        });
        gizmo_gpu_copy_to_host(v, d_v, (size_t)n_gas * sizeof(double));
        gizmo_gpu_copy_to_host(e_bound, d_e, (size_t)n_gas * sizeof(double));
        gizmo_gpu_copy_to_host(corr, d_c, (size_t)n_gas * sizeof(double));
        gizmo_gpu_copy_to_host(envelope, d_env, (size_t)n_gas * sizeof(double));
    }
    mg_device_free(d_x); mg_device_free(d_xr); mg_device_free(d_req); mg_device_free(d_v); mg_device_free(d_e); mg_device_free(d_c); mg_device_free(d_env);
    return !failed;
}


/* The metrics a solve is judged by, from the true edge-form residual r = b - R x over every row of
 * a solved piece (anchor rows included: a pinned solve that has not converged leaves on the anchor
 * row minus the sum of the other rows' residuals, which a global norm can hide):
 *   G = |r|/|b|,   P = sqrt(sum r^2 / sum s^2),   M = max_i (|r_i| - e_i)+ / d_i,
 * with d_i = s_i + sum_j q_ij |x_j - x_i| the row's face-flux scale plus the flux the correction
 * itself puts through its faces (a componentwise backward-error scale: rows whose field is zero or
 * vanishing are judged against the field the correction gives them), and e_i = 16 eps sum_j q_ij
 * (|x_i| + |x_j|) the rounding bound of evaluating r_i. The uncorrected state c = 0 has r = b and
 * d = s: G0 = 1, P0 = sqrt(sum b^2 / sum s^2), M0 = max |b_i| / s_i. The correction flux relative
 * to the field's, C = sqrt(sum (d - s)^2 / sum s^2), is reported so a large correction is visible.
 * A row with d_i = 0 has b_i = 0 and no correction, so a residual there beyond e_i is a structural
 * failure (counted in n_unexplained). e_bound and corr come from mg_residual_terms. */
struct mg_solve_metrics {double G, P, M, P0, M0, C; int any_rows; long long n_unexplained;};
static struct mg_solve_metrics mg_judge_solution(const double *resid, const double *e_bound, const double *corr_flux)
{
    double sum_loc[4] = {0, 0, 0, 0}, sum[4], mx_loc[2] = {0, 0}, mx[2];   /* sum r^2, b^2, s^2, (d-s)^2; max M, M0 */
    long long unexplained_loc = 0, unexplained = 0;
    for(int i = 0; i < N_gas; i++) {
        if(!MG_RowInSolvedPiece[i]) continue;
        double b = MG_Rows[i].rhs, s = MG_Rows[i].flux_scale, e = e_bound[i], corr = corr_flux[i];
        sum_loc[0] += resid[i] * resid[i]; sum_loc[1] += b * b; sum_loc[2] += s * s; sum_loc[3] += corr * corr;
        double excess = DMAX(fabs(resid[i]) - 16.0 * DBL_EPSILON * e, 0.0), d = s + corr;
        if(d > 0) {mx_loc[0] = DMAX(mx_loc[0], excess / d);} else if(excess > 0) {unexplained_loc++;}
        if(s > 0) {mx_loc[1] = DMAX(mx_loc[1], fabs(b) / s);}
    }
    MPI_Allreduce(sum_loc, sum, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(mx_loc, mx, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&unexplained_loc, &unexplained, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    struct mg_solve_metrics m;
    m.any_rows = (sum[1] > 0);
    m.G = m.any_rows ? sqrt(sum[0] / sum[1]) : 0;
    m.P = (sum[2] > 0) ? sqrt(sum[0] / sum[2]) : 0; m.P0 = (sum[2] > 0) ? sqrt(sum[1] / sum[2]) : 0;
    m.M = mx[0]; m.M0 = mx[1]; m.n_unexplained = unexplained;
    m.C = (sum[2] > 0) ? sqrt(sum[3] / sum[2]) : 0;
    return m;
}


/* ==================================================================================== */
/* Built-in solver (MHD_MODIFIED_GRADIENT_CG_ONLY): SSOR-preconditioned CG on the pinned */
/* system. SSOR acts on the local block only (block-Jacobi between tasks).               */
/* ==================================================================================== */

#ifdef MHD_MODIFIED_GRADIENT_CG_ONLY
/* the end of row i's faces to cells on this task in the host copy of the graph */
static inline long long mg_host_local_end(int i)
{
    const long long first_local = MG_TaskRowOffsets[ThisTask];
    return mg_local_end(MG_HostCol.data(), MG_HostOff[i], MG_HostOff[i+1], first_local, first_local + N_gas);
}

static void mg_ssor_precondition(const double *r, double *z)
{
    const long long first_local = MG_TaskRowOffsets[ThisTask];
    for(int i = 0; i < N_gas; i++) {z[i] = 0;}
    for(int sweep = 0; sweep < 2; sweep++) {   /* forward, then backward */
        for(int ii = 0; ii < N_gas; ii++) {
            int i = (sweep == 0) ? ii : N_gas - 1 - ii;
            if(!MG_RowSolved[i]) continue;
            double sum = r[i];
            long long lend = mg_host_local_end(i);
            for(long long k = MG_HostOff[i]; k < lend; k++) {
                if(mg_column_kept(i, MG_HostCol[k])) {sum += MG_HostQ[k] * z[MG_HostCol[k] - first_local];}
            }
            z[i] = sum / MG_Rows[i].Rdiag;
        }
    }
}

/* y = R_pinned x on solved rows (x is 0 on every unsolved row, so their columns contribute nothing) */
static void mg_apply_pinned(const double *x, double *y)
{
    const long long first_local = MG_TaskRowOffsets[ThisTask];
    mg_exchange_values(x);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int i = 0; i < N_gas; i++) {
        if(!MG_RowSolved[i]) {y[i] = 0; continue;}
        double v = MG_Rows[i].Rdiag * x[i];
        long long lend = mg_host_local_end(i);
        for(long long k = MG_HostOff[i]; k < lend; k++) {v -= MG_HostQ[k] * x[MG_HostCol[k] - first_local];}
        for(long long e = MG_RemoteOff[i]; e < MG_RemoteOff[i+1]; e++) {v -= MG_RemoteQ[e] * MG_ExchangeValues[MG_RemoteSlot[e]];}
        y[i] = v;
    }
}

static double mg_global_dot(const double *a, const double *b)
{
    double loc = 0, glob;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:loc)
#endif
    for(int i = 0; i < N_gas; i++) {loc += a[i] * b[i];}
    MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return glob;
}

/* returns the iteration count; x holds the last iterate (the caller judges it by its true residual) */
static int mg_solve_backend(double *x)
{
    std::vector<double> r(MG_RhsSolved), z((size_t)DMAX(N_gas, 1)), p((size_t)DMAX(N_gas, 1)), Ap((size_t)DMAX(N_gas, 1));
    for(int i = 0; i < N_gas; i++) {x[i] = 0;}
    double bnorm2 = mg_global_dot(r.data(), r.data());
    if(bnorm2 <= 0) {return 0;}
    mg_ssor_precondition(r.data(), z.data());
    p = z;
    double rz = mg_global_dot(r.data(), z.data()), rnorm2 = bnorm2;
    int iter;
    for(iter = 0; iter < MG_CG_MAX_ITER; iter++) {
        mg_apply_pinned(p.data(), Ap.data());
        double pAp = mg_global_dot(p.data(), Ap.data());
        if(!(pAp > 0) || !(rz > 0)) {   /* both are positive for an SPD system and preconditioner; anything else is a breakdown */
            if(ThisTask == 0) {PRINT_WARNING("MG built-in CG broke down at iteration %d (p.Ap = %g, r.z = %g); the iterate is judged by its true residual", iter, pAp, rz);}
            break;
        }
        double alpha = rz / pAp;
        for(int i = 0; i < N_gas; i++) {x[i] += alpha * p[i]; r[i] -= alpha * Ap[i];}
        rnorm2 = mg_global_dot(r.data(), r.data());
        if(rnorm2 <= MG_CG_TOL * MG_CG_TOL * bnorm2) {iter++; break;}
        mg_ssor_precondition(r.data(), z.data());
        double rz_new = mg_global_dot(r.data(), z.data()), beta = rz_new / rz;
        rz = rz_new;
        for(int i = 0; i < N_gas; i++) {p[i] = z[i] + beta * p[i];}
    }
    return iter;
}
#endif /* MHD_MODIFIED_GRADIENT_CG_ONLY */


/* ==================================================================================== */
/* HYPRE: BoomerAMG-preconditioned PCG on the pinned system (default)                    */
/* ==================================================================================== */

#if !defined(MHD_MODIFIED_GRADIENT_CG_ONLY) && !defined(MHD_MODIFIED_GRADIENT_USE_PARDISO)
/* The solver's insertion arrays, derived on the device from the physical graph once each row's
   solved flag and anchor are known: an unsolved row is the equation c_i = 0; a solved row has its
   diagonal and every face except the column of its piece's anchor unknown (this task's columns
   first, then the other tasks', as everywhere). They live until HYPRE has assembled its matrix. */
struct mg_insertion {HYPRE_Int *ncols; HYPRE_BigInt *rows, *cols; HYPRE_Real *vals; long long nnz;};

static int mg_build_insertion(struct mg_insertion *ins)
{
    const int n_gas = N_gas;
    const long long first_local = MG_TaskRowOffsets[ThisTask];
    MG_DeviceBytesRequested = 0;
    std::vector<long long> anchor((size_t)DMAX(n_gas, 1));
    std::vector<double> diag((size_t)DMAX(n_gas, 1));
    for(int i = 0; i < n_gas; i++) {anchor[i] = MG_Rows[i].anchor; diag[i] = MG_RowSolved[i] ? MG_Rows[i].Rdiag : 1.0;}
    char *d_solved = mg_device_alloc<char>((size_t)n_gas, "mg_insert_solved");
    long long *d_anchor = mg_device_alloc<long long>((size_t)n_gas, "mg_insert_anchor");
    double *d_diag = mg_device_alloc<double>((size_t)n_gas, "mg_insert_diagonal");
    long long *d_ioff = mg_device_alloc<long long>((size_t)n_gas + 1, "mg_insert_offsets");
    ins->ncols = mg_device_alloc<HYPRE_Int>((size_t)n_gas, "mg_insert_ncols");
    ins->rows = mg_device_alloc<HYPRE_BigInt>((size_t)n_gas, "mg_insert_rows");
    int failed = MG_DeviceAllocFailed;
    MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    long long nnz = 0;
    if(!failed) {
        gizmo_gpu_copy_to_device(d_solved, MG_RowSolved.data(), (size_t)n_gas * sizeof(char));
        gizmo_gpu_copy_to_device(d_anchor, anchor.data(), (size_t)n_gas * sizeof(long long));
        gizmo_gpu_copy_to_device(d_diag, diag.data(), (size_t)n_gas * sizeof(double));
        const long long *gcol = MGd_col, *goff = MGd_off;
        HYPRE_Int *ncols = ins->ncols; HYPRE_BigInt *rows = ins->rows;
        Kokkos::parallel_scan("mg_insert_offsets", Kokkos::RangePolicy<>(0, n_gas + 1), KOKKOS_LAMBDA(const int i, long long &partial, const bool final) {
            long long c = 0;
            if(i < n_gas) {
                c = 1;
                if(d_solved[i]) {for(long long k = goff[i]; k < goff[i+1]; k++) {if(gcol[k] != d_anchor[i]) {c++;}}}
                if(final) {ncols[i] = (HYPRE_Int) c; rows[i] = (HYPRE_BigInt)(first_local + i);}
            }
            if(final) {d_ioff[i] = partial;}
            partial += c;
        }, nnz);
    }
    ins->nnz = nnz;
    ins->cols = mg_device_alloc<HYPRE_BigInt>((size_t)nnz, "mg_insert_columns");
    ins->vals = mg_device_alloc<HYPRE_Real>((size_t)nnz, "mg_insert_values");
    failed = MG_DeviceAllocFailed;
    MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if(!failed) {
        const long long *gcol = MGd_col, *goff = MGd_off; const double *gq = MGd_q;
        HYPRE_BigInt *cols = ins->cols; HYPRE_Real *vals = ins->vals;
        Kokkos::parallel_for("mg_insert_fill", Kokkos::RangePolicy<>(0, n_gas), KOKKOS_LAMBDA(const int i) {
            long long p = d_ioff[i];
            cols[p] = (HYPRE_BigInt)(first_local + i); vals[p] = d_diag[i]; p++;
            if(!d_solved[i]) return;
            for(long long k = goff[i]; k < goff[i+1]; k++) {if(gcol[k] != d_anchor[i]) {cols[p] = (HYPRE_BigInt) gcol[k]; vals[p] = -gq[k]; p++;}}
        });
        Kokkos::fence("mg_insert_fill");
    }
    mg_device_free(d_solved); mg_device_free(d_anchor); mg_device_free(d_diag); mg_device_free(d_ioff);
    MG_DeviceAllocFailed = 0;
    return !failed;
}

static void mg_free_insertion(struct mg_insertion *ins)
{
    mg_device_free(ins->ncols); mg_device_free(ins->rows); mg_device_free(ins->cols); mg_device_free(ins->vals);
}

static int mg_solve_backend(double *x)
{
    static int hypre_initialized = 0;   /* HYPRE's handle (on a GPU build: device, streams, memory pools) is set up once per run */
    if(!hypre_initialized) {HYPRE_Init(); hypre_initialized = 1;}
    HYPRE_BigInt ilower = (HYPRE_BigInt) MG_TaskRowOffsets[ThisTask], iupper = (HYPRE_BigInt)(MG_TaskRowOffsets[ThisTask] + N_gas - 1);
    struct mg_insertion ins = {NULL, NULL, NULL, NULL, 0};
    if(!mg_build_insertion(&ins)) {   /* every task skips the solve together, so no task waits in a HYPRE collective */
        mg_free_insertion(&ins);
        if(ThisTask == 0) {PRINT_WARNING("MG: device memory for the HYPRE system could not be allocated on some task (task 0 asked for %g MB)", MG_DeviceBytesRequested / (1024. * 1024.));}
        return -1;
    }
    std::vector<HYPRE_Real> x_vals((size_t)DMAX(N_gas, 1), 0.0);
    HYPRE_Real *rhs_in = MG_RhsSolved.data(), *x_in = x_vals.data();
#ifdef MG_HYPRE_ON_DEVICE
    /* the matrix arrays are already on the device; the right-hand side and the start go there once and the
       solution comes back once. HYPRE's global memory location and execution policy are switched for this
       solve only and restored to whatever they were. */
    HYPRE_Int *ncols_in = ins.ncols; HYPRE_BigInt *rows_in = ins.rows, *cols_in = ins.cols; HYPRE_Real *vals_in = ins.vals;
    rhs_in = mg_device_alloc<HYPRE_Real>((size_t)N_gas, "mg_insert_rhs");
    x_in   = mg_device_alloc<HYPRE_Real>((size_t)N_gas, "mg_insert_solution");
    int alloc_failed = MG_DeviceAllocFailed;
    MPI_Allreduce(MPI_IN_PLACE, &alloc_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MG_DeviceAllocFailed = 0;
    if(alloc_failed) {
        mg_device_free(rhs_in); mg_device_free(x_in); mg_free_insertion(&ins);
        if(ThisTask == 0) {PRINT_WARNING("MG: device memory for the HYPRE system could not be allocated on some task (task 0 asked for %g MB)", MG_DeviceBytesRequested / (1024. * 1024.));}
        return -1;
    }
    gizmo_gpu_copy_to_device(rhs_in, MG_RhsSolved.data(), (size_t)N_gas * sizeof(HYPRE_Real));
    gizmo_gpu_copy_to_device(x_in, x_vals.data(), (size_t)N_gas * sizeof(HYPRE_Real));
    HYPRE_MemoryLocation prior_location; HYPRE_ExecutionPolicy prior_policy;
    HYPRE_GetMemoryLocation(&prior_location); HYPRE_GetExecutionPolicy(&prior_policy);
    HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
    HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
#else
    /* a host HYPRE reads host memory: on a device build the insertion arrays come back once */
    std::vector<HYPRE_Int> ncols_host; std::vector<HYPRE_BigInt> rows_host, cols_host; std::vector<HYPRE_Real> vals_host;
    HYPRE_Int *ncols_in = ins.ncols; HYPRE_BigInt *rows_in = ins.rows, *cols_in = ins.cols; HYPRE_Real *vals_in = ins.vals;
    if(!Kokkos::SpaceAccessibility<Kokkos::HostSpace, GIZMO_KOKKOS_DEVICE_SPACE>::accessible) {
        ncols_host.resize((size_t)DMAX(N_gas, 1)); rows_host.resize((size_t)DMAX(N_gas, 1));
        cols_host.resize((size_t)DMAX(ins.nnz, 1LL)); vals_host.resize((size_t)DMAX(ins.nnz, 1LL));
        gizmo_gpu_copy_to_host(ncols_host.data(), ins.ncols, (size_t)N_gas * sizeof(HYPRE_Int));
        gizmo_gpu_copy_to_host(rows_host.data(), ins.rows, (size_t)N_gas * sizeof(HYPRE_BigInt));
        gizmo_gpu_copy_to_host(cols_host.data(), ins.cols, (size_t)ins.nnz * sizeof(HYPRE_BigInt));
        gizmo_gpu_copy_to_host(vals_host.data(), ins.vals, (size_t)ins.nnz * sizeof(HYPRE_Real));
        ncols_in = ncols_host.data(); rows_in = rows_host.data(); cols_in = cols_host.data(); vals_in = vals_host.data();
    }
#endif
    HYPRE_IJMatrix A;
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A);
    if(N_gas > 0) {HYPRE_IJMatrixSetValues(A, (HYPRE_Int) N_gas, ncols_in, rows_in, cols_in, vals_in);}
    HYPRE_IJMatrixAssemble(A);
    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_IJMatrixGetObject(A, (void **)&parcsr_A);
    /* HYPRE holds its own copy of the matrix now: the insertion arrays go, except the row ids the vectors still read */
    mg_device_free(ins.ncols); mg_device_free(ins.cols); mg_device_free(ins.vals);
#ifndef MG_HYPRE_ON_DEVICE
    ncols_host.clear(); ncols_host.shrink_to_fit(); cols_host.clear(); cols_host.shrink_to_fit(); vals_host.clear(); vals_host.shrink_to_fit();
#endif

    HYPRE_IJVector b_vec, x_vec;
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b_vec); HYPRE_IJVectorSetObjectType(b_vec, HYPRE_PARCSR); HYPRE_IJVectorInitialize(b_vec);
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &x_vec); HYPRE_IJVectorSetObjectType(x_vec, HYPRE_PARCSR); HYPRE_IJVectorInitialize(x_vec);
    if(N_gas > 0) {
        HYPRE_IJVectorSetValues(b_vec, (HYPRE_Int) N_gas, rows_in, rhs_in);
        HYPRE_IJVectorSetValues(x_vec, (HYPRE_Int) N_gas, rows_in, x_in);
    }
    HYPRE_IJVectorAssemble(b_vec); HYPRE_IJVectorAssemble(x_vec);
    HYPRE_ParVector par_b, par_x;
    HYPRE_IJVectorGetObject(b_vec, (void **)&par_b); HYPRE_IJVectorGetObject(x_vec, (void **)&par_x);

    HYPRE_Solver solver, precond;
    HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);
    HYPRE_PCGSetMaxIter(solver, (HYPRE_Int) MG_CG_MAX_ITER);
    HYPRE_PCGSetTol(solver, (HYPRE_Real) MG_CG_TOL);
    HYPRE_PCGSetTwoNorm(solver, 1);
    HYPRE_PCGSetPrintLevel(solver, 0);
    HYPRE_BoomerAMGCreate(&precond);
    HYPRE_BoomerAMGSetPrintLevel(precond, 0);
#ifdef MG_HYPRE_ON_DEVICE
    HYPRE_BoomerAMGSetCoarsenType(precond, 8);  /* PMIS coarsening (Falgout is not available in a device setup) */
    HYPRE_BoomerAMGSetRelaxType(precond, 18);   /* l1-scaled Jacobi (GPU-parallel) */
#else
    HYPRE_BoomerAMGSetCoarsenType(precond, 6);  /* Falgout coarsening */
    HYPRE_BoomerAMGSetRelaxType(precond, 6);    /* symmetric Gauss-Seidel (CPU) */
#endif
    HYPRE_BoomerAMGSetNumSweeps(precond, 1);
    HYPRE_BoomerAMGSetMaxLevels(precond, 25);
    HYPRE_BoomerAMGSetMaxIter(precond, 1);      /* 1 V-cycle per PCG iteration */
    HYPRE_BoomerAMGSetTol(precond, 0.0);
    HYPRE_PCGSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);
    HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
    HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);
    HYPRE_Int num_iterations;
    HYPRE_PCGGetNumIterations(solver, &num_iterations);
    if(N_gas > 0) {HYPRE_IJVectorGetValues(x_vec, (HYPRE_Int) N_gas, rows_in, x_in);}
    HYPRE_ClearAllErrors();   /* the outcome is judged by the true residual, not HYPRE's flags */

    HYPRE_ParCSRPCGDestroy(solver);
    HYPRE_BoomerAMGDestroy(precond);
    HYPRE_IJMatrixDestroy(A);
    HYPRE_IJVectorDestroy(b_vec);
    HYPRE_IJVectorDestroy(x_vec);
#ifdef MG_HYPRE_ON_DEVICE
    HYPRE_SetMemoryLocation(prior_location);
    HYPRE_SetExecutionPolicy(prior_policy);
    gizmo_gpu_copy_to_host(x_vals.data(), x_in, (size_t)N_gas * sizeof(HYPRE_Real));
    mg_device_free(rhs_in); mg_device_free(x_in);
#endif
    mg_free_insertion(&ins);
    for(int i = 0; i < N_gas; i++) {x[i] = MG_RowSolved[i] ? x_vals[i] : 0;}
    return (int) num_iterations;
}
#endif


/* ==================================================================================== */
/* MKL PARDISO direct solver on rank 0 (MHD_MODIFIED_GRADIENT_USE_PARDISO)               */
/* ==================================================================================== */

#ifdef MHD_MODIFIED_GRADIENT_USE_PARDISO
static int mg_solve_backend(double *x)
{
    /* the pinned system as upper-triangle CSR rows with sorted columns (the symmetric PARDISO input format) */
    std::vector<int> row_ptr(1, 0), col_idx;
    std::vector<double> vals;
    for(int i = 0; i < N_gas; i++) {
        long long gi = mg_global_row(ThisTask, i);
        std::vector<std::pair<long long, double>> row;
        row.push_back({gi, MG_RowSolved[i] ? MG_Rows[i].Rdiag : 1.0});
        if(MG_RowSolved[i]) {
            for(long long k = MG_HostOff[i]; k < MG_HostOff[i+1]; k++) {
                long long g = MG_HostCol[k];
                if(g > gi && mg_column_kept(i, g)) {row.push_back({g, -MG_HostQ[k]});}
            }
        }
        std::sort(row.begin(), row.end());
        for(auto &e : row) {col_idx.push_back((int) e.first); vals.push_back(e.second);}
        row_ptr.push_back((int) col_idx.size());
    }
    int my_rows = N_gas, my_nnz = (int) col_idx.size();
    std::vector<int> all_rows(NTask), all_nnz(NTask), disp_rows(NTask), disp_nnz(NTask);
    MPI_Gather(&my_rows, 1, MPI_INT, all_rows.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&my_nnz, 1, MPI_INT, all_nnz.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    int n_global = 0, nnz_global = 0;
    if(ThisTask == 0) {for(int t = 0; t < NTask; t++) {disp_rows[t] = n_global; disp_nnz[t] = nnz_global; n_global += all_rows[t]; nnz_global += all_nnz[t];}}
    std::vector<int> g_row_counts((size_t)DMAX(n_global, 1)), g_col((size_t)DMAX(nnz_global, 1));
    std::vector<double> g_val((size_t)DMAX(nnz_global, 1)), g_rhs((size_t)DMAX(n_global, 1)), g_x((size_t)DMAX(n_global, 1), 0.0);
    std::vector<int> row_counts((size_t)DMAX(N_gas, 1));
    for(int i = 0; i < N_gas; i++) {row_counts[i] = row_ptr[i+1] - row_ptr[i];}
    MPI_Gatherv(row_counts.data(), my_rows, MPI_INT, g_row_counts.data(), all_rows.data(), disp_rows.data(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(col_idx.data(), my_nnz, MPI_INT, g_col.data(), all_nnz.data(), disp_nnz.data(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(vals.data(), my_nnz, MPI_DOUBLE, g_val.data(), all_nnz.data(), disp_nnz.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(MG_RhsSolved.data(), my_rows, MPI_DOUBLE, g_rhs.data(), all_rows.data(), disp_rows.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(ThisTask == 0 && n_global > 0) {
        std::vector<MKL_INT> ia((size_t)n_global + 1), ja((size_t)nnz_global);
        ia[0] = 1; for(int r = 0; r < n_global; r++) {ia[r+1] = ia[r] + g_row_counts[r];}
        for(int k = 0; k < nnz_global; k++) {ja[k] = g_col[k] + 1;}   /* 1-based */
        void *pt[64]; for(int k = 0; k < 64; k++) pt[k] = NULL;
        MKL_INT iparm[64]; for(int k = 0; k < 64; k++) iparm[k] = 0;
        MKL_INT maxfct = 1, mnum = 1, mtype = 2 /* real symmetric positive definite (pinned) */, phase, nrhs = 1, msglvl = 0, error = 0, perm_dummy, N = n_global;
        iparm[0] = 1; iparm[1] = 2; iparm[7] = 0; iparm[9] = 13;
        phase = 13; /* analysis, factorization, solve */
        PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N, g_val.data(), ia.data(), ja.data(), &perm_dummy, &nrhs, iparm, &msglvl, g_rhs.data(), g_x.data(), &error);
        if(error != 0) {PRINT_WARNING("MG PARDISO error %d; the solve is judged by its residual below", (int)error);}
        phase = -1;
        PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N, g_val.data(), ia.data(), ja.data(), &perm_dummy, &nrhs, iparm, &msglvl, g_rhs.data(), g_x.data(), &error);
    }
    MPI_Scatterv(g_x.data(), all_rows.data(), disp_rows.data(), MPI_DOUBLE, x, my_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for(int i = 0; i < N_gas; i++) {if(!MG_RowSolved[i]) {x[i] = 0;}}
    return 1;
}
#endif /* MHD_MODIFIED_GRADIENT_USE_PARDISO */


/* ==================================================================================== */
/* Public entry point                                                                     */
/* ==================================================================================== */

/* An accepted solve replaces the coefficients, the piece tokens and the envelopes together; anything else leaves the last accepted set. */
static void mg_commit_solution(const double *x, const double *envelope)
{
    for(int i = 0; i < N_gas; i++) {
        bool in_graph = mg_cell_in_graph(P[i], CellP[i]);
        CellP[i].MG_cgcoeff = in_graph ? x[i] : 0;
        CellP[i].MG_piece_token = in_graph ? (MG_Rows[i].piece + 1) : 0;
        CellP[i].MG_envelope = in_graph ? envelope[i] : 0;
    }
#ifdef MHD_MODIFIED_GRADIENT_RELATIVE_FIELD_WEIGHT
    All.MG_RelativeFieldInvMeanB2 = MG_CandidateInvMeanB2;   /* the normalisation these coefficients were solved with */
    GIZMO_GPU_ENSURE_ALL_FRESH();
#endif
}

void mg_gradient_correction_calc(void)
{
    if(ThisTask == 0) PRINT_STATUS("Computing MG gradient correction (exact div(B)=0) ...");

    /* the right-hand side reads neighbour gradients, which ghosts carry only after this refresh */
    gizmo_hydro_corridor_refresh_ghost_values("pre_mg");
    if(NTask > 1 && !ghost_pool_is_live()) {
        PRINT_WARNING("MG solve reached without a ghost pool: the matrix would be missing every cross-task face");
        endrun(8804);
        return;
    }
    GIZMO_GPU_ENSURE_ALL_FRESH();

    MG_Rows = (struct mg_row_data *) calloc((size_t)DMAX(N_gas, 1), sizeof(struct mg_row_data));
    MG_TaskRowOffsets.assign(NTask + 1, 0);
    {long long my_rows = N_gas, offset = 0; MPI_Exscan(&my_rows, &offset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD); if(ThisTask == 0) {offset = 0;}
     MPI_Allgather(&offset, 1, MPI_LONG_LONG, MG_TaskRowOffsets.data(), 1, MPI_LONG_LONG, MPI_COMM_WORLD);
     long long total = 0; MPI_Allreduce(&my_rows, &total, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD); MG_TaskRowOffsets[NTask] = total;}

    /* The corridor's CSR is the MG graph when its rows are every local gas cell (an all-active
       step); otherwise build one over every local gas cell on the corridor's all-gas pool. The
       device reads the device copy each builder keeps beside the host list. */
    int n_eligible = 0;
    for(int i = 0; i < N_gas; i++) {if(P[i].Type == 0 && P[i].Mass > 0) {n_eligible++;}}
    struct mg_csr csr = {NULL, NULL, NULL, 0, NULL, NULL, NULL};
    const gpu_neighbor_list_t *corridor_device = NULL;
    int reuse = (gizmo_hydro_corridor_csr_arrays(&csr.rows, &csr.n_rows, &csr.offsets, &csr.neighbors, &corridor_device) && csr.n_rows == n_eligible) ? 1 : 0, reuse_all = 0;
    MPI_Allreduce(&reuse, &reuse_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    neighbor_list_t private_list = {NULL, NULL, 0, 0};
    gpu_neighbor_list_t private_device = {NULL, NULL, NULL, 0, 0};
    int *private_rows = NULL;
    const gpu_neighbor_list_t *device_list = corridor_device;
    if(!reuse_all) {
        private_rows = (int *) mymalloc("mg_allgas_rows", (size_t)DMAX(n_eligible, 1) * sizeof(int));
        csr.n_rows = 0;
        for(int i = 0; i < N_gas; i++) {if(P[i].Type == 0 && P[i].Mass > 0) {private_rows[csr.n_rows++] = i;}}
        gpu_build_symmetric_neighbor_list(P, NumPart, private_rows, csr.n_rows, &private_list, 1.0, 0, &private_device);
        csr.rows = private_rows; csr.offsets = private_list.offsets; csr.neighbors = private_list.neighbors;
        device_list = &private_device;
    }
    /* the device copy built with this list (the corridor's, or the private one; the counts are a sanity check);
       without one the list is staged from the host copy */
    int *staged_rows = NULL; int64_t *staged_offsets = NULL; int *staged_neighbors = NULL;
    const long long n_pairs = (csr.n_rows > 0) ? (long long) csr.offsets[csr.n_rows] : 0;
    if(device_list && device_list->neighbors && device_list->d_active && device_list->num_active == csr.n_rows && device_list->total_pairs == n_pairs) {
        csr.d_rows = device_list->d_active; csr.d_offsets = device_list->offsets; csr.d_neighbors = device_list->neighbors;
    } else {
        staged_rows = mg_device_alloc<int>((size_t)csr.n_rows, "mg_graph_staged_rows");
        staged_offsets = mg_device_alloc<int64_t>((size_t)csr.n_rows + 1, "mg_graph_staged_offsets");
        staged_neighbors = mg_device_alloc<int>((size_t)n_pairs, "mg_graph_staged_neighbors");
        if(!MG_DeviceAllocFailed) {
            gizmo_gpu_copy_to_device(staged_rows, csr.rows, (size_t)csr.n_rows * sizeof(int));
            if(csr.n_rows > 0) {gizmo_gpu_copy_to_device(staged_offsets, csr.offsets, (size_t)(csr.n_rows + 1) * sizeof(int64_t));}
            gizmo_gpu_copy_to_device(staged_neighbors, csr.neighbors, (size_t)n_pairs * sizeof(int));
        }
        csr.d_rows = staged_rows; csr.d_offsets = staged_offsets; csr.d_neighbors = staged_neighbors;
    }

    std::vector<unsigned char> kept;
    int stop_code = 0;
    if(!mg_build_graph(&csr, kept)) {
        if(ThisTask == 0) {PRINT_WARNING("MG matrix: device memory for the face graph could not be allocated on some task (task 0 asked for %g MB)", MG_DeviceBytesRequested / (1024. * 1024.));}
        stop_code = 8805;
    }
    mg_device_free(staged_rows); mg_device_free(staged_offsets); mg_device_free(staged_neighbors);
    gpu_ngb_list_free(&private_device);

    if(!stop_code) {mg_setup_exchange();}
    std::vector<double> x((size_t)DMAX(N_gas, 1), 0.0), resid((size_t)DMAX(N_gas, 1), 0.0);
    std::vector<double> e_bound((size_t)DMAX(N_gas, 1), 0.0), corr_flux((size_t)DMAX(N_gas, 1), 0.0), envelope((size_t)DMAX(N_gas, 1), 0.0);
    if(!stop_code) {
        long long asymmetric = mg_count_asymmetric_entries();
        if(asymmetric > 0) {
            if(ThisTask == 0) {PRINT_WARNING("MG matrix is not symmetric: %lld entries have no matching reverse entry -- the face graph is incomplete or one-sided", asymmetric);}
            stop_code = 8801;
        }
    }
    struct mg_piece_graph_report pieces = {0, 0, 0, 1};
    if(!stop_code) {   /* labelling and anchors assume every link has its reverse */
        pieces = mg_label_pieces(&csr, kept);
        mg_select_anchors();
    }
    kept.clear(); kept.shrink_to_fit();
    if(!reuse_all) {free_neighbor_list(&private_list); myfree(private_rows);}

    /* per piece: sum b against sum s (b telescopes to roundoff over a closed piece), and ||b|| against ||s|| (the noise floor) */
    long long n_floor_pieces = 0;
    double worst_imbalance = 0;
    MG_RowSolved.assign((size_t)DMAX(N_gas, 1), 0);
    MG_RhsSolved.assign((size_t)DMAX(N_gas, 1), 0.0);
    MG_RowInSolvedPiece.assign((size_t)DMAX(N_gas, 1), 0);
    if(!stop_code) {
        std::vector<double> v((size_t)5 * DMAX(N_gas, 1)), tot((size_t)5 * DMAX(N_gas, 1));
        for(int i = 0; i < N_gas; i++) {double b = MG_Rows[i].rhs, s = MG_Rows[i].flux_scale; v[5*i] = b; v[5*i+1] = s; v[5*i+2] = b*b; v[5*i+3] = s*s; v[5*i+4] = 1.0;}
        mg_piece_reduce(5, v.data(), tot.data(), MG_PIECE_SUM);
        long long floor_loc = 0, nonfinite_loc = 0, nonfinite_coeff = 0;
        double imbalance_loc = 0;
        for(int i = 0; i < N_gas; i++) {
            bool labels_piece = (MG_Rows[i].piece == mg_global_row(ThisTask, i));   /* one row per piece, for per-piece counts */
            bool anchor = (MG_Rows[i].anchor == mg_global_row(ThisTask, i));
            bool below_floor = (tot[5*i+2] <= MG_NOISE_FLOOR * MG_NOISE_FLOOR * tot[5*i+3]);
            if(!isfinite(MG_Rows[i].Rdiag) || !isfinite(MG_Rows[i].rhs) || !isfinite(MG_Rows[i].flux_scale)) {nonfinite_loc++;}
            if(labels_piece && tot[5*i+1] > 0) {imbalance_loc = DMAX(imbalance_loc, fabs(tot[5*i]) / tot[5*i+1]);}
            if(labels_piece && below_floor && MG_Rows[i].Rdiag > 1.0e-60) {floor_loc++;}
            MG_RowInSolvedPiece[i] = !below_floor;
            MG_RowSolved[i] = (MG_Rows[i].Rdiag > 1.0e-60) && !anchor && !below_floor;
            MG_RhsSolved[i] = MG_RowSolved[i] ? MG_Rows[i].rhs : 0.0;
        }
        MPI_Allreduce(&floor_loc, &n_floor_pieces, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&imbalance_loc, &worst_imbalance, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&nonfinite_loc, &nonfinite_coeff, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if(nonfinite_coeff > 0) {
            if(ThisTask == 0) {PRINT_WARNING("MG matrix or right-hand side has %lld non-finite rows", nonfinite_coeff);}
            stop_code = 8803;
        } else if(worst_imbalance > MG_RHS_IMBALANCE_STRUCTURAL) {
            if(ThisTask == 0) {PRINT_WARNING("MG right-hand side does not sum to roundoff on a connected piece (|sum b|/sum s = %g): the face fluxes are not antisymmetric", worst_imbalance);}
            stop_code = 8802;
        } else if(worst_imbalance > MG_CG_TOL && ThisTask == 0) {
            PRINT_WARNING("MG right-hand side sums to |sum b|/sum s = %g on a connected piece, above the solve tolerance %g: that piece cannot converge below it", worst_imbalance, MG_CG_TOL);
        }
    }

    int iterations = 0;
    const char *outcome = "converged";
    double rel_resid = 0;
    struct mg_solve_metrics metrics = {0, 0, 0, 0, 0, 0, 0, 0};
    if(!stop_code) {
        iterations = mg_solve_backend(x.data());
        if(iterations < 0) {stop_code = 8806;}   /* the backend could not allocate its system (same on every task) */
    }
    if(!stop_code) {
        int nonfinite_loc = 0, nonfinite = 0;
        for(int i = 0; i < N_gas; i++) {if(!isfinite(x[i])) {nonfinite_loc = 1;}}
        MPI_Allreduce(&nonfinite_loc, &nonfinite, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if(nonfinite) {
            if(ThisTask == 0) {PRINT_WARNING("MG solve returned a non-finite solution");}
            stop_code = 8803;
        }
    }
    if(!stop_code && !mg_residual_terms(x.data(), resid.data(), e_bound.data(), corr_flux.data(), envelope.data())) {
        if(ThisTask == 0) {PRINT_WARNING("MG: device memory for the residual check could not be allocated on some task (task 0 asked for %g MB)", MG_DeviceBytesRequested / (1024. * 1024.));}
        stop_code = 8806;
    }
    if(!stop_code) {
        /* The anchored solution (c = 0 at each piece's anchor row) is the gauge kept: c spans as many decades
           as the coefficients, so subtracting a piece mean would round away c on the large-coefficient rows
           (measured on FIF: every such row collapsed to the same value). c enters the physics only through
           differences across faces, so a per-piece constant changes nothing. */
        for(int i = 0; i < N_gas; i++) {resid[i] = MG_Rows[i].rhs - resid[i];}
        metrics = mg_judge_solution(resid.data(), e_bound.data(), corr_flux.data());
        rel_resid = metrics.G;
        if(metrics.n_unexplained > 0) {
            if(ThisTask == 0) {PRINT_WARNING("MG solve left a residual on %lld rows that carry neither field nor correction flux", metrics.n_unexplained);}
            endrun(8803);
            outcome = "stopped";
        } else {
            bool converged = (metrics.G <= MG_CG_TOL);
            /* applied only if it lowers the divergence globally, in the field-scaled norm, and in its worst row */
            bool improves = !metrics.any_rows || ((metrics.G < 1) && (metrics.P < metrics.P0) && (metrics.M <= metrics.M0));
            if(!improves) {
                /* keep the last accepted coefficients and pieces; this step is then a step without the solve, except that
                   the gradients pass has already run without its CG correction */
                outcome = converged ? "converged globally but raises the worst row; previous MG correction kept, no CG correction this step"
                                    : "not converged; previous MG correction kept, no CG correction this step";
                All.Flag_SkipMGSolve = 1;
            } else if(!converged) {outcome = "not converged; applied (lowers the divergence)";}
            if((!converged || !improves) && ThisTask == 0) {
                PRINT_WARNING("MG solve %s after %d iterations: |r|/|b| %g (tolerance %g), sqrt(sum r^2/sum s^2) %g -> %g, max |r|/d %g -> %g",
                              outcome, iterations, metrics.G, MG_CG_TOL, metrics.P0, metrics.P, metrics.M0, metrics.M);
            }
            if(improves) {mg_commit_solution(x.data(), envelope.data());}
#ifdef DIVBCLEANING_DEDNER
            if(converged && improves) {   /* the face divergence is now zero, so the cleaning potential it sourced is spurious */
                for(int i = 0; i < N_gas; i++) {if(P[i].Type == 0) {CellP[i].Phi = CellP[i].PhiPred = CellP[i].DtPhi = 0;}}
            }
#endif
        }
    } else {
        endrun(stop_code);
        outcome = "stopped";
    }
    if(ThisTask == 0) {
        PRINT_STATUS(" ..MG solve: %lld connected pieces (%lld below the noise floor), %lld cross-task piece links (%s), %d iterations, |r|/|b| = %g, %s",
                     pieces.pieces, n_floor_pieces, pieces.cross_task_links, pieces.replicated ? "gathered" : "labels propagated",
                     iterations, rel_resid, outcome);
        if(!All.Flag_SkipMGSolve && !stop_code && (metrics.n_unexplained == 0)) {
            PRINT_STATUS(" ..MG residuals (uncorrected -> applied): sqrt(sum r^2/sum s^2) %g -> %g, max |r|/d %g -> %g; correction flux / field flux %g",
                         metrics.P0, metrics.P, metrics.M0, metrics.M, metrics.C);
        } else if(!stop_code) {
            PRINT_STATUS(" ..MG residuals (uncorrected): sqrt(sum r^2/sum s^2) %g, max |r|/d %g; the kept MG correction is not re-evaluated on this graph",
                         metrics.P0, metrics.M0);
        }
        if(!pieces.replicated) {PRINT_STATUS(" ..MG piece labels propagated in %d rounds", pieces.rounds);}
    }

    mg_cleanup_exchange();
    mg_free_graph();
    free(MG_Rows); MG_Rows = NULL;
    MG_RowSolved.clear(); MG_RhsSolved.clear(); MG_RowInSolvedPiece.clear();
    if(ThisTask == 0) PRINT_STATUS(" ..MG correction complete.");
}

#endif /* MHD_MODIFIED_GRADIENT */
