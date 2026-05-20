#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "../mesh/neighbor_list.h"
#include "../mesh/sfc_tiles.h"
#include "../mesh/ghost_writeback.h"
#include "compute_finitevol_faces_functions.h"

/*! \file mg_gradient_correction.cc
 *  \brief Modified-gradient (MG) method for exact div(B)=0 correction.
 *
 *  Implements Tu, Wang, Gao & Tang (2026), arXiv:2603.04077.
 *  This module:
 *    1) Builds a sparse symmetric linear system R*c = b over ALL gas cells,
 *       where c_i are per-particle scalar correction coefficients.
 *    2) Solves the system via preconditioned conjugate gradient (CG) with
 *       MPI ghost exchange for cross-boundary neighbor c values.
 *    3) Stores c_i in CellP[i].MG_cgcoeff for use during face reconstruction.
 *
 *  The correction ensures the discrete face-flux divergence of B is exactly zero.
 */
/*
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO,
 * based on the method described in Tu, Wang, Gao & Tang (2026), arXiv:2603.04077.
 */

#ifdef MHD_MODIFIED_GRADIENT

extern void gpu_build_symmetric_neighbor_list(struct particle_data *P, int num_total,
    int *active_indices, int num_active, neighbor_list_t *out,
    double search_radius_factor);

#if !defined(MHD_MODIFIED_GRADIENT_CG_ONLY) && !defined(MHD_MODIFIED_GRADIENT_USE_PARDISO)
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"
#endif

#ifdef MHD_MODIFIED_GRADIENT_USE_PARDISO
#include <mkl.h>
#include <mkl_pardiso.h>
#endif

#define MG_CG_MAX_ITER 500
#define MG_CG_TOL 1.0e-13
#define MG_MAX_NGB_PER_PARTICLE 128 /* initial allocation per particle; grown if needed */

/* ==================================================================================== */
/* Persistent sparse matrix storage                                                      */
/* ==================================================================================== */

/* entry for a local neighbor (both particles on this task) */
struct mg_local_entry { int j; double Qnorm2; };

/* entry for a remote neighbor (the neighbor is on another MPI task) */
struct mg_remote_entry { int remote_task; int remote_index; double Qnorm2; };

/* per-particle sparse matrix data */
struct mg_row_data {
    double Rdiag;       /* diagonal: sum_j Qnorm2_ij (complete, all pairs) */
    double rhs;         /* RHS: face-flux divergence with slope-limited gradients */
    int n_local;        /* number of local neighbor entries */
    int n_remote;       /* number of remote neighbor entries */
    int alloc_local;    /* allocated size of local array */
    int alloc_remote;   /* allocated size of remote array */
    struct mg_local_entry *local_entries;   /* local neighbor entries */
    struct mg_remote_entry *remote_entries; /* remote neighbor entries */
};

static struct mg_row_data *MG_Rows = NULL;

/* ghost exchange: per-task list of remote particles whose c values we need */
struct mg_ghost_info { int remote_task; int remote_index; double c_value; };
static struct mg_ghost_info *MG_GhostList = NULL;
static int MG_GhostCount = 0;




/* ==================================================================================== */
/* Helper: add a local entry to particle i's row                                         */
/* ==================================================================================== */
static inline void mg_add_local_entry(int i, int j, double Qnorm2)
{
    if(MG_Rows[i].n_local >= MG_Rows[i].alloc_local) {
        MG_Rows[i].alloc_local = DMAX(2 * MG_Rows[i].alloc_local, 32);
        MG_Rows[i].local_entries = (struct mg_local_entry *) realloc(MG_Rows[i].local_entries,
            MG_Rows[i].alloc_local * sizeof(struct mg_local_entry));
    }
    MG_Rows[i].local_entries[MG_Rows[i].n_local].j = j;
    MG_Rows[i].local_entries[MG_Rows[i].n_local].Qnorm2 = Qnorm2;
    MG_Rows[i].n_local++;
}

static inline void mg_add_remote_entry(int i, int remote_task, int remote_index, double Qnorm2)
{
    if(MG_Rows[i].n_remote >= MG_Rows[i].alloc_remote) {
        MG_Rows[i].alloc_remote = DMAX(2 * MG_Rows[i].alloc_remote, 16);
        MG_Rows[i].remote_entries = (struct mg_remote_entry *) realloc(MG_Rows[i].remote_entries,
            MG_Rows[i].alloc_remote * sizeof(struct mg_remote_entry));
    }
    MG_Rows[i].remote_entries[MG_Rows[i].n_remote].remote_task = remote_task;
    MG_Rows[i].remote_entries[MG_Rows[i].n_remote].remote_index = remote_index;
    MG_Rows[i].remote_entries[MG_Rows[i].n_remote].Qnorm2 = Qnorm2;
    MG_Rows[i].n_remote++;
}


/* Legacy CPU tree-walk MG build path retired 2026-04-28 (Step 5 B6.5);
 * mg_build_matrix_modern (symmetric-CSR, called from mg_gradient_correction_calc)
 * is the only build path. See git log for the deleted legacy evaluators,
 * dedup helpers, and the modern-vs-legacy validation harness. */


/* ==================================================================================== */
/* Ghost exchange: sync c values for cross-boundary neighbors each CG iteration          */
/* ==================================================================================== */

static int *mg_ghost_send_counts = NULL;
static int *mg_ghost_recv_counts = NULL;
static int *mg_ghost_send_offsets = NULL;
static int *mg_ghost_recv_offsets = NULL;

/* per-task: indices of local particles to send */
static int **mg_ghost_send_indices = NULL;
/* mapping: for each ghost entry, which position in the recv buffer */
static int *mg_ghost_recv_map = NULL;

static void mg_setup_ghost_exchange(void)
{
    int i, n, task;

    /* count unique (task, remote_index) pairs across all particles */
    /* build per-task send/recv lists */
    mg_ghost_send_counts = (int *) calloc(NTask, sizeof(int));
    mg_ghost_recv_counts = (int *) calloc(NTask, sizeof(int));
    mg_ghost_send_offsets = (int *) calloc(NTask, sizeof(int));
    mg_ghost_recv_offsets = (int *) calloc(NTask, sizeof(int));

    /* count how many remote entries go to each task (these are requests FROM us TO them) */
    /* We need c values from remote particles. We tell the owning task which particles we need. */
    int *task_counts = (int *) calloc(NTask, sizeof(int));
    for(i = 0; i < N_gas; i++) {
        for(n = 0; n < MG_Rows[i].n_remote; n++) {
            task_counts[MG_Rows[i].remote_entries[n].remote_task]++;
        }
    }
    /* total ghost entries */
    MG_GhostCount = 0;
    for(task = 0; task < NTask; task++) MG_GhostCount += task_counts[task];

    if(MG_GhostCount > 0) {
        MG_GhostList = (struct mg_ghost_info *) malloc(MG_GhostCount * sizeof(struct mg_ghost_info));
    }

    /* fill the ghost list, grouped by task */
    int *task_offsets = (int *) calloc(NTask, sizeof(int));
    task_offsets[0] = 0;
    for(task = 1; task < NTask; task++) task_offsets[task] = task_offsets[task-1] + task_counts[task-1];

    int *task_pos = (int *) calloc(NTask, sizeof(int));
    for(i = 0; i < N_gas; i++) {
        for(n = 0; n < MG_Rows[i].n_remote; n++) {
            int t = MG_Rows[i].remote_entries[n].remote_task;
            int pos = task_offsets[t] + task_pos[t];
            MG_GhostList[pos].remote_task = t;
            MG_GhostList[pos].remote_index = MG_Rows[i].remote_entries[n].remote_index;
            MG_GhostList[pos].c_value = 0;
            task_pos[t]++;
        }
    }

    /* recv_counts: how many indices we request from each task */
    for(task = 0; task < NTask; task++) mg_ghost_recv_counts[task] = task_counts[task];

    /* exchange counts: tell each task how many of their particles we need */
    MPI_Alltoall(mg_ghost_recv_counts, 1, MPI_INT, mg_ghost_send_counts, 1, MPI_INT, MPI_COMM_WORLD);

    /* compute offsets */
    mg_ghost_send_offsets[0] = mg_ghost_recv_offsets[0] = 0;
    for(task = 1; task < NTask; task++) {
        mg_ghost_send_offsets[task] = mg_ghost_send_offsets[task-1] + mg_ghost_send_counts[task-1];
        mg_ghost_recv_offsets[task] = mg_ghost_recv_offsets[task-1] + mg_ghost_recv_counts[task-1];
    }
    int total_send = 0;
    for(task = 0; task < NTask; task++) total_send += mg_ghost_send_counts[task];

    /* exchange the requested indices: we send our request list, they tell us which particles they need from us */
    int *send_indices_flat = NULL, *recv_indices_flat = NULL;
    if(MG_GhostCount > 0) {
        recv_indices_flat = (int *) malloc(MG_GhostCount * sizeof(int));
        for(i = 0; i < MG_GhostCount; i++) recv_indices_flat[i] = MG_GhostList[i].remote_index;
    }
    if(total_send > 0) send_indices_flat = (int *) malloc(total_send * sizeof(int));

    MPI_Alltoallv(recv_indices_flat, mg_ghost_recv_counts, mg_ghost_recv_offsets, MPI_INT,
                  send_indices_flat, mg_ghost_send_counts, mg_ghost_send_offsets, MPI_INT, MPI_COMM_WORLD);

    /* store the send indices (local particles that other tasks request from us) */
    mg_ghost_send_indices = (int **) malloc(NTask * sizeof(int *));
    for(task = 0; task < NTask; task++) {
        if(mg_ghost_send_counts[task] > 0) {
            mg_ghost_send_indices[task] = (int *) malloc(mg_ghost_send_counts[task] * sizeof(int));
            memcpy(mg_ghost_send_indices[task], send_indices_flat + mg_ghost_send_offsets[task],
                   mg_ghost_send_counts[task] * sizeof(int));
        } else {
            mg_ghost_send_indices[task] = NULL;
        }
    }

    if(recv_indices_flat) free(recv_indices_flat);
    if(send_indices_flat) free(send_indices_flat);
    free(task_pos); free(task_offsets); free(task_counts);
}


/*! \brief Exchange c values for ghost particles.
 *  Each task sends c values for particles requested by other tasks,
 *  and receives c values for remote particles it needs.
 */
static void mg_exchange_ghost_c(double *c_local)
{
    int task;
    int total_send = 0, total_recv = MG_GhostCount;
    for(task = 0; task < NTask; task++) total_send += mg_ghost_send_counts[task];

    double *send_buf = NULL, *recv_buf = NULL;
    if(total_send > 0) {
        send_buf = (double *) malloc(total_send * sizeof(double));
        for(task = 0; task < NTask; task++) {
            for(int n = 0; n < mg_ghost_send_counts[task]; n++) {
                int idx = mg_ghost_send_indices[task][n];
                send_buf[mg_ghost_send_offsets[task] + n] = c_local[idx];
            }
        }
    }
    if(total_recv > 0) recv_buf = (double *) malloc(total_recv * sizeof(double));

    MPI_Alltoallv(send_buf, mg_ghost_send_counts, mg_ghost_send_offsets, MPI_DOUBLE,
                  recv_buf, mg_ghost_recv_counts, mg_ghost_recv_offsets, MPI_DOUBLE, MPI_COMM_WORLD);

    /* store received c values in the ghost list */
    for(int n = 0; n < MG_GhostCount; n++) MG_GhostList[n].c_value = recv_buf[n];

    if(send_buf) free(send_buf);
    if(recv_buf) free(recv_buf);
}


static void mg_cleanup_ghost_exchange(void)
{
    int task;
    if(mg_ghost_send_indices) {
        for(task = 0; task < NTask; task++) if(mg_ghost_send_indices[task]) free(mg_ghost_send_indices[task]);
        free(mg_ghost_send_indices); mg_ghost_send_indices = NULL;
    }
    if(MG_GhostList) { free(MG_GhostList); MG_GhostList = NULL; }
    if(mg_ghost_send_counts) { free(mg_ghost_send_counts); mg_ghost_send_counts = NULL; }
    if(mg_ghost_recv_counts) { free(mg_ghost_recv_counts); mg_ghost_recv_counts = NULL; }
    if(mg_ghost_send_offsets) { free(mg_ghost_send_offsets); mg_ghost_send_offsets = NULL; }
    if(mg_ghost_recv_offsets) { free(mg_ghost_recv_offsets); mg_ghost_recv_offsets = NULL; }
    MG_GhostCount = 0;
}




/* ==================================================================================== */
/* SSOR preconditioner: z = M^{-1} r                                                     */
/*                                                                                        */
/* Symmetric Gauss-Seidel on the local block (remote entries ignored = block-Jacobi).     */
/* Forward sweep (i=0..n-1) then backward sweep (i=n-1..0), each solving:                 */
/*   z[i] = (r[i] + sum_{local j} Qnorm2_ij * z[j]) / Rdiag[i]                          */
/* using the most recently updated z[j] values.                                           */
/* ==================================================================================== */

static void mg_ssor_precondition(double *r, double *z)
{
    int i, n;
    for(i = 0; i < N_gas; i++) z[i] = 0;

    /* forward sweep */
    for(i = 0; i < N_gas; i++) {
        if(P[i].Type != 0 || MG_Rows[i].Rdiag <= 1.0e-60) continue;
        double sum = r[i];
        for(n = 0; n < MG_Rows[i].n_local; n++) {
            int j = MG_Rows[i].local_entries[n].j;
            sum += MG_Rows[i].local_entries[n].Qnorm2 * z[j]; /* z[j]=0 for j>=i (not yet visited) */
        }
        z[i] = sum / MG_Rows[i].Rdiag;
    }

    /* backward sweep */
    for(i = N_gas - 1; i >= 0; i--) {
        if(P[i].Type != 0 || MG_Rows[i].Rdiag <= 1.0e-60) continue;
        double sum = r[i];
        for(n = 0; n < MG_Rows[i].n_local; n++) {
            int j = MG_Rows[i].local_entries[n].j;
            sum += MG_Rows[i].local_entries[n].Qnorm2 * z[j];
        }
        z[i] = sum / MG_Rows[i].Rdiag;
    }
}


/* ==================================================================================== */
/* CG solver                                                                              */
/* ==================================================================================== */

static void mg_cg_solve(void)
{
    int i, iter;
    double *x  = (double *) calloc(N_gas, sizeof(double));
    double *r  = (double *) calloc(N_gas, sizeof(double));
    double *z  = (double *) calloc(N_gas, sizeof(double));
    double *p  = (double *) calloc(N_gas, sizeof(double));
    double *Ap = (double *) calloc(N_gas, sizeof(double));

    /* Count active gas cells (for mean projection) */
    long long ngas_local = 0;
    for(i = 0; i < N_gas; i++) { if(P[i].Type == 0) ngas_local++; }
    long long ngas_global;
    MPI_Allreduce(&ngas_local, &ngas_global, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    /* R is a graph Laplacian (positive semidefinite, null space = constant vector).
       Project the RHS onto the range of R by subtracting its global mean, ensuring
       consistency. Then project r, z, p each iteration to keep CG in the
       complement of the null space, where R is positive definite. */
    double bsum_local = 0;
    for(i = 0; i < N_gas; i++) { if(P[i].Type == 0) bsum_local += MG_Rows[i].rhs; }
    double bsum_global;
    MPI_Allreduce(&bsum_local, &bsum_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double bmean = bsum_global / (double)ngas_global;
    for(i = 0; i < N_gas; i++) { if(P[i].Type == 0) MG_Rows[i].rhs -= bmean; }

    /* initialize CG: x=0, r=b, z=SSOR^{-1}r, p=z */
    double rz_local = 0, rz_global, bnorm_local = 0, bnorm_global;
    for(i = 0; i < N_gas; i++) {
        if(P[i].Type != 0) continue;
        x[i] = 0;
        r[i] = MG_Rows[i].rhs;
        bnorm_local += r[i] * r[i];
    }
    MPI_Allreduce(&bnorm_local, &bnorm_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if(bnorm_global < 1.0e-60) {
        for(i = 0; i < N_gas; i++) if(P[i].Type == 0) CellP[i].MG_cgcoeff = 0;
        free(Ap); free(p); free(z); free(r); free(x); return;
    }

    /* z = SSOR preconditioner applied to r, then project out mean, set p=z */
    mg_ssor_precondition(r, z);
    double zmean_local = 0, zmean_global;
    for(i = 0; i < N_gas; i++) { if(P[i].Type == 0) zmean_local += z[i]; }
    MPI_Allreduce(&zmean_local, &zmean_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    zmean_global /= (double)ngas_global;
    rz_local = 0;
    for(i = 0; i < N_gas; i++) {
        if(P[i].Type != 0) continue;
        z[i] -= zmean_global;
        p[i] = z[i];
        rz_local += r[i] * z[i];
    }
    MPI_Allreduce(&rz_local, &rz_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* CG iterations */
    double rnorm_global = bnorm_global; /* track for diagnostics outside loop */
    int ghost_cursor;
    for(iter = 0; iter < MG_CG_MAX_ITER; iter++)
    {
        /* ghost exchange for p */
        mg_exchange_ghost_c(p);

        /* Ap = R * p */
        for(i = 0; i < N_gas; i++) {
            if(P[i].Type != 0) { Ap[i] = 0; continue; }
            Ap[i] = MG_Rows[i].Rdiag * p[i];
            for(int n = 0; n < MG_Rows[i].n_local; n++) {
                Ap[i] -= MG_Rows[i].local_entries[n].Qnorm2 * p[MG_Rows[i].local_entries[n].j];
            }
        }
        /* Apply remote contributions from ghost list */
        ghost_cursor = 0;
        for(i = 0; i < N_gas; i++) {
            if(P[i].Type != 0) { ghost_cursor += MG_Rows[i].n_remote; continue; }
            for(int n = 0; n < MG_Rows[i].n_remote; n++) {
                Ap[i] -= MG_Rows[i].remote_entries[n].Qnorm2 * MG_GhostList[ghost_cursor].c_value;
                ghost_cursor++;
            }
        }

        /* alpha = rz / pAp */
        double pAp_local = 0, pAp_global;
        for(i = 0; i < N_gas; i++) { if(P[i].Type != 0) continue; pAp_local += p[i] * Ap[i]; }
        MPI_Allreduce(&pAp_local, &pAp_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if(pAp_global <= 0) break;
        double alpha = rz_global / pAp_global;

        /* x += alpha*p, r -= alpha*Ap */
        double rnorm_local = 0;
        for(i = 0; i < N_gas; i++) {
            if(P[i].Type != 0) continue;
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
            rnorm_local += r[i] * r[i];
        }
        MPI_Allreduce(&rnorm_local, &rnorm_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if(rnorm_global / bnorm_global < MG_CG_TOL * MG_CG_TOL) {
            if(ThisTask == 0) PRINT_STATUS(" ..MG CG converged in %d iterations, |r|/|b| = %g", iter+1, sqrt(rnorm_global/bnorm_global));
            break;
        }

        /* z = SSOR^{-1} r, then project out mean from z */
        mg_ssor_precondition(r, z);
        double zmean_loc = 0, zmean_glob;
        for(i = 0; i < N_gas; i++) { if(P[i].Type == 0) zmean_loc += z[i]; }
        MPI_Allreduce(&zmean_loc, &zmean_glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        zmean_glob /= (double)ngas_global;
        double rz_new_local = 0, rz_new_global;
        for(i = 0; i < N_gas; i++) {
            if(P[i].Type != 0) continue;
            z[i] -= zmean_glob;
            rz_new_local += r[i] * z[i];
        }
        MPI_Allreduce(&rz_new_local, &rz_new_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /* beta, update p, then project out mean from p */
        double beta = rz_new_global / (rz_global + 1.0e-60);
        rz_global = rz_new_global;
        double pmean_loc = 0, pmean_glob;
        for(i = 0; i < N_gas; i++) {
            if(P[i].Type != 0) continue;
            p[i] = z[i] + beta * p[i];
            pmean_loc += p[i];
        }
        MPI_Allreduce(&pmean_loc, &pmean_glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        pmean_glob /= (double)ngas_global;
        for(i = 0; i < N_gas; i++) { if(P[i].Type == 0) p[i] -= pmean_glob; }
    }

    if(iter == MG_CG_MAX_ITER && ThisTask == 0)
        PRINT_STATUS(" ..MG CG did not converge in %d iterations, |r|/|b| = %g", MG_CG_MAX_ITER, sqrt(rnorm_global/bnorm_global));

    /* store solution (subtract mean so c has zero mean — the constant mode is arbitrary) */
    double xmean_local = 0, xmean_global;
    for(i = 0; i < N_gas; i++) { if(P[i].Type == 0) xmean_local += x[i]; }
    MPI_Allreduce(&xmean_local, &xmean_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    xmean_global /= (double)ngas_global;
    for(i = 0; i < N_gas; i++) {
        if(P[i].Type != 0) continue;
        CellP[i].MG_cgcoeff = x[i] - xmean_global;
#ifdef DIVBCLEANING_DEDNER
        CellP[i].Phi = CellP[i].PhiPred = CellP[i].DtPhi = 0; /* zero the dedner potential if this is used, since that will add a now-spurious source term to the divergences */
#endif
    }

    free(Ap); free(p); free(z); free(r); free(x);
}


/* ==================================================================================== */
/* Hypre AMG-preconditioned PCG solver (default, unless MHD_MODIFIED_GRADIENT_CG_ONLY is defined)                  */
/* ==================================================================================== */

#if !defined(MHD_MODIFIED_GRADIENT_CG_ONLY) && !defined(MHD_MODIFIED_GRADIENT_USE_PARDISO)
static void mg_solve_hypre(void)
{
    int i, n;

    /* global row numbering: task t owns rows [offset_t, offset_t + N_gas_t - 1] */
    long long local_ngas = (long long)N_gas;
    long long global_offset;
    MPI_Exscan(&local_ngas, &global_offset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if(ThisTask == 0) global_offset = 0;

    /* gather all task offsets for converting remote entries to global column indices */
    long long *task_offsets = (long long *) malloc(NTask * sizeof(long long));
    MPI_Allgather(&global_offset, 1, MPI_LONG_LONG, task_offsets, 1, MPI_LONG_LONG, MPI_COMM_WORLD);

    HYPRE_BigInt ilower = (HYPRE_BigInt)global_offset;
    HYPRE_BigInt iupper = (HYPRE_BigInt)(global_offset + N_gas - 1);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
    HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
    HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
#endif

    /* project RHS to zero mean (graph Laplacian null-space removal) */
    long long ngas_active_local = 0;
    double bsum_local = 0;
    for(i = 0; i < N_gas; i++) {
        if(P[i].Type != 0) continue;
        ngas_active_local++;
        bsum_local += MG_Rows[i].rhs;
    }
    long long ngas_active_global;
    double bsum_global;
    MPI_Allreduce(&ngas_active_local, &ngas_active_global, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&bsum_local, &bsum_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double bmean = bsum_global / (double)ngas_active_global;
    for(i = 0; i < N_gas; i++) { if(P[i].Type == 0) MG_Rows[i].rhs -= bmean; }

    /* build Hypre IJ matrix */
    HYPRE_IJMatrix A;
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A);

    /* find max non-zeros per row for temp array sizing */
    int max_nnz = 1;
    for(i = 0; i < N_gas; i++) {
        int nnz = 1 + MG_Rows[i].n_local + MG_Rows[i].n_remote;
        if(nnz > max_nnz) max_nnz = nnz;
    }
    HYPRE_BigInt *col_inds = (HYPRE_BigInt *) malloc(max_nnz * sizeof(HYPRE_BigInt));
    HYPRE_Real *col_vals = (HYPRE_Real *) malloc(max_nnz * sizeof(HYPRE_Real));

    for(i = 0; i < N_gas; i++) {
        HYPRE_BigInt global_row = (HYPRE_BigInt)(global_offset + i);
        HYPRE_Int ncols = 0;

        if(P[i].Type != 0 || MG_Rows[i].Rdiag <= 1.0e-60) {
            /* inactive row: trivial equation c_i = 0 */
            col_inds[0] = global_row;
            col_vals[0] = 1.0;
            ncols = 1;
        } else {
            /* diagonal */
            col_inds[ncols] = global_row;
            col_vals[ncols] = MG_Rows[i].Rdiag;
            ncols++;
            /* local off-diagonal */
            for(n = 0; n < MG_Rows[i].n_local; n++) {
                col_inds[ncols] = (HYPRE_BigInt)(global_offset + MG_Rows[i].local_entries[n].j);
                col_vals[ncols] = -MG_Rows[i].local_entries[n].Qnorm2;
                ncols++;
            }
            /* remote off-diagonal */
            for(n = 0; n < MG_Rows[i].n_remote; n++) {
                int rtask = MG_Rows[i].remote_entries[n].remote_task;
                int ridx = MG_Rows[i].remote_entries[n].remote_index;
                col_inds[ncols] = (HYPRE_BigInt)(task_offsets[rtask] + ridx);
                col_vals[ncols] = -MG_Rows[i].remote_entries[n].Qnorm2;
                ncols++;
            }
        }
        HYPRE_IJMatrixSetValues(A, 1, &ncols, &global_row, col_inds, col_vals);
    }
    HYPRE_IJMatrixAssemble(A);
    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_IJMatrixGetObject(A, (void **)&parcsr_A);

    /* build RHS and initial-guess vectors */
    HYPRE_IJVector b_vec, x_vec;
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b_vec);
    HYPRE_IJVectorSetObjectType(b_vec, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(b_vec);
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &x_vec);
    HYPRE_IJVectorSetObjectType(x_vec, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x_vec);

    HYPRE_BigInt *row_inds = (HYPRE_BigInt *) malloc(N_gas * sizeof(HYPRE_BigInt));
    HYPRE_Real *rhs_vals = (HYPRE_Real *) malloc(N_gas * sizeof(HYPRE_Real));
    HYPRE_Real *x_vals = (HYPRE_Real *) calloc(N_gas, sizeof(HYPRE_Real));
    for(i = 0; i < N_gas; i++) {
        row_inds[i] = (HYPRE_BigInt)(global_offset + i);
        rhs_vals[i] = (P[i].Type == 0) ? MG_Rows[i].rhs : 0.0;
    }
    HYPRE_IJVectorSetValues(b_vec, (HYPRE_Int)N_gas, row_inds, rhs_vals);
    HYPRE_IJVectorSetValues(x_vec, (HYPRE_Int)N_gas, row_inds, x_vals);
    HYPRE_IJVectorAssemble(b_vec);
    HYPRE_IJVectorAssemble(x_vec);
    HYPRE_ParVector par_b, par_x;
    HYPRE_IJVectorGetObject(b_vec, (void **)&par_b);
    HYPRE_IJVectorGetObject(x_vec, (void **)&par_x);

    /* set up PCG solver with BoomerAMG preconditioner */
    HYPRE_Solver solver, precond;
    HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);
    HYPRE_PCGSetMaxIter(solver, (HYPRE_Int)MG_CG_MAX_ITER);
    HYPRE_PCGSetTol(solver, (HYPRE_Real)MG_CG_TOL);
    HYPRE_PCGSetTwoNorm(solver, 1);
    HYPRE_PCGSetPrintLevel(solver, 0);

    HYPRE_BoomerAMGCreate(&precond);
    HYPRE_BoomerAMGSetPrintLevel(precond, 0);
    HYPRE_BoomerAMGSetCoarsenType(precond, 6);  /* Falgout coarsening */
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
    HYPRE_BoomerAMGSetRelaxType(precond, 18);   /* L1-scaled hybrid GS (GPU-parallel) */
#else
    HYPRE_BoomerAMGSetRelaxType(precond, 6);    /* symmetric Gauss-Seidel (CPU) */
#endif
    HYPRE_BoomerAMGSetNumSweeps(precond, 1);
    HYPRE_BoomerAMGSetMaxLevels(precond, 25);
    HYPRE_BoomerAMGSetMaxIter(precond, 1);       /* 1 V-cycle per PCG iteration */
    HYPRE_BoomerAMGSetTol(precond, 0.0);

    HYPRE_PCGSetPrecond(solver,
        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,
        precond);

    HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
    HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);

    /* report convergence */
    HYPRE_Int num_iterations;
    HYPRE_Real final_res_norm;
    HYPRE_PCGGetNumIterations(solver, &num_iterations);
    HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
    if(ThisTask == 0)
        PRINT_STATUS(" ..MG Hypre PCG+AMG converged in %d iterations, |r|/|b| = %g", (int)num_iterations, (double)final_res_norm);

    /* extract solution and subtract mean */
    HYPRE_IJVectorGetValues(x_vec, (HYPRE_Int)N_gas, row_inds, x_vals);
    double xmean_local = 0, xmean_global;
    for(i = 0; i < N_gas; i++) { if(P[i].Type == 0) xmean_local += x_vals[i]; }
    MPI_Allreduce(&xmean_local, &xmean_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    xmean_global /= (double)ngas_active_global;
    for(i = 0; i < N_gas; i++) {
        CellP[i].MG_cgcoeff = (P[i].Type == 0) ? x_vals[i] - xmean_global : 0;
    }

    /* cleanup */
    HYPRE_ParCSRPCGDestroy(solver);
    HYPRE_BoomerAMGDestroy(precond);
    HYPRE_IJMatrixDestroy(A);
    HYPRE_IJVectorDestroy(b_vec);
    HYPRE_IJVectorDestroy(x_vec);
    free(x_vals); free(rhs_vals); free(row_inds);
    free(col_vals); free(col_inds); free(task_offsets);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
    HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
    HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);
#endif
}
#endif /* !defined(MHD_MODIFIED_GRADIENT_CG_ONLY) && !defined(MHD_MODIFIED_GRADIENT_USE_PARDISO) */


/* ==================================================================================== */
/* MKL PARDISO direct solver (optional, requires MHD_MODIFIED_GRADIENT_USE_PARDISO)                          */
/* ==================================================================================== */

#ifdef MHD_MODIFIED_GRADIENT_USE_PARDISO
static void mg_solve_pardiso(void)
{
    int i, n;

    /* global row numbering: task t owns rows [offset_t, offset_t + N_gas_t - 1] */
    long long local_ngas = (long long)N_gas;
    long long global_offset;
    MPI_Exscan(&local_ngas, &global_offset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if(ThisTask == 0) global_offset = 0;

    /* gather all task offsets for converting remote entries to global column indices */
    long long *task_offsets = (long long *) malloc(NTask * sizeof(long long));
    MPI_Allgather(&global_offset, 1, MPI_LONG_LONG, task_offsets, 1, MPI_LONG_LONG, MPI_COMM_WORLD);

    long long ngas_total_ll;
    MPI_Allreduce(&local_ngas, &ngas_total_ll, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    int ngas_global = (int)ngas_total_ll;

    /* project RHS to zero mean (graph Laplacian null-space removal) */
    long long ngas_active_local = 0;
    double bsum_local = 0;
    for(i = 0; i < N_gas; i++) {
        if(P[i].Type != 0) continue;
        ngas_active_local++;
        bsum_local += MG_Rows[i].rhs;
    }
    long long ngas_active_global;
    double bsum_global;
    MPI_Allreduce(&ngas_active_local, &ngas_active_global, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&bsum_local, &bsum_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double bmean = bsum_global / (double)ngas_active_global;
    for(i = 0; i < N_gas; i++) { if(P[i].Type == 0) MG_Rows[i].rhs -= bmean; }

    /* --- Step 1: build local CSR arrays for this task's rows --- */
    /* count non-zeros per local row */
    int local_nnz = 0;
    for(i = 0; i < N_gas; i++) {
        if(P[i].Type != 0 || MG_Rows[i].Rdiag <= 1.0e-60) {
            local_nnz += 1; /* trivial row c_i=0 */
        } else {
            local_nnz += 1 + MG_Rows[i].n_local + MG_Rows[i].n_remote; /* diag + off-diag */
        }
    }

    /* build local CSR triplets: global column index + value, plus local RHS */
    int *local_row_ptr = (int *) malloc((N_gas + 1) * sizeof(int));
    int *local_col_idx = (int *) malloc(local_nnz * sizeof(int));
    double *local_vals = (double *) malloc(local_nnz * sizeof(double));
    double *local_rhs  = (double *) malloc(N_gas * sizeof(double));

    int pos = 0;
    for(i = 0; i < N_gas; i++) {
        local_row_ptr[i] = pos;
        int global_row = (int)(global_offset + i);
        local_rhs[i] = (P[i].Type == 0) ? MG_Rows[i].rhs : 0.0;

        if(P[i].Type != 0 || MG_Rows[i].Rdiag <= 1.0e-60) {
            local_col_idx[pos] = global_row;
            local_vals[pos] = 1.0;
            pos++;
        } else {
            /* diagonal */
            local_col_idx[pos] = global_row;
            local_vals[pos] = MG_Rows[i].Rdiag;
            pos++;
            /* local off-diagonal */
            for(n = 0; n < MG_Rows[i].n_local; n++) {
                local_col_idx[pos] = (int)(global_offset + MG_Rows[i].local_entries[n].j);
                local_vals[pos] = -MG_Rows[i].local_entries[n].Qnorm2;
                pos++;
            }
            /* remote off-diagonal */
            for(n = 0; n < MG_Rows[i].n_remote; n++) {
                int rtask = MG_Rows[i].remote_entries[n].remote_task;
                int ridx  = MG_Rows[i].remote_entries[n].remote_index;
                local_col_idx[pos] = (int)(task_offsets[rtask] + ridx);
                local_vals[pos] = -MG_Rows[i].remote_entries[n].Qnorm2;
                pos++;
            }
        }
    }
    local_row_ptr[N_gas] = pos;

    /* --- Step 2: gather CSR data and RHS to rank 0 --- */
    int *all_ngas = NULL, *all_nnz = NULL, *disp_ngas = NULL, *disp_nnz = NULL;
    int *global_row_ptr = NULL, *global_col_idx = NULL;
    double *global_vals = NULL, *global_rhs = NULL, *global_x = NULL;

    if(ThisTask == 0) {
        all_ngas  = (int *) malloc(NTask * sizeof(int));
        all_nnz   = (int *) malloc(NTask * sizeof(int));
        disp_ngas = (int *) malloc(NTask * sizeof(int));
        disp_nnz  = (int *) malloc(NTask * sizeof(int));
    }
    int my_ngas = N_gas, my_nnz = local_nnz;
    MPI_Gather(&my_ngas, 1, MPI_INT, all_ngas, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&my_nnz, 1, MPI_INT, all_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_nnz = 0;
    if(ThisTask == 0) {
        disp_ngas[0] = 0; disp_nnz[0] = 0;
        for(int t = 1; t < NTask; t++) {
            disp_ngas[t] = disp_ngas[t-1] + all_ngas[t-1];
            disp_nnz[t]  = disp_nnz[t-1]  + all_nnz[t-1];
        }
        for(int t = 0; t < NTask; t++) total_nnz += all_nnz[t];
        global_row_ptr = (int *) malloc((ngas_global + 1) * sizeof(int));
        global_col_idx = (int *) malloc(total_nnz * sizeof(int));
        global_vals    = (double *) malloc(total_nnz * sizeof(double));
        global_rhs     = (double *) malloc(ngas_global * sizeof(double));
        global_x       = (double *) malloc(ngas_global * sizeof(double));
    }

    /* gather RHS, col_idx, vals */
    MPI_Gatherv(local_rhs, N_gas, MPI_DOUBLE, global_rhs, all_ngas, disp_ngas, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_col_idx, local_nnz, MPI_INT, global_col_idx, all_nnz, disp_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_vals, local_nnz, MPI_DOUBLE, global_vals, all_nnz, disp_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* gather row_ptr: each task sends N_gas+1 entries, but we need to offset them */
    {
        int *row_ptr_raw = NULL;
        int *all_ngas_p1 = NULL, *disp_ngas_p1 = NULL;
        if(ThisTask == 0) {
            all_ngas_p1  = (int *) malloc(NTask * sizeof(int));
            disp_ngas_p1 = (int *) malloc(NTask * sizeof(int));
            for(int t = 0; t < NTask; t++) all_ngas_p1[t] = all_ngas[t] + 1;
            disp_ngas_p1[0] = 0;
            for(int t = 1; t < NTask; t++) disp_ngas_p1[t] = disp_ngas_p1[t-1] + all_ngas_p1[t-1];
            row_ptr_raw = (int *) malloc((ngas_global + NTask) * sizeof(int));
        }
        MPI_Gatherv(local_row_ptr, N_gas + 1, MPI_INT, row_ptr_raw, all_ngas_p1, disp_ngas_p1, MPI_INT, 0, MPI_COMM_WORLD);
        if(ThisTask == 0) {
            /* reconstruct global row_ptr by offsetting each task's entries */
            int row = 0;
            for(int t = 0; t < NTask; t++) {
                int nnz_offset = disp_nnz[t];
                for(int r = 0; r < all_ngas[t]; r++) {
                    global_row_ptr[row] = row_ptr_raw[disp_ngas_p1[t] + r] + nnz_offset;
                    row++;
                }
            }
            global_row_ptr[ngas_global] = total_nnz;
            free(row_ptr_raw); free(all_ngas_p1); free(disp_ngas_p1);
        }
    }

    free(local_row_ptr); free(local_col_idx); free(local_vals); free(local_rhs);

    /* --- Step 3: solve with PARDISO on rank 0 --- */
    if(ThisTask == 0) {
        /* convert to 1-based indexing for PARDISO */
        for(i = 0; i <= ngas_global; i++) global_row_ptr[i] += 1;
        for(i = 0; i < total_nnz; i++) global_col_idx[i] += 1;

        /* PARDISO control parameters */
        void *pt[64]; for(i = 0; i < 64; i++) pt[i] = NULL;
        MKL_INT iparm[64]; for(i = 0; i < 64; i++) iparm[i] = 0;
        MKL_INT maxfct = 1, mnum = 1, mtype = -2; /* mtype=-2: real symmetric indefinite */
        MKL_INT phase, nrhs = 1, msglvl = 0, error = 0;
        MKL_INT perm_dummy;
        MKL_INT N = (MKL_INT)ngas_global;

        iparm[0]  = 1;   /* use non-default values */
        iparm[1]  = 2;   /* nested dissection from METIS */
        iparm[3]  = 0;   /* no iterative-direct algorithm */
        iparm[4]  = 0;   /* no user fill-in reducing permutation */
        iparm[7]  = 0;   /* max numbers of iterative refinement steps */
        iparm[9]  = 13;  /* perturb the pivot elements with 1E-13 */
        iparm[10] = 1;   /* use nonsymmetric permutation and scaling MPS */
        iparm[12] = 1;   /* maximum weighted matching algorithm (for indefinite) */
        iparm[17] = -1;  /* output: number of nonzeros in the factor LU */
        iparm[18] = -1;  /* output: Mflops for LU factorization */
        iparm[34] = 1;   /* zero-based indexing: OFF (we use 1-based) */

        /* phase 11: symbolic factorization */
        phase = 11;
        PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N, global_vals, global_row_ptr, global_col_idx,
                &perm_dummy, &nrhs, iparm, &msglvl, global_rhs, global_x, &error);
        if(error != 0) { PRINT_STATUS(" ..MG PARDISO symbolic factorization error %d", (int)error); }

        /* phase 22: numerical factorization */
        phase = 22;
        PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N, global_vals, global_row_ptr, global_col_idx,
                &perm_dummy, &nrhs, iparm, &msglvl, global_rhs, global_x, &error);
        if(error != 0) { PRINT_STATUS(" ..MG PARDISO numerical factorization error %d", (int)error); }

        /* phase 33: solve */
        phase = 33;
        PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N, global_vals, global_row_ptr, global_col_idx,
                &perm_dummy, &nrhs, iparm, &msglvl, global_rhs, global_x, &error);
        if(error != 0) { PRINT_STATUS(" ..MG PARDISO solve error %d", (int)error); }

        /* phase -1: release memory */
        phase = -1;
        PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N, global_vals, global_row_ptr, global_col_idx,
                &perm_dummy, &nrhs, iparm, &msglvl, global_rhs, global_x, &error);

        if(ThisTask == 0) PRINT_STATUS(" ..MG PARDISO direct solve complete (N=%d, nnz=%d)", ngas_global, total_nnz);
    }

    /* --- Step 4: scatter solution back to all tasks --- */
    double *local_x = (double *) malloc(N_gas * sizeof(double));
    MPI_Scatterv(global_x, all_ngas, disp_ngas, MPI_DOUBLE, local_x, N_gas, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* subtract mean and store */
    double xmean_local = 0, xmean_global;
    for(i = 0; i < N_gas; i++) { if(P[i].Type == 0) xmean_local += local_x[i]; }
    MPI_Allreduce(&xmean_local, &xmean_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    xmean_global /= (double)ngas_active_global;
    for(i = 0; i < N_gas; i++) {
        CellP[i].MG_cgcoeff = (P[i].Type == 0) ? local_x[i] - xmean_global : 0;
    }

    free(local_x); free(task_offsets);
    if(ThisTask == 0) {
        free(global_row_ptr); free(global_col_idx); free(global_vals);
        free(global_rhs); free(global_x);
        free(all_ngas); free(all_nnz); free(disp_ngas); free(disp_nnz);
    }
}
#endif /* MHD_MODIFIED_GRADIENT_USE_PARDISO */


/* ==================================================================================== */
/* Public entry point                                                                     */
/* ==================================================================================== */

/* ==================================================================================== */
/* Matrix build via a caller-supplied symmetric CSR walk (Step 5 B6).                   */
/*                                                                                        */
/* Pre-condition: the caller has built an active row list and symmetric neighbor list.   */
/* For the global MG solve this row list must cover all local gas cells, not just the    */
/* hydro-active subset, so the Laplacian solve matches the legacy all-gas matrix.        */
/* Ghost cells carry full P/CellP state from the post-gradient ghost refresh.            */
/*                                                                                        */
/* Each pair appears twice in the CSR; we accumulate ONLY target's row per visit.        */
/* No dedup needed — each row's contribution is written exactly once per pair.           */
/* ==================================================================================== */
static void mg_build_matrix_modern(int *mg_active_indices, int mg_num_active,
                                   const neighbor_list_t *mg_neighbor_list)
{
    /* Initialize per-particle matrix rows */
    for(int i = 0; i < N_gas; i++) {
        MG_Rows[i].Rdiag = 0;
        MG_Rows[i].rhs = 0;
        MG_Rows[i].n_local = 0;
        MG_Rows[i].n_remote = 0;
    }

    int *ghost_home_rank  = ghost_get_home_rank();
    int *ghost_home_index = ghost_get_home_index();
    int num_local = ghost_get_num_local();

    /* Walk symmetric CSR; for each row i, accumulate i's row only. */
    for(int aa = 0; aa < mg_num_active; aa++) {
        int i = mg_active_indices[aa];
        if(P[i].Type != 0 || P[i].Mass <= 0 || CellP[i].Density <= 0) continue;
        if(P[i].KernelRadius <= 0) continue;

        double h_i = P[i].KernelRadius;
        double h2_i = h_i * h_i;
        double hinv_i, hinv3_i, hinv4_i;
        kernel_hinv(h_i, &hinv_i, &hinv3_i, &hinv4_i);
        double V_i = P[i].Mass / CellP[i].Density;
        double Bi[3]; for(int k=0;k<3;k++) { Bi[k] = CellP[i].BPred[k] * CellP[i].Density / P[i].Mass; }

        int n_off     = mg_neighbor_list->offsets[aa];
        int n_off_end = mg_neighbor_list->offsets[aa+1];
        for(int nn = n_off; nn < n_off_end; nn++) {
            int j = mg_neighbor_list->neighbors[nn];
            if(P[j].Type != 0 || P[j].Mass <= 0 || CellP[j].Density <= 0) continue;

            struct { Vec3<double> dp; double r, wk_i, wk_j, dwk_i, dwk_j, h_i; } kernel;
            kernel.dp = P[i].Pos - P[j].Pos;
            nearest_xyz(kernel.dp);
            double r2 = kernel.dp.norm_sq();
            double h_j = P[j].KernelRadius;
            if(r2 <= 0 || (r2 >= h2_i && r2 >= h_j * h_j)) continue;
            kernel.r = sqrt(r2);
            kernel.h_i = h_i;

            double u;
            if(kernel.r < h_i) { u = kernel.r * hinv_i; kernel_main(u, hinv3_i, hinv4_i, &kernel.wk_i, &kernel.dwk_i, -1); }
            else { kernel.wk_i = kernel.dwk_i = 0; }
            if(kernel.r < h_j) {
                double hinv_j, hinv3_j, hinv4_j;
                kernel_hinv(h_j, &hinv_j, &hinv3_j, &hinv4_j);
                u = kernel.r * hinv_j;
                kernel_main(u, hinv3_j, hinv4_j, &kernel.wk_j, &kernel.dwk_j, -1);
            } else { kernel.wk_j = kernel.dwk_j = 0; }

            double V_j = P[j].Mass / CellP[j].Density;
            double cnumcrit2 = ((double)CONDITION_NUMBER_DANGER)*((double)CONDITION_NUMBER_DANGER)
                             - CellP[i].ConditionNumber*CellP[i].ConditionNumber;
            Vec3<double> Face_Area_Vec;
            double Face_Area_Norm;
            double Particle_Size_i = pow(P[i].Mass/CellP[i].Density, 1./NUMDIMS);
            double Particle_Size_j = P[j].Get_Particle_Size();
            double rinv_fv = 1.0 / (MIN_REAL_NUMBER + kernel.r);
            double Vi_inv_corr_unused, Vj_inv_corr_unused;
            compute_finitevol_faces(CellP[i], CellP[j], kernel, rinv_fv, r2, V_i, V_j,
                                    Particle_Size_i, Particle_Size_j, cnumcrit2,
                                    Face_Area_Vec, Face_Area_Norm,
                                    Vi_inv_corr_unused, Vj_inv_corr_unused);

            if(Face_Area_Norm <= 0) continue;

            /* Same Qnorm2 formula as legacy */
            double A_dot_dp = dot(Face_Area_Vec, kernel.dp);
            double Qnorm2 = 0.125 * A_dot_dp * A_dot_dp + 1.0e-60;

            /* Accumulate i's row only (j gets its contribution when iterating j separately) */
            MG_Rows[i].Rdiag += Qnorm2;

            double Bj[3]; for(int k=0;k<3;k++) { Bj[k] = CellP[j].BPred[k] * CellP[j].Density / P[j].Mass; }
            double flux = 0;
            for(int k=0;k<3;k++) {
                double Bface_i = Bi[k], Bface_j = Bj[k];
                for(int k2=0;k2<3;k2++) {
                    Bface_i -= 0.5 * CellP[i].Gradients.B[k][k2] * kernel.dp[k2];
                    Bface_j += 0.5 * CellP[j].Gradients.B[k][k2] * kernel.dp[k2];
                }
                flux += 0.5 * (Bface_i + Bface_j) * Face_Area_Vec[k];
            }
            MG_Rows[i].rhs += flux;

            /* Off-diagonal entry: distinguish local vs ghost neighbor.
             * NOTE: ghost_exchange grows NumPart by total_recv during the ghost
             * window, so ghost array indices are [num_local, num_local+num_ghosts).
             * Use num_local from ghost_get_num_local(), not the inflated NumPart. */
            if(j < N_gas) {
                mg_add_local_entry(i, j, Qnorm2);
            } else {
                int ghost_idx = j - num_local;
                mg_add_remote_entry(i, ghost_home_rank[ghost_idx], ghost_home_index[ghost_idx], Qnorm2);
            }
        }
    }
}


/* Build the MG-private all-local-gas symmetric CSR.  The hydro CSR deliberately
 * follows ActiveParticleList, but MG is a global graph solve: inactive local gas
 * rows must be present so inactive neighbors are not converted into c=0
 * Dirichlet boundaries. */
static void mg_build_allgas_neighbor_list(int **mg_active_indices_out, int *mg_num_active_out,
                                          neighbor_list_t *mg_neighbor_list)
{
    int mg_num_active = 0;
    for(int i = 0; i < N_gas; i++) {
        if(P[i].Type == 0 && P[i].Mass > 0) mg_num_active++;
    }

    int *mg_active_indices = (int *) mymalloc("mg_allgas_active",
        (mg_num_active > 0 ? mg_num_active : 1) * sizeof(int));
    int aa = 0;
    for(int i = 0; i < N_gas; i++) {
        if(P[i].Type == 0 && P[i].Mass > 0) mg_active_indices[aa++] = i;
    }

    gpu_build_symmetric_neighbor_list(P, NumPart, mg_active_indices, mg_num_active,
                                      mg_neighbor_list, 1.0);

    *mg_active_indices_out = mg_active_indices;
    *mg_num_active_out = mg_num_active;
}


/* ==================================================================================== */
/* Validation harness (Step 5 B6.3) — RETIRED 2026-04-28 (Step 5 B6.5).                 */
/* Legacy path is retired; harness no longer needed. Vista field_loop PASS confirmed.    */
/* ==================================================================================== */


void mg_gradient_correction_calc(void)
{
    if(ThisTask == 0) PRINT_STATUS("Computing MG gradient correction (exact div(B)=0) ...");

    /* Allocate per-particle matrix rows */
    MG_Rows = (struct mg_row_data *) calloc(N_gas, sizeof(struct mg_row_data));
    for(int i = 0; i < N_gas; i++) {
        MG_Rows[i].alloc_local = 0;
        MG_Rows[i].alloc_remote = 0;
        MG_Rows[i].local_entries = NULL;
        MG_Rows[i].remote_entries = NULL;
    }

    int *mg_active_indices = NULL;
    int mg_num_active = 0;
    neighbor_list_t mg_neighbor_list = {NULL, NULL, 0, 0};
    mg_build_allgas_neighbor_list(&mg_active_indices, &mg_num_active, &mg_neighbor_list);
    mg_build_matrix_modern(mg_active_indices, mg_num_active, &mg_neighbor_list);
    free_neighbor_list(&mg_neighbor_list);
    myfree(mg_active_indices);

#if defined(MHD_MODIFIED_GRADIENT_USE_PARDISO)
    /* Phase 2: solve via MKL PARDISO direct solver (gather to rank 0) */
    mg_solve_pardiso();
#elif !defined(MHD_MODIFIED_GRADIENT_CG_ONLY)
    /* Phase 2: solve via Hypre PCG + BoomerAMG (handles its own MPI communication) */
    mg_solve_hypre();
#else
    mg_setup_ghost_exchange(); /* Phase 2: set up ghost exchange for CG */
    mg_cg_solve(); /* Phase 3: solve R*c = b via SSOR-preconditioned CG */
    mg_cleanup_ghost_exchange(); /* Cleanup ghost exchange */
#endif

    for(int i = 0; i < N_gas; i++) {
        if(MG_Rows[i].local_entries) free(MG_Rows[i].local_entries);
        if(MG_Rows[i].remote_entries) free(MG_Rows[i].remote_entries);
    }
    free(MG_Rows); MG_Rows = NULL;

    if(ThisTask == 0) PRINT_STATUS(" ..MG correction complete.");
}

#endif /* MHD_MODIFIED_GRADIENT */
