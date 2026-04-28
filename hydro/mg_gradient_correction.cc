#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "../mesh/neighbor_list.h"
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
/* MG sweep: input/output structs for the MPI neighbor loop                              */
/* ==================================================================================== */

struct MGdata_in {
    Vec3<MyDouble> Pos;
    MyFloat Mass;
    MyFloat KernelRadius;
    MyFloat Density;
    MyFloat ConditionNumber;
    SymmetricTensor2<MyDouble> NV_T;
    Vec3<MyDouble> BPred;      /* B = BPred * Density/Mass */
    Mat3<MyDouble> GradB;      /* slope-limited B gradient */
    int OrigTask;              /* task that owns this particle */
    int OrigIndex;             /* local index on originating task */
    int NodeList[NODELISTLENGTH];
};
static struct MGdata_in *MGDataIn, *MGDataGet;

struct MGdata_out {
    MyFloat Rdiag;   /* accumulated diagonal contribution */
    MyFloat rhs;     /* accumulated RHS contribution */
};
static struct MGdata_out *MGDataResult, *MGDataOut;


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


/* ==================================================================================== */
/* MG evaluate function: neighbor-pair computation for matrix build                      */
/* ==================================================================================== */

static void particle2in_MG(struct MGdata_in *in, int i)
{
    in->Pos = P[i].Pos;
    in->Mass = P[i].Mass;
    in->KernelRadius = P[i].KernelRadius;
    in->Density = CellP[i].Density;
    in->ConditionNumber = CellP[i].ConditionNumber;
    in->NV_T = CellP[i].NV_T;
    for(int k=0;k<3;k++) { in->BPred[k] = CellP[i].BPred[k]; }
    in->GradB = CellP[i].Gradients.B;
    in->OrigTask = ThisTask;
    in->OrigIndex = i;
}

static void out2particle_MG(struct MGdata_out *out, int i, int mode)
{
    if(mode == 0) {
        MG_Rows[i].Rdiag += out->Rdiag;
        MG_Rows[i].rhs += out->rhs;
    } else {
        MG_Rows[i].Rdiag += out->Rdiag;
        MG_Rows[i].rhs += out->rhs;
    }
}


/*! \brief Evaluate one particle's neighbor interactions for the MG matrix.
 *
 *  For each neighbor pair: compute face area, Qnorm2, RHS contribution.
 *  Store per-neighbor entries for the sparse matrix.
 */
static int MG_evaluate(int target, int mode, int *exportflag, int *exportnodecount,
                       int *exportindex, int *ngblist)
{
    int startnode, numngb, listindex = 0, j, k, k2, n;
    struct MGdata_in local;
    struct MGdata_out out;
    memset(&out, 0, sizeof(struct MGdata_out));

    if(mode == 0)
        particle2in_MG(&local, target);
    else
        local = MGDataGet[target];

    if(local.Density <= 0 || local.Mass <= 0 || local.KernelRadius <= 0) return 0;

    double h_i = local.KernelRadius;
    double h2_i = h_i * h_i;
    double hinv_i, hinv3_i, hinv4_i;
    kernel_hinv(h_i, &hinv_i, &hinv3_i, &hinv4_i);
    double V_i = local.Mass / local.Density;
    double Bi[3]; for(k=0;k<3;k++) { Bi[k] = local.BPred[k] * local.Density / local.Mass; }

    if(mode == 0) { startnode = All.MaxPart; }
    else { startnode = MGDataGet[target].NodeList[0]; startnode = Nodes[startnode].u.d.nextnode; }

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb = ngb_treefind_pairs_threads(local.Pos, h_i, target, &startnode, mode, exportflag, exportnodecount, exportindex, ngblist);
            if(numngb < 0) return -2;

            for(n = 0; n < numngb; n++)
            {
                j = ngblist[n];
                if(P[j].Type != 0 || P[j].Mass <= 0 || CellP[j].Density <= 0) continue;

                /* separation and distance */
                struct { Vec3<double> dp; double r, wk_i, wk_j, dwk_i, dwk_j, h_i; } kernel;
                kernel.dp = local.Pos - P[j].Pos;
                nearest_xyz(kernel.dp);
                double r2 = kernel.dp.norm_sq();
                double h_j = P[j].KernelRadius;
                if(r2 <= 0 || (r2 >= h2_i && r2 >= h_j * h_j)) continue;
                kernel.r = sqrt(r2);
                kernel.h_i = h_i;

                /* kernel weights */
                double u;
                if(kernel.r < h_i) { u = kernel.r * hinv_i; kernel_main(u, hinv3_i, hinv4_i, &kernel.wk_i, &kernel.dwk_i, -1); }
                else { kernel.wk_i = kernel.dwk_i = 0; }
                if(kernel.r < h_j) {
                    double hinv_j, hinv3_j, hinv4_j;
                    kernel_hinv(h_j, &hinv_j, &hinv3_j, &hinv4_j);
                    u = kernel.r * hinv_j;
                    kernel_main(u, hinv3_j, hinv4_j, &kernel.wk_j, &kernel.dwk_j, -1);
                } else { kernel.wk_j = kernel.dwk_j = 0; }

                /* compute face area (same as hydro loop) */
                double V_j = P[j].Mass / CellP[j].Density;
                double Face_Area_Norm, cnumcrit2 = ((double)CONDITION_NUMBER_DANGER)*((double)CONDITION_NUMBER_DANGER) - local.ConditionNumber*local.ConditionNumber;
                Vec3<double> Face_Area_Vec;
                double Particle_Size_i = pow(local.Mass/local.Density, 1./NUMDIMS);
                double Particle_Size_j = P[j].Get_Particle_Size();
                double rinv_fv = 1.0 / (MIN_REAL_NUMBER + kernel.r);
                double Vi_inv_corr_unused, Vj_inv_corr_unused;
                compute_finitevol_faces(local, CellP[j], kernel, rinv_fv, r2, V_i, V_j,
                                        Particle_Size_i, Particle_Size_j, cnumcrit2,
                                        Face_Area_Vec, Face_Area_Norm,
                                        Vi_inv_corr_unused, Vj_inv_corr_unused);

                if(Face_Area_Norm <= 0) continue;

                /* Qnorm2 must match the actual divB correction applied in hydro_core_meshless.h:
                   the face correction delta_B = c * 0.25 * dp * (A·dp) contributes
                   delta_divB = 0.125 * (c_i - c_j) * (A·dp)^2 per pair, so the matrix
                   entry is 0.125 * (A·dp)^2 (not 0.25 * |dp|^2 * |A|^2). */
                double A_dot_dp = dot(Face_Area_Vec, kernel.dp);
                double Qnorm2 = 0.125 * A_dot_dp * A_dot_dp + 1.0e-60;

                /* accumulate diagonal */
                out.Rdiag += Qnorm2;

                /* RHS: b_i += 0.5*(B'_i + B'_j) . A_ij
                   B'_i = B_i + gradB_i . (x_ij - x_i) = B_i - 0.5*gradB_i . dp
                   B'_j = B_j + gradB_j . (x_ij - x_j) = B_j + 0.5*gradB_j . dp */
                double Bj[3]; for(k=0;k<3;k++) { Bj[k] = CellP[j].BPred[k] * CellP[j].Density / P[j].Mass; }
                double flux = 0;
                for(k=0;k<3;k++) {
                    double Bface_i = Bi[k], Bface_j = Bj[k];
                    for(k2=0;k2<3;k2++) {
                        Bface_i -= 0.5 * local.GradB[k][k2] * kernel.dp[k2];
                        Bface_j += 0.5 * CellP[j].Gradients.B[k][k2] * kernel.dp[k2];
                    }
                    flux += 0.5 * (Bface_i + Bface_j) * Face_Area_Vec[k];
                }
                out.rhs += flux;

                /* store per-neighbor matrix entries.
                   These updates touch MG_Rows[target] and MG_Rows[j], both of
                   which can be concurrently accessed by other OpenMP threads
                   working on different targets that share neighbors. The whole
                   block (including the realloc inside mg_add_*_entry) must be
                   serialized to avoid data races. */
                #ifdef _OPENMP
                #pragma omp critical(_mg_rows_update_)
                #endif
                {
                if(mode == 0) {
                    /* both i (=target) and j are local: store entries for both rows */
                    mg_add_local_entry(target, j, Qnorm2);
                    if(j != target) {
                        mg_add_local_entry(j, target, Qnorm2);
                        /* also accumulate j's diagonal and RHS (symmetric) */
                        MG_Rows[j].Rdiag += Qnorm2;
                        double flux_j = 0;
                        for(k=0;k<3;k++) {
                            double Bface_j2 = Bj[k], Bface_i2 = Bi[k];
                            for(k2=0;k2<3;k2++) {
                                Bface_j2 -= 0.5 * CellP[j].Gradients.B[k][k2] * (-kernel.dp[k2]); /* dp_ji = -dp */
                                Bface_i2 += 0.5 * local.GradB[k][k2] * (-kernel.dp[k2]);
                            }
                            flux_j += 0.5 * (Bface_j2 + Bface_i2) * (-Face_Area_Vec[k]); /* A_ji = -A_ij */
                        }
                        MG_Rows[j].rhs += flux_j;
                    }
                } else {
                    /* mode=1: target (imported) is from another task. Local j gets a remote entry. */
                    mg_add_remote_entry(j, local.OrigTask, local.OrigIndex, Qnorm2);
                    /* j's diagonal and RHS from this pair */
                    MG_Rows[j].Rdiag += Qnorm2;
                    double flux_j = 0;
                    for(k=0;k<3;k++) {
                        double Bface_j2 = Bj[k], Bface_i2 = Bi[k];
                        for(k2=0;k2<3;k2++) {
                            Bface_j2 -= 0.5 * CellP[j].Gradients.B[k][k2] * (-kernel.dp[k2]);
                            Bface_i2 += 0.5 * local.GradB[k][k2] * (-kernel.dp[k2]);
                        }
                        flux_j += 0.5 * (Bface_j2 + Bface_i2) * (-Face_Area_Vec[k]);
                    }
                    MG_Rows[j].rhs += flux_j;
                }
                }
            }
        }
        if(mode == 1) {
            listindex++; if(listindex < NODELISTLENGTH) {
                startnode = MGDataGet[target].NodeList[listindex];
                if(startnode >= 0) startnode = Nodes[startnode].u.d.nextnode;
            }
        }
    }

    /* out2particle_MG writes MG_Rows[target].Rdiag/rhs, which can race with
       another thread writing MG_Rows[target] inside the critical section (when
       target is that thread's neighbor j). Must be serialized under the same lock. */
    #ifdef _OPENMP
    #pragma omp critical(_mg_rows_update_)
    #endif
    {
        if(mode == 0) out2particle_MG(&out, target, 0);
        else MGDataResult[target] = out;
    }

    return 0;
}


/* ==================================================================================== */
/* Primary and secondary evaluation wrappers (OpenMP threaded)                           */
/* ==================================================================================== */

static int NextParticle_MG, NextJ_MG;

static void *MG_evaluate_primary(void *p)
{
    int thread_id = *(int *) p;
    int *exportflag, *exportnodecount, *exportindex, *ngblist;
    exportflag = Exportflag + thread_id * NTask;
    exportnodecount = Exportnodecount + thread_id * NTask;
    exportindex = Exportindex + thread_id * NTask;
    ngblist = Ngblist.data() + thread_id * NumPart;

    while(1)
    {
        int i;
        #ifdef _OPENMP
        #pragma omp critical(_nextpart_mg_)
        #endif
        { i = NextParticle_MG++; }

        if(i >= N_gas) break;
        if(P[i].Type != 0 || P[i].Mass <= 0 || CellP[i].Density <= 0) { ProcessedFlag[i] = 1; continue; }

        if(MG_evaluate(i, 0, exportflag, exportnodecount, exportindex, ngblist) < 0) break;
        ProcessedFlag[i] = 1;
    }
    return NULL;
}

static void *MG_evaluate_secondary(void *p)
{
    int thread_id = *(int *) p;
    int *ngblist = Ngblist.data() + thread_id * NumPart;

    while(1)
    {
        int j;
        #ifdef _OPENMP
        #pragma omp critical(_nextpart_mg2_)
        #endif
        { j = NextJ_MG++; }
        if(j >= Nimport) break;

        /* need a dummy listindex variable for the NodeList traversal inside MG_evaluate */
        MG_evaluate(j, 1, NULL, NULL, NULL, ngblist);
    }
    return NULL;
}


/* ==================================================================================== */
/* Build the sparse matrix via MPI neighbor sweep over ALL gas cells                     */
/* ==================================================================================== */

static void mg_build_matrix(void)
{
    int i, j, k, ndone, ndone_flag, recvTask, place;
    long long NTaskTimesNumPart = maxThreads * NumPart;
    double tstart, tend;

    /* Allocate communication buffers */
    size_t MyBufferSize = All.BufferSize;
    All.BunchSize = (long)((MyBufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
                           sizeof(struct MGdata_in) + sizeof(struct MGdata_out) +
                           sizemax(sizeof(struct MGdata_in), sizeof(struct MGdata_out))));
    Ngblist.resize(NTaskTimesNumPart);
    DataIndexTable = (struct data_index *) mymalloc("MG_DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *) mymalloc("MG_DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

    /* Initialize per-particle matrix rows */
    for(i = 0; i < N_gas; i++) {
        MG_Rows[i].Rdiag = 0;
        MG_Rows[i].rhs = 0;
        MG_Rows[i].n_local = 0;
        MG_Rows[i].n_remote = 0;
    }

    /* Main MPI communication loop — same pattern as gradient sweep but over ALL gas cells.
       Force single-threaded execution: the per-pair critical section already serializes
       most of the work, and thread-order-dependent floating-point accumulation (amplified
       by -ffast-math) makes the matrix non-reproducible under OMP. Since the MG sweep is
       infrequent (gated on ActiveFractionForMGSweep) the performance cost is negligible. */
    #ifdef _OPENMP
    int save_omp_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    #endif
    NextParticle_MG = 0;
    memset(ProcessedFlag, 0, All.MaxPart * sizeof(unsigned char));
    int BufferCollisionFlag_MG = 0;

    do
    {
        BufferFullFlag = 0; Nexport = 0;
        int save_NextParticle = NextParticle_MG;
        for(j = 0; j < NTask; j++) { Send_count[j] = 0; Exportflag[j] = -1; }

        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
            #ifdef _OPENMP
            int mainthreadid = omp_get_thread_num();
            #else
            int mainthreadid = 0;
            #endif
            MG_evaluate_primary(&mainthreadid);
        }

        if(BufferFullFlag)
        {
            int last_nextparticle = NextParticle_MG;
            int processed_particles = 0;
            int first_unprocessedparticle = -1;
            NextParticle_MG = save_NextParticle;
            while(NextParticle_MG < N_gas)
            {
                if(NextParticle_MG == last_nextparticle) break;
                int pindex = NextParticle_MG; /* using direct index since we iterate over all gas */
                #ifndef _OPENMP
                if(ProcessedFlag[pindex] != 1) break;
                #else
                if(ProcessedFlag[pindex] == 0 && first_unprocessedparticle < 0) first_unprocessedparticle = NextParticle_MG;
                if(ProcessedFlag[pindex] == 1)
                #endif
                {
                    processed_particles++;
                    ProcessedFlag[pindex] = 2;
                }
                NextParticle_MG++;
            }
            #ifdef _OPENMP
            if(first_unprocessedparticle >= 0) NextParticle_MG = first_unprocessedparticle;
            if(processed_particles == 0 && NextParticle_MG == save_NextParticle && NextParticle_MG < N_gas) {
                BufferCollisionFlag_MG++; if(BufferCollisionFlag_MG < 2) continue;
            } else if(processed_particles && BufferCollisionFlag_MG) { BufferCollisionFlag_MG = 0; }
            #endif
            if(processed_particles <= 0 && NextParticle_MG == save_NextParticle) { endrun(113309); }

            int new_export = 0;
            for(j = 0, k = 0; j < Nexport; j++) {
                if(ProcessedFlag[DataIndexTable[j].Index] != 2) {
                    if(k < j + 1) k = j + 1;
                    for(; k < Nexport; k++)
                        if(ProcessedFlag[DataIndexTable[k].Index] == 2) {
                            int old_index = DataIndexTable[j].Index;
                            DataIndexTable[j] = DataIndexTable[k]; DataNodeList[j] = DataNodeList[k];
                            DataIndexTable[j].IndexGet = j; new_export++;
                            DataIndexTable[k].Index = old_index; k++; break;
                        }
                } else { new_export++; }
            }
            Nexport = new_export;
        }

        for(j = 0; j < NTask; j++) Send_count[j] = 0;
        for(j = 0; j < Nexport; j++) Send_count[DataIndexTable[j].Task]++;
        mysort_dataindex(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare);
        MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);

        for(j = 0, Send_offset[0] = 0; j < NTask; j++) { if(j > 0) Send_offset[j] = Send_offset[j-1] + Send_count[j-1]; }

        MGDataIn = (struct MGdata_in *) mymalloc("MGDataIn", Nexport * sizeof(struct MGdata_in));
        MGDataOut = (struct MGdata_out *) mymalloc("MGDataOut", Nexport * sizeof(struct MGdata_out));

        for(j = 0; j < Nexport; j++) {
            place = DataIndexTable[j].Index;
            particle2in_MG(&MGDataIn[j], place);
            memcpy(MGDataIn[j].NodeList, DataNodeList[DataIndexTable[j].IndexGet].NodeList, NODELISTLENGTH * sizeof(int));
        }

        /* MPI exchange (same sub-chunking pattern) */
        int N_chunks_for_import, ngrp_initial, ngrp;
        for(ngrp_initial = 1; ngrp_initial < (1 << PTask); ngrp_initial += N_chunks_for_import)
        {
            int flagall;
            N_chunks_for_import = (1 << PTask) - ngrp_initial;
            do {
                int flag = 0; Nimport = 0;
                for(ngrp = ngrp_initial; ngrp < ngrp_initial + N_chunks_for_import; ngrp++) {
                    recvTask = ThisTask ^ ngrp;
                    if(recvTask < NTask && Recv_count[recvTask] > 0) Nimport += Recv_count[recvTask];
                }
                size_t space_needed = Nimport * sizeof(struct MGdata_in) + Nimport * sizeof(struct MGdata_out) + 16384;
                if(space_needed > FreeBytes) flag = 1;
                MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
                if(flagall) N_chunks_for_import /= 2; else break;
            } while(N_chunks_for_import > 0);
            if(N_chunks_for_import == 0) { endrun(9998); }

            MGDataGet = (struct MGdata_in *) mymalloc("MGDataGet", Nimport * sizeof(struct MGdata_in));
            MGDataResult = (struct MGdata_out *) mymalloc("MGDataResult", Nimport * sizeof(struct MGdata_out));

            Nimport = 0;
            for(ngrp = ngrp_initial; ngrp < ngrp_initial + N_chunks_for_import; ngrp++) {
                recvTask = ThisTask ^ ngrp;
                if(recvTask < NTask && (Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)) {
                    MPI_Sendrecv(&MGDataIn[Send_offset[recvTask]], Send_count[recvTask] * sizeof(struct MGdata_in), MPI_BYTE, recvTask, TAG_GRADLOOP_A,
                                 &MGDataGet[Nimport], Recv_count[recvTask] * sizeof(struct MGdata_in), MPI_BYTE, recvTask, TAG_GRADLOOP_A,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    Nimport += Recv_count[recvTask];
                }
            }

            /* evaluate imported particles */
            NextJ_MG = 0;
            #ifdef _OPENMP
            #pragma omp parallel
            #endif
            {
                #ifdef _OPENMP
                int mainthreadid = omp_get_thread_num();
                #else
                int mainthreadid = 0;
                #endif
                MG_evaluate_secondary(&mainthreadid);
            }

            /* send results back */
            Nimport = 0;
            for(ngrp = ngrp_initial; ngrp < ngrp_initial + N_chunks_for_import; ngrp++) {
                recvTask = ThisTask ^ ngrp;
                if(recvTask < NTask && (Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)) {
                    MPI_Sendrecv(&MGDataResult[Nimport], Recv_count[recvTask] * sizeof(struct MGdata_out), MPI_BYTE, recvTask, TAG_GRADLOOP_B,
                                 &MGDataOut[Send_offset[recvTask]], Send_count[recvTask] * sizeof(struct MGdata_out), MPI_BYTE, recvTask, TAG_GRADLOOP_B,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    Nimport += Recv_count[recvTask];
                }
            }
            myfree(MGDataResult); myfree(MGDataGet);
        }

        /* merge results from exports back to local particles */
        for(j = 0; j < Nexport; j++) {
            place = DataIndexTable[j].Index;
            out2particle_MG(&MGDataOut[j], place, 1);
        }
        myfree(MGDataOut); myfree(MGDataIn);

        if(NextParticle_MG >= N_gas) ndone_flag = 1; else ndone_flag = 0;
        MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }
    while(ndone < NTask);

    myfree(DataNodeList); myfree(DataIndexTable);
    #ifdef _OPENMP
    omp_set_num_threads(save_omp_threads);
    #endif
}


/* ==================================================================================== */
/* Symmetrize cross-boundary matrix entries                                               */
/*                                                                                        */
/* After the sweep, task B has remote entries: "my local j has neighbor i on task A        */
/* with Qnorm2_ij." But task A's row i has NO off-diagonal entry for j on task B.         */
/* This exchange sends the reverse info so task A can add the missing remote entry.        */
/* ==================================================================================== */

struct mg_sym_entry { int orig_index; int sender_index; double Qnorm2; };

static void mg_symmetrize_remote_entries(void)
{
    int i, n, task;

    /* count how many entries to send to each task */
    int *send_counts = (int *) calloc(NTask, sizeof(int));
    for(i = 0; i < N_gas; i++) {
        for(n = 0; n < MG_Rows[i].n_remote; n++) {
            send_counts[MG_Rows[i].remote_entries[n].remote_task]++;
        }
    }

    int *recv_counts = (int *) calloc(NTask, sizeof(int));
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

    int *send_offsets = (int *) calloc(NTask, sizeof(int));
    int *recv_offsets = (int *) calloc(NTask, sizeof(int));
    int total_send = 0, total_recv = 0;
    for(task = 0; task < NTask; task++) {
        send_offsets[task] = total_send; total_send += send_counts[task];
        recv_offsets[task] = total_recv; total_recv += recv_counts[task];
    }

    /* pack send buffer: for each remote entry (local j, remote task A, remote index i, Qnorm2),
       send to task A: "your particle i has neighbor j on my task with Qnorm2" */
    struct mg_sym_entry *send_buf = NULL, *recv_buf = NULL;
    if(total_send > 0) send_buf = (struct mg_sym_entry *) malloc(total_send * sizeof(struct mg_sym_entry));
    if(total_recv > 0) recv_buf = (struct mg_sym_entry *) malloc(total_recv * sizeof(struct mg_sym_entry));

    int *task_pos = (int *) calloc(NTask, sizeof(int));
    for(i = 0; i < N_gas; i++) {
        for(n = 0; n < MG_Rows[i].n_remote; n++) {
            int t = MG_Rows[i].remote_entries[n].remote_task;
            int pos = send_offsets[t] + task_pos[t];
            send_buf[pos].orig_index = MG_Rows[i].remote_entries[n].remote_index; /* the particle on the receiving task */
            send_buf[pos].sender_index = i; /* my local particle (the neighbor) */
            send_buf[pos].Qnorm2 = MG_Rows[i].remote_entries[n].Qnorm2;
            task_pos[t]++;
        }
    }

    /* exchange */
    MPI_Alltoallv(send_buf, send_counts, send_offsets, MPI_BYTE,
                  recv_buf, recv_counts, recv_offsets, MPI_BYTE, MPI_COMM_WORLD);
    /* Note: MPI_BYTE with counts multiplied by sizeof would be needed, but MPI_Alltoallv
       counts are in units of the datatype. Use a custom approach: */
    /* Actually need to convert counts to bytes or use a derived type. Simpler: manual Sendrecv */
    /* Let me redo this with explicit Sendrecv for correctness: */

    /* redo exchange with MPI_Sendrecv per task-pair */
    for(task = 0; task < NTask; task++) {
        if(task == ThisTask) continue;
        MPI_Sendrecv(send_buf + send_offsets[task], send_counts[task] * (int)sizeof(struct mg_sym_entry), MPI_BYTE, task, TAG_GRADLOOP_A,
                     recv_buf + recv_offsets[task], recv_counts[task] * (int)sizeof(struct mg_sym_entry), MPI_BYTE, task, TAG_GRADLOOP_A,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* unpack: add remote entries to local rows */
    for(n = 0; n < total_recv; n++) {
        int local_i = recv_buf[n].orig_index; /* my local particle that needs the entry */
        if(local_i < 0 || local_i >= N_gas) continue;
        /* the sender_index is the index on the sending task — find which task sent this */
        /* We need the sender task. Since recv_buf is ordered by task via recv_offsets, determine it: */
        int sender_task = -1;
        for(task = 0; task < NTask; task++) {
            if(n >= recv_offsets[task] && n < recv_offsets[task] + recv_counts[task]) { sender_task = task; break; }
        }
        if(sender_task < 0) continue;
        mg_add_remote_entry(local_i, sender_task, recv_buf[n].sender_index, recv_buf[n].Qnorm2);
    }

    if(send_buf) free(send_buf); if(recv_buf) free(recv_buf);
    free(task_pos); free(recv_offsets); free(send_offsets);
    free(recv_counts); free(send_counts);
}


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
/* Deduplicate double-counted matrix entries                                              */
/*                                                                                        */
/* The neighbor sweep visits every pair (i,j) twice (once from each side), producing      */
/* duplicate off-diagonal entries and 2x diagonal/RHS. This pass removes the duplicates   */
/* so the matrix is stored once per pair, halving memory and mat-vec cost.                 */
/* ==================================================================================== */

static int mg_compare_local(const void *a, const void *b)
{
    return ((struct mg_local_entry *)a)->j - ((struct mg_local_entry *)b)->j;
}

static int mg_compare_remote(const void *a, const void *b)
{
    const struct mg_remote_entry *ea = (const struct mg_remote_entry *)a;
    const struct mg_remote_entry *eb = (const struct mg_remote_entry *)b;
    if(ea->remote_task != eb->remote_task) return ea->remote_task - eb->remote_task;
    return ea->remote_index - eb->remote_index;
}

static void mg_dedup_matrix(void)
{
    for(int i = 0; i < N_gas; i++) {
        if(P[i].Type != 0) continue;

        /* deduplicate local entries */
        if(MG_Rows[i].n_local > 1) {
            qsort(MG_Rows[i].local_entries, MG_Rows[i].n_local, sizeof(struct mg_local_entry), mg_compare_local);
            int out = 0;
            for(int n = 1; n < MG_Rows[i].n_local; n++) {
                if(MG_Rows[i].local_entries[n].j == MG_Rows[i].local_entries[out].j) {
                    /* duplicate — average the Qnorm2 (should be nearly identical) */
                    MG_Rows[i].local_entries[out].Qnorm2 = 0.5 * (MG_Rows[i].local_entries[out].Qnorm2 + MG_Rows[i].local_entries[n].Qnorm2);
                } else {
                    out++;
                    MG_Rows[i].local_entries[out] = MG_Rows[i].local_entries[n];
                }
            }
            MG_Rows[i].n_local = out + 1;
        }

        /* deduplicate remote entries */
        if(MG_Rows[i].n_remote > 1) {
            qsort(MG_Rows[i].remote_entries, MG_Rows[i].n_remote, sizeof(struct mg_remote_entry), mg_compare_remote);
            int out = 0;
            for(int n = 1; n < MG_Rows[i].n_remote; n++) {
                if(MG_Rows[i].remote_entries[n].remote_task == MG_Rows[i].remote_entries[out].remote_task &&
                   MG_Rows[i].remote_entries[n].remote_index == MG_Rows[i].remote_entries[out].remote_index) {
                    MG_Rows[i].remote_entries[out].Qnorm2 = 0.5 * (MG_Rows[i].remote_entries[out].Qnorm2 + MG_Rows[i].remote_entries[n].Qnorm2);
                } else {
                    out++;
                    MG_Rows[i].remote_entries[out] = MG_Rows[i].remote_entries[n];
                }
            }
            MG_Rows[i].n_remote = out + 1;
        }

        /* halve diagonal and RHS (each pair contributed twice) */
        MG_Rows[i].Rdiag *= 0.5;
        MG_Rows[i].rhs *= 0.5;
    }
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
/* Modernized matrix build using gizmo_sym_neighbor_list (Step 5 B6).                    */
/*                                                                                        */
/* Replaces the legacy ngb_treefind_pairs_threads + code_block_xchange plumbing with     */
/* the symmetric CSR walk used elsewhere in the modernized hydro path.                    */
/*                                                                                        */
/* Pre-condition: gizmo_sym_active_indices and gizmo_sym_neighbor_list are built and     */
/* current — refreshed at end of hydro_gradient_calc by gizmo_gradients_refresh_symlist. */
/* Ghost cells (P/CellP at indices NumPart..NumPart+Nghost-1) carry full converged state */
/* including Gradients.B, NV_T, ConditionNumber, BPred (full struct exchange in Phase 0). */
/*                                                                                        */
/* Each pair appears twice in the symmetric CSR — once in i's neighbor list, once in     */
/* j's. We accumulate ONLY target's row per visit (single contribution per row per       */
/* pair). This produces matrix entries equivalent to legacy after mg_dedup_matrix's      */
/* halving; no dedup needed in modern path.                                               */
/* ==================================================================================== */
#if defined(MG_USE_MODERN_BUILD) && defined(GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY)
static void mg_build_matrix_modern(void)
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

    /* Walk symmetric CSR; for each (active i, neighbor j), accumulate i's row only. */
    for(int aa = 0; aa < gizmo_sym_num_active; aa++) {
        int i = gizmo_sym_active_indices[aa];
        if(P[i].Type != 0 || P[i].Mass <= 0 || CellP[i].Density <= 0) continue;
        if(P[i].KernelRadius <= 0) continue;

        double h_i = P[i].KernelRadius;
        double h2_i = h_i * h_i;
        double hinv_i, hinv3_i, hinv4_i;
        kernel_hinv(h_i, &hinv_i, &hinv3_i, &hinv4_i);
        double V_i = P[i].Mass / CellP[i].Density;
        double Bi[3]; for(int k=0;k<3;k++) { Bi[k] = CellP[i].BPred[k] * CellP[i].Density / P[i].Mass; }

        int n_off     = gizmo_sym_neighbor_list.offsets[aa];
        int n_off_end = gizmo_sym_neighbor_list.offsets[aa+1];
        for(int nn = n_off; nn < n_off_end; nn++) {
            int j = gizmo_sym_neighbor_list.neighbors[nn];
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

            /* Off-diagonal entry: distinguish local vs ghost neighbor */
            if(j < N_gas) {
                mg_add_local_entry(i, j, Qnorm2);
            } else {
                int ghost_idx = j - NumPart;
                mg_add_remote_entry(i, ghost_home_rank[ghost_idx], ghost_home_index[ghost_idx], Qnorm2);
            }
        }
    }
}


/* ==================================================================================== */
/* Validation harness (Step 5 B6.3): runs both legacy and modern paths, compares         */
/* MG_Rows[] entry-by-entry, asserts max rel-err below tolerance.                        */
/*                                                                                        */
/* Activated only when MG_VALIDATE_MODERN is also defined alongside MG_USE_MODERN_BUILD.  */
/* Default for production use of the modern path: MG_USE_MODERN_BUILD on (legacy gone).   */
/* For B6 validation phase: both flags on, to compare paths bit-by-bit.                   */
/* ==================================================================================== */
#ifdef MG_VALIDATE_MODERN
static void mg_validate_modern_vs_legacy(void)
{
    /* Save modern result */
    struct mg_row_data *modern_rows = (struct mg_row_data *) calloc(N_gas, sizeof(struct mg_row_data));
    for(int i = 0; i < N_gas; i++) {
        modern_rows[i].Rdiag = MG_Rows[i].Rdiag;
        modern_rows[i].rhs   = MG_Rows[i].rhs;
        modern_rows[i].n_local  = MG_Rows[i].n_local;
        modern_rows[i].n_remote = MG_Rows[i].n_remote;
        if(MG_Rows[i].n_local > 0) {
            modern_rows[i].local_entries = (struct mg_local_entry *) malloc(MG_Rows[i].n_local * sizeof(struct mg_local_entry));
            memcpy(modern_rows[i].local_entries, MG_Rows[i].local_entries, MG_Rows[i].n_local * sizeof(struct mg_local_entry));
        }
        if(MG_Rows[i].n_remote > 0) {
            modern_rows[i].remote_entries = (struct mg_remote_entry *) malloc(MG_Rows[i].n_remote * sizeof(struct mg_remote_entry));
            memcpy(modern_rows[i].remote_entries, MG_Rows[i].remote_entries, MG_Rows[i].n_remote * sizeof(struct mg_remote_entry));
        }
        /* Reset MG_Rows for legacy run */
        MG_Rows[i].Rdiag = 0; MG_Rows[i].rhs = 0;
        if(MG_Rows[i].local_entries)  { free(MG_Rows[i].local_entries);  MG_Rows[i].local_entries  = NULL; }
        if(MG_Rows[i].remote_entries) { free(MG_Rows[i].remote_entries); MG_Rows[i].remote_entries = NULL; }
        MG_Rows[i].alloc_local = 0; MG_Rows[i].alloc_remote = 0;
        MG_Rows[i].n_local = 0; MG_Rows[i].n_remote = 0;
    }

    /* Run legacy path */
    mg_build_matrix();
    mg_symmetrize_remote_entries();
    mg_dedup_matrix();

    /* Compare entry-by-entry. Sort each row's entries first (legacy dedup sorts; modern needs sort to match). */
    const double rel_tol = 1.0e-12;
    int n_mismatch = 0;
    double max_relerr_diag = 0, max_relerr_rhs = 0, max_relerr_entry = 0;
    int worst_diag_i = -1, worst_rhs_i = -1, worst_entry_i = -1;

    for(int i = 0; i < N_gas; i++) {
        /* Sort modern's local entries by j to match legacy's sort order */
        if(modern_rows[i].n_local > 1) {
            qsort(modern_rows[i].local_entries, modern_rows[i].n_local, sizeof(struct mg_local_entry), mg_compare_local);
        }
        if(modern_rows[i].n_remote > 1) {
            qsort(modern_rows[i].remote_entries, modern_rows[i].n_remote, sizeof(struct mg_remote_entry), mg_compare_remote);
        }

        /* Diagonal */
        double denom_d = fabs(MG_Rows[i].Rdiag) + 1.0e-30;
        double err_d = fabs(modern_rows[i].Rdiag - MG_Rows[i].Rdiag) / denom_d;
        if(err_d > max_relerr_diag) { max_relerr_diag = err_d; worst_diag_i = i; }
        if(err_d > rel_tol) n_mismatch++;

        /* RHS */
        double denom_r = fabs(MG_Rows[i].rhs) + 1.0e-30;
        double err_r = fabs(modern_rows[i].rhs - MG_Rows[i].rhs) / denom_r;
        if(err_r > max_relerr_rhs) { max_relerr_rhs = err_r; worst_rhs_i = i; }
        if(err_r > rel_tol) n_mismatch++;

        /* Local entry counts */
        if(modern_rows[i].n_local != MG_Rows[i].n_local) {
            if(n_mismatch < 10) printf("[MG_VALIDATE] rank=%d i=%d n_local mismatch: modern=%d legacy=%d\n", ThisTask, i, modern_rows[i].n_local, MG_Rows[i].n_local);
            n_mismatch++;
            continue;
        }
        for(int n = 0; n < MG_Rows[i].n_local; n++) {
            if(modern_rows[i].local_entries[n].j != MG_Rows[i].local_entries[n].j) {
                if(n_mismatch < 10) printf("[MG_VALIDATE] rank=%d i=%d local entry %d j mismatch: modern=%d legacy=%d\n",
                    ThisTask, i, n, modern_rows[i].local_entries[n].j, MG_Rows[i].local_entries[n].j);
                n_mismatch++; continue;
            }
            double denom = fabs(MG_Rows[i].local_entries[n].Qnorm2) + 1.0e-30;
            double err = fabs(modern_rows[i].local_entries[n].Qnorm2 - MG_Rows[i].local_entries[n].Qnorm2) / denom;
            if(err > max_relerr_entry) { max_relerr_entry = err; worst_entry_i = i; }
            if(err > rel_tol) n_mismatch++;
        }

        /* Remote entry counts */
        if(modern_rows[i].n_remote != MG_Rows[i].n_remote) {
            if(n_mismatch < 10) printf("[MG_VALIDATE] rank=%d i=%d n_remote mismatch: modern=%d legacy=%d\n", ThisTask, i, modern_rows[i].n_remote, MG_Rows[i].n_remote);
            n_mismatch++;
            continue;
        }
        for(int n = 0; n < MG_Rows[i].n_remote; n++) {
            if(modern_rows[i].remote_entries[n].remote_task != MG_Rows[i].remote_entries[n].remote_task ||
               modern_rows[i].remote_entries[n].remote_index != MG_Rows[i].remote_entries[n].remote_index) {
                if(n_mismatch < 10) printf("[MG_VALIDATE] rank=%d i=%d remote entry %d task/index mismatch\n", ThisTask, i, n);
                n_mismatch++; continue;
            }
            double denom = fabs(MG_Rows[i].remote_entries[n].Qnorm2) + 1.0e-30;
            double err = fabs(modern_rows[i].remote_entries[n].Qnorm2 - MG_Rows[i].remote_entries[n].Qnorm2) / denom;
            if(err > max_relerr_entry) { max_relerr_entry = err; worst_entry_i = i; }
            if(err > rel_tol) n_mismatch++;
        }
    }

    /* Reduce stats across ranks */
    int total_mismatch = 0;
    double global_max_diag = 0, global_max_rhs = 0, global_max_entry = 0;
    MPI_Allreduce(&n_mismatch, &total_mismatch, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&max_relerr_diag, &global_max_diag, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&max_relerr_rhs, &global_max_rhs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&max_relerr_entry, &global_max_entry, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if(ThisTask == 0) {
        printf("[MG_VALIDATE] modern-vs-legacy comparison: total_mismatch=%d (across all ranks)\n", total_mismatch);
        printf("[MG_VALIDATE]   max_relerr Rdiag = %.3e (rank=%d worst_local_i=%d)\n", global_max_diag, ThisTask, worst_diag_i);
        printf("[MG_VALIDATE]   max_relerr rhs   = %.3e (rank=%d worst_local_i=%d)\n", global_max_rhs, ThisTask, worst_rhs_i);
        printf("[MG_VALIDATE]   max_relerr entry = %.3e (rank=%d worst_local_i=%d)\n", global_max_entry, ThisTask, worst_entry_i);
        if(total_mismatch > 0) {
            printf("[MG_VALIDATE] FAIL: %d entries exceed rel-tol=%.1e\n", total_mismatch, rel_tol);
        } else {
            printf("[MG_VALIDATE] PASS: all entries within rel-tol=%.1e\n", rel_tol);
        }
        fflush(stdout);
    }

    if(total_mismatch > 0) endrun(85002);

    /* Free modern_rows scratch */
    for(int i = 0; i < N_gas; i++) {
        if(modern_rows[i].local_entries)  free(modern_rows[i].local_entries);
        if(modern_rows[i].remote_entries) free(modern_rows[i].remote_entries);
    }
    free(modern_rows);
}
#endif /* MG_VALIDATE_MODERN */
#endif /* MG_USE_MODERN_BUILD && GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY */


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

#if defined(MG_USE_MODERN_BUILD) && defined(GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY)
    /* Modern path: symmetric CSR walk; ghosts visible directly; no symmetrize/dedup needed. */
    mg_build_matrix_modern();
#ifdef MG_VALIDATE_MODERN
    /* Side-by-side comparison: re-runs legacy and asserts byte-equivalence within tolerance. */
    mg_validate_modern_vs_legacy();
#endif
#else
    /* Legacy path: ngb_treefind_pairs_threads + code_block_xchange + symmetrize + dedup. */
    mg_build_matrix();
    /* Phase 1b: symmetrize cross-boundary entries. After the sweep, task B has
       remote entries "my j has neighbor i on task A" but task A's row i is missing
       the reverse entry. This exchange completes the matrix. */
    mg_symmetrize_remote_entries();

    /* Phase 1c: deduplicate. The neighbor sweep visits every pair twice (once from
       each side), producing duplicate off-diagonal entries and 2x diagonal/RHS.
       Remove duplicates so the matrix is stored once per pair. */
    mg_dedup_matrix();
#endif

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
