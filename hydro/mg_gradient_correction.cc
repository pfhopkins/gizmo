#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"

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

#define MG_CG_MAX_ITER 60
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

                #include "compute_finitevol_faces.h"

                if(Face_Area_Norm <= 0) continue;

                /* Qnorm2 = |Q_ij|^2 = 0.25 * |dp|^2 * |A|^2 */
                double dp_sq = kernel.dp.norm_sq();
                double A_sq = Face_Area_Vec.norm_sq();
                double Qnorm2 = 0.25 * dp_sq * A_sq + 1.0e-60;

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

                /* store per-neighbor matrix entries */
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
        if(mode == 1) {
            listindex++; if(listindex < NODELISTLENGTH) {
                startnode = MGDataGet[target].NodeList[listindex];
                if(startnode >= 0) startnode = Nodes[startnode].u.d.nextnode;
            }
        }
    }

    if(mode == 0) out2particle_MG(&out, target, 0);
    else MGDataResult[target] = out;

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

    /* Main MPI communication loop — same pattern as gradient sweep but over ALL gas cells */
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
/* Sparse matrix-vector product: y = R * x                                               */
/* ==================================================================================== */

static void mg_matvec(double *x, double *y, double *ghost_c)
{
    int i, n;
    for(i = 0; i < N_gas; i++) {
        if(P[i].Type != 0) { y[i] = 0; continue; }
        y[i] = MG_Rows[i].Rdiag * x[i];
        /* local off-diagonal entries */
        for(n = 0; n < MG_Rows[i].n_local; n++) {
            y[i] -= MG_Rows[i].local_entries[n].Qnorm2 * x[MG_Rows[i].local_entries[n].j];
        }
        /* remote off-diagonal entries (using ghost c values) */
        /* We need to map each remote entry to the ghost list position.
           Since ghost list is ordered (task, then by insertion order), we
           traverse both lists to find matching entries. For efficiency, we
           store a direct index during matrix build. For now, use a simple approach: */
    }
    /* Remote contributions: iterate over ghost list and distribute to particles */
    /* Build a reverse map: for each particle i, for each remote entry n, find the
       ghost list position. This is set up once after matrix build. */
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

    /* initialize CG: x=0, r=b, z=M^{-1}r, p=z */
    double rz_local = 0, rz_global, bnorm_local = 0, bnorm_global;
    for(i = 0; i < N_gas; i++) {
        if(P[i].Type != 0) continue;
        x[i] = 0;
        r[i] = MG_Rows[i].rhs;
        z[i] = (MG_Rows[i].Rdiag > 1.0e-60) ? r[i] / MG_Rows[i].Rdiag : 0;
        p[i] = z[i];
        rz_local += r[i] * z[i];
        bnorm_local += r[i] * r[i];
    }
    MPI_Allreduce(&rz_local, &rz_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&bnorm_local, &bnorm_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if(bnorm_global < 1.0e-60) {
        for(i = 0; i < N_gas; i++) if(P[i].Type == 0) CellP[i].MG_cgcoeff = 0;
        free(Ap); free(p); free(z); free(r); free(x); return;
    }

    /* project out mean from z and p (r already has zero mean from RHS projection) */
    double zmean_local = 0, pmean_local = 0, zmean_global, pmean_global;
    for(i = 0; i < N_gas; i++) { if(P[i].Type != 0) continue; zmean_local += z[i]; pmean_local += p[i]; }
    MPI_Allreduce(&zmean_local, &zmean_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pmean_local, &pmean_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    zmean_global /= (double)ngas_global; pmean_global /= (double)ngas_global;
    rz_local = 0;
    for(i = 0; i < N_gas; i++) {
        if(P[i].Type != 0) continue;
        z[i] -= zmean_global; p[i] -= pmean_global;
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

        /* z = M^{-1}r, then project out mean from z */
        double zmean_loc = 0, zmean_glob;
        for(i = 0; i < N_gas; i++) {
            if(P[i].Type != 0) continue;
            z[i] = (MG_Rows[i].Rdiag > 1.0e-60) ? r[i] / MG_Rows[i].Rdiag : 0;
            zmean_loc += z[i];
        }
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
    }

    free(Ap); free(p); free(z); free(r); free(x);
}


/* ==================================================================================== */
/* Public entry point                                                                     */
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

    /* Phase 1: build sparse matrix via MPI neighbor sweep */
    mg_build_matrix();

    /* Phase 1b: symmetrize cross-boundary entries. After the sweep, task B has
       remote entries "my j has neighbor i on task A" but task A's row i is missing
       the reverse entry. This exchange completes the matrix. */
    mg_symmetrize_remote_entries();

    /* Phase 2: set up ghost exchange for CG */
    mg_setup_ghost_exchange();

    /* Phase 3: solve R*c = b via preconditioned CG */
    mg_cg_solve();

    /* Cleanup */
    mg_cleanup_ghost_exchange();
    for(int i = 0; i < N_gas; i++) {
        if(MG_Rows[i].local_entries) free(MG_Rows[i].local_entries);
        if(MG_Rows[i].remote_entries) free(MG_Rows[i].remote_entries);
    }
    free(MG_Rows); MG_Rows = NULL;

    if(ThisTask == 0) PRINT_STATUS(" ..MG correction complete.");
}

#endif /* MHD_MODIFIED_GRADIENT */
