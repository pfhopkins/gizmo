/* Lightweight tree walk for column density and radiation flux computation.
 *
 * Computes TREE_RAD column densities (N_H, N_H2, N_CO per HEALPix pixel)
 * and GALSF_RESOLVEDISM_G0_VARIABLE radiation fluxes (UV, LW, NUV, OPT)
 * using the existing gravity tree structure. No gravity computation.
 *
 * Decoupled from gravity so ADAPTIVE_TREEFORCE_UPDATE can skip gravity
 * for gas cells without staling radiation/shielding fields.
 *
 * Opening criterion: node angular size vs HEALPix pixel size.
 * Much looser than gravity → fewer node openings → faster walk.
 */

#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "healpix_utils.h"
#ifdef TREE_RAD_H2
#include "../cooling/chemcool/chemcool_consts.h"
#endif

/* Box-wrapping macros — same as forcetree.cc */
#if defined(BOX_PERIODIC) && !defined(GRAVITY_NOT_PERIODIC)
#define GRAVITY_NEAREST_XYZ(x,y,z,sign) NEAREST_XYZ(x,y,z,sign)
#else
#define GRAVITY_NEAREST_XYZ(x,y,z,sign)
#endif

#ifdef TREE_RAD

/* Opening angle threshold: angular size of one HEALPix pixel.
 * Nodes subtending less than this don't need opening for column accuracy. */
#define TREECOL_OPENING_ANGLE2 (4.0*M_PI/NPIX) /* square of opening angle */

/* MPI tags for treecol communication */
#define TAG_TREECOL_A 600
#define TAG_TREECOL_B 601

/* ============================================================
 *  MPI data structures (much smaller than gravdata_in/out)
 * ============================================================ */
struct treecol_data_in {
    MyFloat Pos[3];
    int NodeList[NODELISTLENGTH];
};

struct treecol_data_out {
    MyDouble Projection[NPIX];
#ifdef TREE_RAD_H2
    MyDouble ProjectionH2[NPIX];
    MyDouble ProjectionCO[NPIX];
#endif
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
    MyDouble UV_flux[NPIX];
    MyDouble LW_flux[NPIX];
    MyDouble NUV_flux[NPIX];
    MyDouble OPT_flux[NPIX];
#endif
};

static struct treecol_data_in *TreecolDataIn, *TreecolDataGet;
static struct treecol_data_out *TreecolDataOut, *TreecolDataResult;

/* ============================================================
 *  Core tree walk: columns + fluxes only, no gravity
 * ============================================================ */
static int treecol_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex)
{
    struct NODE *nop = 0;
    int no, nodesinlist = 0, listindex = 0;
    int maxPart = All.MaxPart, maxNodes = MaxNodes;
    long bunchSize = All.BunchSize;
    integertime ti_Current = All.Ti_Current;
    double r2, dx, dy, dz, r, xtmp, pos_x, pos_y, pos_z;
    xtmp = 0;

    /* local HEALPix accumulation arrays */
    double treecol_Projection[NPIX] = {0};
#ifdef TREE_RAD_H2
    double treecol_ProjectionH2[NPIX] = {0}, treecol_ProjectionCO[NPIX] = {0};
    double h2mass = 0, comass = 0;
#endif
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
    double treecol_UV_flux[NPIX] = {0}, treecol_LW_flux[NPIX] = {0};
    double treecol_NUV_flux[NPIX] = {0}, treecol_OPT_flux[NPIX] = {0};
    double uv_lum = 0, lw_lum = 0, nuv_lum = 0, opt_lum = 0;
#endif
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    double gasmass = 0;
#endif

    double shielding_length = All.ShieldingLength / All.cf_atime; /* comoving */
    double shielding_length2 = shielding_length * shielding_length;

    /* set target position */
    if(mode == 0) {
        pos_x = P[target].Pos[0];
        pos_y = P[target].Pos[1];
        pos_z = P[target].Pos[2];
    } else {
        pos_x = TreecolDataGet[target].Pos[0];
        pos_y = TreecolDataGet[target].Pos[1];
        pos_z = TreecolDataGet[target].Pos[2];
    }

    /* start tree walk */
    if(mode == 0) {
        no = maxPart; /* root node */
    } else {
        nodesinlist++;
        no = TreecolDataGet[target].NodeList[0];
        no = Nodes[no].u.d.nextnode;
    }

    while(no >= 0)
    {
        while(no >= 0)
        {
            if(no < maxPart) /* single particle */
            {
                /* drift particle to current time */
                if(P[no].Ti_current != ti_Current) {
#ifdef _OPENMP
#pragma omp critical(_particledrift_treecol_)
#endif
                    { drift_particle(no, ti_Current); }
                }

                dx = P[no].Pos[0] - pos_x;
                dy = P[no].Pos[1] - pos_y;
                dz = P[no].Pos[2] - pos_z;
                GRAVITY_NEAREST_XYZ(dx, dy, dz, -1);
                r2 = dx*dx + dy*dy + dz*dz;

#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
                gasmass = 0;
                if(P[no].Type == 0) { gasmass = P[no].Mass; }
#endif
#ifdef TREE_RAD_H2
                h2mass = 0; comass = 0;
                if(P[no].Type == 0) {
                    h2mass = 2.0 * CellP[no].TracAbund[IH2] * HYDROGEN_MASSFRAC * P[no].Mass;
#if defined(TREE_RAD_CO) && CHEMISTRYNETWORK != 1 && CHEMISTRYNETWORK != 4
                    comass = 28.0 * CellP[no].TracAbund[ICO] * HYDROGEN_MASSFRAC * P[no].Mass;
#endif
                }
#endif
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
                uv_lum = 0; lw_lum = 0; nuv_lum = 0; opt_lum = 0;
                if(P[no].Type == 4 || P[no].Type == 5) {
                    uv_lum = P[no].UV_luminosity; lw_lum = P[no].LW_luminosity;
                    nuv_lum = P[no].NUV_luminosity; opt_lum = P[no].OPT_luminosity;
                }
#endif
                /* accumulate if within shielding length and nonzero */
                if(r2 > 0) {
                    r = sqrt(r2);
                    /* column densities: gas within shielding length */
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
                    if(gasmass > 0 && r < shielding_length) {
                        long iheal;
                        double vec_hp[3] = {dx, dy, dz};
                        vec2pix_ring(NSIDE, vec_hp, &iheal);
                        double area = (4.0*M_PI / NPIX) * r2;
                        treecol_Projection[iheal] += gasmass / area;
#ifdef TREE_RAD_H2
                        treecol_ProjectionH2[iheal] += h2mass / area;
                        treecol_ProjectionCO[iheal] += comass / area;
#endif
                    }
#endif
                    /* radiation fluxes: no distance cutoff (1/r² falloff is natural) */
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
                    if(uv_lum > 0 || lw_lum > 0 || nuv_lum > 0 || opt_lum > 0) {
                        long iheal;
                        double vec_hp[3] = {dx, dy, dz};
                        vec2pix_ring(NSIDE, vec_hp, &iheal);
                        double inv_4pi_r2 = 1.0 / (4.0*M_PI*r2);
                        treecol_UV_flux[iheal] += uv_lum * inv_4pi_r2;
                        treecol_LW_flux[iheal] += lw_lum * inv_4pi_r2;
                        treecol_NUV_flux[iheal] += nuv_lum * inv_4pi_r2;
                        treecol_OPT_flux[iheal] += opt_lum * inv_4pi_r2;
                    }
#endif
                }

                no = Nextnode[no];
            }
            else /* internal node */
            {
                if(no >= maxPart + maxNodes) /* pseudo-particle: export to remote task */
                {
                    if(mode == 0)
                    {
                        int task, nexp;
                        if(exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != target)
                        {
                            exportflag[task] = target;
                            exportnodecount[task] = NODELISTLENGTH;
                        }
                        if(exportnodecount[task] == NODELISTLENGTH)
                        {
                            int exitFlag = 0;
#ifdef _OPENMP
#pragma omp critical(_nexport_treecol_)
#endif
                            {
                                if(Nexport >= bunchSize) {
                                    BufferFullFlag = 1;
                                    exitFlag = 1;
                                } else {
                                    nexp = Nexport;
                                    Nexport++;
                                }
                            }
                            if(exitFlag) { return -1; }
                            exportnodecount[task] = 0;
                            exportindex[task] = nexp;
                            DataIndexTable[nexp].Task = task;
                            DataIndexTable[nexp].Index = target;
                            DataIndexTable[nexp].IndexGet = nexp;
                        }
                        DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] =
                            DomainNodeIndex[no - (maxPart + maxNodes)];
                        if(exportnodecount[task] < NODELISTLENGTH)
                            DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;
                    }
                    no = Nextnode[no - maxNodes];
                    continue;
                }

                /* local internal node */
                nop = &Nodes[no];

                if(mode == 1) {
                    if(nop->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) {
                        no = -1; continue;
                    }
                }

                double nodemass = nop->u.d.mass;
                if(nodemass <= 0) { no = nop->u.d.sibling; continue; }

                /* single-particle node: open it */
                if(!(nop->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES))) {
                    if(nodemass) { no = nop->u.d.nextnode; continue; }
                }

                /* drift node */
                if(nop->Ti_current != ti_Current) {
#ifdef _OPENMP
#pragma omp critical(_nodedrift_treecol_)
#endif
                    { force_drift_node(no, ti_Current); }
                }

                dx = nop->u.d.s[0] - pos_x;
                dy = nop->u.d.s[1] - pos_y;
                dz = nop->u.d.s[2] - pos_z;
                GRAVITY_NEAREST_XYZ(dx, dy, dz, -1);
                r2 = dx*dx + dy*dy + dz*dz;

                /* Skip entire branch if beyond shielding length AND no luminous sources.
                 * For fluxes we need to go further but 1/r² makes distant nodes negligible. */
                int dominated_by_columns = 1;
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
                if(nop->uv_luminosity > 0 || nop->lw_luminosity > 0 ||
                   nop->nuv_luminosity > 0 || nop->opt_luminosity > 0)
                    dominated_by_columns = 0;
#endif
                if(dominated_by_columns && r2 > shielding_length2) {
                    no = nop->u.d.sibling; continue;
                }

                /* Opening criterion: open if node subtends more than one HEALPix pixel */
                if(nop->len * nop->len > r2 * TREECOL_OPENING_ANGLE2) {
                    no = nop->u.d.nextnode; continue;
                }

                /* use this node as-is */
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
                gasmass = nop->gasmass;
#endif
#ifdef TREE_RAD_H2
                h2mass = nop->h2mass;
                comass = nop->comass;
#endif
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
                uv_lum = nop->uv_luminosity;
                lw_lum = nop->lw_luminosity;
                nuv_lum = nop->nuv_luminosity;
                opt_lum = nop->opt_luminosity;
#endif

                if(r2 > 0) {
                    r = sqrt(r2);
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
                    if(gasmass > 0 && r < shielding_length) {
                        long iheal;
                        double vec_hp[3] = {dx, dy, dz};
                        vec2pix_ring(NSIDE, vec_hp, &iheal);
                        double area = (4.0*M_PI / NPIX) * r2;
                        treecol_Projection[iheal] += gasmass / area;
#ifdef TREE_RAD_H2
                        treecol_ProjectionH2[iheal] += h2mass / area;
                        treecol_ProjectionCO[iheal] += comass / area;
#endif
                    }
#endif
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
                    if(uv_lum > 0 || lw_lum > 0 || nuv_lum > 0 || opt_lum > 0) {
                        long iheal;
                        double vec_hp[3] = {dx, dy, dz};
                        vec2pix_ring(NSIDE, vec_hp, &iheal);
                        double inv_4pi_r2 = 1.0 / (4.0*M_PI*r2);
                        treecol_UV_flux[iheal] += uv_lum * inv_4pi_r2;
                        treecol_LW_flux[iheal] += lw_lum * inv_4pi_r2;
                        treecol_NUV_flux[iheal] += nuv_lum * inv_4pi_r2;
                        treecol_OPT_flux[iheal] += opt_lum * inv_4pi_r2;
                    }
#endif
                }

                no = nop->u.d.sibling;
            }
        } /* inner while(no >= 0) */

        if(mode == 1) {
            listindex++;
            if(listindex < NODELISTLENGTH) {
                no = TreecolDataGet[target].NodeList[listindex];
                if(no >= 0) { nodesinlist++; no = Nodes[no].u.d.nextnode; }
            }
        }
    } /* outer while(no >= 0) */

    /* store results */
    if(mode == 0)
    {
        if(P[target].Type == 0) {
            int kp;
            for(kp = 0; kp < NPIX; kp++) CellP[target].Projection[kp] = treecol_Projection[kp];
#ifdef TREE_RAD_H2
            for(kp = 0; kp < NPIX; kp++) { CellP[target].ProjectionH2[kp] = treecol_ProjectionH2[kp]; CellP[target].ProjectionCO[kp] = treecol_ProjectionCO[kp]; }
#endif
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
            for(kp = 0; kp < NPIX; kp++) { CellP[target].UV_flux[kp] = treecol_UV_flux[kp]; CellP[target].LW_flux[kp] = treecol_LW_flux[kp]; CellP[target].NUV_flux[kp] = treecol_NUV_flux[kp]; CellP[target].OPT_flux[kp] = treecol_OPT_flux[kp]; }
#endif
        }
    }
    else
    {
        int kp;
        for(kp = 0; kp < NPIX; kp++) TreecolDataResult[target].Projection[kp] = treecol_Projection[kp];
#ifdef TREE_RAD_H2
        for(kp = 0; kp < NPIX; kp++) { TreecolDataResult[target].ProjectionH2[kp] = treecol_ProjectionH2[kp]; TreecolDataResult[target].ProjectionCO[kp] = treecol_ProjectionCO[kp]; }
#endif
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
        for(kp = 0; kp < NPIX; kp++) { TreecolDataResult[target].UV_flux[kp] = treecol_UV_flux[kp]; TreecolDataResult[target].LW_flux[kp] = treecol_LW_flux[kp]; TreecolDataResult[target].NUV_flux[kp] = treecol_NUV_flux[kp]; TreecolDataResult[target].OPT_flux[kp] = treecol_OPT_flux[kp]; }
#endif
    }

    return 0;
}

/* ============================================================
 *  Primary/secondary loop functions (threaded)
 * ============================================================ */
static void *treecol_primary_loop(void *p)
{
    int i, j, dummy, *exportflag, *exportnodecount, *exportindex, threadid = *(int *)p;
    exportflag = Exportflag + threadid * NTask;
    exportnodecount = Exportnodecount + threadid * NTask;
    exportindex = Exportindex + threadid * NTask;
    for(j = 0; j < NTask; j++) exportflag[j] = -1;

    while(1)
    {
#ifdef _OPENMP
#pragma omp critical(_nexport_treecol_primary_)
#endif
        {
            i = NextParticle;
            NextParticle++;
        }
        if(i >= NumPart) break;

        if(P[i].Type != 0) { ProcessedFlag[i] = 1; continue; } /* only gas needs columns */
        if(!TimeBinActive[P[i].TimeBin]) { ProcessedFlag[i] = 1; continue; }

        if(treecol_evaluate(i, 0, exportflag, exportnodecount, exportindex) < 0) break;
        ProcessedFlag[i] = 1;
    }
    return NULL;
}

static void *treecol_secondary_loop(void *p)
{
    int j, dummy, threadid = *(int *)p;

    while(1)
    {
#ifdef _OPENMP
#pragma omp critical(_nimport_treecol_secondary_)
#endif
        {
            j = NextJ;
            NextJ++;
        }
        if(j >= Nimport) break;
        treecol_evaluate(j, 1, &dummy, &dummy, &dummy);
    }
    return NULL;
}

/* ============================================================
 *  Driver: MPI export/import loop (follows gravtree.cc pattern)
 * ============================================================ */
void treecol_tree(void)
{
    int i, j, k, ndone, ndone_flag, ngrp, recvTask, place, nexp;
    int Nchunks_per_import = 1;
    double tstart, tend;

    if(ThisTask == 0) { printf("TREECOL: Starting column density + flux walk...\n"); }
    tstart = my_second();

    /* reset processed flag for all particles */
    memset(ProcessedFlag, 0, All.MaxPart * sizeof(unsigned char));

    /* allocate communication buffers */
    size_t MyBufferSize = All.BufferSize;
    All.BunchSize = (long)((MyBufferSize * 1024 * 1024) /
        (sizeof(struct data_index) + sizeof(struct data_nodelist) +
         sizeof(struct treecol_data_in) + sizeof(struct treecol_data_out) +
         sizemax(sizeof(struct treecol_data_in), sizeof(struct treecol_data_out))));
    DataIndexTable = (struct data_index *)mymalloc("TcDataIdx", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *)mymalloc("TcDataNL", All.BunchSize * sizeof(struct data_nodelist));

    NextParticle = 0;
    int ndone_before_buff, iter = 0;

    do {
        BufferFullFlag = 0;
        Nexport = 0;
        for(j = 0; j < NTask; j++) { Send_count[j] = 0; Exportflag[j] = -1; }

        /* primary loop: evaluate local particles */
        int currentParticle = NextParticle;
#ifdef _OPENMP
        int nthreads = omp_get_max_threads();
        int threadid[nthreads];
        for(i = 0; i < nthreads; i++) threadid[i] = i;
#pragma omp parallel for schedule(dynamic)
        for(i = 0; i < nthreads; i++)
            treecol_primary_loop(&threadid[i]);
#else
        int threadid_single = 0;
        treecol_primary_loop(&threadid_single);
#endif

        if(BufferFullFlag) {
            int last_index = NextParticle;
            NextParticle = currentParticle;
            while(NextParticle < last_index) {
                if(ProcessedFlag[NextParticle] != 1) break;
                NextParticle++;
            }
        }

        /* sort export list by task */
        qsort(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare);

        /* count exports per task */
        for(j = 0; j < NTask; j++) Send_count[j] = 0;
        for(j = 0; j < Nexport; j++) Send_count[DataIndexTable[j].Task]++;

        MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);

        Nimport = 0;
        for(j = 0; j < NTask; j++) {
            Send_offset[j] = (j == 0) ? 0 : Send_offset[j-1] + Send_count[j-1];
            Recv_offset[j] = (j == 0) ? 0 : Recv_offset[j-1] + Recv_count[j-1];
            Nimport += Recv_count[j];
        }

        /* pack export data */
        TreecolDataIn = (struct treecol_data_in *)mymalloc("TcDataIn", Nexport * sizeof(struct treecol_data_in));
        for(j = 0; j < Nexport; j++) {
            place = DataIndexTable[j].Index;
            for(k = 0; k < 3; k++) TreecolDataIn[j].Pos[k] = P[place].Pos[k];
            memcpy(TreecolDataIn[j].NodeList, DataNodeList[DataIndexTable[j].IndexGet].NodeList, NODELISTLENGTH * sizeof(int));
        }

        /* MPI exchange: send particle data, receive from other tasks */
        TreecolDataGet = (struct treecol_data_in *)mymalloc("TcDataGet", Nimport * sizeof(struct treecol_data_in));
        for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
        {
            recvTask = ThisTask ^ ngrp;
            if(recvTask < NTask) {
                if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0) {
                    MPI_Sendrecv(&TreecolDataIn[Send_offset[recvTask]], Send_count[recvTask] * sizeof(struct treecol_data_in), MPI_BYTE,
                                 recvTask, TAG_TREECOL_A,
                                 &TreecolDataGet[Recv_offset[recvTask]], Recv_count[recvTask] * sizeof(struct treecol_data_in), MPI_BYTE,
                                 recvTask, TAG_TREECOL_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }

        /* secondary loop: evaluate imported particles against local tree */
        TreecolDataResult = (struct treecol_data_out *)mymalloc("TcDataRes", Nimport * sizeof(struct treecol_data_out));
        memset(TreecolDataResult, 0, Nimport * sizeof(struct treecol_data_out));

        NextJ = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
        for(i = 0; i < nthreads; i++)
            treecol_secondary_loop(&threadid[i]);
#else
        treecol_secondary_loop(&threadid_single);
#endif

        /* send results back */
        TreecolDataOut = (struct treecol_data_out *)mymalloc("TcDataOut", Nexport * sizeof(struct treecol_data_out));
        for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
        {
            recvTask = ThisTask ^ ngrp;
            if(recvTask < NTask) {
                if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0) {
                    MPI_Sendrecv(&TreecolDataResult[Recv_offset[recvTask]], Recv_count[recvTask] * sizeof(struct treecol_data_out), MPI_BYTE,
                                 recvTask, TAG_TREECOL_B,
                                 &TreecolDataOut[Send_offset[recvTask]], Send_count[recvTask] * sizeof(struct treecol_data_out), MPI_BYTE,
                                 recvTask, TAG_TREECOL_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }

        /* add imported results to local particles */
        for(j = 0; j < Nexport; j++) {
            place = DataIndexTable[j].Index;
            if(P[place].Type == 0) {
                int kp;
                for(kp = 0; kp < NPIX; kp++) CellP[place].Projection[kp] += TreecolDataOut[j].Projection[kp];
#ifdef TREE_RAD_H2
                for(kp = 0; kp < NPIX; kp++) { CellP[place].ProjectionH2[kp] += TreecolDataOut[j].ProjectionH2[kp]; CellP[place].ProjectionCO[kp] += TreecolDataOut[j].ProjectionCO[kp]; }
#endif
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
                for(kp = 0; kp < NPIX; kp++) { CellP[place].UV_flux[kp] += TreecolDataOut[j].UV_flux[kp]; CellP[place].LW_flux[kp] += TreecolDataOut[j].LW_flux[kp]; CellP[place].NUV_flux[kp] += TreecolDataOut[j].NUV_flux[kp]; CellP[place].OPT_flux[kp] += TreecolDataOut[j].OPT_flux[kp]; }
#endif
            }
        }

        myfree(TreecolDataOut);
        myfree(TreecolDataResult);
        myfree(TreecolDataGet);
        myfree(TreecolDataIn);

        if(NextParticle >= NumPart) { ndone_flag = 1; } else { ndone_flag = 0; }
        MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        iter++;
    }
    while(ndone < NTask);

    myfree(DataNodeList);
    myfree(DataIndexTable);

    tend = my_second();
    if(ThisTask == 0) { printf("TREECOL: Done (%d iterations, %.3g sec)\n", iter, timediff(tstart, tend)); }
}

#endif /* TREE_RAD */
