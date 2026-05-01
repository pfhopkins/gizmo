#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "../mesh/ghost_symlist_lifecycle.h"
#include "../mesh/ghost_writeback.h"
#include "galsf_gpu_decls.h"

/*! \file dm_dispersion
 *  \brief smoothing length and velocity dispersion calculation for dark matter particles around gas particles
 *
 *  This file contains a loop modeled on the gas density computation which
 *    determines kernel lengths for dark matter particles around a given set of gas particles;
 *    this is used by the flag GALSF_SUBGRID_WIND_SCALING==2 to estimate the local dark
 *    matter velocity dispersion around a given gas particle, which (in turn) is used to set the
 *    sub-grid wind velocity and mass loading. The loop here needs to be called for these models (note this
 *    in general will require a different smoothing length from, say, the dm-dm force softening, or the
 *    gas kernel length for hydro, hence it requires a whole additional loop, even though the loop is
 *    functionally identical - modulo which particles are used - to the loop for adaptive gravitational softening
 *
 * This file was written by Qirong Zhu, for GIZMO, based on Phil Hopkins's adaptive gravitational softening
 *    routine. It has been modified by Phil on re-merger into the main branch of GIZMO with various optimizations
 *
 */

#ifdef GALSF_SUBGRID_WINDS
#if (GALSF_SUBGRID_WIND_SCALING==2)

int disp_density_isactive(int i);
int disp_density_isactive(int i)
{
    if(P[i].TimeBin < 0) return 0;
    if(P[i].Type > 0) return 0; // only gas particles //
    if(P[i].Mass <= 0) return 0;
    return 1;
}


/*! This function represents the core of the density computation. The target particle may either be local, or reside in the communication buffer. */
/*!   -- this subroutine contains no writes to shared memory -- */
void disp_density(void)
{
    /* initialize variables used below, in particlar the structures we need to call throughout the iteration */
    CPU_Step[CPU_MISC] += measure_time(); double t00_truestart = my_second();
    MyFloat *Left, *Right; double desnumngb=64, desnumngbdev=48; long long ntot; 
    int i, npleft, iter=0, redo_particle;
    Left = (MyFloat *) mymalloc("Left", NumPart * sizeof(MyFloat));
    Right = (MyFloat *) mymalloc("Right", NumPart * sizeof(MyFloat));
    /* initialize anything we need to about the active particles before their loop */
    for (int i : ActiveParticleList) {if(disp_density_isactive(i)) {CellP[i].NumNgbDM = 0; Left[i] = Right[i] = 0;}}

    /* GPU neighbor-list path: prep DM ghosts once (ghost_exchange.cc's effective
       hmax already includes KernelRadiusDM via Infra-3), then per-iteration
       build a fresh cross-type CSR (gas→DM) and accumulate on GPU. */
    double timeall=0, timecomp=0, timecomm=0, timewait=0, t0;
    CPU_Step[CPU_MISC] += measure_time(); t0 = my_second();
    double disp_ghost_safety = gizmo_ghost_safety_factor();
    gizmo_density_prep_ghosts(disp_ghost_safety);
    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
    do
    {
        /* Build per-iteration active list + radii, run GPU kernel, scatter back */
        int nl_num_active = 0;
        for (int ii : ActiveParticleList) {if(disp_density_isactive(ii)) nl_num_active++;}
        int *nl_active = (int *) mymalloc("disp_nl_active", (nl_num_active > 0 ? nl_num_active : 1) * sizeof(int));
        double *nl_radii = (double *) mymalloc("disp_nl_radii", (nl_num_active > 0 ? nl_num_active : 1) * sizeof(double));
        {int aa = 0; for (int ii : ActiveParticleList) {
            if(disp_density_isactive(ii)) {
                nl_active[aa] = ii;
                nl_radii[aa] = CellP[ii].KernelRadiusDM;
                aa++;
            }
        }}
        /* Zero per-iteration accumulators (only for currently-active gas; inactive kept finalized) */
        for(int aa = 0; aa < nl_num_active; aa++) {
            int ii = nl_active[aa];
            CellP[ii].NumNgbDM = 0; CellP[ii].DM_Vx = 0; CellP[ii].DM_Vy = 0; CellP[ii].DM_Vz = 0; CellP[ii].DM_VelDisp = 0;
        }
        /* Launch GPU kernel */
        struct dispdens_gpu_out *nl_outs = (struct dispdens_gpu_out *) mymalloc("disp_nl_outs",
            (nl_num_active > 0 ? nl_num_active : 1) * sizeof(struct dispdens_gpu_out));
        ghost_write_detector_begin("disp_density");
        disp_density_evaluate_gpu(P, NumPart, nl_active, nl_num_active, nl_radii, nl_outs);
        ghost_write_detector_end();
        /* Scatter results */
        for(int aa = 0; aa < nl_num_active; aa++) {
            int ii = nl_active[aa];
            CellP[ii].NumNgbDM  = nl_outs[aa].Ngb;
            CellP[ii].DM_Vx     = nl_outs[aa].DM_Vx;
            CellP[ii].DM_Vy     = nl_outs[aa].DM_Vy;
            CellP[ii].DM_Vz     = nl_outs[aa].DM_Vz;
            CellP[ii].DM_VelDisp = nl_outs[aa].DM_Vel_Disp;
        }
        myfree(nl_outs); myfree(nl_radii); myfree(nl_active);

        /* do check on whether we have enough neighbors, and iterate for density-rkern solution */
        double tstart = my_second(), tend;
        npleft = 0; for (int i : ActiveParticleList)
        {
            if(disp_density_isactive(i))
            {
                redo_particle = 0; /* now check whether we have enough neighbours, and are below the maximum search radius */
                double maxsoft = DMIN(All.MaxKernelRadius, 10.0*P[i].KernelRadius);
                if(((CellP[i].NumNgbDM < desnumngb - desnumngbdev) || (CellP[i].NumNgbDM > (desnumngb + desnumngbdev)))
                   && (Right[i]-Left[i] > 0.001*Left[i] || Left[i]==0 || Right[i]==0))
                {
                    redo_particle = 1;
                }
                if(CellP[i].KernelRadiusDM >= maxsoft)
                {
                    CellP[i].KernelRadiusDM = maxsoft;
                    redo_particle = 0;
                }

                if(redo_particle)
                {
                    /* need to redo this particle */
                    npleft++;
                    
                    if(CellP[i].NumNgbDM < desnumngb-desnumngbdev) {Left[i]=DMAX(CellP[i].KernelRadiusDM, Left[i]);}
                    if(CellP[i].NumNgbDM > desnumngb+desnumngbdev) {if(Right[i]>0) {Right[i]=DMIN(Right[i],CellP[i].KernelRadiusDM);} else {Right[i]=CellP[i].KernelRadiusDM;}}
                    
                    if(iter >= MAXITER - 10)
                    {
                        PRINT_WARNING("DM disp: i=%d task=%d ID=%llu Type=%d KernelRadius=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)\n",
                         i, ThisTask, (unsigned long long) P[i].ID, P[i].Type, CellP[i].KernelRadiusDM, Left[i], Right[i], (float) CellP[i].NumNgbDM, Right[i] - Left[i], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
                    }
                    
                    // right/left define upper/lower bounds from previous iterations
                    if(Right[i] > 0 && Left[i] > 0)
                    {
                        // geometric interpolation between right/left //
                        if(CellP[i].NumNgbDM > 1)
                        {
                            CellP[i].KernelRadiusDM *= pow( desnumngb / CellP[i].NumNgbDM , 1./NUMDIMS );
                        } else {
                            CellP[i].KernelRadiusDM *= 2.0;
                        }
                        if((CellP[i].KernelRadiusDM<Right[i])&&(CellP[i].KernelRadiusDM>Left[i]))
                        {
                            CellP[i].KernelRadiusDM = pow(CellP[i].KernelRadiusDM*CellP[i].KernelRadiusDM*CellP[i].KernelRadiusDM*CellP[i].KernelRadiusDM * Left[i]*Right[i] , 1./6.);
                        } else {
                            if(CellP[i].KernelRadiusDM>Right[i]) CellP[i].KernelRadiusDM=Right[i];
                            if(CellP[i].KernelRadiusDM<Left[i]) CellP[i].KernelRadiusDM=Left[i];
                            CellP[i].KernelRadiusDM = pow(CellP[i].KernelRadiusDM * Left[i] * Right[i] , 1.0/3.0);
                        }
                    }
                    else
                    {
                        if(Right[i] == 0 && Left[i] == 0)
                        {
                            char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE];
                            snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "DM disp: Right[i] == 0 && Left[i] == 0 && CellP[i].KernelRadiusDM=%g\n", CellP[i].KernelRadiusDM); terminate(buf);
                        }
                        double fac;
                        if(Right[i] == 0 && Left[i] > 0)
                        {
                            if(CellP[i].NumNgbDM > 1) {fac = log( desnumngb / CellP[i].NumNgbDM ) / NUMDIMS;} else {fac=1.4;}
                            if((CellP[i].NumNgbDM < 2*desnumngb)&&(CellP[i].NumNgbDM > 0.1*desnumngb)) {CellP[i].KernelRadiusDM *= exp(fac);} else {CellP[i].KernelRadiusDM *= 1.26;}
                        }
                        
                        if(Right[i] > 0 && Left[i] == 0)
                        {
                            if(CellP[i].NumNgbDM > 1) {fac = log( desnumngb / CellP[i].NumNgbDM ) / NUMDIMS;} else {fac=1.4;}
                            fac = DMAX(fac,-1.535);
                            if((CellP[i].NumNgbDM < 2*desnumngb)&&(CellP[i].NumNgbDM > 0.1*desnumngb)) {CellP[i].KernelRadiusDM *= exp(fac);} else {CellP[i].KernelRadiusDM /= 1.26;}
                        }
                    }
                }
                else {P[i].TimeBin = -P[i].TimeBin - 1;}	/* Mark as inactive */
            } //  if(disp_density_isactive(i))
        } // npleft = 0; for (int i : ActiveParticleList)

        tend = my_second();
        timecomp += timediff(tstart, tend);
        sumup_large_ints(1, &npleft, &ntot);
        if(ntot > 0)
        {
            iter++;
            if(iter > 0 && ThisTask == 0) {if(iter > 10) printf("DM disp: ngb iteration %d: need to repeat for %d%09d particles.\n", iter, (int) (ntot / 1000000000), (int) (ntot % 1000000000));}
            if(iter > MAXITER) {printf("DM disp: failed to converge in neighbour iteration in disp_density()\n"); fflush(stdout); endrun(1155);}
            /* If KernelRadiusDM grew beyond the exchanged ghost hmax, re-exchange
               so next iteration sees the full DM neighbor pool. */
            gizmo_density_redo_ghosts_if_needed(disp_ghost_safety);
        }
    }
    while(ntot > 0);

    /* Tear down ghosts imported at the top of the neighbor-list path */
    if(NTask > 1) {ghost_exchange_cleanup();}
    myfree(Right); myfree(Left);

    /* mark as active again */
    for (int i : ActiveParticleList)
    {
        if(P[i].TimeBin < 0) {P[i].TimeBin = -P[i].TimeBin - 1;}
    }
    
    /* now that we are DONE iterating to find rkern, we can do the REAL final operations on the results */
    for (int i : ActiveParticleList)
    {
        if(disp_density_isactive(i))
        {
            if(CellP[i].NumNgbDM > 0)
            {
                CellP[i].DM_Vx /= CellP[i].NumNgbDM;
                CellP[i].DM_Vy /= CellP[i].NumNgbDM;
                CellP[i].DM_Vz /= CellP[i].NumNgbDM;
                CellP[i].DM_VelDisp /= CellP[i].NumNgbDM;
                CellP[i].DM_VelDisp = (1./All.cf_atime) * sqrt(CellP[i].DM_VelDisp - CellP[i].DM_Vx * CellP[i].DM_Vx - CellP[i].DM_Vy * CellP[i].DM_Vy - CellP[i].DM_Vz * CellP[i].DM_Vz) / 1.732;//	   1d velocity dispersion
            } else {
                if((CellP[i].DM_VelDisp <= 0) || isnan(CellP[i].DM_VelDisp)) {CellP[i].DM_VelDisp = sqrt(P[i].Vel[0]*P[i].Vel[0]+P[i].Vel[1]*P[i].Vel[1]+P[i].Vel[2]*P[i].Vel[2])/All.cf_atime;}
            }
        }
    }
    
    /* collect some timing information */
    double t1; t1 = WallclockTime = my_second(); timeall = timediff(t00_truestart, t1);
    CPU_Step[CPU_AGSDENSCOMPUTE] += timecomp; CPU_Step[CPU_AGSDENSWAIT] += timewait;
    CPU_Step[CPU_AGSDENSCOMM] += timecomm; CPU_Step[CPU_AGSDENSMISC] += timeall - (timecomp + timewait + timecomm);
}


#endif
#endif




