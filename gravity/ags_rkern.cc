#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <map>
#include <vector>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../sidm/dm_fuzzy_flux_functions.h"
#include "../sidm/sidm_core_flux_functions.h"
#include "../sidm/cbe_integrator_flux_functions.h"
#ifdef GRAIN_COLLISIONS
#include "../solids/grain_helper_functions.h"
#endif
#include "../mesh/kernel.h"
#include "../mesh/ghost_symlist_lifecycle.h"
#include "ags_density_gpu.h"
#include "ags_force_gpu.h"
#include "ags_functions.h"
#include "../mesh/ghost_writeback.h"

/*! \file ags_rkern.c
 *  \brief kernel length determination for non-gas particles
 *
 *  This file contains a loop modeled on the gas density computation which 
 *    determines softening lengths (and appropriate correction terms) 
 *    for all particle types, to make softenings fully adaptive
 */
/*
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */


#define AGS_DSOFT_TOL (0.5)    // amount by which softening lengths are allowed to vary in single timesteps //

/*! this routine is called by the adaptive gravitational softening neighbor search and forcetree (for application 
    of the appropriate correction terms), to determine which particle types "talk to" which other particle types 
    (i.e. which particle types you search for to determine the softening radii for gravity). For effectively volume-filling
    fluids like gas or dark matter, it makes sense for this to be 'matched' to particles of the same type. For other 
    particle types like stars or sink particles, it's more ambiguous, and requires some judgement on the part of the user. 
    The routine specifically returns a bitflag which defines all valid particles to which a particle of type 'primary' 
    can 'see': i.e. SUM(2^n), where n are all the particle types desired for neighbor finding,
    so e.g. if you want particle types 0 and 4, set the bitmask = 17 = 1 + 16 = 2^0 + 2^4
 */
int ags_gravity_kernel_shared_BITFLAG(short int particle_type_primary)
{
#ifdef ADAPTIVE_GRAVSOFT_FORALL
    if(!((1 << particle_type_primary) & (ADAPTIVE_GRAVSOFT_FORALL))) {return 0;} /* particle is NOT one of the designated 'adaptive' types */
#endif

#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    if(!((1 << particle_type_primary) & (ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION))) {return ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION;} /* particle is NOT one of the designated 'adaptive' types */
#endif

    if(particle_type_primary == 0) {return 1;} /* gas particles see gas particles */

#if (ADAPTIVE_GRAVSOFT_FORALL & 32) && defined(SINK_PARTICLES)
    if(particle_type_primary == 5) {return 1;} /* sink particle particles are AGS-active, but using sink physics, they see only gas */
#endif
    
#if defined(GALSF) && ( (ADAPTIVE_GRAVSOFT_FORALL & 16) || (ADAPTIVE_GRAVSOFT_FORALL & 8) || (ADAPTIVE_GRAVSOFT_FORALL & 4) )
    if(All.ComovingIntegrationOn) /* stars [4 for cosmo runs, 2+3+4 for non-cosmo runs] are AGS-active and see baryons (any type) */
    {
        if(particle_type_primary == 4) {return 17;} // 2^0+2^4
    } else {
        if((particle_type_primary == 4)||(particle_type_primary == 2)||(particle_type_primary == 3)) {return 29;} // 2^0+2^2+2^3+2^4
    }
#endif
    
#ifdef DM_SIDM
    if((1 << particle_type_primary) & (DM_SIDM)) {return DM_SIDM;} /* SIDM particles see other SIDM particles, regardless of type/mass */
#endif
    
#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
    return (1 << particle_type_primary); /* if we haven't been caught by one of the above checks, we simply return whether or not we see 'ourselves' */
#endif
    
    return 0;
}



#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE

void ags_density(void)
{
    /* initialize variables used below, in particlar the structures we need to call throughout the iteration */
    CPU_Step[CPU_MISC] += measure_time(); double t00_truestart = my_second(); MyFloat *Left, *Right, *AGS_Prev; double fac, fac_lim, desnumngb, desnumngbdev; long long ntot;
    int i, npleft, iter=0, redo_particle, particle_set_to_minrkern_flag = 0, particle_set_to_maxrkern_flag = 0;
    AGS_Prev = (MyFloat *) mymalloc("AGS_Prev", NumPart * sizeof(MyFloat));
    Left = (MyFloat *) mymalloc("Left", NumPart * sizeof(MyFloat));
    Right = (MyFloat *) mymalloc("Right", NumPart * sizeof(MyFloat));
    /* initialize anything we need to about the active particles before their loop */
    for (int i : ActiveParticleList) {
        if(ags_density_isactive(i)) {
            Left[i] = Right[i] = 0; AGS_Prev[i] = P[i].AGS_KernelRadius; P[i].AGS_vsig = 0;
            P[i].wakeup = 0;
      }}

    /* GPU neighbor-list path — bitmask partition + per-group cross-type CSR. */
    double timeall=0, timecomp=0, timecomm=0, timewait=0, t0;
    CPU_Step[CPU_MISC] += measure_time(); t0 = my_second();
    double ags_ghost_safety = gizmo_ghost_safety_factor();
    /* Clear any ghost particles left by density() before starting the AGS ghost
     * exchange.  Without this, density ghosts compound into the AGS base count
     * and the ghost_exchange repeatedly requests 2x+ the local count each redo. */
    if(NTask > 1) {ghost_exchange_cleanup();}
    gizmo_density_prep_ghosts(ags_ghost_safety);
    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
    do
    {
        /* Partition active AGS particles by their shared neighbor-type bitmask.
           In typical cosmological use only one bitmask is active (e.g. DM→DM),
           so this yields one group and one GPU kernel pass. */
        std::map<int, std::vector<int>> bitmask_groups;
        uint64_t local_bm_presence = 0;
        for (int ii : ActiveParticleList) {
            if(ags_density_isactive(ii)) {
                int bm = ags_gravity_kernel_shared_BITFLAG(P[ii].Type);
                if(bm > 0 && bm < 64) { bitmask_groups[bm].push_back(ii); local_bm_presence |= (1ULL << bm); }
            }
        }
        /* Symmetrise bitmask key set across all ranks so every rank enters the
           same number of ghost_writeback_wakeup() MPI collectives. Without this,
           a rank whose local particles all converged has bitmask_groups={} and
           skips the collective, deadlocking the other ranks. */
        uint64_t global_bm_presence = local_bm_presence;
        if(NTask > 1) MPI_Allreduce(&local_bm_presence, &global_bm_presence, 1, MPI_UINT64_T, MPI_BOR, MPI_COMM_WORLD);
        /* Zero per-iteration accumulators for all AGS-active particles */
        for(auto& kv : bitmask_groups) {
            for(int ii : kv.second) {
                P[ii].NumNgb = 0; P[ii].DrkernNgbFactor = 0; P[ii].AGS_zeta = 0;
                P[ii].AGS_vsig = 0; P[ii].Particle_DivVel = 0;
#if defined(AGS_FACE_CALCULATION_IS_ACTIVE)
                for(int a=0;a<3;a++) for(int b=0;b<3;b++) P[ii].NV_T[a][b] = 0;
#endif
            }
        }
        /* Launch one GPU kernel per bitmask group (iterate global union so all
           ranks always call ghost_writeback_wakeup the same number of times). */
        for(int bm = 1; bm < 64; bm++) {
            if(!(global_bm_presence & (1ULL << bm))) continue;
            std::vector<int>& ilist = bitmask_groups[bm];  /* empty if rank has none */
            int nl_num_active = (int)ilist.size();
            int *nl_active = (int *) mymalloc("ags_nl_active", (nl_num_active > 0 ? nl_num_active : 1) * sizeof(int));
            double *nl_radii = (double *) mymalloc("ags_nl_radii", (nl_num_active > 0 ? nl_num_active : 1) * sizeof(double));
            for(int a=0;a<nl_num_active;a++) {nl_active[a] = ilist[a]; nl_radii[a] = P[ilist[a]].AGS_KernelRadius;}
            struct ags_density_gpu_out *nl_outs = (struct ags_density_gpu_out *) mymalloc(
                "ags_nl_outs", (nl_num_active > 0 ? nl_num_active : 1) * sizeof(struct ags_density_gpu_out));
            /* zero P[j].wakeup on ghosts so post-kernel non-zero values are pure deltas
               to reverse-communicate to home ranks (the kernel writes wakeup atomically
               when a ghost satisfies the wakeup condition). */
            ghost_writeback_zero_wakeup();
            ags_density_evaluate_gpu(P, CellP, NumPart, nl_active, nl_num_active, nl_radii, bm, nl_outs);
            ghost_writeback_wakeup();
            for(int a=0;a<nl_num_active;a++) {
                int ii = nl_active[a];
                P[ii].NumNgb          += nl_outs[a].Ngb;
                P[ii].DrkernNgbFactor += nl_outs[a].DrkernNgb;
                P[ii].AGS_zeta        += nl_outs[a].AGS_zeta;
                if(nl_outs[a].AGS_vsig > P[ii].AGS_vsig) P[ii].AGS_vsig = nl_outs[a].AGS_vsig;
                P[ii].Particle_DivVel += nl_outs[a].Particle_DivVel;
#if defined(AGS_FACE_CALCULATION_IS_ACTIVE)
                for(int u=0;u<3;u++) for(int v=0;v<3;v++) P[ii].NV_T[u][v] += nl_outs[a].NV_T[u][v];
#endif
            }
            myfree(nl_outs); myfree(nl_radii); myfree(nl_active);
        }

      /* do check on whether we have enough neighbors, and iterate for density-rkern solution */
        double tstart = my_second(), tend;
        npleft = 0; for (int i : ActiveParticleList)
        {
            if(ags_density_isactive(i))
            {
#ifdef DM_FUZZY
                P[i].AGS_Density = P[i].Mass * P[i].NumNgb;
#endif
                if(P[i].NumNgb > 0)
                {
                    P[i].DrkernNgbFactor *= P[i].AGS_KernelRadius / (NUMDIMS * P[i].NumNgb);
                    P[i].Particle_DivVel /= P[i].NumNgb;
                    /* spherical volume of the Kernel (use this to normalize 'effective neighbor number') */
                    P[i].NumNgb *= VOLUME_NORM_COEFF_FOR_NDIMS * pow(P[i].AGS_KernelRadius,NUMDIMS);
                } else {
                    P[i].NumNgb = P[i].DrkernNgbFactor = P[i].Particle_DivVel = 0;
                }
                
                // inverse of defined volume element (to satisfy constraint implicit in Lagrange multipliers)
                if(P[i].DrkernNgbFactor > -0.9)	/* note: this would be -1 if only a single particle at zero lag is found */
                    P[i].DrkernNgbFactor = 1 / (1 + P[i].DrkernNgbFactor);
                else
                    P[i].DrkernNgbFactor = 1;
                P[i].Particle_DivVel *= P[i].DrkernNgbFactor;
                
                /* now check whether we have enough neighbours */
                redo_particle = 0;
                
                double minsoft = ags_return_minsoft(i);
                double maxsoft = ags_return_maxsoft(i);
                if(All.Time > All.TimeBegin)
                {
                    minsoft = DMAX(minsoft , AGS_Prev[i]*AGS_DSOFT_TOL);
                    maxsoft = DMIN(maxsoft , AGS_Prev[i]/AGS_DSOFT_TOL);
                }
                desnumngb = All.AGS_DesNumNgb;
                desnumngbdev = All.AGS_MaxNumNgbDeviation;
                /* allow the neighbor tolerance to gradually grow as we iterate, so that we don't spend forever trapped in a narrow iteration */
#if defined(AGS_FACE_CALCULATION_IS_ACTIVE)
                double ConditionNumber = do_cbe_nvt_inversion_for_faces(i); // right now we don't do anything with this, but could use to force expansion of search, as in hydro
                if(ConditionNumber > MAX_REAL_NUMBER) {PRINT_WARNING("CNUM for CBE: ThisTask=%d i=%d ConditionNumber=%g desnumngb=%g NumNgb=%g iter=%d NVT=%g/%g/%g/%g/%g/%g AGS_KernelRadius=%g \n",ThisTask,i,ConditionNumber,desnumngb,P[i].NumNgb,iter,P[i].NV_T[0][0],P[i].NV_T[1][1],P[i].NV_T[2][2],P[i].NV_T[0][1],P[i].NV_T[0][2],P[i].NV_T[1][2],P[i].AGS_KernelRadius);}
                if(iter > 10) {desnumngbdev = DMIN( 0.25*desnumngb , desnumngbdev * exp(0.1*log(desnumngb/(16.*desnumngbdev))*((double)iter - 9.)) );}
#else
                if(iter > 4) {desnumngbdev = DMIN( 0.25*desnumngb , desnumngbdev * exp(0.1*log(desnumngb/(16.*desnumngbdev))*((double)iter - 3.)) );}
#endif
                if(All.Time<=All.TimeBegin) {if(desnumngbdev > 0.0005) desnumngbdev=0.0005; if(iter > 50) {desnumngbdev = DMIN( 0.25*desnumngb , desnumngbdev * exp(0.1*log(desnumngb/(16.*desnumngbdev))*((double)iter - 49.)) );}}


                /* check if we are in the 'normal' range between the max/min allowed values */
                if((P[i].NumNgb < (desnumngb - desnumngbdev) && P[i].AGS_KernelRadius < 0.999*maxsoft) ||
                   (P[i].NumNgb > (desnumngb + desnumngbdev) && P[i].AGS_KernelRadius > 1.001*minsoft))
                    redo_particle = 1;
                
                /* check maximum kernel size allowed */
                particle_set_to_maxrkern_flag = 0;
                if((P[i].AGS_KernelRadius >= 0.999*maxsoft) && (P[i].NumNgb < (desnumngb - desnumngbdev)))
                {
                    redo_particle = 0;
                    if(P[i].AGS_KernelRadius == maxsoft)
                    {
                        /* iteration at the maximum value is already complete */
                        particle_set_to_maxrkern_flag = 0;
                    } else {
                        /* ok, the particle needs to be set to the maximum, and (if gas) iterated one more time */
                        redo_particle = 1;
                        P[i].AGS_KernelRadius = maxsoft;
                        particle_set_to_maxrkern_flag = 1;
                    }
                }
                
                /* check minimum kernel size allowed */
                particle_set_to_minrkern_flag = 0;
                if((P[i].AGS_KernelRadius <= 1.001*minsoft) && (P[i].NumNgb > (desnumngb + desnumngbdev)))
                {
                    redo_particle = 0;
                    if(P[i].AGS_KernelRadius == minsoft)
                    {
                        /* this means we've already done an iteration with the MinKernelRadius value, so the
                         neighbor weights, etc, are not going to be wrong; thus we simply stop iterating */
                        particle_set_to_minrkern_flag = 0;
                    } else {
                        /* ok, the particle needs to be set to the minimum, and (if gas) iterated one more time */
                        redo_particle = 1;
                        P[i].AGS_KernelRadius = minsoft;
                        particle_set_to_minrkern_flag = 1;
                    }
                }
                
                if(redo_particle)
                {
                    if(iter >= MAXITER - 10)
                    {
                        PRINT_WARNING("AGS: i=%d task=%d ID=%llu Type=%d KernelRadius=%g Drkern=%g Left=%g Right=%g Ngbs=%g Right-Left=%g maxh_flag=%d minh_flag=%d  minsoft=%g maxsoft=%g desnum=%g desnumtol=%g redo=%d pos=(%g|%g|%g)\n",
                               i, ThisTask, (unsigned long long) P[i].ID, P[i].Type, P[i].AGS_KernelRadius, P[i].DrkernNgbFactor, Left[i], Right[i],
                               (float) P[i].NumNgb, Right[i] - Left[i], particle_set_to_maxrkern_flag, particle_set_to_minrkern_flag, minsoft,
                               maxsoft, desnumngb, desnumngbdev, redo_particle, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
                    }
                    
                    /* need to redo this particle */
                    npleft++;
                    
                    if(Left[i] > 0 && Right[i] > 0)
                        if((Right[i] - Left[i]) < 1.0e-3 * Left[i])
                        {
                            /* this one should be ok */
                            npleft--;
                            P[i].TimeBin = -P[i].TimeBin - 1;	/* Mark as inactive */
                            continue;
                        }
                    
                    if((particle_set_to_maxrkern_flag==0)&&(particle_set_to_minrkern_flag==0))
                    {
                        if(P[i].NumNgb < (desnumngb - desnumngbdev))
                        {
                            Left[i] = DMAX(P[i].AGS_KernelRadius, Left[i]);
                        }
                        else
                        {
                            if(Right[i] != 0)
                            {
                                if(P[i].AGS_KernelRadius < Right[i])
                                    Right[i] = P[i].AGS_KernelRadius;
                            }
                            else
                                Right[i] = P[i].AGS_KernelRadius;
                        }
                        
                        // right/left define upper/lower bounds from previous iterations
                        if(Right[i] > 0 && Left[i] > 0)
                        {
                            // geometric interpolation between right/left //
                            double maxjump=0;
                            if(iter>1) {maxjump = 0.2*log(Right[i]/Left[i]);}
                            if(P[i].NumNgb > 1)
                            {
                                double jumpvar = P[i].DrkernNgbFactor * log( desnumngb / P[i].NumNgb ) / NUMDIMS;
                                if(iter>1) {if(fabs(jumpvar) < maxjump) {if(jumpvar<0) {jumpvar=-maxjump;} else {jumpvar=maxjump;}}}
                                P[i].AGS_KernelRadius *= exp(jumpvar);
                            } else {
                                P[i].AGS_KernelRadius *= 2.0;
                            }
                            if((P[i].AGS_KernelRadius<Right[i])&&(P[i].AGS_KernelRadius>Left[i]))
                            {
                                if(iter > 1)
                                {
                                    double hfac = exp(maxjump);
                                    if(P[i].AGS_KernelRadius > Right[i] / hfac) {P[i].AGS_KernelRadius = Right[i] / hfac;}
                                    if(P[i].AGS_KernelRadius < Left[i] * hfac) {P[i].AGS_KernelRadius = Left[i] * hfac;}
                                }
                            } else {
                                if(P[i].AGS_KernelRadius>Right[i]) P[i].AGS_KernelRadius=Right[i];
                                if(P[i].AGS_KernelRadius<Left[i]) P[i].AGS_KernelRadius=Left[i];
                                P[i].AGS_KernelRadius = pow(P[i].AGS_KernelRadius * Left[i] * Right[i] , 1.0/3.0);
                            }
                        }
                        else
                        {
                            if(Right[i] == 0 && Left[i] == 0)
                            {
                                char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE];
                                snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "AGS: Right[i] == 0 && Left[i] == 0 && P[i].AGS_KernelRadius=%g\n", P[i].AGS_KernelRadius); terminate(buf);
                            }
                            
                            if(Right[i] == 0 && Left[i] > 0)
                            {
                                if (P[i].NumNgb > 1)
                                    fac_lim = log( desnumngb / P[i].NumNgb ) / NUMDIMS; // this would give desnumgb if constant density (+0.231=2x desnumngb)
                                else
                                    fac_lim = 1.4; // factor ~66 increase in N_NGB in constant-density medium
                                
                                if((P[i].NumNgb < 2*desnumngb)&&(P[i].NumNgb > 0.1*desnumngb))
                                {
                                    double slope = P[i].DrkernNgbFactor;
                                    if(iter>2 && slope<1) slope = 0.5*(slope+1);
                                    fac = fac_lim * slope; // account for derivative in making the 'corrected' guess
                                    if(iter>=4)
                                        if(P[i].DrkernNgbFactor==1) fac *= 10; // tries to help with being trapped in small steps
                                    
                                    if(fac < fac_lim+0.231)
                                    {
                                        P[i].AGS_KernelRadius *= exp(fac); // more expensive function, but faster convergence
                                    }
                                    else
                                    {
                                        P[i].AGS_KernelRadius *= exp(fac_lim+0.231);
                                        // fac~0.26 leads to expected doubling of number if density is constant,
                                        //   insert this limiter here b/c we don't want to get *too* far from the answer (which we're close to)
                                    }
                                }
                                else
                                    P[i].AGS_KernelRadius *= exp(fac_lim); // here we're not very close to the 'right' answer, so don't trust the (local) derivatives
                            }
                            
                            if(Right[i] > 0 && Left[i] == 0)
                            {
                                if (P[i].NumNgb > 1)
                                    fac_lim = log( desnumngb / P[i].NumNgb ) / NUMDIMS; // this would give desnumgb if constant density (-0.231=0.5x desnumngb)
                                else
                                    fac_lim = 1.4; // factor ~66 increase in N_NGB in constant-density medium
                                
                                if (fac_lim < -1.535) fac_lim = -1.535; // decreasing N_ngb by factor ~100
                                
                                if((P[i].NumNgb < 2*desnumngb)&&(P[i].NumNgb > 0.1*desnumngb))
                                {
                                    double slope = P[i].DrkernNgbFactor;
                                    if(iter>2 && slope<1) slope = 0.5*(slope+1);
                                    fac = fac_lim * slope; // account for derivative in making the 'corrected' guess
                                    if(iter>=10)
                                        if(P[i].DrkernNgbFactor==1) fac *= 10; // tries to help with being trapped in small steps
                                    
                                    if(fac > fac_lim-0.231)
                                    {
                                        P[i].AGS_KernelRadius *= exp(fac); // more expensive function, but faster convergence
                                    }
                                    else
                                        P[i].AGS_KernelRadius *= exp(fac_lim-0.231); // limiter to prevent --too-- far a jump in a single iteration
                                }
                                else
                                    P[i].AGS_KernelRadius *= exp(fac_lim); // here we're not very close to the 'right' answer, so don't trust the (local) derivatives
                            }
                        } // closes if(Right[i] > 0 && Left[i] > 0) else clause
                        
                    } // closes if[particle_set_to_max/minrkern_flag]
                    /* resets for max/min values */
                    if(P[i].AGS_KernelRadius < minsoft) P[i].AGS_KernelRadius = minsoft;
                    if(particle_set_to_minrkern_flag==1) P[i].AGS_KernelRadius = minsoft;
                    if(P[i].AGS_KernelRadius > maxsoft) P[i].AGS_KernelRadius = maxsoft;
                    if(particle_set_to_maxrkern_flag==1) P[i].AGS_KernelRadius = maxsoft;
                } // closes redo_particle
                else
                    P[i].TimeBin = -P[i].TimeBin - 1;	/* Mark as inactive */
            } //  if(ags_density_isactive(i))
        } // npleft = 0; for (int i : ActiveParticleList)
        
        tend = my_second();
        timecomp += timediff(tstart, tend);
        sumup_large_ints(1, &npleft, &ntot);
        if(ntot > 0)
        {
            iter++;
            if(iter > 10 && ThisTask == 0) {printf("AGS-ngb iteration %d: need to repeat for %d%09d particles.\n", iter, (int) (ntot / 1000000000), (int) (ntot % 1000000000));}
            if(iter > MAXITER) {printf("ags-failed to converge in neighbour iteration in density()\n"); fflush(stdout); endrun(1155);}
            /* If AGS_KernelRadius grew beyond the exchanged ghost hmax, re-exchange */
            gizmo_density_redo_ghosts_if_needed(ags_ghost_safety);
        }
    }
    while(ntot > 0);

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
        if(ags_density_isactive(i))
        {
            if((P[i].Mass>0)&&(P[i].AGS_KernelRadius>0)&&(P[i].NumNgb>0))
            {
                double minsoft = ags_return_minsoft(i);
                double maxsoft = ags_return_maxsoft(i);
                minsoft = DMAX(minsoft , AGS_Prev[i]*AGS_DSOFT_TOL);
                maxsoft = DMIN(maxsoft , AGS_Prev[i]/AGS_DSOFT_TOL);
                if(P[i].AGS_KernelRadius >= maxsoft) {P[i].AGS_zeta = 0;} /* check that we're within the 'valid' range for adaptive softening terms, otherwise zeta=0 */

                double z0 = 0.5 * P[i].AGS_zeta * P[i].AGS_KernelRadius / (NUMDIMS * P[i].Mass * P[i].NumNgb / ( VOLUME_NORM_COEFF_FOR_NDIMS * pow(P[i].AGS_KernelRadius,NUMDIMS) )); // zeta before various prefactors
                double h_eff = 2. * (KERNEL_CORE_SIZE*All.ForceSoftening[P[i].Type]); // force softening defines where Jeans pressure needs to kick in; prefactor = NJeans [=2 here]
                double Prho = 0 * h_eff*h_eff/2.; if(P[i].Particle_DivVel>0) {Prho=-Prho;} // truelove criterion. NJeans[above] , gamma=2 for effective EOS when this dominates, rho=ma*na; h_eff here can be KernelRadius [P/rho~H^-1] or gravsoft_min to really enforce that, as MIN, with P/rho~H^-3; if-check makes it so this term always adds KE to the system, pumping it up
                P[i].AGS_zeta = P[i].Mass*P[i].Mass * P[i].DrkernNgbFactor * ( z0 + Prho ); // force correction, including corrections for adaptive softenings and EOS terms
                P[i].NumNgb = pow(P[i].NumNgb , 1./NUMDIMS); /* convert NGB to the more useful format, NumNgb^(1/NDIMS), which we can use to obtain the corrected particle sizes */
            } else {
                P[i].AGS_zeta = 0; P[i].NumNgb = 0; P[i].AGS_KernelRadius = All.ForceSoftening[P[i].Type];
            }
        }
    }
    myfree(AGS_Prev);

    /* collect some timing information */
    double t1; t1 = WallclockTime = my_second(); timeall = timediff(t00_truestart, t1);
    CPU_Step[CPU_AGSDENSCOMPUTE] += timecomp; CPU_Step[CPU_AGSDENSWAIT] += timewait;
    CPU_Step[CPU_AGSDENSCOMM] += timecomm; CPU_Step[CPU_AGSDENSMISC] += timeall - (timecomp + timewait + timecomm);
}









/* routine to determine if we need to use ags_density to calculate KernelRadius */
int ags_density_isactive(int i)
{
    int default_to_return = 0; // default to not being active - needs to be pro-actively 'activated' by some physics
#ifdef ADAPTIVE_GRAVSOFT_FORALL
    default_to_return = 1;
    if(!((1 << P[i].Type) & (ADAPTIVE_GRAVSOFT_FORALL))) /* particle is NOT one of the designated 'adaptive' types */
    {
        P[i].AGS_KernelRadius = All.ForceSoftening[P[i].Type];
        P[i].AGS_zeta = 0;
        default_to_return = 0;
    } else {default_to_return = 1;} /* particle is AGS-active */
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || (ADAPTIVE_GRAVSOFT_FORALL & 1)
    if(P[i].Type==0)
    {
        P[i].AGS_KernelRadius = P[i].KernelRadius; // gas sees gas, these are identical
        default_to_return = 0; // don't actually need to do the loop //
    }
#endif
#ifdef DM_SIDM
    if((1 << P[i].Type) & (DM_SIDM)) {default_to_return = 1;}
#endif
#if defined(DM_FUZZY) || defined(CBE_INTEGRATOR)
    if(P[i].Type == 1) {default_to_return = 1;}
#endif
    if(P[i].TimeBin < 0) {default_to_return = 0;} /* check our 'marker' for particles which have finished iterating to an KernelRadius solution (if they have, dont do them again) */
    return default_to_return;
}
    

/* routine to return the maximum allowed softening */
double ags_return_maxsoft(int i)
{
    double maxsoft = All.MaxKernelRadius; // user-specified maximum: nothing is allowed to exceed this
#ifdef PMGRID /* Maximum allowed gravitational softening when using the TreePM method. The quantity is given in units of the scale used for the force split (PM_ASMTH) */
    maxsoft = DMIN(maxsoft, 1e3 * 0.5 * All.Asmth[0]); /* no more than 1/2 the size of the largest PM cell, times a 'safety factor' which can be pretty big */
#endif
#if (ADAPTIVE_GRAVSOFT_FORALL & 32) && defined(SINK_PARTICLES) && !defined(SINGLE_STAR_SINK_DYNAMICS)
    if(P[i].Type == 5) {maxsoft = All.SinkMaxAccretionRadius  / All.cf_atime;}   // MaxAccretionRadius is now defined in params.txt in PHYSICAL units
#endif
    return maxsoft;
}

    
/* routine to return the minimum allowed softening */
double ags_return_minsoft(int i)
{
    double minsoft = All.ForceSoftening[P[i].Type]; // this is the user-specified minimum
#if !defined(ADAPTIVE_GRAVSOFT_FORALL)
    minsoft = DMIN(All.MinKernelRadius, minsoft);
#endif
    return minsoft;
}


/* CPU wrappers around the GPU-callable _P forms in ags_functions.h. The
   wrappers exist so existing CPU call sites that rely on the global P stay
   untouched, while GPU kernels and the flux_functions.h templates call the
   _P forms with an explicit particle_data pointer. */
double INLINE_FUNC Get_Particle_Size_AGS(int i) { return Get_Particle_Size_AGS_P(i, P); }
double get_particle_volume_ags(int j) { return get_particle_volume_ags_P(j, P); }


#ifdef AGS_FACE_CALCULATION_IS_ACTIVE

/* --------------------------------------------------------------------------
 Subroutine here exists to calculate the MFM-like effective faces for purposes of face-interaction evaluation
 -------------------------------------------------------------------------- */

/* routine to invert the NV_T matrix after neighbor pass */
double do_cbe_nvt_inversion_for_faces(int i)
{
    /* initialize the matrix to be inverted */
    MyDouble NV_T[3][3], Tinv[3][3]; int j,k; for(j=0;j<3;j++) {for(k=0;k<3;k++) {NV_T[j][k]=P[i].NV_T[j][k];}}
    /* want to work in dimensionless units for defining certain quantities robustly, so normalize out the units */
    double dimensional_NV_T_normalizer = pow( P[i].KernelRadius , 2-NUMDIMS ); /* this has the same dimensions as NV_T here */
    for(j=0;j<3;j++) {for(k=0;k<3;k++) {NV_T[j][k] /= dimensional_NV_T_normalizer;}} /* now NV_T should be dimensionless */
    /* Also, we want to be able to calculate the condition number of the matrix to be inverted, since
        this will tell us how robust our procedure is (and let us know if we need to improve the conditioning) */
    double ConditionNumber=0, ConditionNumber_threshold = 10. * CONDITION_NUMBER_DANGER; /* set a threshold condition number - above this we will 'pre-condition' the matrix for better behavior */
    double trace_initial = NV_T[0][0] + NV_T[1][1] + NV_T[2][2]; /* initial trace of this symmetric, positive-definite matrix; used below as a characteristic value for adding the identity */
    double conditioning_term_to_add = 1.05 * (trace_initial / NUMDIMS) / ConditionNumber_threshold; /* this will be added as a test value if the code does not reach the desired condition number */
    /* now enter an iterative loop to arrive at a -well-conditioned- inversion to use */
    while(1)
    {
        /* initialize the matrix this will go into */
        ConditionNumber = matrix_invert_ndims(NV_T, Tinv); // compute the matrix inverse, and return the condition number
        if(ConditionNumber < ConditionNumber_threshold) {break;} // end loop if we have reached target conditioning for the matrix
        for(j=0;j<NUMDIMS;j++) {NV_T[j][j] += conditioning_term_to_add;} /* add the conditioning term which should make the matrix better-conditioned for subsequent use: this is a normalization times the identity matrix in the relevant number of dimensions */
        conditioning_term_to_add *= 1.2; /* multiply the conditioning term so it will grow and eventually satisfy our criteria */
    } // end of loop broken when condition number is sufficiently small
    for(j=0;j<3;j++) {for(k=0;k<3;k++) {P[i].NV_T[j][k] = Tinv[j][k] / dimensional_NV_T_normalizer;}} // now P[i].NV_T holds the inverted matrix elements //
    return ConditionNumber;
}

#endif





/* ------------------------------------------------------------------------------------------------------
 Everything below here is a giant block to define the sub-routines needed to calculate additional force
  terms for particle types that do not fall into the 'hydro' category.
 -------------------------------------------------------------------------------------------------------- */
int AGSForce_isactive(int i);
int AGSForce_isactive(int i)
{
    if(P[i].TimeBin < 0) return 0; /* check our 'marker' for particles which have finished iterating to an KernelRadius solution (if they have, dont do them again) */
#ifdef DM_SIDM
    if((1 << P[i].Type) & (DM_SIDM)) return 1;
#endif
#if defined(DM_FUZZY) || defined(CBE_INTEGRATOR)
    if(P[i].Type == 1) return 1;
#endif
    return 0; // default to no-action, need to affirm calculation above //
}


void AGSForce_calc(void)
{
    CPU_Step[CPU_MISC] += measure_time(); double t00_truestart = my_second();
    PRINT_STATUS(" ..entering AGS-Force calculation [as hydro loop for non-gas elements]\n");
    /* before doing any operations, need to zero the appropriate memory so we can correctly do pair-wise operations */
#if defined(DM_SIDM)
    {int i; for (int i : ActiveParticleList) {P[i].dtime_sidm = 10.*get_particle_timestep_in_physical(i);}}
#endif
#ifdef CBE_INTEGRATOR
    /* need to zero values for active particles (which will be re-calculated) before they are added below */
    //for (int i : ActiveParticleList) {int k1,k2; for(k1=0;k1<CBE_INTEGRATOR_NBASIS;k1++) {for(k2=0;k2<CBE_INTEGRATOR_NMOMENTS;k2++) {P[i].CBE_basis_moments_dt[k1][k2] = 0;}}}
#endif
    /* GPU neighbor-list path for AGSForce_calc. Partition active particles
       (isactive == 1) by their shared neighbor-type bitmask and launch the
       GPU kernel once per group, same pattern as ags_density(). */
    double timeall = 0, timecomp = 0, timecomm = 0, timewait = 0, t0 = 0;
    CPU_Step[CPU_MISC] += measure_time(); t0 = my_second();
    double ags_ghost_safety = gizmo_ghost_safety_factor();
    gizmo_density_prep_ghosts(ags_ghost_safety);

    std::map<int, std::vector<int>> bitmask_groups;
    uint64_t local_bm_presence_f = 0;
    for (int ii : ActiveParticleList) {
        if(AGSForce_isactive(ii)) {
            int bm = ags_gravity_kernel_shared_BITFLAG(P[ii].Type);
            if(bm > 0 && bm < 64) { bitmask_groups[bm].push_back(ii); local_bm_presence_f |= (1ULL << bm); }
        }
    }
    /* Symmetrise across ranks so all ranks call ghost_writeback_agsforce the same number of times. */
    uint64_t global_bm_presence_f = local_bm_presence_f;
    if(NTask > 1) MPI_Allreduce(&local_bm_presence_f, &global_bm_presence_f, 1, MPI_UINT64_T, MPI_BOR, MPI_COMM_WORLD);

    /* Zero per-iteration i-side accumulators for active AGSForce particles.
       These correspond to the OUTPUTFUNCTION_NAME fields that use mode==0
       ASSIGN (not ASSIGN_ADD). */
    for(auto& kv : bitmask_groups) {
        for(int ii : kv.second) {
#ifdef DM_FUZZY
            P[ii].AGS_Dt_Numerical_QuantumPotential = 0;
#if (DM_FUZZY > 0)
            P[ii].AGS_Dt_Psi_Re = P[ii].AGS_Dt_Psi_Im = P[ii].AGS_Dt_Psi_Mass = 0;
#endif
#endif
#if defined(CBE_INTEGRATOR)
            P[ii].AGS_vsig = 0;
            for(int k1 = 0; k1 < CBE_INTEGRATOR_NBASIS; k1++) {
                for(int k2 = 0; k2 < CBE_INTEGRATOR_NMOMENTS; k2++) {
                    P[ii].CBE_basis_moments_dt[k1][k2] = 0;
                }
            }
#endif
        }
    }

    /* Iterate global bitmask union so all ranks call ghost_writeback_agsforce the same number of times. */
    for(int bm = 1; bm < 64; bm++) {
        if(!(global_bm_presence_f & (1ULL << bm))) continue;
        std::vector<int>& ilist = bitmask_groups[bm];  /* empty if rank has none */
        int nl_num_active = (int)ilist.size();
        int *nl_active = (int *) mymalloc("agsforce_nl_active", (nl_num_active > 0 ? nl_num_active : 1) * sizeof(int));
        double *nl_radii = (double *) mymalloc("agsforce_nl_radii", (nl_num_active > 0 ? nl_num_active : 1) * sizeof(double));
        for(int a = 0; a < nl_num_active; a++) { nl_active[a] = ilist[a]; nl_radii[a] = P[ilist[a]].AGS_KernelRadius; }
        struct ags_force_gpu_out *nl_outs = (struct ags_force_gpu_out *) mymalloc(
            "agsforce_nl_outs", (nl_num_active > 0 ? nl_num_active : 1) * sizeof(struct ags_force_gpu_out));

        /* Snapshot ghost Vel/dp/NInteractions + zero wakeup so post-kernel
           values become pure deltas to reverse-communicate. */
        ghost_writeback_zero_agsforce();
        ags_force_evaluate_gpu(P, NumPart, nl_active, nl_num_active, nl_radii, bm, nl_outs);
        ghost_writeback_agsforce();

        /* Scatter i-side accumulators into P[ii] (match CPU OUTPUT semantics). */
        for(int a = 0; a < nl_num_active; a++) {
            int ii = nl_active[a];
#if defined(DM_SIDM)
            for(int k = 0; k < 3; k++) {
                P[ii].Vel[k] += nl_outs[a].sidm_kick[k];
                P[ii].dp[k]  += nl_outs[a].sidm_kick[k] * P[ii].Mass;
            }
            if(nl_outs[a].dtime_sidm < P[ii].dtime_sidm) P[ii].dtime_sidm = nl_outs[a].dtime_sidm;
            P[ii].NInteractions += nl_outs[a].si_count;
#endif
#ifdef DM_FUZZY
            for(int k = 0; k < 3; k++) P[ii].GravAccel[k] += nl_outs[a].acc[k];
            P[ii].AGS_Dt_Numerical_QuantumPotential += nl_outs[a].AGS_Dt_Numerical_QuantumPotential;
#if (DM_FUZZY > 0)
            P[ii].AGS_Dt_Psi_Re   += nl_outs[a].AGS_Dt_Psi_Re;
            P[ii].AGS_Dt_Psi_Im   += nl_outs[a].AGS_Dt_Psi_Im;
            P[ii].AGS_Dt_Psi_Mass += nl_outs[a].AGS_Dt_Psi_Mass;
#endif
#endif
#if defined(CBE_INTEGRATOR)
            if(nl_outs[a].AGS_vsig > P[ii].AGS_vsig) P[ii].AGS_vsig = nl_outs[a].AGS_vsig;
            for(int k1 = 0; k1 < CBE_INTEGRATOR_NBASIS; k1++) {
                for(int k2 = 0; k2 < CBE_INTEGRATOR_NMOMENTS; k2++) {
                    P[ii].CBE_basis_moments_dt[k1][k2] += nl_outs[a].CBE_basis_moments_dt[k1][k2];
                }
            }
#endif
        }
        myfree(nl_outs); myfree(nl_radii); myfree(nl_active);
    }

    if(NTask > 1) { ghost_exchange_cleanup(); }
    timecomp += timediff(t0, my_second());
    /* do final operations on results: these are operations that can be done after the complete set of iterations */
#ifdef CBE_INTEGRATOR
        for (int i : ActiveParticleList) {do_postgravity_cbe_calcs(i);} // do any final post-tree-walk calcs from the CBE integrator here //
#endif
    /* collect timing information */
    double t1; t1 = WallclockTime = my_second(); timeall = timediff(t00_truestart, t1);
    CPU_Step[CPU_AGSDENSCOMPUTE] += timecomp; CPU_Step[CPU_AGSDENSWAIT] += timewait;
    CPU_Step[CPU_AGSDENSCOMM] += timecomm; CPU_Step[CPU_AGSDENSMISC] += timeall - (timecomp + timewait + timecomm);
}


#endif // AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
