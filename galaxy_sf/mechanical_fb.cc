#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "mechanical_fb_types.h"  /* provides struct MechFBGasDelta */

/* Routines for mechanical feedback/enrichment models: stellar winds, supernovae, etc
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifdef GALSF_FB_MECHANICAL

int N_Gas_Couplings_ThisTask; // define variable to use below to record if and how many times a coupling to a cell occurs in a timestep per MPI task

int addFB_evaluate_active_check(int i, int fb_loop_iteration);
int addFB_evaluate_active_check(int i, int fb_loop_iteration)
{
    if(P[i].Type <= 1) {return 0;} // note quantities used here must -not- change in the loop [hence not using mass here], b/c can change offsets for return from different processors, giving a negative mass and undefined behaviors
    if(P[i].KernelRadius <= 0) {return 0;}
    if(P[i].NumNgb <= 0) {return 0;}
#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
    if(P[i].Type == 5 && !P[i].do_gas_search_this_timestep) {return 0;}
#endif
    if(P[i].SNe_ThisTimeStep>0) {if(fb_loop_iteration<0 || fb_loop_iteration==0) {return 1;}}
#ifdef GALSF_FB_FIRE_STELLAREVOLUTION
    if(P[i].MassReturn_ThisTimeStep>0) {if(fb_loop_iteration<0 || fb_loop_iteration==1) {return 1;}}
#ifdef GALSF_FB_FIRE_RPROCESS
    if(P[i].RProcessEvent_ThisTimeStep>0) {if(fb_loop_iteration<0 || fb_loop_iteration==2) {return 1;}}
#endif
#ifdef GALSF_FB_FIRE_AGE_TRACERS
    if(P[i].AgeDeposition_ThisTimeStep>0) {if(fb_loop_iteration<0 || fb_loop_iteration==3) {return 1;}}
#endif
#endif
#if defined(SINGLE_STAR_FB_WINDS) && defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
    if(P[i].wind_mode != 2 || P[i].ProtoStellarStage != 5) {return 0;}
#endif
    return 0;
}


void determine_where_SNe_occur(void)
{
    if(All.Time<=0) return;
    int i; double dt,star_age,npossible,nhosttotal,ntotal,ptotal,dtmean,rmean;
    npossible=nhosttotal=ntotal=ptotal=dtmean=rmean=0;
    double mpi_npossible,mpi_nhosttotal,mpi_ntotal,mpi_ptotal,mpi_dtmean,mpi_rmean;
    mpi_npossible=mpi_nhosttotal=mpi_ntotal=mpi_ptotal=mpi_dtmean=mpi_rmean=0;
    // loop over particles //
    for (int i : ActiveParticleList)
    {
        P[i].SNe_ThisTimeStep=0;
#ifdef GALSF_FB_FIRE_STELLAREVOLUTION
        P[i].MassReturn_ThisTimeStep=0;
#ifdef GALSF_FB_FIRE_RPROCESS
        P[i].RProcessEvent_ThisTimeStep=0;
#endif
#ifdef GALSF_FB_FIRE_AGE_TRACERS
        P[i].AgeDeposition_ThisTimeStep=0;
#endif
#endif
    

#if defined(SINGLE_STAR_SINK_DYNAMICS)
        if(P[i].Type == 0) {continue;} // any non-gas type is eligible to be a 'star' here
#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
        if(P[i].Type == 5)
        {   // single sink type, for type 5
            if((P[i].ProtoStellarStage != 5) && (P[i].ProtoStellarStage != 6)) {continue;} // only MS or SN-tagged particles eligible to have winds or SN in this routine
        } else { // stellar population type allowed here
            if(All.ComovingIntegrationOn) {if(P[i].Type != 4) {continue;}} // in cosmological simulations, 'stars' have particle type=4
            if(All.ComovingIntegrationOn==0) {if((P[i].Type<2)||(P[i].Type>4)) {continue;}} // in non-cosmological sims, types 2,3,4 are valid 'stars'
        }
#endif
#else
        if(All.ComovingIntegrationOn) {if(P[i].Type != 4) {continue;}} // in cosmological simulations, 'stars' have particle type=4
        if(All.ComovingIntegrationOn==0) {if((P[i].Type<2)||(P[i].Type>4)) {continue;}} // in non-cosmological sims, types 2,3,4 are valid 'stars'
#endif
        if(P[i].Mass<=0) {continue;}
        dt = get_particle_feedback_timestep_in_physical(i);
#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
        if(P[i].Type == 5) {dt = P[i].dt_since_last_gas_search;}
#endif
        if(dt<=0) {continue;} // no time, no events
        star_age = evaluate_stellar_age_Gyr(i);
        if(star_age<=0) {continue;} // unphysical age, no events
        // now use a calculation of mechanical event rates to determine where/when the events actually occur //
        npossible++;
        double RSNe = mechanical_fb_calculate_eventrates(i,dt);
        rmean += RSNe; ptotal += RSNe * (P[i].Mass*UNIT_MASS_IN_SOLAR) * (dt*UNIT_TIME_IN_MYR);
#ifdef GALSF_SFR_IMF_SAMPLING
        if(P[i].Type<5) {if(P[i].IMF_NumMassiveStars>0) {P[i].IMF_NumMassiveStars=DMAX(0,P[i].IMF_NumMassiveStars-P[i].SNe_ThisTimeStep);}} // lose an O-star for every SNe //
#endif
        if(P[i].SNe_ThisTimeStep>0) {ntotal+=P[i].SNe_ThisTimeStep; nhosttotal++;}
        dtmean += dt;
    } // for (int i : ActiveParticleList) //

    MPI_Reduce(&dtmean, &mpi_dtmean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&rmean, &mpi_rmean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ptotal, &mpi_ptotal, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nhosttotal, &mpi_nhosttotal, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ntotal, &mpi_ntotal, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&npossible, &mpi_npossible, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(ThisTask == 0)
    {
        if(mpi_ntotal > 0 && mpi_nhosttotal > 0 && mpi_dtmean > 0)
        if(mpi_npossible>0)
        {
            mpi_dtmean /= mpi_npossible; mpi_rmean /= mpi_npossible;
            fprintf(FdSNeFBLogFile, "%.16g %g %g %g %g %g %g \n", All.Time,mpi_npossible,mpi_nhosttotal,mpi_ntotal,mpi_ptotal,mpi_dtmean,mpi_rmean); fflush(FdSNeFBLogFile);
        }
        if(All.HighestActiveTimeBin == All.HighestOccupiedTimeBin) {fflush(FdSNeFBLogFile);}
    } // if(ThisTask == 0) //

} // void determine_where_SNe_occur() //



/* this is a temporary structure for quantities used ONLY in the loops below,
 for example for ensuring conservation if there are many overlapping events.
 Struct type MechFBGasDelta is defined in mechanical_fb_functions.h so the
 GPU port (B8) can share the exact layout. */
static struct MechFBGasDelta *LocalGasMechFBInfoTemp;






/* subroutine to check for total kinetic energy change and thermal energy change after integrating the effects of all SNe over all cells, and coupling this to particles,  */
void verify_and_assign_local_mechfb_integrals(void)
{
    if(N_Gas_Couplings_ThisTask <= 0) {return;} /* no cells had feedback deposited */
    int j,k,ndone=0; for(j=0;j<N_gas;j++)
    {
        if(LocalGasMechFBInfoTemp[j].N_injected <= 0) {continue;} /* all mechanisms deposit non-zero mass, so skip if this is not >0*/
        if(P[j].Type==0 && P[j].Mass>0)
        {
            double m0=P[j].Mass, dm=LocalGasMechFBInfoTemp[j].m_injected; P[j].Mass += dm; /* update mass */ if(P[j].Type==0) {CellP[j].Mass = P[j].Mass;}
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
            m0=CellP[j].MassTrue; CellP[j].MassTrue += dm; /* update conserved mass */
#endif
            double mf=m0+dm; /* save for below */
            for(k=0;k<NUM_METAL_SPECIES;k++) {P[j].Metallicity[k] = (m0/mf)*P[j].Metallicity[k] + (1./mf)*LocalGasMechFBInfoTemp[j].Z_injected[k];} /* update metallicity */
#if defined(GALSF_ISMDUSTCHEM_MODEL)
            update_ISMDustChem_after_mechanical_injection(j, LocalGasMechFBInfoTemp[j].Mass_Where_Dust_Shocked, m0, mf, LocalGasMechFBInfoTemp[j].Z_injected); /* update dust chemistry quantities */
#endif
            CellP[j].Density *= mf/m0; /* update density [semi-drift] */
            double dTE=LocalGasMechFBInfoTemp[j].TE_injected;
            if(dTE > 0)
            {
                double TE_0=m0*CellP[j].InternalEnergy; dTE=DMAX(-TE_0,dTE); /* ensure against non-negative values */
                double dU = (-dm/mf)*CellP[j].InternalEnergy + (1./mf)*dTE; /* using new mass get updated internal energy */
                double dt = get_particle_timestep_in_physical(j), implied_heating_cgs=(dU*UNIT_SPECEGY_IN_CGS*PROTONMASS_CGS)/(dt*UNIT_TIME_IN_CGS), typical_cooling_cgs=1.e-23*(CellP[j].Density*All.cf_a3inv*UNIT_DENSITY_IN_NHCGS);
                if((implied_heating_cgs < 0.3*typical_cooling_cgs) && (dt > MIN_REAL_NUMBER) && ((dU < 4.*CellP[j].InternalEnergy) || ((dU < 1000.*CellP[j].InternalEnergy) && ((dU+CellP[j].InternalEnergy)*U_TO_TEMP_UNITS*2./3.*1.28 < 5.e5)))) {CellP[j].DtInternalEnergy += dU/dt;} else {CellP[j].InternalEnergy += dU; CellP[j].InternalEnergyPred += dU;}
                //CellP[j].InternalEnergy += dU; CellP[j].InternalEnergyPred += dU; /* update internal energy; simpler (old) way to do it - less accurate phase diagrams at high density, however */
            }
            double dKE=LocalGasMechFBInfoTemp[j].KE_injected, dp[3];
            if(dKE != 0 || LocalGasMechFBInfoTemp[j].p_injected[0] != 0 || LocalGasMechFBInfoTemp[j].p_injected[1] != 0 || LocalGasMechFBInfoTemp[j].p_injected[2] != 0 )
            {
                double KE_0=0, p0[3], p2=0, dp2=0, pdp=0; /* define variables for momentum update (the non-trivial update term) */
                for(k=0;k<3;k++) {p0[k]=m0*P[j].Vel[k]/All.cf_atime; dp[k]=LocalGasMechFBInfoTemp[j].p_injected[k]; p2+=p0[k]*p0[k]; dp2+=dp[k]*dp[k]; pdp+=p0[k]*dp[k];} /* define variables */
                KE_0=p2/(2.*m0); dKE=DMAX(-KE_0,dKE); /* limit to retain positive definite kinetic energy */
                double a0=dp2, b0=2.*pdp, c0=p2*dm/m0+2.*mf*dKE, sfac=b0*b0+4.*a0*c0, f0=0; /* assume coupled delta_p = f0*delta_p, and solve for the multiplier f0 that gives the desired total kinetic energy change */
                if(sfac<=0 || !isfinite(sfac)) {if((fabs(a0)<1.e-40) || !isfinite(b0/a0)) {f0=0;} else {f0=-b0/(2.*a0);}} // catches for floating-point error
                    else {if((b0>0) && fabs(4.*a0*c0)<1.e-3*b0*b0) {f0=c0/b0;} // catches for floating-point error
                    else {if(fabs(a0)<1.e-40 || !isfinite(a0)) {f0=0;} else {f0=(-b0+sqrt(b0*b0+4.*a0*c0))/(2.*a0);}}} // catches for floating-point error, if pass all of them, use exact solution for desired KE
                if(f0<0 || !isfinite(f0)) {f0=0;} else {if(f0>1.) {f0=1.;}} /* limit to physical values (should never be an issue but again because of float error it could be) */
                for(k=0;k<3;k++) {
                    double dv = (-dm/mf)*P[j].Vel[k] + f0*(1./mf)*dp[k]*All.cf_atime; /* calculate total momentum change and mass change and therefore final velocity (in code units) */
                    P[j].Vel[k] += dv; CellP[j].VelPred[k] += dv; P[j].dp[k] += f0*dp[k]; /* update velocities */
                }
            }
#if defined(COSMIC_RAY_FLUID)
            /* Apply CR deltas accumulated by the B8 Phase 2 GPU kernel. On CPU-only
             * builds, the existing inject_cosmic_rays in the pair loop writes
             * directly to CellP, so these delta fields remain zero and this block
             * is a no-op. */
            {
                double cr_tot = 0;
                for(int kb = 0; kb < N_CR_PARTICLE_BINS; kb++) cr_tot += LocalGasMechFBInfoTemp[j].CR_energy_injected[kb];
                if(cr_tot > 0) {
                    /* compute unit injection direction from weighted sum */
                    double dmag_sq = 0;
                    for(int c = 0; c < 3; c++) dmag_sq += LocalGasMechFBInfoTemp[j].CR_dir_weighted[c] * LocalGasMechFBInfoTemp[j].CR_dir_weighted[c];
                    double dmag = sqrt(dmag_sq);
                    double dir_hat[3];
                    if(dmag > 0) { for(int c = 0; c < 3; c++) dir_hat[c] = LocalGasMechFBInfoTemp[j].CR_dir_weighted[c] / dmag; }
                    else         { dir_hat[0] = 0; dir_hat[1] = 0; dir_hat[2] = 1; }
#ifdef MAGNETIC
                    /* project onto B, keeping sign from requested direction */
                    {
                        double B_dot_d = 0;
                        for(int c = 0; c < 3; c++) B_dot_d += dir_hat[c] * CellP[j].BPred[c];
                        for(int c = 0; c < 3; c++) dir_hat[c] = B_dot_d * CellP[j].BPred[c];
                        double bmag_sq = 0;
                        for(int c = 0; c < 3; c++) bmag_sq += dir_hat[c] * dir_hat[c];
                        double bmag = sqrt(bmag_sq);
                        if(bmag > 0) { for(int c = 0; c < 3; c++) dir_hat[c] /= bmag; }
                        else         { dir_hat[0] = 0; dir_hat[1] = 0; dir_hat[2] = 1; }
                    }
#endif
                    /* apply per-bin energy / number / flux */
                    for(int kb = 0; kb < N_CR_PARTICLE_BINS; kb++) {
                        double dEcr = LocalGasMechFBInfoTemp[j].CR_energy_injected[kb];
                        if(dEcr <= 0) continue;
                        CellP[j].CosmicRayEnergy[kb]     += dEcr;
                        CellP[j].CosmicRayEnergyPred[kb] += dEcr;
#if defined(CRFLUID_EVOLVE_SPECTRUM)
                        CellP[j].CosmicRay_Number_in_Bin[kb] += LocalGasMechFBInfoTemp[j].CR_number_injected[kb];
#endif
                        double flux_mag = dEcr * CRFLUID_REDUCED_C_CODE(kb);
                        for(int c = 0; c < 3; c++) {
                            double dflux = flux_mag * dir_hat[c];
                            CellP[j].CosmicRayFlux[kb][c]     += dflux;
                            CellP[j].CosmicRayFluxPred[kb][c] += dflux;
                        }
                    }
                }
            }
#endif /* COSMIC_RAY_FLUID */
            ndone++; /* note another cell accounted for */
            if(ndone >= N_Gas_Couplings_ThisTask) {break;} /* we have done all cells (note is possible if the same cell is hit many times, N_Gas_Couplings_ThisTask can be much larger than ndone after the full loop. but if its smaller, then we -must- be done with this loop */
        }
    }
    return;
}



/* parent routine which calls all of the mechanical fb loops and verifies coupling, if needed */
void mechanical_fb_calc_toplevel(void)
{
    PRINT_STATUS("Start mechanical feedback computation...");

    /* B8 GPU port: dispatch all 6 modes at once on the GPU, sharing a single
       neighbor list across modes. See galaxy_sf/mechanical_fb_gpu.cc. */
    {
#include "../galaxy_sf/galsf_gpu_decls.h"
        /* Build SUPERSET active list: any star active in ANY mode is included.
           The per-mode mask inside the kernel skips stars not active in that mode. */
        int num_active = 0;
        for(int ii : ActiveParticleList) {
            int any = 0;
            for(int mm = -2; mm <= 3 && !any; mm++) {
                if(addFB_evaluate_active_check(ii, mm)) any = 1;
            }
            if(any) num_active++;
        }
        /* Multi-rank correctness: ghost_prep is collective in mech_fb_gpu so all
         * ranks must agree on whether to call it. MPI_Allreduce num_active first;
         * if no rank has any source, every rank skips the entire mech_fb path
         * (no allocation, no kernel, no ghost_prep). Saves ~1s/step on tiny-N
         * timebins where no SNe/winds fire. Architectural rule (Phil): NEVER
         * loop over N_gas in a tiny-N step when no work will be done; use
         * ActiveParticleList instead. The N_gas memset of LocalGasMechFBInfoTemp
         * (was here pre-fix) was paying ~0.5s/step even on no-work timebins. */
        int global_num_active = num_active;
        MPI_Allreduce(&num_active, &global_num_active, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if(global_num_active == 0) {
            return;
        }

        /* allocate temporary stucture which will hold the total change, to compare when done to check for non-linear effects if too many cells act at once */
        LocalGasMechFBInfoTemp = (struct MechFBGasDelta *) mymalloc("LocalGasMechFBInfoTemp",N_gas * sizeof(struct MechFBGasDelta));
        N_Gas_Couplings_ThisTask = 0; /* initialize this to zero [default to assume no coupled feedback] */
        /* Zero only gas particles. memset of N_gas is unavoidable here because
         * this struct is indexed by particle index across all gas (kernel writes
         * to neighbor j-indices, not just i-active sources). But it only fires
         * when global_num_active > 0, so no waste on tiny-N no-work steps. */
        for(int j = 0; j < N_gas; j++) {if(P[j].Type==0) {memset(&LocalGasMechFBInfoTemp[j], 0, sizeof(struct MechFBGasDelta));}}
        int *nl_active = (int *) mymalloc("mechfb_nl_active",
            (num_active > 0 ? num_active : 1) * sizeof(int));
        double *nl_radii = (double *) mymalloc("mechfb_nl_radii",
            (num_active > 0 ? num_active : 1) * sizeof(double));
        {int aa = 0; for(int ii : ActiveParticleList) {
            int any = 0;
            for(int mm = -2; mm <= 3 && !any; mm++) {
                if(addFB_evaluate_active_check(ii, mm)) any = 1;
            }
            if(any) { nl_active[aa] = ii; nl_radii[aa] = (double)P[ii].KernelRadius; aa++; }
        }}
        int n_coup_gpu = 0;
        /* Always call evaluate_gpu — its wrapper fast-path handles num_active==0
         * including the multi-rank ghost_writeback_mechfb collective, which other
         * ranks may rely on even when this rank has no local sources. */
        mechanical_fb_evaluate_gpu(P, CellP, NumPart,
                                    LocalGasMechFBInfoTemp, N_gas,
                                    nl_active, num_active, nl_radii, &n_coup_gpu);
        N_Gas_Couplings_ThisTask = n_coup_gpu;
        myfree(nl_radii); myfree(nl_active);
    }



    verify_and_assign_local_mechfb_integrals();
    myfree(LocalGasMechFBInfoTemp); /* free the structure */
}




#endif /* GALSF_FB_MECHANICAL */
