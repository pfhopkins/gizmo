#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"

/*! \file accel.c
 *  \brief driver routines to carry out force computation
 */


/*! This routine computes the accelerations for all active particles.  First, the gravitational forces are
 * computed. This also reconstructs the tree, if needed, otherwise the drift/kick operations have updated the
 * tree to make it fullu usable at the current time.
 *
 * If gas particles are presented, the `interior' of the local domain is determined. This region is guaranteed
 * to contain only particles local to the processor. This information will be used to reduce communication in
 * the hydro part.  The density for active gas/fluid cells is computed next. If the number of neighbours should
 * be outside the allowed bounds, it will be readjusted by the function ensure_neighbours(), and for those
 * particle, the densities are recomputed accordingly. Finally, the hydrodynamical forces are added.
 */
/*!
 * This file was originally part of the GADGET3 code developed by
 * Volker Springel. The code has been rewritten and extensively modified (re-arranged, consolidated, and a number of additional
 * sub-loops and other structures for e.g. feedback, gradients, neighbor operations on non-gas, etc,
 * added) by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

void compute_grav_accelerations(void)
{
  CPU_Step[CPU_MISC] += measure_time();
  PRINT_STATUS("Start gravity force computation...");

#ifdef PMGRID
  if(All.PM_Ti_endstep == All.Ti_Current)
    {
      long_range_force();
      CPU_Step[CPU_MESH] += measure_time();
    }
#endif

  gravity_tree();		/* computes gravity accel. */

  /* For the first timestep, we redo it to allow usage of relative opening criterion for consistent accuracy.
     Raise the rebuild flag so the second pass runs on a freshly built tree rather than reusing the first
     pass's: the BH->relative switch changes which node data the opening test consumes (cf. gizmo-cpp
     5b490b93). One extra build at t=0 only. */
  if(All.TypeOfOpeningCriterion == 1 && All.Ti_Current == 0) {TreeReconstructFlag = 1; gravity_tree();}
#if defined(KETJU_REGULARIZATION) && defined(KETJU_TYPE4_STARS)
  ketju_snapshot_tree_softening(); /* softening/zeta this gravity call used, for the KETJU pair subtraction */
#endif

  PRINT_STATUS(" ..gravity force computation done");
}



void compute_hydro_densities_and_forces(void)
{
  if(All.TotN_gas > 0)
    {
        PRINT_STATUS("Start hydrodynamics computation...");
        density();		/* computes density, and pressure */
#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
        ags_density();
#endif
        force_update_hmax();	/* update kernel lengths in tree */
        /*! This function updates the hmax-values in tree nodes that hold gas. These values are needed to find all neighbors in the hydro-force computation.  Since the KernelRadius-values are potentially changed in the gas-denity computation, force_update_hmax() should be carried out before the hydrodynamical forces are computed, i.e. after density(). */

        PRINT_STATUS(" ..density & tree-update computation done...");

#ifdef HYDRO_VOLUME_CORRECTIONS
        cellcorrections_calc(); /* must be called after density, and after the update of hmax in the tree [because it depends on bi-directional search], but before gradients where quantities dependent on volumetric elements such as density are needed */
#endif
#ifdef TURB_DIFF_DYNAMIC
        dynamic_diff_vel_calc(); /* This must be called between density and gradient calculations */
#endif
#if defined(RADTRANSFER) && defined(GRAIN_RDI_TESTPROBLEM_LIVE_RADIATION_INJECTION)
        rt_source_injection(); /* doing source injection here (just before interpolation and hydro gradients) is slightly more accurate for this setup, but not possible in total generality owing to dependence of some injection modules on quantities calculated below */
#endif
#if defined(RT_OPACITY_FROM_EXPLICIT_GRAINS)
        interpolate_fluxes_opacities_gasgrains(); /* this must be called here for the computation of opacities and radiative quantity gradients below to be correct */
#endif
#if defined(GALSF) && !defined(FB_G17_BOUNDARY_INJECTION) /* PFH set of feedback routines; here because for e.g. strong SNe, obtain better stability if they are coupled discretely just -before- the hydro force is computed */
        MCBAL_LOG("pre_FB");
        compute_stellar_feedback();
        MCBAL_LOG("post_FB");
#endif
        /* FB_G17_BOUNDARY_INJECTION: compute_stellar_feedback() moved to the sync boundary
         * (run.cc calculate_non_standard_physics, gizmo2017 run.c:220 slot).  At that position
         * the weighting inputs are FRESH -- density()/ags_density() above ran THIS step, so
         * DensityAroundParticle/KernelRadius/NumNgb are current and the 2026-05-07
         * stale-density concern (which motivated the mid-step placement) does not apply. */

        MCBAL_LOG("pre_grad");
        hydro_gradient_calc(); /* calculates the gradients of hydrodynamical quantities  */
        MCBAL_LOG("post_grad");
#ifdef MHD_MODIFIED_GRADIENT
        {   /* determine whether the active gas fraction is large enough to justify the global MG solve */
            long long ngas_active_local = 0;
            for(int tbin = 0; tbin < TIMEBINS; tbin++) {if(TimeBinActive[tbin]) {ngas_active_local += TimeBinCountGas[tbin];}}
            long long ngas_active_global; MPI_Allreduce(&ngas_active_local, &ngas_active_global, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            double active_gas_fraction = (All.TotN_gas > 0) ? ((double)ngas_active_global / (double)All.TotN_gas) : 1.0;
            if(active_gas_fraction >= All.ActiveFractionForMGSweep) {
                All.Flag_SkipMGSolve = 0;
                mg_gradient_correction_calc(); /* MG method: global sparse-matrix solve for exact div(B)=0 correction coefficients (Tu et al. 2026) */
            } else {
                All.Flag_SkipMGSolve = 1;
                if(ThisTask == 0) {PRINT_STATUS("Skipping MG global solve (active gas fraction %g < %g), using CG fallback for active cells", active_gas_fraction, All.ActiveFractionForMGSweep);}
            }
        }
#endif
#if defined(COOLING) && defined(GALSF_FB_FIRE_RT_LONGRANGE)
        selfshield_local_incident_uv_flux(); /* needs to be called after gravity tree (where raw flux is calculated) and the local gradient calculation (GradRho) to properly self-shield the particles that had this calculated */
#endif
        PRINT_STATUS(" ..gradient computation done.");

#if (SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_SPECIALBOUNDARIES >= 4)
        special_rt_feedback_injection(); /* do before proper hydro loop */
#endif

#ifdef TURB_DIFF_DYNAMIC
        dynamic_diff_calc(); /* This MUST be called immediately following gradient calculations */
#endif
        hydro_force();		/* adds hydrodynamical accelerations and computes du/dt  */
        MCBAL_LOG("post_hydro_force");
        compute_additional_forces_for_all_particles(); /* other accelerations that need to be computed are done here */
        PRINT_STATUS(" ..hydro force computation done.");

    } else {
#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
        ags_density(); // if there are no gas particles but ags-all is active, still need to enter this loop //
        force_update_hmax();    /* update kernel lengths in tree */
#endif
        compute_additional_forces_for_all_particles();
    }
}



#ifdef BOX_SHEARING
/* Shearing-sheet Coriolis force, as a Boris step over each particle's timestep (as for the grain Lorentz force): half a kick from
   every other acceleration, a rotation at 2 Omega that keeps the speed, half a kick. The rotation's velocity change over dt enters
   GravAccel as an effective acceleration, which the kicks apply as usual. Evaluated explicitly, a velocity-dependent force makes the
   epicycles of the kick-drift-kick integration grow; every other acceleration has to be final for the step, so this runs last.
   GravAccel holds it for this step only (GRAVACCEL_RETAINED_ON_SKIPPED_WALK keeps it out of a skipped walk's kept gravity).
   Not comoving: velocities and accelerations are physical. */
static void shearing_sheet_coriolis_boris(void)
{
    const int ph = BOX_SHEARING_PHI_COORDINATE;
    const double omega = BOX_SHEARING_OMEGA_BOX_CENTER;
    const int n_active = (int) ActiveParticleList.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int ii = 0; ii < n_active; ii++)
    {
        const int i = ActiveParticleList[ii];
        const double dt = get_particle_timestep_in_physical(i);
        double coriolis_x, coriolis_ph;
        if(dt <= 0) /* no timestep yet (the first force evaluation): the dt -> 0 limit, the explicit force on the predicted velocity */
        {
            const double v_x = (P[i].Type == 0) ? CellP[i].VelPred[0] : P[i].Vel[0], v_ph = (P[i].Type == 0) ? CellP[i].VelPred[ph] : P[i].Vel[ph];
            coriolis_x = 2. * omega * v_ph;
            coriolis_ph = -2. * omega * v_x;
        } else {
            double acc_x = P[i].GravAccel[0], acc_ph = P[i].GravAccel[ph];
            if(P[i].Type == 0)
            {
                acc_x += CellP[i].HydroAccel[0]; acc_ph += CellP[i].HydroAccel[ph];
#ifdef TURB_DRIVING
                acc_x += CellP[i].TurbAccel[0]; acc_ph += CellP[i].TurbAccel[ph];
#endif
#ifdef RT_RAD_PRESSURE_OUTPUT
                acc_x += CellP[i].Rad_Accel[0]; acc_ph += CellP[i].Rad_Accel[ph];
#endif
            }
            const double vm_x = P[i].Vel[0] + 0.5 * dt * acc_x, vm_ph = P[i].Vel[ph] + 0.5 * dt * acc_ph;
            const double t = omega * dt, s = 2. * t / (1. + t * t);   /* rotation by 2 Omega dt: its half-angle tangent is Omega dt */
            const double vt_x = vm_x + t * vm_ph, vt_ph = vm_ph - t * vm_x;
            coriolis_x = s * vt_ph / dt;
            coriolis_ph = -s * vt_x / dt;
        }
        P[i].GravAccel[0] += coriolis_x;
        P[i].GravAccel[ph] += coriolis_ph;
    }
}
#endif


void compute_additional_forces_for_all_particles(void)
{
#if defined(DM_FUZZY) || defined(CBE_INTEGRATOR_WITHGRADIENTS)
    DMGrad_gradient_calc();
#endif
#if defined(DM_FUZZY) || defined(CBE_INTEGRATOR) || defined(DM_SIDM)
    AGSForce_calc();
#endif
#ifdef GRAIN_FLUID
    apply_grain_dragforce(); /* if we are solving a coupled set of grains via aerodynamic drag, this is where their acceleration should be calculated */
#endif
#ifdef BOX_SHEARING
    shearing_sheet_coriolis_boris(); /* last: it needs every other acceleration */
#endif
}




#ifdef GALSF
void compute_stellar_feedback(void)
{
    CPU_Step[CPU_MISC] += measure_time();
#ifdef GALSF_LIMIT_FBTIMESTEPS_FROM_BELOW
    if(All.Dt_Since_LastFBCalc_Gyr > All.Dt_Min_Between_FBCalc_Gyr) {All.Dt_Since_LastFBCalc_Gyr = 0;}
    All.Dt_Since_LastFBCalc_Gyr += All.TimeStep / All.cf_hubble_a * UNIT_TIME_IN_GYR; // augment by timestep
    if(All.Dt_Since_LastFBCalc_Gyr < All.Dt_Min_Between_FBCalc_Gyr) {
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
        { static long nthr=0; if((nthr++%200)==0 && ThisTask==0) {printf("FBTL_THROTTLE skipped (n=%ld) t=%g\n", nthr, All.Time); fflush(stdout);} }
#endif
        return;}
#endif

#ifdef GALSF_FB_MECHANICAL /* check the mechanical sources of feedback */
    {   /* Rank-uniform early-out, as rt_source_injection does. With no active source on any rank the sub-loops
           are exact no-ops (nothing is evaluated, so nothing is exported or imported) but each still pays an
           Alltoall, a Barrier and its buffer setup. The gate predicate is the loops' own CONDITIONFUNCTION with
           fb_loop_iteration = -1, which is true iff any channel's condition is, so the set of steps on which
           anything is deposited is unchanged. */
        int any_fb_local = 0, any_fb_global = 0;
        for(int i : ActiveParticleList) {if(addFB_evaluate_active_check(i, -1)) {any_fb_local = 1; break;}}
        MPI_Allreduce(&any_fb_local, &any_fb_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#ifndef SNE_BROADCAST_TIMESTEP /* with it, deposited after the closing kick instead (calculate_non_standard_physics) */
        if(any_fb_global) {mechanical_fb_calc_toplevel();}  /* call the parent loop for the different mechanical fb sub-loops */
#endif
    }
    CPU_Step[CPU_SNIIHEATING] += measure_time(); /* collect timings and reset clock for next timing */
#endif

#ifdef GALSF_FB_THERMAL
    thermal_fb_calc(); /* thermal feedback */
    MPI_Barrier(MPI_COMM_WORLD); CPU_Step[CPU_SNIIHEATING] += measure_time(); /* collect timings and reset clock for next timing */
#endif

#ifdef GALSF_RESOLVEDISM_FB
    resolvedism_inject_fb_energy(); /* resolved-ISM SN/Ia + AGB/wind + radpressure injection.  Uses FIRE-style measured-coupling: dying star is reduced by the actually-deposited mass (M_coupled accumulated in each FB walk's out2particle) → bit-exact mass conservation per event regardless of Σ wk_j ≠ 1. */
    MPI_Barrier(MPI_COMM_WORLD); CPU_Step[CPU_SNIIHEATING] += measure_time();
#endif

#if defined(GALSF_FB_FIRE_RT_HIIHEATING)
    HII_heating_singledomain(); /* local photo-ionization heating */
    MPI_Barrier(MPI_COMM_WORLD); CPU_Step[CPU_HIIHEATING] += measure_time(); /* collect timings and reset clock for next timing */
#endif
    
#ifdef GALSF_FB_FIRE_RT_LOCALRP
    radiation_pressure_winds_consolidated(); /* local radiation pressure */
    MPI_Barrier(MPI_COMM_WORLD); CPU_Step[CPU_LOCALWIND] += measure_time(); /* collect timings and reset clock for next timing */
#endif
    
    CPU_Step[CPU_MISC] += measure_time();
}
#endif // GALSF //
