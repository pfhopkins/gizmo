#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../hydro/hydro_corridor.h" /* gizmo_hydro_corridor_mg_step */

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
  {
    if(All.PM_Ti_endstep == All.Ti_Current)
      {
        long_range_force();
        CPU_Step[CPU_MESH] += measure_time();
      }
  }
#endif

  {
#if defined(RT_USE_GRAVTREE) && defined(GALSF_SINGLE_STARS_FROM_IMF) && defined(GALSF_SINGLE_STARS_EVOLUTION_TABLES)
    single_star_light_switch_check(); /* node luminosities follow a star that starts to shine or goes dark */
#endif
    gravity_tree();		/* computes gravity accel. */

    /* For the first timestep, we redo it to allow usage of relative opening criterion for consistent accuracy.
       The estimator this seeds is refreshed only when the tree is built, and the import is pruned against it,
       so the repeat pass has to rebuild -- otherwise it walks the first pass's tree and seeds nothing. */
    if(All.TypeOfOpeningCriterion == 1 && All.Ti_Current == 0) {TreeReconstructFlag = 1; gravity_tree();}
  }
#ifdef TEST_DUMP_TREECOL_PIXELS_AND_EXIT
  test_dump_treecol_pixels_and_exit();
#endif

  PRINT_STATUS(" ..gravity force computation done");
}



void compute_hydro_densities_and_forces(void)
{
  if(All.TotN_gas > 0)
    {
        PRINT_STATUS("Start hydrodynamics computation...");
        /* Hydro corridor entry: decide Mode A vs Mode B once for the whole
         * corridor span (density → cellcorrections → gradients → hydro_force),
         * so every consumer in the span dispatches coherently. */
        gizmo_hydro_corridor_decide_mode();
        /* density() internally handles ghost_exchange prep + redo (neighbor-list path)
           via the ghost_symlist_lifecycle helpers. Same for gradients and hydro_force. */
        double t_bench_density_start = my_second(); double child0_density = CPU_ChildCharged;
        density();		/* computes density, and pressure */
        double t_bench_density_only = cpu_minus_children(timediff(t_bench_density_start, my_second()), child0_density);
#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
        ags_density();
#endif
        force_update_hmax();	/* update kernel lengths in tree */
        /*! This function updates the hmax-values in tree nodes that hold gas. These values are needed to find all neighbors in the hydro-force computation.  Since the KernelRadius-values are potentially changed in the gas-denity computation, force_update_hmax() should be carried out before the hydrodynamical forces are computed, i.e. after density(). */

        PRINT_STATUS(" ..density & tree-update computation done...");

#if defined(RADTRANSFER) && defined(GRAIN_RDI_TESTPROBLEM_LIVE_RADIATION_INJECTION)
        rt_source_injection(); /* doing source injection here (just before interpolation and hydro gradients) is slightly more accurate for this setup, but not possible in total generality owing to dependence of some injection modules on quantities calculated below */
#endif
#if defined(RT_OPACITY_FROM_EXPLICIT_GRAINS)
        interpolate_fluxes_opacities_gasgrains(); /* this must be called here for the computation of opacities and radiative quantity gradients below to be correct */
        CPU_Step[CPU_RTNONFLUXOPS] += measure_time();
#endif
#ifdef GALSF /* PFH set of feedback routines (thermal, photo-ionization, radiation pressure), coupled discretely just -before- the hydro force is computed; mechanical feedback is coupled after the closing kick (run.cc) */
        LEDGER_STEP_BREAKDOWN("pre_feedback");
        LEDGER_FB(compute_stellar_feedback());
        LEDGER_STEP_BREAKDOWN("post_feedback");
#endif

        /* Hydro corridor begin: in Mode A (any rank count) imports the gas
         * ghost pool and builds the shared active list + symmetric gas CSR
         * consumed by cellcorrections + gradients + hydro_force via
         * gizmo_hydro_corridor_external_csr(). In Mode B builds only the
         * shared active list (request-driven, no CSR). See hydro_corridor.h
         * for the ownership contract and mid-span refresh points.
         *
         * It sits below the injection/feedback block above because each of
         * those routines runs its own neighbor loop, and a neighbor loop's
         * ghost import first tears down the single global ghost pool. With
         * them above the corridor they can no longer tear its pool, so the
         * refresh before gradients stays on the in-place path instead of
         * re-importing and rebuilding the CSR for a neighbor list that never
         * changed. This does NOT make the span tear-free: a loop inside the
         * span that still runs its own ghost import tears it just the same.
         * Under TURB_DIFF_DYNAMIC, dynamic_diff_calc between gradients and
         * hydro_force does exactly that and costs one corridor re-import per
         * step. Consuming this CSR, the way cellcorrections and gradients do,
         * is what removes such a tear. */
        gizmo_hydro_corridor_begin();
        CPU_Step[CPU_DENSMISC] += measure_time();

#ifdef HYDRO_VOLUME_CORRECTIONS
        /* Running below feedback is NOT neutral for this routine, by intent.
         * Its volume correction reads only positions, kernel radii and the
         * zeroth-order volumes, none of which feedback touches, so that part
         * is unchanged. But its closing Density = Mass/Volume_1 and pressure
         * update read mass and internal energy, which feedback DOES write, so
         * in a feedback build those primitives are now derived from the
         * post-feedback state — the same state the gradients below consume,
         * which is the self-consistent choice. */
        cellcorrections_calc(); /* must be called after density, and after the update of hmax in the tree [because it depends on bi-directional search], but before gradients where quantities dependent on volumetric elements such as density are needed */
#endif
#ifdef TURB_DIFF_DYNAMIC
        /* Between density and gradients, and below feedback: the diffusion
         * coefficient this filter estimates is applied to the post-feedback
         * state, so measuring it on the post-feedback velocity field is the
         * self-consistent choice. */
        dynamic_diff_vel_calc();
#endif

        double t_bench_grad_start = my_second(); double child0_grad = CPU_ChildCharged;
        hydro_gradient_calc(); /* calculates the gradients of hydrodynamical quantities  */
#ifdef MHD_MODIFIED_GRADIENT
        if(gizmo_hydro_corridor_mg_step()) { /* decided once for the step by gizmo_hydro_corridor_decide_mode */
            const double t_mg_solve = my_second();
            mg_gradient_correction_calc(); /* MG method: global sparse-matrix solve for exact div(B)=0 correction coefficients (Tu et al. 2026) */
            /* The solve is charged to its own row rather than left in the gradient
             * residual.  It has to be charged SOMEWHERE: measure_time() discounts only
             * what a child bucket claims, so an unclaimed span of this size lands both
             * in the enclosing bracket and in the next residual, and is counted twice. */
            cpu_charge_child(CPU_MHD_MG, timediff(t_mg_solve, my_second()));
        }
#endif
#if defined(COOLING) && defined(GALSF_FB_FIRE_RT_LONGRANGE)
        selfshield_local_incident_uv_flux(); /* needs to be called after gravity tree (where raw flux is calculated) and the local gradient calculation (GradRho) to properly self-shield the particles that had this calculated */
#endif
        double t_bench_grad = cpu_minus_children(timediff(t_bench_grad_start, my_second()), child0_grad);
        PRINT_STATUS(" ..gradient computation done.");

#if (SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_SPECIALBOUNDARIES >= 4)
        special_rt_feedback_injection(); /* do before proper hydro loop */
#endif

#ifdef TURB_DIFF_DYNAMIC
        dynamic_diff_calc(); /* This MUST be called immediately following gradient calculations */
#endif
        double t_bench_hydro_start = my_second(); double child0_hydro = CPU_ChildCharged;
        hydro_force();		/* adds hydrodynamical accelerations and computes du/dt  */
        double t_bench_hydro = cpu_minus_children(timediff(t_bench_hydro_start, my_second()), child0_hydro);
        /* Hydro corridor exit: reset mode to UNSET. Any unexpected
         * gizmo_hydro_corridor_get_mode() outside the corridor span will
         * read UNSET and is therefore detectable. */
        gizmo_hydro_corridor_end();
        compute_additional_forces_for_all_particles(); /* other accelerations that need to be computed are done here */
        /* Feed GPU-path kernel timings into CPU_Step accumulators for cpu.txt output.
           drift+ghost+redo feeds are done inside density()/gradients()/hydro_force() via
           the ghost_symlist_lifecycle helpers. */
        CPU_Step[CPU_DENSCOMPUTE] += t_bench_density_only;
        CPU_Step[CPU_DENSWAIT] += t_bench_grad;
        CPU_Step[CPU_HYDCOMPUTE] += t_bench_hydro;
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
#pragma omp parallel for schedule(static) if(gizmo_host_threads_take(HOST_THREAD_PARTICLE_LOOP, n_active))
#endif
    for(int ii = 0; ii < n_active; ii++)
    {
        const int i = ActiveParticleList[ii];
        const double dt = get_particle_timestep_in_physical(i, P);
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
#if defined(DM_FUZZY)
    DMGrad_gradient_calc();
#endif
#if defined(CBE_INTEGRATOR_WITHGRADIENTS)
    /* Persistent CBE basis-moment gradients refreshed for AGSForce-active
     * particles each call; inactive particles retain prior-step gradients
     * (hydro semantics). Independent of DMGrad_gradient_calc (different
     * Spec, different storage); both fire when both flags are on. */
#ifdef CBE_NEIGHBOUR_CORRIDOR
    cbe_corridor_begin();   /* CBEGrad + AGSForce share one import and list per subgroup; AGSForce_calc ends it */
#endif
    CBEGrad_gradient_calc();
#endif
#if defined(DM_FUZZY) || defined(CBE_INTEGRATOR) || defined(DM_SIDM)
    AGSForce_calc();
#endif
#if defined(DO_FLUID_ALTSPECIES_DRAG_CALCULATION)
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
    if(All.Dt_Since_LastFBCalc_Gyr < All.Dt_Min_Between_FBCalc_Gyr) {return;}
#endif

    /* mechanical feedback is coupled after the closing half-kick instead (run.cc), under the same cadence decision made here */

#ifdef GALSF_FB_THERMAL
    thermal_fb_calc(); /* thermal feedback */
    gizmo_exit_bad_stop_if_requested("accel:after_thermal_fb"); CPU_Step[CPU_SNIIHEATING] += measure_time(); /* collect timings and reset clock for next timing */
#endif

#ifdef GALSF_FB_PHOTOIONIZATION_PER_DIRECTION
    photoionization_per_direction(); /* per-direction Stromgren photoionisation around single stars */
    gizmo_exit_bad_stop_if_requested("accel:after_photoionization"); CPU_Step[CPU_HIIHEATING] += measure_time();
#endif
#if defined(GALSF_FB_FIRE_RT_HIIHEATING)
    HII_heating_singledomain(); /* local photo-ionization heating */
    gizmo_exit_bad_stop_if_requested("accel:after_hii_heating"); CPU_Step[CPU_HIIHEATING] += measure_time(); /* collect timings and reset clock for next timing */
#endif

#ifdef GALSF_FB_FIRE_RT_LOCALRP
    radiation_pressure_winds_consolidated(); /* local radiation pressure */
    gizmo_exit_bad_stop_if_requested("accel:after_local_rp"); CPU_Step[CPU_LOCALWIND] += measure_time(); /* collect timings and reset clock for next timing */
#endif
    
    CPU_Step[CPU_MISC] += measure_time();
}
#endif // GALSF //
