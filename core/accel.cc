#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#ifdef GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY
#include "../mesh/neighbor_list.h"
#include "../mesh/sfc_tiles.h"
#endif

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

  /* For the first timestep, we redo it to allow usage of relative opening criterion for consistent accuracy */
  if(All.TypeOfOpeningCriterion == 1 && All.Ti_Current == 0) {gravity_tree();}

  PRINT_STATUS(" ..gravity force computation done");
}



void compute_hydro_densities_and_forces(void)
{
  if(All.TotN_gas > 0)
    {
        /* Drift ALL particles to current time before any neighbor operations.
           This eliminates lazy drifting during the tree walk — required for
           GPU neighbor finding (no critical sections) and for halo exchange
           (halo particles must be at current positions before exchange). */
        move_particles(All.Ti_Current);
        /* Ghost exchange: import boundary particles from neighboring MPI ranks.
           Use safety_factor > 1 on first timestep (restartflag=0) since initial h values
           are guesses that may grow significantly during density iteration. */
        ghost_exchange(1.0);

        PRINT_STATUS("Start hydrodynamics computation...");
        density();		/* computes density, and pressure */
#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
        ags_density();
#endif
        force_update_hmax();	/* update kernel lengths in tree */
        /*! This function updates the hmax-values in tree nodes that hold gas. These values are needed to find all neighbors in the hydro-force computation.  Since the KernelRadius-values are potentially changed in the gas-denity computation, force_update_hmax() should be carried out before the hydrodynamical forces are computed, i.e. after density(). */

        /* Check if h grew beyond the ghost pool during density iteration.
           If so, re-exchange with converged hmax to ensure complete ghost pool.
           The tree walk result (P[i].NumNgb) is independent of ghosts, so no
           need to re-run density — just need correct ghosts for the cell-list
           (and later GPU dispatch). */
        if(ghost_exchange_needs_redo()) {
            ghost_exchange_cleanup();
            ghost_exchange(1.0);
        }
        validate_neighbor_list(); /* temporary: compare cell-list neighbor finder against tree walk */

#ifdef GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY
        /* Build symmetric neighbor list (r < max(h_i, h_j)) with converged h values.
           This list is reused by gradients, volume corrections, and hydro force —
           built once here, freed after hydro_force completes. */
        {
            gizmo_sym_num_active = 0;
            for(int ii : ActiveParticleList) {if(P[ii].Type == 0 && P[ii].Mass > 0) gizmo_sym_num_active++;}
            gizmo_sym_active_indices = (int *) mymalloc("sym_active", (gizmo_sym_num_active > 0 ? gizmo_sym_num_active : 1) * sizeof(int));
            {int aa = 0; for(int ii : ActiveParticleList) {if(P[ii].Type == 0 && P[ii].Mass > 0) gizmo_sym_active_indices[aa++] = ii;}}

            double t_sym_start = my_second();
            build_neighbor_list_sfc(P, CellP, NumPart, gizmo_sym_active_indices, gizmo_sym_num_active, NGB_SEARCH_SYMMETRIC, 1, &gizmo_sym_neighbor_list);
            double t_sym_end = my_second();

            if(ThisTask == 0) {
                PRINT_STATUS("Symmetric neighbor list: %d active, %d pairs (%.4f s) — cached for gradients+hydro",
                             gizmo_sym_num_active, gizmo_sym_neighbor_list.total_pairs, timediff(t_sym_start, t_sym_end));
            }
        }
#endif

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
#ifdef GALSF /* PFH set of feedback routines; here because for e.g. strong SNe, obtain better stability if they are coupled discretely just -before- the hydro force is computed */
        compute_stellar_feedback();
#endif

        hydro_gradient_calc(); /* calculates the gradients of hydrodynamical quantities  */
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
#ifdef GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY
        /* Free symmetric neighbor list — no longer needed after hydro_force */
        free_neighbor_list(&gizmo_sym_neighbor_list);
        myfree(gizmo_sym_active_indices);
        gizmo_sym_active_indices = NULL;
        gizmo_sym_num_active = 0;
#endif
        ghost_exchange_cleanup(); /* remove ghost particles — must be before any particle count-dependent operations */
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

#ifdef GALSF_FB_MECHANICAL /* check the mechanical sources of feedback */
    mechanical_fb_calc_toplevel();  /* call the parent loop for the different mechanical fb sub-loops */
    MPI_Barrier(MPI_COMM_WORLD); CPU_Step[CPU_SNIIHEATING] += measure_time(); /* collect timings and reset clock for next timing */
#endif

#ifdef GALSF_FB_THERMAL
    thermal_fb_calc(); /* thermal feedback */
    MPI_Barrier(MPI_COMM_WORLD); CPU_Step[CPU_SNIIHEATING] += measure_time(); /* collect timings and reset clock for next timing */
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
