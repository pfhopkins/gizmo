#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"

/*! \file accel.c
 *  \brief driver routines to carry out force computation
 */


/*! This routine computes the accelerations for all active particles.  First, the gravitational forces are
 * computed. This also reconstructs the tree, if needed, otherwise the drift/kick operations have updated the
 * tree to make it fullu usable at the current time.
 *
 * If gas particles are presented, the `interior' of the local domain is determined. This region is guaranteed
 * to contain only particles local to the processor. This information will be used to reduce communication in
 * the hydro part.  The density for active SPH particles is computed next. If the number of neighbours should
 * be outside the allowed bounds, it will be readjusted by the function ensure_neighbours(), and for those
 * particle, the densities are recomputed accordingly. Finally, the hydrodynamical forces are added.
 */

/*
 * This file was originally part of the GADGET3 code developed by
 * Volker Springel (volker.springel@h-its.org). The code has been modified
 * slightly (re-arranged, consolidated, and added compute_stellar_feedback and 
 * the gradients loop) by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

void compute_grav_accelerations(void)
{
  CPU_Step[CPU_MISC] += measure_time();

  if(ThisTask == 0)
    {
      printf("Start gravity force computation...\n");
#ifndef IO_REDUCED_MODE
      fflush(stdout);
#endif
    }

#ifdef PMGRID
  if(All.PM_Ti_endstep == All.Ti_Current)
    {
      long_range_force();
      CPU_Step[CPU_MESH] += measure_time();
    }
#endif

  gravity_tree();		/* computes gravity accel. */

  /* For the first timestep, we redo it to allow usage of 
   relative opening criterion for consistent accuracy */
  if(All.TypeOfOpeningCriterion == 1 && All.Ti_Current == 0)
    gravity_tree();

#ifndef IO_REDUCED_MODE
  if(ThisTask == 0)
    {
      printf("gravity force computation done.\n");
      fflush(stdout);
    }
#endif
}



void compute_hydro_densities_and_forces(void)
{
  if(All.TotN_gas > 0)
    {
        if(ThisTask == 0)
        {
            printf("Start hydrodynamics computation...\n");
        }
#ifndef IO_REDUCED_MODE
        if(ThisTask == 0)
        {
            printf("Start density & tree-update computation...\n");
        }
#endif
        density();		/* computes density, and pressure */
#ifdef ADAPTIVE_GRAVSOFT_FORALL
        ags_density();
#endif
        force_update_hmax();	/* update kernel lengths in tree */
        /*! This function updates the hmax-values in tree nodes that hold SPH
         *  particles. These values are needed to find all neighbors in the
         *  hydro-force computation.  Since the Hsml-values are potentially changed
         *  in the SPH-denity computation, force_update_hmax() should be carried
         *  out before the hydrodynamical SPH forces are computed, i.e. after
         *  density().
         */
        
#ifndef IO_REDUCED_MODE
        if(ThisTask == 0)
        {
            printf("density & tree-update computation done...\n");
        }
#endif
        hydro_gradient_calc(); /* calculates the gradients of hydrodynamical quantities  */
#if defined(COOLING) && defined(GALSF_FB_LOCAL_UV_HEATING)
        selfshield_local_incident_uv_flux();
        /* needs to be called after gravity tree (where raw flux is calculated) 
         and the local gradient calculation (GradRho) to
         properly self-shield the particles that had this calculated */
#endif
#ifndef IO_REDUCED_MODE
        if(ThisTask == 0)
        {
            printf("gradient computation done.\n");
        }
#endif
        hydro_force();		/* adds hydrodynamical accelerations and computes du/dt  */
#ifndef IO_REDUCED_MODE
        if(ThisTask == 0)
        {
            printf("hydro force computation done.\n");
        }
#endif
#ifdef GRAIN_FLUID
        apply_grain_dragforce(); /* if we are solving a coupled set of grains via aerodynamic drag, this is where their acceleration should be calculated */
#ifndef IO_REDUCED_MODE
        if(ThisTask == 0)
        {
            printf("grain aerodynamic force evaluation done.\n");
        }
#endif
#endif

    } else {
#ifdef ADAPTIVE_GRAVSOFT_FORALL
        ags_density(); // if there are no gas particles but ags-all is active, still need to enter this loop //
        force_update_hmax();    /* update kernel lengths in tree */
#endif
    }
}


#ifdef GALSF
void compute_stellar_feedback(void)
{
    CPU_Step[CPU_MISC] += measure_time();

    /* first, check the mechanical sources of feedback */
#if defined(GALSF_FB_SNE_HEATING) || defined(GALSF_FB_GASRETURN) || defined(GALSF_FB_RPROCESS_ENRICHMENT)
    mechanical_fb_calc(-2); /* compute weights for coupling [first weight-calculation pass] */
    mechanical_fb_calc(-1); /* compute weights for coupling [second weight-calculation pass] */
    CPU_Step[CPU_SNIIHEATING] += measure_time();
#ifdef GALSF_FB_SNE_HEATING
    mechanical_fb_calc(0); /* actually do the SNe coupling */
    CPU_Step[CPU_SNIIHEATING] += measure_time();
#endif
#if defined(GALSF_FB_GASRETURN)
    mechanical_fb_calc(1); /* do the gas return coupling */
    CPU_Step[CPU_GASRETURN] += measure_time();
#endif
#ifdef GALSF_FB_RPROCESS_ENRICHMENT
    mechanical_fb_calc(2); /* do the R-process element injection */
    CPU_Step[CPU_GASRETURN] += measure_time();
#endif
#endif // (defined(GALSF_FB_SNE_HEATING)||defined(GALSF_FB_GASRETURN) || defined(GALSF_FB_RPROCESS_ENRICHMENT))
    
    /* alternatively use the 'turn off cooling' sub-grid feedback model */
#ifdef GALSF_GASOLINE_RADHEATING
    luminosity_heating_gasoline();
    CPU_Step[CPU_SNIIHEATING] += measure_time();
#endif
    
    /* now do the local photo-ionization heating */
#ifdef GALSF_FB_HII_HEATING
    HII_heating_singledomain();
    CPU_Step[CPU_HIIHEATING] += measure_time();
#endif
    
    /* finally (if we're not doing it in the star formation routine), do the local radiation pressure */
#if defined(GALSF_FB_RPWIND_FROMSTARS) && !defined(GALSF_FB_RPWIND_DO_IN_SFCALC)
    radiation_pressure_winds_consolidated();
    CPU_Step[CPU_LOCALWIND] += measure_time();
#endif
    
    CPU_Step[CPU_MISC] += measure_time();
}
#endif // GALSF //
