#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../declarations/allvars.h"
#include "../declarations/multifluid_helpers.h"  
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "../solids/grain_physics_gpu.h"

/*

 This module contains the self-contained sub-routines needed for
 grain-specific physics in proto-planetary/proto-stellar/planetary cases,
 GMC and ISM/CGM/IGM dust dynamics, dust dynamics in cool-star atmospheres,
 winds, and SNe remnants, as well as terrestrial turbulence and
 particulate-laden turbulence. Anywhere where particles coupled to gas
 via coulomb, aerodynamic, or lorentz forces are interesting.

 This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.

 */



#if defined(DO_FLUID_ALTSPECIES_DRAG_CALCULATION)

#if defined(GRAIN_BACKREACTION)
void grain_backrx(void);
#endif

extern void grain_drag_evaluate_gpu(struct particle_data *, struct gas_cell_data *, int *, int);

/* function to apply the drag on the grains from surrounding gas properties */
void apply_grain_dragforce(void)
{
    CPU_Step[CPU_MISC] += measure_time();
    PRINT_STATUS("Beginning particulate/grain/PIC force evaluation.");

    /* Gather only true active grains — non-grain actives are never processed by the
       drag kernel, so filtering here makes the grain count honest for the tiny-N
       dispatch decision inside grain_drag_evaluate_gpu. */
    {
        auto drag_eligible_caller = [](int ii) {
            return IS_PARTICLE_DRAGVALID(P[ii].Type, P[ii].FluidType) && (P[ii].Mass > 0);
        };
        int num_grain = 0;
        for(int ii : ActiveParticleList) if(drag_eligible_caller(ii)) num_grain++;
        if(num_grain > 0) {
            int *grain_indices = (int *) malloc(num_grain * sizeof(int));
            int aa = 0;
            for(int ii : ActiveParticleList) if(drag_eligible_caller(ii)) grain_indices[aa++] = ii;
            grain_drag_evaluate_gpu(P, CellP, grain_indices, num_grain);
            free(grain_indices);
        }
    }
#if defined(GRAIN_BACKREACTION)
    /* Reset active gas Grain_AccelTimeMin to MAX_REAL_NUMBER before grain_backrx min-applies
     * grain constraints. grain_backrx only reduces the field, never resets it. Without this,
     * gas particles retain zero from IC struct zero-fill, collapsing their timestep to zero.
     * This matches the implicit reset the old code did via scatter-back on every drag call. */
    for(int ii : ActiveParticleList)
        if(P[ii].Type == 0) P[ii].Grain_AccelTimeMin = MAX_REAL_NUMBER;
    grain_backrx(); /* call parent routine to assign the back-reaction force among neighbors */
#endif
    PRINT_STATUS(" ..particulate/grain/PIC force evaluation done.");
    CPU_Step[CPU_DRAGFORCE] += measure_time();
}




/* this is a template for fully-automated parallel (hybrid MPI+OpenMP/Pthreads) neighbor communication
 written in a completely modular fashion. this works as long as what you are trying to do isn't too
 complicated (from a communication point-of-view). You specify a few key variables, and then
 define the variables that need to be passed, and write the actual sub-routine that does the actual
 'work' between neighbors, but all of the parallelization, looping, communication blocks,
 etc, are all handled for you. */

void grain_backrx(void)
{
    PRINT_STATUS(" ..assigning grain back-reaction to gas");
    {
        /* Build LOCAL active grain list (i-type restricted by GRAIN_PTYPES). */
        int num_active = 0;
        for(int i : ActiveParticleList) {
            if((IS_PARTICLE_DRAGVALID(P[i].Type, P[i].FluidType)) && (P[i].TimeBin >= 0) && (P[i].Mass > 0) && (P[i].KernelRadius > 0)) num_active++;
        }
        int *nl_active = (int *) mymalloc("grainbackrx_nl_active",
            (num_active > 0 ? num_active : 1) * sizeof(int));
        double *nl_radii = (double *) mymalloc("grainbackrx_nl_radii",
            (num_active > 0 ? num_active : 1) * sizeof(double));
        {int aa = 0; for(int i : ActiveParticleList) {
            if((IS_PARTICLE_DRAGVALID(P[i].Type, P[i].FluidType)) && (P[i].TimeBin >= 0) && (P[i].Mass > 0) && (P[i].KernelRadius > 0)) {
                nl_active[aa] = i; nl_radii[aa] = (double)P[i].KernelRadius; aa++;
            }
        }}
        grain_backrx_evaluate_gpu(P, CellP, NumPart, nl_active, num_active, nl_radii);
        myfree(nl_radii); myfree(nl_active);
        CPU_Step[CPU_DRAGFORCE] += measure_time();
    }
}







#ifdef GRAIN_COLLISIONS
/* return_grain_cross_section_per_unit_mass / prob_of_grain_interaction /
   calculate_interact_kick were host-only functions with implicit global-P
   access and GSL RNG. They have been moved to:
     - solids/grain_helper_functions.h (return_grain_cross_section_per_unit_mass_P,
       prob_of_grain_interaction_tab)
     - sidm/sidm_helper_functions.h (calculate_interact_kick_rng)
   as KOKKOS_INLINE_FUNCTION forms that thread particle_data and the
   GeoFactorTable as explicit args and use the counter-based gpu_rng.
   CPU callers pass `GeoFactorTable` directly; the B2 AGSForce GPU kernel
   passes a SharedSpace mirror. */
#endif






#if defined(RT_OPACITY_FROM_EXPLICIT_GRAINS)

void interpolate_fluxes_opacities_gasgrains(void)
{
    PRINT_STATUS(" ..assigning opacities to gas from the grain distribution, and interpolating radiation fields to grains");
    {
        int num_gas = 0, num_grain = 0;
        for(int i : ActiveParticleList) {
            if(P[i].TimeBin < 0 || P[i].Mass <= 0 || P[i].KernelRadius <= 0) continue;
            if(P[i].Type == 0) num_gas++;
            if((1 << P[i].Type) & (GRAIN_PTYPES)) num_grain++;
        }
        int ng = (num_gas > 0 ? num_gas : 1), nr = (num_grain > 0 ? num_grain : 1);
        int *nl_active_gas   = (int *)    mymalloc("gasgrainrt_nl_gas_active",   ng * sizeof(int));
        double *nl_radii_gas = (double *) mymalloc("gasgrainrt_nl_gas_radii",    ng * sizeof(double));
        int *nl_active_grain   = (int *)    mymalloc("gasgrainrt_nl_grain_active", nr * sizeof(int));
        double *nl_radii_grain = (double *) mymalloc("gasgrainrt_nl_grain_radii",  nr * sizeof(double));
        {int ag = 0, ar = 0; for(int i : ActiveParticleList) {
            if(P[i].TimeBin < 0 || P[i].Mass <= 0 || P[i].KernelRadius <= 0) continue;
            if(P[i].Type == 0) { nl_active_gas[ag] = i; nl_radii_gas[ag] = (double)P[i].KernelRadius; ag++; }
            if((1 << P[i].Type) & (GRAIN_PTYPES)) { nl_active_grain[ar] = i; nl_radii_grain[ar] = (double)P[i].KernelRadius; ar++; }
        }}
        interpolate_fluxes_opacities_gasgrains_evaluate_gpu(P, CellP, NumPart,
            nl_active_gas, num_gas, nl_radii_gas,
            nl_active_grain, num_grain, nl_radii_grain);
        myfree(nl_radii_grain); myfree(nl_active_grain);
        myfree(nl_radii_gas); myfree(nl_active_gas);
        CPU_Step[CPU_DRAGFORCE] += measure_time();
    }
}



double return_grain_extinction_efficiency_Q(int i, int k_freq)
{
    double Q = 1; /* default to geometric opacity */
#if defined(GRAIN_RDI_TESTPROBLEM)
    Q *= All.Grain_Q_at_MaxGrainSize; // this needs to be set by-hand, Q for the maximum sized grains. irrelevant for the scale-free problem (degenerate with flux), but important here */
#if !defined(GRAIN_RDI_TESTPROBLEM_ACCEL_DEPENDS_ON_SIZE)
    Q *= P[i].Grain_Size / All.Grain_Size_Max;
#endif
#else
    /* INSERT PHYSICS HERE -- this is where you want to specify the optical properties of grains relative to the frequency bins being evolved. could code up something for -ALL- the bins we do, but that's a lot, so we'll do these as-needed, for runs with different frequencies */
#if defined(RADTRANSFER) || defined(RT_USE_GRAVTREE)
    double nu_min_ev=All.RHD_bins_nu_min_ev[k_freq], nu_max_ev=All.RHD_bins_nu_max_ev[k_freq]; // get the radiation frequency range in eV
    double x_min = 5.068e4 * P[i].Grain_Size * nu_min_ev, x_max = 5.068e4 * P[i].Grain_Size * nu_max_ev; // this is the 'x' parameter for Q, defined as 2*pi*a_grain/lambda_light
    /* don't have a detailed model here for dielectric properties of grains, so instead use an extremely simple model as follows */
    double x_eff=sqrt(x_min*x_max); if(x_min<=MIN_REAL_NUMBER) {x_eff=0.5*x_max;} // take geometric mean or linear mean if former not well-defined
    return DMIN(x_eff, 1.); // simple model where Q~x for x<<1, Q=1 for x>>1
#else
    if(ThisTask==0) {PRINT_WARNING("Code does not have entered grain absorption efficiency/optical properties for your specific wavelength being evolved. Please enter that information in the routine 'return_grain_extinction_efficiency_Q'. For now will assume geometric absorption (Q=1). \n");}
#endif
#endif
    return Q;
}



#endif //defined(RT_OPACITY_FROM_EXPLICIT_GRAINS)




#endif
