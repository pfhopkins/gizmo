#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"


/* Routines for pure thermal/scalar feedback/enrichment models: these are intended to represent
    extremely simplified models for stellar feedback manifest as a "pure thermal energy dump"
    (potentially with some cooling turnoff). This is -not- a model for mechanical feedback
    (which, critically, must include the actual momentum and solve for wind/SNe shock properties
    at the interface with the ISM). Those physics are included in the mechanical_fb.c file and
    algorithms therein. This also uses an extremely simple kernel-weighting, rather than a
    more self-consistent area weighting. */

/*
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */


#if defined(GALSF_FB_THERMAL)

/* routine that evaluates whether a FB event occurs in a given particle, in a given timestep */
void determine_where_addthermalFB_events_occur(void)
{
    int i; double check = 0;
    for (int i : ActiveParticleList)
    {
        if(P[i].Type != 4) {continue;}
        if(P[i].Mass <= 0) {continue;}
        check += mechanical_fb_calculate_eventrates(i,1); // this should do the calculation and add to number of SNe as needed //
    } // for (int i : ActiveParticleList) //
}

int addthermalFB_evaluate_active_check(int i);
int addthermalFB_evaluate_active_check(int i)
{
    if(P[i].Type != 4) {return 0;} // note quantities used here must -not- change in the loop [hence not using mass here], b/c can change offsets for return from different processors, giving a negative mass and undefined behaviors
    if(P[i].KernelRadius <= 0) {return 0;}
    if(P[i].NumNgb <= 0) {return 0;}
    if(P[i].SNe_ThisTimeStep>0) {return 1;}
    return 0;
}


/*!   -- this subroutine writes to shared memory [updating the neighbor values]: need to protect these writes for openmp below */
void thermal_fb_calc(void)
{
    PRINT_STATUS(" ..depositing thermal feedback to gas");
    {
#include "../galaxy_sf/thermal_fb_gpu.h"
        /* Build LOCAL active-source list from ActiveParticleList; iterating
           NumPart here would include ghost imports and double-deposit. */
        int num_active = 0;
        for(int i : ActiveParticleList) { if(addthermalFB_evaluate_active_check(i)) num_active++; }
        int *nl_active = (int *) mymalloc("thermalfb_nl_active",
            (num_active > 0 ? num_active : 1) * sizeof(int));
        double *nl_radii = (double *) mymalloc("thermalfb_nl_radii",
            (num_active > 0 ? num_active : 1) * sizeof(double));
        {int aa = 0; for(int i : ActiveParticleList) {
            if(addthermalFB_evaluate_active_check(i)) {
                nl_active[aa] = i; nl_radii[aa] = (double)P[i].KernelRadius; aa++;
            }
        }}
        thermal_fb_evaluate_gpu(P, CellP, NumPart, nl_active, num_active, nl_radii);
        myfree(nl_radii); myfree(nl_active);
        CPU_Step[CPU_SNIIHEATING] += measure_time();
    }
}




#endif /* GALSF_FB_THERMAL */

