#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#ifdef RT_SOURCE_INJECTION
#include "rt_source_injection_gpu.h"
#endif

/*! \file rt_source_injection.c
 *  \brief inject luminosity from point sources to neighboring gas particles
 *
 *  This file contains a loop modeled on the gas density computation which will
 *    share luminosity from non-gas particles to the surrounding gas particles,
 *    so that it can be treated within e.g. the flux-limited diffusion or other
 *    radiation-hydrodynamic approximations. Basically the same concept as
 *    injecting the radiation 'in a cell' surrounding the particle
 */
/*
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifdef RT_SOURCE_INJECTION


int rt_sourceinjection_active_check(int i);
int rt_sourceinjection_active_check(int i)
{
    if(P[i].NumNgb <= 0) return 0;
    if(P[i].KernelRadius <= 0) return 0;
    if(P[i].Mass <= 0) return 0;
    if(P[i].KernelSum_Around_RT_Source <= 0) return 0;
#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
    if(P[i].Type == 5 && !P[i].do_gas_search_this_timestep) return 0;
#endif
    double lum[N_RT_FREQ_BINS];
    return rt_get_source_luminosity(i,-1,lum, P, CellP);
}


/*! operations that need to be performed before entering the main loop */
void rt_source_injection_initial_operations_preloop(void);
void rt_source_injection_initial_operations_preloop(void)
{
    /* first, we do a loop over the gas particles themselves. these are trivial -- they don't need to share any information,
     they just determine their own source functions. so we don't need to do any loops. and we can zero everything before the loop below. */
    if(!(RT_SOURCES & 1)) return; // we skip this if gas cells don't have explicit source terms

    int j;
    for(j=0;j<NumPart;j++) {
        if(P[j].Type==0) {
            double lum[N_RT_FREQ_BINS]; int k;
            for(k=0;k<N_RT_FREQ_BINS;k++) {CellP[j].Rad_Je[k]=0;} // need to zero -before- calling injection //
            int active_check = rt_get_source_luminosity(j,0,lum, P, CellP);
            /* here is where we would need to code some source luminosity for the gas */
            for(k=0;k<N_RT_FREQ_BINS;k++) if(active_check) {CellP[j].Rad_Je[k]=lum[k];}
        }
    }
}




/*! subroutine that actually distributes the luminosity as desired to neighbor particles in the kernel */
/*!   -- this subroutine writes to shared memory [updating the neighbor values]: need to protect these writes for openmp below. none of the modified values are read, so only the write block is protected. */



/*! routine to do the top-level loop over particles, for the source injection (photons put into surrounding gas) */
void rt_source_injection(void)
{
    PRINT_STATUS(" ..injecting radiation onto grid for RHD steps");
    rt_source_injection_initial_operations_preloop(); /* operations before the main loop */
    {
        /* Build LOCAL active-source list from ActiveParticleList; iterating
           NumPart here would include ghost imports and double-deposit. */
        int num_active = 0;
        for(int i : ActiveParticleList) { if(rt_sourceinjection_active_check(i)) num_active++; }
        int *nl_active = (int *) mymalloc("rtsrc_nl_active",
            (num_active > 0 ? num_active : 1) * sizeof(int));
        double *nl_radii = (double *) mymalloc("rtsrc_nl_radii",
            (num_active > 0 ? num_active : 1) * sizeof(double));
        {int aa = 0; for(int i : ActiveParticleList) {
            if(rt_sourceinjection_active_check(i)) {
                nl_active[aa] = i;
                double sr = (double)P[i].KernelRadius;
#ifdef RT_SINK_ANGLEWEIGHT_PHOTON_INJECTION
                sr *= 3.0; /* matches dispatcher: RT_SINK_ANGLEWEIGHT checks P[j].KernelRadius too */
#endif
                nl_radii[aa] = sr; aa++;
            }
        }}
        rt_source_injection_evaluate_gpu(P, CellP, NumPart, nl_active, num_active, nl_radii);
        myfree(nl_radii); myfree(nl_active);
    }
    CPU_Step[CPU_RTNONFLUXOPS] += measure_time(); /* collect timings and reset clock for next timing */
}





#endif
