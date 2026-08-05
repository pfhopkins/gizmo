#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../system/gpu_particles_arena.h"
#include "../mesh/kernel.h"
#define GRAVTREE_SOURCE_HOST_OWNER_TU  /* expose the source-helper core bodies (host-inline) for the wrappers below */
#include "sink_functions.h"
#undef GRAVTREE_SOURCE_HOST_OWNER_TU

/*! \file sink_util.c
 *  \brief util routines for memory (de)allocation and array setting for sink particles
 */
/*
* This file is largely written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
* see notes in sink.c for details on code history.
*/

#ifdef SINK_PARTICLES // top-level flag [needs to be here to prevent compiler breaking when this is not active] //

/* function for allocating temp BH data struc needed for feedback routines*/
void sink_start(void)
{
    int i, Nbh;

    /* count the num BHs on this task */
    N_active_loc_Sink=0;
    for (int i : ActiveParticleList)
    {
        if(sink_isactive(i))
        {
            P[i].IndexMapToTempStruc = N_active_loc_Sink;         /* allows access via SinkTempInfo[P[i].IndexMapToTempStruc] */
            N_active_loc_Sink++;                     /* N_active_loc_Sink now set for BH routines */
        }
    }

    /* allocate the sink temp struct */
    if(N_active_loc_Sink>0)
    {
        SinkTempInfo = (struct sink_temp_particle_data *) mymalloc("SinkTempInfo", N_active_loc_Sink * sizeof(struct sink_temp_particle_data));
    } else {
        SinkTempInfo = (struct sink_temp_particle_data *) mymalloc("SinkTempInfo", 1 * sizeof(struct sink_temp_particle_data));
    }

    memset( &SinkTempInfo[0], 0, N_active_loc_Sink * sizeof(struct sink_temp_particle_data) );

    Nbh=0;
    for (int i : ActiveParticleList)
    {
        if(sink_isactive(i))
        {
            SinkTempInfo[Nbh].index = i;               /* only meaningful field set here */
            Nbh++;
        }
    }

    /* all future loops can now take the following form:
     for(i=0; i<N_active_loc_Sink; i++)
     {
     i_old = SinkTempInfo[i].index;
     ...
     }
     */

}


/* function for freeing temp BH data struc needed for feedback routines*/
void sink_end(void)
{
#ifndef OUTPUT_ADDITIONAL_RUNINFO
    if(All.HighestActiveTimeBin == All.HighestOccupiedTimeBin)
#endif
    {
        int bin;
        double mdot, mdot_in_msun_per_year;
        double mass_real, total_mass_real, medd, total_mdoteddington;
        double mass_holes, total_mass_holes, total_mdot;

        /* sum up numbers to print for summary of the sink step  */
        mdot = mass_holes = mass_real = medd = 0;
        for(bin = 0; bin < TIMEBINS; bin++)
        {
            if(TimeBinCount[bin])
            {
                mass_holes += TimeBin_Sink_mass[bin];
                mass_real += TimeBin_Sink_dynamicalmass[bin];
                mdot += TimeBin_Sink_Mdot[bin];
                medd += TimeBin_Sink_Medd[bin];
            }
        }
        MPI_Reduce(&mass_holes, &total_mass_holes, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&mass_real, &total_mass_real, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&mdot, &total_mdot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&medd, &total_mdoteddington, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if((ThisTask == 0) && (total_mdot > 0) && (total_mass_real > 0))
        {
            /* convert to solar masses per yr */
            mdot_in_msun_per_year = total_mdot * UNIT_MASS_IN_SOLAR/UNIT_TIME_IN_YR;
            total_mdoteddington *= 1.0 / (sink_eddington_mdot(1) * All.TotSinks);
            fprintf(FdSinks, "%.16g %d %g %g %g %g %g\n", All.Time, All.TotSinks, total_mass_holes, total_mdot, mdot_in_msun_per_year, total_mass_real, total_mdoteddington);
        }
        fflush(FdSinks);
#ifdef OUTPUT_SINK_ACCRETION_HIST
        fflush(FdSinkSwallowDetails);
#endif
#ifdef OUTPUT_SINK_FORMATION_PROPS
        fflush(FdSinkFormationDetails);
#endif
#if defined(OUTPUT_ADDITIONAL_RUNINFO) || defined(SINK_OUTPUT_MOREINFO)
        fflush(FdSinksDetails);
#ifdef SINK_OUTPUT_MOREINFO
        fflush(FdSinkMergerDetails);
#ifdef SINK_WIND_KICK
        fflush(FdSinkWindDetails);
#endif
#endif
#endif
    }
    myfree(SinkTempInfo);
}


/* Bodies of the four accretion-luminosity helpers below live in
 * sink_functions.h (single source for host + device); these are the host
 * external wrappers. */

/* return the eddington accretion-rate = L_edd/(epsilon_r*c*c) - for a variable radiative efficiency, this scales off of the 'canonical' value given as the constant in the parameterfile, per usual convention in the literature */
double sink_eddington_mdot(double sink_mass)
{
    return sink_eddington_mdot_core(sink_mass);
}


/* return the bh luminosity given some accretion rate and mass (allows for non-standard models: radiatively inefficient flows, stellar sinks, etc) */
double sink_lum_bol(double mdot, double mass, long pindex)
{
    return sink_lum_bol_core(mdot, mass, pindex, P);
}


/* return the bh radiative efficiency - allows for models with dynamical or accretion-dependent radiative efficiencies */
double evaluate_sink_radiative_efficiency(double mdot, double mass, long pindex)
{
    return evaluate_sink_radiative_efficiency_core(mdot, mass, pindex);
}


/* return the bh cosmic ray acceleration efficiency - allows for models with dynamical or accretion-dependent cosmic ray efficiencies */
double evaluate_sink_cosmicray_efficiency(double mdot, double mass, long pindex)
{
    return evaluate_sink_cosmicray_efficiency_core(mdot, mass, pindex);
}


void sink_properties_loop(void) /* Note, normalize_temp_info_struct is now done at the end of sink_environment_loop(), so that final quantities are available for the second environment loop if needed */
{
    int i, n; double dt;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) private(n, dt) if(N_active_loc_Sink >= 16)
#endif
    for(i=0; i<N_active_loc_Sink; i++)
    {
        n = SinkTempInfo[i].index;
        dt = get_particle_feedback_timestep_in_physical(n, P);
#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
        dt = P[n].dt_since_last_gas_search;
#endif
        P[n].Sink_Mdot=0;  /* always initialize/default to zero accretion rate */
        set_sink_long_range_rp(i, n);
        set_sink_mdot(i, n, dt);
        set_sink_drag(i, n, dt);
        set_sink_new_mass(i, n, dt);
        /* results dumped to 'sink_details' files at the end of sink_final_operations so that BH mass is corrected for mass loss to radiation/bal outflows */
    }// for(i=0; i<N_active_loc_Sink; i++)
    /* No arena invalidate needed under UVM-canonical: arena aliases host P[],
     * the set_sink_* helpers' host writes ARE the arena write — coherent by
     * construction. */
}


#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
/* routine to assess whether stellar properties allow for a star to be merged if in a too-close binary */
int is_star_eligible_for_binary_merge_away(int j)
{
    int merge_key = 1;
    if(P[j].ProtoStellarStage < 5) {merge_key = 0;} /* only allow mergers once the stars reach the main sequence */
    double star_age = evaluate_stellar_age_Gyr(j) / UNIT_TIME_IN_GYR; /* stellar age */
    if(P[j].Sink_Mass < 2.*P[j].Sink_Mdot*star_age) {merge_key = 0;} /* dont allow mergers if the star is still growing very rapidly */
    return merge_key;
}
#endif


#endif // top-level flag
