/* Copyright (c) 2026 Ulrich Steinwandel
 * Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* galaxy_sf/photoion_per_direction.cc -- per-direction photoionisation around single stars (GALSF_FB_PHOTOIONIZATION_PER_DIRECTION).
 * All ionising single stars share one cadence: their timestep is at most the bin of PhotoionMaxTimestep, so every one of
 * them is active whenever that bin is. Reference: stella_dev galaxy_sf/resolvedism_photoion.cc. */
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "../declarations/allvars.h"
#include "../core/proto.h"

#ifdef GALSF_FB_PHOTOIONIZATION_PER_DIRECTION
#if !defined(GALSF_SINGLE_STARS_FROM_IMF) || !defined(GALSF_SINGLE_STARS_EVOLUTION_TABLES)
#error "GALSF_FB_PHOTOIONIZATION_PER_DIRECTION: the sources are single stars (GALSF_SINGLE_STARS_FROM_IMF, GALSF_SINGLE_STARS_EVOLUTION_TABLES)"
#endif
#ifdef GALSF_FB_FIRE_RT_HIIHEATING
#error "GALSF_FB_PHOTOIONIZATION_PER_DIRECTION replaces the HII-region loop (GALSF_FB_FIRE_RT_HIIHEATING): switch on one of them"
#endif

/* ionising photon rate of source i [1/s] */
static double photoion_source_rate(int i)
{
    if(P[i].StarIdx <= 0 || P[i].Mass <= 0) {return 0;}
    return All.HIIRegion_fLum_Coupled * single_star_ionizing_photon_rate(i);
}

/* the photoionisation timebin: the largest bin whose step is within PhotoionMaxTimestep and MaxSizeTimestep */
static int photoion_timebin(void)
{
    double dt = DMIN(All.PhotoionMaxTimestep / UNIT_TIME_IN_YR, All.MaxSizeTimestep);
    integertime ti = (integertime)(dt / All.Timebase_interval);
    int b = 1;
    while(b + 1 < TIMEBINS && ((integertime)1 << (b + 1)) <= ti) {b++;}
    return b;
}

/* largest timestep of single star p [code]: an ionising source keeps to the photoionisation bin; a star that will
   ionise steps onto its onset age */
double photoion_max_timestep(int p)
{
    double t_on = single_star_ionizing_onset_yr(p);
    if(t_on < 0) {return 0;}
    if(photoion_source_rate(p) > 0) {return (double)((integertime)1 << photoion_timebin()) * All.Timebase_interval;}
    double age = evaluate_stellar_age_Gyr(p) * 1.e9;
    if(age < t_on) {return DMAX(t_on - age, 10.) / UNIT_TIME_IN_YR;}
    return 0;
}

#endif /* GALSF_FB_PHOTOIONIZATION_PER_DIRECTION */
