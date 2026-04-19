#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include "../declarations/gizmo_quadrature.h"

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"

#define GSLWORKSIZE 100000

/*! \file sidm_routines.c
 *  \brief Fuctions and routines needed for the calculations of dark matter self interactions
 *
 *  This file contains the functions and routines necesary for the computation of
 *  the self-interaction probabilities and the velocity kicks due to the interactios.
 *  Originally written by Miguel Rocha, rocham@uci.edu. Oct 2010. Updated on 2014 & re-written by PFH March 2018
 */

/*! This function calculates the interaction probability between two particles.
 *  It checks if comoving integration is on and does the necesary change of
 *  variables and units.
 */

#ifdef DM_SIDM

/* prob_of_interaction / calculate_interact_kick / g_geo used to live here as
   host-only functions with GSL RNG and global-GeoFactorTable access. They
   have been moved to sidm/sidm_helper_functions.h as KOKKOS_INLINE_FUNCTION
   so both the CPU tree-walk (via sidm_core_flux_functions.h) and the B2
   GPU AGSForce kernel can call them unchanged. CPU callers now pass
   `GeoFactorTable` explicitly; the scatter RNG uses the counter-based
   generator in declarations/gpu_rng.h. See that header for the migration
   rationale. */

/*! This routine initializes the table that will be used to get the geometrical factor
 *  as a function of the two particle separations. It populates a table with the results of the numerical integration */
void init_geofactor_table(void)
{
    int i; double r;
    for(i = 0; i < GEOFACTOR_TABLE_LENGTH; i++)
    {
        r = 2.0/GEOFACTOR_TABLE_LENGTH * (i + 1);
        GeoFactorTable[i] = 2*M_PI * gizmo_gl20_integrate(&geofactor_integ, 0.0, 1.0, &r);
    }
}

/*! This function returns the integrand of the numerical integration done on init_geofactor_table(). */
double geofactor_integ(double x, void * params)
{
    double r = *(double *) params, newparams[2];
    newparams[0] = r; newparams[1] = x;
    double result = gizmo_gl20_integrate(&geofactor_angle_integ, -1.0, 1.0, newparams);
    double wk=0; if(x<1) kernel_main(x, 1, 1, &wk, &wk, -1);
    return x*x*wk*result;
}

/*! This function returns the integrand of the angular part of the integral done on init_geofactor_table(). */
double geofactor_angle_integ(double u, void * params)
{
    double x,r,f;
    double *dparams = (double *) params;
    r = dparams[0];
    x = dparams[1];
    f = sqrt(x*x + r*r + 2*x*r*u);
    double wk=0; if(f<1) kernel_main(f, 1, 1, &wk, &wk, -1); /*! This function returns the value W(x). The values of the density kernel as a funtion of x=r/h */
    return wk;
}

/*! This function simply initializes some variables to prevent memory errors */
void init_self_interactions() {int i; for(i = 0; i < NumPart; i++) {P[i].dtime_sidm = 0; P[i].NInteractions = 0;}}

#endif
