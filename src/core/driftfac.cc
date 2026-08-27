#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/gizmo_quadrature.h"

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../core/timestep_functions.h"

/*!
 * This routine calculates the pre-factors and timesteps for cosmological 
 *  simulations where the time-stepping is done in co-moving units and the 
 *  time units are the scale factor. Basically the pre-factors that need to be 
 *  combined and/or integrated are calculated here, to be combined with the 
 *  appropriate time derivatives as calculated in code units elsewhere.
 */

/*!
 * This file was originally part of the GADGET3 code developed by
 * Volker Springel. The code has been modified heavily
 * by Phil Hopkins (phopkins@caltech.edu) for GIZMO; the conventions for the
 * timestep units are different so this is revised here. Computation uses
 * different libraries now as well. The GADGET approximations have been replaced
 * entirely to allow for more general cosmologies.
 */

static double logTimeBegin;
static double logTimeMax;


double drift_integ(double a, void *param)
{
  double h;

  h = hubble_function(a);

  return 1 / (h * a * a * a);
}

double gravkick_integ(double a, void *param)
{
  double h;

  h = hubble_function(a);

  return 1 / (h * a * a);
}

double growthfactor_integ(double a, void *param)
{
  double s;

  s = hubble_function(a) / All.Hubble_H0_CodeUnits * sqrt(a * a * a);

  return pow(sqrt(a) / s, 3);
}


void init_drift_table(void)
{
  int i;
  logTimeBegin = log(All.TimeBegin);
  logTimeMax = log(All.TimeMax);

  for(i = 0; i < DRIFT_TABLE_LENGTH; i++)
    {
      double a0 = exp(logTimeBegin);
      double a1 = exp(logTimeBegin + ((logTimeMax - logTimeBegin) / DRIFT_TABLE_LENGTH) * (i + 1));
      DriftTable[i]   = gizmo_gl20_integrate(&drift_integ,   a0, a1, NULL);
      GravKickTable[i] = gizmo_gl20_integrate(&gravkick_integ, a0, a1, NULL);
    }
}


/* init_drift_table() runs only under ComovingIntegrationOn, so on a non-cosmological run the
 * tables below are never filled and never read; the elapsed-time branch is taken instead.
 */

/*! This function integrates the cosmological prefactor for a drift step between time0 and time1. The value returned is
 *  \f[ \int_{a_0}^{a_1} \frac{{\rm d}a}{H(a)} \f]
 *  A lookup-table is used for reasons of speed. mode selects whose timestep dilation applies:
 *  0 for particle i, 1 for tree node i.
 */

double get_drift_factor(integertime time0, integertime time1, int i, int mode)
{
    double dilation = mode ? return_node_timestep_dilation_factor(i) : timestep_dilation_factor(i, P);
    struct DriftKickTableView view = drift_kick_table_view(DriftTable, GravKickTable,
            logTimeBegin, logTimeMax, All.Timebase_interval, All.ComovingIntegrationOn);
    return get_drift_factor_impl(time0, time1, dilation, &view);
}


double get_gravkick_factor(integertime time0, integertime time1, int i, int mode)
{
    double dilation = mode ? return_node_timestep_dilation_factor(i) : timestep_dilation_factor(i, P);
    struct DriftKickTableView view = drift_kick_table_view(DriftTable, GravKickTable,
            logTimeBegin, logTimeMax, All.Timebase_interval, All.ComovingIntegrationOn);
    return get_gravkick_factor_impl(time0, time1, dilation, &view);
}
