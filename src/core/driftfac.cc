#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/gizmo_quadrature.h"

#include "../declarations/allvars.h"
#include "../core/proto.h"

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


/*! This function integrates the cosmological prefactor for a drift step between time0 and time1. The value returned is
 *  \f[ \int_{a_0}^{a_1} \frac{{\rm d}a}{H(a)} \f]
 *  A lookup-table is used for reasons of speed.
 */

double get_drift_factor(integertime time0, integertime time1, int i, int mode)
{
    double a1, a2, df1, df2, u1, u2;
    int i1, i2;
    double dilation = mode ? return_node_timestep_dilation_factor(i) : timestep_dilation_factor(i, P);

    if(All.ComovingIntegrationOn == 0)
    {
        return (time1 - time0) * All.Timebase_interval * dilation;
    }
    else
    {
        /* note: will only be called for cosmological integration */
        a1 = logTimeBegin + time0 * All.Timebase_interval;
        a2 = logTimeBegin + time1 * All.Timebase_interval;
        
        if(logTimeMax > logTimeBegin)
            u1 = (a1 - logTimeBegin) / (logTimeMax - logTimeBegin) * DRIFT_TABLE_LENGTH;
        else
            u1 = 0;
        i1 = (int) u1;
        if(i1 >= DRIFT_TABLE_LENGTH)
            i1 = DRIFT_TABLE_LENGTH - 1;
        
        if(i1 <= 1)
            df1 = u1 * DriftTable[0];
        else
            df1 = DriftTable[i1 - 1] + (DriftTable[i1] - DriftTable[i1 - 1]) * (u1 - i1);
        
        if(logTimeMax > logTimeBegin)
            u2 = (a2 - logTimeBegin) / (logTimeMax - logTimeBegin) * DRIFT_TABLE_LENGTH;
        else
            u2 = 0;
        i2 = (int) u2;
        if(i2 >= DRIFT_TABLE_LENGTH)
            i2 = DRIFT_TABLE_LENGTH - 1;
        
        if(i2 <= 1)
            df2 = u2 * DriftTable[0];
        else
            df2 = DriftTable[i2 - 1] + (DriftTable[i2] - DriftTable[i2 - 1]) * (u2 - i2);
        
        return (df2 - df1) * dilation;
    }
}


double get_gravkick_factor(integertime time0, integertime time1, int i, int mode)
{
  double a1, a2, df1, df2, u1, u2;
  int i1, i2;
  double dilation = mode ? return_node_timestep_dilation_factor(i) : timestep_dilation_factor(i, P);

    if(All.ComovingIntegrationOn == 0)
    {
        return (time1 - time0) * All.Timebase_interval * dilation;
    }
    else
    {

        /* note: will only be called for cosmological integration */
        
        a1 = logTimeBegin + time0 * All.Timebase_interval;
        a2 = logTimeBegin + time1 * All.Timebase_interval;
        
        if(logTimeMax > logTimeBegin)
            u1 = (a1 - logTimeBegin) / (logTimeMax - logTimeBegin) * DRIFT_TABLE_LENGTH;
        else
            u1 = 0;
        i1 = (int) u1;
        if(i1 >= DRIFT_TABLE_LENGTH)
            i1 = DRIFT_TABLE_LENGTH - 1;
        
        if(i1 <= 1)
            df1 = u1 * GravKickTable[0];
        else
            df1 = GravKickTable[i1 - 1] + (GravKickTable[i1] - GravKickTable[i1 - 1]) * (u1 - i1);
        
        if(logTimeMax > logTimeBegin)
            u2 = (a2 - logTimeBegin) / (logTimeMax - logTimeBegin) * DRIFT_TABLE_LENGTH;
        else
            u2 = 0;
        i2 = (int) u2;
        if(i2 >= DRIFT_TABLE_LENGTH)
            i2 = DRIFT_TABLE_LENGTH - 1;
        
        if(i2 <= 1)
            df2 = u2 * GravKickTable[0];
        else
            df2 = GravKickTable[i2 - 1] + (GravKickTable[i2] - GravKickTable[i2 - 1]) * (u2 - i2);
        
        return (df2 - df1) * dilation;
    }
}
