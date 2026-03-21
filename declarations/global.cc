#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
#include "../core/proto.h"


/*! This routine computes various global properties of the particle
 * distribution, if desired for some options. It also holds some subroutines
 * (unlike the GADGET file of this name originally written by Volker Springel)
 * for different meta-options, like dynamic refinement centers, and
 * time-dilation schemes, etc., all of which are unique to GIZMO.
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

void compute_global_quantities_of_system(void)
{
  int i, j, n;
  struct state_of_system sys;
  double a1, a2, a3;
  double entr = 0, egyspec; Vec3<double> vel;
  double dt_entr, dt_gravkick, dt_hydrokick;

  if(All.ComovingIntegrationOn)
    {
      a1 = All.Time;
      a2 = All.Time * All.Time;
      a3 = All.Time * All.Time * All.Time;
    }
  else
    {
      a1 = a2 = a3 = 1;
    }


  for(n = 0; n < 6; n++)
    {
      sys.MassComp[n] = sys.EnergyKinComp[n] = sys.EnergyPotComp[n] = sys.EnergyIntComp[n] = 0;
      for(j = 0; j < 4; j++) {sys.CenterOfMassComp[n][j] = sys.MomentumComp[n][j] = sys.AngMomentumComp[n][j] = 0;}
    }

  for(i = 0; i < NumPart; i++)
    {
        sys.MassComp[P[i].Type] += P[i].Mass;
#if defined(EVALPOTENTIAL) || defined(COMPUTE_POTENTIAL_ENERGY)
        sys.EnergyPotComp[P[i].Type] += 0.5 * P[i].Mass * P[i].Potential / a1;
#endif
        integertime dt_integerstep = GET_PARTICLE_INTEGERTIME(i);
        dt_entr = dt_hydrokick = (All.Ti_Current - (P[i].Ti_begstep + dt_integerstep / 2)) * UNIT_INTEGERTIME_IN_PHYSICAL(i);
        dt_gravkick = get_gravkick_factor((P[i].Ti_begstep + dt_integerstep / 2), All.Ti_Current, i, 0);

        vel = P[i].Vel + P[i].GravAccel * dt_gravkick;
        if(P[i].Type == 0) {vel += CellP[i].HydroAccel * (dt_hydrokick * All.cf_atime);} /* convert from physical to code vel */
        if(P[i].Type == 0) {entr = DMAX(0.1*CellP[i].InternalEnergy, CellP[i].InternalEnergy + CellP[i].DtInternalEnergy * dt_entr);}

#ifdef PMGRID
        double dt_gravkick_pm = get_gravkick_factor((All.PM_Ti_begstep + All.PM_Ti_endstep) / 2, All.Ti_Current, -1, 0);
        vel += P[i].GravPM * dt_gravkick_pm;
#endif
        
        sys.EnergyKinComp[P[i].Type] += 0.5 * P[i].Mass * vel.norm_sq() / a2;
        if(P[i].Type == 0) {egyspec = entr; sys.EnergyIntComp[0] += P[i].Mass * egyspec;}
        
        for(j = 0; j < 3; j++)
        {
            sys.MomentumComp[P[i].Type][j] += P[i].Mass * vel[j];
            sys.CenterOfMassComp[P[i].Type][j] += P[i].Mass * P[i].Pos[j];
        }
        
        sys.AngMomentumComp[P[i].Type][0] += P[i].Mass * (P[i].Pos[1] * vel[2] - P[i].Pos[2] * vel[1]);
        sys.AngMomentumComp[P[i].Type][1] += P[i].Mass * (P[i].Pos[2] * vel[0] - P[i].Pos[0] * vel[2]);
        sys.AngMomentumComp[P[i].Type][2] += P[i].Mass * (P[i].Pos[0] * vel[1] - P[i].Pos[1] * vel[0]);
    }
  
  /* some the stuff over all processors */
  MPI_Reduce(&sys.MassComp[0], &SysState.MassComp[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&sys.EnergyPotComp[0], &SysState.EnergyPotComp[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&sys.EnergyIntComp[0], &SysState.EnergyIntComp[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&sys.EnergyKinComp[0], &SysState.EnergyKinComp[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&sys.MomentumComp[0][0], &SysState.MomentumComp[0][0], 6 * 4, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
  MPI_Reduce(&sys.AngMomentumComp[0][0], &SysState.AngMomentumComp[0][0], 6 * 4, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
  MPI_Reduce(&sys.CenterOfMassComp[0][0], &SysState.CenterOfMassComp[0][0], 6 * 4, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);


  if(ThisTask == 0)
    {
      for(i = 0; i < 6; i++) {SysState.EnergyTotComp[i] = SysState.EnergyKinComp[i] + SysState.EnergyPotComp[i] + SysState.EnergyIntComp[i];}
      SysState.Mass = SysState.EnergyKin = SysState.EnergyPot = SysState.EnergyInt = SysState.EnergyTot = 0;
      for(j = 0; j < 3; j++) {SysState.Momentum[j] = SysState.AngMomentum[j] = SysState.CenterOfMass[j] = 0;}

      for(i = 0; i < 6; i++)
	{
	  SysState.Mass += SysState.MassComp[i];
	  SysState.EnergyKin += SysState.EnergyKinComp[i];
	  SysState.EnergyPot += SysState.EnergyPotComp[i];
	  SysState.EnergyInt += SysState.EnergyIntComp[i];
	  SysState.EnergyTot += SysState.EnergyTotComp[i];

	  for(j = 0; j < 3; j++)
	    {
	      SysState.Momentum[j] += SysState.MomentumComp[i][j];
	      SysState.AngMomentum[j] += SysState.AngMomentumComp[i][j];
	      SysState.CenterOfMass[j] += SysState.CenterOfMassComp[i][j];
	    }
	}

      for(i = 0; i < 6; i++) {for(j = 0; j < 3; j++) {if(SysState.MassComp[i] > 0) {SysState.CenterOfMassComp[i][j] /= SysState.MassComp[i];}}}

      for(j = 0; j < 3; j++) {if(SysState.Mass > 0) {SysState.CenterOfMass[j] /= SysState.Mass;}}
	  

      for(i = 0; i < 6; i++)
	{
	  SysState.CenterOfMassComp[i][3] = SysState.MomentumComp[i][3] = SysState.AngMomentumComp[i][3] = 0;
	  for(j = 0; j < 3; j++)
	    {
	      SysState.CenterOfMassComp[i][3] += SysState.CenterOfMassComp[i][j] * SysState.CenterOfMassComp[i][j];
	      SysState.MomentumComp[i][3] += SysState.MomentumComp[i][j] * SysState.MomentumComp[i][j];
	      SysState.AngMomentumComp[i][3] += SysState.AngMomentumComp[i][j] * SysState.AngMomentumComp[i][j];
	    }
	  SysState.CenterOfMassComp[i][3] = sqrt(SysState.CenterOfMassComp[i][3]);
	  SysState.MomentumComp[i][3] = sqrt(SysState.MomentumComp[i][3]);
	  SysState.AngMomentumComp[i][3] = sqrt(SysState.AngMomentumComp[i][3]);
	}

      SysState.CenterOfMass[3] = SysState.Momentum[3] = SysState.AngMomentum[3] = 0;

      for(j = 0; j < 3; j++)
	{
	  SysState.CenterOfMass[3] += SysState.CenterOfMass[j] * SysState.CenterOfMass[j];
	  SysState.Momentum[3] += SysState.Momentum[j] * SysState.Momentum[j];
	  SysState.AngMomentum[3] += SysState.AngMomentum[j] * SysState.AngMomentum[j];
	}

      SysState.CenterOfMass[3] = sqrt(SysState.CenterOfMass[3]);
      SysState.Momentum[3] = sqrt(SysState.Momentum[3]);
      SysState.AngMomentum[3] = sqrt(SysState.AngMomentum[3]);
    }
  
  /* give everyone the result, maybe the want to do something with it */
  MPI_Bcast(&SysState, sizeof(struct state_of_system), MPI_BYTE, 0, MPI_COMM_WORLD);
}


/* Sigmoid ("turn-on") function (1 + x/(1+x^2))/2, interpolates between 0 as x->-infty and 1 as x->infty. Useful for cheaply doing smooth fits of e.g. EOS where different thermo processes turn on at certain temps */
double sigmoid_sqrt(double x)
{
    return 0.5*(1 + x/sqrt(1+x*x));
}

/* Returns the Frobenius norm of the velocity gradient in physical units */
double velocity_gradient_norm(int i)
{
    double dv2=0; int j,k; for(j=0;j<3;j++) {for(k=0;k<3;k++) {double vt = CellP[i].Gradients.Velocity[j][k]*All.cf_a2inv; /* physical velocity gradient */
        if(All.ComovingIntegrationOn) {if(j==k) {vt += All.cf_hubble_a;}} /* add hubble-flow correction */
        dv2 += vt*vt;}} // calculate magnitude of the velocity shear across cell from || grad -otimes- v ||^(1/2)
    return sqrt(dv2);
}


#if defined(FIRE_SUPERLAGRANGIAN_JEANS_REFINEMENT) || defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM)
int is_particle_a_special_zoom_target(int i)
{
#ifdef SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM
    if(P[i].Type == 3) {return 1;}
#endif
    return 0;
}
#endif



/* timestep dilation factor for computing quantities in zoom-in runs with variable extreme dynamic range */
double return_timestep_dilation_factor(int i, int mode)
{
#if !defined(USE_TIMESTEP_DILATION_FOR_ZOOMS)
    return 1;
#else
    
    if(All.Time <= All.TimeBegin) {return 1;}
    if(i < 0) {return 1;}
#ifdef DILATION_FOR_STELLAR_KINEMATICS_ONLY
    if(mode != 0) {return 1;}
#ifdef SPECIAL_POINT_WEIGHTED_MOTION
    if(P[i].Type != 4 && P[i].Type != SPECIAL_POINT_TYPE_FOR_NODE_DISTANCES) {return 1;} /* only do cosmological 'stars' type -and- the special smoothing-source-types */
#else
    if(P[i].Type != 4) {return 1;} /* only do cosmological 'stars' type */
#endif
#endif
    
    /* now specify some dilation factor a(r) or otherwise */
    double a = 1;
    
#ifdef SPECIAL_POINT_WEIGHTED_MOTION
    double r = P[i].Min_Distance_to_Sink;
    if(P[i].Type == SPECIAL_POINT_TYPE_FOR_NODE_DISTANCES) {r = 0;}
    double wt = weight_function_for_weighted_motion_smoothing(r, 0);
    if(wt > 0 && wt < 1) {a = 1. / wt;}
#endif

#if defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM)
    double fac_amax = 100.;
#ifdef SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_SPECIALBOUNDARIES
#if (SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_SPECIALBOUNDARIES >= 3)
    fac_amax = 1.e6;
#endif
#endif
    double amax = fac_amax;
    double r_amax = fac_amax * All.ForceSoftening[3]; // modify as needed
    double index = 1;
    int j, k; double rmin = MAX_REAL_NUMBER, r=0, a=1;
    for(j=0;j<SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM;j++)
    {
        double p0[3]={0}, dp[3]={0}, r2=0, pos_i[3];
        for(k=0;k<3;k++) {p0[k] = All.SpecialParticle_Position_ForRefinement[j][k];}
        if(mode==0) {for(k=0;k<3;k++) {pos_i[k]=P[i].Pos[k];}} /* the reference index refers to a real particle */
            else {for(k=0;k<3;k++) {pos_i[k]=Nodes[i].u.d.s[k];}} /* the reference index refers to a node or pseudo-particle */
        for(k=0;k<3;k++) {dp[k] = All.cf_atime*(pos_i[k] - p0[k]); r2 += dp[k]*dp[k];}
        r = sqrt(r2); if(r < rmin) {rmin = r;}
    }
    r = rmin;
#if (SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_SPECIALBOUNDARIES >= 3)
    if(mode==0) {r = sqrt(P[i].Pos[0]*P[i].Pos[0] + P[i].Pos[1]*P[i].Pos[1] + P[i].Pos[2]*P[i].Pos[2]);}
#endif
    if(r < 1.e-10 || isnan(r) || isfinite(r)==0) {r = 1.e-10;}
    a = 1. + 1. / (1./amax + pow(r / r_amax, index));
#endif
    
    return 1. / a;
#endif
}

