#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"

/*! \file timestep.c
 *  routines for assigning new timesteps
 */
/*!
 * This file was originally part of the GADGET3 code developed by
 * Volker Springel. The code has been modified
 * substantially by Phil Hopkins (phopkins@caltech.edu) for GIZMO; these
 * modifications include the addition of various timestep criteria, the WAKEUP
 * additions, and various changes of units and variable naming conventions throughout,
 * as well as timestep conditions for all physics and alternative solver options
 * and different timestep schemes entirely.
 */

static double dt_displacement = 0;


/*! This function advances the system in momentum space, i.e. it does apply the 'kick' operation after the
 *  forces have been computed. Additionally, it assigns new timesteps to particles. At start-up, a
 *  half-timestep is carried out, as well as at the end of the simulation. In between, the half-step kick that
 *  ends the previous timestep and the half-step kick for the new timestep are combined into one operation.
 */
void find_timesteps(void)
{
    CPU_Step[CPU_MISC] += measure_time();

    int i, bin, binold, prev, next;
    integertime ti_step, ti_step_old, ti_min, ti_stepmax, ti_max;
    double aphys;
#ifdef SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM
    int special_particle_active_with_this_index[SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM], j_specialpartical_counter=0;
    double xyz_local[SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM][3], xyz_global[SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM][3], special_particle_mass_local[SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM]={0}, special_particle_mass_global[SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM]={0};
    for(i=0;i<SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM;i++) {special_particle_active_with_this_index[i] = -1; xyz_local[i][0]=xyz_local[i][1]=xyz_local[i][2] = -MAX_REAL_NUMBER;}
#endif

    if(All.HighestActiveTimeBin == All.HighestOccupiedTimeBin || dt_displacement == 0)
        find_dt_displacement_constraint(All.cf_hubble_a * All.cf_atime * All.cf_atime);

#ifdef DIVBCLEANING_DEDNER
    /* need to calculate the global fastest wave speed to manage the damping terms stably */
    if((All.HighestActiveTimeBin == All.HighestOccupiedTimeBin)||(All.FastestWaveSpeed == 0))
    {
        double fastwavespeed = 0.0;
        double fastwavedecay = 0.0;
        double fac_magnetic_pressure = 1. / All.cf_atime;
        for(i=0;i<NumPart;i++)
        {
            if(P[i].Type==0)
            {
                double vsig2 = 0.5  * fabs(CellP[i].MaxSignalVel); // in v_phys units //
                double vsig1 = sqrt( Get_Gas_effective_soundspeed_i(i)*Get_Gas_effective_soundspeed_i(i) + fac_magnetic_pressure * (Get_Gas_BField(i,0)*Get_Gas_BField(i,0)+Get_Gas_BField(i,1)*Get_Gas_BField(i,1)+Get_Gas_BField(i,2)*Get_Gas_BField(i,2)) / CellP[i].Density );
                double vsig0 = DMAX(vsig1,vsig2);

                if(vsig0 > fastwavespeed) fastwavespeed = vsig0; // physical unit
                double hsig0 = Get_Particle_Size(i) * All.cf_atime; // physical unit
                if(vsig0/hsig0 > fastwavedecay) fastwavedecay = vsig0 / hsig0; // physical unit
            }
        }
        /* if desired, can just do this by domain; otherwise we use an MPI call over all domains to collect */
        double fastwavespeed_max_glob=fastwavespeed;
        MPI_Allreduce(&fastwavespeed, &fastwavespeed_max_glob, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        double fastwavedecay_max_glob=fastwavedecay;
        MPI_Allreduce(&fastwavedecay, &fastwavedecay_max_glob, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        /* now set the variables */
        All.FastestWaveSpeed = fastwavespeed_max_glob;
        All.FastestWaveDecay = fastwavedecay_max_glob;
    }
#endif

#if defined(FORCE_EQUAL_TIMESTEPS) || defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM)
    ti_max = 0;
    for(i = FirstActiveParticle, ti_min = TIMEBASE; i >= 0; i = NextActiveParticle[i])
    {
#if defined(FORCE_EQUAL_TIMESTEPS)
        ti_step = get_timestep(i, &aphys, 0);
#elif defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM)
        if(is_particle_a_special_zoom_target(i)==0) {ti_step = P[i].dt_step;} else {ti_step = TIMEBASE;} // set the source particle to have a timestep no more than 4 bins larger than the previous smallest active particle/cell bin timestep
#endif
        if(ti_step < ti_min) {ti_min = ti_step;}
        if(ti_step > ti_max) {ti_max = ti_step;}
    }
    if(ti_min > (dt_displacement / All.Timebase_interval)) {ti_min = (dt_displacement / All.Timebase_interval);}

    ti_step = TIMEBASE;
    while(ti_step > ti_min) {ti_step >>= 1;}
    ti_stepmax = TIMEBASE;
    while(ti_stepmax > ti_max) {ti_stepmax >>= 1;}
    integertime ti_min_glob, ti_max_glob;
    MPI_Allreduce(&ti_step, &ti_min_glob, 1, MPI_TYPE_TIME, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&ti_stepmax, &ti_max_glob, 1, MPI_TYPE_TIME, MPI_MAX, MPI_COMM_WORLD);
#if defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM)
#if defined(USE_TIMESTEP_DILATION_FOR_ZOOMS)
    ti_min_glob <<= 2; // 2^N times min timestep - shift to N bins higher
#else
    ti_min_glob <<= 4; // 2^N times min timestep - shift to N bins higher
#endif
    if(ti_min_glob > ti_max_glob) {ti_min_glob = ti_max_glob;}
#endif
#endif


    /* Now assign new timesteps  */
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#ifdef FORCE_EQUAL_TIMESTEPS
        ti_step = ti_min_glob;
#else
        ti_step = get_timestep(i, &aphys, 0);
#endif
        ti_step = (integertime)(((double)ti_step) / TIMESTEP_DILATION_FACTOR(i,0));
        
#if defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM)
        if(ti_step < 0) {ti_step = ti_min_glob;}
        if(is_particle_a_special_zoom_target(i)) {
            if(ti_step > ti_min_glob) {ti_step = ti_min_glob;}
            if(ti_step > ti_max_glob) {ti_step = ti_max_glob;}
        }
        //if(ti_min_glob > 0) {if(is_particle_a_special_zoom_target(i)) {while(ti_step > ti_min_glob) {ti_step >>= 1;}}} // set this per the above loop to minimum threshold relative to previous steps
#endif
        /* make it a power 2 subdivision */
        ti_min = TIMEBASE;
        while(ti_min > ti_step) {ti_min >>= 1;}
        ti_step = ti_min;
        bin = get_timestep_bin(ti_step);
        binold = P[i].TimeBin;
        if(bin > binold)		/* timestep wants to increase */
        {
            while(TimeBinActive[bin] == 0 && bin > binold) {bin--;}	/* make sure the new step is synchronized */
            ti_step = GET_INTEGERTIME_FROM_TIMEBIN(bin);
        }
        if(All.Ti_Current >= TIMEBASE) {ti_step = 0; bin = 0;} /* we here finish the last timestep. */

        if((TIMEBASE - All.Ti_Current) < ti_step)	/* check that we don't run beyond the end */
        {
            terminate("we are beyond the end of the timeline");	/* should not happen */
            ti_step = TIMEBASE - All.Ti_Current;
            ti_min = TIMEBASE;
            while(ti_min > ti_step) {ti_min >>= 1;}
            ti_step = ti_min;
        }

        if(bin != binold)
        {
            TimeBinCount[binold]--;
            if(P[i].Type == 0)
            {
                TimeBinCountGas[binold]--;
#ifdef GALSF
                TimeBinSfr[binold] -= CellP[i].Sfr;
                TimeBinSfr[bin] += CellP[i].Sfr;
#endif
            }

#ifdef SINK_PARTICLES
            if(P[i].Type == 5)
            {
                TimeBin_Sink_mass[binold] -= P[i].Sink_Mass;
                TimeBin_Sink_dynamicalmass[binold] -= P[i].Mass;
                TimeBin_Sink_Mdot[binold] -= P[i].Sink_Mdot;
                if(P[i].Sink_Mass > 0) {TimeBin_Sink_Medd[binold] -= P[i].Sink_Mdot / P[i].Sink_Mass;}
                TimeBin_Sink_mass[bin] += P[i].Sink_Mass;
                TimeBin_Sink_dynamicalmass[bin] += P[i].Mass;
                TimeBin_Sink_Mdot[bin] += P[i].Sink_Mdot;
                if(P[i].Sink_Mass > 0) {TimeBin_Sink_Medd[bin] += P[i].Sink_Mdot / P[i].Sink_Mass;}
            }
#endif
            prev = PrevInTimeBin[i];
            next = NextInTimeBin[i];

            if(FirstInTimeBin[binold] == i) {FirstInTimeBin[binold] = next;}
            if(LastInTimeBin[binold] == i) {LastInTimeBin[binold] = prev;}
            if(prev >= 0) {NextInTimeBin[prev] = next;}
            if(next >= 0) {PrevInTimeBin[next] = prev;}

            if(TimeBinCount[bin] > 0)
            {
                PrevInTimeBin[i] = LastInTimeBin[bin];
                NextInTimeBin[LastInTimeBin[bin]] = i;
                NextInTimeBin[i] = -1;
                LastInTimeBin[bin] = i;
            }
            else
            {
                FirstInTimeBin[bin] = LastInTimeBin[bin] = i;
                PrevInTimeBin[i] = NextInTimeBin[i] = -1;
            }
            TimeBinCount[bin]++;
            if(P[i].Type == 0) {TimeBinCountGas[bin]++;}
            P[i].TimeBin = bin;
        }

#ifndef WAKEUP
        ti_step_old = GET_INTEGERTIME_FROM_TIMEBIN(binold);
#else
        ti_step_old = P[i].dt_step;
#endif
        P[i].Ti_begstep += ti_step_old;
#if defined(WAKEUP)
        P[i].dt_step = ti_step;
#endif
#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
        if(P[i].Type == 5){
            if(All.Ti_Current == 0) { // first timestep
                P[i].dt_since_last_gas_search = GET_PHYSICAL_TIMESTEP_FROM_TIMEBIN(P[i].TimeBin,i);
                P[i].do_gas_search_this_timestep = 1;
            } else {
                P[i].dt_since_last_gas_search += GET_PHYSICAL_TIMESTEP_FROM_TIMEBIN(P[i].TimeBin,i);
                if(P[i].dt_since_last_gas_search > 0.49 * GET_PHYSICAL_TIMESTEP_FROM_TIMEBIN(P[i].Sink_TimeBinGasNeighbor,i)){
                    P[i].do_gas_search_this_timestep = 1;
                } else {P[i].do_gas_search_this_timestep = 0;}
            }
#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
	    if(P[i].ProtoStellarStage == 6) {P[i].do_gas_search_this_timestep = 1;} // always do gas search if we're rapidly spawning in new gas shells
#endif
        }
#endif
        
#ifdef SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM
        if(is_particle_a_special_zoom_target(i) && P[i].Mass > 0) {xyz_local[j_specialpartical_counter][0]=P[i].Pos[0]; xyz_local[j_specialpartical_counter][1]=P[i].Pos[1]; xyz_local[j_specialpartical_counter][2]=P[i].Pos[2]; special_particle_active_with_this_index[j_specialpartical_counter]=i; special_particle_mass_local[j_specialpartical_counter]=P[i].Mass; j_specialpartical_counter++;} // active on this processor, set
#endif
        
    }

#ifdef SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM
    MPI_Allreduce(xyz_local, xyz_global, 3*SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); // broadcast the new position of the special particle
    double mass_to_sum_local[SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM],  mass_to_sum_global[SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM]={0}; // define mass variables for passing
    int k; for(k=0;k<SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM;k++) {mass_to_sum_local[k] = All.Mass_Accreted_By_SpecialParticle[k];}
    MPI_Allreduce(mass_to_sum_local, mass_to_sum_global, SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // broadcast the mass update of the special particle
    for(k=0;k<SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM;k++)
    {
        if(xyz_global[k][0] > -1.e10) { // this indicates that the special particle was active on one task
            All.SpecialParticle_Position_ForRefinement[k][0] = xyz_global[k][0]; All.SpecialParticle_Position_ForRefinement[k][1] = xyz_global[k][1]; All.SpecialParticle_Position_ForRefinement[k][2] = xyz_global[k][2]; // variable was updated, update global variable as needed
            if(special_particle_active_with_this_index[k]>=0) {P[special_particle_active_with_this_index[k]].Mass += mass_to_sum_global[k]; special_particle_mass_local[k] += mass_to_sum_global[k];} // the special particle lives here with this id, so we can update it with this mass
            All.Mass_Accreted_By_SpecialParticle[k] = 0; // reset this variable on all processors because we have added it now to the special particle, to conserve mass properly
        }
    }
    MPI_Allreduce(special_particle_mass_local, special_particle_mass_global, SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); // broadcast the mass of the special particle
    for(k=0;k<SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM;k++) {if(special_particle_mass_global[k] > 0) {All.Mass_of_SpecialParticle[k] = special_particle_mass_global[k];}} // update the mass of the special particle for everyone to use
    // ???
#endif


#ifdef PMGRID
    if(All.PM_Ti_endstep == All.Ti_Current)	/* need to do long-range kick */
    {
        ti_step = TIMEBASE;
        while(ti_step > (dt_displacement / All.Timebase_interval)) {ti_step >>= 1;}
        if(ti_step > (All.PM_Ti_endstep - All.PM_Ti_begstep))	/* PM-timestep wants to increase */
        {
            bin = get_timestep_bin(ti_step);
            binold = get_timestep_bin(All.PM_Ti_endstep - All.PM_Ti_begstep);
            while(TimeBinActive[bin] == 0 && bin > binold) {bin--;}	/* make sure the new step is synchronized */
            ti_step = GET_INTEGERTIME_FROM_TIMEBIN(bin);
        }
        if(All.Ti_Current == TIMEBASE) {ti_step = 0;} /* we here finish the last timestep. */
        All.PM_Ti_begstep = All.PM_Ti_endstep;
        All.PM_Ti_endstep = All.PM_Ti_begstep + ti_step;
    }
#endif

#ifdef WAKEUP
    process_wake_ups();
#endif

    CPU_Step[CPU_TIMELINE] += measure_time();
}



/*! This function normally (for flag==0) returns the maximum allowed timestep of a particle, expressed in
 *  terms of the integer mapping that is used to represent the total simulated timespan. The physical
 *  acceleration is returned in aphys. The latter is used in conjunction with the PSEUDOSYMMETRIC integration
 *  option, which also makes of the second function of get_timestep. When it is called with a finite timestep
 *  for flag, it returns the physical acceleration that would lead to this timestep, assuming timestep
 *  criterion 0.
 */
integertime get_timestep(int p,		/*!< particle index */
                         double *aphys,	/*!< acceleration (physical units) */
                         int flag	/*!< either 0 for normal operation, or finite timestep to get corresponding aphys */ )
{
    double ax, ay, az, ac, csnd = 0, dt = All.MaxSizeTimestep, dt_courant = 0, dt_divv = 0;
    integertime ti_step; int k; k=0;

#ifdef IO_GRADUAL_SNAPSHOT_RESTART // if on the first timestep of a snapshot restart, start at the lowest allowed timestep to minimize any transient effects
    if(RestartFlag == 2 && All.Ti_Current == 0) {return 2;}
#endif
#if (SINGLE_STAR_TIMESTEPPING > 0)
    P[p].SuperTimestepFlag = 0;
    if( (P[p].Type == 5) && P[p].is_in_a_binary ) // candidate: need to decide whether to use super timestepping for binaries
    {
#if (SINGLE_STAR_TIMESTEPPING == 1) // to be conservative, use the semimajor axis, ie. the internal timescale is the orbital period
	    double dt_bin = P[p].Min_Sink_OrbitalTime / (2.*M_PI); // sqrt(a^3/GM) for binary
	    if(0.03*P[p].COM_dt_tidal>dt_bin) {P[p].SuperTimestepFlag=2;
	    } // external timestep is appropriately larger than 'internal' timestep, so use super-timestepping routine
#else // to be more aggressive, use the instantaneous orbital timescale, ie. freefall time from the CURRENT orbital separation. This lets us super step an orbit on the close passages, even when it is affected by tides at apopase
	    double dr = sqrt(P[p].comp_dx[0]*P[p].comp_dx[0] + P[p].comp_dx[1]*P[p].comp_dx[1] + P[p].comp_dx[2]*P[p].comp_dx[2]);
	    double dt_bin = sqrt(dr*dr*dr / (All.G * (P[p].Mass + P[p].comp_Mass)));
        if(0.005*P[p].COM_dt_tidal>dt_bin) {P[p].SuperTimestepFlag=2;} // external timestep is appropriately larger than 'internal' timestep, so use super-timestepping routine [constant here stricter for more aggressive routine]
#endif
    }
#endif

    
    
#if defined(SPECIAL_POINT_MOTION)
    {int k;
#ifdef SPECIAL_POINT_WEIGHTED_MOTION
        if(P[p].Type != SPECIAL_POINT_TYPE_FOR_NODE_DISTANCES)
#endif
        for(k=0;k<3;k++) {
            double acc = All.cf_a2inv * P[p].GravAccel[k];
#ifdef PMGRID
            acc += All.cf_a2inv * P[p].GravPM[k];
#endif
            if(P[p].Type==0) {
                acc += CellP[p].HydroAccel[k];
#ifdef TURB_DRIVING
                acc += CellP[p].TurbAccel[k];
#endif
#ifdef RT_RAD_PRESSURE_OUTPUT
                acc += CellP[p].Rad_Accel[k];
#endif
            }
            P[p].Acc_Total_PrevStep[k] = acc;
        }
    }
#endif

    
    if(flag == 0)
    {
        ax = All.cf_a2inv * P[p].GravAccel[0];
        ay = All.cf_a2inv * P[p].GravAccel[1];
        az = All.cf_a2inv * P[p].GravAccel[2];
#ifdef PMGRID
        ax += All.cf_a2inv * P[p].GravPM[0];
        ay += All.cf_a2inv * P[p].GravPM[1];
        az += All.cf_a2inv * P[p].GravPM[2];
#endif
        
#if defined(TIDAL_TIMESTEP_CRITERION)
#if defined(RT_USE_GRAVTREE) && !defined(SINGLE_STAR_FB_RT_HEATING)
        if(P[p].Type>0) // strictly this is better for accuracy, but not necessary
#endif
        ax = ay = az = 0.0; // we're getting our gravitational timestep criterion from the tidal tensor, but still want to do the accel criterion for other forces
#endif

        if(P[p].Type == 0)
        {
            ax += CellP[p].HydroAccel[0];
            ay += CellP[p].HydroAccel[1];
            az += CellP[p].HydroAccel[2];
#ifdef TURB_DRIVING
            ax += CellP[p].TurbAccel[0];
            ay += CellP[p].TurbAccel[1];
            az += CellP[p].TurbAccel[2];
#endif
#ifdef RT_RAD_PRESSURE_OUTPUT
            ax += CellP[p].Rad_Accel[0];
            ay += CellP[p].Rad_Accel[1];
            az += CellP[p].Rad_Accel[2];
#endif
        }

        ac = sqrt(ax * ax + ay * ay + az * az);	/* this is now the physical acceleration */
        *aphys = ac;
    }
    else
    {ac = *aphys;}

    if(ac == 0) {ac = 1.0e-30;}


    if(flag > 0)
    {
        /* this is the non-standard mode; use timestep to get the maximum acceleration tolerated */
        dt = flag * UNIT_INTEGERTIME_IN_PHYSICAL(p); /* convert dloga to physical timestep  */
        ac = 2 * All.ErrTolIntAccuracy * All.cf_atime * KERNEL_CORE_SIZE * ForceSoftening_KernelRadius(p) / (dt * dt);
        *aphys = ac;
        return flag;
    }
    dt = sqrt(2 * All.ErrTolIntAccuracy * All.cf_atime * KERNEL_CORE_SIZE * ForceSoftening_KernelRadius(p) / ac);

#if (defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL)) && defined(GALSF) && defined(GALSF_FB_MECHANICAL)
    if(((P[p].Type == 4)||((All.ComovingIntegrationOn==0)&&((P[p].Type == 2)||(P[p].Type==3))))&&(P[p].Mass>0))
    {
        if((All.ComovingIntegrationOn)) // sort of a hack here, but acceptable in applications
        {
            double h_min = All.ForceSoftening[P[p].Type], ags_h = DMIN(DMAX(P[p].KernelRadius, h_min), 10.*h_min);
#ifdef ADAPTIVE_GRAVSOFT_FORALL
            ags_h = DMIN(DMAX(P[p].AGS_KernelRadius , DMAX(P[p].KernelRadius, h_min)) , DMAX(100.*h_min, 10.*P[p].AGS_KernelRadius));
#endif
            dt = sqrt(2 * All.ErrTolIntAccuracy * All.cf_atime  * KERNEL_CORE_SIZE * ags_h / ac);
        }
    }
#endif


#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    double tidal_mag = 0.; {int k,j; for(k=0;k<3;k++) {for(j=0;j<3;j++) {tidal_mag += P[p].tidal_tensorps[k][j]*P[p].tidal_tensorps[k][j];}}} // get the frobenius norm
    tidal_mag = sqrt(tidal_mag); // can estimate time derivative here, via: dt_ttmag = (tidal_mag-P[p].tidal_tensor_mag_prev) / GET_PARTICLE_TIMESTEP_IN_PHYSICAL(p); 
    double dt_tidalsoft = All.CourantFac * NUMDIMS * DMAX(DMAX(GET_PARTICLE_TIMESTEP_IN_PHYSICAL(p), dt), All.MinSizeTimestep) * (tidal_mag+P[p].tidal_tensor_mag_prev) / (fabs(tidal_mag-P[p].tidal_tensor_mag_prev) + MIN_REAL_NUMBER);
    if(((1 << P[p].Type) & (ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION)) && (P[p].tidal_tensor_mag_prev>0 && All.Time>All.TimeBegin)) {dt = DMIN(dt, dt_tidalsoft);} // use as a timestep criterion for tidal-ags-active particles
    P[p].tidal_tensor_mag_prev = tidal_mag; // save it (overwriting previous value)
    {
        int j,k; double tt2=0,tracett=0; for(j=0;j<3;j++) {for(k=0;k<3;k++) {tt2+=P[p].tidal_tensorps[j][k]*P[p].tidal_tensorps[j][k]; if(k==j) {tracett+=P[p].tidal_tensorps[j][k];}}} /* compute numbers needed below */
        double H_eff = ForceSoftening_KernelRadius(p); /* get value to calculate H we need to use in the equations below */
        if(tidal_mag > 0) {P[p].tidal_zeta *= -All.G*(H_eff-All.ForceSoftening[P[p].Type])/(2.*NUMDIMS*tt2 - 0 * 8.*M_PI*tracett*(All.G*P[p].Mass/pow(H_eff,NUMDIMS)));} else {P[p].tidal_zeta = 0;}
        for(j=0;j<3;j++) {for(k=0;k<3;k++) {P[p].tidal_tensorps_prevstep[j][k]=P[p].tidal_zeta*P[p].tidal_tensorps[j][k];}} /* save for next iteration in gravtree */
    }
#endif


#ifdef TIDAL_TIMESTEP_CRITERION // tidal criterion obtains the same energy error in an optimally-softened Plummer sphere over ~100 crossing times as the Power 2003 criterion
    double tidal_mag_dt = 0.; {int k,j; for(k=0;k<3;k++) {for(j=0;j<3;j++) {tidal_mag_dt += P[p].tidal_tensorps[k][j]*P[p].tidal_tensorps[k][j];}}} // get the frobenius norm
    double dt_tidal = sqrt(All.ErrTolIntAccuracy / (All.cf_a3inv * sqrt(tidal_mag_dt / 6))); // recovers sqrt(eta) * tdyn for a Keplerian potential
    if(P[p].Type == 0) {dt_tidal = DMIN(sqrt(All.ErrTolIntAccuracy/(All.G*CellP[p].Density*All.cf_a3inv)), dt_tidal);} // gas self-gravity timescale as a bare minimum
    if(All.ComovingIntegrationOn){ // floor to the dynamical time of the universe
        double rho0 = (H0_CGS*H0_CGS*(3./(8.*M_PI*GRAVITY_G_CGS))*All.cf_a3inv / UNIT_DENSITY_IN_CGS);
        dt_tidal = DMIN(dt_tidal, sqrt(All.ErrTolIntAccuracy / (All.G * rho0)));
    } 
#ifdef ADAPTIVE_TREEFORCE_UPDATE
    P[p].tdyn_step_for_treeforce = dt_tidal; // hang onto this to decide how frequently to update the treeforce
#endif
    
#if (SINGLE_STAR_TIMESTEPPING > 0)
    if(P[p].SuperTimestepFlag>=2) {dt_tidal = sqrt(2*All.ErrTolIntAccuracy) * P[p].COM_dt_tidal;}
#endif
    dt=DMIN(dt,dt_tidal);
#endif

#ifdef SINGLE_STAR_TIMESTEPPING // this ensures that binaries advance in lock-step, which gives superior conservation
    if(P[p].Type == 5)
    {
        double dt_2body = sqrt(2*All.ErrTolIntAccuracy) * 0.3 / (1./P[p].Min_Sink_Approach_Time + 1./P[p].Min_Sink_Freefall_time); // timestep is harmonic mean of freefall and approach time
#ifdef HERMITE_INTEGRATION
        if(eligible_for_hermite(p)) dt_2body /= 0.3;
#endif
#if (SINGLE_STAR_TIMESTEPPING > 0)
    	if(P[p].is_in_a_binary && (P[p].SuperTimestepFlag >= 2)) //binary candidate or a confirmed binary
	    {    // First we need to construct the same 2-body timescale as above, but from the binary parameters. If this is longer than the above, there is another star that is requiring us to
	         // take a short timestep, so we better not super-timestep otherwise we risk messing up that star's integration. But if it is consistent with the above, then we can safely super-timestep
	        double Mtot=P[p].comp_Mass+P[p].Mass, dr=0,dv=0,dv_dot_dx=0, binary_dt_2body=0;
	        for(k=0;k<3;k++) {dr+=P[p].comp_dx[k]*P[p].comp_dx[k]; dv+=P[p].comp_dv[k]*P[p].comp_dv[k]; dv_dot_dx+=P[p].comp_dx[k]*P[p].comp_dv[k];}
            double r_effective = KERNEL_FAC_FROM_FORCESOFT_TO_PLUMMER * ForceSoftening_KernelRadius(p); // plummer-equivalent softening
	        dr += r_effective*r_effective; // add in quadrature for simple softening estimate
            dr=sqrt(dr); if(dv>0) {dv=sqrt(dv);} else {dv=0;}
            double dt_2body_base = 1/(1./P[p].Min_Sink_Approach_Time + 1./P[p].Min_Sink_Freefall_time); // timestep is harmonic mean of freefall and approach time
	        binary_dt_2body = 1. / (dv / dr + sqrt(All.G * Mtot / (dr*dr*dr)));
	        if(fabs(binary_dt_2body - dt_2body_base)/dt_2body_base < 1e-2)
	        { // If consistent with the binary parameters, we choose a super-timestep that gives ~constant number of timesteps per orbit
                double SUPERTIMESTEPPING_NUM_STEPS_PER_ORBIT = 50;
                dt_2body = 2.*M_PI / SUPERTIMESTEPPING_NUM_STEPS_PER_ORBIT * (binary_dt_2body*2); // orbital frequency is |dr x dv| / r^2, so timestep will be inverse to this
	        } else {P[p].SuperTimestepFlag = 0;}  // we still have to take a proper short N-body integration timestep due to a third body whose approach requires careful integration, so no super timestepping is possible
	    }
#endif
        dt = DMIN(dt, dt_2body);
#ifdef HERMITE_INTEGRATION
        if(eligible_for_hermite(p)) dt *= 1.4; // gives 10^-6 energy error per orbit for a 0.9 eccentricity binary
#endif
    }
#if defined(SINGLE_STAR_FB_TIMESTEPLIMIT) && !defined(SELFGRAVITY_OFF)
    if(P[p].Type == 0) {dt = DMIN(dt, 0.5 * All.CourantFac * DMIN(P[p].Min_Sink_FeedbackTime, P[p].Min_Sink_Approach_Time));}
#endif    
#endif // SINGLE_STAR_TIMESTEPPING

#ifdef ADAPTIVE_GRAVSOFT_FORALL
    /* make sure smoothing length of non-gas particles doesn't change too much in one timestep */
    if(((1 << P[p].Type) & (ADAPTIVE_GRAVSOFT_FORALL)) && (P[p].Type > 0))
    {
        double dt_divv = 0.1 / (MIN_REAL_NUMBER + All.cf_a2inv*fabs(P[p].Particle_DivVel)); // with new integration accuracy in gravtree, we may not need to be super-conservative here. old code used pre-factor 0.25 here, see if we can get away with the larger value which is standard for gas below
        if(dt_divv < dt) {dt = dt_divv;}
        double dt_cour = 2. * All.CourantFac * (Get_Particle_Size_AGS(p)*All.cf_atime) / (MIN_REAL_NUMBER + 0.5*P[p].AGS_vsig); // can be generous here, really the signal velocity isn't that important in the collisionless case, but it is important with some of the physics above //
#if defined(CBE_INTEGRATOR)
        dt_cour *= 0.25; // need a much stricter criterion here, to account for fluxes de-stabilizing the method //
#endif
        if(dt_cour < dt) {dt = dt_cour;}
    }
#endif


#ifdef DM_FUZZY
    if((P[p].Type > 0) && (P[p].AGS_Density > 0))
    {
        /* fuzzy DM admits longitudinal waves with group velocity =(hbar/m_dm)*k, so need a courant criterion, but because of scaling with k (like diffusion), timestep is quadratic in resolution */
        double L_particle_ags_x = Get_Particle_Size_AGS(p) * All.cf_atime;
        double dt_cour_ags_fuzzy = 0.25 * (L_particle_ags_x*L_particle_ags_x) / All.ScalarField_hbar_over_mass; // wavespeed of resolve-able waves
        if(dt_cour_ags_fuzzy < dt) {dt = dt_cour_ags_fuzzy;}
        dt_cour_ags_fuzzy = 0.25 * L_particle_ags_x / sqrt(MIN_REAL_NUMBER + (10./9.)*P[p].AGS_Numerical_QuantumPotential/P[p].Mass); // wavespeed based on 'stored' sub-grid energy [can get comparable]
        if(dt_cour_ags_fuzzy < dt) {dt = dt_cour_ags_fuzzy;}
    }
#endif


#ifdef GRAIN_FLUID
    if((1 << P[p].Type) & (GRAIN_PTYPES))
    {
        csnd = convert_internalenergy_soundspeed2(p, P[p].Gas_InternalEnergy);
        int k; for(k=0;k<3;k++) {csnd += (P[p].Gas_Velocity[k]-P[p].Vel[k])*(P[p].Gas_Velocity[k]-P[p].Vel[k]);}
#if defined(GRAIN_LORENTZFORCE)
        for(k=0;k<3;k++) {csnd += P[p].Gas_B[k]*P[p].Gas_B[k] / (2.0 * P[p].Gas_Density);}
#endif
        csnd = sqrt(csnd);
        double L_particle = Get_Particle_Size(p);
        dt_courant = 0.5 * All.CourantFac * (L_particle*All.cf_atime) / csnd;
#if defined(GRAIN_BACKREACTION)
        if(6.*P[p].Grain_AccelTimeMin < dt_courant) {dt_courant = 6.*P[p].Grain_AccelTimeMin;}
#endif
#if defined(GRAIN_LORENTZFORCE) && defined(GRAIN_RDI_TESTPROBLEM)
        if(All.Grain_Charge_Parameter != 0) {double bmag=0; for(k=0;k<3;k++) {bmag += P[p].Gas_B[k]*P[p].Gas_B[k];}
            if(bmag>0) {double dt_gyro = 1. / ((All.Grain_Charge_Parameter*sqrt(1.)/((All.Grain_Internal_Density/UNIT_DENSITY_IN_CGS)*(All.Grain_Size_Max/UNIT_LENGTH_IN_CGS))) * DMIN(100.,pow(All.Grain_Size_Max/P[p].Grain_Size,2)) * sqrt(bmag)); if(dt_gyro>0 && dt_gyro<dt_courant) {dt_courant=dt_gyro;}}} /* this gives t_Lorentz in code units; sqrt[1] reflects expected unity mean density definition, hard-coded for rdi testproblem options here */
#endif
#ifdef PIC_MHD
        if(P[p].MHD_PIC_SubType>=3)
        {
            double lorentz_units = UNIT_B_IN_GAUSS * UNIT_VEL_IN_CGS * (ELECTRONCHARGE_CGS/(PROTONMASS_CGS*C_LIGHT_CGS)) / (UNIT_VEL_IN_CGS/UNIT_TIME_IN_CGS); // code velocity to CGS and B to Gauss, times base units e/(mp*c), then convert 'back' to code-units acceleration
            double reduced_C = PIC_SPEEDOFLIGHT_REDUCTION * C_LIGHT_CODE, charge_to_mass_ratio_dimensionless = All.PIC_Charge_to_Mass_Ratio;
#ifdef PIC_MHD_NEW_RSOL_METHOD
            lorentz_units *= PIC_SPEEDOFLIGHT_REDUCTION; // the rsol enters by slowing down the forces here, acts as a unit shift for time
#endif
            double beta2=0,B2=0; for(k=0;k<3;k++) {double b=P[p].Gas_B[k]*All.cf_a2inv, v=P[p].Vel[k]/(All.cf_atime*reduced_C); B2+=b*b; beta2+=v*v;} /* get magnitude and unit vector for B, and vector beta [-true- beta here] */
            double gamma_lorentz = 1./sqrt(DMAX(1.-DMAX(DMIN(beta2,1.),0.),MIN_REAL_NUMBER)); // calculate lorentz factor (with safety factors included to prevent accidental nan here //
            double dt_courant_pic = 0.5 / ((charge_to_mass_ratio_dimensionless/gamma_lorentz) * sqrt(B2) * lorentz_units); /* dt = 0.5/omega_gyro*/
            if(dt_courant_pic < dt_courant) dt_courant = dt_courant_pic;
        }
#endif
        if(dt_courant < dt) dt = dt_courant;
    }
#ifdef GRAIN_RDI_TESTPROBLEM_LIVE_RADIATION_INJECTION
    if(P[p].Type>-1) {double dt_inj = 0.1 * P[p].KernelRadius / C_LIGHT_CODE_REDUCED(p); if(P[p].Type==4) {dt_inj*=0.25;} if(dt_inj < dt) {dt = dt_inj;}}
#endif
#endif


    if((P[p].Type == 0) && (P[p].Mass > 0))
        {
            csnd = 0.5 * CellP[p].MaxSignalVel ;
            double L_particle = Get_Particle_Size(p);
            dt_courant = All.CourantFac * (L_particle*All.cf_atime) / csnd;
#if defined(SINK_WIND_SPAWN) && !defined(SINK_RIAF_SUBEDDINGTON_MODEL)
            if(P[p].ID == All.SpawnedWindCellID) {dt_courant *= 0.5;} // be more careful if this is a spawned-in gas cell
#endif
            if(dt_courant < dt) dt = dt_courant;

            double dt_prefac_diffusion;
            dt_prefac_diffusion = 0.5;
#if (defined(GALSF) || defined(DIFFUSION_OPTIMIZERS)) && !defined(MHD_NON_IDEAL)
            dt_prefac_diffusion = 1.8;
#endif
#ifdef SUPER_TIMESTEP_DIFFUSION
            double dt_superstep_explicit = 1.e10 * dt;
#endif



#ifdef CONDUCTION
            {
                double L_cond_inv = sqrt(CellP[p].Gradients.InternalEnergy[0]*CellP[p].Gradients.InternalEnergy[0] +
                                         CellP[p].Gradients.InternalEnergy[1]*CellP[p].Gradients.InternalEnergy[1] +
                                         CellP[p].Gradients.InternalEnergy[2]*CellP[p].Gradients.InternalEnergy[2]) / CellP[p].InternalEnergy;
                double L_cond = DMAX(L_particle , 1./(L_cond_inv + 1./L_particle)) * All.cf_atime;
                double dt_conduction = dt_prefac_diffusion * L_cond*L_cond / (MIN_REAL_NUMBER + CellP[p].Kappa_Conduction);
                // since we use CONDUCTIVITIES, not DIFFUSIVITIES, we need to add a power of density to get the right units //
                dt_conduction *= CellP[p].Density * All.cf_a3inv;
#ifdef SUPER_TIMESTEP_DIFFUSION
                if(dt_conduction < dt_superstep_explicit) dt_superstep_explicit = dt_conduction; // explicit time-step
                double dt_advective = dt_conduction * DMAX(1,DMAX(L_particle , 1/(MIN_REAL_NUMBER + L_cond_inv))*All.cf_atime / L_cond);
                if(dt_advective < dt) dt = dt_advective; // 'advective' timestep: needed to limit super-stepping
#else
                if(dt_conduction < dt) dt = dt_conduction; // normal explicit time-step
#endif
            }
#endif


#ifdef MHD_NON_IDEAL
            {
                double b_grad = 0, b_mag = 0;
                int k; for(k=0;k<3;k++)
                {
                    int k2;
                    for(k2=0;k2<3;k2++)
                    {
                        double tmp_grad = CellP[p].Gradients.B[k][k2];
                        b_grad += tmp_grad * tmp_grad;
                    }
                    double tmp_grad = Get_Gas_BField(p,k);
                    b_mag += tmp_grad * tmp_grad;
                }
                double L_cond_inv = MIN_REAL_NUMBER + sqrt(b_grad / (MIN_REAL_NUMBER + b_mag));
                double L_cond = DMAX(0.5*L_particle , DMIN(L_particle , 1./(L_cond_inv + 1./L_particle))) * All.cf_atime;
                L_cond = DMIN( L_particle , DMAX(1./L_cond_inv, 0.5*L_particle) ) * All.cf_atime; // more conservative estimator - may be needed sometimes to deal accurately with steep local gradients //
                double diff_coeff = fabs(CellP[p].Eta_MHD_OhmicResistivity_Coeff) + fabs(CellP[p].Eta_MHD_HallEffect_Coeff) + fabs(CellP[p].Eta_MHD_AmbiPolarDiffusion_Coeff);
                double dt_conduction = dt_prefac_diffusion * L_cond*L_cond / (MIN_REAL_NUMBER + diff_coeff);
#ifdef SUPER_TIMESTEP_DIFFUSION
                if(dt_conduction < dt_superstep_explicit) dt_superstep_explicit = dt_conduction; // explicit time-step
                double dt_advective = dt_conduction * DMAX(1,DMAX(L_particle , 1/(MIN_REAL_NUMBER + L_cond_inv))*All.cf_atime / L_cond);
                if(dt_advective < dt) dt = dt_advective; // 'advective' timestep: needed to limit super-stepping
#else
                if(dt_conduction < dt) dt = dt_conduction; // normal explicit time-step
#endif
            }
#endif


#ifdef COSMIC_RAY_FLUID
            int k_CRegy;
            for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++)
            {
                if(Get_Gas_CosmicRayPressure(p,k_CRegy) > 1.0e-20)
                {
                    int explicit_timestep_on, cr_diffusion_opt = 1;
                    double CRPressureGradScaleLength = Get_CosmicRayGradientLength(p,k_CRegy);
                    double L_cr_weak; L_cr_weak = CRPressureGradScaleLength;
                    double kappa_cr_eff = fabs(CellP[p].CosmicRayDiffusionCoeff[k_CRegy]);
                    kappa_cr_eff *= CosmicRayFluid_RSOL_Corrfac(k_CRegy); // account for RSOL factor as it actually appears in the flux eqn in code units with this RSOL form
                    double L_cr_strong = DMAX(L_particle*All.cf_atime , 1./(1./CRPressureGradScaleLength + 1./(L_particle*All.cf_atime)));
                    double coeff_inv = 0.67 * L_cr_strong * dt_prefac_diffusion / (1.e-33 + kappa_cr_eff * (GAMMA_COSMICRAY(k_CRegy)-1.));
                    double dt_conduction =  L_cr_strong * coeff_inv; /* true diffusion requires the stronger timestep criterion be applied */
                    explicit_timestep_on = 1;
#if (CRFLUID_DIFFUSION_MODEL < 0)
                    dt_conduction = L_cr_weak * coeff_inv; /* streaming allows weaker timestep criterion because it's really an advection equation */
                    explicit_timestep_on = 0;
#endif
#ifdef GALSF
                    /* for multi-physics problems, we will use a more aggressive timestep criterion
                     based on whether or not the cosmic ray physics are relevant for what we are modeling */
                    if((CellP[p].CosmicRayEnergy[k_CRegy]==0)||(CellP[p].DtCosmicRayEnergy[k_CRegy]==0))
                    {
                        dt_conduction = 10. * dt;
                    } else {
                        double delta_cr = dt_conduction*fabs(CellP[p].DtCosmicRayEnergy[k_CRegy]);
                        double dL_cr = CRPressureGradScaleLength / (L_particle*All.cf_atime);
                        double thres_dL = 2., thres_egy = 1.e-3;
                        if(cr_diffusion_opt==1) {thres_dL = 1.; thres_egy = 1.e-2;}
                        if((dL_cr > thres_dL) || (delta_cr < thres_egy*CellP[p].CosmicRayEnergy[k_CRegy]))
                        {
                            double dt_weak = DMIN(L_cr_weak*coeff_inv , (delta_cr + 1.e-4*CellP[p].CosmicRayEnergy[k_CRegy])/fabs(CellP[p].DtCosmicRayEnergy[k_CRegy]));
                            if((dL_cr > thres_dL+1.) && (delta_cr < 0.1*thres_egy*CellP[p].CosmicRayEnergy[k_CRegy])) {dt_conduction = dt_weak; explicit_timestep_on = 0;}
                        }
                    }
#endif
#ifdef SUPER_TIMESTEP_DIFFUSION
                    if(explicit_timestep_on==1)
                    {
                        if(dt_prefac_diffusion > 1) {dt_conduction *= 0.5;}
                        if(dt_conduction < dt_superstep_explicit) dt_superstep_explicit = dt_conduction; // explicit time-step
                        double dt_advective = dt_conduction * DMAX(1 , DMAX(L_cr_strong,L_cr_weak)/L_cr_strong);
                        if(dt_advective < dt) dt = dt_advective; // 'advective' timestep: needed to limit super-stepping
                    } else {
                        if(dt_conduction < dt) dt = dt_conduction; // this is an advective timestep and super-stepping doesn't apply
                    }
#else
                    double cr_m1_speed = CRFLUID_REDUCED_C_CODE(k_CRegy); // pull for use below
                    if(cr_diffusion_opt==1)
                    {
                        if(CellP[p].CosmicRayEnergy[k_CRegy] > 0)
                        {
                            double cr_speed = cr_m1_speed;
                            //double crv=0; int k; for(k=0;k<3;k++) {crv+=CellP[p].CosmicRayFlux[k_CRegy][k]*CellP[p].CosmicRayFlux[k_CRegy][k];} if(crv > 0) {crv = sqrt(crv) / CellP[p].CosmicRayEnergy[k_CRegy];}
                            cr_speed = DMAX( DMIN(cr_m1_speed , CellP[p].MaxSignalVel) , DMIN(cr_m1_speed , kappa_cr_eff/(Get_Particle_Size(p)*All.cf_atime))); // default to min of free-streaming/diffusion speed
                            double dt_courant_CR = 0.4 * (L_particle*All.cf_atime) / cr_speed;
                            dt_conduction = dt_courant_CR; // per TK, strictly enforce this timestep //
                        } else {dt_conduction=10.*dt;}
                    } else {
                        double dt_courant_CR = 0.4 * (L_particle*All.cf_atime) / cr_m1_speed;
                        dt_conduction = dt_courant_CR; // per TK, strictly enforce this timestep //
                    }
                    if(dt_conduction < dt) dt = dt_conduction; // normal explicit time-step
#endif
                }
            }
#endif


#if defined(RADTRANSFER)
            {
                double dt_rad = 1.e10 * dt; // make some ridiculously large number here
                    
                /* first check if we are using an explicit diffusion-type solver (FLD, OTVET). need to consider the standard diffusive timestep, which we calculate below */
#if (defined(RT_OTVET) || defined(RT_FLUXLIMITEDDIFFUSION)) && defined(RT_COMPGRAD_EDDINGTON_TENSOR) && !defined(RT_EVOLVE_FLUX) /* for explicit diffusion, we include the usual second-order diffusion timestep */
                int kf; for(kf=0;kf<N_RT_FREQ_BINS;kf++)
                {
#if defined(RT_SOLVER_EXPLICIT) // explicit solver -- need diffusion timestep //
                    double gradETmag=0; for(k=0;k<3;k++) {gradETmag += CellP[p].Gradients.Rad_E_gamma_ET[kf][k]*CellP[p].Gradients.Rad_E_gamma_ET[kf][k];}
                    double L_ETgrad_inv = sqrt(gradETmag) / (1.e-37 + CellP[p].Rad_E_gamma[kf] * CellP[p].Density/P[p].Mass);
                    double L_RT_diffusion = DMIN(L_particle , 1./(3.*L_ETgrad_inv)) * All.cf_atime;
                    double dt_rt_diffusion = dt_prefac_diffusion * L_RT_diffusion*L_RT_diffusion / (MIN_REAL_NUMBER + rt_diffusion_coefficient(p,kf));
                    double dt_advective = dt_rt_diffusion * DMAX(1,DMAX(L_particle , 1/(MIN_REAL_NUMBER + L_ETgrad_inv))*All.cf_atime / L_RT_diffusion);
                    double dt_rt_work = All.CourantFac * DMIN( L_RT_diffusion / csnd , L_particle*All.cf_atime / ((2./3.)*sqrt(CellP[p].Rad_E_gamma[kf]/P[p].Mass)) ); /* time-step related to radiation work, radiation soundspeed, relevant in strongly-coupled limit */
#ifdef RT_FLUXLIMITER /* if we are flux-limited, we can account for the flux limiter making the timestep advective */
                    if(dt_advective > dt_rt_diffusion) {dt_rt_diffusion *= 1. + (1.-CellP[p].Rad_Flux_Limiter[kf]) * DMAX(0,(dt_advective/dt_rt_diffusion-1.));}
                    dt_advective = All.CourantFac * 0.5 * (L_particle*All.cf_atime) / C_LIGHT_CODE_REDUCED(p);
                    dt_rt_diffusion = DMAX(dt_rt_diffusion, dt_advective);
                    dt_rt_work /= MIN_REAL_NUMBER + CellP[p].Rad_Flux_Limiter[kf];
                    if((CellP[p].Rad_Flux_Limiter[kf] <= 0)||(dt_rt_diffusion<=0)) {dt_rt_diffusion = 1.e9 * dt;}
#endif
                    if((CellP[p].Rad_E_gamma[kf] <= MIN_REAL_NUMBER) || (CellP[p].Rad_E_gamma_Pred[kf] <= MIN_REAL_NUMBER)) {dt_rt_diffusion = dt_advective;} /* if the radiation is totally negligible, just use an advective timestep instead */
#ifdef SUPER_TIMESTEP_DIFFUSION /* if super-timestepping, limit the -super- step with the advective step, since you can still get inaccuracies if this is not respected */
                    if(dt_rt_diffusion < dt_superstep_explicit) dt_superstep_explicit = dt_rt_diffusion; // explicit time-step
                    dt_advective = dt_rt_diffusion * DMAX(1,DMAX(L_particle , 1/(MIN_REAL_NUMBER + L_ETgrad_inv))*All.cf_atime / L_RT_diffusion);
                    if(dt_advective < dt_rad) dt_rad = dt_advective; // 'advective' timestep: needed to limit super-stepping
#else
                    if(dt_rt_diffusion < dt_rad) dt_rad = dt_rt_diffusion; // normal explicit time-step
                    if(dt_rt_work < dt_rad) {dt_rad = dt_rt_work;} // normal explicit time-step
#endif
#endif // explicit-solver check
#if defined(RT_RAD_PRESSURE_FORCES) // -regardless- of if using an explicit solver, here the acceleration isn't saved to Rad_Accel so we calculate that timestep constraint
                    double gradErad=0; for(k=0;k<3;k++) {gradErad+=CellP[p].Gradients.Rad_E_gamma_ET[kf][k]*CellP[p].Gradients.Rad_E_gamma_ET[kf][k];}
                    double radacc = return_flux_limiter(p,kf) * (sqrt(gradErad) / CellP[p].Density) / All.cf_atime; // radiation acceleration for a timestep criterion
                    if(gradErad > 0 && radacc > 0)
                    {
                        double dt_radacc = sqrt(2 * All.ErrTolIntAccuracy * All.cf_atime * KERNEL_CORE_SIZE * DMAX(ForceSoftening_KernelRadius(p), P[p].KernelRadius) / radacc);
                        if(dt_radacc < dt_rad) {dt_rad = dt_radacc;}
                    }
#endif
                } // end of loop over frequency bins
#endif // end of conditional to check if we're using FLD or OTVET with an explicit solver

                
                /* now consider the (simpler) CFL-type condition required for advective solvers like M1 or intensity/ray integrators */
#if defined(RT_M1) || defined(RT_LOCALRAYGRID)
                dt_courant = All.CourantFac * (L_particle*All.cf_atime) / C_LIGHT_CODE_REDUCED(p); /* courant-type criterion, using the reduced speed of light */
#if defined(SINGLE_STAR_STARFORGE_DEFAULTS)
                dt_courant = 0.4 * (L_particle*All.cf_atime) / C_LIGHT_CODE_REDUCED(p); /* hacked here for starforge, where mike's experimentation suggests we can get away with a slightly larger courant factor. remains experimental. courant-type criterion, using the reduced speed of light - here we hardcode the most aggressive possible Courant factor as an optimization */
#ifdef SINK_WIND_SPAWN
                if((CellP[p].MaxSignalVel > 0.5*C_LIGHT_CODE_REDUCED(p)) || (P[p].ID == All.SpawnedWindCellID && P[p].Type == 0)) {dt_courant *= 0.5}; // be more careful if this is a jet cell or there are transluminal velocities
#endif
#endif                
#if defined(GALSF) && !defined(SINGLE_STAR_SINK_DYNAMICS) && defined(GALSF_FB_FIRE_STELLAREVOLUTION) // custom hacks for FIRE-RT tests; can override CFL condition with diffusion timestep certain limits
                int kf; for(kf=0;kf<N_RT_FREQ_BINS;kf++)
                {
                    double dt_rt_diffusion = dt_prefac_diffusion * (L_particle*All.cf_atime)*(L_particle*All.cf_atime) / (MIN_REAL_NUMBER + rt_diffusion_coefficient(p,kf));
                    if((CellP[p].Rad_E_gamma[kf] <= MIN_REAL_NUMBER) || (CellP[p].Rad_E_gamma_Pred[kf] <= MIN_REAL_NUMBER) || (CellP[p].Rad_E_gamma[kf] < 1.e-5*P[p].Mass*CellP[p].InternalEnergy)) {dt_rt_diffusion = 1.e10 * dt;} /* ignore particles where the radiation energy density is negligible */
                    dt_rad = DMIN(dt_rad, dt_rt_diffusion);
                }
                if(All.ComovingIntegrationOn) {dt_courant = DMAX(dt_courant, DMIN(dt_rad, 1.e3*dt_courant));}
#endif
                if(dt_courant < dt_rad) {dt_rad = dt_courant;}
#endif // explicit advective-type solver check

                
                /* one more check - we can optionally limit the timestep for explicit chemical timesteps: implicit solve is fine locally, but gets propagation somewhat wrong if timesteps too large, as that depends on opacity, which depends on ionization step! */
#if defined(RT_CHEM_PHOTOION) && defined(RT_TIMESTEP_LIMIT_RECOMBINATION) /* make sure this doesn't overshoot the recombination time for the opacity to change for ionizing photons */
                double ne_cgs = (CellP[p].Density * All.cf_a3inv * UNIT_DENSITY_IN_NHCGS), dt_recombination = All.CourantFac * (3.3e12/ne_cgs) / UNIT_TIME_IN_CGS;
                double dt_change = 1.e10*dt; if((CellP[p].Rad_E_gamma[RT_FREQ_BIN_H0] > 0)&&(fabs(CellP[p].Dt_Rad_E_gamma[RT_FREQ_BIN_H0])>0)) {dt_change = CellP[p].Rad_E_gamma[RT_FREQ_BIN_H0] / fabs(CellP[p].Dt_Rad_E_gamma[RT_FREQ_BIN_H0]);}
                dt_recombination = DMIN(DMAX(dt_recombination,dt_change), DMAX(dt_courant,dt_rad));
                if(dt_recombination < dt_rad) {dt_rad = dt_recombination;}
#endif

                if(dt_rad < dt) dt = dt_rad; // set the actual radiation timestep!
            }
#endif // RADTRANSFER
            

#ifdef VISCOSITY
            {
                int kv1,kv2; double dv_mag=0,v_mag=1.0e-33;
                for(kv1=0;kv1<3;kv1++) {v_mag+=P[p].Vel[kv1]*P[p].Vel[kv1];}
                double dv_mag_all = 0.0;
                for(kv1=0;kv1<3;kv1++)
                {
                    double dvmag_tmp = 0;
                    for(kv2=0;kv2<3;kv2++) {dvmag_tmp+=CellP[p].Gradients.Velocity[kv1][kv2]*CellP[p].Gradients.Velocity[kv1][kv2];}
                    dv_mag += dvmag_tmp /DMAX(P[p].Vel[kv1]*P[p].Vel[kv1],0.01*v_mag);
                    dv_mag_all += dvmag_tmp;
                }
                dv_mag = sqrt(DMAX(dv_mag, dv_mag_all/v_mag));
                double L_visc = DMAX(L_particle , 1. / (dv_mag + 1./L_particle)) * All.cf_atime;
                double visc_coeff = sqrt(CellP[p].Eta_ShearViscosity*CellP[p].Eta_ShearViscosity + CellP[p].Zeta_BulkViscosity*CellP[p].Zeta_BulkViscosity);
                double dt_viscosity = 0.25 * L_visc*L_visc / (1.0e-33 + visc_coeff) * CellP[p].Density * All.cf_a3inv;
                // since we use VISCOSITIES, not DIFFUSIVITIES, we need to add a power of density to get the right units //
#ifdef SUPER_TIMESTEP_DIFFUSION
                if(dt_viscosity < dt_superstep_explicit) dt_superstep_explicit = dt_viscosity; // explicit time-step
                double dt_advective = dt_viscosity * DMAX(1,DMAX(L_particle , 1/(MIN_REAL_NUMBER + dv_mag))*All.cf_atime / L_visc);
                if(dt_advective < dt) dt = dt_advective; // 'advective' timestep: needed to limit super-stepping
#else
                if(dt_viscosity < dt) dt = dt_viscosity; // normal explicit time-step
#endif
            }
#endif

#if defined(GRAIN_BACKREACTION)
            if(6.*P[p].Grain_AccelTimeMin < dt) {dt = 6.*P[p].Grain_AccelTimeMin;}
#endif


#ifdef TURB_DIFFUSION
            {
#ifdef TURB_DIFF_METALS
                int k_species; double L_tdiff = L_particle * All.cf_atime; // don't use gradient b/c ill-defined pre-enrichment
                for(k_species=0;k_species<NUM_METAL_SPECIES;k_species++)
                {
                    double dt_tdiff = L_tdiff*L_tdiff / (1.0e-33 + CellP[p].TD_DiffCoeff); // here, we use DIFFUSIVITIES, so there is no extra density power in the equation //
                    if(dt_tdiff < dt) dt = dt_tdiff; // normal explicit time-step
                }
#endif
            }
#endif


#if defined(DIVBCLEANING_DEDNER)
            double fac_magnetic_pressure = 1. / All.cf_atime;
            double phi_b_units = Get_Gas_PhiField(p) / ( All.cf_atime * CellP[p].MaxSignalVel);
            double vsig1 =  sqrt( Get_Gas_effective_soundspeed_i(p)*Get_Gas_effective_soundspeed_i(p) +
                    fac_magnetic_pressure * (Get_Gas_BField(p,0)*Get_Gas_BField(p,0) +
                                             Get_Gas_BField(p,1)*Get_Gas_BField(p,1)+
                                             Get_Gas_BField(p,2)*Get_Gas_BField(p,2) +
                                             phi_b_units*phi_b_units) / CellP[p].Density );

            dt_courant = 0.8 * All.CourantFac * (All.cf_atime*L_particle) / vsig1; // 2.0 factor may be added (PFH) //
            if(dt_courant < dt) {dt = dt_courant;}
#endif

            /* make sure that the velocity divergence does not imply a too large change of density or kernel length in the step */
            double divVel = P[p].Particle_DivVel;
            if(divVel != 0)
            {
                dt_divv = 1.5 / fabs(All.cf_a2inv * divVel);
                if(dt_divv < dt) {dt = dt_divv;}
            }


#if defined(TURB_DRIVING) && !defined(TURB_DRIVING_UPDATE_FORCE_ON_TURBUPDATE)
                /* gas cannot step larger than major updates to turbulent driving routine */
                double dt_turb_driving = 1.9 * st_return_dt_between_updates();
                if (dt > dt_turb_driving) {dt = dt_turb_driving;}
#endif
            

#ifdef SUPER_TIMESTEP_DIFFUSION
            /* now use the timestep information above to limit the super-stepping timestep */
            {
                int N_substeps = 5; /*!< number of sub-steps per super-timestep for super-timestepping algorithm */
                double nu_substeps = 0.04; /*!< damping parameter (0<nu<1), optimal behavior around ~1/sqrt[N_substeps] */

                /*!< pre-calculate the multipliers needed for the super-timestep sub-step */
                if(CellP[p].Super_Timestep_j == 0) {CellP[p].Super_Timestep_Dt_Explicit = dt_superstep_explicit;} // reset dt_explicit //
                double j_p_super = (double)(CellP[p].Super_Timestep_j + 1);
                double dt_superstep = CellP[p].Super_Timestep_Dt_Explicit / ((nu_substeps+1) + (nu_substeps-1) * cos(M_PI * (2*j_p_super - 1) / (2*(double)N_substeps)));

                double dt_touse = dt_superstep;
                if((dt <= dt_superstep)||(CellP[p].Super_Timestep_j > 0))
                {
                    /* if(dt <= dt_superstep): other constraints beat our super-step, so it doesn't matter: iterate */
                    /* if(CellP[p].Super_Timestep_j > 0): don't break mid-cycle, so iterate */
                    CellP[p].Super_Timestep_j++; if(CellP[p].Super_Timestep_j>=N_substeps) {CellP[p].Super_Timestep_j=0;} /*!< increment substep 'j' and loop if it cycles fully */
                } else {
                    /* ok, j=0 and dt > dt_superstep [the next super-step matters for starting a new cycle]: think about whether to start */
                    double dt_pred = dt_superstep * All.cf_hubble_a;
                    if(dt_pred > All.MaxSizeTimestep) {dt_pred = All.MaxSizeTimestep;}
                    if(dt_pred < All.MinSizeTimestep) {dt_pred = All.MinSizeTimestep;}
                    /* convert our physical timestep into the dimensionless units of the code */
                    integertime ti_min=TIMEBASE, ti_step = (integertime) (dt_pred / All.Timebase_interval);
                    /* check against valid limits */
                    if(ti_step<=1) {ti_step=2;}
                    if(ti_step>=TIMEBASE) {ti_step=TIMEBASE-1;}
                    while(ti_min > ti_step) {ti_min >>= 1;}  /* make it a power 2 subdivision */
                    ti_step = ti_min;
                    /* now turn it into a timebin */
                    int bin = get_timestep_bin(ti_step);
                    int binold = P[p].TimeBin;
                    if(bin > binold)  /* timestep wants to increase: check whether it wants to move into a valid timebin */
                    {
                        while(TimeBinActive[bin] == 0 && bin > binold) {bin--;} /* make sure the new step is synchronized */
                    }
                    /* now convert this -back- to a physical timestep */
                    double dt_allowed = GET_INTEGERTIME_FROM_TIMEBIN(bin) * UNIT_INTEGERTIME_IN_PHYSICAL(-1);
                    if(dt_superstep > 1.5*dt_allowed)
                    {
                        /* the next allowed timestep [because of synchronization] is not big enough to fit the 'big step'
                            part of the super-stepping cycle. rather than 'waste' our timestep which will knock us into a
                            lower bin and defeat the super-stepping, we simply take the -safe- explicit timestep and
                            wait until the desired time-bin synchs up, so we can super-step */
                        dt_touse = dt_superstep_explicit; // use the safe [normal explicit] timestep and -do not- cycle j //
                    } else {
                        /* ok, we can jump up in bins to use our super-step; begin the cycle! */
                        CellP[p].Super_Timestep_j++; if(CellP[p].Super_Timestep_j>=N_substeps) {CellP[p].Super_Timestep_j=0;}
                    }
                }
                if(dt < dt_touse) {dt = dt_touse;} // set the actual timestep [now that we've appropriately checked everything above] //
            }
#endif

        } // closes if(P[p].Type == 0) [gas particle check] //


#if defined(DM_SIDM)
    /* Reduce time-step if this particle got interaction probabilities > 0.2 during the last time-step */
    if((1 << P[p].Type) & (DM_SIDM))
    {
        if(P[p].dtime_sidm > 0) {if(P[p].dtime_sidm < dt) {dt = P[p].dtime_sidm;}}
        if(dt > 0)
        {
            double p_target = 0.2; // desired maximum probability per timestep
            double dV[3]; for(k=0;k<3;k++) {dV[k]=P[p].AGS_vsig*All.cf_atime/sqrt(3.);} // convert signal vel to velocity dispersion for estimating rates
#ifdef GRAIN_COLLISIONS
            double p_dt = prob_of_grain_interaction(return_grain_cross_section_per_unit_mass(p),P[p].Mass,0.,P[p].AGS_KernelRadius,dV,dt,p); // probability of interacting with another grain super-particle well within kernel, assuming same mass, H, and V~signalvel, for current timestep dt
#else
            double p_dt = prob_of_interaction(P[p].Mass,0.,P[p].AGS_KernelRadius,dV,dt); // probability of interacting with another DM particle well within kernel, assuming same mass, H, and V~signalvel, for current timestep dt
#endif
            if(p_dt > p_target) {dt *= p_target / p_dt;}
        }
    }
#endif


    // add a 'stellar evolution timescale' criterion to the timestep, to prevent too-large jumps in feedback //
#if defined(GALSF_FB_FIRE_RT_HIIHEATING) || defined(GALSF_FB_MECHANICAL) || defined(GALSF_FB_FIRE_RT_LONGRANGE) || (defined(GALSF) && defined(RADTRANSFER))
    if(((P[p].Type == 4)||((All.ComovingIntegrationOn==0)&&((P[p].Type == 2)||(P[p].Type==3))))&&(P[p].Mass>0))
    {
        double star_age = evaluate_stellar_age_Gyr(p);
        double dt_stellar_evol;
        dt_stellar_evol = DMAX(2.0e-4, star_age/250.); // restrict to small steps for young stars //
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
#if defined(SNE_NONSINK_SPAWN)
        double mcorr = 1.e-4 * (P[p].Mass*UNIT_MASS_IN_SOLAR) / 0.1; // expectation of ()/X SNe per timestep -- here 0.1
        if(star_age > 0.044) {mcorr *= 0.02;} // into Ia regime, lower SNR means we can substantially relax this mass-dependent criterion
        if(mcorr > 1) {dt_stellar_evol /= DMIN(mcorr, 100.);} // don't use - ok to have multiple at low-res, but don't want too-big a jump or miss key stellar evolution
#endif
/* // below not necessary with newer code, can be safely skipped for optimization //
        double mcorr = 1.e-4 * (P[p].Mass*UNIT_MASS_IN_SOLAR) / 0.1; // expectation of ()/X SNe per timestep -- here 0.1
        if(star_age > 0.044) {mcorr *= 0.02;} // into Ia regime, lower SNR means we can substantially relax this mass-dependent criterion
        if(mcorr > 1) {dt_stellar_evol /= DMIN(mcorr, 10.);} // don't use - ok to have multiple at low-res, but don't want too-big a jump or miss key stellar evolution
*/
#else
        double mcorr = 1.e-5 * (P[p].Mass*UNIT_MASS_IN_SOLAR);
        if(mcorr < 1 && mcorr > 0) {dt_stellar_evol /= mcorr;}
#endif
        if(dt_stellar_evol < 1.e-6) {dt_stellar_evol = 1.e-6;}
        dt_stellar_evol /= (UNIT_TIME_IN_GYR); // convert to code units //
        if(dt_stellar_evol>0) {if(dt_stellar_evol<dt) {dt = dt_stellar_evol;}}
    }
#endif


#ifdef SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM
    if(is_particle_a_special_zoom_target(p))
    {
        double dt_special_max = 1000./UNIT_TIME_IN_YR; // set a maximum physical timestep to prevent this centering from jumping
        if(dt > dt_special_max) {dt = dt_special_max;}
    }
#endif
    
    
#ifdef SINK_PARTICLES
    if(P[p].Type == 5)
    {
#if !defined(SINGLE_STAR_SINK_DYNAMICS) && defined(GALSF)
      double dt_accr = 4.2e5 / UNIT_TIME_IN_YR; // this is the 1% of Salpeter timescale; not relevant for low radiative efficiency
#else
      double dt_accr = All.MaxSizeTimestep;
#endif
        if(P[p].Sink_Mdot > 0 && P[p].Sink_Mass > 0 && All.Time > All.TimeBegin)
        {
#if (defined(SINK_GRAVCAPTURE_GAS) || defined(SINK_WIND_KICK)) && !defined(SINGLE_STAR_SINK_DYNAMICS)
            /* really want prefactor to be ratio of median gas mass to sink mass */
            dt_accr = 0.001 * DMAX(P[p].Sink_Mass, All.MaxMassForParticleSplit) / P[p].Sink_Mdot;
#if defined(SINK_WIND_KICK)
            dt_accr *= DMAX(0.1, All.Sink_accreted_fraction);
#endif
#else
            dt_accr = 0.001 * DMIN(P[p].Sink_Mass, All.MaxMassForParticleSplit) / P[p].Sink_Mdot;
#endif
#ifdef SINGLE_STAR_FB_JETS	    
            dt_accr = DMIN(dt_accr, target_mass_for_wind_spawning(p) / P[p].Sink_Mdot); 
#endif
        } // if(P[p].Sink_Mdot > 0 && P[p].Sink_Mass > 0)
#if defined(SINK_SEED_GROWTH_TESTS) || defined(FIRE_BHS)
        double dt_evol = 4.2e5 / UNIT_TIME_IN_YR; // totally arbitrary hard-coding here //
#ifdef TURB_DRIVING
        if(dt_evol > 1.e-3*st_return_mode_correlation_time()) {dt_evol=1.e-3*st_return_mode_correlation_time();}
#endif
        if(dt_accr > dt_evol) {dt_accr=dt_evol;}
#endif
        if(dt_accr > 0 && dt_accr < dt) {dt = dt_accr;}

        double dt_ngbs = 4.1 * GET_PHYSICAL_TIMESTEP_FROM_TIMEBIN(P[p].Sink_TimeBinGasNeighbor,p); /* standard wakeup-type threshold: use this by default here, unless dynamical interaction important (e.g. back-rx term from oscillation of sink c-o-m, which is important for single-sink sims */
        if(dt > dt_ngbs && dt_ngbs > 0) {dt = 1.01 * dt_ngbs; }

#if defined(SINGLE_STAR_TIMESTEPPING)
	    if(P[p].DensityAroundParticle > 0)
	    {
            double eps = DMAX( KERNEL_CORE_SIZE*ForceSoftening_KernelRadius(p), P[p].Sink_dr_to_NearestGasNeighbor);
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
            eps = DMAX(eps, P[p].SinkRadius);
#endif
            if(eps < MAX_REAL_NUMBER) {eps = DMAX(Get_Particle_Size(p), eps);} else {eps = Get_Particle_Size(p);}
#if (ADAPTIVE_GRAVSOFT_FORALL & 32)
            eps = DMAX(eps, KERNEL_CORE_SIZE*P[p].AGS_KernelRadius);
#endif
            double dt_ff = sqrt(2*All.ErrTolIntAccuracy * pow(eps*All.cf_atime,3) / (All.G * P[p].Mass)); // fraction of the freefall time of the nearest gas particle from rest
            if(dt > dt_ff && dt_ff > 0) {dt = 1.01 * dt_ff;}

            double L_particle = Get_Particle_Size(p);
            double vsig = P[p].Sink_SurroundingGasVel;
#if defined(SINGLE_STAR_FB_TIMESTEPLIMIT) && !defined(NOGRAVITY)
            vsig += P[p].MaxFeedbackVel;
#endif                        
            double dt_cour_sink = All.CourantFac * (L_particle*All.cf_atime) / vsig;
            if(dt > dt_cour_sink && dt_cour_sink > 0 && isfinite(dt_cour_sink)) {dt = 1.01 * dt_cour_sink;}
        }
        if(P[p].StellarAge == All.Time)
        {   // want a brand new sink to be on the lowest occupied timebin
            long bin; for(bin = 0; bin < TIMEBINS; bin++) {if(TimeBinCount[bin] > 0) break;}
            double dt_min =  GET_PHYSICAL_TIMESTEP_FROM_TIMEBIN(bin,p);
            if(dt > dt_min && dt_min > 0) dt = 1.01 * dt_min;
        }
#endif // SINGLE_STAR_TIMESTEPPING
#ifdef SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION
#ifdef SINGLE_STAR_FB_WINDS
        if(P[p].ProtoStellarStage == 5) {
            double mdot_spawn = single_star_wind_mdot(p,1);
            if(mdot_spawn > 0) {
                double dm_spawn = target_mass_for_wind_spawning(p), dt_spawn = dm_spawn / mdot_spawn;
                if(dt > dt_spawn && dt_spawn > 0) {dt = 1.01 * dt_spawn;}
            }}
#endif
#ifdef SINGLE_STAR_FB_SNE
        if ( (P[p].ProtoStellarStage == 6) && ( (P[p].Sink_Mass > 0) || (P[p].unspawned_wind_mass > 0) ) ) { //Star going supernova, still has mass to eject
            double eps = DMIN(KERNEL_CORE_SIZE*ForceSoftening_KernelRadius(p), P[p].KernelRadius);
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
            eps = DMAX(eps, P[p].SinkRadius);
#endif
            double t_clear=eps/single_star_SN_velocity(p);
            if(t_clear > 0 && dt > 0) {dt=DMIN(dt, DMAX(0.5*t_clear, 1.01*All.MinSizeTimestep));}; // time needed spawned wind particles to clear the sink so that we don't spawn on top of them (leading to progressively smaller timesteps from each spawn until crashing the code)
        }
#endif
#endif
    } // if(P[p].Type == 5)

#if defined(SINK_WIND_SPAWN_SET_BFIELD_POLTOR) /* KYSu: here for de-bugging jet injection model right now */
    if((P[p].Type==5) || (P[p].Type==0 && P[p].ID==All.SpawnedWindCellID && CellP[p].IniDen<0)) {if(dt>All.Sink_spawn_injectionradius/All.Sink_outflow_velocity && All.Sink_spawn_injectionradius>0 && All.Sink_outflow_velocity>0) {dt=All.Sink_spawn_injectionradius/All.Sink_outflow_velocity;}}
#endif
#endif // SINK_PARTICLES
    



    /* convert the physical timestep to dloga if needed. Note: If comoving integration has not been selected, All.cf_hubble_a=1. */
    dt *= All.cf_hubble_a;

#ifdef ONLY_PM
    dt = All.MaxSizeTimestep;
#endif

    if(dt >= All.MaxSizeTimestep) {dt = All.MaxSizeTimestep;}

    if(dt >= dt_displacement) {dt = dt_displacement;}

    if((dt < All.MinSizeTimestep)||(((integertime) (dt / All.Timebase_interval)) <= 1))
    {
        PRINT_WARNING("Timestep wants to be below the limit `MinSizeTimestep'");
        double agrav_pm=0, agrav = sqrt(P[p].GravAccel[0]*P[p].GravAccel[0] + P[p].GravAccel[1]*P[p].GravAccel[1] + P[p].GravAccel[2]*P[p].GravAccel[2]) * All.cf_a2inv;
#ifdef PMGRID
        agrav_pm = sqrt(P[p].GravPM[0]*P[p].GravPM[0] + P[p].GravPM[1]*P[p].GravPM[1] + P[p].GravPM[2]*P[p].GravPM[2]) * All.cf_a2inv;
#endif
        if(P[p].Type == 0)
        {
            double aturb=0, arad=0, ahydro = sqrt(CellP[p].HydroAccel[0]*CellP[p].HydroAccel[0] + CellP[p].HydroAccel[1]*CellP[p].HydroAccel[1] + CellP[p].HydroAccel[2]*CellP[p].HydroAccel[2]);
#ifdef TURB_DRIVING
            aturb = sqrt(CellP[p].TurbAccel[0]*CellP[p].TurbAccel[0] + CellP[p].TurbAccel[1]*CellP[p].TurbAccel[1] + CellP[p].TurbAccel[2]*CellP[p].TurbAccel[2]);
#endif
#ifdef RT_RAD_PRESSURE_OUTPUT
            arad = sqrt(CellP[p].Rad_Accel[0]*CellP[p].Rad_Accel[0] + CellP[p].Rad_Accel[1]*CellP[p].Rad_Accel[1] + CellP[p].Rad_Accel[2]*CellP[p].Rad_Accel[2]);
#endif
            PRINT_WARNING("\n Cell-ID=%llu  dt_desired=%g dt_Courant=%g dt_Accel=%g\n accel_tot=%g accel_gravTree=%g accel_gravPM=%g accel_hydro=%g accel_rad=%g accel_turb=%g Pos_xyz=(%g|%g|%g) Vel_xyz=(%g|%g|%g)\n KernelRadius=%g Density=%g InternalEnergy=%g dtInternalEnergy=%g divV=%g Pressure=%g Cs_Eff=%g vAlfven=%g f_ion=%g\n csnd_for_signalspeed=%g eps_forcesoftening=%g mass=%g type=%d condition_number=%g Nngb=%g\n NVT=%.17g/%.17g/%.17g %.17g/%.17g/%.17g %.17g/%.17g/%.17g\n",
                          (unsigned long long) P[p].ID, dt, dt_courant*All.cf_hubble_a, sqrt(2*All.ErrTolIntAccuracy*All.cf_atime*ForceSoftening_KernelRadius(p) / ac)*All.cf_hubble_a,
                          ac, agrav, agrav_pm, ahydro, arad, aturb, P[p].Pos[0], P[p].Pos[1], P[p].Pos[2], P[p].Vel[0]/All.cf_atime, P[p].Vel[1]/All.cf_atime, P[p].Vel[2]/All.cf_atime,
                          P[p].KernelRadius*All.cf_atime, CellP[p].Density*All.cf_a3inv, CellP[p].InternalEnergy, CellP[p].DtInternalEnergy, P[p].Particle_DivVel*All.cf_a2inv,
                          CellP[p].Pressure*All.cf_a3inv, Get_Gas_effective_soundspeed_i(p), Get_Gas_Alfven_speed_i(p), Get_Gas_Ionized_Fraction(p),
                          csnd, ForceSoftening_KernelRadius(p)*All.cf_atime, P[p].Mass, P[p].Type, CellP[p].ConditionNumber, P[p].NumNgb,
                          CellP[p].NV_T[0][0],CellP[p].NV_T[0][1],CellP[p].NV_T[0][2],CellP[p].NV_T[1][0],CellP[p].NV_T[1][1],CellP[p].NV_T[1][2],CellP[p].NV_T[2][0],CellP[p].NV_T[2][1],CellP[p].NV_T[2][2]);
        }
        else // if(P[p].Type == 0)
        {
            PRINT_WARNING("Part-ID=%llu  dt_desired=%g dt_Accel=%g\n accel_tot=%g accel_gravTree=%g accel_gravPM=%g  mass=%g pos_xyz=(%g|%g|%g) vel_xyz=(%g|%g|%g) soft=%g type=%d\n",
                          (unsigned long long) P[p].ID, dt, sqrt(2*All.ErrTolIntAccuracy*All.cf_atime*ForceSoftening_KernelRadius(p) / ac)*All.cf_hubble_a,
                          ac, agrav, agrav_pm, P[p].Mass, P[p].Pos[0], P[p].Pos[1], P[p].Pos[2], P[p].Vel[0], P[p].Vel[1], P[p].Vel[2], ForceSoftening_KernelRadius(p), P[p].Type);
        }
        fflush(stdout); fprintf(stderr, "\n @ fflush \n");
#ifdef STOP_WHEN_BELOW_MINTIMESTEP
        if(P[p].Mass > 0) {endrun(888);}
#endif
        dt = All.MinSizeTimestep;
    }

    ti_step = (integertime) (dt / All.Timebase_interval);
#ifndef STOP_WHEN_BELOW_MINTIMESTEP
    if(ti_step<=1) ti_step=2;
#endif

    if(!(ti_step > 0 && ti_step < TIMEBASE))
    {
        printf("\nError: A timestep of size zero was assigned on the integer timeline. Code must stop.\n"
               "Task=%d Part-ID=%llu dt=%g dtc=%g dtv=%g dtdis=%g tibase=%g ti_step=%lld ac=%g xyz=(%g|%g|%g) tree=(%g|%g|%g)\n\n",
               ThisTask, (unsigned long long) P[p].ID, dt, dt_courant, dt_divv, dt_displacement,
               All.Timebase_interval, (long long) ti_step, ac, P[p].Pos[0], P[p].Pos[1], P[p].Pos[2], P[p].GravAccel[0], P[p].GravAccel[1], P[p].GravAccel[2]);
#ifdef PMGRID
        printf("pm_force=(%g|%g|%g)\n", P[p].GravPM[0], P[p].GravPM[1], P[p].GravPM[2]);
#endif
        fflush(stdout); endrun(818);
    }

    return ti_step;
}


/*! This function computes an upper limit ('dt_displacement') to the global timestep of the system based on
 *  the rms velocities of particles. For cosmological simulations, the criterion used is that the rms
 *  displacement should be at most a fraction MaxRMSDisplacementFac of the mean particle separation. Note that
 *  the latter is estimated using the assigned particle masses, separately for each particle type. If comoving
 *  integration is not used, the function imposes no constraint on the timestep.
 */
void find_dt_displacement_constraint(double hfac /*!<  should be  a^2*H(a)  */ )
{
    int i, type;
    int count[6];
    long long count_sum[6];
    double v[6], v_sum[6], mim[6], mnm[6], min_mass[6], mean_mass[6];
    double dt, dmean, asmth = 0;

    dt_displacement = All.MaxSizeTimestep;

    if(All.ComovingIntegrationOn)
    {
        for(type = 0; type < 6; type++)
        {
            count[type] = 0;
            v[type] = 0;
            mim[type] = 1.0e30;
            mnm[type] = 0;
        }

        for(i = 0; i < NumPart; i++)
        {
            if(P[i].Mass > 0)
            {
                double v2 = P[i].Vel[0] * P[i].Vel[0] + P[i].Vel[1] * P[i].Vel[1] + P[i].Vel[2] * P[i].Vel[2];
                if(v2 > 0 && isfinite(v2)) {
                    count[P[i].Type]++;
                    if(P[i].Type == 0) {v[P[i].Type] += P[i].Mass * v2;} else {v[P[i].Type] += v2;} /* for gas use a weighted average to deal with extreme cell-mass difference situations */
                    if(mim[P[i].Type] > P[i].Mass) {mim[P[i].Type] = P[i].Mass;}
                    mnm[P[i].Type] += P[i].Mass;
                }
            }
        }

        MPI_Allreduce(v, v_sum, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(mim, min_mass, 6, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(mnm, mean_mass, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        sumup_large_ints(6, count, count_sum);
        if(mean_mass[0] > 0) {v_sum[0] /= mean_mass[0];} /* for gas use a weighted average to deal with extreme cell-mass difference situations */

#ifdef GALSF
        /* add star and gas particles together to treat them on equal footing, using the original gas particle spacing. */
        double vsum0_0 = v_sum[0], minmass0_0 = min_mass[0], meanmass0_0 = mean_mass[0]; long long countsum0_0=count_sum[0];
        v_sum[0] += v_sum[4]; count_sum[0] += count_sum[4];
        if(count_sum[0] > 0) {
            if(count_sum[4]<=1 || (v_sum[0]+v_sum[4])*count_sum[4] < v_sum[4]*(count_sum[0]+count_sum[4])) {
                v_sum[4] += vsum0_0; count_sum[4] += countsum0_0; mean_mass[4] += meanmass0_0; min_mass[4] = DMAX(DMAX(min_mass[4],minmass0_0),mean_mass[4]/count_sum[4]);}}
        //v_sum[4] = v_sum[0]; count_sum[4] = count_sum[0]; if(count_sum[0] > 0) {min_mass[0] = min_mass[4] = (mean_mass[0] + mean_mass[4]) / count_sum[0];}
#ifdef SINK_PARTICLES
        vsum0_0 = v_sum[0]; minmass0_0 = min_mass[0]; meanmass0_0 = mean_mass[0]; countsum0_0=count_sum[0];
        v_sum[0] += v_sum[5]; count_sum[0] += count_sum[5];
        if(count_sum[0] > 0) {
            if(count_sum[5]<=1 || (v_sum[0]+v_sum[5])*count_sum[5] < v_sum[5]*(count_sum[0]+count_sum[5])) {
                v_sum[5] += vsum0_0; count_sum[5] += countsum0_0; mean_mass[5] += meanmass0_0; min_mass[5] = DMAX(DMAX(min_mass[5],minmass0_0),mean_mass[5]/count_sum[5]);}}
        //v_sum[5] = v_sum[0]; count_sum[5] = count_sum[0]; min_mass[5] = min_mass[0];
#endif
#ifdef SPECIAL_POINT_MOTION
        v_sum[SPECIAL_POINT_TYPE_FOR_NODE_DISTANCES] = v_sum[0];
        count_sum[SPECIAL_POINT_TYPE_FOR_NODE_DISTANCES] = count_sum[0];
        min_mass[SPECIAL_POINT_TYPE_FOR_NODE_DISTANCES] = min_mass[0];
#endif
#endif

        if(ThisTask == 0) {printf("Global displacement time constraint computation: \n");}
        for(type = 0; type < 6; type++)
        {
            if(count_sum[type] > 0 && v_sum[type] > 0)
            {
#ifdef GALSF
                if(type == 0 || type == 4)
#else
                if(type == 0)
#endif
                    dmean = pow(min_mass[type] / (All.OmegaBaryon * 3 * All.Hubble_H0_CodeUnits * All.Hubble_H0_CodeUnits / (8 * M_PI * All.G)), 1.0 / 3);
                else
                    dmean = pow(min_mass[type] / ((All.OmegaMatter - All.OmegaBaryon) * 3 * All.Hubble_H0_CodeUnits * All.Hubble_H0_CodeUnits / (8 * M_PI * All.G)), 1.0 / 3);

#ifdef SINK_PARTICLES
                if(type == 5) {dmean = pow(min_mass[type] / (All.OmegaBaryon * 3 * All.Hubble_H0_CodeUnits * All.Hubble_H0_CodeUnits / (8 * M_PI * All.G)), 1.0 / 3);}
#endif
                dt = All.MaxRMSDisplacementFac * hfac * dmean / sqrt(v_sum[type] / count_sum[type]);

#ifdef PMGRID
                asmth = All.Asmth[0];
#ifdef PM_PLACEHIGHRESREGION
                if(((1 << type) & (PM_PLACEHIGHRESREGION))) {asmth = All.Asmth[1];}
#endif
                if(asmth < dmean) {dt = All.MaxRMSDisplacementFac * hfac * asmth / sqrt(v_sum[type] / count_sum[type]);}
#endif
                if(ThisTask == 0) {printf(" ..type=%d  dmean=%g asmth=%g minmass=%g a=%g  sqrt(<p^2>)=%g  dlogmax=%g\n",type, dmean, asmth, min_mass[type], All.Time, sqrt(v_sum[type] / count_sum[type]), dt);}
                if(dt < dt_displacement && dt > 0) {dt_displacement = dt;}
            }
        }

        if(ThisTask == 0) {printf(" ..global displacement time constraint: %g  (All.MaxSizeTimestep=%g)\n", dt_displacement, All.MaxSizeTimestep);}
    }
}



int get_timestep_bin(integertime ti_step)
{
    int bin = -1;

    if(ti_step == 0)
        return 0;

    if(ti_step == 1)
        terminate("time-step of integer size 1 not allowed\n");

    while(ti_step)
    {
        bin++;
        ti_step >>= 1;
    }

    return bin;
}





#ifdef WAKEUP
void process_wake_ups(void)
{
    int i, n, max_time_bin_active, bin, binold, prev, next; long long ntot;
    integertime dt_bin, ti_next_for_bin, ti_next_kick, ti_next_kick_global;

    /* find the next kick time */
    for(n = 0, ti_next_kick = TIMEBASE; n < TIMEBINS; n++)
    {
        if(TimeBinCount[n])
        {
            if(n > 0)
            {
                dt_bin = GET_INTEGERTIME_FROM_TIMEBIN(n);
                ti_next_for_bin = (All.Ti_Current / dt_bin) * dt_bin + dt_bin;	/* next kick time for this timebin */
            }
            else {dt_bin = 0; ti_next_for_bin = All.Ti_Current;}
            if(ti_next_for_bin < ti_next_kick) {ti_next_kick = ti_next_for_bin;}
        }
    }

    MPI_Allreduce(&ti_next_kick, &ti_next_kick_global, 1, MPI_TYPE_TIME, MPI_MIN, MPI_COMM_WORLD);

    PRINT_STATUS("Predicting next timestep: %g", (ti_next_kick_global - All.Ti_Current) * All.Timebase_interval);
    max_time_bin_active = 0;
    /* get the highest bin, that is active next time */
    for(n = 0; n < TIMEBINS; n++)
    {
        dt_bin = (((integertime) 1) << n);
        if((ti_next_kick_global % dt_bin) == 0) {max_time_bin_active = n;}
    }

    /* move the particle into the highest bin, that is active in the next timestep and that is lower than its last timebin */
    bin = 0; for(n = 0; n < TIMEBINS; n++) {if(TimeBinCount[n] > 0) {bin = n; break;}}
    n = 0;

    MPI_Allreduce(&NeedToWakeupParticles_local, &NeedToWakeupParticles, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD); // if one process processes wakeups then they all should, just in case a woke particle gets swapped to another process before we get here

    if(NeedToWakeupParticles){
	for(i = 0; i < NumPart; i++)
	{
	    if(!P[i].wakeup) {continue;}
#if !defined(AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE)
	    if(P[i].Type != 0) {continue;} // only gas particles can be awakened
#endif
	    if(P[i].Mass <= 0) {continue;}
	    binold = P[i].TimeBin;
	    if(TimeBinActive[binold]) {continue;}

	    bin = max_time_bin_active < binold ? max_time_bin_active : binold;

	    if(bin != binold)
	    {
		integertime dt_0 = GET_INTEGERTIME_FROM_TIMEBIN(P[i].TimeBin);
		integertime tstart = P[i].Ti_begstep + dt_0;
		integertime t_2 = P[i].Ti_current;
		if(t_2 > tstart) {tstart = t_2;}
		integertime tend = All.Ti_Current;

		TimeBinCount[binold]--;
		if(P[i].Type == 0) {TimeBinCountGas[binold]--;}

		prev = PrevInTimeBin[i];
		next = NextInTimeBin[i];

		if(FirstInTimeBin[binold] == i) {FirstInTimeBin[binold] = next;}
		if(LastInTimeBin[binold] == i) {LastInTimeBin[binold] = prev;}
		if(prev >= 0) {NextInTimeBin[prev] = next;}
		if(next >= 0) {PrevInTimeBin[next] = prev;}

		if(TimeBinCount[bin] > 0)
		{
		    PrevInTimeBin[i] = LastInTimeBin[bin];
		    NextInTimeBin[LastInTimeBin[bin]] = i;
		    NextInTimeBin[i] = -1;
		    LastInTimeBin[bin] = i;
		}
		else
		{
		    FirstInTimeBin[bin] = LastInTimeBin[bin] = i;
		    PrevInTimeBin[i] = NextInTimeBin[i] = -1;
		}
		TimeBinCount[bin]++;
		if(P[i].Type == 0) {TimeBinCountGas[bin]++;}
		P[i].TimeBin = bin;
        if(TimeBinActive[bin]) {NumForceUpdate++;}
		n++;

		/* reverse part of the last second-half kick this particle received
		   (to correct it back to its new active time) */
		if(tend < tstart)
		{
		    do_the_kick(i, tstart, tend, P[i].Ti_current, 1);
		    set_predicted_quantities_for_extra_physics(i);
		}
		P[i].Ti_begstep = All.Ti_Current;
		P[i].dt_step = GET_INTEGERTIME_FROM_TIMEBIN(bin);
		if(P[i].Ti_current < All.Ti_Current) {P[i].Ti_current=All.Ti_Current;}
	    }
	}
    }

    sumup_large_ints(1, &n, &ntot);
    if(ThisTask == 0) {if(ntot > 0) {printf("%d%09d particles activated (in wakeup check).\n", (int) (ntot / 1000000000), (int) (ntot % 1000000000));}}
    NeedToWakeupParticles = 0;
    NeedToWakeupParticles_local = 0;
}
#endif




#ifdef BOX_SHEARING
void calc_shearing_box_pos_offset(void) /* function that calculates the shear-offset between the shear-periodic boundaries in a shearing box */
{
    Shearing_Box_Pos_Offset = Shearing_Box_Vel_Offset * All.Time;
    while(Shearing_Box_Pos_Offset > boxSize_Y) {Shearing_Box_Pos_Offset -= boxSize_Y;}
}
#endif
