/* predict_functions.h — Canonical KOKKOS_INLINE_FUNCTION implementation of
 * evaluate_NH_from_GradRho.  Single source of truth for both CPU and GPU.
 *
 * Proto.h has an inline Vec3<MyFloat> wrapper that forwards to this.
 *
 * Include order: after allvars.h (for All, MyFloat). */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

KOKKOS_INLINE_FUNCTION
double evaluate_NH_from_GradRho(MyFloat gradrho[3], double rkern, double rho, double numngb_ndim, double include_h, int target, struct particle_data *pp)
{
    double gradrho_mag=0;
    if(rho>0)
    {
#ifdef RT_USE_TREECOL_FOR_NH
        gradrho_mag = include_h * rho * rkern / numngb_ndim; if(target>=0) {gradrho_mag += pp[target].SigmaEff;}
#else
        gradrho_mag = sqrt(gradrho[0]*gradrho[0]+gradrho[1]*gradrho[1]+gradrho[2]*gradrho[2]);
        if(gradrho_mag > 0) {gradrho_mag = rho*rho/gradrho_mag;} else {gradrho_mag=0;}
        if(include_h > 0) if(numngb_ndim > 0) gradrho_mag += include_h * rho * rkern / numngb_ndim;
#endif
    }
    return gradrho_mag * All.cf_a2inv;
}

/* calculate_face_area_for_cartesian_mesh — migrated from predict.cc to fix
 * #20011-D (host-only fn called from KOKKOS_INLINE_FUNCTION
 * compute_finitevol_faces template under HYDRO_REGULAR_GRID). Body uses only
 * All.cf_atime (mirror-safe), std::max, fabs — all device-callable.
 * Phase D 2026-05-21. */
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
KOKKOS_INLINE_FUNCTION
double calculate_face_area_for_cartesian_mesh(const Vec3<double>& dp, double rinv, double l_side, Vec3<double>& Face_Area_Vec)
{
    Face_Area_Vec = {}; double Face_Area_Norm;
#if (NUMDIMS==1)
    Face_Area_Norm = 1; Face_Area_Vec[0] = Face_Area_Norm * dp[0]/fabs(dp[0]);
#elif (NUMDIMS==2)
    if(fabs(dp[0]) > fabs(dp[1])) {Face_Area_Vec[0] = Face_Area_Norm = DMAX(0.,l_side-fabs(dp[1])) * dp[0]/fabs(dp[0]) * All.cf_atime;} else {Face_Area_Vec[1] = Face_Area_Norm = DMAX(0.,l_side-fabs(dp[0])) * dp[1]/fabs(dp[1]) * All.cf_atime;}
#else
    Vec3<double> dp_abs = {fabs(dp[0]), fabs(dp[1]), fabs(dp[2])};
    int kdir;
    if((dp_abs[0]>=dp_abs[1])&&(dp_abs[0]>=dp_abs[2])) {kdir=0;} else if ((dp_abs[1]>=dp_abs[0])&&(dp_abs[1]>=dp_abs[2])) {kdir=1;} else {kdir=2;}
    Face_Area_Norm=1; for(int k=0;k<3;k++) {if(k!=kdir) {Face_Area_Norm *= DMAX(0.,l_side-dp_abs[k]) * All.cf_atime*All.cf_atime;}}
    Face_Area_Vec[kdir] = Face_Area_Norm * dp[kdir]/fabs(dp[kdir]);
#endif
    return fabs(Face_Area_Norm);
}
#endif

/* Get_Particle_Expected_Area — migrated from predict.cc to fix #20011-D
 * (host-only fn called from KOKKOS_INLINE_FUNCTION compute_finitevol_faces
 * under SLOPE_LIMITER_TOLERANCE==0 — surfaced 2026-05-21 Phase D config 152).
 * Pure-compute function of `h`, dimension-dependent. */
KOKKOS_INLINE_FUNCTION
double Get_Particle_Expected_Area(double h)
{
#if (NUMDIMS == 1)
    return 2;
#endif
#if (NUMDIMS == 2)
    return 2 * M_PI * h;
#endif
#if (NUMDIMS == 3)
    return 4 * M_PI * h * h;
#endif
}


#ifdef DIVBCLEANING_DEDNER
KOKKOS_INLINE_FUNCTION
double Get_Gas_PhiField_P(int i_particle_id, struct particle_data *pp, struct gas_cell_data *cell)
{
    //return cell[i_particle_id].PhiPred * cell[i_particle_id].Density / pp[i_particle_id].Mass; // volumetric phy-flux (requires extra term compared to mass-based flux)
    return cell[i_particle_id].PhiPred / pp[i_particle_id].Mass; // mass-based phi-flux
}
#endif /* DIVBCLEANING_DEDNER */


#ifdef DIVBCLEANING_DEDNER
KOKKOS_INLINE_FUNCTION
double Get_Gas_PhiField_DampingTimeInv_P(int i_particle_id, struct particle_data *pp, struct gas_cell_data *cell)
{
    /* this timescale should always be returned as a -physical- time */
#ifdef HYDRO_SPH
    /* PFH: add simple damping (-phi/tau) term */
    double damping_tinv = 0.5 * All.DivBcleanParabolicSigma * (cell[i_particle_id].MaxSignalVel / (All.cf_atime*pp[i_particle_id].Get_Particle_Size()));
#else
    double damping_tinv;
#ifdef SELFGRAVITY_OFF
    damping_tinv = All.DivBcleanParabolicSigma * All.FastestWaveSpeed / (All.cf_atime*pp[i_particle_id].Get_Particle_Size()); // fastest wavespeed has units of [vphys]
    //double damping_tinv = All.DivBcleanParabolicSigma * All.FastestWaveDecay * All.cf_a2inv; // no improvement over fastestwavespeed; decay has units [vphys/rphys]
#else
    // only see a small performance drop from fastestwavespeed above to maxsignalvel below, despite the fact that below is purely local (so allows more flexible adapting to high dynamic range)
    damping_tinv = 0.0;

    if(pp[i_particle_id].KernelRadius > 0)
    {
        double h_eff = pp[i_particle_id].Get_Particle_Size();
        double vsig2 = 0.5 * fabs(cell[i_particle_id].MaxSignalVel);
        double phi_B_eff = 0.0;
        if(vsig2 > 0) {phi_B_eff = Get_Gas_PhiField_P(i_particle_id, pp, cell) / (All.cf_atime * vsig2);}
        double vsig1 = 0.0;
        if(cell[i_particle_id].Density > 0)
        {
            vsig1 = sqrt( cell[i_particle_id].effective_soundspeed()*cell[i_particle_id].effective_soundspeed() +
                 (1. / All.cf_atime) *
                 (cell[i_particle_id].Bfield().norm_sq() +
                  phi_B_eff*phi_B_eff) / cell[i_particle_id].Density );
        }
        vsig1 = DMAX(vsig1, vsig2);
        vsig2 = 0.0;
        vsig2 = cell[i_particle_id].Gradients.Velocity.frobenius_norm();
        vsig2 = 3.0 * h_eff * DMAX( vsig2, fabs(pp[i_particle_id].Particle_DivVel)) / All.cf_atime;
        double prefac_fastest = 0.1;
        double prefac_tinv = 0.5;
        double area_0 = 0.1;
#ifdef MHD_CONSTRAINED_GRADIENT
        prefac_fastest = 1.0;
        prefac_tinv = 2.0;
        area_0 = 0.05;
        vsig2 *= 5.0;
        if(cell[i_particle_id].FlagForConstrainedGradients <= 0) prefac_tinv *= 30;
#endif
        prefac_tinv *= sqrt(1. + cell[i_particle_id].ConditionNumber/100.);
        double area = fabs(cell[i_particle_id].Face_Area[0]) + fabs(cell[i_particle_id].Face_Area[1]) + fabs(cell[i_particle_id].Face_Area[2]);
        area /= Get_Particle_Expected_Area(pp[i_particle_id].KernelRadius);
        prefac_tinv *= (1. + area/area_0)*(1. + area/area_0);

        double vsig_max = DMAX( DMAX(vsig1,vsig2) , prefac_fastest * All.FastestWaveSpeed );
        damping_tinv = prefac_tinv * All.DivBcleanParabolicSigma * (vsig_max / (All.cf_atime * h_eff));
    }
#endif
#endif
    return damping_tinv;
}
#endif /* DIVBCLEANING_DEDNER */


#ifdef HYDRO_MESHLESS_FINITE_VOLUME
KOKKOS_INLINE_FUNCTION
void advect_mesh_point_P(int i, double dt, struct particle_data *pp, struct gas_cell_data *cell)
{
#if (HYDRO_FIX_MESH_MOTION == 2) || (HYDRO_FIX_MESH_MOTION == 3) // cylindrical or spherical coordinates
    // define the location relative to the origin (needed in these coordinate systems)
    Vec3<double> dp = pp[i].Pos; Vec3<double> dp_offset = {}; // assume center is at coordinate origin
#if defined(GRAVITY_ANALYTIC_ANCHOR_TO_PARTICLE) // unless we use a special anchor, to define the center
    dp_offset = pp[i].Pos - pp[i].Min_xyz_to_Sink;
#elif defined(BOX_PERIODIC) // or if periodic, the box mid-point is instead the center
#if (NUMDIMS==1)
    dp_offset[0] = -boxHalf_X;
#elif (NUMDIMS==2)
    dp_offset[0] = -boxHalf_X; dp_offset[1] = -boxHalf_Y;
#else
    dp_offset = Vec3<double>{-boxHalf_X, -boxHalf_Y, -boxHalf_Z};
#endif
#endif
    dp += dp_offset;
#if (HYDRO_FIX_MESH_MOTION == 2) // cylindrical
    double r2=dp[0]*dp[0]+dp[1]*dp[1], r=sqrt(r2), c0=dp[0]/r, s0=dp[1]/r, z=dp[2]; // get r, sin/cos theta, z
    double vr=c0*cell[i].ParticleVel[0] + s0*cell[i].ParticleVel[1], vt=s0*cell[i].ParticleVel[0] - c0*cell[i].ParticleVel[1], vz=cell[i].ParticleVel[2]; // velocities in these directions
    double r_n=r+vr*dt, z_n=z+vz*dt, c_n=c0-s0*(vt/r)*dt, s_n=s0+c0*(vt/r)*dt; // updated cylindrical values
    dp[0] = c_n*r_n; dp[1] = s_n*r_n; dp[2] = z_n; // back to coordinates
    cell[i].ParticleVel[0] = c_n*vr + s_n*vt; // re-set velocities in these coordinates //
    cell[i].ParticleVel[1] = s_n*vr - c_n*vt;
    cell[i].ParticleVel[2] = vz;
    return;
#elif (HYDRO_FIX_MESH_MOTION == 3) // spherical
    Vec3<double> v = cell[i].ParticleVel; double r2=dp.norm_sq(); // assume center is at coordinate origin
    double r=sqrt(r2), rxy=sqrt(dp[0]*dp[0]+dp[1]*dp[1]), vr=dot(dp,v)/r; // updated r is easy
    double ct = 1./sqrt(1.+dp[1]*dp[1]/(dp[0]*dp[0])), st = (dp[1]/dp[0])*ct; // cos and sin theta
    double cp = sqrt(1.-dp[2]*dp[2]/(r*r)), sp = dp[2]/r; // cos and sin phi
    double t_dot = (v[0]*dp[1]-v[1]*dp[0])/(rxy*rxy), p_dot = (dp[2]*(dp[0]*v[0]+dp[1]*v[1])-rxy*rxy*v[2])/(r*r*rxy); // theta, phi derivatives
    double r_n=r+vr*dt, ct_n=ct-st*t_dot, st_n=st+ct*t_dot, cp_n=cp-sp*t_dot, sp_n=sp+cp*t_dot; // updated angles and positions in spherical
    dp[0] = r_n * ct_n * cp_n; dp[1] = r_n * st_n * cp_n; dp[2] = r_n * sp_n; // back to coordinates
    rxy = sqrt(dp[0]*dp[0] + dp[1]*dp[1]); // updated rxy
    cell[i].ParticleVel[0] = (dp[0]/r_n) * vr + dp[1] * t_dot + dp[0]*dp[2]/rxy * p_dot; // back to cartesian velocities
    cell[i].ParticleVel[1] = (dp[1]/r_n) * vr - dp[0] * t_dot + dp[1]*dp[2]/rxy * p_dot; // back to cartesian velocities
    cell[i].ParticleVel[2] = (dp[2]/r_n) * vr - rxy * p_dot; // back to cartesian velocities
    return;
#endif
    // ok now have the updated x/y/z positions relative to the origin, convert these back to the simulation coordinate frame
    pp[i].Pos = dp - dp_offset;
#endif // ok done with cylindrical/spherical coordinates


    // ok anything else ('normal' coordinates), does down here
    pp[i].Pos += cell[i].ParticleVel * dt; // for standard grid velocities, this is trivial //
    return;
}
#endif /* HYDRO_MESHLESS_FINITE_VOLUME */


KOKKOS_INLINE_FUNCTION
void apply_special_boundary_conditions_P(int i, double mass_for_dp, int mode, struct particle_data *pp, struct gas_cell_data *cell)
{
#if BOX_DEFINED_SPECIAL_XYZ_BOUNDARY_CONDITIONS_ARE_ACTIVE
    double box_upper[3]; int j;
    box_upper[0]=boxSize_X; box_upper[1]=boxSize_Y; box_upper[2]=boxSize_Z;
    for(j=0; j<3; j++)
    {
        if(pp[i].Pos[j] <= 0)
        {
            if(special_boundary_condition_xyz_def_reflect[j] == 0 || special_boundary_condition_xyz_def_reflect[j] == -1)
            {
                if(pp[i].Vel[j]<0) {pp[i].Vel[j]=-pp[i].Vel[j]; if(pp[i].Type==0) {cell[i].VelPred[j]=pp[i].Vel[j]; cell[i].HydroAccel[j]=0;} if(mode==1) {pp[i].dp[j]+=2*pp[i].Vel[j]*mass_for_dp;}}
                pp[i].Pos[j]=DMAX((0.+((double)pp[i].ID)*2.e-8)*box_upper[j], 0.1*pp[i].Pos[j]); // old  was 1e-9, safer on some problems, but can artificially lead to 'trapping' in some low-res tests
#ifdef GRAIN_RDI_TESTPROBLEM_LIVE_RADIATION_INJECTION
                pp[i].Pos[j]+=3.e-3*boxSize_X; pp[i].Vel[j] += 0.1; /* special because of our wierd boundary condition for this problem, sorry to have so many hacks for this! */
#endif
#ifdef RT_EVOLVE_FLUX
                if(pp[i].Type==0) {int kf; for(kf=0;kf<N_RT_FREQ_BINS;kf++) {if(cell[i].Rad_Flux[kf][j]<0) {cell[i].Rad_Flux[kf][j]=-cell[i].Rad_Flux[kf][j]; cell[i].Rad_Flux_Pred[kf][j]=cell[i].Rad_Flux[kf][j];}}}
#endif

#ifdef COSMIC_RAY_FLUID
                if(pp[i].Type==0) {int kf; for(kf=0;kf<N_CR_PARTICLE_BINS;kf++) {if(cell[i].CosmicRayFlux[kf][j]<0) {cell[i].CosmicRayFlux[kf][j]=-cell[i].CosmicRayFlux[kf][j]; cell[i].CosmicRayFluxPred[kf][j]=cell[i].CosmicRayFlux[kf][j];}}}
#endif
            }
            if(special_boundary_condition_xyz_def_outflow[j] == 0 || special_boundary_condition_xyz_def_outflow[j] == -1) {pp[i].Mass=0; if(pp[i].Type==0) {cell[i].Mass=0;} if(mode==1) {pp[i].dp[0]=pp[i].dp[1]=pp[i].dp[2]=0;}}
        }
        else if (pp[i].Pos[j] >= box_upper[j])
        {
            if(special_boundary_condition_xyz_def_reflect[j] == 0 || special_boundary_condition_xyz_def_reflect[j] == 1)
            {
                if(pp[i].Vel[j]>0) {pp[i].Vel[j]=-pp[i].Vel[j]; if(pp[i].Type==0) {cell[i].VelPred[j]=pp[i].Vel[j]; cell[i].HydroAccel[j]=0;} if(mode==1) {pp[i].dp[j]+=2*pp[i].Vel[j]*mass_for_dp;}}
                pp[i].Pos[j]=box_upper[j]*(1.-((double)pp[i].ID)*2.e-8);
#ifdef RT_EVOLVE_FLUX
                if(pp[i].Type==0) {int kf; for(kf=0;kf<N_RT_FREQ_BINS;kf++) {if(cell[i].Rad_Flux[kf][j]>0) {cell[i].Rad_Flux[kf][j]=-cell[i].Rad_Flux[kf][j]; cell[i].Rad_Flux_Pred[kf][j]=cell[i].Rad_Flux[kf][j];}}}
#endif
#ifdef COSMIC_RAY_FLUID
                if(pp[i].Type==0) {int kf; for(kf=0;kf<N_CR_PARTICLE_BINS;kf++) {if(cell[i].CosmicRayFlux[kf][j]>0) {cell[i].CosmicRayFlux[kf][j]=-cell[i].CosmicRayFlux[kf][j]; cell[i].CosmicRayFluxPred[kf][j]=cell[i].CosmicRayFlux[kf][j];}}}
#endif
            }
            if(special_boundary_condition_xyz_def_outflow[j] == 0 || special_boundary_condition_xyz_def_outflow[j] == 1) {pp[i].Mass=0; if(pp[i].Type==0) {cell[i].Mass=0;} if(mode==1) {pp[i].dp[0]=pp[i].dp[1]=pp[i].dp[2]=0;}}
        }
    }
#endif
    return;
}
