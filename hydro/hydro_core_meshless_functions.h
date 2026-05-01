/* hydro_core_meshless_functions.h -- MFM / MFV Riemann core, as a function.
 *
 * Replaces the #include-fragment hydro/hydro_core_meshless.h that carried the
 * full face reconstruction + Riemann solve + flux projection for the meshless
 * methods. The original fragment relied on about two dozen enclosing-scope
 * variables; here those become explicit arguments (with references for the
 * ones that are mutated: face_vel_i/j, Face_Area_Vec, Face_Area_Norm,
 * face_area_dot_vel, kernel.vsig via MFM+GALSF, Fluxes, Riemann_vec,
 * Riemann_out, mdot_estimated, out.MaxShockMachNumber).
 *
 * Body compiles to nothing under HYDRO_SPH — hydro_core_sph has its own
 * function (hydro_core_sph_functions.h).
 *
 * Requires allvars.h, kernel.h, hydro_structs.h, reimann.h, hydro_pair_types.h,
 * compute_finitevol_faces_functions.h included.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef HYDRO_CORE_MESHLESS_FUNCTIONS_H
#define HYDRO_CORE_MESHLESS_FUNCTIONS_H

#include "hydro_pair_types.h"
#include "compute_finitevol_faces_functions.h"

KOKKOS_INLINE_FUNCTION
void hydro_core_meshless_compute_pair(
    const struct hydro_data_in &local,
    int j,
    struct particle_data *P,
    struct gas_cell_data *CellP,
    const Vec3<MyDouble> &VelPred_j,
    const Vec3<MyDouble> &ParticleVel_j,
    const Vec3<double> &BPred_j,
    double PhiPred_j,
    struct kernel_hydra &kernel,
    double rinv, double r2,
    double V_i, double V_j,
    double Particle_Size_i, double Particle_Size_j,
    double cnumcrit2,
    double fac_magnetic_pressure,
    double tensile_correction_factor,
    double epsilon_entropic_eos_big,
    double epsilon_entropic_eos_small,
    const double *CosmicRayPressure_j,
    Vec3<double> &Face_Area_Vec,
    double &Face_Area_Norm,
    double &face_vel_i,
    double &face_vel_j,
    double &face_area_dot_vel,
    double &mdot_estimated,
    struct Input_vec_Riemann &Riemann_vec,
    struct Riemann_outputs &Riemann_out,
    struct Conserved_var_Riemann &Fluxes,
    struct hydro_data_out &out)
{
#ifndef HYDRO_SPH
    int k;
    double s_star_ij, s_i, s_j, dummy_pressure;
    double distance_from_i[3], distance_from_j[3];
    double leak_vs_tol = 0;
    Vec3<double> v_frame;
#if !(defined(HYDRO_KERNEL_SURFACE_VOLCORR) || defined(EOS_ELASTIC))
    leak_vs_tol = 0.5 * (local.FaceClosureError + CellP[j].FaceClosureError);
#endif
    dummy_pressure = 0;
    face_area_dot_vel = 0; face_vel_i = 0; face_vel_j = 0; Face_Area_Norm = 0;

    double Pressure_i = local.Pressure, Pressure_j = CellP[j].Pressure;
#if defined(EOS_TILLOTSON) || defined(EOS_ELASTIC) || defined(EOS_ANEOS)
    if((Pressure_i < 0) || (Pressure_j < 0)) {
        dummy_pressure = -DMIN(Pressure_i, Pressure_j);
        Pressure_i += dummy_pressure; Pressure_j += dummy_pressure;
        dummy_pressure *= 1. - tensile_correction_factor;
    }
#endif
#ifdef COSMIC_RAY_FLUID
    for(k=0; k<N_CR_PARTICLE_BINS; k++) {
        Fluxes.CosmicRayPressure[k] = 0;
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
        Fluxes.CosmicRayAlfvenEnergy[k][0] = Fluxes.CosmicRayAlfvenEnergy[k][1] = 0;
#endif
    }
#endif

    V_j = P[j].Mass / CellP[j].Density;
    s_star_ij = 0;

    double Vi_inv_corr, Vj_inv_corr;
    compute_finitevol_faces(local, CellP[j], kernel, rinv, r2, V_i, V_j,
                            Particle_Size_i, Particle_Size_j, cnumcrit2,
                            Face_Area_Vec, Face_Area_Norm, Vi_inv_corr, Vj_inv_corr);

    if(Face_Area_Norm == 0) {
        memset(&Fluxes, 0, sizeof(struct Conserved_var_Riemann));
#ifdef DIVBCLEANING_DEDNER
        Riemann_out.phi_normal_mean = Riemann_out.phi_normal_db = 0;
#endif
        return;
    }

    if((Face_Area_Norm <= 0) || (isnan(Face_Area_Norm))) {
        PRINT_WARNING("PANIC! Face_Area_Norm=%g Mij=%g/%g wk_ij=%g/%g Vij=%g/%g dx/dy/dz=%g/%g/%g NVT=%g/%g/%g NVT_j=%g/%g/%g \n",
            Face_Area_Norm, local.Mass, P[j].Mass, kernel.wk_i, kernel.wk_j, V_i, V_j,
            kernel.dp[0], kernel.dp[1], kernel.dp[2],
            local.NV_T[0][0], local.NV_T[0][1], local.NV_T[0][2],
            CellP[j].NV_T[0][0], CellP[j].NV_T[0][1], CellP[j].NV_T[0][2]);
    }
    Vec3<double> n_unit = Face_Area_Vec / Face_Area_Norm;

    s_i =  0.5 * kernel.r;
    s_j = -0.5 * kernel.r;
    s_i = s_star_ij - s_i;
    s_j = s_star_ij - s_j;
    distance_from_i[0] = kernel.dp[0] * rinv;
    distance_from_i[1] = kernel.dp[1] * rinv;
    distance_from_i[2] = kernel.dp[2] * rinv;
    for(k=0; k<3; k++) { distance_from_j[k] = distance_from_i[k] * s_j; distance_from_i[k] *= s_i; }
    v_frame = rinv * (-s_i * VelPred_j + s_j * local.Vel);
#if defined(HYDRO_MESHLESS_FINITE_VOLUME)
    v_frame = rinv * (-s_i * ParticleVel_j + s_j * local.ParticleVel);
#endif

    face_vel_i += dot(local.Vel, n_unit); face_vel_j += dot(VelPred_j, n_unit);
    face_vel_i /= All.cf_atime; face_vel_j /= All.cf_atime;
    face_area_dot_vel = rinv * (-s_i * face_vel_j + s_j * face_vel_i);

    double v2_approach = 0;
    double vdotr2_phys = kernel.vdotr2;
    if(All.ComovingIntegrationOn) { vdotr2_phys -= All.cf_hubble_a2 * r2; }
    vdotr2_phys *= 1.0 / (kernel.r * All.cf_atime);
    if(vdotr2_phys < 0) { v2_approach = vdotr2_phys * vdotr2_phys; }
    double vdotf2_phys = face_vel_i - face_vel_j;
    if(vdotf2_phys < 0) { v2_approach = DMAX(v2_approach, vdotf2_phys * vdotf2_phys); }

    volatile int recon_mode = 1; /* volatile: nvc++ may otherwise fold this runtime gate to its initializer inside device code */
#if defined(GALSF) || defined(COOLING)
    if(fabs(vdotr2_phys) * UNIT_VEL_IN_KMS > 1000.) { recon_mode = 0; }
#endif
    if(leak_vs_tol > 1) { recon_mode = 0; }

    double rho_i = local.Density, rho_j = CellP[j].Density;
    double P_i = Pressure_i, P_j = Pressure_j;
#if defined(HYDRO_FACE_VOLUME_RECONSTRUCTION_CORRECTION)
    if(Vi_inv_corr * Vj_inv_corr < 0.999) { recon_mode = 0; }
    rho_i *= Vi_inv_corr; P_i *= Vi_inv_corr;
    rho_j *= Vj_inv_corr; P_j *= Vj_inv_corr;
#endif

    reconstruct_face_states(rho_i, local.Gradients.Density, rho_j, CellP[j].Gradients.Density,
                            distance_from_i, distance_from_j, &Riemann_vec.L.rho, &Riemann_vec.R.rho, recon_mode);
    reconstruct_face_states(P_i, local.Gradients.Pressure, P_j, CellP[j].Gradients.Pressure,
                            distance_from_i, distance_from_j, &Riemann_vec.L.p, &Riemann_vec.R.p, recon_mode);
#ifdef EOS_GENERAL
    reconstruct_face_states(local.InternalEnergyPred, local.Gradients.InternalEnergy,
                            CellP[j].InternalEnergyPred, CellP[j].Gradients.InternalEnergy,
                            distance_from_i, distance_from_j, &Riemann_vec.L.u, &Riemann_vec.R.u, recon_mode);
    reconstruct_face_states(kernel.sound_i, local.Gradients.SoundSpeed, kernel.sound_j, CellP[j].Gradients.SoundSpeed,
                            distance_from_i, distance_from_j, &Riemann_vec.L.cs, &Riemann_vec.R.cs, recon_mode);
#endif
    for(k=0; k<3; k++) {
        reconstruct_face_states(local.Vel[k], local.Gradients.Velocity[k], VelPred_j[k], CellP[j].Gradients.Velocity[k],
                                distance_from_i, distance_from_j, &Riemann_vec.L.v[k], &Riemann_vec.R.v[k], recon_mode);
        Riemann_vec.L.v[k] -= v_frame[k]; Riemann_vec.R.v[k] -= v_frame[k];
    }
#ifdef MAGNETIC
    int slim_mode = 1;
#ifdef MHD_CONSTRAINED_GRADIENT
    if((local.ConditionNumber < 0) || (CellP[j].FlagForConstrainedGradients == 0)) { slim_mode = 1; } else { slim_mode = -1; }
#endif
    for(k=0; k<3; k++) {
        reconstruct_face_states(local.BPred[k], local.Gradients.B[k], BPred_j[k], CellP[j].Gradients.B[k],
                                distance_from_i, distance_from_j, &Riemann_vec.L.B[k], &Riemann_vec.R.B[k], slim_mode);
    }
#ifdef MHD_MODIFIED_GRADIENT
    {
        double A_dot_dp = dot(Face_Area_Vec, kernel.dp);
        double mg_ci = local.MG_cgcoeff;
        double mg_cj = CellP[j].MG_cgcoeff;
        double mg_fac = 1.0;
        if(All.Flag_SkipMGSolve) {
            double dBn_before = 0, dBn_delta = 0;
            for(k=0; k<3; k++) {
                dBn_before += (Riemann_vec.R.B[k] - Riemann_vec.L.B[k]) * Face_Area_Vec[k];
                dBn_delta  += (-mg_ci - mg_cj) * 0.25 * kernel.dp[k] * A_dot_dp * Face_Area_Vec[k];
            }
            double abs_before = fabs(dBn_before);
            double dBn_after = dBn_before + dBn_delta;
            double abs_after = fabs(dBn_after);
            if(abs_after > abs_before && fabs(dBn_delta) > 1.0e-60) {
                if(dBn_before * dBn_delta > 0) {
                    mg_fac = 0.0;
                } else {
                    mg_fac = fabs(dBn_before) / fabs(dBn_delta);
                    if(mg_fac > 1.0) mg_fac = 1.0;
                }
            }
        }
        for(k=0; k<3; k++) {
            Riemann_vec.R.B[k] += mg_fac * (-mg_ci) * 0.25 * kernel.dp[k] * A_dot_dp;
            Riemann_vec.L.B[k] += mg_fac * ( mg_cj) * 0.25 * kernel.dp[k] * A_dot_dp;
        }
    }
#endif
#ifdef DIVBCLEANING_DEDNER
    reconstruct_face_states(local.PhiPred, local.Gradients.Phi, PhiPred_j, CellP[j].Gradients.Phi,
                            distance_from_i, distance_from_j, &Riemann_vec.L.phi, &Riemann_vec.R.phi, 2);
#endif
#endif

    double press_i_tot = Pressure_i + local.Density * v2_approach;
    double press_j_tot = Pressure_j + CellP[j].Density * v2_approach;
#ifdef MAGNETIC
    press_i_tot += 0.5 * kernel.b2_i * fac_magnetic_pressure;
    press_j_tot += 0.5 * kernel.b2_j * fac_magnetic_pressure;
#endif
    double press_tot_limiter;
#ifdef MAGNETIC
    press_tot_limiter = 2.0 * 1.1 * All.cf_a3inv * (press_i_tot + press_j_tot);
#else
    press_tot_limiter = 1.1 * All.cf_a3inv * DMAX(press_i_tot, press_j_tot);
#endif
#if defined(EOS_GENERAL) || defined(HYDRO_MESHLESS_FINITE_VOLUME)
    press_tot_limiter *= 2.0;
#endif
#if (SLOPE_LIMITER_TOLERANCE == 2)
    press_tot_limiter *= 100.0;
#endif
    if(recon_mode == 0) {
        press_tot_limiter = DMAX(press_tot_limiter,
                                 DMAX(DMAX(Pressure_i, Pressure_j),
                                      2. * DMAX(local.Density, CellP[j].Density) * v2_approach));
    }
#if defined(EOS_TILLOTSON) || defined(EOS_ELASTIC) || defined(EOS_ANEOS)
    press_tot_limiter = 1.e10 * (press_tot_limiter + 1.);
#endif

    Riemann_solver(Riemann_vec, &Riemann_out, n_unit, press_tot_limiter);
    if((Riemann_out.P_M < 0) || (isnan(Riemann_out.P_M)) || (Riemann_out.P_M > 1.4 * press_tot_limiter))
    {
        /* retry with linear reconstruction */
        Riemann_vec.R.p = Pressure_i; Riemann_vec.L.p = Pressure_j;
        Riemann_vec.R.rho = local.Density; Riemann_vec.L.rho = CellP[j].Density;
        Riemann_vec.R.v = local.Vel - v_frame; Riemann_vec.L.v = VelPred_j - v_frame;
#ifdef MAGNETIC
        Riemann_vec.R.B = local.BPred; Riemann_vec.L.B = BPred_j;
#ifdef DIVBCLEANING_DEDNER
        Riemann_vec.R.phi = local.PhiPred; Riemann_vec.L.phi = PhiPred_j;
#endif
#endif
#ifdef EOS_GENERAL
        Riemann_vec.R.u = local.InternalEnergyPred; Riemann_vec.L.u = CellP[j].InternalEnergyPred;
        Riemann_vec.R.cs = kernel.sound_i; Riemann_vec.L.cs = kernel.sound_j;
#endif
        Riemann_solver(Riemann_vec, &Riemann_out, n_unit, 1.4 * press_tot_limiter);
        if((Riemann_out.P_M < 0) || (isnan(Riemann_out.P_M)))
        {
            /* zero-velocity fallback */
            Riemann_vec.R.p = Pressure_i; Riemann_vec.L.p = Pressure_j;
            Riemann_vec.R.rho = local.Density; Riemann_vec.L.rho = CellP[j].Density;
            Riemann_vec.R.v = {}; Riemann_vec.L.v = {};
#ifdef MAGNETIC
            Riemann_vec.R.B = local.BPred; Riemann_vec.L.B = BPred_j;
#ifdef DIVBCLEANING_DEDNER
            Riemann_vec.R.phi = local.PhiPred; Riemann_vec.L.phi = PhiPred_j;
#endif
#endif
#ifdef EOS_GENERAL
            Riemann_vec.R.u = local.InternalEnergyPred; Riemann_vec.L.u = CellP[j].InternalEnergyPred;
            Riemann_vec.R.cs = kernel.sound_i; Riemann_vec.L.cs = kernel.sound_j;
#endif
            Riemann_solver(Riemann_vec, &Riemann_out, n_unit, 2.0 * press_tot_limiter);
            if((Riemann_out.P_M < 0) || (isnan(Riemann_out.P_M)))
            {
#if defined(MAGNETIC) && defined(DIVBCLEANING_DEDNER)
                printf("Riemann Solver Failed to Find Positive Pressure!: Pmax=%g PL/M/R=%g/%g/%g Mi/j=%g/%g rhoL/R=%g/%g H_ij=%g/%g vL=%g/%g/%g vR=%g/%g/%g n_unit=%g/%g/%g BL=%g/%g/%g BR=%g/%g/%g phiL/R=%g/%g \n",
                       press_tot_limiter, Riemann_vec.L.p, Riemann_out.P_M, Riemann_vec.R.p,
                       local.Mass, P[j].Mass, Riemann_vec.L.rho, Riemann_vec.R.rho,
                       local.KernelRadius, P[j].KernelRadius,
                       local.Vel[0] - v_frame[0], local.Vel[1] - v_frame[1], local.Vel[2] - v_frame[2],
                       VelPred_j[0] - v_frame[0], VelPred_j[1] - v_frame[1], VelPred_j[2] - v_frame[2],
                       n_unit[0], n_unit[1], n_unit[2],
                       Riemann_vec.L.B[0], Riemann_vec.L.B[1], Riemann_vec.L.B[2],
                       Riemann_vec.R.B[0], Riemann_vec.R.B[1], Riemann_vec.R.B[2],
                       Riemann_vec.L.phi, Riemann_vec.R.phi);
#else
                printf("Riemann Solver Failed to Find Positive Pressure!: Pmax=%g PL/M/R=%g/%g/%g Mi/j=%g/%g rhoL/R=%g/%g vL=%g/%g/%g vR=%g/%g/%g n_unit=%g/%g/%g \n",
                       press_tot_limiter, Riemann_vec.L.p, Riemann_out.P_M, Riemann_vec.R.p,
                       local.Mass, P[j].Mass, Riemann_vec.L.rho, Riemann_vec.R.rho,
                       Riemann_vec.L.v[0], Riemann_vec.L.v[1], Riemann_vec.L.v[2],
                       Riemann_vec.R.v[0], Riemann_vec.R.v[1], Riemann_vec.R.v[2],
                       n_unit[0], n_unit[1], n_unit[2]);
#endif
                endrun(1234);
            }
        }
    }

    if(!((Riemann_out.P_M > 0) && (!isnan(Riemann_out.P_M)))) {
        /* nothing but bad riemann solutions found */
        memset(&Fluxes, 0, sizeof(struct Conserved_var_Riemann));
#ifdef DIVBCLEANING_DEDNER
        Riemann_out.phi_normal_mean = Riemann_out.phi_normal_db = 0;
#endif
        return;
    }

    if(All.ComovingIntegrationOn) { v_frame /= All.cf_atime; }
#ifdef TURB_DIFF_METALS
    mdot_estimated = Riemann_out.Mdot_estimated * Face_Area_Norm;
#endif
    if(dummy_pressure != 0) {
        Riemann_out.Fluxes.v -= dummy_pressure * n_unit;
        Riemann_out.Fluxes.p -= dummy_pressure * Riemann_out.S_M;
    }

#if defined(HYDRO_MESHLESS_FINITE_MASS) && defined(GALSF)
    kernel.vsig = 2. * Riemann_out.S_M + DMAX(0, face_vel_j - face_vel_i);
#endif

    /* de-boost to simulation frame (Pakmor et al. 2011) */
    Riemann_out.Fluxes.p += dot(v_frame, Riemann_out.Fluxes.v);
#if defined(HYDRO_MESHLESS_FINITE_VOLUME)
    Riemann_out.Fluxes.p += 0.5 * v_frame.norm_sq() * Riemann_out.Fluxes.rho;
    Riemann_out.Fluxes.v += v_frame * Riemann_out.Fluxes.rho;
#endif
#ifdef MAGNETIC
    Riemann_out.Fluxes.B += -v_frame * Riemann_out.B_normal_corrected;
#endif

#if defined(HYDRO_MESHLESS_FINITE_VOLUME)
    Fluxes.rho = Face_Area_Norm * Riemann_out.Fluxes.rho;
#endif
    Fluxes.p = Face_Area_Norm * Riemann_out.Fluxes.p;
    Fluxes.v = Face_Area_Norm * Riemann_out.Fluxes.v;
#if defined(COSMIC_RAY_FLUID) && defined(HYDRO_MESHLESS_FINITE_VOLUME)
    for(k=0; k<N_CR_PARTICLE_BINS; k++) {
        if(Fluxes.rho < 0) {
            Fluxes.CosmicRayPressure[k] = Fluxes.rho * (local.CosmicRayPressure[k] * V_i / ((GAMMA_COSMICRAY(k) - 1.) * local.Mass));
        } else {
            Fluxes.CosmicRayPressure[k] = Fluxes.rho * (CosmicRayPressure_j[k] * V_j / ((GAMMA_COSMICRAY(k) - 1.) * P[j].Mass));
        }
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
        for(int kAlf=0; kAlf<2; kAlf++) {
            if(Fluxes.rho < 0) {
                Fluxes.CosmicRayAlfvenEnergy[k][kAlf] += local.CosmicRayAlfvenEnergy[k][kAlf] * Fluxes.rho / local.Mass;
            } else {
                Fluxes.CosmicRayAlfvenEnergy[k][kAlf] += CellP[j].CosmicRayAlfvenEnergy[k][kAlf] * Fluxes.rho / local.Mass;
            }
        }
#endif
    }
#endif
#ifdef MAGNETIC
    Fluxes.B = Face_Area_Norm * Riemann_out.Fluxes.B;
    Fluxes.B_normal_corrected = -Riemann_out.B_normal_corrected * Face_Area_Norm;
#if defined(DIVBCLEANING_DEDNER) && defined(HYDRO_MESHLESS_FINITE_VOLUME)
    if(Fluxes.rho < 0) { Fluxes.phi = Fluxes.rho * Riemann_vec.R.phi; }
    else               { Fluxes.phi = Fluxes.rho * Riemann_vec.L.phi; }
#endif
#endif

#if defined(HYDRO_MESHLESS_FINITE_MASS) && (SLOPE_LIMITER_TOLERANCE < 2) && !(defined(EOS_TILLOTSON) || defined(EOS_ELASTIC) || defined(EOS_ANEOS))
    {
        double SM_over_ceff = fabs(Riemann_out.S_M) / DMIN(kernel.sound_i, kernel.sound_j);
        if((SM_over_ceff < epsilon_entropic_eos_big && All.ComovingIntegrationOn >= 0) || (leak_vs_tol > 1))
        {
#ifdef MAGNETIC
            Riemann_out.P_M -= 0.5 * Riemann_out.Face_B.norm_sq();
#endif
            volatile int use_entropic_energy_equation = 1; /* volatile: nvc++ may otherwise fold this runtime gate to its initializer inside device code */
            double facenorm_pm = Riemann_out.P_M * Face_Area_Norm;
            double PdV_fac = Riemann_out.P_M * vdotr2_phys / All.cf_a2inv;
            double PdV_i = kernel.dwk_i * V_i * V_i * local.DrkernNgbFactor * PdV_fac;
            double PdV_j = kernel.dwk_j * V_j * V_j * P[j].DrkernNgbFactor * PdV_fac;
            double du_old = facenorm_pm * (Riemann_out.S_M + face_area_dot_vel);
            double du_new = 0.5 * (PdV_i - PdV_j + facenorm_pm * (face_vel_i + face_vel_j));
            double cnum2 = CellP[j].ConditionNumber * CellP[j].ConditionNumber;
            double epsilon_threshold_entropic_eos_ratio = 1.001;
            if(SM_over_ceff > epsilon_entropic_eos_small && cnum2 < cnumcrit2)
            {
                if(Pressure_i / local.Density > epsilon_threshold_entropic_eos_ratio * Pressure_j / CellP[j].Density)
                {
                    double dtoj = -du_old + facenorm_pm * face_vel_j;
                    if(dtoj > 0) { use_entropic_energy_equation = 0; }
                    else { if(dtoj < 0) { if(dtoj > -du_new + facenorm_pm * face_vel_j) { use_entropic_energy_equation = 0; } } }
                } else if(epsilon_threshold_entropic_eos_ratio * Pressure_i / local.Density < Pressure_j / CellP[j].Density) {
                    double dtoi = +du_old - facenorm_pm * face_vel_i;
                    if(dtoi > 0) { use_entropic_energy_equation = 0; }
                    else { if(dtoi < 0) { if(dtoi > +du_new - facenorm_pm * face_vel_i) { use_entropic_energy_equation = 0; } } }
                } else {
                    double dtoj = -du_old + facenorm_pm * face_vel_j;
                    double dtoi = +du_old - facenorm_pm * face_vel_i;
                    if(dtoj < 0) { if(dtoj > -du_new + facenorm_pm * face_vel_j) { use_entropic_energy_equation = 0; } }
                    if(dtoi < 0) { if(dtoi > +du_new - facenorm_pm * face_vel_i) { use_entropic_energy_equation = 0; } }
                }
            }
            if(cnum2 >= cnumcrit2) { use_entropic_energy_equation = 1; }
            if(use_entropic_energy_equation) { Fluxes.p += du_new - du_old; }
        }
    }
#endif

#if defined(OUTPUT_SHOCK_MACH_NUMBER)
    {
        double dv_face_phys = face_vel_i - face_vel_j;
        if(dv_face_phys < 0 && Riemann_out.P_M > 0) {
            double cf_a3inv_fac = (All.ComovingIntegrationOn) ? All.cf_a3inv : 1.0;
            double P_L_phys = Riemann_vec.L.p * cf_a3inv_fac;
            double P_R_phys = Riemann_vec.R.p * cf_a3inv_fac;
            if(P_L_phys > 0 && P_R_phys > 0) {
                int upstream_is_L = (P_L_phys <= P_R_phys);
                double P_up   = upstream_is_L ? P_L_phys : P_R_phys;
                double cs_up  = upstream_is_L ? kernel.sound_j : kernel.sound_i;
                double rho_up = upstream_is_L ? Riemann_vec.L.rho : Riemann_vec.R.rho;
                if(cs_up > 0 && rho_up > 0) {
                    double pjump = Riemann_out.P_M / P_up;
                    if(pjump > 1.05) {
                        double gamma_eos = cs_up * cs_up * rho_up * cf_a3inv_fac / P_up;
                        gamma_eos = DMAX(1.001, DMIN(gamma_eos, 5./3. + 0.01));
                        double mach2_RH = ((gamma_eos + 1.0) * pjump + (gamma_eos - 1.0)) / (2.0 * gamma_eos);
                        double mach_RH = (mach2_RH > 1.0) ? sqrt(mach2_RH) : 1.0;
                        if(mach_RH > 1.0) {
                            if(mach_RH > out.MaxShockMachNumber) { out.MaxShockMachNumber = (MyFloat)mach_RH; }
                        }
                    }
                }
            }
        }
    }
#endif
#else
    /* HYDRO_SPH path: handled by hydro_core_sph_functions.h; this function is a no-op. */
    (void)local; (void)j; (void)P; (void)CellP;
    (void)VelPred_j; (void)ParticleVel_j; (void)BPred_j; (void)PhiPred_j;
    (void)kernel; (void)rinv; (void)r2; (void)V_i; (void)V_j;
    (void)Particle_Size_i; (void)Particle_Size_j; (void)cnumcrit2;
    (void)fac_magnetic_pressure; (void)tensile_correction_factor;
    (void)epsilon_entropic_eos_big; (void)epsilon_entropic_eos_small;
    (void)CosmicRayPressure_j; (void)Face_Area_Vec; (void)Face_Area_Norm;
    (void)face_vel_i; (void)face_vel_j; (void)face_area_dot_vel;
    (void)mdot_estimated; (void)Riemann_vec; (void)Riemann_out; (void)Fluxes; (void)out;
#endif /* !HYDRO_SPH */
}

#endif /* HYDRO_CORE_MESHLESS_FUNCTIONS_H */
