/* rt_diffusion_explicit_functions.h -- per-pair explicit radiation
 * diffusion / flux transport (M1 or FLD, depending on RT flags).
 *
 * Replaces the fragment radiation/rt_diffusion_explicit.h. Body guarded by
 * RT_SOLVER_EXPLICIT && !RT_EVOLVE_INTENSITIES so callers invoke
 * unconditionally. Pure i-accumulation into out.Dt_Rad_E_gamma,
 * out.Dt_Rad_Flux, out.Dt_Rad_E_gamma_T_weighted_IR. (j-side writes removed
 * with the j_is_active_for_fluxes sweep.)
 *
 * Templated on LocalT and OutT so both the hydro pass (hydro_data_in /
 * hydro_data_out) and the RT transport subcycle (transport INPUT_STRUCT_NAME /
 * OUTPUT_STRUCT_NAME) can call the function. LocalT must expose the RT / ET
 * / gradient fields the body touches under the active flag combination;
 * see rt_diffusion_explicit.h's original dependencies.
 *
 * Requires allvars.h, kernel.h, hydro_structs.h, hydro_pair_types.h,
 * and reimann.h (for reconstruct_face_states).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef RT_DIFFUSION_EXPLICIT_FUNCTIONS_H
#define RT_DIFFUSION_EXPLICIT_FUNCTIONS_H

#include "../hydro/hydro_pair_types.h"

/* increase numerical diffusion for smoother solutions (akin to
   slope-limiters ~ 0 model). Original was defined inside the fragment; moving
   to the function header preserves behaviour for all call sites. */
#ifndef RT_ENHANCED_NUMERICAL_DIFFUSION
#define RT_ENHANCED_NUMERICAL_DIFFUSION
#endif

template <typename LocalT, typename OutT>
KOKKOS_INLINE_FUNCTION
void rt_diffusion_explicit_compute_pair(
    const LocalT &local,
    int j,
    struct particle_data *P,
    struct gas_cell_data *CellP,
    const Vec3<MyDouble> &VelPred_j,
    const Vec3<MyDouble> &ParticleVel_j,
    const struct kernel_hydra &kernel,
    double rinv,
    const Vec3<double> &Face_Area_Vec,
    double Face_Area_Norm,
    double V_i, double V_j,
    double Particle_Size_i, double Particle_Size_j,
    double face_vel_i, double face_vel_j,
    const double *tau_c_i,
    double dt_hydrostep,
    OutT &out)
{
#if defined(RT_SOLVER_EXPLICIT) && !defined(RT_EVOLVE_INTENSITIES)
    int k;
    double c_light_eff = C_LIGHT_CODE_REDUCED;
    double rsol_corr   = RSOL_CORRECTION_FACTOR_FOR_VELOCITY_TERMS;
#if defined(HYDRO_MESHLESS_FINITE_VOLUME)
    Vec3<double> v_frame = 0.5 * (ParticleVel_j + local.ParticleVel) / All.cf_atime;
#else
    Vec3<double> v_frame = 0.5 * (VelPred_j + local.Vel) / All.cf_atime;
#endif
#if defined(RT_INFRARED)
    double Fluxes_Rad_E_gamma_T_weighted_IR = 0;
#endif
    double Fluxes_Rad_E_gamma[N_RT_FREQ_BINS];

#if !defined(RT_EVOLVE_FLUX)
    /* diffusion-only (FLD-like) path */
    for(int k_freq=0; k_freq<N_RT_FREQ_BINS; k_freq++)
    {
        Fluxes_Rad_E_gamma[k_freq] = 0;
        double kappa_ij = 0.5 * (local.RT_DiffusionCoeff[k_freq] + rt_diffusion_coefficient(j, k_freq, CellP));
        if(!((kappa_ij>0) && (local.Mass>0) && (P[j].Mass>0))) { continue; }

        double scalar_i = local.Rad_E_gamma[k_freq] / V_i;
        double scalar_j = CellP[j].Rad_E_gamma_Pred[k_freq] / V_j;

        double d_scalar = scalar_i - scalar_j;
        double conduction_wt = kappa_ij * All.cf_a3inv / All.cf_atime;
        double cmag=0., grad_norm=0, grad_dot_x_ij=0.0;
        for(k=0; k<3; k++) {
            double grad = 0.5 * (local.Gradients.Rad_E_gamma_ET[k_freq][k] + CellP[j].Gradients.Rad_E_gamma_ET[k_freq][k]);
            double grad_direct = d_scalar * kernel.dp[k] * rinv * rinv;
            grad_dot_x_ij += grad * kernel.dp[k];
            grad = MINMOD_G(grad, grad_direct);
#if defined(GALSF) || defined(COOLING) || defined(SINK_PARTICLES)
            double grad_direct_vs_abs_fac = 2.0;
#else
            double grad_direct_vs_abs_fac = 5.0;
#endif
            if(grad*grad_direct < 0) { if(fabs(grad_direct) > grad_direct_vs_abs_fac*fabs(grad)) { grad = 0.0; } }
            cmag += Face_Area_Vec[k] * grad;
            grad_norm += grad*grad;
        }
        double A_dot_grad_alignment = cmag*cmag / (Face_Area_Norm*Face_Area_Norm * grad_norm);
        cmag *= -conduction_wt;

        double v_eff_touse = DMIN(c_light_eff, kappa_ij / Particle_Size_j);
        double c_hll = DMIN(0.5*fabs(face_vel_i - face_vel_j) + v_eff_touse, c_light_eff);
        double q = 0.5 * c_hll * kernel.r * All.cf_atime / fabs(1.e-37 + kappa_ij);
        q = (0.2 + q) / (0.2 + q + q*q);
        double d_scalar_tmp = d_scalar - grad_dot_x_ij;
        double d_scalar_hll = MINMOD(d_scalar, d_scalar_tmp) * All.cf_a3inv;
        double hll_tmp = -A_dot_grad_alignment * q * Face_Area_Norm * c_hll * d_scalar_hll;

        double tau_c_j = Particle_Size_j * CellP[j].Rad_Kappa[k_freq] * CellP[j].Density * All.cf_a3inv;
        double hll_corr = 1. / (1. + 1.5 * DMAX(tau_c_i[k_freq], tau_c_j));
        hll_tmp *= hll_corr;

        double thold_hll_x = 2.0 * fabs(cmag);
        if(fabs(hll_tmp) > thold_hll_x) { hll_tmp *= thold_hll_x / fabs(hll_tmp); }
        double cmag_corr = cmag + hll_tmp;
        cmag = MINMOD(HLL_DIFFUSION_COMPROMISE_FACTOR * cmag, cmag_corr);

        double f_direct = -conduction_wt * (1./9.) * Face_Area_Norm * d_scalar * rinv;
        double check_for_stability_sign = f_direct * cmag;
        if((check_for_stability_sign < 0) && (fabs(f_direct) > HLL_DIFFUSION_OVERSHOOT_FACTOR * fabs(cmag))) { cmag = 0; }

        double R_flux = fabs(cmag) / (3. * Face_Area_Norm * c_light_eff * (fabs(d_scalar) * All.cf_a3inv) + 1.e-37);
        R_flux = (1. + 12.*R_flux) / (1. + 12.*R_flux*(1. + R_flux));
#ifndef FREEZE_HYDRO
        cmag *= R_flux;
#endif

        double cmag_adv = 0, fluxlim_ij = 1;
        double v_Area_dot_rt  = dot(v_frame, Face_Area_Vec);
        double v_Fluid_dot_rt = dot(0.5 * (local.Vel + VelPred_j) / All.cf_atime, Face_Area_Vec);
        double scalar_ij_phys = 2. * scalar_i * scalar_j / (scalar_i + scalar_j) * All.cf_a3inv;
#ifdef RT_FLUXLIMITER
        double fluxlim_j = CellP[j].flux_limiter(k_freq);
        double fluxlim_i = local.RT_DiffusionCoeff[k_freq] * local.Density * local.Rad_Kappa[k_freq] / c_light_eff;
        fluxlim_ij = 0.5 * (fluxlim_i + fluxlim_j);
#endif
        double fac_fluxlim = rsol_corr * (4./3.) * fluxlim_ij * v_Fluid_dot_rt - v_Area_dot_rt;
#ifdef RT_RADPRESSURE_IN_HYDRO
        fac_fluxlim = rsol_corr * fluxlim_ij * v_Fluid_dot_rt - v_Area_dot_rt
                    + (2. * rt_absorb_frac_albedo(j, k_freq, P, CellP)) * (fluxlim_ij / 3.) * v_Fluid_dot_rt;
#endif
        cmag_adv += scalar_ij_phys * fac_fluxlim;
        cmag_adv += fabs(v_Area_dot_rt) * (scalar_j - scalar_i);
        if(fabs(cmag_adv) > 0) {
            double cmag_max = 0.5 * DMAX(1, 0.5*(tau_c_i[k_freq] + tau_c_j))
                            * (fabs(v_Area_dot_rt / Face_Area_Norm) / c_light_eff) * fabs(cmag);
            if(fabs(cmag_adv) > cmag_max) { cmag_adv *= cmag_max / fabs(cmag_adv); }
            cmag += cmag_adv;
        }

        cmag *= dt_hydrostep;
        if(fabs(cmag) > 0) {
            double thold_hll_y = 0.25 * DMIN(fabs(scalar_i*V_i - scalar_j*V_j),
                                              DMAX(fabs(scalar_i*V_i), fabs(scalar_j*V_j)));
#ifdef RT_ENHANCED_NUMERICAL_DIFFUSION
            thold_hll_y *= 2.0;
#endif
            if(check_for_stability_sign < 0) { thold_hll_y *= 1.e-2; }
            if(fabs(cmag) > thold_hll_y) { cmag *= thold_hll_y / fabs(cmag); }
            cmag /= dt_hydrostep;
            Fluxes_Rad_E_gamma[k_freq] += cmag;
#ifdef RT_INFRARED
            if(k_freq == RT_FREQ_BIN_INFRARED) {
                if(cmag > 0) { Fluxes_Rad_E_gamma_T_weighted_IR = cmag / (MIN_REAL_NUMBER + CellP[j].Radiation_Temperature); }
                else         { Fluxes_Rad_E_gamma_T_weighted_IR = cmag / (MIN_REAL_NUMBER + local.Radiation_Temperature); }
            }
#endif
        }
    }

#else  /* RT_EVOLVE_FLUX: two advection-like equations for E and F */

    Vec3<double> Fluxes_Rad_Flux[N_RT_FREQ_BINS];
    double V_i_phys = V_i / All.cf_a3inv, V_j_phys = V_j / All.cf_a3inv;
#ifdef RT_M1_SECONDORDER
    double dist_rt_i[3], dist_rt_j[3];
    for(k=0; k<3; k++) { dist_rt_i[k] = -0.5 * kernel.dp[k]; dist_rt_j[k] = 0.5 * kernel.dp[k]; }
#endif
    for(int k_freq=0; k_freq<N_RT_FREQ_BINS; k_freq++)
    {
        Fluxes_Rad_E_gamma[k_freq] = 0;
        Fluxes_Rad_Flux[k_freq] = {};
        double scalar_i = local.Rad_E_gamma[k_freq] / V_i_phys;
        double scalar_j = CellP[j].Rad_E_gamma_Pred[k_freq] / V_j_phys;
#ifdef RT_M1_SECONDORDER
        {
            MyFloat grad_E_i[3], grad_E_j[3];
            for(k=0; k<3; k++) {
                grad_E_i[k] = (MyFloat)(local.Gradients.Rad_E_gamma_Grad[k_freq][k] * All.cf_a3inv);
                grad_E_j[k] = (MyFloat)(CellP[j].Gradients.Rad_E_gamma_Grad[k_freq][k] * All.cf_a3inv);
            }
            double scalar_i_face, scalar_j_face;
            reconstruct_face_states(scalar_i, grad_E_i, scalar_j, grad_E_j,
                                    dist_rt_i, dist_rt_j, &scalar_j_face, &scalar_i_face, 1);
            scalar_i = scalar_i_face; scalar_j = scalar_j_face;
        }
#endif
        if(!((scalar_i + scalar_j > 0) && (local.Mass > 0) && (P[j].Mass > 0)
             && (dt_hydrostep > 0) && (Face_Area_Norm > 0))) { continue; }

        double d_scalar = scalar_i - scalar_j;
        double cmag = 0., thold_hll_z;
        Vec3<double> cmag_flux = {};
        Vec3<double> flux_i = local.Rad_Flux[k_freq] / V_i_phys - rsol_corr * v_frame * scalar_i;
        Vec3<double> flux_j = CellP[j].Rad_Flux_Pred[k_freq] / V_j_phys - rsol_corr * v_frame * scalar_j;
        double kappa_i = local.RT_DiffusionCoeff[k_freq];
        double kappa_j = rt_diffusion_coefficient(j, k_freq, CellP);
        double kappa_ij = 0.5 * (kappa_i + kappa_j);

#ifdef RT_M1_SECONDORDER
        double grad_norm = 0, face_dot_flux = 0;
        for(k=0; k<3; k++) {
            double flux_i_raw = local.Rad_Flux[k_freq][k] / V_i_phys;
            double flux_j_raw = CellP[j].Rad_Flux_Pred[k_freq][k] / V_j_phys;
            {
                MyFloat grad_F_i[3], grad_F_j[3];
                for(int k3=0; k3<3; k3++) {
                    grad_F_i[k3] = (MyFloat)(local.Gradients.Rad_Flux_Grad[k_freq][k][k3] * All.cf_a3inv);
                    grad_F_j[k3] = (MyFloat)(CellP[j].Gradients.Rad_Flux_Grad[k_freq][k][k3] * All.cf_a3inv);
                }
                double flux_i_face, flux_j_face;
                reconstruct_face_states(flux_i_raw, grad_F_i, flux_j_raw, grad_F_j,
                                        dist_rt_i, dist_rt_j, &flux_j_face, &flux_i_face, 1);
                flux_i_raw = flux_i_face; flux_j_raw = flux_j_face;
            }
            flux_i[k] = flux_i_raw - rsol_corr * v_frame[k] * scalar_i;
            flux_j[k] = flux_j_raw - rsol_corr * v_frame[k] * scalar_j;
            double grad = 0.5 * (flux_i[k] + flux_j[k]);
            grad_norm += grad * grad;
            face_dot_flux += Face_Area_Vec[k] * grad;
        }
        grad_norm = sqrt(grad_norm) + MIN_REAL_NUMBER;
#else
        Vec3<double> grad_vec = 0.5 * (flux_i + flux_j);
        double grad_norm = sqrt(grad_vec.norm_sq()) + MIN_REAL_NUMBER;
        double face_dot_flux = dot(Face_Area_Vec, grad_vec);
#endif
        double cos_theta_face_flux = face_dot_flux / (Face_Area_Norm * grad_norm);
        if(cos_theta_face_flux < -1)      { cos_theta_face_flux = -1; }
        else if(cos_theta_face_flux > 1)  { cos_theta_face_flux = 1; }
        double hlle_wtfac_u = 0.5, hlle_wtfac_f = 0.5;
        cmag = dot(Face_Area_Vec, hlle_wtfac_f * flux_i + (1. - hlle_wtfac_f) * flux_j);

        Vec3<double> ET_dot_Face_i = local.ET[k_freq].matvec(Face_Area_Vec);
        Vec3<double> ET_dot_Face_j = CellP[j].ET[k_freq].matvec(Face_Area_Vec);
#ifdef RT_ENHANCED_NUMERICAL_DIFFUSION
        cmag_flux += (c_light_eff * c_light_eff * 0.5 * (scalar_i + scalar_j) / 3.) * Face_Area_Vec;
#else
        for(k=0; k<3; k++) {
            cmag_flux[k] += c_light_eff * c_light_eff * (hlle_wtfac_f * scalar_i * ET_dot_Face_i[k]
                                                         + (1. - hlle_wtfac_f) * scalar_j * ET_dot_Face_j[k]);
        }
#endif

        double v_eff_touse = DMIN(c_light_eff, kappa_ij / Particle_Size_j);
        double c_hll = DMIN(0.5 * fabs(face_vel_i - face_vel_j) + DMAX(1., hlle_wtfac_u) * v_eff_touse, c_light_eff);
        double tau_c_j = Particle_Size_j * CellP[j].Rad_Kappa[k_freq] * (CellP[j].Density * All.cf_a3inv);
        double hll_corr = 1. / (1. + 1.5 * DMAX(tau_c_i[k_freq], tau_c_j));
        double q = 0.5 * c_hll * (kernel.r * All.cf_atime) / fabs(MIN_REAL_NUMBER + kappa_ij);
        q = (0.2 + q) / (0.2 + q + q*q);
        double renormerFAC = DMIN(1., fabs(cos_theta_face_flux * cos_theta_face_flux * q * hll_corr));
#ifdef RT_ENHANCED_NUMERICAL_DIFFUSION
        renormerFAC = 1;
#endif
#if !defined(RT_ENHANCED_NUMERICAL_DIFFUSION)
        double RTopticaldepth = DMIN(Particle_Size_i, Particle_Size_j) * c_light_eff / kappa_ij;
        double reductionfactor = sqrt((1.0 - exp(-RTopticaldepth * RTopticaldepth))) / RTopticaldepth;
        double reducedcM1 = reductionfactor * c_light_eff / sqrt(3.0);
        v_eff_touse = DMAX(reducedcM1, 2. * kernel.vsig);
        c_hll = DMIN(0.5 * fabs(face_vel_i - face_vel_j) + v_eff_touse, c_light_eff);
        hll_corr = 1. / (1. + 1.5 * v_eff_touse * DMAX(Particle_Size_j / kappa_j, Particle_Size_i / kappa_i));
        q = 0.5 * c_hll * (kernel.r * All.cf_atime) / fabs(MIN_REAL_NUMBER + kappa_ij);
        q = (0.2 + q) / (0.2 + q + q*q);
        renormerFAC = DMIN(1., fabs(cos_theta_face_flux * cos_theta_face_flux * q * hll_corr));
#ifndef RT_M1_SECONDORDER
        double scalar_jr = scalar_j, scalar_ir = scalar_i, d_scalar_hll = d_scalar, d_scalar_ij = 0;
        scalar_jr += 0.5 * dot(kernel.dp, local.Gradients.Rad_E_gamma_ET[k_freq]) * All.cf_a3inv;
        scalar_ir -= 0.5 * dot(kernel.dp, CellP[j].Gradients.Rad_E_gamma_ET[k_freq]) * All.cf_a3inv;
        d_scalar_ij = scalar_ir - scalar_jr;
        if((d_scalar_ij * d_scalar > 0) && (fabs(d_scalar_ij) < fabs(d_scalar))) { d_scalar_hll = d_scalar_ij; }
        d_scalar = d_scalar_hll;
#endif
#endif

        double f_direct = -Face_Area_Norm * c_hll * d_scalar * renormerFAC;
        if(dt_hydrostep > 0) {
            double f_direct_max = 0.25 * fabs(d_scalar) * DMIN(V_i_phys, V_j_phys) / dt_hydrostep;
            if(fabs(f_direct) > f_direct_max) { f_direct *= f_direct_max / fabs(f_direct); }
        } else { f_direct = 0; }
        double sign_c0 = f_direct * cmag;
        if((sign_c0 < 0) && (fabs(f_direct) > fabs(cmag))) { cmag = 0; }
#ifdef RT_ENHANCED_NUMERICAL_DIFFUSION
        if(cmag != 0)
#endif
        {
            if(f_direct != 0) {
                thold_hll_z = (0.5 * hlle_wtfac_u) * fabs(cmag);
                if(fabs(f_direct) > thold_hll_z) { f_direct *= thold_hll_z / fabs(f_direct); }
                cmag += f_direct;
            }
            cmag *= dt_hydrostep;
            double sVi = scalar_i * V_i_phys, sVj = scalar_j * V_j_phys;
            thold_hll_z = 0.25 * DMAX(fabs(sVi - sVj), DMAX(fabs(sVi), fabs(sVj)));
#ifdef RT_ENHANCED_NUMERICAL_DIFFUSION
            thold_hll_z *= 2.0;
#ifdef SINK_WIND_SPAWN
            if(local.ConditionNumber < 0 || P[j].ID == All.SpawnedWindCellID) { thold_hll_z *= 0.25; }
#endif
#endif
            if(sign_c0 < 0) { thold_hll_z *= 1.e-2; }
            if(fabs(cmag) > thold_hll_z) { cmag *= thold_hll_z / fabs(cmag); }
            cmag /= dt_hydrostep;
            Fluxes_Rad_E_gamma[k_freq] += cmag;
#ifdef RT_INFRARED
            if(k_freq == RT_FREQ_BIN_INFRARED) {
                if(cmag > 0) { Fluxes_Rad_E_gamma_T_weighted_IR = cmag / (MIN_REAL_NUMBER + CellP[j].Radiation_Temperature); }
                else         { Fluxes_Rad_E_gamma_T_weighted_IR = cmag / (MIN_REAL_NUMBER + local.Radiation_Temperature); }
            }
#endif
        }

        double hll_mult_dmin = 1;
        for(k=0; k<3; k++) {
            double f_direct_k = -Face_Area_Norm * c_hll * (flux_i[k] - flux_j[k]);
            if(f_direct_k != 0) {
                double thold_fd = 1.0 * fabs(cmag_flux[k]) / fabs(f_direct_k);
                if(thold_fd < hll_mult_dmin) { hll_mult_dmin = thold_fd; }
            }
        }

        double v_Area_dot_rt = rsol_corr * dot(v_frame, Face_Area_Vec);
        cmag_flux += -v_Area_dot_rt * 0.5 * (flux_i + flux_j) + (0.5 * fabs(v_Area_dot_rt)) * (flux_j - flux_i);

        for(k=0; k<3; k++) {
            double f_direct_k = -Face_Area_Norm * c_hll * (flux_i[k] - flux_j[k]);
            double sign_agreement = f_direct_k * cmag_flux[k];
            if((sign_agreement < 0) && (fabs(f_direct_k) > fabs(cmag_flux[k]))) { cmag_flux[k] = 0; }
            if(cmag_flux[k] != 0) {
                cmag_flux[k] += hll_mult_dmin * f_direct_k;
                cmag_flux[k] *= dt_hydrostep;
                thold_hll_z = DMIN(DMAX(DMIN(fabs(local.Rad_Flux[k_freq][k]), fabs(CellP[j].Rad_Flux_Pred[k_freq][k])),
                                         fabs(local.Rad_Flux[k_freq][k] - CellP[j].Rad_Flux_Pred[k_freq][k])),
                                    DMAX(fabs(local.Rad_Flux[k_freq][k]), fabs(CellP[j].Rad_Flux_Pred[k_freq][k])));
                double fii = V_i_phys * scalar_i * c_light_eff;
                double fjj = V_j_phys * scalar_j * c_light_eff;
                double tij = DMIN(DMAX(DMIN(fabs(fii), fabs(fjj)), fabs(fii - fjj)), DMAX(fabs(fii), fabs(fjj)));
                thold_hll_z = 0.25 * DMAX(thold_hll_z, 0.5 * tij);
                if(fabs(cmag_flux[k]) > thold_hll_z) { cmag_flux[k] *= thold_hll_z / fabs(cmag_flux[k]); }
                Fluxes_Rad_Flux[k_freq][k] += cmag_flux[k] / dt_hydrostep;
            }
        }
    }
#endif /* RT_EVOLVE_FLUX */

    for(k=0; k<N_RT_FREQ_BINS; k++) { out.Dt_Rad_E_gamma[k] += Fluxes_Rad_E_gamma[k]; }
#if defined(RT_INFRARED)
    out.Dt_Rad_E_gamma_T_weighted_IR += Fluxes_Rad_E_gamma_T_weighted_IR;
#endif
#ifdef RT_EVOLVE_FLUX
    for(k=0; k<N_RT_FREQ_BINS; k++) { out.Dt_Rad_Flux[k] += Fluxes_Rad_Flux[k]; }
#endif
#else
    (void)local; (void)j; (void)P; (void)CellP; (void)VelPred_j; (void)ParticleVel_j;
    (void)kernel; (void)rinv; (void)Face_Area_Vec; (void)Face_Area_Norm;
    (void)V_i; (void)V_j; (void)Particle_Size_i; (void)Particle_Size_j;
    (void)face_vel_i; (void)face_vel_j; (void)tau_c_i; (void)dt_hydrostep; (void)out;
#endif /* RT_SOLVER_EXPLICIT && !RT_EVOLVE_INTENSITIES */
}

#endif /* RT_DIFFUSION_EXPLICIT_FUNCTIONS_H */
