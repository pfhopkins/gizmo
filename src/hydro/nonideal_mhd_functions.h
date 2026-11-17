/* nonideal_mhd_functions.h -- per-pair non-ideal MHD (ohmic / Hall / ambipolar
 * diffusion) flux. Replaces the fragment hydro/nonideal_mhd.h.
 *
 * Body guarded by MHD_NON_IDEAL. Pure i-accumulation into Fluxes.B.
 *
 * Cosmological units (commit 9/N): predict.cc:268 requires anything added to
 * Fluxes.B/DtB to be in physical d(B*V)/dt (the integrator does the
 * conversion via *dt/cf_atime to get the code-unit BPred update). The native
 * scaling of this routine is cf_atime^3 too large vs physical:
 *   B_code = a^2 * B_phys                   (cf. bhat = (B_i+B_j)/2 * cf_a2inv)
 *   Bfield() differences  d_scalar  ~ a^2 * dB_phys
 *   gradient(Bfield) in code-coord  ~ a^3 * gradB_phys      (extra factor of
 *     a from x_code = x_phys/a)
 *   J_current, J_direct, b_flux, db_direct, db_direct_diff:  all  ~ a^3 * physical
 *   hll_correction_fn(d_scalar_hll, ...) output:  ~ a^2 * physical (because
 *     d_scalar_hll comes from Bfield differences, NOT gradients).
 * Internal consistency requires a uniform a-power throughout the limiter; we
 * lift hll_corr from a^2 to a^3 with one cf_atime, then strip the uniform
 * a^3 from the final OUT-param with one cf_a3inv. cf_atime=1 bit-identical.
 *
 * Requires allvars.h, kernel.h, hydro_structs.h, hydro_pair_types.h.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef NONIDEAL_MHD_FUNCTIONS_H
#define NONIDEAL_MHD_FUNCTIONS_H

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif
#include "hydro_structs.h"
#include "hydro_pair_types.h"

/* Shared linear-algebra helper for assembling the comoving electric field E'
 * (or, in code units, the b_flux) from a current J and its decomposition
 * along/perpendicular to B^:
 *   b_flux = -coef_O J  -  coef_H (J x B^)  -  coef_A ((J x B^) x B^)
 * where coef_{O,H,A} are Ohmic, Hall, and ambipolar coefficients respectively.
 *
 * Used by:
 *   - nonideal_mhd_compute_pair (this file): coef = eta = (c^2/4pi)*alpha
 *     with J = curl B (the standard MHD current).
 *   - dust battery (solids/dust_battery_functions.h): builds E_dust per-cell;
 *     gradient pass takes ∇E_dust; curl applied in hydro_toplevel.cc.
 *     = (c^2/4pi)*alpha_dust with J = J_d (dust current).
 *
 * The structural identity between non-ideal MHD and the dust battery is the
 * key insight from Soliman, Hopkins & Squire 2025 (Eq. 9): both produce the
 * same E' form, just with different (coef, J) pairs. */
KOKKOS_INLINE_FUNCTION
Vec3<double> nonideal_mhd_assemble_bflux(
    double coef_O, double coef_H, double coef_A,
    const Vec3<double>& J, const Vec3<double>& bhat)
{
    Vec3<double> b_flux = {};
    Vec3<double> JcrossB = cross(J, bhat);
    Vec3<double> JcrossBcrossB = cross(JcrossB, bhat);
    if(fabs(coef_O) > 0) { b_flux += -coef_O * J; }
    if(fabs(coef_A) > 0) { b_flux +=  coef_A * JcrossBcrossB; }
    if(fabs(coef_H) > 0) { b_flux += -coef_H * JcrossB; }
    return b_flux;
}

/* Component k of the current from a B-gradient tensor g (g[a][b] = d B_a / d x_b) is g[k_xyz_B][k_xyz_A] - g[k_xyz_A][k_xyz_B]. */
KOKKOS_INLINE_FUNCTION
void nonideal_mhd_curl_axes(int k, int &k_xyz_A, int &k_xyz_B) { k_xyz_A = (k+2)%3; k_xyz_B = (k+1)%3; }

#ifdef MHD_NON_IDEAL_EVOLVED_FLUX
/* Inverse of nonideal_mhd_assemble_bflux for given coefficients and unit B^: the current J that the coefficients map onto b_flux.
   Along B^ only the Ohmic term acts, J = -b/coef_O; across it the Ohmic+ambipolar terms scale and the Hall term rotates, so
   J = -[a b - coef_H (b x B^)] / (a^2 + coef_H^2) with a = coef_O + coef_A. A part the coefficients cannot produce (coef_O = 0
   along B^, a = coef_H = 0 across it, or a quotient that is not representable) is dropped. With B^ = 0 only the Ohmic term exists. */
KOKKOS_INLINE_FUNCTION
Vec3<double> nonideal_mhd_current_from_bflux(
    double coef_O, double coef_H, double coef_A,
    const Vec3<double>& b_flux, const Vec3<double>& bhat)
{
    Vec3<double> J = {};
    if(bhat.norm_sq() <= 0) { if(coef_O != 0) { J = (-1./coef_O) * b_flux; } }
    else {
        double b_par = dot(b_flux, bhat);
        Vec3<double> b_perp = b_flux - b_par * bhat;
        if(coef_O != 0) { J += (-b_par/coef_O) * bhat; }
        double a = coef_O + coef_A, d = a*a + coef_H*coef_H;
        if(d > 0) { J += (-1./d) * (a * b_perp - coef_H * cross(b_perp, bhat)); }
    }
    for(int k=0; k<3; k++) { if(!isfinite(J[k])) { return {}; } }
    return J;
}

/* Exact update over dt of the evolved non-ideal field F (stored as the b_flux it produces, physical units), with the cell's
   coefficients, B^ and current held fixed over the step:
     (eta_NI / c~^2) dF/dt = -M (F - F_eq),  eta_NI^2 = (eta_O+eta_A)^2 + eta_H^2,
   where F_eq is the standard non-ideal b_flux of the cell's own current. The deviation F - F_eq decays at c~^2 dt/eta_O along B^
   and at c~^2 dt (eta_O+eta_A)/eta_NI^2 across it, while the Hall term rotates it by c~^2 dt eta_H/eta_NI^2 about B^; its magnitude
   never grows, so no step limit is needed. mode 0 updates NonIdealMHDFlux (kick), mode 1 NonIdealMHDFluxPred (drift). */
KOKKOS_INLINE_FUNCTION
void nonideal_mhd_flux_update_driftkick(int i, double dt_entr, int mode, struct particle_data *pp, struct gas_cell_data *cell)
{
    if(dt_entr <= 0) { return; }
    double eta_O = cell[i].Eta_MHD_OhmicResistivity_Coeff, eta_H = cell[i].Eta_MHD_HallEffect_Coeff, eta_A = cell[i].Eta_MHD_AmbiPolarDiffusion_Coeff;
    double eta_NI2 = (eta_O + eta_A)*(eta_O + eta_A) + eta_H*eta_H;
    /* a cell that has not completed a step since the start may not have had its coefficients computed yet (with cooling they are
       zero until then): keep the field read back from a snapshot rather than clearing it */
    if((eta_NI2 <= 0) && (pp[i].Ti_begstep == 0)) { return; }
    Vec3<double> F_new = {};
    if(eta_NI2 > 0)
    {
        Vec3<double> F = (mode == 0) ? Vec3<double>(cell[i].NonIdealMHDFlux) : Vec3<double>(cell[i].NonIdealMHDFluxPred);
        Vec3<double> bhat = (mode == 0) ? Vec3<double>(cell[i].B) : Vec3<double>(cell[i].BPred);
        double bmag2 = bhat.norm_sq(); if(bmag2 > 0) { bhat /= sqrt(bmag2); }
        Vec3<double> J = {}; /* same component formula as the face current in nonideal_mhd_compute_pair, in physical units */
        for(int k=0; k<3; k++) {
            int k_xyz_A, k_xyz_B; nonideal_mhd_curl_axes(k, k_xyz_A, k_xyz_B);
            J[k] = (cell[i].Gradients.B[k_xyz_B][k_xyz_A] - cell[i].Gradients.B[k_xyz_A][k_xyz_B]) * All.cf_a3inv;
        }
        Vec3<double> F_eq = nonideal_mhd_assemble_bflux(eta_O, eta_H, eta_A, J, bhat);
        Vec3<double> dF = F - F_eq;
        double c2dt = MHD_NON_IDEAL_REDUCED_C_CODE * MHD_NON_IDEAL_REDUCED_C_CODE * dt_entr;
        /* written as increments of F (expm1, and 1-cos = 2 sin^2) so a small step does not lose its update to cancellation */
        double decay_par = (eta_O != 0) ? expm1(-c2dt/fabs(eta_O)) : -1.; /* along B^ only the Ohmic term damps; none -> fully relaxed */
        if(bmag2 <= 0) { /* no field: only the Ohmic term exists */
            F_new = F + decay_par * dF;
        } else {
            double dF_par = dot(dF, bhat);
            Vec3<double> dF_perp = dF - dF_par * bhat;
            double lam_perp = c2dt * (eta_O + eta_A) / eta_NI2, angle = c2dt * eta_H / eta_NI2;
            double perp_scale = expm1(-lam_perp), perp_rotate = 0; /* no Hall term: no rotation */
            if(angle != 0) { double sin_half = sin(0.5*angle); perp_scale = perp_scale * cos(angle) - 2.*sin_half*sin_half; perp_rotate = exp(-lam_perp) * sin(angle); }
            F_new = F + (decay_par * dF_par) * bhat + perp_scale * dF_perp + perp_rotate * cross(dF_perp, bhat);
        }
    }
    if(mode == 0) { cell[i].NonIdealMHDFlux = F_new; } else { cell[i].NonIdealMHDFluxPred = F_new; }
}
#endif

KOKKOS_INLINE_FUNCTION
void nonideal_mhd_compute_pair(
    const struct hydro_data_in &local,
    const struct particle_data &Pj,
    const struct gas_cell_data &CPj,
    const Vec3<double> &BPred_j,
    const struct kernel_hydra &kernel,
    double rinv,
    const Vec3<double> &Face_Area_Vec,
    double Face_Area_Norm,
    double v_hll,
    const Vec3<double> &bhat,
    double bhat_mag,
    double dt_hydrostep,
    struct Conserved_var_Riemann &Fluxes,
    Vec3<double> &bflux_from_nonideal_effects)
{
#ifdef MHD_NON_IDEAL
    (void)bhat_mag;  /* bhat_mag unused here — bhat already carries magnitude */
    if(!((local.Mass > 0) && (Pj.Mass > 0) && (dt_hydrostep > 0))) { return; }

    double eta_i, eta_j, eta_ohmic, eta_hall, eta_ad;
    eta_i = local.Eta_MHD_OhmicResistivity_Coeff; eta_j = CPj.Eta_MHD_OhmicResistivity_Coeff;
    if(eta_i*eta_j > 0) { eta_ohmic = 2*eta_i*eta_j / (eta_i+eta_j); } else { eta_ohmic = 0; }
    eta_i = local.Eta_MHD_HallEffect_Coeff; eta_j = CPj.Eta_MHD_HallEffect_Coeff;
    if(eta_i*eta_j > 0) { eta_hall = 2*eta_i*eta_j / (eta_i+eta_j); } else { eta_hall = 0; }
    eta_i = local.Eta_MHD_AmbiPolarDiffusion_Coeff; eta_j = CPj.Eta_MHD_AmbiPolarDiffusion_Coeff;
    if(eta_i*eta_j > 0) { eta_ad = 2*eta_i*eta_j / (eta_i+eta_j); } else { eta_ad = 0; }

    double eta_max = DMAX(fabs(eta_ohmic), DMAX(fabs(eta_hall), fabs(eta_ad)));
    if(eta_max <= MIN_REAL_NUMBER) { return; }

    Vec3<double> J_current, d_scalar;
    double rinv2 = rinv * rinv;
    Vec3<double> J_direct = {}, grad_dot_x_ij = {};
#ifdef MHD_NON_IDEAL_EVOLVED_FLUX
    /* evolved non-ideal field: average the two cells' fields at the face and express the result as the current the face
       coefficients would map onto it, then run that current through the same limiters as the standard solver below.
       The stored fields are physical; the current here carries the same cf_atime^3 as the gradient-based one. */
    Vec3<double> b_flux_face = 0.5 * (Vec3<double>(local.NonIdealMHDFlux) + Vec3<double>(CPj.NonIdealMHDFluxPred));
    Vec3<double> J_evolved = nonideal_mhd_current_from_bflux(eta_ohmic, eta_hall, eta_ad, b_flux_face, bhat) / All.cf_a3inv;
#endif
    for(int k=0; k<3; k++) { d_scalar[k] = local.BPred[k] - BPred_j[k]; }
    for(int k=0; k<3; k++) {
        int k_xyz_A, k_xyz_B; nonideal_mhd_curl_axes(k, k_xyz_A, k_xyz_B);
#ifdef MHD_NON_IDEAL_EVOLVED_FLUX
        J_current[k] = J_evolved[k];
#else
        double tmp_grad_A = 0.5 * (local.Gradients.B[k_xyz_A][k_xyz_B] + CPj.Gradients.B[k_xyz_A][k_xyz_B]);
        double tmp_grad_B = 0.5 * (local.Gradients.B[k_xyz_B][k_xyz_A] + CPj.Gradients.B[k_xyz_B][k_xyz_A]);
        J_current[k] = tmp_grad_B - tmp_grad_A;
#endif
        J_direct[k]  = rinv2 * (kernel.dp[k_xyz_A]*d_scalar[k_xyz_B] - kernel.dp[k_xyz_B]*d_scalar[k_xyz_A]);
        if(J_current[k]*J_direct[k] < 0) { if(fabs(J_direct[k]) > 5.*fabs(J_current[k])) { J_current[k] = 0.0; } }
        for(int k2=0; k2<3; k2++) {
            grad_dot_x_ij[k] += 0.5*(local.Gradients.B[k][k2] + CPj.Gradients.B[k][k2]) * kernel.dp[k2];
        }
    }
    double Jmag = J_current.norm_sq();

    Vec3<double> b_flux = nonideal_mhd_assemble_bflux(eta_ohmic, eta_hall, eta_ad, J_current, bhat);
    bflux_from_nonideal_effects = cross(Face_Area_Vec, b_flux);

    double eta_0 = v_hll * kernel.r * All.cf_atime;
    double eta_ohmic_0 = 0, eta_ad_0 = 0, eta_hall_0 = 0, q = 0;
    if(fabs(eta_ohmic) > 0) { q = eta_0/fabs(eta_ohmic); eta_ohmic_0 = eta_0*(0.2 + q)/(0.2 + q + q*q); }
    if(fabs(eta_ad)    > 0) { q = eta_0/fabs(eta_ad);    eta_ad_0    = eta_0*(0.2 + q)/(0.2 + q + q*q); }
    if(fabs(eta_hall)  > 0) { q = eta_0/fabs(eta_hall);  eta_hall_0  = eta_0*(0.2 + q)/(0.2 + q + q*q); if(eta_hall < 0) { eta_hall_0 *= -1; } }
    Vec3<double> b_flux_direct = nonideal_mhd_assemble_bflux(eta_ohmic_0, eta_hall_0, eta_ad_0, J_direct, bhat);
    Vec3<double> db_direct = cross(Face_Area_Vec, b_flux_direct);

    double db_dot_direct_diff = 0;
    double F_ddiff_prefac = -Face_Area_Norm * rinv * DMAX(eta_ohmic, fabs(eta_hall) + eta_ad);
    Vec3<double> db_direct_diff = F_ddiff_prefac * d_scalar;

    double bfluxmag = bflux_from_nonideal_effects.norm_sq();
    bfluxmag /= (1.e-37 + Jmag * eta_max*eta_max * Face_Area_Norm*Face_Area_Norm);

    for(int k=0; k<3; k++) {
        double d_scalar_tmp = d_scalar[k] - grad_dot_x_ij[k];
        double d_scalar_hll = MINMOD(d_scalar[k], d_scalar_tmp);
        /* hll_correction_fn output is built from d_scalar_hll which carries
           cf_atime^2 (Bfield difference), but everything else in this loop
           (b_flux, db_direct, db_direct_diff) carries cf_atime^3. Lift to
           cf_atime^3 here so the +/MINMOD/sign-check operations below compare
           like-with-like. The final cf_a3inv at the end of the function then
           strips the uniform cf_atime^3 from all of them simultaneously. */
        double hll_corr = bfluxmag * hll_correction_fn(d_scalar_hll, 0., 1., eta_max,
                                                       v_hll, Face_Area_Norm, kernel.r, All.cf_atime)
                          * All.cf_atime;
        double db_corr = bflux_from_nonideal_effects[k] + hll_corr;
        bflux_from_nonideal_effects[k] = MINMOD(1.1 * bflux_from_nonideal_effects[k], db_corr);
        if(hll_corr != 0) {
            double db_direct_tmp = fabs(db_direct[k]) * hll_corr / fabs(hll_corr);
            if((bflux_from_nonideal_effects[k] * db_direct_tmp < 0)
               && (fabs(db_direct_tmp) > 1.0*fabs(bflux_from_nonideal_effects[k]))) { bflux_from_nonideal_effects[k] = 0; }
        }
        if((bflux_from_nonideal_effects[k] * db_direct[k] < 0)
           && (fabs(db_direct[k]) > 10.0 * fabs(bflux_from_nonideal_effects[k]))) { bflux_from_nonideal_effects[k] = 0; }

        db_dot_direct_diff += bflux_from_nonideal_effects[k] * db_direct_diff[k];
    }

#ifdef MHD_NON_IDEAL_EVOLVED_FLUX
    /* the anti-diffusion projection below guards against dissipative flux running up the field gradient. The Hall flux is not
       dissipative, and the evolved field legitimately lags the instantaneous current (finite response at the reduced speed of
       light), which turns part of the large Hall flux along the gradient: the projection must not read that as anti-diffusion.
       So split the limited flux into its Hall and Ohmic+ambipolar parts (each component scaled as the limiters above scaled the
       total), project only the Ohmic+ambipolar part, and add the Hall part back. Without a Hall term this is the standard projection. */
    Vec3<double> bflux_hall_limited = {};
    if(eta_hall != 0) {
        Vec3<double> bflux_raw = cross(Face_Area_Vec, b_flux);
        Vec3<double> bflux_hall_raw = cross(Face_Area_Vec, nonideal_mhd_assemble_bflux(0., eta_hall, 0., J_current, bhat));
        for(int k=0; k<3; k++) { if(bflux_raw[k] != 0) { bflux_hall_limited[k] = (bflux_from_nonideal_effects[k] / bflux_raw[k]) * bflux_hall_raw[k]; } }
        bflux_from_nonideal_effects -= bflux_hall_limited;
        db_dot_direct_diff = dot(bflux_from_nonideal_effects, db_direct_diff);
    }
#endif
    if(db_dot_direct_diff < 0) {
        double db_direct_diff_mag2 = db_direct_diff.norm_sq();
        bflux_from_nonideal_effects -= (db_dot_direct_diff / db_direct_diff_mag2) * db_direct_diff;
    }
#ifdef MHD_NON_IDEAL_EVOLVED_FLUX
    if(eta_hall != 0) { bflux_from_nonideal_effects += bflux_hall_limited; }
#endif

    /* Cosmological units (commit 9/N): all flux contributions inside this
       block are uniformly cf_atime^3 too large vs physical d(B*V)/dt. Strip
       the factor here so Fluxes.B and the OUT param bflux_from_nonideal_effects
       (consumed by the energy hook in hydro_functions.h) are both in physical
       units, matching the integrator convention asserted at predict.cc:268.
       cf_atime=1: bit-identical (cf_a3inv=1). */
    bflux_from_nonideal_effects *= All.cf_a3inv;
    Fluxes.B += bflux_from_nonideal_effects;
#endif /* MHD_NON_IDEAL */

    /* Battery sources (Biermann, radiative-ionization, dust) are NOT applied
       here. They are evaluated as cell-centered source terms in
       hydro_toplevel.cc::out2particle_hydra after the pair loop closes;
       face-flux/curl-of-EMF evaluation cannot be made discretely div-B
       preserving on an unstructured/meshless mesh (cf. AREPO's cell-centered
       choice; see Garaldi+2021 §2.2). */
}

#endif /* NONIDEAL_MHD_FUNCTIONS_H */
