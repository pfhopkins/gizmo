/* hydro_core_sph_functions.h -- SPH equation-of-motion per-pair flux.
 *
 * Replaces the #include-fragment hydro/hydro_core_sph.h. Body guarded by
 * HYDRO_SPH so callers can invoke unconditionally.
 *
 * Includes: SPH pressure term (density-entropy or pressure-entropy form,
 * with Dedner phi update if applicable), ideal-MHD acceleration and induction
 * (including the TP12 artificial resistivity), Monaghan artificial viscosity,
 * and optional artificial conductivity.
 *
 * Mutates kernel.dwk_ij (sets it to the symmetric mean); outputs magfluxv and
 * resistivity_heatflux for the caller to fold into Fluxes.p downstream.
 *
 * Requires allvars.h, kernel.h, hydro_structs.h, reimann.h, hydro_pair_types.h.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef HYDRO_CORE_SPH_FUNCTIONS_H
#define HYDRO_CORE_SPH_FUNCTIONS_H

#include "hydro_pair_types.h"

KOKKOS_INLINE_FUNCTION
void hydro_core_sph_compute_pair(
    const struct hydro_data_in &local,
    int j,
    struct particle_data *P,
    struct gas_cell_data *CellP,
    const Vec3<MyDouble> &VelPred_j,
    const Vec3<double> &BPred_j,
    double PhiPred_j,
    struct kernel_hydra &kernel,
    double r2,
    double V_j,
    double magneticspeed_i, double magneticspeed_j,
    double fac_mu, double fac_vsic_fix,
    double tensile_correction_factor,
    double dt_hydrostep,
    Vec3<double> &magfluxv,
    double &resistivity_heatflux,
    struct Conserved_var_Riemann &Fluxes)
{
#ifdef HYDRO_SPH
    (void)V_j; /* kept in the signature for symmetry with hydro_core_meshless; not used here */

    int k;
    Fluxes.rho = Fluxes.p = Fluxes.v[0] = Fluxes.v[1] = Fluxes.v[2] = 0;
    kernel.dwk_ij = 0.5 * (kernel.dwk_i + kernel.dwk_j);
    double vdotr2_phys = kernel.vdotr2;
    if(All.ComovingIntegrationOn) { vdotr2_phys -= All.cf_hubble_a2 * r2; }
#ifdef COSMIC_RAY_FLUID
    for(k=0; k<N_CR_PARTICLE_BINS; k++) {
        Fluxes.CosmicRayPressure[k] = 0;
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
        Fluxes.CosmicRayAlfvenEnergy[k][0] = Fluxes.CosmicRayAlfvenEnergy[k][1] = 0;
#endif
    }
#endif

    double vi_dot_r, hfc, hfc_visc, hfc_i, hfc_j, hfc_dwk_i, hfc_dwk_j;
    double hfc_egy = 0, wt_corr_i = 1, wt_corr_j = 1;

#if defined(EOS_TILLOTSON) || defined(EOS_ELASTIC) || defined(EOS_ANEOS)
    if(local.Pressure < 0) { wt_corr_i = 1. - tensile_correction_factor; }
    if(CellP[j].Pressure < 0) { wt_corr_j = 1. - tensile_correction_factor; }
#endif

#ifdef HYDRO_PRESSURE_SPH
    double p_over_rho2_j = CellP[j].Pressure / (CellP[j].EgyWtDensity * CellP[j].EgyWtDensity);
    hfc_i = kernel.p_over_rho2_i * (CellP[j].InternalEnergyPred / local.InternalEnergyPred)
          * (1 + local.DrkernHydroSumFactor / (P[j].Mass * CellP[j].InternalEnergyPred));
    hfc_j = p_over_rho2_j * (local.InternalEnergyPred / CellP[j].InternalEnergyPred)
          * (1 + CellP[j].DrkernHydroSumFactor / (local.Mass * local.InternalEnergyPred));
#else
    double p_over_rho2_j = CellP[j].Pressure / (CellP[j].Density * CellP[j].Density);
    hfc_i = kernel.p_over_rho2_i * (1 + local.DrkernHydroSumFactor / P[j].Mass);
    hfc_j = p_over_rho2_j     * (1 + CellP[j].DrkernHydroSumFactor / local.Mass);
#endif
    hfc_i *= wt_corr_i; hfc_j *= wt_corr_j;
    hfc_egy = hfc_i;

    hfc_dwk_i = hfc_dwk_j = local.Mass * P[j].Mass / kernel.r;
    hfc_dwk_i *= kernel.dwk_i;
    hfc_dwk_j *= kernel.dwk_j;
    hfc = hfc_i * hfc_dwk_i + hfc_j * hfc_dwk_j;

    Fluxes.v[0] += -hfc * kernel.dp[0];
    Fluxes.v[1] += -hfc * kernel.dp[1];
    Fluxes.v[2] += -hfc * kernel.dp[2];
    vi_dot_r = dot(local.Vel, kernel.dp);
    Fluxes.p += hfc_egy * hfc_dwk_i * vdotr2_phys - hfc * vi_dot_r;

#ifdef MAGNETIC
    Fluxes.B[0] = Fluxes.B[1] = Fluxes.B[2] = 0;
    double mj_r = P[j].Mass / kernel.r;
    double mf_i = kernel.mf_i * mj_r * kernel.dwk_i;
    double mf_j = kernel.mf_j * mj_r * kernel.dwk_j / (CellP[j].Density * CellP[j].Density);
#ifndef HYDRO_PRESSURE_SPH
    mf_i *= (1 + local.DrkernHydroSumFactor / P[j].Mass);
    mf_j *= (1 + CellP[j].DrkernHydroSumFactor / local.Mass);
#endif

#ifdef DIVBCLEANING_DEDNER
    double phifac = -(mf_i * local.PhiPred + mf_j * PhiPred_j);
    Fluxes.B[0] += phifac * kernel.dp[0];
    Fluxes.B[1] += phifac * kernel.dp[1];
    Fluxes.B[2] += phifac * kernel.dp[2];
#else
    (void)PhiPred_j;
#endif

    Vec3<double> dB = {local.BPred[0] - BPred_j[0], local.BPred[1] - BPred_j[1], local.BPred[2] - BPred_j[2]};

    double mm_i[3][3], mm_j[3][3];
    for(k=0; k<3; k++) {
        for(int k1=0; k1<3; k1++) {
            mm_i[k][k1] = local.BPred[k] * local.BPred[k1];
            mm_j[k][k1] = BPred_j[k]     * BPred_j[k1];
        }
    }
    for(k=0; k<3; k++) {
        mm_i[k][k] -= 0.5 * kernel.b2_i;
        mm_j[k][k] -= 0.5 * kernel.b2_j;
    }

    Fluxes.B_normal_corrected = ((local.BPred[0] * mf_i + BPred_j[0] * mf_j) * kernel.dp[0]
                               + (local.BPred[1] * mf_i + BPred_j[1] * mf_j) * kernel.dp[1]
                               + (local.BPred[2] * mf_i + BPred_j[2] * mf_j) * kernel.dp[2]) * All.cf_atime;
    for(k=0; k<3; k++) {
        magfluxv[k] = ((mm_i[k][0] * mf_i + mm_j[k][0] * mf_j) * kernel.dp[0]
                     + (mm_i[k][1] * mf_i + mm_j[k][1] * mf_j) * kernel.dp[1]
                     + (mm_i[k][2] * mf_i + mm_j[k][2] * mf_j) * kernel.dp[2]) / All.cf_atime;
        Fluxes.v[k] += magfluxv[k] * All.cf_atime;
    }

#ifdef SPH_TP12_ARTIFICIAL_RESISTIVITY
    {
        double mf_dissInd = local.Mass * mj_r * kernel.dwk_ij * kernel.rho_ij_inv * kernel.rho_ij_inv / fac_mu;
        double vsigb = 0.5 * (sqrt(kernel.alfven2_i) + sqrt(kernel.alfven2_j));
        double eta = 0.5 * (local.Balpha + CellP[j].Balpha) * vsigb * kernel.r;
        mf_dissInd *= eta;
        Fluxes.B += mf_dissInd / All.cf_atime * dB;
        resistivity_heatflux = -0.5 * mf_dissInd * dB.norm_sq();
    }
#else
    resistivity_heatflux = 0;
#endif
#else  /* !MAGNETIC */
    (void)BPred_j; (void)PhiPred_j;
    magfluxv = {};
    resistivity_heatflux = 0;
#endif /* MAGNETIC */

    if(kernel.vdotr2 < 0) {
        double c_ij = 0.5 * (kernel.sound_i + kernel.sound_j);
#ifdef MAGNETIC
        c_ij = 0.5 * (magneticspeed_i + magneticspeed_j);
#else
        (void)magneticspeed_i; (void)magneticspeed_j;
#endif
#if defined(SPHAV_CD10_VISCOSITY_SWITCH)
        double BulkVisc_ij = 0.5 * (local.alpha + CellP[j].alpha_limiter * CellP[j].alpha);
        double mu_ij = fac_mu * kernel.vdotr2 / kernel.r;
        double visc = -BulkVisc_ij * mu_ij * (c_ij - mu_ij) * kernel.rho_ij_inv;
#else
        double BulkVisc_ij = 0.5 * All.ArtBulkViscConst;
#if !defined(EOS_ELASTIC)
        BulkVisc_ij *= (local.alpha + CellP[j].alpha_limiter);
#endif
        double h_ij = KERNEL_CORE_SIZE * 0.5 * (kernel.h_i + kernel.h_j);
        double mu_ij = fac_mu * h_ij * kernel.vdotr2 / (r2 + 0.0001 * h_ij * h_ij);
        double visc = -BulkVisc_ij * mu_ij * (c_ij - 2 * mu_ij) * kernel.rho_ij_inv;
#endif
#ifndef NOVISCOSITYLIMITER
        if(dt_hydrostep > 0 && kernel.dwk_ij < 0) {
            visc = DMIN(visc, 0.5 * fac_vsic_fix * kernel.vdotr2
                              / ((local.Mass + P[j].Mass) * kernel.dwk_ij * kernel.r * (2. * dt_hydrostep)));
        }
#endif
        hfc_visc = -local.Mass * P[j].Mass * visc * kernel.dwk_ij / kernel.r;
        Fluxes.v[0] += hfc_visc * kernel.dp[0];
        Fluxes.v[1] += hfc_visc * kernel.dp[1];
        Fluxes.v[2] += hfc_visc * kernel.dp[2];
        Fluxes.p += hfc_visc * (vi_dot_r - 0.5 * vdotr2_phys);
    }

#ifdef SPHAV_ARTIFICIAL_CONDUCTIVITY
    {
        double vsigu = (kernel.sound_i + kernel.sound_j - 3 * fac_mu * kernel.vdotr2 / kernel.r) / fac_mu;
        if((vsigu > 0) && (fabs(local.Pressure) + fabs(CellP[j].Pressure) > 0)) {
            vsigu *= fabs(local.Pressure - CellP[j].Pressure) / (fabs(local.Pressure) + fabs(CellP[j].Pressure));
            double du_ij = kernel.spec_egy_u_i - CellP[j].InternalEnergyPred;
#if defined(SPHAV_CD10_VISCOSITY_SWITCH)
            du_ij *= 0.5 * (local.alpha + CellP[j].alpha_limiter * CellP[j].alpha);
#endif
            Fluxes.p += local.Mass * All.ArtCondConstant * P[j].Mass * kernel.rho_ij_inv * vsigu * du_ij * kernel.dwk_ij;
        }
    }
#endif

    Fluxes.p *= All.cf_a2inv;
    Fluxes.v *= 1. / All.cf_atime;
#else  /* !HYDRO_SPH */
    (void)local; (void)j; (void)P; (void)CellP; (void)VelPred_j;
    (void)BPred_j; (void)PhiPred_j; (void)kernel; (void)r2; (void)V_j;
    (void)magneticspeed_i; (void)magneticspeed_j; (void)fac_mu; (void)fac_vsic_fix;
    (void)tensile_correction_factor; (void)dt_hydrostep;
    (void)magfluxv; (void)resistivity_heatflux; (void)Fluxes;
#endif /* HYDRO_SPH */
}

#endif /* HYDRO_CORE_SPH_FUNCTIONS_H */
