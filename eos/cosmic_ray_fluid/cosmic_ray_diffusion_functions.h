/* cosmic_ray_diffusion_functions.h -- per-pair cosmic-ray fluid
 * diffusion / streaming / Alfven / shock-injection fluxes.
 *
 * Replaces the fragment eos/cosmic_ray_fluid/cosmic_ray_diffusion.h. Body
 * guarded by COSMIC_RAY_FLUID so callers invoke unconditionally.
 *
 * Pure i-accumulation into:
 *   - Fluxes.CosmicRayPressure / CosmicRayAlfvenEnergy (Riemann workspace)
 *   - out.DtCosmicRayEnergy / DtCosmicRayAlfvenEnergy /
 *     DtCosmicRay_Number_in_Bin / Face_DivVel_ForAdOps /
 *     DtCREgyNewInjectionFromShocks
 *
 * j-side writes removed with the j_is_active_for_fluxes sweep.
 *
 * Only used on the hydro pass (not the RT transport subcycle), so the local
 * struct is fixed as hydro_data_in.
 *
 * Requires allvars.h, kernel.h, hydro_structs.h, hydro_pair_types.h.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef COSMIC_RAY_DIFFUSION_FUNCTIONS_H
#define COSMIC_RAY_DIFFUSION_FUNCTIONS_H

#include "../../hydro/hydro_pair_types.h"

KOKKOS_INLINE_FUNCTION
void cosmic_ray_diffusion_compute_pair(
    const struct hydro_data_in &local,
    int j,
    struct particle_data *P,
    struct gas_cell_data *CellP,
    const Vec3<MyDouble> &VelPred_j,
    const struct kernel_hydra &kernel,
    const Vec3<double> &Face_Area_Vec,
    double Face_Area_Norm,
    double V_i, double V_j,
    double Particle_Size_i, double Particle_Size_j,
    double face_vel_i, double face_vel_j,
    double face_area_dot_vel,
    double Riemann_S_M,
    const double *CosmicRayPressure_j,
    const Vec3<double> &bhat,
    double dt_hydrostep,
    struct Conserved_var_Riemann &Fluxes,
    struct hydro_data_out &out)
{
#ifdef COSMIC_RAY_FLUID
    double sqrtthreeinv = 1. / sqrt(3.0);
    double cosmo_unit = All.cf_a3inv;
    double V_i_phys = V_i / All.cf_a3inv;
    double V_j_phys = V_j / All.cf_a3inv;

    for(int k_CRegy=0; k_CRegy<N_CR_PARTICLE_BINS; k_CRegy++)
    {
        double scalar_i = local.CosmicRayPressure[k_CRegy] * cosmo_unit;
        double scalar_j = CosmicRayPressure_j[k_CRegy] * cosmo_unit;
        double kappa_i  = fabs(local.CosmicRayDiffusionCoeff[k_CRegy]);
        double kappa_j  = fabs(CellP[j].CosmicRayDiffusionCoeff[k_CRegy]);
        double d_scalar = scalar_i - scalar_j;
        double gamma_cr_m1 = GAMMA_COSMICRAY(k_CRegy) - 1.;

        if(((kappa_i > MIN_REAL_NUMBER) || (kappa_j > MIN_REAL_NUMBER))
           && (local.Mass > 0) && (P[j].Mass > 0)
           && (dt_hydrostep > MIN_REAL_NUMBER) && (Face_Area_Norm > MIN_REAL_NUMBER))
        {
            double cmag = 0., flux_norm = 0;
            double flux_i[3] = {0}, flux_j[3] = {0};
            double thold_hll;
            for(int k=0; k<3; k++) {
                flux_i[k] = local.CosmicRayFlux[k_CRegy][k] / V_i_phys;
                flux_j[k] = CellP[j].CosmicRayFlux[k_CRegy][k] / V_j_phys;
                double flux_ij = 0.5 * (flux_i[k] + flux_j[k]);
                flux_norm += flux_ij * flux_ij;
                cmag += Face_Area_Vec[k] * flux_ij;
            }
            flux_norm = sqrt(flux_norm) + MIN_REAL_NUMBER;
            double cos_theta_face_flux = cmag / (Face_Area_Norm * flux_norm);
            if(cos_theta_face_flux < -1)     { cos_theta_face_flux = -1; }
            else if(cos_theta_face_flux > 1) { cos_theta_face_flux = 1; }

            double cr_m1_speed_touse = CRFLUID_REDUCED_C_CODE(k_CRegy);
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
            double c_hll = 0.5 * fabs(face_vel_i - face_vel_j) + cr_m1_speed_touse;
            double renormerFAC = cos_theta_face_flux * cos_theta_face_flux;
#else
            double kappa_ij = cosmicrayfluid_rsol_corrfac(k_CRegy) * 0.5 * (kappa_i + kappa_j);
            double CRopticaldepth = DMIN(Particle_Size_i, Particle_Size_j) * cr_m1_speed_touse / kappa_ij;
            double reductionfactor = (4. + CRopticaldepth * (4. + 3. * CRopticaldepth))
                                   / (4. + CRopticaldepth * (4. + CRopticaldepth * (4. + 3. * CRopticaldepth)));
            double reducedcM1 = reductionfactor * cr_m1_speed_touse * sqrtthreeinv;
            double c_light_eff = DMAX(2. * kernel.vsig, reducedcM1);
            double v_eff_touse = DMIN(c_light_eff, kappa_ij / Particle_Size_j);
            double c_hll = 0.5 * fabs(face_vel_i - face_vel_j) + v_eff_touse;
            double hll_corr = 1. / (1. + 1.5 * c_light_eff * DMAX(Particle_Size_j / kappa_j, Particle_Size_i / kappa_i));
            double q = 0.5 * c_hll * (kernel.r * All.cf_atime) / fabs(MIN_REAL_NUMBER + kappa_ij);
            q = (0.2 + q) / (0.2 + q + q*q);
            double renormerFAC = DMIN(1., fabs(cos_theta_face_flux * cos_theta_face_flux * q * hll_corr));
#endif

            double f_direct = -Face_Area_Norm * c_hll * (d_scalar / gamma_cr_m1) * renormerFAC;
            if(dt_hydrostep > 0) {
                double f_direct_max = 0.25 * fabs(d_scalar) * DMIN(V_i_phys, V_j_phys) / dt_hydrostep;
                if(fabs(f_direct) > f_direct_max) { f_direct *= f_direct_max / fabs(f_direct); }
            } else { f_direct = 0; }
            double sign_c0 = f_direct * cmag;
            if((sign_c0 < 0) && (fabs(f_direct) > fabs(cmag))) { cmag = 0; }
            if(cmag != 0) {
                if(f_direct != 0) {
                    thold_hll = 1.0 * fabs(cmag);
                    if(fabs(f_direct) > thold_hll) { f_direct *= thold_hll / fabs(f_direct); }
                    cmag += f_direct;
                }
                cmag *= dt_hydrostep;
                double sVi = scalar_i * V_i_phys / gamma_cr_m1;
                double sVj = scalar_j * V_j_phys / gamma_cr_m1;
                thold_hll = DMAX(DMIN(fabs(sVi), fabs(sVj)), DMIN(fabs(sVi - sVj), DMAX(fabs(sVi), fabs(sVj))));
                if(sign_c0 < 0) { thold_hll *= 0.001; }
                if(fabs(cmag) > thold_hll) { cmag *= thold_hll / fabs(cmag); }
                cmag /= dt_hydrostep;
                Fluxes.CosmicRayPressure[k_CRegy] = cmag;
            }

#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
            {
                double A_dot_bhat = 0;
                double flux_tmp[2];
                int k_j_to_i = 0, k_i_to_j = 1;
                for(int k=0; k<3; k++) { A_dot_bhat += Face_Area_Vec[k] * bhat[k]; }
                if(A_dot_bhat < 0) { k_j_to_i = 1; k_i_to_j = 0; }
                A_dot_bhat = fabs(A_dot_bhat);
                flux_tmp[k_j_to_i] = +A_dot_bhat * sqrt(kernel.alfven2_j) / V_j_phys;
                flux_tmp[k_i_to_j] = -A_dot_bhat * sqrt(kernel.alfven2_i) / V_i_phys;
                if(flux_tmp[k_j_to_i] * dt_hydrostep > +0.5) { flux_tmp[k_j_to_i] = +0.5 / dt_hydrostep; }
                if(flux_tmp[k_i_to_j] * dt_hydrostep < -0.5) { flux_tmp[k_i_to_j] = -0.5 / dt_hydrostep; }
                Fluxes.CosmicRayAlfvenEnergy[k_CRegy][k_j_to_i] += flux_tmp[k_j_to_i] * CellP[j].CosmicRayAlfvenEnergyPred[k_CRegy][k_j_to_i];
                Fluxes.CosmicRayAlfvenEnergy[k_CRegy][k_i_to_j] += flux_tmp[k_i_to_j] * local.CosmicRayAlfvenEnergy[k_CRegy][k_i_to_j];
                for(int k=0; k<2; k++) { out.DtCosmicRayAlfvenEnergy[k_CRegy][k] += Fluxes.CosmicRayAlfvenEnergy[k_CRegy][k]; }
            }
#endif
        }

        out.DtCosmicRayEnergy[k_CRegy] += Fluxes.CosmicRayPressure[k_CRegy];
#if defined(CRFLUID_EVOLVE_SPECTRUM)
        double CR_number_to_energy_ratio = 0;
        if(Fluxes.CosmicRayPressure[k_CRegy] > 0) {
            CR_number_to_energy_ratio = CellP[j].Flux_Number_to_Energy_Correction_Factor[k_CRegy]
                                      * CellP[j].CosmicRay_Number_in_Bin[k_CRegy]
                                      / (CellP[j].CosmicRayEnergy[k_CRegy] + MIN_REAL_NUMBER);
        } else {
            CR_number_to_energy_ratio = local.CR_number_to_energy_ratio[k_CRegy];
        }
        out.DtCosmicRay_Number_in_Bin[k_CRegy] += Fluxes.CosmicRayPressure[k_CRegy] * CR_number_to_energy_ratio;
#endif
    }

    out.Face_DivVel_ForAdOps += -(All.cf_a3inv / V_i) * Face_Area_Norm * (Riemann_S_M + face_area_dot_vel) / All.cf_a2inv;

#if defined(CRFLUID_INJECTION_AT_SHOCKS)
    {
        double min_shockvel_for_accel = 100., min_machnum_for_accel = 5.;
        double min_machnum_for_pressurecrit = 1.3;
        double saturation_fraction_for_craccel = (double)(CRFLUID_INJECTION_AT_SHOCKS);
        double vdotf2_phys = face_vel_i - face_vel_j;
        double vdotr2_phys = kernel.vdotr2 / (kernel.r * All.cf_atime);
        if(All.ComovingIntegrationOn) { vdotr2_phys -= All.cf_hubble_a2 * kernel.r / All.cf_atime; }
        if(vdotr2_phys < 0 && vdotf2_phys < 0
           && (DMAX(vdotf2_phys, vdotr2_phys) * UNIT_VEL_IN_KMS > min_shockvel_for_accel)) {
            if((local.InternalEnergyPred - CellP[j].InternalEnergyPred) * (local.Density - CellP[j].Density) > 0) {
                if((local.Pressure - CellP[j].Pressure) * (local.InternalEnergyPred - CellP[j].InternalEnergyPred) > 0) {
                    double gamma_touse = 5./3.;
                    double m2 = min_machnum_for_accel * min_machnum_for_accel;
                    double gplus = gamma_touse + 1., gminus = gamma_touse - 1.;
                    double pjump_crit = (2. * gamma_touse * min_machnum_for_pressurecrit * min_machnum_for_pressurecrit - gminus) / gplus;
                    double tjump_crit = (2. * gamma_touse * m2 - gminus) * (gminus * m2 + 2.) / (gplus * gplus * m2);
                    double tjump = local.InternalEnergyPred / CellP[j].InternalEnergyPred;
                    double pjump = local.Pressure / CellP[j].Pressure;
                    if(tjump < 1.) { tjump = 1. / tjump; pjump = 1. / pjump; }
                    if((tjump > tjump_crit) && (pjump > pjump_crit)) {
                        double m2_guess = (16./5.) * tjump;
                        double dissipation_fac = (0.5625 - 2.9607 / m2_guess);
                        double upwind_density = DMIN(local.Density, CellP[j].Density);
                        double zeta_obliquity_fac = 1.;
                        double velforflux = fabs(vdotf2_phys);
#ifdef MAGNETIC
                        double cos_t = 0, dvmag = 0, bmag = 0;
                        for(int k=0; k<3; k++) {
                            double dv = local.Vel[k] - VelPred_j[k];
                            cos_t += bhat[k] * dv;
                            dvmag += dv * dv;
                            bmag  += bhat[k] * bhat[k];
                        }
                        if(dvmag > 0 && bmag > 0) { cos_t /= sqrt(dvmag * bmag); } else { cos_t = 0; }
                        double q = (1. - fabs(cos_t)) / 0.34;
                        zeta_obliquity_fac = exp(-q * q * q);
#endif
                        double DtCREgyNewInjection = saturation_fraction_for_craccel * zeta_obliquity_fac
                                                   * dissipation_fac * 0.5 * upwind_density
                                                   * velforflux * velforflux * velforflux * Face_Area_Norm;
                        if(local.InternalEnergyPred > CellP[j].InternalEnergyPred) {
                            out.DtCREgyNewInjectionFromShocks += DtCREgyNewInjection;
                        }
                    }
                }
            }
        }
    }
#endif
#else
    (void)local; (void)j; (void)P; (void)CellP; (void)VelPred_j; (void)kernel;
    (void)Face_Area_Vec; (void)Face_Area_Norm; (void)V_i; (void)V_j;
    (void)Particle_Size_i; (void)Particle_Size_j; (void)face_vel_i; (void)face_vel_j;
    (void)face_area_dot_vel; (void)Riemann_S_M; (void)CosmicRayPressure_j;
    (void)bhat; (void)dt_hydrostep; (void)Fluxes; (void)out;
#endif /* COSMIC_RAY_FLUID */
}

#endif /* COSMIC_RAY_DIFFUSION_FUNCTIONS_H */
