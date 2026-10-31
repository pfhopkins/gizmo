/* hydro/gradients_tail.h -- per-particle bodies of the gradients writeback and finalization loop.
 *
 * One definition for both the host loops in hydro/gradients_loop.cc and their device kernels in
 * hydro/gpu_hydro_tails.cc (see hydro/gpu_hydro_tails.h for the contract). Each body touches only particle i
 * (Pp[i], Cp[i]) and its own gradient scratch `pas`. Include after gradients_loop.h.
 */
#ifndef GIZMO_GRADIENTS_TAIL_H
#define GIZMO_GRADIENTS_TAIL_H

#include "gpu_hydro_tails.h"

/* MAX/MIN macros mirroring the legacy preset versions from hydro/gradients.cc (mode parameter ignored -- these
 * are the symmetric re-formulation versions). */
#define ASSIGN_ADD_PRESET(x, y, mode) (x += y)
#define MAX_ADD(x, y, mode)           ((y > x) ? (x = y) : (1))
#define MIN_ADD(x, y, mode)           ((y < x) ? (x = y) : (1))

HYDRO_TAIL_FUNCTION void gradients_slopelimiter(double *grad, double valmax, double valmin, double alim, double h,
                                                double shoot_tol, int pos_preserve, double d_max, double val_cen)
{
    Vec3<double>& g = *reinterpret_cast<Vec3<double>*>(grad);
    double d_abs = g.norm_sq();
    if(d_abs > 0) {
        d_abs = sqrt(d_abs); double cfac = 1 / (alim * h * d_abs);
        double fabs_max = fabs(valmax), fabs_min = fabs(valmin), abs_max = fabs_max, abs_min = fabs_min, f_corr_overshoot;
        if(abs_max < abs_min) { abs_max = fabs_min; abs_min = fabs_max; }
        f_corr_overshoot = DMIN(abs_min + shoot_tol * abs_max, abs_max);
        cfac *= f_corr_overshoot;
        if(pos_preserve == 1) {
            double fmin = DMIN(val_cen, DMAX(0, DMAX(MIN_REAL_NUMBER * val_cen,
                                                    DMIN(0.5 * (val_cen + valmin),
                                                         val_cen - f_corr_overshoot))));
            cfac = DMIN(((val_cen - fmin) / d_max) / d_abs, cfac);
        }
        if(cfac < 1) { g *= cfac; }
    }
}
HYDRO_TAIL_FUNCTION void gradients_slopelimiter(Vec3<double>& grad, double valmax, double valmin, double alim, double h,
                                                double shoot_tol, int pos_preserve, double d_max, double val_cen)
{
    gradients_slopelimiter(grad.data, valmax, valmin, alim, h, shoot_tol, pos_preserve, d_max, val_cen);
}

HYDRO_TAIL_FUNCTION void gradients_construct(Vec3<MyDouble>& grad, int i, struct particle_data *Pp, struct gas_cell_data *Cp)
{
    if(SHOULD_I_USE_SPH_GRADIENTS(Cp[i].ConditionNumber)) {
        if(Cp[i].Density > 0) { grad *= Pp[i].DrkernNgbFactor / Cp[i].Density; }
    } else {
        grad = Cp[i].NV_T.matvec(grad);
    }
}
HYDRO_TAIL_FUNCTION void gradients_construct(double *grad, int i, struct particle_data *Pp, struct gas_cell_data *Cp)
{
    Vec3<MyDouble> v{grad[0], grad[1], grad[2]}; gradients_construct(v, i, Pp, Cp); grad[0] = v[0]; grad[1] = v[1]; grad[2] = v[2];
}

/* The first-iteration writeback (out2particle_GasGrad) of active gas cell i from its accumulated sums `out`, into
 * Cp[i] and the scratch `pas` (zeroed before). */
HYDRO_TAIL_FUNCTION void gradients_writeback_cell(struct particle_data *Pp, struct gas_cell_data *Cp, struct temporary_data_topass &pas,
                                                  int i, const struct GasGraddata_out_ *out)
{
    (void) Pp;
#ifdef MHD_CONSTRAINED_GRADIENT
    {
        ASSIGN_ADD_PRESET(pas.FaceDotB, out->FaceDotB, 0);
#ifdef MHD_CONSTRAINED_GRADIENT_MIDPOINT
        for(int k = 0; k < 3; k++) {
            ASSIGN_ADD_PRESET(pas.PhiGrad[k], out->Gradients[k].Phi, 0);
        }
#endif
    }
#endif

    {
        int j, k;
        MAX_ADD(pas.MaxDistance, out->MaxDistance, 0);
#ifdef TURB_DIFF_DYNAMIC
        for(j = 0; j < 3; j++) {
            MAX_ADD(pas.Maxima.Velocity_bar[j], out->Maxima.Velocity_bar[j], 0);
            MIN_ADD(pas.Minima.Velocity_bar[j], out->Minima.Velocity_bar[j], 0);
            ASSIGN_ADD_PRESET(Cp[i].Velocity_hat[j], out->Velocity_hat[j], 0);
            for(k = 0; k < 3; k++) {
                ASSIGN_ADD_PRESET(pas.GradVelocity_bar[j][k], out->Gradients[k].Velocity_bar[j], 0);
            }
        }
#endif

#if defined(KERNEL_CRK_FACES)
        ASSIGN_ADD_PRESET(pas.m0, out->m0, 0);
        for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(pas.dm0[k], out->dm0[k], 0); }
        for(j = 0; j < 3; j++) {
            ASSIGN_ADD_PRESET(pas.m1[j], out->m1[j], 0);
            for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(pas.dm1[j][k], out->dm1[j][k], 0); }
        }
        for(j = 0; j < 6; j++) {
            ASSIGN_ADD_PRESET(pas.m2[j], out->m2[j], 0);
            for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(pas.dm2[j][k], out->dm2[j][k], 0); }
        }
#endif

#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && (HYDRO_FIX_MESH_MOTION==6)
        ASSIGN_ADD_PRESET(pas.GlassAcc, out->GlassAcc, 0);
#endif
#ifdef SPHAV_CD10_VISCOSITY_SWITCH
        ASSIGN_ADD_PRESET(Cp[i].alpha_limiter, out->alpha_limiter, 0);
#endif

        MAX_ADD(pas.Maxima.Density,  out->Maxima.Density,  0);
        MIN_ADD(pas.Minima.Density,  out->Minima.Density,  0);
        MAX_ADD(pas.Maxima.Pressure, out->Maxima.Pressure, 0);
        MIN_ADD(pas.Minima.Pressure, out->Minima.Pressure, 0);
        for(k = 0; k < 3; k++) {
            ASSIGN_ADD_PRESET(Cp[i].Gradients.Density[k],  out->Gradients[k].Density,  0);
            ASSIGN_ADD_PRESET(Cp[i].Gradients.Pressure[k], out->Gradients[k].Pressure, 0);
        }
#ifdef DOGRAD_INTERNAL_ENERGY
        MAX_ADD(pas.Maxima.InternalEnergy, out->Maxima.InternalEnergy, 0);
        MIN_ADD(pas.Minima.InternalEnergy, out->Minima.InternalEnergy, 0);
        for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(Cp[i].Gradients.InternalEnergy[k], out->Gradients[k].InternalEnergy, 0); }
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 1)
        MAX_ADD(pas.Maxima.ElectronNumberDensity, out->Maxima.ElectronNumberDensity, 0);
        MIN_ADD(pas.Minima.ElectronNumberDensity, out->Minima.ElectronNumberDensity, 0);
        for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(Cp[i].Gradients.ElectronNumberDensity[k], out->Gradients[k].ElectronNumberDensity, 0); }
        MAX_ADD(pas.Maxima.ElectronTemperature, out->Maxima.ElectronTemperature, 0);
        MIN_ADD(pas.Minima.ElectronTemperature, out->Minima.ElectronTemperature, 0);
        for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(Cp[i].Gradients.ElectronTemperature[k], out->Gradients[k].ElectronTemperature, 0); }
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & (2|4|8))
        for(j = 0; j < 3; j++) {
            MAX_ADD(pas.Maxima.E_battery_T2[j], out->Maxima.E_battery_T2[j], 0);
            MIN_ADD(pas.Minima.E_battery_T2[j], out->Minima.E_battery_T2[j], 0);
            for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(Cp[i].Gradients.E_battery_T2[j][k], out->Gradients[k].E_battery_T2[j], 0); }
        }
#endif
#ifdef COSMIC_RAY_FLUID
        for(j = 0; j < N_CR_PARTICLE_BINS; j++) {
            MAX_ADD(pas.Maxima.CosmicRayPressure[j], out->Maxima.CosmicRayPressure[j], 0);
            MIN_ADD(pas.Minima.CosmicRayPressure[j], out->Minima.CosmicRayPressure[j], 0);
            for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(Cp[i].Gradients.CosmicRayPressure[j][k], out->Gradients[k].CosmicRayPressure[j], 0); }
        }
#endif
#ifdef DOGRAD_SOUNDSPEED
        MAX_ADD(pas.Maxima.SoundSpeed, out->Maxima.SoundSpeed, 0);
        MIN_ADD(pas.Minima.SoundSpeed, out->Minima.SoundSpeed, 0);
        for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(Cp[i].Gradients.SoundSpeed[k], out->Gradients[k].SoundSpeed, 0); }
#endif

        for(j = 0; j < 3; j++) {
            MAX_ADD(pas.Maxima.Velocity[j], out->Maxima.Velocity[j], 0);
            MIN_ADD(pas.Minima.Velocity[j], out->Minima.Velocity[j], 0);
            for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(Cp[i].Gradients.Velocity[j][k], out->Gradients[k].Velocity[j], 0); }
        }

#ifdef MAGNETIC
#ifdef HYDRO_SPH
#ifdef DIVBCLEANING_DEDNER
        ASSIGN_ADD_PRESET(Cp[i].divB, out->divB, 0);
#endif
        ASSIGN_ADD_PRESET(Cp[i].DtB, out->DtB, 0);
#endif
#ifdef MHD_CONSTRAINED_GRADIENT
        for(j = 0; j < 3; j++) {
            ASSIGN_ADD_PRESET(Cp[i].Face_Area[j], out->Face_Area[j], 0);
            for(k = 0; k < 3; k++) {
                ASSIGN_ADD_PRESET(pas.BGrad[j][k],       out->Gradients[k].B[j],     0);
                ASSIGN_ADD_PRESET(pas.FaceCrossX[j][k],  out->FaceCrossX[j][k],      0);
            }
        }
#endif
        for(j = 0; j < 3; j++) {
            MAX_ADD(pas.Maxima.B[j], out->Maxima.B[j], 0);
            MIN_ADD(pas.Minima.B[j], out->Minima.B[j], 0);
            for(k = 0; k < 3; k++) {
#ifndef MHD_CONSTRAINED_GRADIENT
                ASSIGN_ADD_PRESET(Cp[i].Gradients.B[j][k], out->Gradients[k].B[j], 0);
#endif
            }
        }
#ifdef DIVBCLEANING_DEDNER
        MAX_ADD(pas.Maxima.Phi, out->Maxima.Phi, 0);
        MIN_ADD(pas.Minima.Phi, out->Minima.Phi, 0);
#ifndef MHD_CONSTRAINED_GRADIENT_MIDPOINT
        for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(Cp[i].Gradients.Phi[k], out->Gradients[k].Phi, 0); }
#endif
#endif
#endif /* MAGNETIC */

#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
        for(j = 0; j < NUM_METAL_SPECIES; j++) {
            MAX_ADD(pas.Maxima.Metallicity[j], out->Maxima.Metallicity[j], 0);
            MIN_ADD(pas.Minima.Metallicity[j], out->Minima.Metallicity[j], 0);
            for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(Cp[i].Gradients.Metallicity[j][k], out->Gradients[k].Metallicity[j], 0); }
        }
#endif

#ifdef RT_COMPGRAD_EDDINGTON_TENSOR
        for(j = 0; j < N_RT_FREQ_BINS; j++) {
            MAX_ADD(pas.Maxima.Rad_E_gamma[j], out->Maxima.Rad_E_gamma[j], 0);
            MIN_ADD(pas.Minima.Rad_E_gamma[j], out->Minima.Rad_E_gamma[j], 0);
            for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(pas.Gradients_Rad_E_gamma[j][k], out->Gradients[k].Rad_E_gamma[j], 0); }
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
            for(int k_d = 0; k_d < 3; k_d++) {
                MAX_ADD(pas.Maxima.Rad_Flux[j][k_d], out->Maxima.Rad_Flux[j][k_d], 0);
                MIN_ADD(pas.Minima.Rad_Flux[j][k_d], out->Minima.Rad_Flux[j][k_d], 0);
                for(k = 0; k < 3; k++) {
                    ASSIGN_ADD_PRESET(Cp[i].Gradients.Rad_Flux_Grad[j][k_d][k], out->Gradients[k].Rad_Flux[j][k_d], 0);
                }
            }
#endif
        }
        /* The gradient dotted into the Eddington tensor (NV_T matvec replay) —
         * gradients.cc:432-448 verbatim. */
        for(int k_freq = 0; k_freq < N_RT_FREQ_BINS; k_freq++) {
            for(int k_xyz = 0; k_xyz < 3; k_xyz++) {
                for(int j_xyz = 0; j_xyz < 3; j_xyz++) {
                    for(int i_xyz = 0; i_xyz < 3; i_xyz++) {
                        Cp[i].Gradients.Rad_E_gamma_ET[k_freq][k_xyz] +=
                            Cp[i].NV_T[j_xyz][i_xyz] *
                            out->Gradients[i_xyz].Rad_E_gamma_ET[k_freq][k_xyz][j_xyz];
                    }
                }
            }
        }
#endif

#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || (ADAPTIVE_GRAVSOFT_FORALL & 1)
        ASSIGN_ADD_PRESET(Pp[i].AGS_zeta, out->AGS_zeta, 0);
#endif
    }
}

/* The post-iteration finalization of active gas cell i (construct_gradient, slope limiters, derived quantities). */
HYDRO_TAIL_FUNCTION void gradients_finalize_cell(struct particle_data *Pp, struct gas_cell_data *Cp, struct temporary_data_topass &pas, int i)
{
    int k, k1;
    gradients_construct(Cp[i].Gradients.Density, i, Pp, Cp);
    gradients_construct(Cp[i].Gradients.Pressure, i, Pp, Cp);
    for(k = 0; k < 3; k++) { gradients_construct(Cp[i].Gradients.Velocity[k], i, Pp, Cp); }
#ifdef TURB_DIFF_DYNAMIC
    for(k = 0; k < 3; k++) { gradients_construct(pas.GradVelocity_bar[k], i, Pp, Cp); }
#endif
#ifdef DOGRAD_INTERNAL_ENERGY
    gradients_construct(Cp[i].Gradients.InternalEnergy, i, Pp, Cp);
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 1)
    gradients_construct(Cp[i].Gradients.ElectronNumberDensity, i, Pp, Cp);
    gradients_construct(Cp[i].Gradients.ElectronTemperature, i, Pp, Cp);
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & (2|4|8))
    for(k = 0; k < 3; k++) { gradients_construct(Cp[i].Gradients.E_battery_T2[k], i, Pp, Cp); }
#endif
#ifdef COSMIC_RAY_FLUID
    for(k = 0; k < N_CR_PARTICLE_BINS; k++) { gradients_construct(Cp[i].Gradients.CosmicRayPressure[k], i, Pp, Cp); }
    int is_particle_local_extremum[N_CR_PARTICLE_BINS] = {0};
    is_particle_local_extremum[0] = 0;
#endif
#ifdef DOGRAD_SOUNDSPEED
    gradients_construct(Cp[i].Gradients.SoundSpeed, i, Pp, Cp);
#endif
#ifdef MAGNETIC
#ifndef MHD_CONSTRAINED_GRADIENT
    for(k = 0; k < 3; k++) { gradients_construct(Cp[i].Gradients.B[k], i, Pp, Cp); }
#endif
#if defined(DIVBCLEANING_DEDNER) && !defined(MHD_CONSTRAINED_GRADIENT_MIDPOINT)
    gradients_construct(Cp[i].Gradients.Phi, i, Pp, Cp);
#endif
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
    for(k = 0; k < NUM_METAL_SPECIES; k++) { gradients_construct(Cp[i].Gradients.Metallicity[k], i, Pp, Cp); }
#endif
#ifdef RT_COMPGRAD_EDDINGTON_TENSOR
    for(k = 0; k < N_RT_FREQ_BINS; k++) { gradients_construct(pas.Gradients_Rad_E_gamma[k], i, Pp, Cp); }
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
    for(int k_f = 0; k_f < N_RT_FREQ_BINS; k_f++) {
        for(int k_d = 0; k_d < 3; k_d++) { gradients_construct(Cp[i].Gradients.Rad_Flux_Grad[k_f][k_d], i, Pp, Cp); }
    }
#endif
#endif

#ifdef DO_DENSITY_AROUND_NONGAS_PARTICLES
    Pp[i].GradRho = Cp[i].Gradients.Density;
#endif

#if defined(TURB_DRIVING) || defined(OUTPUT_VORTICITY)
    Cp[i].Vorticity = Cp[i].Gradients.Velocity.curl();
#endif

#ifdef SPH_TP12_ARTIFICIAL_RESISTIVITY
    {
        double GradBMag = Cp[i].Gradients.B.frobenius_norm_sq();
        double rho_over_m = Cp[i].Density / Pp[i].Mass;
        double BMag = Cp[i].BPred.norm_sq() * rho_over_m * rho_over_m;
        Cp[i].Balpha = DMAX(DMIN(Pp[i].KernelRadius * sqrt(GradBMag / (BMag + 1.0e-33)),
                                    0.1 * All.ArtMagDispConst), 0.005);
    }
#endif

#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || (ADAPTIVE_GRAVSOFT_FORALL & 1)
    {
        double ngb_eff = pow(Pp[i].NumNgb, NUMDIMS);
        int softening_is_free = (Pp[i].KernelRadius > All.ForceSoftening[0]); /* at the softening floor the radius does not vary with the density */
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) && defined(ADAPTIVE_GRAVSOFT_MAX_SOFT_HARD_LIMIT)
        if(Pp[i].KernelRadius >= ADAPTIVE_GRAVSOFT_MAX_SOFT_HARD_LIMIT/All.cf_atime) {softening_is_free = 0;} /* nor at the hard upper limit */
#endif
        if((fabs(ngb_eff - All.DesNumNgb) / All.DesNumNgb < 0.05)
           && (Pp[i].KernelRadius > 1.001 * All.MinKernelRadius)
           && (Pp[i].KernelRadius < 0.999 * All.MaxKernelRadius) && softening_is_free) {
            double ndenNGB = ngb_eff / (VOLUME_NORM_COEFF_FOR_NDIMS * pow(Pp[i].KernelRadius, NUMDIMS));
            Pp[i].AGS_zeta *= Pp[i].Mass * Pp[i].KernelRadius / (NUMDIMS * ndenNGB) * Pp[i].DrkernNgbFactor;
        } else {
            Pp[i].AGS_zeta = 0;
        }
    }
#endif

#ifdef HYDRO_SPH
#ifdef MAGNETIC
    if(Cp[i].Density > 0) {
        Cp[i].DtB *= Pp[i].DrkernNgbFactor * Pp[i].Mass / (Cp[i].Density * Cp[i].Density) / All.cf_atime;
#ifdef DIVBCLEANING_DEDNER
        Cp[i].divB *= Pp[i].DrkernNgbFactor * Pp[i].Mass / (Cp[i].Density * Cp[i].Density);
        if((!isnan(Cp[i].divB)) && (Pp[i].KernelRadius > 0) && (Cp[i].divB != 0) && (Cp[i].Density > 0)) {
            double tmp_ded = 0.5 * Cp[i].MaxSignalVel;
            double rho_over_m_divb = Cp[i].Density / Pp[i].Mass;
            double b2_max = Cp[i].BPred.norm_sq() * rho_over_m_divb * rho_over_m_divb;
            b2_max = 100.0 * fabs(sqrt(b2_max) * All.cf_a2inv * Pp[i].Mass
                      / (Cp[i].Density * All.cf_a3inv) * 1.0
                      / (Pp[i].KernelRadius * All.cf_atime));
            if(fabs(Cp[i].divB) > b2_max) { Cp[i].divB *= b2_max / fabs(Cp[i].divB); }
            Cp[i].DtPhi = -tmp_ded * tmp_ded * All.DivBcleanHyperbolicSigma
                              * Cp[i].divB * Cp[i].Density * All.cf_a3inv;
        } else {
            Cp[i].DtPhi = 0; Cp[i].divB = 0; Cp[i].DtB = {};
        }
        Cp[i].divB = 0.0;
#endif
    } else {
        Cp[i].DtB = {};
#ifdef DIVBCLEANING_DEDNER
        Cp[i].divB = 0; Cp[i].DtPhi = 0;
#endif
    }
#endif

#ifdef SPHAV_CD10_VISCOSITY_SWITCH
    {
        Cp[i].alpha_limiter /= Cp[i].Density;
        double NV_dt = get_particle_timestep_in_physical(i, Pp);
        double NV_dummy = fabs(1.0 * pow(1.0 - Cp[i].alpha_limiter, 4.0) * Cp[i].NV_DivVel);
        double NV_limiter = NV_dummy * NV_dummy / (NV_dummy * NV_dummy + Cp[i].NV_trSSt);
        double NV_A = DMAX(-Cp[i].NV_dt_DivVel, 0.0);
        double divVel_physical = Cp[i].NV_DivVel;
        if(All.ComovingIntegrationOn) { divVel_physical += 3 * All.cf_hubble_a; }
        if(divVel_physical >= 0.0) { NV_A = 0.0; }
        double h_eff = Pp[i].Get_Particle_Size() * All.cf_atime / 0.5;
        double cs_nv = Cp[i].effective_soundspeed();
        double alphaloc = All.ViscosityAMax * h_eff * h_eff * NV_A / (0.36 * cs_nv * cs_nv + h_eff * h_eff * NV_A);
        if(Cp[i].alpha < alphaloc) { Cp[i].alpha = alphaloc; }
        else if(Cp[i].alpha > alphaloc) {
            Cp[i].alpha = alphaloc + (Cp[i].alpha - alphaloc)
                              * exp(-NV_dt * (0.5 * fabs(Cp[i].MaxSignalVel)) / (0.5 * h_eff) * 0.05);
        }
        if(Cp[i].alpha < All.ViscosityAMin) { Cp[i].alpha = All.ViscosityAMin; }
        Cp[i].alpha_limiter = DMAX(NV_limiter, All.ViscosityAMin / Cp[i].alpha);
    }
#else
    {
        double divVel = All.cf_a2inv * fabs(Cp[i].Gradients.Velocity.trace());
        if(All.ComovingIntegrationOn) { divVel += 3 * All.cf_hubble_a; }
        Vec3<double> CurlVel = Cp[i].Gradients.Velocity.curl();
        double MagCurl = All.cf_a2inv * CurlVel.norm();
        double fac_mu = 1 / All.cf_atime;
        Cp[i].alpha_limiter = divVel / (divVel + MagCurl
                                   + 0.0001 * Cp[i].effective_soundspeed()
                                             / (Pp[i].Get_Particle_Size()) / fac_mu);
    }
#endif
#endif

#ifndef __CUDA_ARCH__   /* host pass only: both are empty in every configuration the device tail admits (gpu_hydro_tails.h) */
    calculate_and_assign_conduction_and_viscosity_coefficients(i, Pp, Cp);
    calculate_and_assign_nonideal_mhd_coefficients(i, Pp, Cp);
#endif

#ifdef RADTRANSFER
    for(int k_freq = 0; k_freq < N_RT_FREQ_BINS; k_freq++) {
        Cp[i].Rad_Kappa[k_freq] = rt_kappa(i, k_freq, Pp, Cp);
#if defined(RT_FLUXLIMITER) && defined(RT_COMPGRAD_EDDINGTON_TENSOR)
        double lambda = 1;
        if(Cp[i].Rad_E_gamma_Pred[k_freq] > 0) {
            double R_ET = Cp[i].Gradients.Rad_E_gamma_ET[k_freq].norm()
                          / (MIN_REAL_NUMBER + Cp[i].Rad_E_gamma_Pred[k_freq]
                                                * Cp[i].Density / (MIN_REAL_NUMBER + Pp[i].Mass));
            R_ET = 3. * DMAX(R_ET, 1.e-6 / Pp[i].Get_Particle_Size())
                    / (1.e-55 + All.cf_atime * Cp[i].Rad_Kappa[k_freq]
                                 * (Cp[i].Density * All.cf_a3inv));
            lambda = DMIN(1., DMAX(3. * (2. + R_ET) / (6. + 3. * R_ET + R_ET * R_ET), MIN_REAL_NUMBER));
#ifdef RT_OTVET
            double chi = DMAX(1./3., DMIN(1., (3. + 4. * lambda * lambda)
                          / (5. + 2. * sqrt(4. - 3. * lambda * lambda))));
            double chifac_iso = 3. * (1 - chi) / 2., chifac_ot = (3. * chi - 1.) / 2.;
            Cp[i].Gradients.Rad_E_gamma_ET[k_freq] =
                chifac_ot * Cp[i].Gradients.Rad_E_gamma_ET[k_freq]
              + (chifac_iso / 3.) * pas.Gradients_Rad_E_gamma[k_freq];
#endif
        }
        Cp[i].Rad_Flux_Limiter[k_freq] = lambda;
#endif
#if defined(RT_COMPGRAD_EDDINGTON_TENSOR) && !defined(RT_OTVET)
        {
            Vec3<MyDouble> g{pas.Gradients_Rad_E_gamma[k_freq][0],
                             pas.Gradients_Rad_E_gamma[k_freq][1],
                             pas.Gradients_Rad_E_gamma[k_freq][2]};
            Cp[i].Gradients.Rad_E_gamma_ET[k_freq] = Cp[i].ET[k_freq].matvec(g);
        }
#endif
#if defined(GRAIN_RDI_TESTPROBLEM_LIVE_RADIATION_INJECTION)
        if(Cp[i].Interpolated_Opacity[0] < 1.e-3 * All.Dust_to_Gas_Mass_Ratio * 0.75
                 * All.Grain_Q_at_MaxGrainSize
                 / ((All.Grain_Internal_Density / UNIT_DENSITY_IN_CGS)
                    * (All.Grain_Size_Max / UNIT_LENGTH_IN_CGS))) {
            double gmax = -1;
            if(Pp[i].GravAccel[GRAV_DIRECTION_RDI] < gmax) {
                Pp[i].GravAccel[GRAV_DIRECTION_RDI] = gmax;
            }
        }
#endif
    }
#endif /* RADTRANSFER */

#if defined(EOS_ELASTIC)
    elastic_body_update_driftkick(i, 1., 2);
#endif

    /* Slope limiter block */
    {
        double stol = 0.0, stol_tmp, stol_diffusion;
        stol_diffusion = 0.1;
        stol_tmp = stol;
        double h_lim = Pp[i].KernelRadius;
        double d_max = DMAX(Pp[i].KernelRadius, pas.MaxDistance);
        h_lim = d_max;
        double a_limiter = 0.25;
        if(Cp[i].ConditionNumber > 100)
            a_limiter = DMIN(0.5, 0.25 + 0.25 * (Cp[i].ConditionNumber - 100) / 100);
#if defined(SELFGRAVITY_OFF) && (!defined(MAGNETIC) && !defined(GALSF))
        h_lim = Pp[i].KernelRadius; stol = 0.1;
#endif
#if (SLOPE_LIMITER_TOLERANCE == 2)
        h_lim = Pp[i].KernelRadius; a_limiter *= 0.5; stol = 0.125;
#endif
#if (SLOPE_LIMITER_TOLERANCE == 0)
        a_limiter *= 2.0; stol = 0.0;
#endif

#if (SINGLE_STAR_SINK_FORMATION & 4)
        Cp[i].Density_Relative_Maximum_in_Kernel = pas.Maxima.Density;
#endif
        gradients_slopelimiter(Cp[i].Gradients.Density, pas.Maxima.Density, pas.Minima.Density,
                           a_limiter, h_lim, 0, 1, d_max, Cp[i].Density);
        int pressure_is_positive_definite = 1;
#if defined(EOS_TILLOTSON) || defined(EOS_ELASTIC) || defined(EOS_ANEOS)
        pressure_is_positive_definite = 0;
#endif
        gradients_slopelimiter(Cp[i].Gradients.Pressure, pas.Maxima.Pressure, pas.Minima.Pressure,
                           a_limiter, h_lim, stol, pressure_is_positive_definite, d_max, Cp[i].Pressure);
        stol_tmp = stol;
#if defined(VISCOSITY)
        stol_tmp = DMAX(stol, stol_diffusion);
#endif
#ifdef TURB_DIFF_DYNAMIC
        for(k1 = 0; k1 < 3; k1++) {
            gradients_slopelimiter(pas.GradVelocity_bar[k1],
                               pas.Maxima.Velocity_bar[k1],
                               pas.Minima.Velocity_bar[k1],
                               a_limiter, h_lim, stol, 0, 0, 0);
        }
#endif
        for(k1 = 0; k1 < 3; k1++) {
            gradients_slopelimiter(Cp[i].Gradients.Velocity[k1],
                               pas.Maxima.Velocity[k1],
                               pas.Minima.Velocity[k1],
                               a_limiter, h_lim, stol_tmp, 0, 0, 0);
        }
#ifdef DOGRAD_INTERNAL_ENERGY
        stol_tmp = stol;
#if defined(CONDUCTION)
        stol_tmp = DMAX(stol, stol_diffusion);
#endif
        gradients_slopelimiter(Cp[i].Gradients.InternalEnergy,
                           pas.Maxima.InternalEnergy, pas.Minima.InternalEnergy,
                           a_limiter, h_lim, stol_tmp, 1, d_max, Cp[i].InternalEnergyPred);
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 1)
        gradients_slopelimiter(Cp[i].Gradients.ElectronNumberDensity,
                           pas.Maxima.ElectronNumberDensity, pas.Minima.ElectronNumberDensity,
                           a_limiter, h_lim, stol, 1, d_max, Cp[i].n_e());
        gradients_slopelimiter(Cp[i].Gradients.ElectronTemperature,
                           pas.Maxima.ElectronTemperature, pas.Minima.ElectronTemperature,
                           a_limiter, h_lim, stol, 1, d_max, Cp[i].T_e());
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & (2|4|8))
        for(k1 = 0; k1 < 3; k1++) {
            gradients_slopelimiter(Cp[i].Gradients.E_battery_T2[k1],
                               pas.Maxima.E_battery_T2[k1], pas.Minima.E_battery_T2[k1],
                               a_limiter, h_lim, stol, 0, 0, 0);
        }
#endif
#ifdef DOGRAD_SOUNDSPEED
        gradients_slopelimiter(Cp[i].Gradients.SoundSpeed,
                           pas.Maxima.SoundSpeed, pas.Minima.SoundSpeed,
                           a_limiter, h_lim, stol, 1, d_max, Cp[i].effective_soundspeed());
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
        for(k1 = 0; k1 < NUM_METAL_SPECIES; k1++) {
            gradients_slopelimiter(Cp[i].Gradients.Metallicity[k1],
                               pas.Maxima.Metallicity[k1], pas.Minima.Metallicity[k1],
                               a_limiter, h_lim, DMAX(stol, stol_diffusion), 1, d_max, Pp[i].Metallicity[k1]);
        }
#endif
#if defined(RT_COMPGRAD_EDDINGTON_TENSOR) && !defined(RT_EVOLVE_FLUX)
        for(k1 = 0; k1 < N_RT_FREQ_BINS; k1++) {
            gradients_slopelimiter(Cp[i].Gradients.Rad_E_gamma_ET[k1],
                               pas.Maxima.Rad_E_gamma[k1], pas.Minima.Rad_E_gamma[k1],
                               a_limiter, h_lim, stol, 1, d_max,
                               Cp[i].Rad_E_gamma_Pred[k1] * Cp[i].Density / Pp[i].Mass);
            gradients_slopelimiter(pas.Gradients_Rad_E_gamma[k1],
                               pas.Maxima.Rad_E_gamma[k1], pas.Minima.Rad_E_gamma[k1],
                               a_limiter, h_lim, DMAX(stol, stol_diffusion), 1, d_max,
                               Cp[i].Rad_E_gamma_Pred[k1] * Cp[i].Density / Pp[i].Mass);
        }
#endif
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
        {
            double V_i_inv_rt = Cp[i].Density / Pp[i].Mass;
            for(k1 = 0; k1 < N_RT_FREQ_BINS; k1++) {
                double val_cen_e = Cp[i].Rad_E_gamma_Pred[k1] * V_i_inv_rt;
                gradients_slopelimiter(pas.Gradients_Rad_E_gamma[k1],
                                   pas.Maxima.Rad_E_gamma[k1], pas.Minima.Rad_E_gamma[k1],
                                   a_limiter, h_lim, stol, 1, d_max, val_cen_e);
                for(k = 0; k < 3; k++) {
                    Cp[i].Gradients.Rad_E_gamma_Grad[k1][k] = pas.Gradients_Rad_E_gamma[k1][k];
                }
                for(int k_d = 0; k_d < 3; k_d++) {
                    double val_cen_f = Cp[i].Rad_Flux_Pred[k1][k_d] * V_i_inv_rt;
                    gradients_slopelimiter(Cp[i].Gradients.Rad_Flux_Grad[k1][k_d],
                                       pas.Maxima.Rad_Flux[k1][k_d], pas.Minima.Rad_Flux[k1][k_d],
                                       a_limiter, h_lim, stol, 0, d_max, val_cen_f);
                }
            }
        }
#endif
#ifdef MAGNETIC
#ifndef MHD_CONSTRAINED_GRADIENT
        {
            double v_tmp = Pp[i].Mass / Cp[i].Density;
            double tmp_d = sqrt(1.0e-37 + (2. * All.cf_atime / Cp[i].Pressure * v_tmp * v_tmp)
                                + Cp[i].BPred.norm_sq());
            double q = fabs(Cp[i].divB) * Pp[i].KernelRadius / tmp_d;
            double alim2 = a_limiter * (1. + q * q);
            if(alim2 > 0.5) alim2 = 0.5;
            stol_tmp = stol;
#ifdef MHD_NON_IDEAL
            stol_tmp = DMAX(stol, stol_diffusion);
#endif
            for(k1 = 0; k1 < 3; k1++) {
                gradients_slopelimiter(Cp[i].Gradients.B[k1],
                                   pas.Maxima.B[k1], pas.Minima.B[k1],
                                   alim2, h_lim, stol_tmp, 0, 0, 0);
            }
        }
#endif
#if defined(DIVBCLEANING_DEDNER) && !defined(MHD_CONSTRAINED_GRADIENT_MIDPOINT)
        gradients_slopelimiter(Cp[i].Gradients.Phi,
                           pas.Maxima.Phi, pas.Minima.Phi,
                           a_limiter, h_lim, stol, 0, 0, 0);
#endif
#endif
    }

#ifdef TURB_DIFFUSION
#ifdef TURB_DIFF_DYNAMIC
    for(int k1_ = 0; k1_ < 3; k1_++) for(int k2_ = 0; k2_ < 3; k2_++) {
        Cp[i].VelShear_bar[k1_][k2_] = 0.5 * (pas.GradVelocity_bar[k1_][k2_]
                                              + pas.GradVelocity_bar[k2_][k1_]);
    }
#endif
    calculate_and_assign_turbulent_diffusion_coefficients(i, Pp, Cp);
#endif

#if defined(COSMIC_RAY_FLUID) && !defined(CRFLUID_EVOLVE_SCATTERINGWAVES)
    for(k = 0; k < N_CR_PARTICLE_BINS; k++) { Cp[i].CosmicRayDiffusionCoeff[k] = 0; }
    if(Cp[i].Density > 0 && Pp[i].Mass > 0) {
        CalculateAndAssign_CosmicRay_DiffusionAndStreamingCoefficients(i, Pp, Cp);
    }
#endif

#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && (HYDRO_FIX_MESH_MOTION==6)
    if(All.Time > 0) {
        double cs_invelunits = Cp[i].effective_soundspeed() * All.cf_atime;
        double L_i_code = Pp[i].Get_Particle_Size();
        Vec3<double> dvel = L_i_code * L_i_code * pas.GlassAcc;
        double velnorm = dvel.norm();
        double dtx = get_particle_timestep_in_physical(i, Pp);
        if(velnorm > 0 && dtx > 0) {
            double v00 = 0.5 * DMIN(cs_invelunits * (0.5 * velnorm),
                                   All.CourantFac * (L_i_code / dtx) / All.cf_a2inv);
            Cp[i].ParticleVel += v00 * (dvel / velnorm);
        }
    }
#endif

#if defined(KERNEL_CRK_FACES)
    {
        double m0, dm0[3], m1[3], dm1[3][3], m2[3][3], m2i[3][3], dm2[3][3][3], Cnum_m2;
        m0 = pas.m0;
        for(k = 0; k < 3; k++) {
            dm0[k] = pas.dm0[k];
            m1[k]  = pas.m1[k];
            for(int k_x = 0; k_x < 3; k_x++) {
                dm1[k][k_x] = pas.dm1[k][k_x];
                int k_tmp = 0;
                if((k == 0) && (k_x == 0)) { k_tmp = 0; }
                if((k == 1) && (k_x == 1)) { k_tmp = 1; }
                if((k == 2) && (k_x == 2)) { k_tmp = 2; }
                if((k == 0) && (k_x == 1)) { k_tmp = 3; }
                if((k == 1) && (k_x == 0)) { k_tmp = 3; }
                if((k == 0) && (k_x == 2)) { k_tmp = 4; }
                if((k == 2) && (k_x == 0)) { k_tmp = 4; }
                if((k == 1) && (k_x == 2)) { k_tmp = 5; }
                if((k == 2) && (k_x == 1)) { k_tmp = 5; }
                m2[k][k_x] = pas.m2[k_tmp]; m2i[k][k_x] = 0;
                for(int k_y = 0; k_y < 3; k_y++) { dm2[k][k_x][k_y] = pas.dm2[k_tmp][k_y]; }
            }
        }
        for(k = 0; k < 3; k++) { dm1[k][k] += m0; }
        for(k = 0; k < 3; k++) for(int k_x = 0; k_x < 3; k_x++) {
            dm2[k][k_x][k_x] += m1[k]; dm2[k_x][k][k_x] += m1[k];
        }
        Cnum_m2 = matrix_invert_ndims(m2, m2i);
        (void)Cnum_m2;
        double A = 0, B[3] = {0}, Bdotm1 = 0, dB[3][3] = {{0}}, dA[3] = {0};
        for(k = 0; k < 3; k++) {
            for(int k_x = 0; k_x < 3; k_x++) { B[k] += -m2i[k][k_x] * m1[k_x]; }
            Bdotm1 += B[k] * m1[k];
        }
        A = 1. / (m0 + Bdotm1);

        double minus_m2i_dm1_dotm1[3] = {0}, contracted_twotensor[3][3] = {{0}};
        double contracted_twotensor_x[3][3] = {{0}}, contracted_twotensor_dotm1[3] = {0};
        for(int k_gamma = 0; k_gamma < 3; k_gamma++) {
            for(int k_alpha = 0; k_alpha < 3; k_alpha++) {
                for(int k_beta = 0; k_beta < 3; k_beta++) {
                    contracted_twotensor[k_beta][k_gamma] = 0;
                    for(int k_delta = 0; k_delta < 3; k_delta++) {
                        contracted_twotensor[k_beta][k_gamma] += dm2[k_beta][k_delta][k_gamma] * B[k_delta];
                    }
                    contracted_twotensor_x[k_alpha][k_gamma] += dm2[k_alpha][k_beta][k_gamma] * B[k_beta];
                    dB[k_alpha][k_gamma] += -m2i[k_alpha][k_beta]
                        * (dm1[k_beta][k_gamma] + contracted_twotensor[k_beta][k_gamma]);
                }
                minus_m2i_dm1_dotm1[k_gamma] += 2. * B[k_alpha] * dm1[k_alpha][k_gamma];
                contracted_twotensor_dotm1[k_gamma] += B[k_alpha] * contracted_twotensor_x[k_alpha][k_gamma];
            }
            dA[k_gamma] = -A * A * (dm0[k_gamma] + minus_m2i_dm1_dotm1[k_gamma]
                            + contracted_twotensor_dotm1[k_gamma]);
        }

        double vector_corr[3] = {0}, tensor_corr[3][3] = {{0}};
        for(k = 0; k < 3; k++) {
            vector_corr[k] = dA[k] + A * B[k];
            for(int k_x = 0; k_x < 3; k_x++) {
                tensor_corr[k][k_x] = B[k] * dA[k_x] + A * dB[k][k_x];
            }
        }
        Cp[i].Tensor_CRK_Face_Corrections[0] = A;
        for(k = 0; k < 3; k++) { Cp[i].Tensor_CRK_Face_Corrections[1 + k] = B[k]; }
        for(k = 0; k < 3; k++) { Cp[i].Tensor_CRK_Face_Corrections[1 + 3 + k] = vector_corr[k]; }
        for(k = 0; k < 3; k++) for(int k_x = 0; k_x < 3; k_x++) {
            Cp[i].Tensor_CRK_Face_Corrections[1 + 3 + 3 + 3 * k + k_x] = tensor_corr[k][k_x];
        }
    }
#endif
}

#if GIZMO_HYDRO_TAILS_DEVICE
/* The host finalization loop of an admitted configuration (hydro/gpu_hydro_tails.cc). It is compiled with the
 * device kernels' translation unit, without contraction to fused multiply-add on either side, so the host and the
 * device loops give the same bits (the host compiler contracts by default where its target has FMA). */
void gradients_finalize_host_loop(struct temporary_data_topass *passer_base);
#endif

#endif /* GIZMO_GRADIENTS_TAIL_H */
