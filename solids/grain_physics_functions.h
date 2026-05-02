/* grain_physics_functions.h — GPU-callable structs + per-pair kernel bodies
 * for the grain_physics neighbor loops (B7).
 *
 * Covers two independent loops:
 *   1. grain_backrx_evaluate (GRAIN_BACKREACTION): grain→gas momentum
 *      backreaction, atomic j-writes to Vel/VelPred/dp/Grain_AccelTimeMin.
 *   2. interpolate_fluxes_opacities_gasgrains_evaluate
 *      (RT_OPACITY_FROM_EXPLICIT_GRAINS): bidirectional gas↔grain RT
 *      coupling, per-source output struct only (no j-writes).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 8)
#include "grain_charge_functions.h"
#endif


/* -------------------------------------------------------------------------
 *  GPU-callable grain extinction efficiency (used by B7b kernels).
 *  Mirrors solids/grain_physics.cc:return_grain_extinction_efficiency_Q.
 *  Kept GPU-safe: no ThisTask/PRINT_WARNING fallback (that path only fires
 *  when neither RADTRANSFER nor RT_USE_GRAVTREE is on, i.e. a non-RT build
 *  — in which case RT_OPACITY_FROM_EXPLICIT_GRAINS won't be meaningful). */
#if defined(GRAIN_FLUID)
KOKKOS_INLINE_FUNCTION
static double grain_extinction_Q_inline(int i, int k_freq, struct particle_data *P_arr)
{
    double Q = 1;
#if defined(GRAIN_RDI_TESTPROBLEM)
    Q *= All.Grain_Q_at_MaxGrainSize;
#if !defined(GRAIN_RDI_TESTPROBLEM_ACCEL_DEPENDS_ON_SIZE)
    Q *= (double)P_arr[i].Grain_Size / All.Grain_Size_Max;
#endif
#else
#if defined(RADTRANSFER) || defined(RT_USE_GRAVTREE)
    double nu_min_ev = All.RHD_bins_nu_min_ev[k_freq];
    double nu_max_ev = All.RHD_bins_nu_max_ev[k_freq];
    double x_min = 5.068e4 * (double)P_arr[i].Grain_Size * nu_min_ev;
    double x_max = 5.068e4 * (double)P_arr[i].Grain_Size * nu_max_ev;
    double x_eff = sqrt(x_min * x_max);
    if(x_min <= MIN_REAL_NUMBER) x_eff = 0.5 * x_max;
    return DMIN(x_eff, 1.0);
#endif
#endif
    return Q;
}
#endif /* GRAIN_FLUID */


/* -------------------------------------------------------------------------
 *  B7a: grain_backrx (GRAIN_BACKREACTION)
 * ------------------------------------------------------------------------- */
#if defined(GRAIN_FLUID) && defined(GRAIN_BACKREACTION)

struct GrainBackrxLocalIn
{
    Vec3<MyDouble> Pos;
    MyFloat KernelRadius;
    Vec3<MyFloat> Grain_DeltaMomentum;
    MyFloat Gas_Density;
    MyFloat Grain_AccelTimeMin;
};

/* No per-source output (matches the CPU OUTPUT_STRUCT_NAME which is empty). */
struct GrainBackrxOut { char _unused; };

KOKKOS_INLINE_FUNCTION
static void grain_backrx_pair_kernel(
    const struct GrainBackrxLocalIn& local,
    int j,
    struct particle_data *P,
    struct gas_cell_data *CellP,
    const Vec3<double>& dp,    /* = local.Pos - P[j].Pos, nearest_xyz-corrected */
    double r2)
{
    if(P[j].Mass <= 0) return;
    if(P[j].KernelRadius <= 0) return;
    double h = (double)local.KernelRadius;
    if(r2 <= 0 || r2 >= h * h) return;
#ifdef BOX_BND_PARTICLES
    if(P[j].ID > 0) return;
#endif
    double hinv, hinv3, hinv4, wk_i = 0, dwk_i = 0;
    kernel_hinv(h, &hinv, &hinv3, &hinv4);
    double r = sqrt(r2);
    kernel_main(r * hinv, hinv3, hinv4, &wk_i, &dwk_i, 0);

    double gas_rho = (double)local.Gas_Density;
    if(gas_rho <= 0) return;
    double wt = -wk_i / gas_rho;

    double dv[3]; double dv2 = 0;
    for(int k = 0; k < 3; k++) {
        dv[k] = wt * (double)local.Grain_DeltaMomentum[k];
        dv2 += dv[k] * dv[k];
    }
    double Mass_j = (double)P[j].Mass;
    for(int k = 0; k < 3; k++) {
        Kokkos::atomic_add(&P[j].Vel[k],        (MyDouble)dv[k]);
        Kokkos::atomic_add(&CellP[j].VelPred[k], (MyDouble)dv[k]);
        Kokkos::atomic_add(&P[j].dp[k],         (MyFloat)(dv[k] * Mass_j));
    }

    /* atomic minimum update on P[j].Grain_AccelTimeMin.
     * The CPU path reads prev, computes new = min(X, Y, prev), and writes if
     * new < prev — equivalent to atomic_min(*, min(X, Y)) without the race. */
    double pgsize = P[j].Get_Particle_Size();
    double taccel_cand = DMIN(
        2.0 * All.ErrTolIntAccuracy * pgsize * All.cf_atime * All.cf_atime / sqrt(dv2 + MIN_REAL_NUMBER),
        4.0 * (double)local.Grain_AccelTimeMin);
    Kokkos::atomic_min(&P[j].Grain_AccelTimeMin, (MyFloat)taccel_cand);
}

#endif /* GRAIN_FLUID && GRAIN_BACKREACTION */


/* -------------------------------------------------------------------------
 *  B7b: gas↔grain RT opacity coupling (RT_OPACITY_FROM_EXPLICIT_GRAINS)
 *  Bidirectional — two independent kernels:
 *    - gas searches grains:    writes InterpolatedGeometricDustCrossSection +
 *                              Interpolated_Opacity[k_freq] to gas source.
 *    - grains search gas:      writes Interpolated_Radiation_Acceleration[3]
 *                              to grain source.
 *  Per-source output only — no j-writes, no ghost writeback needed.
 * ------------------------------------------------------------------------- */
#if defined(GRAIN_FLUID) && defined(RT_OPACITY_FROM_EXPLICIT_GRAINS)

/* Shared per-source input (both directions). */
struct GasGrainRTLocalIn
{
    int Type;
    Vec3<MyDouble> Pos, Vel;
    MyFloat KernelRadius, Mass;
    MyFloat Grain_Size;                 /* grain only; unused for gas source */
    MyFloat Grain_Abs_Coeff[N_RT_FREQ_BINS]; /* grain only; unused for gas source */
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 8)
    MyFloat Ne_gas;                     /* gas electron fraction (cell.Ne); gas source only */
    MyFloat Density_gas;                /* comoving gas density (code units); gas source only */
#endif
};

/* Per-source outputs — the kernel writes only the fields relevant to its
 * direction; the host scatter inspects the source type to dispatch. */
struct GasGrainRTOut
{
    Vec3<MyDouble> Interpolated_Radiation_Acceleration;   /* grain direction */
    MyDouble InterpolatedGeometricDustCrossSection;       /* gas direction */
    MyDouble Interpolated_Opacity[N_RT_FREQ_BINS];        /* gas direction */
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 8)
    Vec3<MyDouble> J_dust_contribution;                   /* gas direction; cgs esu/cm^2/s */
#endif
};

/* Gas source (Type==0) searches grain neighbors: computes dust opacity
 * contribution from each grain weighted by its SPH kernel at the gas position. */
KOKKOS_INLINE_FUNCTION
static void gasgrain_rt_gas_search_pair_kernel(
    const struct GasGrainRTLocalIn& local,  /* source is gas */
    int j,                                   /* neighbor is grain */
    struct particle_data *P,
    struct gas_cell_data *CellP,
    const Vec3<double>& dp,
    double r2,
    struct GasGrainRTOut& out)
{
    if(P[j].Mass <= 0) return;
    if(P[j].KernelRadius <= 0) return;
    double h_to_use = (double)P[j].KernelRadius;
    if(r2 <= 0 || r2 >= h_to_use * h_to_use) return;
    double hinv, hinv3, hinv4, wk_i = 0, dwk_i = 0;
    kernel_hinv(h_to_use, &hinv, &hinv3, &hinv4);
    double r = sqrt(r2);
    kernel_main(r * hinv, hinv3, hinv4, &wk_i, &dwk_i, 0);

    if((double)P[j].Gas_Density <= 0) return;
    double wt = (double)P[j].Mass * (wk_i / (double)P[j].Gas_Density);
    double R_grain_code = (double)P[j].Grain_Size / UNIT_LENGTH_IN_CGS;
    double rho_grain_code = All.Grain_Internal_Density / UNIT_DENSITY_IN_CGS;
    double geom = wt * 3.0 / (4.0 * rho_grain_code * R_grain_code);
    out.InterpolatedGeometricDustCrossSection += (MyDouble)geom;
    for(int k_freq = 0; k_freq < N_RT_FREQ_BINS; k_freq++) {
        double Q_abs = grain_extinction_Q_inline(j, k_freq, P);
        out.Interpolated_Opacity[k_freq] += (MyDouble)(Q_abs * geom);
    }
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 8)
    /* Dust battery J_dust accumulation (SHS25 Eq. 8). Per-grain-particle
       contribution to the gas-cell-centered current density:
         dJ_d = (Z_j q_e / m_grain_single) * M_j * wk_i * (v_grain - v_gas)
       Z_grain estimated inline from the grain Draine-Sutin tau parameter
       (same expression as solids/grain_drag_gpu.cc::grain_drag_kernel:91 for
       GRAIN_LORENTZFORCE — this is the existing per-grain-charge approximation).
       Result accumulated in cgs (esu/cm^2/s); units are conservative because
       all factors are converted to physical cgs explicitly here. */
    {
        const double R_grain_cgs = (double)P[j].Grain_Size;
        const double m_grain_single_cgs = (4.0/3.0) * M_PI * R_grain_cgs*R_grain_cgs*R_grain_cgs * All.Grain_Internal_Density;
        if((m_grain_single_cgs > 0) && (P[j].Gas_InternalEnergy > 0)) {
            const double gamma_eff = GAMMA_DEFAULT;
            const double cs_code = sqrt(gamma_eff*(gamma_eff-1.0) * (double)P[j].Gas_InternalEnergy);
            const double cs_cgs = cs_code * UNIT_VEL_IN_CGS;
            const double T_K_est = cs_cgs * cs_cgs * (2.3 * PROTONMASS_CGS)
                                   / (gamma_eff * BOLTZMANN_CGS);
            /* Use gas cell's electron density (local.Ne_gas, local.Density_gas) when
               available (bit 8 + gas source); otherwise n_e=0 returns Z≈0. */
            const double rho_gas_cgs = (double)local.Density_gas * All.cf_a3inv * UNIT_DENSITY_IN_CGS;
            const double n_eff_for_Z = rho_gas_cgs / PROTONMASS_CGS;
            const double mean_mol_weight = 2.38;  /* matches eos.cc default */
            const double n_e_for_Z = (double)local.Ne_gas * HYDROGEN_MASSFRAC
                                     * mean_mol_weight * n_eff_for_Z;
            const double a_grain_micron_j = R_grain_cgs * 1.0e4;  /* cm → microns */
            const double f_dustgas_est = 0.01;                     /* default, matches eos.cc */
            const double m_grain_in_mp  = m_grain_single_cgs / PROTONMASS_CGS;
            const double ngr_ngas_est   = (mean_mol_weight / m_grain_in_mp) * f_dustgas_est;
            double Z_grain = grain_charge_equilibrium_simple(a_grain_micron_j, n_eff_for_Z,
                                                              T_K_est, n_e_for_Z, ngr_ngas_est,
                                                              0.0 /* zeta_cr not available */, 24.3);
            const double q_per_grain_esu = Z_grain * ELECTRONCHARGE_CGS;
            const double M_j_cgs = (double)P[j].Mass * UNIT_MASS_IN_CGS;
            /* wk_i has units 1/code-volume; convert to physical cgs cm^-3:
                 wk_phys = wk_i / (UNIT_LENGTH * cf_atime)^3
               Then M_j_cgs * wk_phys gives a contribution to local gas density
               from grain particle j (g/cm^3 per (mass/density-norm) weight). */
            const double L_phys_cgs = UNIT_LENGTH_IN_CGS * All.cf_atime;
            const double wk_phys = wk_i / (L_phys_cgs * L_phys_cgs * L_phys_cgs);
            const double charge_density_cgs = (q_per_grain_esu / m_grain_single_cgs) * M_j_cgs * wk_phys;
            for(int kd = 0; kd < 3; kd++) {
                const double dv_phys_cgs = ((double)P[j].Vel[kd] - (double)local.Vel[kd])
                                           * UNIT_VEL_IN_CGS / All.cf_atime;
                out.J_dust_contribution[kd] += (MyDouble)(charge_density_cgs * dv_phys_cgs);
            }
        }
    }
#endif
}

/* Grain source (Type in GRAIN_PTYPES) searches gas neighbors: computes
 * radiation pressure acceleration on the grain from each gas cell's RT state. */
KOKKOS_INLINE_FUNCTION
static void gasgrain_rt_grain_search_pair_kernel(
    const struct GasGrainRTLocalIn& local,  /* source is grain */
    int j,                                   /* neighbor is gas */
    struct particle_data *P,
    struct gas_cell_data *CellP,
    const Vec3<double>& dp,
    double r2,
    struct GasGrainRTOut& out)
{
    if(P[j].Mass <= 0) return;
    if(P[j].KernelRadius <= 0) return;
    double h_to_use = (double)local.KernelRadius;
    if(r2 <= 0 || r2 >= h_to_use * h_to_use) return;
    double hinv, hinv3, hinv4, wk_i = 0, dwk_i = 0;
    kernel_hinv(h_to_use, &hinv, &hinv3, &hinv4);
    double r = sqrt(r2);
    kernel_main(r * hinv, hinv3, hinv4, &wk_i, &dwk_i, 0);

    double wt = (double)CellP[j].Density * All.cf_a3inv * wk_i;
    double radacc[3] = {0}, vel_i[3] = {0};
    for(int k = 0; k < 3; k++) {
        vel_i[k] = RSOL_CORRECTION_FACTOR_FOR_VELOCITY_TERMS * (double)local.Vel[k] / All.cf_atime;
    }
    for(int k_freq = 0; k_freq < N_RT_FREQ_BINS; k_freq++)
    {
        double f_kappa_abs = 0.5;
        double vdot_h[3] = {0}, flux_i[3] = {0}, flux_mag = 0, erad_i = 0, flux_corr = 1;
#if defined(RT_EVOLVE_FLUX) || (defined(RT_USE_GRAVTREE_SAVE_RAD_FLUX) && defined(RT_USE_GRAVTREE_SAVE_RAD_ENERGY))
        erad_i = (double)CellP[j].Rad_E_gamma_Pred[k_freq];
        {
            Vec3<double> v_i{vel_i[0], vel_i[1], vel_i[2]};
            Vec3<double> vdh = erad_i * (v_i + CellP[j].ET[k_freq].matvec(v_i));
            vdot_h[0] = vdh[0]; vdot_h[1] = vdh[1]; vdot_h[2] = vdh[2];
        }
        for(int k = 0; k < 3; k++) {
            flux_i[k] = (double)CellP[j].Rad_Flux_Pred[k_freq][k];
            flux_mag += flux_i[k] * flux_i[k];
        }
        if(flux_mag > 0 && isfinite(flux_mag)) { flux_mag = sqrt(flux_mag); }
        else { flux_mag = MIN_REAL_NUMBER; flux_i[0] = flux_i[1] = 0; flux_i[2] = flux_mag; }
#elif (defined(RT_OTVET) || defined(RT_FLUXLIMITEDDIFFUSION))
        erad_i = (double)CellP[j].Rad_E_gamma_Pred[k_freq];
        for(int k = 0; k < 3; k++) {
            flux_i[k] = -(double)CellP[j].Gradients.Rad_E_gamma_ET[k_freq][k];
            flux_mag += flux_i[k] * flux_i[k];
        }
        if(flux_mag > 0) { for(int k = 0; k < 3; k++) flux_i[k] /= sqrt(flux_mag); }
        else             { flux_i[0] = 0; flux_i[1] = 0; flux_i[2] = 1; }
        flux_mag = erad_i * C_LIGHT_CODE_REDUCED;
        for(int k = 0; k < 3; k++) flux_i[k] *= flux_mag;
#endif
        if(!isfinite(flux_mag) || flux_mag <= MIN_REAL_NUMBER) {
            flux_mag = MIN_REAL_NUMBER; flux_i[0] = flux_i[1] = 0; flux_i[2] = flux_mag;
        }
        double flux_thin = erad_i * C_LIGHT_CODE_REDUCED;
        if(!isfinite(flux_thin) || flux_thin <= 0) flux_thin = 0;
        flux_corr = DMIN(1.0, 100.0 * flux_thin / flux_mag);
        for(int k = 0; k < 3; k++) {
            radacc[k] += (double)local.Grain_Abs_Coeff[k_freq] * wt * (flux_corr * flux_i[k] - vdot_h[k]);
        }
    }
    for(int kk = 0; kk < 3; kk++) out.Interpolated_Radiation_Acceleration[kk] += (MyDouble)radacc[kk];
}

#endif /* GRAIN_FLUID && RT_OPACITY_FROM_EXPLICIT_GRAINS */
