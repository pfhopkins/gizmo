/* thermal_fb_functions.h — GPU-callable per-source struct and per-pair kernel
 * for thermal_fb GPU port (B6).
 *
 * Include after: allvars.h, kernel.h, particle_data.h, gas_cell_data.h.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#ifdef GALSF_FB_THERMAL

/* Per-source-particle input — mirrors CPU INPUT_STRUCT_NAME fields used in kernel. */
struct ThermalFBLocalIn {
    Vec3<MyDouble> Pos;
    MyFloat KernelRadius;
    MyFloat wt_sum;       /* P[i].DensityAroundParticle */
    MyFloat Msne;         /* ejecta mass */
    MyFloat Esne;         /* ejecta kinetic energy = 0.5*Msne*v_ej^2 */
    MyFloat kernel_zero;  /* kernel_main(0.0,1.0,1.0,...) — center kernel value */
#ifdef METALS
    MyFloat yields[NUM_METAL_SPECIES];
#endif
};

/* Per-source output: mass returned to caller for source particle update. */
struct ThermalFBOut {
    MyDouble M_coupled;
};


/* Per-pair kernel: source local -> gas receiver j.
 * Caller ensures r2 > 0 and r2 < KernelRadius^2 (pre-checked by neighbor list + r2 guard
 * in the lambda).  dp = local.Pos - P[j].Pos (with nearest_xyz applied). */
KOKKOS_INLINE_FUNCTION
static void thermal_fb_pair_kernel(
    const struct ThermalFBLocalIn& local,
    int j,
    struct particle_data *P,
    struct gas_cell_data *CellP,
    double r2,
    const Vec3<double>& dp,
    struct ThermalFBOut& out)
{
    if(P[j].Type != 0) return;
    double Mass_j = P[j].Mass;
    if(Mass_j <= 0) return;
    double h2 = (double)local.KernelRadius * (double)local.KernelRadius;
    if(r2 >= h2 || r2 <= 0) return;

    double hinv, hinv3, hinv4;
    kernel_hinv((double)local.KernelRadius, &hinv, &hinv3, &hinv4);
    double r = sqrt(r2);
    double u = r * hinv;
    double wk, dwk;
    if(u < 1) { kernel_main(u, hinv3, hinv4, &wk, &dwk, 0); } else { wk = dwk = 0; }
    if(wk <= 0 || wk != wk) return;

    double rho_j_0 = CellP[j].Density;

    /* kernel-weighted fraction of ejecta for this neighbor */
    wk = Mass_j * wk / (double)local.wt_sum;

    /* mass injected */
    double dM = wk * (double)local.Msne;

    /* delta density */
    double delta_rho = 0;
    if(P[j].KernelRadius <= 0) {
        if(rho_j_0 > 0) { delta_rho = rho_j_0 * dM / Mass_j; }
        else             { delta_rho = (double)local.kernel_zero * dM * hinv3; }
    } else {
        double kr3 = (double)P[j].KernelRadius * (double)P[j].KernelRadius * (double)P[j].KernelRadius;
        delta_rho = (double)local.kernel_zero * dM / kr3;
    }

    /* energy: IE += wk*Esne / Mass_j */
    double dIE = wk * (double)local.Esne / Mass_j;

    /* atomic updates to j */
    Kokkos::atomic_add(&P[j].Mass, (MyDouble)dM);
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    Kokkos::atomic_add(&CellP[j].MassTrue, (MyDouble)dM);
#endif
    if(delta_rho != 0) {
        Kokkos::atomic_add(&CellP[j].Density, (MyDouble)delta_rho);
    }
    for(int k = 0; k < 3; k++) {
        Kokkos::atomic_add(&P[j].dp[k], (MyFloat)(dM * P[j].Vel[k]));
    }
    Kokkos::atomic_add(&CellP[j].InternalEnergy,     (MyDouble)dIE);
    Kokkos::atomic_add(&CellP[j].InternalEnergyPred, (MyDouble)dIE);

#ifdef METALS
    for(int k = 0; k < NUM_METAL_SPECIES; k++) {
        double dMet = (dM / Mass_j) * ((double)local.yields[k] - (double)P[j].Metallicity[k]);
        Kokkos::atomic_add(&P[j].Metallicity[k], (MyDouble)dMet);
    }
#endif

#ifdef GALSF_FB_TURNOFF_COOLING
    {
        double Esne51 = (double)local.Esne * UNIT_ENERGY_IN_CGS / 1.e51;
        double density_to_n = All.cf_a3inv * UNIT_DENSITY_IN_NHCGS;
        double pressure_to_p4 = density_to_n * U_TO_TEMP_UNITS / 1.0e4;
        double dt_ram = 7.08 * pow(Esne51 * rho_j_0 * density_to_n, 0.34)
                        * pow(CellP[j].Pressure * pressure_to_p4, -0.70) / UNIT_TIME_IN_MYR;
        Kokkos::atomic_max(&CellP[j].DelayTimeCoolingSNe, (MyDouble)dt_ram);
    }
#endif

    out.M_coupled += dM;
}

#endif /* GALSF_FB_THERMAL */
