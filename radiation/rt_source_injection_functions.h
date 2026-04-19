/* rt_source_injection_functions.h — GPU-callable per-source-particle struct and
 * per-pair injection kernel for rt_source_injection GPU port (B5).
 *
 * Include after: allvars.h, kernel.h, particle_data.h, gas_cell_data.h,
 *   rt_functions.h, sinks/sink_functions.h.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#ifdef RT_SOURCE_INJECTION

#include "rt_functions.h"
#include "../sinks/sink_functions.h"

/* GPU-local copy renamed to avoid conflict with non-static proto.h declaration */
KOKKOS_INLINE_FUNCTION
static double rt_slab_avg(double x)
{
    return (1.00000000000 + x*(0.21772719088733913 + x*(0.047076512011644776 + x*0.005068307557496351))) /
           (1.00000000000 + x*(0.71772719088733920 + x*(0.239273440788647680 + x*(0.046750496137263675 + x*0.005068307557496351))));
}

/* Per-source-particle input — mirrors CPU INPUT_STRUCT_NAME. */
struct RtSrcLocalIn {
    Vec3<MyDouble> Pos;
    MyFloat KernelRadius;
    MyFloat KernelSum_Around_RT_Source;
    MyFloat Luminosity[N_RT_FREQ_BINS];
#if defined(RT_EVOLVE_FLUX)
    Vec3<MyFloat> Vel;
#endif
#if defined(RT_REPROCESS_INJECTED_PHOTONS) && defined(RT_CHEM_PHOTOION)
    MyDouble Dt;
    MyDouble Density;
#endif
};


/* Per-pair injection kernel: source local -> gas receiver j.
 * Caller ensures j is a gas particle with Mass > 0, r > 0, inside kernel. */
KOKKOS_INLINE_FUNCTION
static void rt_source_injection_pair_kernel(
    const struct RtSrcLocalIn& local,
    int j,
    struct particle_data *P,
    struct gas_cell_data *CellP,
    double r2, const Vec3<double>& dp)
{
    double r = sqrt(r2);
    double hinv, hinv3, hinv4;
    kernel_hinv(local.KernelRadius, &hinv, &hinv3, &hinv4);
    double wk;
#ifdef RT_SINK_ANGLEWEIGHT_PHOTON_INJECTION
    if(All.TimeStep > 0) {
        wk = sink_fb_angleweight_localcoupling_gpu(P[j], CellP[j], 0., r, local.KernelRadius)
             / local.KernelSum_Around_RT_Source;
    } else {
        wk = (1. - r2*hinv*hinv) / local.KernelSum_Around_RT_Source;
    }
#else
    wk = (1. - r2*hinv*hinv) / local.KernelSum_Around_RT_Source;
#endif

#ifdef RT_EVOLVE_INTENSITIES
    int kx; double angle_wt_Inu_sum=0, angle_wt_Inu[N_RT_INTENSITY_BINS];
    for(kx=0; kx<N_RT_INTENSITY_BINS; kx++) {
        double cos_t=0;
        for(int kq=0; kq<3; kq++) { cos_t += All.Rad_Intensity_Direction[kx][kq]*dp[kq]/r; }
        double wt = cos_t*cos_t*cos_t*cos_t; if(cos_t < 0) wt=0;
        angle_wt_Inu[kx] = wt; angle_wt_Inu_sum += wt;
    }
#endif

    for(int k=0; k<N_RT_FREQ_BINS; k++) {
        double dE = wk * local.Luminosity[k];
        Vec3<double> dfluxes = {};

#if !defined(RT_INJECT_PHOTONS_DISCRETELY)
        Kokkos::atomic_add(&CellP[j].Rad_Je[k], dE);
#endif

#if defined(RT_INJECT_PHOTONS_DISCRETELY_ADD_MOMENTUM_FOR_LOCAL_EXTINCTION) || defined(RT_REPROCESS_INJECTED_PHOTONS)
        double x_abs = 2. * CellP[j].Rad_Kappa[k] * (CellP[j].Density*All.cf_a3inv)
                       * (DMAX(2.*P[j].Get_Particle_Size(), DMAX(local.KernelRadius, r))) * All.cf_atime;
        double slabfac_x = x_abs * rt_slab_avg(x_abs);
        if(slabfac_x != slabfac_x || slabfac_x <= 0) { slabfac_x=0; } else if(slabfac_x > 1) { slabfac_x=1; }
#if !defined(RT_DISABLE_RAD_PRESSURE) && defined(RT_INJECT_PHOTONS_DISCRETELY_ADD_MOMENTUM_FOR_LOCAL_EXTINCTION)
        double dv = -slabfac_x * dE / (C_LIGHT_CODE_REDUCED * P[j].Mass);
        for(int kv=0; kv<3; kv++) {
            double dv_tmp = dv * (dp[kv]/r) * All.cf_atime;
            Kokkos::atomic_add(&P[j].Vel[kv], dv_tmp);
            Kokkos::atomic_add(&CellP[j].VelPred[kv], dv_tmp);
            Kokkos::atomic_add(&P[j].dp[kv], dv_tmp * P[j].Mass);
        }
#endif
#endif

#ifdef RT_REPROCESS_INJECTED_PHOTONS
        double dE_donation=0;
        int donation_bin=rt_get_donation_target_bin(k), do_donation=(donation_bin > -1) ? 1 : 0;
#ifdef RT_CHEM_PHOTOION
        if(RT_BAND_IS_IONIZING(k)) {
            double stellum=0;
            if(local.Dt > 0) {
#if (RT_CHEM_PHOTOION==1)
                stellum = local.Luminosity[RT_FREQ_BIN_H0];
#else
                for(int k2=RT_FREQ_BIN_H0; k2 < RT_FREQ_BIN_H0+4; k2++) { stellum += local.Luminosity[k2]; }
#endif
                stellum *= 1./(C_LIGHT_CODE_REDUCED/C_LIGHT_CODE)/local.Dt*UNIT_LUM_IN_CGS;
            }
            double RHII = 4.01e-9*pow(stellum,0.333)*pow(local.Density*All.cf_a3inv*UNIT_DENSITY_IN_CGS,-0.66667)/UNIT_LENGTH_IN_CGS;
            if(DMAX(r, P[j].Get_Particle_Size())*All.cf_atime < RHII) { do_donation=0; }
        }
#endif
#ifdef RT_INFRARED
        if(k == RT_FREQ_BIN_INFRARED) { do_donation=0; }
#endif
        if(do_donation) { dE_donation = slabfac_x*dE; dE *= fabs(1.-slabfac_x); }
#endif /* RT_REPROCESS_INJECTED_PHOTONS */

#if defined(RT_EVOLVE_FLUX) && defined(RT_INJECT_PHOTONS_DISCRETELY_ADD_MOMENTUM_FOR_LOCAL_EXTINCTION)
        {
            double dflux = -dE * C_LIGHT_CODE_REDUCED / r;
            dfluxes += dflux * dp;
        }
#endif

#if defined(RT_INJECT_PHOTONS_DISCRETELY)
        Kokkos::atomic_add(&CellP[j].Rad_E_gamma[k], dE);
#ifdef RT_EVOLVE_ENERGY
        Kokkos::atomic_add(&CellP[j].Rad_E_gamma_Pred[k], dE);
#endif
#ifdef RT_REPROCESS_INJECTED_PHOTONS
        if(donation_bin > -1) {
            Kokkos::atomic_add(&CellP[j].Rad_E_gamma[donation_bin], dE_donation);
#ifdef RT_EVOLVE_ENERGY
            Kokkos::atomic_add(&CellP[j].Rad_E_gamma_Pred[donation_bin], dE_donation);
#endif
        }
#endif
#ifdef RT_EVOLVE_INTENSITIES
        {
            double dflux_int = dE / angle_wt_Inu_sum;
            for(int kv=0; kv<N_RT_INTENSITY_BINS; kv++) {
                double dI_temp = dflux_int * angle_wt_Inu[kv];
                Kokkos::atomic_add(&CellP[j].Rad_Intensity[k][kv], dI_temp);
                Kokkos::atomic_add(&CellP[j].Rad_Intensity_Pred[k][kv], dI_temp);
            }
        }
#endif
#if defined(RT_EVOLVE_FLUX)
        {
            for(int kv=0; kv<3; kv++) {
                dfluxes[kv] += dE * (RSOL_CORRECTION_FACTOR_FOR_VELOCITY_TERMS * local.Vel[kv] / All.cf_atime);
            }
        }
#ifdef GRAIN_RDI_TESTPROBLEM_LIVE_RADIATION_INJECTION
        {
            double qtau=(0.75*All.Grain_Q_at_MaxGrainSize)/((All.Grain_Internal_Density/UNIT_DENSITY_IN_CGS)*(All.Grain_Size_Max/UNIT_LENGTH_IN_CGS));
            double e0=(P[j].Mass/CellP[j].Density)*All.Vertical_Grain_Accel/qtau;
            double tau_tot=All.Dust_to_Gas_Mass_Ratio*qtau;
            double flux_egy_0=C_LIGHT_CODE_REDUCED*DMAX(CellP[j].Rad_E_gamma[k],CellP[j].Rad_E_gamma_Pred[k])/(1.+3.*tau_tot);
            double f0=DMAX(flux_egy_0, DMAX(C_LIGHT_CODE_REDUCED*e0, DMAX(CellP[j].Rad_Flux[k][2],CellP[j].Rad_Flux_Pred[k][2])));
            dfluxes[0]=0; dfluxes[1]=0; dfluxes[2]=f0;
            /* assignment (not add) for this special case */
            Kokkos::atomic_exchange(&CellP[j].Rad_Flux[k][0], (MyFloat)0.);
            Kokkos::atomic_exchange(&CellP[j].Rad_Flux[k][1], (MyFloat)0.);
            Kokkos::atomic_exchange(&CellP[j].Rad_Flux[k][2], (MyFloat)0.);
            Kokkos::atomic_exchange(&CellP[j].Rad_Flux_Pred[k][0], (MyFloat)0.);
            Kokkos::atomic_exchange(&CellP[j].Rad_Flux_Pred[k][1], (MyFloat)0.);
            Kokkos::atomic_exchange(&CellP[j].Rad_Flux_Pred[k][2], (MyFloat)0.);
        }
#endif
        for(int kv=0; kv<3; kv++) {
            Kokkos::atomic_add(&CellP[j].Rad_Flux[k][kv], (MyFloat)dfluxes[kv]);
            Kokkos::atomic_add(&CellP[j].Rad_Flux_Pred[k][kv], (MyFloat)dfluxes[kv]);
        }
#endif /* RT_EVOLVE_FLUX */
#endif /* RT_INJECT_PHOTONS_DISCRETELY */
    } /* for k < N_RT_FREQ_BINS */
}

#endif /* RT_SOURCE_INJECTION */
