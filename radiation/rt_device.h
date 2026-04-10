/* rt_device.h — KOKKOS_INLINE_FUNCTION stubs for RT functions called from
 * the GPU cooling chain.  rt_utilities.cc is NOT in GPU_OBJS.
 *
 * These are used when the RT modules (RT_INFRARED, RT_CHEM_PHOTOION, etc.)
 * are active.  For configurations without those modules, the callers are
 * behind #ifdefs and these are never reached at runtime, but nvcc still
 * needs device-callable declarations to avoid #20011 warnings and stubs. */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

/* rt_kappa_adaptive_IR_band: only meaningful when RT_INFRARED is active.
 * Full implementation requires dust_planck_mean_opacity tables which are
 * not yet ported to device.  Provide a device-callable version that returns
 * a reasonable default so the compiler doesn't generate a host stub. */
#ifdef RT_INFRARED
KOKKOS_INLINE_FUNCTION
double rt_kappa_adaptive_IR_band(int i, double T_dust, double Trad, int do_emission_absorption_scattering_opacity, int dust_or_gas_opacity_only_flag, struct particle_data *pp, struct gas_cell_data *cell)
{
    /* TODO: port full Semenov opacity tables to device.  For now return a
     * reasonable Rosseland-mean estimate: kappa ~ 0.1 Z (T/10K)^2 cm^2/g
     * capped at 5 Z cm^2/g, converted to code units. */
    double Zfac = 1.0;
#ifdef METALS
    if(i>=0) {Zfac = pp[i].Metallicity[0]/All.SolarAbundances[0];}
#endif
    double T_use = (do_emission_absorption_scattering_opacity==1) ? T_dust : Trad;
    double kappa_cgs = DMIN(0.1 * Zfac * (T_use/10.)*(T_use/10.), 5.*Zfac);
    return kappa_cgs / UNIT_SURFDEN_IN_CGS;
}

KOKKOS_INLINE_FUNCTION
double rt_eqm_dust_temp(int i, double T, double dust_absorption_rate, struct particle_data *pp, struct gas_cell_data *cell)
{
    /* Simple estimate: T_dust ~ T_gas for tightly coupled regime */
    return DMAX(DMIN(T, 1500.), 2.73);
}
#endif /* RT_INFRARED */

/* rt_irband_egydensity_in_band: returns energy density in a given frequency range
 * from the adaptive IR band.  Only called under RT_INFRARED. */
#ifdef RT_INFRARED
KOKKOS_INLINE_FUNCTION
double rt_irband_egydensity_in_band(int i, double nu_lo_eV, double nu_hi_eV, struct gas_cell_data *cell)
{
    /* Approximate: fraction of IR band in the given frequency range.
     * For the cooling chain this is a small correction; return 0 if not
     * properly ported yet. */
    return 0;
}
#endif
