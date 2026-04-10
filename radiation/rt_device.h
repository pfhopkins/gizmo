/* rt_device.h — KOKKOS_INLINE_FUNCTION versions of RT functions called from
 * the GPU cooling chain.  rt_utilities.cc is NOT in GPU_OBJS.
 *
 * rt_kappa_adaptive_IR_band is called from CoolingRate and
 * get_equilibrium_dust_temperature_estimate even WITHOUT RT_INFRARED
 * (under COOL_LOW_TEMPERATURES for optically-thick cooling).
 * rt_eqm_dust_temp is called from get_equilibrium_dust_temperature_estimate.
 * rt_irband_egydensity_in_band is only called under RT_INFRARED. */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

/* Always available — called from COOL_LOW_TEMPERATURES path regardless of RT_INFRARED */
KOKKOS_INLINE_FUNCTION
double rt_kappa_adaptive_IR_band(int i, double T_dust, double Trad, int do_emission_absorption_scattering_opacity, int dust_or_gas_opacity_only_flag, struct particle_data *pp, struct gas_cell_data *cell)
{
    /* Approximate Rosseland-mean dust opacity: kappa ~ 0.1 Z (T/10K)^2 cm^2/g
     * capped at 5 Z cm^2/g, converted to code units.  Full Semenov tables
     * not yet ported to device. */
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

#ifdef RT_INFRARED
KOKKOS_INLINE_FUNCTION
double rt_irband_egydensity_in_band(int i, double nu_lo_eV, double nu_hi_eV, struct gas_cell_data *cell)
{
    return 0;
}
#endif
