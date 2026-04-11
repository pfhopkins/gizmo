/* dust_to_metals_functions.h — Canonical KOKKOS_INLINE_FUNCTION implementation
 * of return_dust_to_metals_ratio_vs_solar.  Single source of truth for both
 * CPU and GPU paths.
 *
 * Include order: after allvars.h (for struct types, All). */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

KOKKOS_INLINE_FUNCTION
double return_dust_to_metals_ratio_vs_solar(int i, double T_dust_manual_override, struct particle_data *pp, struct gas_cell_data *cell)
{
    if(i<0 || pp[i].Type!=0) {return 1;}
#if defined(RT_OPACITY_FROM_EXPLICIT_GRAINS)
    double kappa_interp_geo_cgs = cell[i].InterpolatedGeometricDustCrossSection / UNIT_SURFDEN_IN_CGS;
    double kappa_solar_geo_cgs = 3300.;
    double Z_scaled = pp[i].Metallicity[0]/All.SolarAbundances[0];
    return (kappa_interp_geo_cgs / kappa_solar_geo_cgs) / (Z_scaled);
#endif
#if defined(GALSF_ISMDUSTCHEM_MODEL)
    if(pp[i].Metallicity[0]>0) {return (cell[i].ISMDustChem_Dust_Metal[0]/pp[i].Metallicity[0])/0.5;} else {return 0;}
#endif
#if defined(RT_INFRARED)
    double T_evap = 1500.;
    double T_dust = T_dust_manual_override; if(T_dust == 0) {T_dust = cell[i].Dust_Temperature;}
    double Tdust_Tsub = T_dust / T_evap;
    double fdust = sigmoid_sqrt(9.*(1.-Tdust_Tsub)) * exp(-DMIN(40.,Tdust_Tsub*Tdust_Tsub/9.));
    return DMAX(fdust, 1.e-25);
#endif
#if defined(COOL_LOW_TEMPERATURES) && !defined(SINGLE_STAR_SINK_DYNAMICS)
    double Tdust = T_dust_manual_override; if(Tdust == 0) {Tdust = get_equilibrium_dust_temperature_estimate(i,0,0, pp, cell);}
    if(Tdust >= 2000.) {return 1.e-4;} else {return exp(-pow(Tdust/1000.,3));}
#endif
    return 1;
}
