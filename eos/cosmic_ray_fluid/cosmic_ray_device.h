/* cosmic_ray_device.h — KOKKOS_INLINE_FUNCTION versions of CR utility
 * functions needed by the GPU cooling kernel.  The originals live in
 * cosmic_ray_utilities.cc which is NOT in GPU_OBJS.
 *
 * Only the code paths relevant to non-explicit-CR-bins builds (i.e. the
 * default COOL_LOW_TEMPERATURES path with no COSMIC_RAY_FLUID) are included.
 * The explicit N_CR_PARTICLE_BINS > 2 paths call return_CRbin_* helpers that
 * are too complex to inline here and would need a deeper port.
 */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

KOKKOS_INLINE_FUNCTION
double Get_CosmicRayEnergyDensity_cgs(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    if(i<0) {return 0;}
#ifdef COSMIC_RAY_FLUID
    double u_cr=0; int k; for(k=0;k<N_CR_PARTICLE_BINS;k++) {u_cr += cell[i].CosmicRayEnergyPred[k];}
    return u_cr * (cell[i].Density*All.cf_a3inv / pp[i].Mass) * UNIT_PRESSURE_IN_CGS;
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    return cell[i].SubGrid_CosmicRayEnergyDensity*All.cf_a3inv * UNIT_PRESSURE_IN_CGS;
#endif
#ifdef RT_ISRF_BACKGROUND
    double column = evaluate_NH_from_GradRho(pp[i].GradRho,pp[i].KernelRadius,cell[i].Density,pp[i].NumNgb,1,i,pp) * UNIT_SURFDEN_IN_CGS, sigma_0=2.23e-3;
    double u_cr_0 = sqrt(All.InterstellarRadiationFieldStrength) * 1.6e-12;
    if(column < sigma_0) {return u_cr_0;} else {
        double atten_fac = exp(DMAX(-column/100.,-90)) * (sigma_0/(column+MIN_REAL_NUMBER));
        return atten_fac * u_cr_0;
    }
#endif
    return 1.6e-12; // eV/cm-3 default
}

KOKKOS_INLINE_FUNCTION
double Get_CosmicRayIonizationRate_cgs(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    double zeta_cr = 0;
#if defined(COSMIC_RAY_FLUID) && (N_CR_PARTICLE_BINS > 2)
    /* explicit multi-bin CR spectrum: too complex for device inline, fall back to
     * approximate scaling from energy density for now on GPU */
    zeta_cr = 1.e-5 * Get_CosmicRayEnergyDensity_cgs(i, pp, cell);
#else
    zeta_cr = 1.e-5 * Get_CosmicRayEnergyDensity_cgs(i, pp, cell);
#if !defined(COSMIC_RAY_FLUID) && !defined(COSMIC_RAY_SUBGRID_LEBRON) && !defined(RT_ISRF_BACKGROUND)
    double prefac_CR=1.; if(All.ComovingIntegrationOn) {double rhofac = (cell[i].Density*All.cf_a3inv*UNIT_DENSITY_IN_CGS) / (1000.*COSMIC_BARYON_DENSITY_CGS); if(rhofac < 0.2) {prefac_CR=0;} else {if(rhofac > 200.) {prefac_CR=1;} else {prefac_CR=exp(-1./(rhofac*rhofac));}}}
    zeta_cr *= prefac_CR;
#endif
#endif
#ifdef METALS
    zeta_cr += 1.e-21 * pp[i].Metallicity[0]/All.SolarAbundances[0];
#if (NUM_METAL_SPECIES > 10)
    zeta_cr += 1.e-19 * pp[i].Metallicity[10]/All.SolarAbundances[10];
#endif
#endif
    return zeta_cr;
}

KOKKOS_INLINE_FUNCTION
double CR_gas_heating(int target, double n_elec, double nH0, double nHcgs, struct particle_data *pp, struct gas_cell_data *cell)
{
#if defined(CRFLUID_ALT_DISABLE_LOSSES)
    return 0;
#endif
    double a_hadronic, b_coulomb_ion_per_GeV, f_heat_hadronic;
    a_hadronic = 6.37e-16; b_coulomb_ion_per_GeV = 3.09e-16*(n_elec + 0.57*nH0)*HYDROGEN_MASSFRAC; f_heat_hadronic=1./6.;
#if defined(COSMIC_RAY_FLUID) || defined(COSMIC_RAY_SUBGRID_LEBRON)
#if (N_CR_PARTICLE_BINS > 2)
    /* multi-bin: approximate on device */
    return (0.87*f_heat_hadronic*a_hadronic + 0.53*b_coulomb_ion_per_GeV) * Get_CosmicRayEnergyDensity_cgs(target, pp, cell) / nHcgs;
#else
    return (0.87*f_heat_hadronic*a_hadronic + 0.53*b_coulomb_ion_per_GeV) * Get_CosmicRayEnergyDensity_cgs(target, pp, cell) / nHcgs;
#endif
#elif defined(COOL_LOW_TEMPERATURES)
    double prefac_CR=1.; if(All.ComovingIntegrationOn) {double rhofac = (PROTONMASS_CGS*nHcgs/HYDROGEN_MASSFRAC) / (1000.*COSMIC_BARYON_DENSITY_CGS); if(rhofac < 0.2) {prefac_CR=0;} else {if(rhofac > 200.) {prefac_CR=1;} else {prefac_CR=exp(-1./(rhofac*rhofac));}}}
#ifdef RT_ISRF_BACKGROUND
    prefac_CR *= Get_CosmicRayEnergyDensity_cgs(target, pp, cell)/1.6e-12;
#endif
    return (0.87*f_heat_hadronic*a_hadronic + 0.53*b_coulomb_ion_per_GeV) * (1.6e-12*prefac_CR) / (1.e-2 + nHcgs);
#else
    return 0;
#endif
}

