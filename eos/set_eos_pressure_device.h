/* set_eos_pressure_device.h — KOKKOS_INLINE_FUNCTION version of
 * set_eos_pressure for use in the GPU cooling kernel.
 *
 * cooling.cc includes this header so the GPU cooling kernel can call
 * set_eos_pressure without crossing TU boundaries (no -rdc=true needed).
 * The inlined body's call to ThermalProperties resolves to the KOKKOS_FUNCTION
 * definition in cooling.cc itself.
 *
 * eos.cc keeps its own plain-host definition (unchanged) for all other callers.
 *
 * Include order: must come after allvars.h and after a ThermalProperties
 * forward declaration is visible (proto.h covers this).
 */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

KOKKOS_INLINE_FUNCTION
void set_eos_pressure(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    double soundspeed, press=0, temp=0, mu_meanwt=1, gamma_eos_index = cell[i].gamma_eos_value(); soundspeed=0; cell[i].Gamma = gamma_eos_index; /* get effective adiabatic index */
    press = (gamma_eos_index-1) * cell[i].InternalEnergyPred * cell[i].density_for_energy(); /* ideal gas EOS (will get over-written if more complex EOS assumed) */

#ifdef COOLING
    double ne=1, nh0=0, nHe0, nHepp, nhp, nHeII, rho_fortemp=cell[i].Density*All.cf_a3inv, u0=cell[i].InternalEnergyPred;
    temp = ThermalProperties(u0, rho_fortemp, i, &mu_meanwt, &ne, &nh0, &nhp, &nHe0, &nHeII, &nHepp, pp, cell); // get thermodynamic properties
    cell[i].Gamma = cell[i].gamma_eos_value(); // cache the adiabatic index; this will reuse the pre-computed cell[i].Temperature assigned above
#else
    temp = cell[i].InternalEnergyPred * (gamma_eos_index-1.) * PROTONMASS_CGS / (BOLTZMANN_CGS) * UNIT_ENERGY_IN_CGS / UNIT_MASS_IN_CGS; // convert to temperature for caching
#endif
    cell[i].Temperature = temp; // cache the temperature

#ifdef EOS_SUBSTELLAR_ISM
    press = cell[i].density_for_energy() * BOLTZMANN_CGS * temp / UNIT_ENERGY_IN_CGS / (mu_meanwt * PROTONMASS_CGS / UNIT_MASS_IN_CGS);
#endif

#ifdef GALSF_EFFECTIVE_EQS /* modify pressure to 'interpolate' between effective EOS and isothermal, with the Springel & Hernquist 2003 'effective' EOS */
    if(cell[i].Density*All.cf_a3inv >= All.PhysDensThresh) {press = All.FactorForSofterEQS * press + (1 - All.FactorForSofterEQS)  * (gamma_eos_index-1) * cell[i].Density * All.InitGasU;}
#endif

#ifdef EOS_HELMHOLTZ /* pass the necessary quantities to wrappers for the Timms EOS */
    struct eos_input eos_in;
    struct eos_output eos_out;
    eos_in.rho  = cell[i].Density;
    eos_in.eps  = cell[i].InternalEnergyPred;
    eos_in.Ye   = cell[i].Ye;
    eos_in.Abar = cell[i].Abar;
    eos_in.temp = cell[i].Temperature;
    int ierr = eos_compute(&eos_in, &eos_out);
    assert(!ierr);
    press      = eos_out.press;
    soundspeed = eos_out.csound;
    cell[i].Temperature = eos_out.temp;
#endif

#ifdef EOS_TILLOTSON
    press = cell[i].calculate_tillotson_eos(); soundspeed = cell[i].SoundSpeed; /* done in subroutine, save for below */
#endif

#ifdef EOS_ANEOS
    {
        int aneos_mat = cell[i].CompositionType;
        double aneos_rho_cgs = cell[i].Density * UNIT_DENSITY_IN_CGS;
        double aneos_u_cgs   = cell[i].InternalEnergyPred * UNIT_SPECEGY_IN_CGS;
        double aneos_T_guess = cell[i].Temperature;
        double aneos_P, aneos_cs, aneos_S, aneos_cv, aneos_grun;
        int aneos_phase;
        aneos_compute(aneos_mat, aneos_rho_cgs, aneos_u_cgs, &aneos_T_guess,
                      &aneos_P, &aneos_cs, &aneos_S, &aneos_cv, &aneos_grun, &aneos_phase);
        press      = aneos_P / UNIT_PRESSURE_IN_CGS;
        soundspeed = aneos_cs / UNIT_VEL_IN_CGS;
        cell[i].Temperature = aneos_T_guess;
        cell[i].PhaseID = aneos_phase;
    }
#endif

#ifdef EOS_MHD_CORE_BAROTROPIC
    press = 0.04*cell[i].Density*sqrt(1.+pow(cell[i].Density/1.47705e8 ,4./3.)); /* special barotropic EOS for core collapse test (Hopkins 2015) */
#endif
#ifdef EOS_ENFORCE_ADIABAT
    press = EOS_ENFORCE_ADIABAT * pow(cell[i].Density, gamma_eos_index);
#endif
#if defined(EOS_ENFORCE_ADIABAT) || defined(EOS_MHD_CORE_BAROTROPIC)
#ifdef TURB_DRIVING
    cell[i].EgyDiss += (cell[i].InternalEnergy - press / (cell[i].Density * (gamma_eos_index-1.))); /* save the change in energy imprinted by this enforced equation of state here */
#endif
    cell[i].InternalEnergy = cell[i].InternalEnergyPred = press / (cell[i].Density * (gamma_eos_index-1.)); /* reset internal energy: particles live -exactly- along this relation */
#endif

#ifdef EOS_GMC_BAROTROPIC
    gamma_eos_index=7./5.; double rho=cell[i].density_for_energy(), nH_cgs=rho*All.cf_a3inv*UNIT_DENSITY_IN_NHCGS;
    if(nH_cgs > 2.30181e16) {gamma_eos_index=5./3.;}
    if (nH_cgs < 1.49468e8) {press = 6.60677e-16 * nH_cgs;}
    else if (nH_cgs < 2.30181e11) {press = 1.00585e-16 * pow(nH_cgs, 1.1);}
    else if (nH_cgs < 2.30181e16) {press = 3.92567e-20 * pow(nH_cgs, gamma_eos_index);}
    else if (nH_cgs < 2.30181e21) {press = 3.1783e-15 * pow(nH_cgs, 1.1);}
    else {press = 2.49841e-27 * pow(nH_cgs, gamma_eos_index);}
#if CHECK_IF_PREPROCESSOR_HAS_NUMERICAL_VALUE_(EOS_GMC_BAROTROPIC)
#if (EOS_GMC_BAROTROPIC==1)
    if (nH_cgs < 6e10) {press = 6.60677e-16 * nH_cgs;}
    else press = 3.964062e-5 * pow(nH_cgs/6e10,1.4);
#endif
#endif
    press /= UNIT_PRESSURE_IN_CGS;
    cell[i].InternalEnergy = cell[i].InternalEnergyPred = press / (rho * (gamma_eos_index-1.));
#endif

#ifdef COSMIC_RAY_FLUID
    double soundspeed2 = gamma_eos_index*(gamma_eos_index-1) * cell[i].InternalEnergyPred;
    int k_CRegy; for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++)
    {
        press += Get_Gas_CosmicRayPressure(i, k_CRegy, cell);
        soundspeed2 += GAMMA_COSMICRAY(k_CRegy) * (GAMMA_COSMICRAY(k_CRegy)-1.) * cell[i].CosmicRayEnergyPred[k_CRegy] / cell[i].Mass;
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
        press += (1.5-1) * cell[i].Density * (cell[i].CosmicRayAlfvenEnergy[k_CRegy][0]+cell[i].CosmicRayAlfvenEnergy[k_CRegy][1]);
        soundspeed2 += 1.5*(1.5-1)*(cell[i].CosmicRayAlfvenEnergy[k_CRegy][0]+cell[i].CosmicRayAlfvenEnergy[k_CRegy][1]) / cell[i].Mass;
#endif
    }
    soundspeed = sqrt(soundspeed2);
#endif

#ifdef COSMIC_RAY_SUBGRID_LEBRON
    soundspeed = sqrt(gamma_eos_index*(gamma_eos_index-1) * cell[i].InternalEnergyPred + (4./3.)*(1./3.)*cell[i].SubGrid_CosmicRayEnergyDensity/cell[i].Density);
    press += (1./3.) * cell[i].SubGrid_CosmicRayEnergyDensity;
#endif

#ifdef RT_RADPRESSURE_IN_HYDRO
    int k_freq; double gamma_rad=4./3., fluxlim=1; double soundspeed2 = gamma_eos_index*(gamma_eos_index-1) * cell[i].InternalEnergyPred;
    if(cell[i].Mass>0 && cell[i].Density>0) {for(k_freq=0;k_freq<N_RT_FREQ_BINS;k_freq++)
    {
        press += (gamma_rad-1.) * cell[i].flux_limiter(k_freq) * cell[i].Rad_E_gamma_Pred[k_freq] * cell[i].Density / cell[i].Mass;
        soundspeed2 += gamma_rad*(gamma_rad-1.) * cell[i].Rad_E_gamma_Pred[k_freq] / cell[i].Mass;
    }}
    soundspeed = sqrt(soundspeed2);
#endif

#if defined(EOS_TRUELOVE_PRESSURE) || defined(TRUELOVE_CRITERION_PRESSURE)
    double h_eff = std::max(pp[i].Get_Particle_Size(), KERNEL_FAC_FROM_FORCESOFT_TO_PLUMMER*ForceSoftening_KernelRadius(i));
    double NJeans = 4;
    double xJeans = (NJeans * NJeans / gamma_eos_index) * All.G * h_eff*h_eff * cell[i].Density * cell[i].Density /All.cf_atime;
    if(xJeans>press) press=xJeans;
#endif

#if defined(HYDRO_GENERATE_TARGET_MESH)
    press = return_user_desired_target_pressure(i) * (cell[i].Density / return_user_desired_target_density(i));
    cell[i].InternalEnergy = cell[i].InternalEnergyPred = return_user_desired_target_pressure(i) / ((gamma_eos_index-1) * cell[i].Density);
#endif

#ifdef EOS_GENERAL /* need to be sure soundspeed variable is set */
    if(soundspeed == 0) {cell[i].SoundSpeed = sqrt(gamma_eos_index * press / cell[i].density_for_energy());} else {cell[i].SoundSpeed = soundspeed;}
#endif

    /* Finally, set the pressure as advertised */
    cell[i].Pressure = press;
}
