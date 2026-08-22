#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../nuclear/nuclear.h"   /* nuclear_neutrino_ye_feedback (NUCLEAR_NETWORK_NEUTRINOS) */
#include "../mesh/kernel.h"
/* This TU emits the strong host-symbol externals for rt_functions.h. Pre-include
 * the source-helper leaf headers with inline-semantic KOKKOS_INLINE_FUNCTION so
 * their #pragma once / include guards lock them BEFORE the non-inline re-include
 * of rt_functions.h below; otherwise the non-inline pass would re-emit their
 * inline helpers as strong symbols, duplicating the owning TUs and failing the
 * link (the pattern eos_functions.h already relies on). GRAVTREE_SOURCE_HOST_OWNER_TU
 * opens the source-helper gate for the host pass regardless of compiler. */
#define GRAVTREE_SOURCE_HOST_OWNER_TU
#include "../eos/eos_functions.h"
#include "../gravity/cosmology_functions.h"
#include "../galaxy_sf/stellar_evolution_functions.h"
#include "../sinks/sink_functions.h"
#include "../core/predict_functions.h"
/* Function bodies now in rt_functions.h (single source of truth).
   Define KOKKOS_INLINE_FUNCTION as empty so functions are non-inline here,
   providing externally-visible symbols for other TUs that link via proto.h. */
#undef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION
#include "rt_functions.h"
#undef GRAVTREE_SOURCE_HOST_OWNER_TU

/*! \file rt_utilities.c
 *  \brief useful functions for radiation modules
 *
 *  This file contains a variety of useful functions having to do with radiation in different modules
 *    A number of the radiative transfer subroutines and more general mass-to-light ratio calculations
 *    will refer to these routines.
 */
/*
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */



/***********************************************************************************************************
 *
 * ROUTINES IN THIS BLOCK MUST BE MODIFIED FOR NEW MODULES USING DIFFERENT WAVEBANDS/PHYSICS
 *
 *  (these routines depend on compiler-time choices for which frequencies will be followed, and the 
 *    physics used to determine things like the types of source particles, source luminosities, 
 *    and how opacities are calculated)
 *
 ***********************************************************************************************************/



#if defined(RADTRANSFER) || defined(RT_USE_GRAVTREE)



/***********************************************************************************************************/
/* routine which returns the luminosity [total volume/mass integrated] for the desired source particles in physical code units (energy/time),
    as a function of whatever the user desires, in the relevant bands. inpute here:
    'i' = index of target particle/cell for which the luminosity should be computed
    'mode' = flag for special behaviors. if <0 (e.g. -1), just returns whether or not a particle is 'active' (eligible as an RT source). if =0, normal behavior. if =1, then some bands have special behavior, for example self-shielding estimated -at the source-
    'lum' = pointer to vector of length N_RT_FREQ_BINS to hold luminosities for all bands.
    Note that for a number of the bands below where 'sources' are stars or star-like objects,
        there are two default 'versions' implemented: one assuming the sources are -individual- stars/protostars/compact objects, the other
        assuming the sources represent stellar -populations-. Make sure you implement the correct assumptions with appropriate flags for the simulations you wish to run.
 */
/***********************************************************************************************************/
/* rt_get_source_luminosity, rt_get_lum_band_{stellarpopulation,agn,singlestar}:
   bodies now in rt_functions.h (single source for host + device). */

/* host external wrapper for the ionizing-luminosity core (body in rt_functions.h) */
#if defined(GALSF_FB_FIRE_RT_HIIHEATING) || (defined(RT_CHEM_PHOTOION) && defined(GALSF))
double particle_ionizing_luminosity_in_cgs(long i)
{
    return particle_ionizing_luminosity_in_cgs_core(i, P);
}
#endif




/* this initializes the list of the effective min and max frequencies associated with each waveband, to be used throughout the code */
void rt_define_effective_frequencies_in_bands(void)
{
    /* initialize the table that contains the effective wavelengths of all the different bansd we are actually evolving */
    int k; double rhd_bins_nu_min_ev[N_RT_FREQ_BINS], rhd_bins_nu_max_ev[N_RT_FREQ_BINS]; for(k=0;k<N_RT_FREQ_BINS;k++) {rhd_bins_nu_min_ev[k]=0; rhd_bins_nu_max_ev[k]=MAX_REAL_NUMBER;}
#ifdef RT_CHEM_PHOTOION
#if defined(RT_PHOTOION_MULTIFREQUENCY)
    int i_vec[4] = {RT_FREQ_BIN_H0, RT_FREQ_BIN_He0, RT_FREQ_BIN_He1, RT_FREQ_BIN_He2};
    rhd_bins_nu_min_ev[i_vec[3]]=All.rt_ion_nu_min[i_vec[3]]; rhd_bins_nu_max_ev[i_vec[3]]=500; for(k=0;k<3;k++) {rhd_bins_nu_min_ev[i_vec[k]]=All.rt_ion_nu_min[i_vec[k]]; rhd_bins_nu_max_ev[i_vec[k]]=All.rt_ion_nu_min[i_vec[k+1]];}
#else
    k=RT_FREQ_BIN_H0; rhd_bins_nu_min_ev[k]=13.6; rhd_bins_nu_max_ev[k]=500;
#endif
#endif
#ifdef RT_SOFT_XRAY
    k=RT_FREQ_BIN_SOFT_XRAY; rhd_bins_nu_min_ev[k]=500; rhd_bins_nu_max_ev[k]=2000;
#endif
#ifdef RT_HARD_XRAY
    k=RT_FREQ_BIN_HARD_XRAY; rhd_bins_nu_min_ev[k]=2000; rhd_bins_nu_max_ev[k]=10000;
#endif
#ifdef RT_PHOTOELECTRIC
    k=RT_FREQ_BIN_PHOTOELECTRIC; rhd_bins_nu_min_ev[k]=8; rhd_bins_nu_max_ev[k]=13.6;
#endif
#ifdef RT_LYMAN_WERNER
    k=RT_FREQ_BIN_LYMAN_WERNER; rhd_bins_nu_min_ev[k]=11.2; rhd_bins_nu_max_ev[k]=13.6;
#endif
#ifdef RT_NUV
    k=RT_FREQ_BIN_NUV; rhd_bins_nu_min_ev[k]=3.444; rhd_bins_nu_max_ev[k]=8.;
#endif
#ifdef RT_OPTICAL_NIR
    k=RT_FREQ_BIN_OPTICAL_NIR; rhd_bins_nu_min_ev[k]=0.4133; rhd_bins_nu_max_ev[k]=3.444;
#endif
#ifdef RT_GENERIC_USER_FREQ
    k=RT_FREQ_BIN_GENERIC_USER_FREQ; rhd_bins_nu_min_ev[k]=0; rhd_bins_nu_max_ev[k]=MAX_REAL_NUMBER;
#endif
#ifdef GALSF_FB_FIRE_RT_LONGRANGE
    k=RT_FREQ_BIN_FIRE_UV; rhd_bins_nu_min_ev[k]=3.444; rhd_bins_nu_max_ev[k]=13.6;
    k=RT_FREQ_BIN_FIRE_OPT; rhd_bins_nu_min_ev[k]=0.365; rhd_bins_nu_max_ev[k]=3.444;
    k=RT_FREQ_BIN_FIRE_IR; rhd_bins_nu_min_ev[k]=0.01; rhd_bins_nu_max_ev[k]=0.365;
#endif
#ifdef RT_INFRARED
    k=RT_FREQ_BIN_INFRARED; rhd_bins_nu_min_ev[k]=0.001; rhd_bins_nu_max_ev[k]=0.4133;
#endif
#ifdef RT_FREEFREE
    k=RT_FREQ_BIN_FREEFREE; rhd_bins_nu_min_ev[k]=0; rhd_bins_nu_max_ev[k]=MAX_REAL_NUMBER;
#endif
#ifdef RT_NEUTRINO_ELECTRON
    k=RT_FREQ_BIN_NU_E;    rhd_bins_nu_min_ev[k]=1.0e6; rhd_bins_nu_max_ev[k]=100.0e6; /* electron neutrino: ~1-100 MeV */
#endif
#ifdef RT_NEUTRINO_ANTIELECTRON
    k=RT_FREQ_BIN_NU_EBAR; rhd_bins_nu_min_ev[k]=1.0e6; rhd_bins_nu_max_ev[k]=100.0e6; /* electron antineutrino: ~1-100 MeV */
#endif
#ifdef RT_NEUTRINO_HEAVY
    k=RT_FREQ_BIN_NU_X;    rhd_bins_nu_min_ev[k]=1.0e6; rhd_bins_nu_max_ev[k]=100.0e6; /* heavy-flavor neutrino: ~1-100 MeV */
#endif
    for(k=0;k<N_RT_FREQ_BINS;k++) {All.RHD_bins_nu_min_ev[k]=rhd_bins_nu_min_ev[k]; All.RHD_bins_nu_max_ev[k]=rhd_bins_nu_max_ev[k];}
}



#endif // #if defined(RADTRANSFER) || defined(RT_USE_GRAVTREE)






/***********************************************************************************************************
 *
 * ROUTINES WHICH DO NOT NEED TO BE MODIFIED SHOULD GO BELOW THIS BREAK
 *
 *  (these routines may depend on the RT solver or other numerical choices, but below here, place routines 
 *    which shouldn't needed to be hard-coded for different assumptions about the bands of the RT module, etc)
 *
 ***********************************************************************************************************/



/***********************************************************************************************************/
/* rt_absorption_rate: definition now in rt_functions.h */




#ifdef RADTRANSFER

/* rt_diffusion_coefficient is now defined inline in proto.h for GPU compatibility */



/* rt_eddington_update_calculation: definition now in rt_functions.h (RADTRANSFER). */


/* eddington_tensor_dot_vector: now replaced by SymmetricTensor2::matvec() */




/***********************************************************************************************************/
/***********************************************************************************************************/
/*
  routine which does the drift/kick operations on radiation quantities. separated here because we use a non-trivial
    update to deal with potentially stiff absorption terms (could be done more rigorously with something fully implicit in this 
    step, in fact). 
    mode = 0 == kick operation (update the conserved quantities)
    mode = 1 == predict/drift operation (update the predicted quantities)
 */
/***********************************************************************************************************/
void rt_update_driftkick(int i, double dt_entr, int mode, struct particle_data *pp, struct gas_cell_data *cell)
{
#if defined(RT_EVOLVE_ENERGY) || defined(RT_EVOLVE_INTENSITIES)
    int kf, k_tmp; double total_erad_emission_minus_absorption = 0;
#if defined(RT_EVOLVE_INTENSITIES)
    for(kf=0;kf<N_RT_FREQ_BINS;kf++) {cell[i].Rad_E_gamma[kf]=0; for(k_tmp=0;k_tmp<N_RT_INTENSITY_BINS;k_tmp++) {cell[i].Rad_E_gamma[kf]+=RT_INTENSITY_BINS_DOMEGA*cell[i].Rad_Intensity[kf][k_tmp];}}
    double de_emission_minus_absorption_saved[N_RT_FREQ_BINS][N_RT_INTENSITY_BINS]; // save this for use below
#endif
#ifdef RT_INFRARED
    double E_abs_tot_toIR = 0;/* energy absorbed in other bands is transfered to IR, by default: track it here */
    double Rad_E_gamma_tot = 0; // dust temperature defined by total radiation energy density //
    {int j; for(j=0;j<N_RT_FREQ_BINS;j++) {Rad_E_gamma_tot += cell[i].Rad_E_gamma[j];}}
    double a_rad_inverse=C_LIGHT_CGS/(4.*5.67e-5), vol_inv_phys=(cell[i].Density*All.cf_a3inv/cell[i].Mass), u_gamma = Rad_E_gamma_tot * vol_inv_phys * UNIT_PRESSURE_IN_CGS; // photon energy density in CGS //
    double Dust_Temperature_4 = u_gamma * a_rad_inverse; // estimated effective temperature of local rad field in equilibrium with dust emission. note that for our definitions, rad energy density has its 'normal' value independent of RSOL, so Tdust should as well; emission -and- absorption are both lower by a factor of RSOL, but these cancel in the Tdust4 here //
#if !defined(COOLING) // if cooling is active, don't reset this here, because it needs to include the gas coupling term which will be self-consistently calculated there
    cell[i].Dust_Temperature = sqrt(sqrt(Dust_Temperature_4)); // just set this to the local radiation equilibrium temperature
#endif
    double T_min = get_min_allowed_dustIRrad_temperature();
    if(cell[i].Dust_Temperature <= T_min) {cell[i].Dust_Temperature = T_min;} // dust temperature shouldn't be below CMB
    double IRBand_opacity_fraction_from_gas_absorption = 0; // needed below to know what fraction is immediately re-radiated or not
    double T_gas = cell[i].Dust_Temperature;
#ifdef COOLING
    T_gas = cell[i].temperature();
#endif
#endif

    for(k_tmp=0; k_tmp<N_RT_FREQ_BINS; k_tmp++)
    {
#ifdef RT_INFRARED
        // need to do IR last after summing absorption from other bands //
        if(RT_FREQ_BIN_INFRARED < N_RT_FREQ_BINS-1) {if(kf == RT_FREQ_BIN_INFRARED) {kf = N_RT_FREQ_BINS-1;} if(kf == N_RT_FREQ_BINS-1) {kf = RT_FREQ_BIN_INFRARED;}}
#endif
#if defined(RT_EVOLVE_INTENSITIES)
        int k_angle; for(k_angle=0;k_angle<N_RT_INTENSITY_BINS;k_angle++)
#endif
        {
            kf = k_tmp; // normal loop
            double e0, dt_e_gamma_band=0, total_de_dt=0, a0_abs = -rt_absorption_rate(i,kf, pp, cell);
#if defined(RT_EVOLVE_INTENSITIES)
            if(mode==0) {e0 = RT_INTENSITY_BINS_DOMEGA*cell[i].Rad_Intensity[kf][k_angle];} else {e0 = RT_INTENSITY_BINS_DOMEGA*cell[i].Rad_Intensity_Pred[kf][k_angle];}
            dt_e_gamma_band = RT_INTENSITY_BINS_DOMEGA*cell[i].Dt_Rad_Intensity[kf][k_angle];
#else
            if(mode==0) {e0 = cell[i].Rad_E_gamma[kf];} else {e0 = cell[i].Rad_E_gamma_Pred[kf];}
            dt_e_gamma_band = cell[i].Dt_Rad_E_gamma[kf];
#endif
#ifdef RT_COMOVING
            double ET_dotdot_GradVcom = cell[i].ET[kf].double_contract(cell[i].Gradients.Velocity);
            double VolP_dotdot_GradV = e0 * ET_dotdot_GradVcom * All.cf_a2inv; // convert to physical units and multiply by radiation energy density to get into appropriate units
            dt_e_gamma_band += (C_LIGHT_CODE_REDUCED/C_LIGHT_CODE) * (-VolP_dotdot_GradV); // account for RSOL term here as usual
#endif
            total_de_dt = cell[i].Rad_Je[kf] + dt_e_gamma_band;

#ifdef RT_INFRARED
            if(kf == RT_FREQ_BIN_INFRARED)
            {
                if((mode==0) && (dt_e_gamma_band!=0) && (dt_entr>0)) // only update temperatures on kick operations //
                {
                    // advected radiation changes temperature of radiation field, before absorption //
                    double dE_fac = dt_e_gamma_band * dt_entr; // change in energy from advection
                    double dTE_fac = cell[i].Dt_Rad_E_gamma_T_weighted_IR * dt_entr; // T-weighted change from advection
                    double dE_abs = -e0 * (1. - exp(a0_abs*dt_entr)); // change in energy from absorption
                    double T_max = DMAX(cell[i].Radiation_Temperature , dE_fac / dTE_fac); 
                    double rfac=1; if(dE_fac < -0.5*(e0+dE_abs)) {rfac=fabs(0.5*(e0+dE_abs))/fabs(dE_fac);} else {if(dE_fac > 0.5*e0) {rfac=0.5*e0/dE_fac;}}
                    dE_fac*=rfac; dTE_fac*=rfac; // limit temperature change from advection to prevent spurious divergences
                    
                    cell[i].Radiation_Temperature = (e0 + dE_fac) / (MIN_REAL_NUMBER + DMAX(0., e0 / cell[i].Radiation_Temperature + dTE_fac));
                    cell[i].Radiation_Temperature = DMIN(cell[i].Radiation_Temperature, T_max);
                    cell[i].Radiation_Temperature = DMAX(cell[i].Radiation_Temperature, DMAX(T_min, MIN_REAL_NUMBER)); // numerator above can go negative in extreme dynamic-range regimes; floor before use below to avoid log10(negative)=NaN propagating into the opacity table lookup
                    a0_abs = -rt_absorption_rate(i,kf, pp, cell); // update absorption rate using the new radiation temperature //
                }
                double total_absorption_rate = E_abs_tot_toIR + fabs(a0_abs)*e0; // add the summed absorption and equate to dust emission //
#ifdef COOLING
                cell[i].Dust_Temperature = rt_eqm_dust_temp(i, T_gas, total_absorption_rate * vol_inv_phys * C_LIGHT_CODE / C_LIGHT_CODE_REDUCED, pp, cell);
#endif
                if(cell[i].Dust_Temperature < T_min) {cell[i].Dust_Temperature = T_min;}
                double Tdust_eff = cell[i].Dust_Temperature, Trad_eff = cell[i].Radiation_Temperature;
                double kappa_gas = rt_kappa_adaptive_IR_band(i,Tdust_eff,Trad_eff,-1,-1, pp, cell), kappa_total = rt_kappa_adaptive_IR_band(i,Tdust_eff,Trad_eff,0,0, pp, cell);
                IRBand_opacity_fraction_from_gas_absorption = kappa_gas / (kappa_total + MIN_REAL_NUMBER); /* gas absorption opacity only, relative to total opacity (all sources+scattering) */
                double total_emission_rate = total_absorption_rate * (1.-IRBand_opacity_fraction_from_gas_absorption) + cell[i].Rad_Je[kf]; /* we will re-radiate this much because the component due to gas-dust coupling is accounted for in the cooling loop */
                total_de_dt = E_abs_tot_toIR + cell[i].Rad_Je[kf] + dt_e_gamma_band;
                if((mode==0) && (Tdust_eff <= MAX_DUST_TEMP)) // only update temperatures on kick operations and Tdust is meaningful //
                {
                    /* dust absorption and re-emission brings T_rad towards T_dust: */
                    double T_max = DMAX(DMAX(Trad_eff , Tdust_eff), T_gas); /* should not exceed either initial temperature */
                    double frac_unabsorbed = exp(-fabs(a0_abs*dt_entr));
                    double e_absorbed = E_abs_tot_toIR*dt_entr + e0*(1.-frac_unabsorbed);
                    double e_unabsorbed = e0*frac_unabsorbed + DMAX(dt_e_gamma_band*dt_entr, 0); /* include incoming radiation, which has already been used to corect Trad, but outgoing doesn't modify T */
                    double etot_em_gas = e_absorbed * IRBand_opacity_fraction_from_gas_absorption + cell[i].Rad_Je[kf]*dt_entr;
                    double etot_em_dust = e_absorbed * (1.-IRBand_opacity_fraction_from_gas_absorption);
                    double Trad_new = (e_unabsorbed + etot_em_gas + etot_em_dust + MIN_REAL_NUMBER) / (MIN_REAL_NUMBER + e_unabsorbed/Trad_eff + etot_em_gas/DMAX(Trad_eff,T_gas) + etot_em_dust/Tdust_eff);
                    cell[i].Radiation_Temperature = DMIN( Trad_new , T_max );
                }
                if(cell[i].Radiation_Temperature < T_min) {cell[i].Radiation_Temperature = T_min;} // radiation temperature shouldn't be below CMB
            }
#endif
            
            /*---------------------------------------------------------------------------------------------------
             the following block is for absorption and special behavior where
             photons absorbed in one band are re-radiated [or up/down-scattered] into another.
             this must be hard-coded to maintain conservation (as opposed to treated as a source term)
             -----------------------------------------------------------------------------------------------------*/
            if(fabs(a0_abs)*dt_entr > 50.) {a0_abs *= 50./(fabs(a0_abs)*dt_entr);}
            double abs_0 = DMAX(0,fabs(a0_abs)*dt_entr); double slabfac = slab_averaging_function(abs_0); double e_abs_0=exp(-abs_0); if(abs_0>100.) {e_abs_0=0;}
            /* since we're taking exponentials and inverses of some large numbers here, need to be careful not to let floating point errors cause nan's */
            if((dt_entr <= 0.)||(a0_abs >= 0.)||(abs_0 <= 0.)) {abs_0=0.; slabfac=e_abs_0=1.;} else {if(abs_0 < 1.e-5) {slabfac=1.-0.5*abs_0; e_abs_0 = 1.-abs_0;} else {if(abs_0 > 100.) {slabfac = 1./abs_0; e_abs_0 = 0.;}}}
            double e0_postabs = e0*e_abs_0, de_postabs = total_de_dt * dt_entr * slabfac, f_min = 0.01;
            if(e0_postabs+de_postabs < f_min*e0_postabs) {slabfac *= fabs((1.-f_min)*e0_postabs)/(fabs(de_postabs)+MIN_REAL_NUMBER);}
            
            double ef = e0 * e_abs_0 + total_de_dt * dt_entr * slabfac; // gives exact solution for dE/dt = -E*abs + de , the 'reduction factor' appropriately suppresses the source term //
#ifdef RT_INFRARED
            if(isnan(ef)) {PRINT_WARNING("\n ef energy prediction is NaN for cell-ID=%llu, e0=%g e_abs_0=%g abs_0=%g a0_abs=%g total_de_dt=%g dt_entr=%g slabfac=%g Trad=%g Tdust=%g\n", (unsigned long long) (long long)i /* particle index */,e0, e_abs_0,abs_0, a0_abs, total_de_dt,dt_entr,slabfac,cell[i].Radiation_Temperature,cell[i].Dust_Temperature);}
#else
            if(isnan(ef)) {PRINT_WARNING("\n ef energy prediction is NaN for cell-ID=%llu, e0=%g e_abs_0=%g abs_0=%g a0_abs=%g total_de_dt=%g dt_entr=%g slabfac=%g\n", (unsigned long long) (long long)i /* particle index */,e0, e_abs_0,abs_0, a0_abs, total_de_dt,dt_entr,slabfac);}
#endif
            if(ef < 0) {ef=0;}
            double de_abs = e0 + total_de_dt * dt_entr - ef; // energy removed by absorption alone
            double de_emission_minus_absorption = (ef - DMAX(0, (e0 + dt_e_gamma_band * dt_entr * slabfac))); // total change, relative to what we would get with just advection (positive = net energy increase in the gas)
            if((dt_entr <= 0) || (de_abs <= 0)) {de_abs = 0;}
            
#if defined(RT_RAD_PRESSURE_FORCES) && defined(RT_COMPGRAD_EDDINGTON_TENSOR) && !defined(RT_EVOLVE_FLUX) && !defined(RT_RADPRESSURE_IN_HYDRO)
            // for OTVET/FLD methods, need to apply radiation pressure term here so can limit this b/c just based on a gradient which is not flux-limited [as in hydro operators] //
            {
                double radacc[3]={0}, rmag=0, vel_i[3], L_particle = pp[i].Get_Particle_Size()*All.cf_atime; // particle effective size/slab thickness
                double Sigma_particle = cell[i].Mass / (M_PI*L_particle*L_particle), abs_per_kappa_dt = C_LIGHT_CODE_REDUCED * (cell[i].Density*All.cf_a3inv) * dt_entr; // effective surface density through particle & fractional absorption over timestep
                double f_kappa_abs = rt_absorb_frac_albedo(i,kf, pp, cell); // get albedo, we'll need this below
                double slabfac_rp=1; if(check_if_absorbed_photons_can_be_reemitted_into_same_band(kf)==0) {slabfac_rp=slab_averaging_function(f_kappa_abs*cell[i].Rad_Kappa[kf]*Sigma_particle) * slab_averaging_function(f_kappa_abs*cell[i].Rad_Kappa[kf]*abs_per_kappa_dt);} // reduction factor for absorption over dt
                int kx; for(kx=0;kx<3;kx++)
                {
                    radacc[kx] = -dt_entr * slabfac_rp * cell[i].flux_limiter(kf) * (cell[i].Gradients.Rad_E_gamma_ET[kf][kx] / cell[i].Density) / All.cf_atime; // naive radiation-pressure calc for FLD methods [physical units]
                    rmag += radacc[kx]*radacc[kx]; // compute magnitude
                    if(mode==0) {vel_i[kx]=(C_LIGHT_CODE_REDUCED/C_LIGHT_CODE)*pp[i].Vel[kx]/All.cf_atime;} else {vel_i[kx]=(C_LIGHT_CODE_REDUCED/C_LIGHT_CODE)*cell[i].VelPred[kx]/All.cf_atime;} // [for comoving] note this is the 'effective' u appearing in the RHD equations for an RSOL, care needed with these factors!
                }
                if(rmag > 0)
                {
                    rmag = sqrt(rmag); for(kx=0;kx<3;kx++) {radacc[kx] /= rmag;} // normalize
                    double rmag_max = de_abs / (cell[i].Mass * C_LIGHT_CODE_REDUCED * (MIN_REAL_NUMBER + f_kappa_abs)); // limit magnitude to absorbed photon momentum
                    if(check_if_absorbed_photons_can_be_reemitted_into_same_band(kf)==0) {if(rmag > rmag_max) {rmag=rmag_max;}}
#if defined(RT_ENABLE_R15_GRADIENTFIX)
                    rmag = rmag_max; // set to maximum (optically thin limit)
#endif
                    double work_band = 0;
                    for(kx=0;kx<3;kx++)
                    {
                        double radacc_eff = radacc[kx] * rmag; // re-normalize according to the criterion above
                        work_band += vel_i[kx] * radacc_eff * cell[i].Mass; // PdV work done by photons [absorbed ones are fully-destroyed, so their loss of energy and momentum is already accounted for by their deletion in this limit //
                        if(mode==0) {double dv_rt=radacc_eff*All.cf_atime; pp[i].Vel[kx]+=dv_rt; pp[i].dp[kx]+=dv_rt*cell[i].Mass;} else {cell[i].VelPred[kx] += radacc_eff * All.cf_atime;}
                    }
                    double d_egy_rad = (2.*f_kappa_abs-1.)*work_band , d_egy_int = -2.*f_kappa_abs*work_band * (C_LIGHT_CODE/C_LIGHT_CODE_REDUCED); // correct for rsol factor above which reduced vel_i by rsol; -only- add back this term for gas
                    if(mode==0) {cell[i].InternalEnergy += d_egy_int;} else {cell[i].InternalEnergyPred += d_egy_int;}
#if defined(RT_EVOLVE_INTENSITIES)
                    {int k_q; for(k_q=0;k_q<N_RT_INTENSITY_BINS;k_q++) {if(mode==0) {cell[i].Rad_Intensity[kf][k_q]+=d_egy_rad/RT_INTENSITY_BINS_DOMEGA;} else {cell[i].Rad_Intensity_Pred[kf][k_q]+=d_egy_rad/RT_INTENSITY_BINS_DOMEGA;}}}
#else
                    if(mode==0) {cell[i].Rad_E_gamma[kf]+=d_egy_rad;} else {cell[i].Rad_E_gamma_Pred[kf]+=d_egy_rad;}
#endif
                }
            }
#endif
            
            int donation_target_bin = rt_get_donation_target_bin(kf); // frequency into which the photons will be deposited, if any //
#ifdef RT_INFRARED
            if((donation_target_bin == RT_FREQ_BIN_INFRARED) && (kf != RT_FREQ_BIN_INFRARED)) {E_abs_tot_toIR += de_abs/(MIN_REAL_NUMBER + dt_entr);} /* donor bin is yourself in the IR - some self-absorption is re-emitted, but this is handled explicitly below, so don't need to include it in sum here */
            if(kf==RT_FREQ_BIN_INFRARED) {
#ifdef COOLING
                ef += de_abs*(1.-IRBand_opacity_fraction_from_gas_absorption); /* update: assume a fraction de_abs * IRBand_opacity_fraction_from_gas_absorption is absorbed by the gas, which will not be instantly re-emitted here, but later in the cooling subroutines */
                if(mode==0) {cell[i].DtInternalEnergy += (de_abs * IRBand_opacity_fraction_from_gas_absorption) / ((MIN_REAL_NUMBER + dt_entr) * cell[i].Mass);} /* this fraction absorbed by gas goes into a heating rate which can be balanced implicitly in the cooling function later */
#else
                ef = e0 + total_de_dt * dt_entr; // previous version: assumes all self-absorption is re-emitted
#endif
            } /* donor bin is yourself in the IR - just need to decide what to do with the photons */
#endif
            // isotropically re-emit the donated radiation into the target bin[s] //
#if defined(RT_EVOLVE_INTENSITIES)
            // this is the leading-order (isotropic) emission-absorption step, i.e. the psi_a * (j_e - I) term in the intensity equation. solved by the methods above to deal generically with stiff emission-absorption problems, re-used below if needed //
            if(donation_target_bin >= 0) {int k_q; for(k_q=0;k_q<N_RT_INTENSITY_BINS;k_q++) {if(mode==0) {cell[i].Rad_Intensity[donation_target_bin][k_q] += de_abs/RT_INTENSITY_BINS_DOMEGA;} else {cell[i].Rad_Intensity_Pred[donation_target_bin][k_q] += de_abs/RT_INTENSITY_BINS_DOMEGA;}}}
            if(ef < 0) {ef=0;}
            if(mode==0) {cell[i].Rad_Intensity[kf][k_angle] = ef/RT_INTENSITY_BINS_DOMEGA;} else {cell[i].Rad_Intensity_Pred[kf][k_angle] = ef/RT_INTENSITY_BINS_DOMEGA;}
#else
            if(donation_target_bin >= 0) {if(mode==0) {cell[i].Rad_E_gamma[donation_target_bin] += de_abs;} else {cell[i].Rad_E_gamma_Pred[donation_target_bin] += de_abs;}}
            if(ef < 0) {ef=0;}

            if(mode==0) {cell[i].Rad_E_gamma[kf] = ef;} else {cell[i].Rad_E_gamma_Pred[kf] = ef;}
#endif

#if defined(RT_EVOLVE_FLUX)
            int k_dir; double f_mag=0, E_rad_forflux=0, vdot_h[3]={0}, vel_i[3]={0}, DeltaFluxEff[3]={0}, rho=cell[i].Density*All.cf_a3inv; E_rad_forflux=0.5*(e0+ef); // use energy density averaged over this update for the operation below
            for(k_dir=0;k_dir<3;k_dir++) {if(mode==0) {vel_i[k_dir]=RSOL_CORRECTION_FACTOR_FOR_VELOCITY_TERMS*pp[i].Vel[k_dir]/All.cf_atime;} else {vel_i[k_dir]=RSOL_CORRECTION_FACTOR_FOR_VELOCITY_TERMS*cell[i].VelPred[k_dir]/All.cf_atime;}} // need gas velocity at this time [effective v - note RSOL terms]
            double teqm_inv = cell[i].Rad_Kappa[kf] * rho * C_LIGHT_CODE_REDUCED + MIN_REAL_NUMBER; // physical code units of 1/time, defines characteristic timescale for coming to equilibrium flux. see notes for CR second-order module for details. //
            {Vec3<double> v_i{vel_i[0],vel_i[1],vel_i[2]}; Vec3<double> vdh = E_rad_forflux * (v_i + cell[i].ET[kf].matvec(v_i)); vdot_h[0]=vdh[0]; vdot_h[1]=vdh[1]; vdot_h[2]=vdh[2];} // calculate P_rad term and eI term, multiply by radiation energy //
#ifdef RT_COMPGRAD_EDDINGTON_TENSOR // definitely favor this for greater accuracy and reduced noise //
            for(k_dir=0;k_dir<3;k_dir++) {DeltaFluxEff[k_dir] -= (cell[i].Mass/rho) * (C_LIGHT_CODE_REDUCED*C_LIGHT_CODE_REDUCED/teqm_inv) * cell[i].Gradients.Rad_E_gamma_ET[kf][k_dir]*All.cf_a3inv/All.cf_atime;} // here we compute the nabla.pressure_gradient_tensor term from gradients directly, and use this in the next step after multiplying the flux equation by (tilde[c]^2/dt_eqm_inv) and working in dimensionless time units
#else
            for(k_dir=0;k_dir<3;k_dir++) {DeltaFluxEff[k_dir] += (cell[i].Dt_Rad_Flux[kf][k_dir]/teqm_inv);} // the nabla.pressure_gradient_tensor is computed in the finite-volume solver, here
#endif
            for(k_dir=0;k_dir<3;k_dir++) {DeltaFluxEff[k_dir] += vdot_h[k_dir];} // add the 'enthalpy advection' term here, vdot_h = Erad v.(e*I + P_rad)

            double tau=dt_entr*teqm_inv, f00=exp(-tau), f11=1.-f00;
            if(tau > 0 && isfinite(tau))
            {
                if(tau <= 0.04) {f11=tau-0.5*tau*tau+tau*tau*tau/6.; f00=1.-f11;} // some limits to prevent small/large number problems here
                if(!isfinite(f00) || !isfinite(f11)) {f00=1.; f11=0.;} // some limits to prevent small/large number problems here
                if(tau >= 20.) {f00=DMAX(0.,DMIN(1.e-11,f00)); f11=1.-f00;} // some limits to prevent small/large number problems here
                for(k_dir=0;k_dir<3;k_dir++)
                {
                    double flux_0; if(mode==0) {flux_0 = cell[i].Rad_Flux[kf][k_dir];} else {flux_0 = cell[i].Rad_Flux_Pred[kf][k_dir];}
                    flux_0 += vel_i[k_dir] * de_emission_minus_absorption; // add Lorentz term from net energy injected by absorption and re-emission (effectively, we operator-split this term and solve it -BEFORE- going to the next step). note our definitions of vel_i above ensure this is zero when the rhd is comoving-frame
                    double flux_f = flux_0 * f00 + DeltaFluxEff[k_dir] * f11; // exact solution for dE/dt = -E*abs + de , the 'reduction factor' appropriately suppresses the source term //
                    if(mode==0) {cell[i].Rad_Flux[kf][k_dir] = flux_f;} else {cell[i].Rad_Flux_Pred[kf][k_dir] = flux_f;}
                    f_mag += flux_f*flux_f; // magnitude of flux vector
                }
                if(f_mag > 0) // limit the flux according the physical (optically thin) maximum //
                {
                    f_mag=sqrt(f_mag); double fmag_max = C_LIGHT_CODE_REDUCED * ef; // maximum flux should be optically-thin limit: e_gamma*c: here allow some tolerance for numerical leapfrogging in timestepping. should be the RSOL here, although in principle equations can allow exceeding this if we have reached equilibrium, it really violates the M1 closure assumptions. see discussion in Skinner+Ostriker 2013 or Levermore et al. 1984
                    if(f_mag > fmag_max) {for(k_dir=0;k_dir<3;k_dir++) {if(mode==0) {cell[i].Rad_Flux[kf][k_dir] *= fmag_max/f_mag;} else {cell[i].Rad_Flux_Pred[kf][k_dir] *= fmag_max/f_mag;}}}
#if defined(GRAIN_RDI_TESTPROBLEM_LIVE_RADIATION_INJECTION)
                    if(pp[i].Pos[2]<=0.1) {if(mode==0) {cell[i].Rad_Flux[kf][0]=cell[i].Rad_Flux[kf][1]=0; cell[i].Rad_Flux[kf][2]=fmag_max;} else {cell[i].Rad_Flux_Pred[kf][0]=cell[i].Rad_Flux_Pred[kf][1]=0; cell[i].Rad_Flux_Pred[kf][2]=fmag_max;}}
#endif
                }
            }
#endif
            total_erad_emission_minus_absorption += de_emission_minus_absorption; // add to cumulative sum for back-reaction to gas
#if defined(RT_EVOLVE_INTENSITIES)
            de_emission_minus_absorption_saved[k_tmp][k_angle] = de_emission_minus_absorption; // save this for use below
#endif
        } // clause for radiation angle [needed for evolving intensities]	
    } // loop over frequencies
    
#if defined(RT_EVOLVE_INTENSITIES)
    if(dt_entr > 0) { // none of this is worth doing if we don't have a finite timestep here
    for(kf=0;kf<N_RT_FREQ_BINS;kf++)
    {
        int k,k_om; double rho=cell[i].Density*All.cf_a3inv, ceff=C_LIGHT_CODE_REDUCED, ctrue=C_LIGHT_CODE, teq_inv=cell[i].Rad_Kappa[kf]*rho*ceff, f_a=rt_absorb_frac_albedo(i,kf, pp, cell), f_s=1.-f_a, b_dot_n[N_RT_INTENSITY_BINS]={0}, beta_2=0.; Vec3<double> beta;
        int n_iter = 1 + (int)(DMIN(DMAX(4. , dt_entr/teq_inv), 1000.)); // number of iterations to subcycle everything below //
        double dt=dt_entr/n_iter, tau=dt*teq_inv, i0[N_RT_INTENSITY_BINS]={0}, invfourpi=1./(4.*M_PI), J, b_dot_H, b2_dot_K; int i_iter;
        for(i_iter=0; i_iter<n_iter; i_iter++)
        {
            double egy_0=0, egy_f=0; Vec3<double> flux_0={}, flux_f={}; // compute total change over sub-cycle, to update gas properties
            // load all the gas and intensity properties we need [all can change on the subcycle so some re-computing here]
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {if(mode==0) {i0[k_om] = RT_INTENSITY_BINS_DOMEGA*cell[i].Rad_Intensity[kf][k_om];} else {i0[k_om] = RT_INTENSITY_BINS_DOMEGA*cell[i].Rad_Intensity_Pred[kf][k_om];}}
            if(mode==0) {beta=pp[i].Vel/(All.cf_atime*ctrue);} else {beta=cell[i].VelPred/(All.cf_atime*ctrue);} // need gas velocity at this time; with equations written this way, the 'beta' term is the -true- beta, so we have to use the true SOL
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {b_dot_n[k_om]=0; for(k=0;k<3;k++) {b_dot_n[k_om]+=All.Rad_Intensity_Direction[k_om][k]*beta[k];}}
            beta_2 = beta.norm_sq();
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {egy_0+=i0[k_om]; for(k=0;k<3;k++) {flux_0[k]+=All.Rad_Intensity_Direction[k_om][k]*i0[k_om];}}
            J=0,b_dot_H=0,b2_dot_K=0; for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {J+=i0[k_om]*invfourpi; b_dot_H+=b_dot_n[k_om]*i0[k_om]*invfourpi; b2_dot_K+=b_dot_n[k_om]*b_dot_n[k_om]*i0[k_om]*invfourpi;}

            // isotropic terms that change total energy in bin (part of the 'work term' for the photon momentum): this includes the beta.beta*(J+K) and beta.H terms
            double work = (1. * (f_s-f_a)*(beta_2*J + b2_dot_K) - 2.*f_s*b_dot_H) * tau; // will be shared isotropically.
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {if((work>0) || (i0[k_om]<=0)) {i0[k_om]+=work;} else {i0[k_om]/=(1-work/i0[k_om]);}} // gaurantees linearized sum is still correct, and symmetric with positive changes, but can't get negative energies. shared isotropically.

            // isotropic scattering term [scattering * (J - I) term in the intensity equation] [recall, our general update for the 'energy term' for absorption and emission above already took care of the psi_a*(j_em - I) term in the intensity equation
            J=0; for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {J+=i0[k_om]*invfourpi;} // prepare to calculate isotropic scattering term
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {i0[k_om] = J + (i0[k_om]-J)*exp(-f_s*tau);} // isotropic scattering conserving total energy over step
            
            // flux 'boost' and 'beaming' terms (go as n.beta). note we replace je -> je-I + I, and use the fact that we have solved already for the time-integral of (psi_a*(je-I)*dt) = de_emission_minus_absorption_saved, which can be re-used here, in average form <psi_a*(je-I)> = dE/dt --> just make sure the units are correct! because we're working in dimensionless units below, we should divide by tau, to be working in the same delta-units here: these are the 3 n.beta * [ psi_a*(j_em-J_nu)*(creduced/c)^2 + (psi_a+psi_s)*J_nu) ] in the intensity equation
            double fboost[N_RT_INTENSITY_BINS], fboost_avg=0, fboost_avg_p=0, fboost_avg_m=0; // calculate flux 'boost' terms
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {fboost[k_om] = 3.*b_dot_n[k_om] * ((de_emission_minus_absorption_saved[kf][k_om]/tau) + ((f_a+f_s)*J)); fboost_avg += fboost[k_om]/N_RT_INTENSITY_BINS;} // pre-calculate to get mean value, will divide out
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {work=(fboost[k_om]-fboost_avg)*tau; if((work>0) || (i0[k_om]<=0)) {fboost[k_om]=work; fboost_avg_p+=fboost[k_om];} else {fboost[k_om]=work/(1.-work/i0[k_om]); fboost_avg_m+=fboost[k_om];}} // zero total energy change at linear order ensured by subtracting out sum here; non-linearization ensures i0 cannot be negative, but does allow second-order dt work term to appear, that's ok for now
            if(fboost_avg_p>0 && fboost_avg_m<0) {double fc=-fboost_avg_m/fboost_avg_p; fboost_avg_m=(1.+fc)/(1.+fc*fc); fboost_avg_p=fc*fboost_avg_m;} else {fboost_avg_m=fboost_avg_p=0;} // // these re-weight to gaurantee the non-linear sum is identically zero while preserving positive-definite behavior
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {if(fboost[k_om]>0) {i0[k_om]+=fboost_avg_p*fboost[k_om];} else {i0[k_om]+=fboost_avg_m*fboost[k_om];}} // alright done!
            
            // flux work term, allowed to both do work and be asymmetric so just need to ensure it retains positive-definite intensities: n.beta * (psi_a+psi_s) * I term in intensity equation
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {work=b_dot_n[k_om]*(f_a+f_s)*i0[k_om] * tau; if((work>0) || (i0[k_om]<=0)) {i0[k_om]+=work;} else {i0[k_om]/=(1-work/i0[k_om]);}}

            // ok -now- calculate the net change in momentum and energy, for updating the gas quantities
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {egy_f+=i0[k_om]; for(k=0;k<3;k++) {flux_f[k]+=All.Rad_Intensity_Direction[k_om][k]*i0[k_om];}}
            Vec3<double> dv_gas = -(flux_f-flux_0)/(ceff*cell[i].Mass); double ke_gas_0=0, ke_gas_f=0, v0g=0, u0=0;
            {Vec3<double> v0_gas = ctrue*beta; ke_gas_0 = v0_gas.norm_sq(); ke_gas_f = (v0_gas+dv_gas).norm_sq();} // note everything is volume-integrated, accounted for above, and we defined flux for convience without the c, so just one power of c here.
            double d_ke_gas = 0.5*(ke_gas_f - ke_gas_0)*cell[i].Mass, de_gas=-(ctrue/ceff)*(egy_f-egy_0), de_gas_internal=(de_gas-d_ke_gas)/cell[i].Mass; // note ctrue/ceff factor here, accounting for rsol difference in gas heating/cooling rates vs RHD
            if(mode==0) {u0=cell[i].InternalEnergy;} else {u0=cell[i].InternalEnergyPred;} // for updating gas internal energy (work terms, after subtracting kinetic energy changes)
            if(de_gas_internal<=-0.9*u0) {de_gas_internal = DMIN(de_gas_internal/(1.-de_gas_internal/u0), -0.9*u0);} // just a catch to avoid negative energies (will break energy conservation if you are slamming into it, however!
            
            // assign everything back to the appropriate variables after update
            if(mode==0) {auto dv_kick=dv_gas*All.cf_atime; pp[i].Vel+=dv_kick; pp[i].dp+=dv_kick*cell[i].Mass;} else {cell[i].VelPred += dv_gas*All.cf_atime;} // update gas velocities (radiation pressure forces here)
            if(mode==0) {cell[i].InternalEnergy += de_gas_internal;} else {cell[i].InternalEnergyPred += de_gas_internal;} // update gas internal energy (work terms, after subtracting kinetic energy changes)
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {if(mode==0) {cell[i].Rad_Intensity[kf][k_om] = i0[k_om]/RT_INTENSITY_BINS_DOMEGA;} else {cell[i].Rad_Intensity_Pred[kf][k_om] = i0[k_om]/RT_INTENSITY_BINS_DOMEGA;}} // update intensities (all of the above)
            cell[i].Rad_E_gamma[kf]=egy_f; // set this every time this subroutine is called, so it is accessible everywhere else //
        } // loop over iterations
    } // loop over frequencies
    } // finite timestep requirement
#else
    double mom_fac = 1. - RSOL_CORRECTION_FACTOR_FOR_VELOCITY_TERMS * total_erad_emission_minus_absorption / (cell[i].Mass * C_LIGHT_CODE_REDUCED*C_LIGHT_CODE_REDUCED); // back-reaction on gas from emission, which is isotropic in the fluid frame but anisotropic in the lab frame. this effect is only important in actually semi-relativistic problems so we use "real" C here, not a RSOL, and match the corresponding term above in the radiation flux equation (if that is evolved explicitly). careful checking-through gives the single termm here, not both
    if(fabs(mom_fac - 1) > 0.1) {printf("WARNING: Large radiation backreaction for cell %d (mom_fac=%g), check the RT solver stability if this is not a relativistic problem.\n",i,mom_fac);}
    {int k_dir; for(k_dir=0;k_dir<3;k_dir++) {if(mode==0) {pp[i].dp[k_dir]+=pp[i].Vel[k_dir]*(mom_fac-1.)*cell[i].Mass; pp[i].Vel[k_dir]*=mom_fac;} else {cell[i].VelPred[k_dir] *= mom_fac;}}}
#endif

    if(mode > 0) {rt_eddington_update_calculation(i, pp, cell);} /* update the eddington tensor (if we calculate it) as well */

#ifdef RT_ISRF_BACKGROUND
    if(mode==0) {rt_apply_boundary_conditions(i, pp, cell);} /* if we have any special boundary conditions (e.g. fixed ISRF at box edge) apply this here */
#endif
#ifdef NUCLEAR_NETWORK_NEUTRINOS
    if(mode==0) { /* after RT kick, update Ye from neutrino absorption */
        nuclear_neutrino_ye_feedback(i, dt_entr, pp, cell);
    }
#endif
#endif
}



#endif

/* rt_apply_boundary_conditions, get_background_isrf_urad, background_isrf_cmb_Teff:
 * definitions now in rt_functions.h (RT_ISRF_BACKGROUND && RADTRANSFER). */



#ifdef RADTRANSFER
/***********************************************************************************************************/
/* this function initializes some of the variables we need */
/***********************************************************************************************************/
void rt_set_simple_inits(int RestartFlag)
{
    if(RestartFlag==1) return;
    int flag_to_reset_values_on_startup = 0;
    if(RestartFlag==0) {flag_to_reset_values_on_startup = 1;}
#if defined(SINGLE_STAR_AND_SSP_HYBRID_MODEL) && defined(SINGLE_STAR_RESTART_FROM_FIRESIM)
    if(RestartFlag==2) {flag_to_reset_values_on_startup = 1;}
#endif
    
    int i; for(i = 0; i < NumPart; i++)
    {
        if(P[i].Type == 0)
        {
            int k;
#ifdef RT_INFRARED
            if(flag_to_reset_values_on_startup) {CellP[i].Radiation_Temperature = CellP[i].Dust_Temperature = DMIN(All.InitRadiationTemp,100.);} //get_min_allowed_dustIRrad_temperature(); // in K, floor = CMB temperature or 10K
#ifdef RT_ISRF_BACKGROUND
            if(flag_to_reset_values_on_startup) {CellP[i].Radiation_Temperature = background_isrf_cmb_Teff();} //CellP[i].Dust_Temperature;
#endif
            CellP[i].Dt_Rad_E_gamma_T_weighted_IR = 0;
#endif
#ifdef RT_CHEM_PHOTOION
            if(flag_to_reset_values_on_startup)
            {
                CellP[i].HII = MIN_REAL_NUMBER;
                CellP[i].HI = 1.0 - CellP[i].HII;
                CellP[i].Ne = CellP[i].HII;
#ifdef RT_CHEM_PHOTOION_HE
                double fac = (1-HYDROGEN_MASSFRAC) / (4.0 * HYDROGEN_MASSFRAC);
                CellP[i].HeIII = MIN_REAL_NUMBER * fac;
                CellP[i].HeII = MIN_REAL_NUMBER * fac;
                CellP[i].HeI = (1.0 - CellP[i].HeII - CellP[i].HeIII) * fac;
                CellP[i].Ne += CellP[i].HeII + 2.0 * CellP[i].HeIII;
#endif
            } else {
#ifdef RT_CHEM_PHOTOION_HE
                CellP[i].HeIII = DMIN(1.0, DMAX(MIN_REAL_NUMBER, 1.0 - CellP[i].HeII - CellP[i].HeI * (4.0 * HYDROGEN_MASSFRAC)/(1-HYDROGEN_MASSFRAC))); // not read in since for this subroutine follows from others
#endif
            }
#endif
#ifdef RT_ISRF_BACKGROUND
            double urad[N_RT_FREQ_BINS];
            get_background_isrf_urad(i, urad);
#endif
            for(k = 0; k < N_RT_FREQ_BINS; k++)
            {
                if(flag_to_reset_values_on_startup) {CellP[i].Rad_E_gamma[k] = MIN_REAL_NUMBER;}
                int flag_to_reset_values_on_startup_et = flag_to_reset_values_on_startup;
#if !defined(OUTPUT_EDDINGTON_TENSOR)
                flag_to_reset_values_on_startup_et = 1;
#endif
                if(flag_to_reset_values_on_startup_et) {CellP[i].ET[k].set_isotropic(1./3.);}
                CellP[i].Rad_Je[k] = 0;
#ifdef RT_FLUXLIMITER
                CellP[i].Rad_Flux_Limiter[k] = 1;
#endif

#ifdef RT_INFRARED
                if(flag_to_reset_values_on_startup && k==RT_FREQ_BIN_INFRARED) { // only initialize the IR energy if starting a new run, otherwise use what's in the snapshot
                    CellP[i].Rad_E_gamma[RT_FREQ_BIN_INFRARED] = (4.*5.67e-5 / C_LIGHT_CGS) * pow(DMIN(All.InitRadiationTemp,100.),4.) / UNIT_PRESSURE_IN_CGS * P[i].Mass / (CellP[i].Density*All.cf_a3inv);
                }
#endif
#ifdef RT_ISRF_BACKGROUND
                if(flag_to_reset_values_on_startup) {CellP[i].Rad_E_gamma[k] = urad[k] * P[i].Mass / (CellP[i].Density*All.cf_a3inv);}
#endif
#ifdef RT_EVOLVE_ENERGY
                CellP[i].Rad_E_gamma_Pred[k] = CellP[i].Rad_E_gamma[k]; 
                CellP[i].Dt_Rad_E_gamma[k] = 0;
#endif
#ifdef RT_EVOLVE_FLUX
                int k_dir, flag_to_reset_values_on_startup_flux = flag_to_reset_values_on_startup;
#if !defined(OUTPUT_RT_RAD_FLUX)
                flag_to_reset_values_on_startup_flux = 1;
#endif
                if(flag_to_reset_values_on_startup_flux) {
                    for(k_dir=0;k_dir<3;k_dir++) {CellP[i].Rad_Flux_Pred[k][k_dir] = 0;}
                } else {
                    CellP[i].Rad_Flux_Pred[k] *= P[i].Mass/(CellP[i].Density*All.cf_a3inv); // need to correct the units here before using
                }
                CellP[i].Rad_Flux[k] = CellP[i].Rad_Flux_Pred[k];
                CellP[i].Dt_Rad_Flux[k] = {};
#endif
#ifdef RT_EVOLVE_INTENSITIES
                int k_dir; for(k_dir=0;k_dir<N_RT_INTENSITY_BINS;k_dir++) {CellP[i].Rad_Intensity_Pred[k][k_dir] = CellP[i].Rad_Intensity[k][k_dir] = MIN_REAL_NUMBER; CellP[i].Dt_Rad_Intensity[k][k_dir] = 0;}
#endif
                
#ifdef GRAIN_RDI_TESTPROBLEM_LIVE_RADIATION_INJECTION
                double q_a = (0.75*All.Grain_Q_at_MaxGrainSize) / ((All.Grain_Internal_Density/UNIT_DENSITY_IN_CGS)*(All.Grain_Size_Max/UNIT_LENGTH_IN_CGS)), e0 = All.Vertical_Grain_Accel / q_a, kappa_0 = All.Grain_Absorbed_Fraction_vs_Total_Extinction * q_a * All.Dust_to_Gas_Mass_Ratio, cell_vol = (P[i].Mass/CellP[i].Density);
                double rho_base_setup = 1., H_scale_setup = 1.*boxSize_X; // define in code units the -assumed- initial scaling of the base gas density and vertical scale-length (PROBLEM SPECIFIC HERE!)
#ifdef GRAIN_RDI_TESTPROBLEM_ACCEL_DEPENDS_ON_SIZE
                kappa_0 *= sqrt(All.Grain_Size_Max / All.Grain_Size_Min); // opacity must be corrected for dependence of Q on grainsize or lack thereof
#endif
                double E_cell_thin = cell_vol * e0 * exp(-kappa_0*rho_base_setup*H_scale_setup*(1.-exp(-P[i].Pos[2]/H_scale_setup))), E_cell=E_cell_thin; // attenuate according to equilibrium expectation, if we're using single-scattering radiation pressure [otherwise comment this line out] //
                double tau_tot = q_a * All.Dust_to_Gas_Mass_Ratio * rho_base_setup*H_scale_setup; if(tau_tot>1) {E_cell = cell_vol * (3.*All.Vertical_Grain_Accel*All.Dust_to_Gas_Mass_Ratio*rho_base_setup*H_scale_setup) * (exp(-P[i].Pos[2]/H_scale_setup) + 1./tau_tot);} // attenuate according to approximate optically-thick expression with free-streaming from the 'photosphere' when optically thin
                CellP[i].Rad_E_gamma_Pred[k] = CellP[i].Rad_E_gamma[k] = E_cell;
#if defined(RT_EVOLVE_FLUX)
                CellP[i].Rad_Flux_Pred[k][2]=CellP[i].Rad_Flux[k][2] = E_cell_thin*C_LIGHT_CODE_REDUCED;
                CellP[i].Rad_Flux[k][0]=CellP[i].Rad_Flux[k][1]=CellP[i].Rad_Flux_Pred[k][0]=CellP[i].Rad_Flux_Pred[k][1]=0;
#endif
#endif
                CellP[i].Rad_Kappa[k] = rt_kappa(i,k, P, CellP); // let everything else get reset first before calling this //
            }
        }
    }
}
#endif



#if defined(RT_EVOLVE_INTENSITIES)
/***********************************************************************************************************/
/* routine to initialize the distribution of ray angles along which the intensities are explicitly evolved.
    this follows Bruhls et al. 1999 [Appendix B]. the rays uniformly sample the unit sphere, are distributed
    isotropically, and satisfy the numerical quadtratures through second order
    J=SUM[dOmega*I]/4PI=I, F=SUM[dOmega*I*Omega_hat]=0, K=SUM[dOmega*I*Omega_hat.x.Omega_hat]/4PI=(1/3)*Identity*J
    to machine accuracy for an isotropic (I=constant along all discrete directions) radiation field.
    The user specifies the number of independent polar angles Np=RT_LOCALRAYGRID (any integer). The total number of rays
    per octant of the unit sphere is then Np*(Np+1)/2, and the total number of rays
    altogether is 4*Np*(Np+1). The exact numerical quadrature assumption is used throughout.
 */
/***********************************************************************************************************/
void rt_init_intensity_directions(void)
{
    int n_polar = RT_LOCALRAYGRID;
    if(n_polar < 1) {printf("Number of rays is invalid (<1). Terminating.\n"); fflush(stdout); endrun(5346343); return;} // soft-stop + return: skip the invalid mu[n_polar] VLA (n_polar<1 is stack-UB) below

    double mu[n_polar]; int i,j,k,l,n=0,n_oct=n_polar*(n_polar+1)/2;
    double Rad_Intensity_Direction_tmp[n_oct][3];
    for(j=0;j<n_polar;j++) {mu[j] = sqrt( (j + 1./6.) / (n_polar - 1./2.) );}
    
    for(i=0;i<n_polar;i++)
    {
        for(j=0;j<n_polar-i;j++)
        {
            k=n_polar-1-i-j;
            Rad_Intensity_Direction_tmp[n][0]=mu[i]; Rad_Intensity_Direction_tmp[n][1]=mu[j]; Rad_Intensity_Direction_tmp[n][2]=mu[k];
            n++;
        }
    }
    n=0;
    for(i=0;i<2;i++)
    {
        double sign_x = 1 - 2*i;
        for(j=0;j<2;j++)
        {
            double sign_y = 1 - 2*j;
            for(k=0;k<2;k++)
            {
                double sign_z = 1 - 2*k;
                for(l=0;l<n_oct;l++)
                {
                    All.Rad_Intensity_Direction[n][0] = Rad_Intensity_Direction_tmp[l][0] * sign_x;
                    All.Rad_Intensity_Direction[n][1] = Rad_Intensity_Direction_tmp[l][1] * sign_y;
                    All.Rad_Intensity_Direction[n][2] = Rad_Intensity_Direction_tmp[l][2] * sign_z;
                    n++;
                }
            }
        }
    }
}
#endif


/***********************************************************************************************************/
/* optional routine to distribute cooling or other sources from gas: currently empty */
/***********************************************************************************************************/
/* rt_get_lum_gas: body now in rt_functions.h (single source for host + device). */




/***********************************************************************************************************/
/* rt_get_donation_target_bin: definition now in rt_functions.h */




/* slab_averaging_function: definition now in rt_functions.h */

/* check_if_absorbed_photons_can_be_reemitted_into_same_band: definition now in rt_functions.h */



/*****************************************************************************
Routines specifically for handling thermal and radiative coupling between gas,
dust, and the IR radiation field component (RT_INFRARED)
*****************************************************************************/

#ifdef RT_INFRARED

/* get_min_allowed_dustIRrad_temperature, dust_dE_cooling, rt_ir_lambdadust:
   bodies live in radiation/rt_functions.h (single source for host + device). */

#endif /* RT_INFRARED */

/* dust_dEdt: definition now in rt_functions.h (single source of truth) */

/* rt_eqm_dust_temp: definition now in rt_functions.h (single source of truth, inside #ifdef COOLING) */

/* blackbody_lum_frac: definition now in rt_functions.h (single source of truth) */

/* rt_irband_egydensity_in_band: definition now in rt_functions.h (single source of truth) */

/***********************************************************************************************************/
/* returns the fraction of a star's SED (approximated as a blackbody) in a given photon energy band - accurate to <1% over all wavelengths
   i - index of star particle
   E_lower - lower end of the energy band in eV
   E_upper - upper end of the energy band in eV
*/
/***********************************************************************************************************/
/* stellar_lum_in_band: body now in rt_functions.h (single source for host + device). */




/* chimes_G0_luminosity, chimes_ion_luminosity, rt_get_source_luminosity_chimes:
   bodies now in rt_functions.h (single source for host + device). */



/*--------------------------------------------------------------------
  calculate the IR dust opacity [in physical code units = Length^2/Mass].
   NOTE: The flag do_emission_absorption_scattering_opacity toggles special behaviour.
   -1: returns the absorption opacity only, using the radiation temperature
    0: returns the scattering+absorption opacity using the radiation temperature (usually want this)
    1: returns the *emission* opacity, assuming the dust+gas radiates as a blackbody (depends only on T_dust)
   likewise, dust_or_gas_opacity_only_flag toggles different behaviors:
    0: total IR-band opacity,
    1: opacity -only- from dust,
   -1: opacity -only- from non-dust 
--------------------------------------------------------------------*/
/* rt_kappa_adaptive_IR_band: definition now in rt_functions.h (single source of truth) */
