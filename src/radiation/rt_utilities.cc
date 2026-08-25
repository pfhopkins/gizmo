#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
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
/* rt_functions.h pulls this in for the neutrino routines. It MUST be locked here,
 * with inline semantics, so the non-inline pass below does not re-emit the nuclear
 * inline bodies as strong host symbols owned by this TU. */
#include "../nuclear/nuclear_physics_functions.h"
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
/* rt_update_driftkick: body now in rt_functions.h (single source for host +
   device). Host external emitted by the non-inline re-include at the top of
   this file. */



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
