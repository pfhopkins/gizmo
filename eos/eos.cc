/* Standard/Kokkos headers before global_data_all_struct.h to avoid #define terminate
 * colliding with std::terminate in <exception> (same issue as cooling.cc). */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#ifdef OPENMP_GPU_OFFLOAD
#include <Kokkos_Core.hpp>
#endif

/* GPU All mirror: same pattern as cooling.cc — must precede allvars.h. */
#ifdef OPENMP_GPU_OFFLOAD
#include "../declarations/global_data_all_struct.h"
#endif
#if defined(OPENMP_GPU_OFFLOAD) && defined(GIZMO_GPU_COMPILER)
static __managed__ struct global_data_all_processes All_dev;
#define All All_dev  /* redirect All -> managed copy for ALL nvcc-compiled code (host+device) */
#endif

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"

/* GPU-safe isfinite/isnan overrides (see cooling.cc for explanation) */
#ifdef GIZMO_GPU_COMPILER
#undef isfinite
#undef isnan
#define isfinite(x) (((double)(x) == (double)(x)) && ((double)(x) - (double)(x) == 0.0))
#define isnan(x) ((double)(x) != (double)(x))
#endif

/*! Routines for gas equation-of-state terms (collects things like calculation of gas pressure)
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

/*! this pair of functions: 'return_user_desired_target_density' and 'return_user_desired_target_pressure' should be used
 together with 'HYDRO_GENERATE_TARGET_MESH'. This will attempt to move the mesh and mass
 towards the 'target' pressure profile. Use this to build your ICs.
 The 'desired' pressure and density as a function of particle properties (most commonly, position) should be provided in the function below */
double return_user_desired_target_density(int i)
{
    return 1; // uniform density everywhere -- will try to generate a glass //
    /*
     // this example would initialize a constant-density (density=rho_0) spherical cloud (radius=r_cloud) with a smooth density 'edge' (width=interp_width) surrounded by an ambient medium of density =rho_0/rho_contrast //
     double dx=pp[i].Pos[0]-boxHalf_X, dy=pp[i].Pos[1]-boxHalf_Y, dz=pp[i].Pos[2]-boxHalf_Z, r=sqrt(dx*dx+dy*dy+dz*dz);
     double rho_0=1, r_cloud=0.5*boxHalf_X, interp_width=0.1*r_cloud, rho_contrast=10.;
     return rho_0 * ((1.-1./rho_contrast)*0.5*erfc(2.*(r-r_cloud)/interp_width) + 1./rho_contrast);
     */
}
double return_user_desired_target_pressure(int i)
{
    return 1; // uniform pressure everywhere -- will try to generate a constant-pressure medium //
    /*
     // this example would initialize a radial pressure gradient corresponding to a self-gravitating, spherically-symmetric, infinite power-law
     //   density profile rho ~ r^(-b) -- note to do this right, you need to actually set that power-law for density, too, in 'return_user_desired_target_density' above
     double dx=pp[i].Pos[0]-boxHalf_X, dy=pp[i].Pos[1]-boxHalf_Y, dz=pp[i].Pos[2]-boxHalf_Z, r=sqrt(dx*dx+dy*dy+dz*dz);
     double b = 2.; return 2.*M_PI/fabs((3.-b)*(1.-b)) * pow(return_user_desired_target_density(i),2) * r*r;
     */
}

/* ---- BEGIN device-compilable EOS functions (for OPENMP_GPU_OFFLOAD cooling loop) ---- */
#ifdef OPENMP_GPU_OFFLOAD
#pragma omp begin declare target
#endif

/*!
    Updates the thermodynamic quantities determined by the current internal
    energy, density, and chemistry: pressure, and potentially adiabatic index
    and temperature if those are getting cached.

    this subroutine needs to set the value of the 'press' variable (pressure), which you can see from the
    templates below can follow an arbitrary equation-of-state. for more general equations-of-state you want to specifically set the soundspeed
    variable as well. */
/* set_eos_pressure: definition moved to cooling/cooling_functions.h for cross-TU GPU callability.
   eos.cc includes cooling_functions.h (below) so the inline definition is available to callers
   that link against eos.o. */


/* Get_Gas_Ionized_Fraction: definition now in eos_functions.h */


/* returns the dust-to-metals ratio normalized to the canonical solar value of 1/2: i.e. for 'standard' conditions, should = 1, but if e.g. all dust is sublimated, = 0;
    explicit dust formation-destruction modules should plug in here, to communicate to relevant opacity and other routines in heating/cooling/molecular chemistry cross-code
 T_dust_manual_override here designates whether we want to pass a specific dust temp or allow the code to call one itself here, to avoid creating circular dependencies and to
    allow for self-consistent coupling to various other dust dynamics/formation/chemistry modules. if you just want 'default' behavior and aren't worried about this, call with this parameter set to 0
 */
/* Function bodies now in _functions.h headers (single source of truth).
   Define KOKKOS_INLINE_FUNCTION as empty so functions are non-inline here,
   providing externally-visible symbols for other TUs that link via proto.h. */
#undef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION
#include "eos_functions.h"

/* CoolTables extern + aliases for cooling_functions.h (set_eos_pressure etc.) */
#ifdef COOLING
#if !defined(CHIMES)
#include "../cooling/cooling_tables.h"
extern struct cooling_tables_t CoolTables;
#define Tmin      CoolTables.Tmin
#define Tmax      CoolTables.Tmax
#define deltaT    CoolTables.deltaT
#define BetaH0    CoolTables.BetaH0
#define BetaHep   CoolTables.BetaHep
#define Betaff    CoolTables.Betaff
#define AlphaHp   CoolTables.AlphaHp
#define AlphaHep  CoolTables.AlphaHep
#define Alphad    CoolTables.Alphad
#define AlphaHepp CoolTables.AlphaHepp
#define GammaeH0  CoolTables.GammaeH0
#define GammaeHe0 CoolTables.GammaeHe0
#define GammaeHep CoolTables.GammaeHep
#define J_UV      CoolTables.J_UV
#define gJH0      CoolTables.gJH0
#define gJHep     CoolTables.gJHep
#define gJHe0     CoolTables.gJHe0
#define epsH0     CoolTables.epsH0
#define epsHep    CoolTables.epsHep
#define epsHe0    CoolTables.epsHe0
#ifdef COOL_METAL_LINES_BY_SPECIES
#define SpCoolTable0 CoolTables.SpCoolTable0
#define SpCoolTable1 CoolTables.SpCoolTable1
#endif
#endif
#endif
#include "../cooling/cooling_functions.h"

#ifdef OPENMP_GPU_OFFLOAD
#pragma omp end declare target
#endif
/* ---- END device-compilable EOS functions ---- */



/* subroutine to calculate the conductivities and assign the various non-ideal MHD coefficients (Ohmic, Hall, Ambipolar). can modify or expand chemistry or assumptions about grains herein. */
void calculate_and_assign_nonideal_mhd_coefficients(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
#ifdef MHD_NON_IDEAL
#ifdef COOLING // only try to get self-consistent resistivities after the first timestep, when we have calculated the self-consistent ionization state - on the first timestep default to 0 (=ideal MHD)
    if(All.Time <= All.TimeBegin) {cell[i].Eta_MHD_OhmicResistivity_Coeff = cell[i].Eta_MHD_HallEffect_Coeff = cell[i].Eta_MHD_AmbiPolarDiffusion_Coeff = 0; return;} // =0 on the first timestep, since we don't know the ionization yet
#endif
    /* calculations below follow Wardle 2007 and Keith & Wardle 2014, for the equation sets */
    double mean_molecular_weight = 2.38; // molecular H2, +He with solar mass fractions and metals
    double a_grain_micron = 0.1, f_dustgas = 0.01; // effective size of grains that matter at these densities
    double m_ion = 24.3; // Mg dominates ions in dense gas [where this is relevant]; this is ion mass in units of proton mass
    double zeta_cr = Get_CosmicRayIonizationRate_cgs(i, pp, cell); // cosmic ray ionization rate (fixed as constant for non-CR runs)
#ifdef COOLING
    double T_eff_atomic = 1.23 * (5./3.-1.) * U_TO_TEMP_UNITS * cell[i].InternalEnergyPred; /* we'll use this to make a quick approximation to the actual mean molecular weight here */
    double nH_cgs = cell[i].Density*All.cf_a3inv*UNIT_DENSITY_IN_NHCGS, T_transition=DMIN(8000.,nH_cgs), f_mol=1./(1. + T_eff_atomic*T_eff_atomic/(T_transition*T_transition));
    mean_molecular_weight = 4. / (1. + (3. + 4.*cell[i].Ne - 2.*f_mol) * HYDROGEN_MASSFRAC);
#endif
#ifdef METALS
    f_dustgas = 0.5 * pp[i].Metallicity[0] * return_dust_to_metals_ratio_vs_solar(i,0, pp, cell); // appropriate dust-to-metals ratio
#endif
    double temperature = mean_molecular_weight * (cell[i].gamma_eos_value()-1.) * U_TO_TEMP_UNITS * cell[i].InternalEnergyPred; // will use appropriate EOS to estimate temperature
    // now everything should be fully-determined (given the inputs above and the known properties of the gas) //
    double m_neutral = mean_molecular_weight; // in units of the proton mass
    double ag01 = a_grain_micron/0.1, m_grain = 7.51e9 * ag01*ag01*ag01; // grain mass [internal density =3 g/cm^3]
    double rho = cell[i].Density*All.cf_a3inv * UNIT_DENSITY_IN_CGS, n_eff = rho / PROTONMASS_CGS; // density in cgs
    // calculate ionization fraction in dense gas; use rate coefficients k to estimate grain charge
    double k0 = 1.95e-4 * ag01*ag01 * sqrt(temperature); // prefactor for rate coefficient for electron-grain collisions
    double ngr_ngas = (m_neutral/m_grain) * f_dustgas; // number of grains per neutral
    double psi_prefac = 167.1 / (ag01 * temperature); // e*e/(a_grain*k_boltzmann*T): Z_grain = psi/psi_prefac where psi is constant determines charge
    double alpha = zeta_cr * psi_prefac / (ngr_ngas*ngr_ngas * k0 * (n_eff/m_neutral)); // coefficient for equation that determines Z_grain
    // psi solves the equation: psi = alpha * (exp[psi] - y/(1+psi)) where y=sqrt(m_ion/m_electron); note the solution for small alpha is independent of m_ion, only large alpha
    //   (where the non-ideal effects are weak, generally) produces a difference: at very high-T, appropriate m_ion should be hydrogen+helium, but in this limit our cooling
    //    routines will already correctly determine the ionization states. so we can safely adopt Mg as our ion of consideration
    double y=sqrt(m_ion*PROTONMASS_CGS/ELECTRONMASS_CGS), psi_0 = 0.5188025-0.804386*log(y), psi=psi_0; // solution for large alpha [>~10]
    if(alpha<0.002) {psi=alpha*(1.-y)/(1.+alpha*(1.+y));} else if(alpha<10.) {psi=psi_0/(1.+0.027/alpha);} // accurate approximation for intermediate values we can use here
    double k_e = k0 * exp(psi); // e-grain collision rate coefficient
    double k_i = k0 * sqrt(ELECTRONMASS_CGS / (m_ion*PROTONMASS_CGS)) * (1 - psi); // i-grain collision rate coefficient
    double n_elec = zeta_cr / (ngr_ngas * k_e); // electron number density
    double n_ion = zeta_cr / (ngr_ngas * k_i); // ion number density
    double Z_grain = psi / psi_prefac; // mean grain charge (note this is signed, will be negative)
#ifdef COOLING
    double mu_eff=2.38, x_elec=DMAX(1.e-18, cell[i].Ne*HYDROGEN_MASSFRAC*mu_eff), R=x_elec*psi_prefac/ngr_ngas; psi_0=-3.787124454911839; n_elec=x_elec*n_eff/mu_eff; // R is essentially the ratio of negative charge in e- to dust: determines which regime we're in to set quantities below
    if(R > 100.) {psi=psi_0;} else if(R < 0.002) {psi=R*(1.-y)/(1.+2.*y*R);} else {psi=psi_0/(1.+pow(R/0.18967,-0.5646));} // simple set of functions to solve for psi, given R above, using the same equations used to determine low-temp ion fractions
    n_ion = n_elec * y * exp(psi)/(1.-psi); Z_grain = psi / psi_prefac; // we can immediately now calculate these from the above
#endif
    // now define more variables we will need below //
    double gizmo2gauss = UNIT_B_IN_GAUSS; // convert to B-field to gauss (units)
    double B_Gauss = cell[i].Bfield().norm_sq(); // get magnitude of B //
    if(B_Gauss<=0) {B_Gauss=0;} else {B_Gauss = sqrt(B_Gauss) * All.cf_a2inv * gizmo2gauss;} // B-field magnitude in Gauss
    double xe = n_elec / n_eff;
    double xi = n_ion / n_eff;
    double xg = ngr_ngas;
    // get collision rates/cross sections for different species //
    double nu_g = 7.90e-6 * ag01*ag01 * sqrt(temperature/m_neutral) / (m_neutral+m_grain); // Pinto & Galli 2008
    double nu_ei = 51.*xe*pow(temperature,-1.5); // Pandey & Wardle 2008 (e-ion)
    double nu_e = nu_ei + 6.21e-9*pow(temperature/100.,0.65)/m_neutral; // Pinto & Galli 2008 for latter (e-neutral)
    double nu_ie = ((ELECTRONMASS_CGS*xe)/(m_ion*PROTONMASS_CGS*xi))*nu_ei; // Pandey & Wardle 2008 for former (e-ion)
    double nu_i = nu_ie + 1.57e-9/(m_neutral+m_ion); // Pandey & Wardle 2008 for former (e-ion), Pinto & Galli 2008 for latter (i-neutral)
    // use the cross sections to determine the hall parameters and conductivities //
    double beta_prefac = ELECTRONCHARGE_CGS * B_Gauss / (PROTONMASS_CGS * C_LIGHT_CGS * n_eff);
    double beta_i = beta_prefac / (m_ion * nu_i); // standard beta factors (Hall parameters)
    double beta_e = beta_prefac / (ELECTRONMASS_CGS/PROTONMASS_CGS * nu_e);
    double beta_g = beta_prefac / (m_grain * nu_g) * fabs(Z_grain);
    double be_inv = 1/(1 + beta_e*beta_e), bi_inv = 1/(1 + beta_i*beta_i), bg_inv = 1/(1 + beta_g*beta_g);
    double sigma_O = xe*beta_e + xi*beta_i + xg*fabs(Z_grain)*beta_g; // ohmic conductivity
    double sigma_H = -xe*be_inv + xi*bi_inv + xg*Z_grain*bg_inv; // hall conductivity
    double sigma_P = xe*beta_e*be_inv + xi*beta_i*bi_inv + xg*fabs(Z_grain)*beta_g*bg_inv; // pedersen conductivity
    double sign_Zgrain = Z_grain/fabs(Z_grain); if(Z_grain==0) {sign_Zgrain=0;}
    double sigma_A2 = (xe*beta_e*be_inv)*(xi*beta_i*bi_inv)*pow(beta_i+beta_e,2) +
    (xe*beta_e*be_inv)*(xg*fabs(Z_grain)*beta_g*bg_inv)*pow(sign_Zgrain*beta_g+beta_e,2) +
    (xi*beta_i*bi_inv)*(xg*fabs(Z_grain)*beta_g*bg_inv)*pow(sign_Zgrain*beta_g-beta_i,2); // alternative formulation which is automatically positive-definite
    // now we can finally calculate the diffusivities //
    double eta_prefac = B_Gauss * C_LIGHT_CGS / (4 * M_PI * ELECTRONCHARGE_CGS * n_eff );
    double eta_O = eta_prefac / sigma_O;
    double sigma_perp2 = sigma_H*sigma_H + sigma_P*sigma_P;
    double eta_H = eta_prefac * sigma_H / sigma_perp2;
    double eta_A = eta_prefac * (sigma_A2)/(sigma_O*sigma_perp2);
    eta_O = fabs(eta_O); eta_A = fabs(eta_A); // these depend on the absolute values and should be written as such, so eta is always positive [not true for eta_h]
    // convert units to code units
    double x_neutral = DMAX(0., 1-m_ion*xi); // neutral fraction of mass
    double units_cgs_to_code = UNIT_TIME_IN_CGS / (UNIT_LENGTH_IN_CGS * UNIT_LENGTH_IN_CGS); // convert coefficients (L^2/t) to code units [physical]
    double eta_ohmic = eta_O*units_cgs_to_code, eta_hall = eta_H*units_cgs_to_code, eta_ad = x_neutral * eta_A*units_cgs_to_code;
#ifdef MHD_NON_IDEAL_CORRECTIONTERMS /* account for unphysical or not internally self-consistent drift/slip speeds (PFH+Squire 24; arXiv:2405.06026) */
    double gradbmag2=0,gradbmag=0,btmp=0,L_B=MAX_REAL_NUMBER;
    gradbmag2 = cell[i].Gradients.B.frobenius_norm_sq(); // need to get magnitude of B gradient for below
    if(gradbmag2>0) {gradbmag=sqrt(gradbmag2)*(All.cf_a2inv/All.cf_atime)*gizmo2gauss; L_B=(B_Gauss/gradbmag)*UNIT_LENGTH_IN_CGS;} // L_B is gradient length in cgs
    double xi_AbsZi_eff = (xe*beta_e + (xi+1.e-25)*beta_i + xg*fabs(Z_grain)*beta_g) / (beta_e + beta_i + beta_g); // weighted mean xi_qi to use
    double psi_n = (xe/(1.+beta_e) + xi/(1.+beta_i) + xg*fabs(Z_grain)/(1.+beta_g)) / (xe+xi+xg*fabs(Z_grain) + 1.e-20); // coupling parameter for weight of neutrals in effective speed
    if(xe+xi+xg < 1.e-20) {psi_n = 1.;}
    double m_carrier_weighted = PROTONMASS_CGS * (xe*ELECTRONMASS_CGS/PROTONMASS_CGS + xi*m_ion + xg*fabs(Z_grain)*m_grain + psi_n*m_neutral); // effective weight of the dragged carriers for speeds below
    double vT_crit = sqrt( BOLTZMANN_CGS*temperature * (xe + xi + xg*fabs(Z_grain) + psi_n) / m_carrier_weighted ); // salient thermal speed for superthermal drift
    double vA_crit = B_Gauss / sqrt(4.*M_PI*m_carrier_weighted*n_eff); // effective Alfven speed to compare as well
    double vT_an = DMIN(vT_crit, vA_crit); // speed for anomalous term to kick on
    double vdrift_mag = ((B_Gauss / L_B) * C_LIGHT_CGS) / (4.*M_PI*ELECTRONCHARGE_CGS * n_eff * xi_AbsZi_eff); // drift magnitude in CGS
    double vslip_mag_over_vT = (eta_A/L_B) / vT_crit; // vslip_perp over vT_crit in CGS
    double one_plus_vslip2_veff2 = 0.5 + sqrt(0.25 + vslip_mag_over_vT*vslip_mag_over_vT); // 'effective' vslip, accounting for full nonlinear terms
    eta_ohmic *= sqrt(one_plus_vslip2_veff2 + vdrift_mag*vdrift_mag/(vT_crit*vT_crit)); // epstein-type correction for superthermal drift or slip - here rescaling of Ohmic for drift+slip
    eta_ad /= sqrt(one_plus_vslip2_veff2); // epstein-type correction for superthermal drift or slip - here rescaling of AD for slip
    double eta_an = (eta_prefac * units_cgs_to_code / xi_AbsZi_eff) * sqrt(1. + 4.*M_PI*C_LIGHT_CGS*C_LIGHT_CGS*ELECTRONMASS_CGS*n_eff*xe/(B_Gauss*B_Gauss)); // anomalous resistivity, set to max of either gyro frequency or electron plasma frequency (dust plasma frequency should be lower unless in dusty plasma regime, where behavior is much more complicated)
    if(eta_an < fabs(eta_hall)+fabs(eta_ad)) {eta_an = fabs(eta_hall)+fabs(eta_ad);} // anomalous term should always be largest frequency
    eta_an *= sqrt(1. + (vdrift_mag*vdrift_mag)/(vT_an*vT_an)); // epstein-type correction for the anomalous term itself
    if(vdrift_mag > vT_an/40.) {eta_ohmic += eta_an * exp(-vT_an/vdrift_mag);} // factor to represent boosted ohmic to diffuse when exceed critical limit
    double eta_max_corr = 1.e24 * units_cgs_to_code; // limit maximum value to the diffusivity
    if(eta_ohmic > eta_max_corr) {double fcorr=eta_max_corr/eta_ohmic; eta_ohmic*=fcorr; eta_hall*=fcorr; eta_ad*=fcorr;} // rescale so relative magnitude is preserved
#endif
    double etamag = fabs(eta_ohmic) + fabs(eta_hall) + fabs(eta_ad); // maximum (total) magnitude of the non-ideal terms
    double etamax = 1.e24 * units_cgs_to_code; // cap coefficients -- this is problem specific but for disks and related problems, can safely assume this cap
    double etacor = 1.; if(etamag > etamax) {etacor = etamax/etamag;} // suppression factor
    eta_ohmic *= etacor; eta_hall *= etacor; eta_ad *= etacor; // applied

    cell[i].Eta_MHD_OhmicResistivity_Coeff = eta_ohmic;     /*!< Ohmic resistivity coefficient [physical units of L^2/t] */
    cell[i].Eta_MHD_HallEffect_Coeff = eta_hall;            /*!< Hall effect coefficient [physical units of L^2/t] */
    cell[i].Eta_MHD_AmbiPolarDiffusion_Coeff = eta_ad;      /*!< Hall effect coefficient [physical units of L^2/t] */
#endif
}


/* subroutine to calculate and assign the Spitzer-Braginskii conduction and viscosity coefficients, with appropriate limiters for high-beta plasmas, saturated behaviors, and optional practical numerical limiters when in extreme regimes */
void calculate_and_assign_conduction_and_viscosity_coefficients(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
#if defined(CONDUCTION) || defined(VISCOSITY)
    
#if (defined(CONDUCTION_SPITZER) || defined(VISCOSITY_BRAGINSKII))
    double ion_frac, beta_i; int k; k=0; ion_frac=1; beta_i=1.e20; double rho=cell[i].Density*All.cf_a3inv, u_int=cell[i].InternalEnergyPred;
#if defined(COOLING) /* get the ionized fraction. NOTE we CANNOT call 'ThermalProperties' or functions like 'Get_Ionized_Fraction' here in gradients.c, as we have not done self-shielding steps yet and most modules will yield unphysical answers! */
    ion_frac = cell[i].Ne / (1. + 2.*yhelium(i, pp)); /* quick estimator. this is actually what we need for conduction since its the free electrons conducting, and we want number relative to fully-ionized gas */
#endif
    double vf_lim,cs,cs_therm; cs=cell[i].effective_soundspeed(); cs_therm=cell[i].thermal_soundspeed(); vf_lim = cs;
#ifdef MAGNETIC
    Vec3<double> bhat = cell[i].Bfield(); double bmag=0,double_dot_dv=0;
    bmag=bhat.norm_sq(); if(bmag>0) {bmag = sqrt(bmag); bhat/=bmag;}
    beta_i = bmag*bmag  * All.cf_a3inv / (All.cf_atime * rho * cs_therm * cs_therm);
    vf_lim *= DMIN(1.e4 , sqrt(1.+beta_i));
#endif
#endif
    
#ifdef CONDUCTION
    cell[i].Kappa_Conduction = All.ConductionCoeff;
#ifdef CONDUCTION_SPITZER
    /* calculate the thermal conductivities: use the Spitzer formula */
    cell[i].Kappa_Conduction *= ion_frac * pow(u_int, 2.5);
    /* account for saturation (when the mean free path of electrons is large): estimate whether we're in that limit with the gradients */
    double electron_free_path = All.ElectronFreePathFactor * u_int * u_int / rho;
    double du_conduction=0; for(k=0;k<3;k++) {du_conduction += cell[i].Gradients.InternalEnergy[k] * cell[i].Gradients.InternalEnergy[k];}
    double temp_scale_length = u_int / sqrt(du_conduction) * All.cf_atime;
    /* also the Whistler instability limits the heat flux at high-beta; Komarov et al., arXiv:1711.11462 (2017) */
    cell[i].Kappa_Conduction /= (1 + (4.2 + 1./(3.*beta_i)) * electron_free_path / temp_scale_length); /* should be in physical units */
#ifdef DIFFUSION_OPTIMIZERS
    cell[i].Kappa_Conduction = DMIN(cell[i].Kappa_Conduction , 42.85 * rho * vf_lim * DMIN(20.*pp[i].Get_Particle_Size()*All.cf_atime , temp_scale_length));
#endif
#endif
#endif
    
#ifdef VISCOSITY
    cell[i].Eta_ShearViscosity = All.ShearViscosityCoeff; cell[i].Zeta_BulkViscosity = All.BulkViscosityCoeff;
#ifdef VISCOSITY_BRAGINSKII
    /* calculate the viscosity coefficients: use the Braginskii shear tensor formulation expanded to first order */
    cell[i].Eta_ShearViscosity *= ion_frac * pow(u_int, 2.5); cell[i].Zeta_BulkViscosity = 0;
    /* again need to account for possible saturation (when the mean free path of ions is large): estimate whether we're in that limit with the gradients */
    double ion_free_path = All.ElectronFreePathFactor * u_int * u_int / rho; double dv_magnitude=0, v_magnitude=0;
    /* need an estimate of the internal energy gradient scale length, which we get by d(P/rho) = P/rho * (dP/P - drho/rho) */
    for(k=0;k<3;k++) {int k1;
        for(k1=0;k1<3;k1++) {
            dv_magnitude += cell[i].Gradients.Velocity[k][k1]*cell[i].Gradients.Velocity[k][k1];
#ifdef MAGNETIC
            double_dot_dv += cell[i].Gradients.Velocity[k][k1] * bhat[k]*bhat[k1] * All.cf_a2inv; // physical units
#endif
        }
        v_magnitude += cell[i].VelPred[k]*cell[i].VelPred[k];
    }
    double vel_scale_length = sqrt( v_magnitude / dv_magnitude ) * All.cf_atime;
    /* also limit to saturation magnitude ~ signal_speed / lambda_MFP^2 */
    cell[i].Eta_ShearViscosity /= (1 + 4.2 * ion_free_path / vel_scale_length); /* should be in physical units */
#ifdef MAGNETIC
    // following Jono Squire's notes, the mirror and firehose instabilities limit pressure anisotropies [which scale as the viscous term inside the gradient: nu_braginskii*(bhat.bhat:grad.v)] to >-2*P_magnetic and <1*P_magnetic
    double P_effective_visc = cell[i].Eta_ShearViscosity * double_dot_dv;
    double P_magnetic = 0.5 * (bmag*All.cf_a2inv) * (bmag*All.cf_a2inv);
    if(P_effective_visc < -2.*P_magnetic) {cell[i].Eta_ShearViscosity = 2.*P_magnetic / fabs(double_dot_dv);}
    if(P_effective_visc > P_magnetic) {cell[i].Eta_ShearViscosity = P_magnetic / fabs(double_dot_dv);}
#endif
#ifdef DIFFUSION_OPTIMIZERS
    double eta_sat = rho * vf_lim / (ion_free_path * (1 + 4.2 * ion_free_path / vel_scale_length));
    if(eta_sat <= 0) {cell[i].Eta_ShearViscosity=0;}
    if(cell[i].Eta_ShearViscosity>0) {cell[i].Eta_ShearViscosity = 1. / (1./cell[i].Eta_ShearViscosity + 1./eta_sat);} // again, all physical units //
#endif
#endif
#endif
    
#endif
}


#ifdef TURB_DIFFUSION
void calculate_and_assign_turbulent_diffusion_coefficients(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    /* estimate local turbulent diffusion coefficient from velocity gradients using Smagorinsky mixing model:
     we do this after slope-limiting to prevent the estimated velocity gradients from being unphysically large */
    double h_turb = pp[i].Get_Particle_Size() * All.cf_atime; // physical
    if(h_turb > 0)
    {
        // overall normalization //
        double C_Smagorinsky_Lilly = 0.15; // this is the standard Smagorinsky-Lilly constant, calculated from Kolmogorov theory: should be 0.1-0.2 //
        double turb_prefactor = 0.25 * All.TurbDiffusion_Coefficient * C_Smagorinsky_Lilly*C_Smagorinsky_Lilly * sqrt(2.0);
        // then scale with inter-particle spacing //
        turb_prefactor *= h_turb*h_turb;
        // calculate frobenius norm of symmetric shear velocity gradient tensor //
        double shear_factor = sqrt((1./2.)*((cell[i].Gradients.Velocity[1][0]+cell[i].Gradients.Velocity[0][1]) *
                                            (cell[i].Gradients.Velocity[1][0]+cell[i].Gradients.Velocity[0][1]) +
                                            (cell[i].Gradients.Velocity[2][0]+cell[i].Gradients.Velocity[0][2]) *
                                            (cell[i].Gradients.Velocity[2][0]+cell[i].Gradients.Velocity[0][2]) +
                                            (cell[i].Gradients.Velocity[2][1]+cell[i].Gradients.Velocity[1][2]) *
                                            (cell[i].Gradients.Velocity[2][1]+cell[i].Gradients.Velocity[1][2])) +
                                   (2./3.)*((cell[i].Gradients.Velocity[0][0]*cell[i].Gradients.Velocity[0][0] +
                                             cell[i].Gradients.Velocity[1][1]*cell[i].Gradients.Velocity[1][1] +
                                             cell[i].Gradients.Velocity[2][2]*cell[i].Gradients.Velocity[2][2]) -
                                            (cell[i].Gradients.Velocity[1][1]*cell[i].Gradients.Velocity[2][2] +
                                             cell[i].Gradients.Velocity[0][0]*cell[i].Gradients.Velocity[1][1] +
                                             cell[i].Gradients.Velocity[0][0]*cell[i].Gradients.Velocity[2][2])));
        // slope-limit and convert to physical units //
        double shearfac_max = 0.5 * sqrt(cell[i].VelPred[0]*cell[i].VelPred[0]+cell[i].VelPred[1]*cell[i].VelPred[1]+cell[i].VelPred[2]*cell[i].VelPred[2]) / h_turb;
        shear_factor = DMIN(shear_factor , shearfac_max * All.cf_atime) * All.cf_a2inv; // physical
#ifdef TURB_DIFF_DYNAMIC
        int u, v; double trace = 0;
        shearfac_max = 0.5 * sqrt(cell[i].Velocity_bar[0] * cell[i].Velocity_bar[0] + cell[i].Velocity_bar[1] * cell[i].Velocity_bar[1]+cell[i].Velocity_bar[2] * cell[i].Velocity_bar[2]) * All.cf_atime / h_turb;
        for (u = 0; u < 3; u++) {
            for (v = 0; v < 3; v++) {
                if (cell[i].VelShear_bar[u][v] < 0) {cell[i].VelShear_bar[u][v] = DMAX(cell[i].VelShear_bar[u][v], -shearfac_max);}
                else {cell[i].VelShear_bar[u][v] = DMIN(cell[i].VelShear_bar[u][v], shearfac_max);}
                if (u == v) {trace += cell[i].VelShear_bar[u][u];}}}
        /* If it was already trace-free, don't zero out the diagonal components */
        if (trace != 0 && NUMDIMS > 1) {for (u = 0; u < NUMDIMS; u++) {cell[i].VelShear_bar[u][u] -= 1.0 / NUMDIMS * trace;}}
        for (u = 0; u < 3; u++) { /* Don't want to recalculate these a bunch later on, so save them */
            cell[i].Velocity_hat[u] *= All.TurbDynamicDiffSmoothing;
            for (v = 0; v < 3; v++) {cell[i].MagShear_bar += cell[i].VelShear_bar[u][v] * cell[i].VelShear_bar[u][v];}}
        cell[i].MagShear = sqrt(2.0) * shear_factor / All.cf_a2inv; // Don't want this physical
        cell[i].MagShear_bar = DMIN(sqrt(2.0 * cell[i].MagShear_bar), shearfac_max); turb_prefactor /= 0.25;
#endif
        // ok, combine to get the diffusion coefficient //
        cell[i].TD_DiffCoeff = turb_prefactor * shear_factor; // physical
    } else {
        cell[i].TD_DiffCoeff = 0;
    }
#ifdef TURB_DIFF_ENERGY
#if !defined(CONDUCTION_SPITZER)
    cell[i].Kappa_Conduction = 0;
#endif
    cell[i].Kappa_Conduction += All.ConductionCoeff * cell[i].TD_DiffCoeff * cell[i].Density * All.cf_a3inv; // physical
#endif
#ifdef TURB_DIFF_VELOCITY
#if !defined(VISCOSITY_BRAGINSKII)
    cell[i].Eta_ShearViscosity = 0; cell[i].Zeta_BulkViscosity = 0;
#endif
    cell[i].Eta_ShearViscosity += All.ShearViscosityCoeff * cell[i].TD_DiffCoeff * cell[i].Density * All.cf_a3inv; // physical
    cell[i].Zeta_BulkViscosity += All.BulkViscosityCoeff * cell[i].TD_DiffCoeff * cell[i].Density * All.cf_a3inv; // physical
#endif
}
#endif


#if defined(OPENMP_GPU_OFFLOAD) && defined(GIZMO_GPU_COMPILER)
void gizmo_gpu_sync_all_eos(void) {
#pragma push_macro("All")
#undef All
    extern struct global_data_all_processes All;
    All_dev = All;
#pragma pop_macro("All")
}
#elif defined(OPENMP_GPU_OFFLOAD)
void gizmo_gpu_sync_all_eos(void) {} /* no-op: no __managed__ copy with OpenMP backend */
#endif
