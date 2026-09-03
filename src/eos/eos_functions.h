/* eos_functions.h — Canonical KOKKOS_INLINE_FUNCTION implementations of EOS
 * utility routines (Get_Gas_Molecular_Mass_Fraction, yhelium,
 * Get_Gas_Mean_Molecular_Weight_mu).  These are the single source of truth —
 * included by both eos.cc and cooling.cc.  No separate _device.h copy.
 *
 * Include order: after allvars.h (for struct types and All).
 *
 * This header is included by:
 *   cooling/cooling.cc  — so the GPU kernel can call these inline
 *   eos/eos.cc          — so the non-GPU host path still has one definition
 */
#pragma once

/* Fallback for non-Kokkos builds: KOKKOS_INLINE_FUNCTION is defined by
 * <Kokkos_Core.hpp> as "inline __device__ __host__ __forceinline__" in GPU
 * builds.  In non-GPU builds Kokkos is not included, so define it as plain
 * inline so host-only compilation still works. */
#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif
#ifndef KOKKOS_FUNCTION
#define KOKKOS_FUNCTION
#endif

/* NOTE (Phase D 2026-05-21 config 118): the EOS_TRUELOVE_PRESSURE branch in
 * set_eos_pressure_impl below uses KERNEL_FAC_FROM_FORCESOFT_TO_PLUMMER from
 * mesh/kernel.h. We do NOT include mesh/kernel.h here: kernel.h lacks
 * header guards and re-defining its inline-static helpers in the same TU
 * (when another header already pulled it in) causes redefinition errors.
 * Consumers (cooling/cooling.cc, eos/eos.cc) MUST include mesh/kernel.h
 * BEFORE this header under the EOS_TRUELOVE_PRESSURE /
 * TRUELOVE_CRITERION_PRESSURE gate. */

/* Forward declaration for get_equilibrium_dust_temperature_estimate (body in
 * radiation/rt_functions.h). Needed here because return_dust_to_metals_ratio_vs_solar
 * (defined below in this header) calls it under the COOL_LOW_TEMPERATURES branch.
 * KOKKOS_FUNCTION (NOT KOKKOS_INLINE_FUNCTION) attribute so the host external symbol
 * still gets emitted by radiation/rt_utilities.cc's non-inline re-include pass;
 * KOKKOS_INLINE_FUNCTION here would make the function inline everywhere, suppressing
 * the strong host symbol and causing link errors from eos.o/io.o/gravtree.cc callers.
 * The KOKKOS_FUNCTION attribute is enough for nvc++ to recognize the call as
 * device-callable inside __host__ __device__ inlining contexts (avoids silent
 * #20011-D physics error on the device pass). */
KOKKOS_FUNCTION double get_equilibrium_dust_temperature_estimate(int i, double shielding_factor_for_exgalbg, double T, struct particle_data *pp, struct gas_cell_data *cell);

/* Forward declaration for return_dust_to_metals_ratio_vs_solar (defined below
 * in this header at line ~298). Needed because Get_Gas_Molecular_Mass_Fraction
 * (below at line ~49) calls it under the (COOL_MOLECFRAC == 5) branch, and not
 * every TU that includes eos_functions.h has seen proto.h first (cooling.cc
 * includes eos_functions.h at line 64, proto.h at line 69). Surface-and-fix
 * 2026-05-27 -- bug surfaced under GALSF_FB_FIRE_STELLAREVOLUTION>2 + COOLING
 * + GALSF_SFR_CRITERION & 256 (eos_functions.h:75-77 implicitly sets
 * COOL_MOLECFRAC=5, activating the line 104+ block). KOKKOS_INLINE_FUNCTION
 * storage class matches the eventual definition. */
KOKKOS_INLINE_FUNCTION double return_dust_to_metals_ratio_vs_solar(int i, double T_dust_manual_override, struct particle_data *pp, struct gas_cell_data *cell);

#include "../declarations/multifluid_helpers.h"

/* return an estimate of the Hydrogen molecular fraction of gas */
KOKKOS_INLINE_FUNCTION double Get_Gas_Molecular_Mass_Fraction(int i, double temperature, double neutral_fraction, double free_electron_ratio, double urad_from_uvb_in_G0, struct particle_data *pp, struct gas_cell_data *cell)
{
#ifdef HYDRO_MULTIFLUID_DM
    if(pp[i].FluidType == FLUID_DM) return 0; /* dark fluid: no molecular tracking in placeholder model */
#endif
#ifdef GALSF_EFFECTIVE_EQS
    return 0; /* in the effective equation of state, H2 is not tracked explicitly here and the cooling function explicitly assumes an ionized+atomic medium. the 'molecular' compoennt is part of the implicit sub-grid model for clouds. so any non-zero value here will be invalid */
#endif

    /* if tracking chemistry explicitly, return the explicitly-evolved H2 fraction */
#ifdef CHIMES // use the CHIMES molecular network for H2
    return DMIN(1,DMAX(0, ChimesGasVars[i].abundances[ChimesGlobalVars.speciesIndices[sp_H2]] * 2.0)); // factor 2 converts to mass fraction in molecular gas, as desired
#endif

#if (COOL_GRACKLE_CHEMISTRY >= 2) // Use GRACKLE explicitly-tracked H2 [using the molecular network if this is valid]
    return DMIN(1,DMAX(0, cell[i].grH2I + cell[i].grH2II)); // include both states of H2 tracked
#endif

#if defined(COOL_MOLECFRAC_NONEQM) // use our simple 1-species network for explicitly-evolved H2 fraction
    return DMIN(1, DMAX(0, cell[i].MolecularMassFraction));
#endif

#if defined(GALSF_FB_FIRE_RT_HIIHEATING)
    if(cell[i].DelayTimeHII > 0) {return 0;} // force gas flagged as in HII regions to have zero molecular fraction
#endif

#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2) && !defined(COOL_MOLECFRAC_NONEQM) && !defined(COOL_MOLECFRAC) && defined(COOLING) /* set default module we will use here */
#define COOL_MOLECFRAC 5
#endif

#if (COOL_MOLECFRAC == 5) || (COOL_MOLECFRAC == 4) || (COOL_MOLECFRAC == 3) // here are some of the 'fancy' molecular fraction estimators which need various additional properties
    double T=1, nH_cgs=1, Z_Zsol=1, urad_G0=1, xH0=1, x_e=0; // initialize definitions of some variables used below to prevent compiler warnings
    if(temperature > 3.e5) {return 0;} else {T=temperature;} // approximations below not designed for high temperatures, should simply give null
    xH0 = DMIN(DMAX(neutral_fraction,0.),1.); // get neutral fraction [given by call to this program]
    x_e = DMIN(DMAX(free_electron_ratio,0.),2.); // get free electron ratio [number per H nucleon]
    nH_cgs = cell[i].Density*All.cf_a3inv * UNIT_DENSITY_IN_NHCGS; // get nH defined as number of nucleons per cm^3
    Z_Zsol=1; urad_G0=1; // initialize metal and radiation fields. will assume solar-Z and spatially-uniform Habing field for incident FUV radiation unless reset below.
#ifdef METALS
    Z_Zsol = pp[i].Metallicity[0]/All.SolarAbundances[0]; // metallicity in solar units [scale to total Z, since this mixes dust and C opacity], and enforce a low-Z floor to prevent totally unphysical behaviors at super-low Z [where there is still finite opacity in reality; e.g. Kramer's type and other opacities enforce floor around ~1e-3]
#endif
    /* get incident radiation field from whatever module we are using to track it */
#ifdef GALSF_FB_FIRE_RT_LONGRANGE
    urad_G0 = DMAX(cell[i].Rad_Flux_UV, 1.e-10); // note this is ALREADY self-shielded, so we need to be careful about 2x-counting the self-shielding approximation below; hence limit this to a rather sizeable value  //
#endif
#if defined(RT_PHOTOELECTRIC) || defined(RT_LYMAN_WERNER)
    int whichbin = RT_FREQ_BIN_LYMAN_WERNER;
#if !defined(RT_LYMAN_WERNER)
    whichbin = RT_FREQ_BIN_PHOTOELECTRIC; // use photo-electric bin as proxy (very close) if don't evolve LW explicitly
#endif
    urad_G0 = cell[i].Rad_E_gamma[whichbin] * (cell[i].Density*All.cf_a3inv/pp[i].Mass) * UNIT_PRESSURE_IN_CGS / 3.9e-14; // convert to Habing field //
#endif
    urad_G0 += urad_from_uvb_in_G0; // include whatever is contributed from the meta-galactic background, fed into this routine
    urad_G0 = DMIN(DMAX( urad_G0 , 1.e-10 ) , 1.e10 ); // limit values, because otherwise exponential self-shielding approximation easily artificially gives 0 incident field
#endif

#if (COOL_MOLECFRAC == 5) // ??? -- update to match noneqm fancier cooling functions --
    /* estimate local equilibrium molecular fraction actually using the real formation and destruction rates. expressions for the different rate terms
        as used here are collected in Nickerson, Teyssier, & Rosdahl et al. 2018. Expression for the line self-shielding here
        including turbulent and cell line blanketing terms comes from Gnedin & Draine 2014. below solves this all exactly, using the temperature, metallicity,
        density, ionization states, FUV incident radiation field, and column densities in the simulations. */
    /* take eqm of dot[nH2] = a_H2*rho_dust*nHI [dust formation] + a_GP*nHI*ne [gas-phase formation] + b_3B*nHI*nHI*(nHI+nH2/8) [3-body collisional form] - b_H2HI*nHI*nH2 [collisional dissociation]
        - b_H2H2*nH2*nH2 [collisional mol-mol dissociation] - Gamma_H2^LW * nH2 [photodissociation] - Gamma_H2^+ [photoionization] - xi_H2*nH2 [CR ionization/dissociation] */
    double fH2=0, sqrt_T=sqrt(T), nH0=xH0*nH_cgs, n_e=x_e*nH_cgs, EXPmax=40., clumping_factor=1; // define some variables for below, including neutral H number density, free electron number, etc.
    double f_dustgas_solar = 0.5*Z_Zsol*return_dust_to_metals_ratio_vs_solar(i,0, pp, cell); // dust-to-gas ratio locally
    double Tdust = 30.; // need to assume something about dust temperature for reaction rates below for dust-phase formation
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2) || defined(SINGLE_STAR_SINK_DYNAMICS)
    Tdust = get_equilibrium_dust_temperature_estimate(i, 1, T, pp, cell);
#endif
    double a_Z = 3.e-18*sqrt_T / ((1. +4.e-2*sqrt(T+Tdust) +2.e-3*T +8.e-6*T*T )*(1. +1.e4/exp(DMIN(EXPmax,600./Tdust)))) * f_dustgas_solar * nH_cgs * nH0 * clumping_factor; // dust surface formation (assuming dust-to-metals ratio is 0.5*(Z/solar)*dust-to-gas-relative-to-solar in all regions where this is significant), from Glover & Jappsen 2007
    //double a_GP = (1.833e-21 * pow(T,0.88)) * nH0 * n_e; // gas-phase formation [old form, from Nickerson et al., appears to be a significant typo in their expression compared to the sources from which they extracted it]
    double a_GP = (1.833e-18 * pow(T,0.88)) * nH0 * n_e / (1. + x_e*1846.*(1.+T/20000.)/sqrt(T)); // gas-phase formation [Glover & Abel 2008, using fitting functions slightly more convenient and assuming H-->H2 much more rapid than other reactions, from Krumholz & McKee 2010; denominator factor accounts for p+H- -> H + H, instead of H2]
    double b_3B = (6.0e-32/sqrt(sqrt_T) + 2.0e-31/sqrt_T) * nH0 * nH0 * nH0; // 3-body collisional formation
    double b_H2HI = (7.073e-19 * pow(T,2.012) * exp(-DMIN(5.179e4/T,EXPmax)) / pow(1. + 2.130e-5*T , 3.512)) * nH0 * (nH0/2.); // collisional dissociation
    b_H2HI += 4.49e-9 * pow(T,0.11) * exp(-DMIN(101858./T,EXPmax)) * (n_e) * (nH0/2.); // collisional H2-e- dissociation [note assuming ground-state optically thin dissociation here as thats where this is most relevant, see Glover+Abel 2008)
    double b_H2H2 = (5.996e-30 * pow(T,4.1881) * exp(-DMIN(5.466e4/T,EXPmax)) / pow(1. + 6.761e-6*T , 5.6881)) * (nH0/2.) * (nH0/2.); // collisional mol-mol dissociation
    double G_LW = 3.3e-11 * urad_G0 * (nH0/2.); // photo-dissociation (+ionization); note we're assuming a spectral shape identical to the MW background mean, scaling by G0
    double xi_cr_H2 = (7.525e-16) * (nH0/2.); // CR dissociation (+ionization)
    // can write this as a quadtratic: 0 = x_a*f^2 - x_b*f + x_c, with f = molec mass fraction
    double x_a = (b_3B + b_H2HI - b_H2H2); // terms quadratic in f -- this term can in principle be positive or negative, usually positive
    double x_b = (a_GP + a_Z + 2.*b_3B + b_H2HI + G_LW + xi_cr_H2); // terms linear in f [note sign, pulling the -sign out here] -- positive-definite
    double x_c = (a_GP + a_Z + b_3B); // terms independent of f -- positive-definite
    double y_a = x_a / (x_c + MIN_REAL_NUMBER), y_b = x_b / (x_c + MIN_REAL_NUMBER), z_a = 4. * y_a / (y_b*y_b + MIN_REAL_NUMBER); // convenient to convert to dimensionless variable needed for checking definite-ness
    if(z_a>1.) {fH2=1.;} else {if(fabs(z_a)<0.1) {fH2=(1.+0.25*z_a*(1.+0.5*z_a))/(y_b + MIN_REAL_NUMBER);} else {fH2=(2./(y_b + MIN_REAL_NUMBER))*(1.-sqrt(1.-z_a))/z_a;}} // checking limits of terms for accuracy

    /* now comes the tricky bit -- need to account for the -molecular- self-shielding [depends on fH2, not just the dust external shielding already accounted for */
    double xb0 = a_GP + a_Z + 2.*b_3B + b_H2HI + xi_cr_H2;
    if(fH2 > 1.e-10 && fH2 < 0.99 && G_LW > 0.1*xb0) // fH2 is non-trivial, and the radiation term is significant, so we need to think about molecular self-shielding
    {
        double fH2_min = fH2; // we have just calculated fH2 with -no- molecular self-shielding, so this number can only go up from here
        // calculate a bundle of variables we will need below, to account for the velocity-gradient Sobolev approximation and slab attenuation of G0 //
        double dx_cell = pp[i].Get_Particle_Size() * All.cf_atime; // cell size
        double surface_density_H2_0 = 5.e14 * PROTONMASS_CGS, x_exp_fac=0.00085, w0=0.2; // characteristic cgs column for -molecular line- self-shielding
        double surface_density_local = xH0 * cell[i].Density * All.cf_a3inv * dx_cell * UNIT_SURFDEN_IN_CGS; // this is -just- the [neutral] depth through the local cell/slab. that's closer to what we want here, since G0 is -already- attenuated in the pre-processing step!
        double v_thermal_rms = 0.111*sqrt(T); // sqrt(3*kB*T/2*mp), since want rms thermal speed of -molecular H2- in kms
        double gradv=cell[i].velocity_gradient_norm();
        double dv_turb=gradv*dx_cell*UNIT_VEL_IN_KMS; // delta-velocity across cell
        double x00 = surface_density_local / surface_density_H2_0, x01 = x00 / (sqrt(1. + 3.*dv_turb*dv_turb/(v_thermal_rms*v_thermal_rms)) * sqrt(2.)*v_thermal_rms), y_ss, x_ss_1, x_ss_sqrt, fH2_tmp, fH2_max, Qmax, Qmin; // variable needed below. note the x01 term corrects following Gnedin+Draine 2014 for the velocity gradient at the sonic scale, assuming a Burgers-type spectrum [their Eq. 3]

        fH2_tmp = 1.; // now consider the maximally shielded case, if you had fmol = 1 in the shielding terms
        x_ss_1=1.+fH2_tmp*x01; x_ss_sqrt=sqrt(1.+fH2_tmp*x00); y_ss=(1.-w0)/(x_ss_1*x_ss_1) + w0/x_ss_sqrt*exp(-DMIN(EXPmax,x_exp_fac*x_ss_sqrt)); x_b=xb0+y_ss*G_LW; y_b=x_b/(x_c + MIN_REAL_NUMBER); // recalculate all terms that depend on the shielding
        z_a=4.*y_a/(y_b*y_b + MIN_REAL_NUMBER); if(z_a>1.) {fH2=1.;} else {if(fabs(z_a)<0.1) {fH2=(1.+0.25*z_a*(1.+0.5*z_a))/(y_b + MIN_REAL_NUMBER);} else {fH2=(2./(y_b + MIN_REAL_NUMBER))*(1.-sqrt(1.-z_a))/z_a;}} // calculate f assuming the shielding term is constant
        fH2_max = DMAX(0,DMIN(1,fH2)); // this serves as an upper-limit for f

        if(fH2_max > 1.1*fH2_min)
        {
            fH2_tmp = fH2_max; // re-calculate the maximally-shielded case
            x_ss_1=1.+fH2_tmp*x01; x_ss_sqrt=sqrt(1.+fH2_tmp*x00); y_ss=(1.-w0)/(x_ss_1*x_ss_1) + w0/x_ss_sqrt*exp(-DMIN(EXPmax,x_exp_fac*x_ss_sqrt)); x_b=xb0+y_ss*G_LW; y_b=x_b/(x_c + MIN_REAL_NUMBER); // recalculate all terms that depend on the shielding
            z_a=4.*y_a/(y_b*y_b + MIN_REAL_NUMBER); if(z_a>1.) {fH2=1.;} else {if(fabs(z_a)<0.1) {fH2=(1.+0.25*z_a*(1.+0.5*z_a))/(y_b + MIN_REAL_NUMBER);} else {fH2=(2./(y_b + MIN_REAL_NUMBER))*(1.-sqrt(1.-z_a))/z_a;}} // calculate f assuming the shielding term is constant
            fH2_tmp=fH2; x_ss_1=1.+fH2_tmp*x01; x_ss_sqrt=sqrt(1.+fH2_tmp*x00); y_ss=(1.-w0)/(x_ss_1*x_ss_1) + w0/x_ss_sqrt*exp(-DMIN(EXPmax,x_exp_fac*x_ss_sqrt)); x_b=xb0+y_ss*G_LW; y_b=x_b/(x_c + MIN_REAL_NUMBER); // calculate all the terms we need to solve for the zeros of this function
            fH2_max = fH2; Qmax = 1 + y_a*fH2_tmp*fH2_tmp - y_b*fH2_tmp; // set the new max fH2, from this, and set the corresponding value of the function we are trying to root-find for

            fH2_tmp = fH2_min; // re-calculate the minimally-shielded case
            x_ss_1=1.+fH2_tmp*x01; x_ss_sqrt=sqrt(1.+fH2_tmp*x00); y_ss=(1.-w0)/(x_ss_1*x_ss_1) + w0/x_ss_sqrt*exp(-DMIN(EXPmax,x_exp_fac*x_ss_sqrt)); x_b=xb0+y_ss*G_LW; y_b=x_b/(x_c + MIN_REAL_NUMBER); // recalculate all terms that depend on the shielding
            z_a=4.*y_a/(y_b*y_b + MIN_REAL_NUMBER); if(z_a>1.) {fH2=1.;} else {if(fabs(z_a)<0.1) {fH2=(1.+0.25*z_a*(1.+0.5*z_a))/(y_b + MIN_REAL_NUMBER);} else {fH2=(2./(y_b + MIN_REAL_NUMBER))*(1.-sqrt(1.-z_a))/z_a;}} // calculate f assuming the shielding term is constant
            fH2_tmp=fH2; x_ss_1=1.+fH2_tmp*x01; x_ss_sqrt=sqrt(1.+fH2_tmp*x00); y_ss=(1.-w0)/(x_ss_1*x_ss_1) + w0/x_ss_sqrt*exp(-DMIN(EXPmax,x_exp_fac*x_ss_sqrt)); x_b=xb0+y_ss*G_LW; y_b=x_b/(x_c + MIN_REAL_NUMBER); // calculate all the terms we need to solve for the zeros of this function
            fH2_min = fH2; Qmin = 1 + y_a*fH2_tmp*fH2_tmp - y_b*fH2_tmp; // set the new min fH2, from this, and set the corresponding value of the function we are trying to root-find for

            fH2 = exp( (log(fH2_min)*Qmax - log(fH2_max)*Qmin) / (Qmax-Qmin) ); // do a Newton-Raphson step in log[f_H2] space now that we have good initial brackets
            if((fH2_max > 1.5*fH2_min) && (Qmax*Qmin < 0) && (fH2_max > 1.1*fH2)) // have a big enough dynamic range, and bracketing Qmin/max, to make further iteration meaningful
            {
                double f_p=fH2_min, Q_p=Qmin, Q, fH2_new; int iter=0; // define variables for iteration below
                while(1)
                {
                    x_ss_1=1.+fH2*x01; x_ss_sqrt=sqrt(1.+fH2*x00); y_ss=(1.-w0)/(x_ss_1*x_ss_1) + w0/x_ss_sqrt*exp(-DMIN(EXPmax,x_exp_fac*x_ss_sqrt)); x_b=xb0+y_ss*G_LW; y_b=x_b/(x_c + MIN_REAL_NUMBER); // calculate all the terms we need to solve for the zeros of this function
                    Q = 1 + y_a*fH2*fH2 - y_b*fH2; // update the value of the function we are trying to zero
                    if(iter==0) {if(Q*Q_p>=0) {f_p=fH2_max; Q_p=Qmax;}} // check in case we attempted to bracket from the 'wrong side'
                    if(Q*Q_p >= 0) {break;} // no longer bracketing, end while loop
                    fH2_new = exp( (log(f_p)*Q - log(fH2)*Q_p) / (Q-Q_p) ); f_p=fH2; fH2=fH2_new; Q_p=Q; // update guess and previous values //
                    iter++; // count iterations
                    if(fabs(fH2-f_p) < 0.1*0.5*(f_p+fH2)) {break;} // converged well enough for our purposes!
                    if((y_ss > 0.85) || (y_ss*G_LW < xb0)) {break;} // negligible shielding, or converged to point where external LW is not dominant dissociator so no further iteration needed
                    if((fH2 > 0.95*fH2_max) || (fH2 > 0.99) || (fH2 < 1.e-10) || (fH2 < 1.05*fH2_min) || (iter > 10)) {break;} // approached physical limits or bounds of validity, or end of iteration cycle
                } // end of convergence iteration to find solution for fmol
            } // opening condition for iteration requiring large enough dynamic range, valid bracketing
        } // opening condition for even checking iteration with fmax > 1.5*fmin
    } // opening condition for considering any molecular self-shielding terms at all
    if(!isfinite(fH2)) {fH2=0;} else {if(fH2>1) {fH2=1;} else if(fH2<0) {fH2=0;}} // check vs nans, valid values
    return xH0 * fH2; // return answer
#endif


#if (COOL_MOLECFRAC == 4)
    /* use the simpler Kumholz, McKee, & Tumlinson 2009 sub-grid model for molecular fractions in equilibrium, which is a function modeling spherical clouds
        of internally uniform properties exposed to incident radiation. Depends on column density, metallicity, and incident FUV field. */
    /* get estimate of mass column density integrated away from this location for self-shielding */
    double surface_density_Msun_pc2_infty = 0.05 * evaluate_NH_from_GradRho(pp[i].GradRho,pp[i].KernelRadius,cell[i].Density,pp[i].NumNgb,1,i,pp) * UNIT_SURFDEN_IN_CGS / 0.000208854; // approximate column density with Sobolev or Treecol methods as appropriate; converts to M_solar/pc^2
    /* 0.05 above is in testing, based on calculations by Laura Keating: represents a plausible re-scaling of the shielding length for sub-grid clumping */
    double surface_density_Msun_pc2_local = cell[i].Density * pp[i].Get_Particle_Size() * All.cf_a2inv * UNIT_SURFDEN_IN_CGS / 0.000208854; // this is -just- the depth through the local cell/slab. that's closer to what we want here, since G0 is -already- attenuated in the pre-processing step!
    double surface_density_Msun_pc2 = DMIN( surface_density_Msun_pc2_local, surface_density_Msun_pc2_infty);
    //double surface_density_Msun_pc2 = surface_density_Msun_pc2_local;
    /* now actually do the relevant calculation with the KMT fitting functions */
    double clumping_factor_for_unresolved_densities = 1; // Gnedin et al. add a large clumping factor to account for inability to resolve high-densities, here go with what is resolved
    double chi = 0.766 * (1. + 3.1*pow(Z_Zsol, 0.365)); // KMT estimate of chi, useful if we do -not- know anything about actual radiation field
    if(urad_G0 >= 0) {chi = 71. * urad_G0 / (clumping_factor_for_unresolved_densities * nH_cgs);} // their actual fiducial value including radiation information
    double psi = chi * (1.+0.4*chi)/(1.+1.08731*chi); // slightly-transformed chi variable
    double s = (Z_Zsol + 1.e-3) * surface_density_Msun_pc2 / (MIN_REAL_NUMBER + psi); // key variable controlling shielding in the KMT approximaton
    double q = s * (125. + s) / (11. * (96. + s)); // convert to more useful form from their Eq. 37
    double fH2 = 1. - pow(1.+q*q*q , -1./3.); // full KMT expression [unlike log-approximation, this extrapolates physically at low-q]
    if(q<0.2) {fH2 = q*q*q * (1. - 2.*q*q*q/3.)/3.;} // catch low-q limit more accurately [prevent roundoff error problems]
    if(q>10.) {fH2 = 1. - 1./q;} // catch high-q limit more accurately [prevent roundoff error problems]
    fH2 = DMIN(1,DMAX(0, fH2)); // multiple by neutral fraction, as this is ultimately the fraction of the -neutral- gas in H2
    return xH0 * fH2;
#endif


#if (COOL_MOLECFRAC == 3)
    /* use the sub-grid final expression calibrated to ~60pc resolution simulations with equilibrium molecular chemistry and post-processing radiative
        transfer from Gnedin & Draine 2014 (Eqs. 5-7) */
    double S_slab = pp[i].Get_Particle_Size() * All.cf_atime * UNIT_LENGTH_IN_PC / 100.; // slab size in units of 100 pc
    double D_star = 0.17 * (2. + S_slab*S_slab*S_slab*S_slab*S_slab) / (1. + S_slab*S_slab*S_slab*S_slab*S_slab); // intermediate variable
    double U_star = 9. * D_star / S_slab, n_star = 14. * sqrt(D_star) / S_slab; // intermediate variables
    double g_eff = sqrt(D_star*D_star + Z_Zsol*Z_Zsol); // intermediate variable parameterizing the dust-to-gas ratio here [assuming the dust-to-gas ratio relative to solar scales linearly with metallicity, giving Z_Zsol = their D_MW parameter]
    double Lambda_incident = log(1. + pow(0.05/g_eff + urad_G0, 2./3.) * pow(g_eff, 1./3.) / U_star); // intermediate variable parameterizing the incident radiation, takes input UV radiation field relative to MW
    double nHalf = n_star * Lambda_incident / g_eff; // intermediate variable
    double w_x = 0.8 + sqrt(Lambda_incident) / pow(S_slab, 1./3.); // intermediate variable
    double x_f = w_x * log(nH_cgs / nHalf); // intermediate variable
    double fH2_gd = 1./(1. + exp(-x_f*(1.-0.02*x_f+0.001*x_f*x_f)));
    return xH0 * fH2_gd;
#endif


/* Subordinate the SINGLE_STAR_SINK_FORMATION-bit-256 fallback to any other
 * already-selected MOLECFRAC method. The original gate `(SINGLE_STAR_SINK_FORMATION & 256) ||
 * (COOL_MOLECFRAC == 2)` triggered simultaneously with the COOL_MOLECFRAC==5
 * block above for any Config carrying GALSF_FB_FIRE_STELLAREVOLUTION>2 +
 * COOLING + GALSF_SFR_CRITERION & 256 (e.g. an isodisk_mechfb run with the
 * `1+256` SF criterion plus stellar evolution). precompiler_logic.h:356
 * mirrors GALSF_SFR_CRITERION onto SINGLE_STAR_SINK_FORMATION, and eos_functions.h:75-77
 * implicitly sets COOL_MOLECFRAC=5 under that combo, so both branches got
 * compiled and `clumping_factor` was declared twice in the same function
 * scope -> redefinition error. The intent is mutual exclusion: this KG10
 * fallback only when no other explicit method is selected. */
#if (COOL_MOLECFRAC == 2) || ((SINGLE_STAR_SINK_FORMATION & 256) && !((COOL_MOLECFRAC == 5) || (COOL_MOLECFRAC == 4) || (COOL_MOLECFRAC == 3))) /* estimate f_H2 with Krumholz & Gnedin 2010 fitting function, assuming simple scalings of radiation field, clumping, and other factors with basic gas properties so function only of surface density and metallicity, truncated at low values (or else it gives non-sensical answers) */
    double clumping_factor=1, fH2_kg=0, tau_fmol = (0.1 + pp[i].Metallicity[0]/All.SolarAbundances[0]) * evaluate_NH_from_GradRho(pp[i].GradRho,pp[i].KernelRadius,cell[i].Density,pp[i].NumNgb,1,i,pp) * 434.78 * UNIT_SURFDEN_IN_CGS; // convert units for surface density. also limit to Z>=0.1, where their fits were actually good, or else get unphysically low molecular fractions
    if(tau_fmol>0) {double y = 0.756 * (1 + 3.1*pow(pp[i].Metallicity[0]/All.SolarAbundances[0],0.365)) / clumping_factor; // this assumes all the equilibrium scalings of radiation field, density, SFR, etc, to get a trivial expression
        y = log(1 + 0.6*y + 0.01*y*y) / (0.6*tau_fmol); y = 1 - 0.75*y/(1 + 0.25*y); fH2_kg=DMIN(1,DMAX(0,y));}
    return fH2_kg * neutral_fraction;
#endif


#if defined(COOLING) || (COOL_MOLECFRAC == 1) /* if none of the above is set, default to a wildly-oversimplified scaling set by fits to the temperature below which gas at a given density becomes molecular from cloud simulations in Glover+Clark 2012 */
    double T_mol = DMAX(1.,DMIN(8000., cell[i].Density*All.cf_a3inv*UNIT_DENSITY_IN_NHCGS));
    return neutral_fraction / (1. + temperature*temperature/(T_mol*T_mol));
#endif

    return 0; // catch //
}


/* return helium -number- fraction, not mass fraction */
KOKKOS_INLINE_FUNCTION double yhelium(int target, struct particle_data *pp)
{
#ifdef COOL_METAL_LINES_BY_SPECIES
    if(target >= 0) {double ytmp=DMIN(0.5,pp[target].Metallicity[1]); return 0.25*ytmp/(1.-ytmp);} else {return ((1.-HYDROGEN_MASSFRAC)/(4.*HYDROGEN_MASSFRAC));}
#else
    return ((1.-HYDROGEN_MASSFRAC)/(4.*HYDROGEN_MASSFRAC)); // assume uniform H-He gas
#endif
}


/* return mean molecular weight, appropriate for the approximations of the user-selected chemical network[s] */
KOKKOS_INLINE_FUNCTION double Get_Gas_Mean_Molecular_Weight_mu(double T_guess, double rho, double *xH0, double *ne_guess, double urad_from_uvb_in_G0, int target, struct particle_data *pp, struct gas_cell_data *cell)
{
#if defined(CHIMES)
    if(target >= 0) { return calculate_mean_molecular_weight(&(ChimesGasVars[target]), &ChimesGlobalVars); } else { return 0.59; }
#elif defined(COOLING)
    double X=HYDROGEN_MASSFRAC, Y=1.-X, Z=0, fmol=0;
#ifdef METALS
    if(target >= 0)
    {
        Z = DMIN(0.25,pp[target].Metallicity[0]); if(NUM_METAL_SPECIES>=10) {Y = DMIN(0.35,pp[target].Metallicity[1]);}
        X = 1. - (Y+Z);
    }
#endif
    if(target >= 0) { fmol = Get_Gas_Molecular_Mass_Fraction(target, T_guess, *xH0, *ne_guess, urad_from_uvb_in_G0, pp, cell); }
    return 1. / ( X*(1-0.5*fmol) + Y/4. + *ne_guess*HYDROGEN_MASSFRAC + Z/(16.+12.*fmol) ); // since our ne is defined in some routines with He, should multiply by universal
#else
    return 4./(3.+5.*HYDROGEN_MASSFRAC); // fully-ionized H-He plasma
#endif
}


/* ========================================================================
 * return_dust_to_metals_ratio_vs_solar
 * (formerly in dust_to_metals_functions.h, consolidated here)
 * ======================================================================== */

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


/* ========================================================================
 * Get_Gas_Ionized_Fraction — ionized fraction of gas
 * (moved from eos/eos.cc)
 * ======================================================================== */

KOKKOS_INLINE_FUNCTION
double Get_Gas_Ionized_Fraction(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
#ifdef COOLING
#ifdef CHIMES
  return (double) ChimesGasVars[i].abundances[ChimesGlobalVars.speciesIndices[sp_HII]];
#else
    double ne = cell[i].Ne, nh0 = cell[i].HI; // saved by the cooling solver; no need to re-solve the chemistry to answer this
    double f_ion = DMIN(DMAX(DMAX(1-nh0, ne/1.2), 1.e-8), 1.); // account for different measures above (assuming primordial composition)
    if((!isfinite(f_ion)) || (f_ion<0)) {f_ion=0;}
    return f_ion;
#endif
#endif
    return 1;
}


#ifdef HYDRO_GENERATE_TARGET_MESH
/* ==========================================================================
 * USER EDIT POINT for HYDRO_GENERATE_TARGET_MESH.  EDIT THESE TWO BODIES.
 *
 * This pair of functions gives the 'target' density and pressure the mesh is
 * driven towards, as a function of particle properties (most commonly
 * position).  Use it to build ICs: the code moves mesh and mass towards the
 * profile you return here.
 *
 * They live in this header rather than in eos.cc because set_eos_pressure_impl
 * calls them from the device pass; eos.cc still emits the host symbols through
 * its usual #undef KOKKOS_INLINE_FUNCTION block, so host callers are unchanged.
 * Write ordinary arithmetic on pp[i] / cell[i] and All.*; anything calling a
 * host-only library here would put that library on the device path.
 * ========================================================================== */
KOKKOS_INLINE_FUNCTION double return_user_desired_target_density(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    return 1; // uniform density everywhere -- will try to generate a glass //
    /*
     // this example would initialize a constant-density (density=rho_0) spherical cloud (radius=r_cloud) with a smooth density 'edge' (width=interp_width) surrounded by an ambient medium of density =rho_0/rho_contrast //
     double dx=pp[i].Pos[0]-boxHalf_X, dy=pp[i].Pos[1]-boxHalf_Y, dz=pp[i].Pos[2]-boxHalf_Z, r=sqrt(dx*dx+dy*dy+dz*dz);
     double rho_0=1, r_cloud=0.5*boxHalf_X, interp_width=0.1*r_cloud, rho_contrast=10.;
     return rho_0 * ((1.-1./rho_contrast)*0.5*erfc(2.*(r-r_cloud)/interp_width) + 1./rho_contrast);
     */
}
KOKKOS_INLINE_FUNCTION double return_user_desired_target_pressure(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    return 1; // uniform pressure everywhere -- will try to generate a constant-pressure medium //
    /*
     // this example would initialize a radial pressure gradient corresponding to a self-gravitating, spherically-symmetric, infinite power-law
     //   density profile rho ~ r^(-b) -- note to do this right, you need to actually set that power-law for density, too, in 'return_user_desired_target_density' above
     double dx=pp[i].Pos[0]-boxHalf_X, dy=pp[i].Pos[1]-boxHalf_Y, dz=pp[i].Pos[2]-boxHalf_Z, r=sqrt(dx*dx+dy*dy+dz*dz);
     double b = 2.; return 2.*M_PI/fabs((3.-b)*(1.-b)) * pow(return_user_desired_target_density(i, pp, cell),2) * r*r;
     */
}
#endif


#if defined(COOLING) && !defined(CHIMES)
/* ================================================================
   ThermalProperties — get temperature and ionization state from u
   ================================================================ */
/* Reads the thermochemical state the cooling solver saved on the cell. It does not
   solve anything, reach a table, or write to the cell: the values it hands back are
   the ones the last cooling update determined for this gas. A source with no cell of
   its own (target < 0) has no such state, so the call reports nothing and returns 0 --
   build the estimate you need locally instead. Helium is deliberately absent: nothing
   outside the cooling solver consumed it, and it is not saved. */
KOKKOS_INLINE_FUNCTION
double ThermalProperties(double u, double rho, int target, double *mu_guess, double *ne_guess, double *nH0_guess, double *nHp_guess, struct particle_data *pp, struct gas_cell_data *cell)
{
    if(target < 0) {*ne_guess = 0; *nH0_guess = 0; *nHp_guess = 0; *mu_guess = 1; return 0;}
#ifdef HYDRO_MULTIFLUID_DM
    /* dark-fluid short-circuit: trivial adiabat T(u), no chemistry call-stack. */
    if(pp[target].FluidType == FLUID_DM) {
        *ne_guess = 0; *nH0_guess = 1; *nHp_guess = 0; *mu_guess = 1.0;
        return (GAMMA_DEFAULT - 1.0) * (*mu_guess) * (PROTONMASS_CGS / BOLTZMANN_CGS)
               * u * (UNIT_ENERGY_IN_CGS / UNIT_MASS_IN_CGS);
    }
#endif
#if defined(CHIMES)
    /* Unreachable: the enclosing guard is !defined(CHIMES), so under CHIMES this
       function does not exist at all. Kept because it records what the CHIMES
       state would be read from, and it is inert either way -- CHIMES needs
       SUNDIALS, which does not build for the device. */
    int i = target; *ne_guess = ChimesGasVars[i].abundances[ChimesGlobalVars.speciesIndices[sp_elec]]; *nH0_guess = ChimesGasVars[i].abundances[ChimesGlobalVars.speciesIndices[sp_HI]];
    *nHp_guess = ChimesGasVars[i].abundances[ChimesGlobalVars.speciesIndices[sp_HII]];
    double temp = ChimesGasVars[target].temperature;
#else
    *ne_guess = cell[target].Ne; *nH0_guess = cell[target].HI; *nHp_guess = DMAX(0, 1. - *nH0_guess);
    double temp = cell[target].gas_temperature_from_u(u);
#if (GALSF_FB_FIRE_STELLAREVOLUTION <= 2) && defined(GALSF_FB_FIRE_RT_HIIHEATING) && !defined(CHIMES_HII_REGIONS)
    /* gas inside an HII region is held ionized by the local source, which the saved state
       only picks up once the cell next cools */
    if(cell[target].DelayTimeHII > 0) {*ne_guess = 1.0 + 2.0*yhelium(target, pp); *nH0_guess = 0; *nHp_guess = 1;}
#endif
#endif
    *mu_guess = cell[target].MeanMolecularWeight;
    return temp;
}
#endif /* COOLING && !CHIMES */


/* ==========================================================================
 * set_eos_pressure_impl — compute pressure and soundspeed from EOS.
 *
 * Migrated from eos/eos.cc.  This is the device-callable inline body.  The
 * public host symbol `set_eos_pressure` in eos.cc is now a one-line wrapper
 * that calls this function.
 *
 * Calls ThermalProperties, which is defined ABOVE in this header.  It used to live
 * in cooling_functions.h and be reached by external linkage; that worked only for
 * callers inside cooling.cc, because a device TU cannot resolve an external device
 * call.  It reads saved cell state and needs nothing from the cooling solver, so it
 * moved here beside yhelium and this body.
 * ⚠ This header still must NOT include cooling_functions.h: that would emit non-inline
 * strong symbols for convert_u_to_temp / find_abundances_and_rates competing with
 * cooling.cc's, which reads a different per-TU All mirror and gives wrong results on
 * CUDA only (confirmed by bisection).
 *
 * Every branch runs wherever this body runs. The tabulated equations of state
 * used to be the exception: EOS_HELMHOLTZ and EOS_ANEOS reached table-lookup
 * routines that only existed on the host, so those two branches were fenced out
 * of the device pass and a compile-time gate stopped the device kernel calling
 * this function at all in such builds. Both tables now live in memory the
 * device can read and both evaluation chains are header bodies, so the fence,
 * the gate and the split have gone -- there is no configuration in which this
 * function computes less than the whole equation of state.
 *
 * The tables are passed in, never fetched here. Their owners are host globals,
 * and a device body that reached for one would read host memory and say
 * nothing; the caller builds an EosTableView on the host and hands it over.
 * ========================================================================== */

/* set_eos_pressure_impl below calls Get_Gas_CosmicRayPressure under
 * COSMIC_RAY_FLUID. Include cosmic_ray_functions.h here (AFTER
 * Get_Gas_Ionized_Fraction above, which cosmic_ray_functions.h calls) so the
 * inline body is visible to any TU pulling this header in (cooling.cc etc.)
 * without requiring each caller to pre-include cosmic_ray_functions.h.
 *
 * MUST be gated on COSMIC_RAY_FLUID to mirror eos.cc's own pre-include
 * (eos/eos.cc:77-79). The three functions Get_CosmicRayEnergyDensity_cgs,
 * Get_CosmicRayIonizationRate_cgs, CR_gas_heating live OUTSIDE the
 * COSMIC_RAY_FLUID #ifdef in cosmic_ray_functions.h; including unconditionally
 * here would cause eos.cc's non-inline re-include of eos_functions.h to emit
 * strong symbols for those three, dup'ing cosmic_ray_utilities.o. */
#ifdef COSMIC_RAY_FLUID
#include "cosmic_ray_fluid/cosmic_ray_functions.h"
#endif

/* HYDRO_MULTIFLUID_DM: dark-fluid early-return into set_dark_eos_pressure.
   set_dark_eos_pressure is KOKKOS_INLINE_FUNCTION in sidm/dm_fluid_functions.h
   (pure arithmetic on macro constants + device-callable cell_data accessors),
   so we dispatch into it unconditionally under HYDRO_MULTIFLUID_DM. */
#ifdef HYDRO_MULTIFLUID_DM
#include "../declarations/multifluid_helpers.h"
#include "../sidm/dm_fluid_functions.h"
#endif

/* Solid-EOS dispatch: eos_branch_of keys the switch below on CompositionType and
   is pure arithmetic, so it is GPU-marked and the switch runs on either pass.
   The functions here are `static inline`, i.e. internal linkage, so eos.cc's
   non-inline re-include of this header cannot duplicate a strong symbol. */
#if defined(EOS_TILLOTSON) || defined(EOS_ANEOS)
#include "composition_registry.h"
#endif

/* The tabulated equations of state. Their evaluation routines are header
   bodies so that a device kernel can call them -- without relocatable device
   code, a call that crosses a translation unit has nothing to resolve to. Each
   include is gated exactly as its call site inside set_eos_pressure_impl is, so
   a build that does not use one never parses it. */
#ifdef EOS_ANEOS
#include "aneos.h"
#endif
#ifdef EOS_HELMHOLTZ
#include "helmholtz/helmholtz.h"
#endif

/* The tables a call to set_eos_pressure_impl may need, as data.

   The fields are gated once, here, and the struct itself always exists, so the
   routines that take it keep one signature in every configuration. It is built
   on the host immediately before a dispatch and captured by value; a device
   body must never fetch it, because the owners are host globals and reading one
   from a kernel gives the wrong memory with nothing to say so -- which is
   exactly the defect the blocker survey caught when an earlier cut of this rung
   called the accessor from inside the body instead of passing the result in. */
struct EosTableView
{
#ifdef EOS_HELMHOLTZ
    const HelmTable *helm;
#endif
#ifdef EOS_ANEOS
    const struct aneos_table *aneos;
#endif
};

/* Fill a view from the live tables.

   ⚠ HOST ONLY, AND THAT IS A CONTRACT THE COMPILER CANNOT ENFORCE. It names the
   owning globals, so calling it from a device body reads host memory and says
   nothing -- the same silent-wrong-answer class the table pointers exist to
   prevent. It is deliberately NOT device-annotated, but an unmarked function is
   callable from an unmarked caller, so the guarantee rests on nothing being
   allowed to call it from inside a KOKKOS_INLINE_FUNCTION body or a
   KOKKOS_LAMBDA. The rule is easy to state and easy to check: the two
   device-callable bodies that take an EosTableView -- set_eos_pressure_impl
   and drift_particle_impl -- must not mention this function or either table
   owner anywhere in their bodies.

   It is also the single place a view is assembled -- call sites take the
   result, they never build their own, because a hand-built view at four sites
   is four places for a gated field to be forgotten. */
inline struct EosTableView eos_tables_view(void)
{
    struct EosTableView view = {};
#ifdef EOS_HELMHOLTZ
    view.helm = helm_table_view();
#endif
#ifdef EOS_ANEOS
    view.aneos = ANEOS_Tables;
#endif
    return view;
}

#ifdef EOS_HELMHOLTZ
/* The Helmholtz router: unit conversion, range validation with 0th-order
   clamping, the Newton inversion, and the ideal-gas fallback for a solve that
   fails even after clamping. Moved here verbatim from eos/eos_interface.cc,
   which now owns only the table's lifetime. Every routine takes the table
   explicitly: a body that reached for the file-scope instance would read host
   memory from a kernel, silently. */

#define BITMASK_SET_FLAG(BITMASK,FLAG)      (BITMASK) |= (FLAG)
#define BITMASK_SET_ALL_FLAGS(BITMASK)      (BITMASK = ~(0))
#define BITMASK_UNSET_FLAG(BITMASK,FLAG)    (BITMASK) &= ~(FLAG)
#define BITMASK_UNSET_ALL_FLAGS(BITMASK)    (BITMASK) = 0
#define BITMASK_CHECK_FLAG(BITMASK,FLAG)    (((BITMASK) & (FLAG)) == (FLAG))

#define EOS_ERR_VALID         0
#define EOS_ERR_RHO_LT_RHOMIN 1
#define EOS_ERR_RHO_GT_RHOMAX 2
#define EOS_ERR_EPS_LT_EPSMIN 4
#define EOS_ERR_EPS_GT_EPSMAX 8
#define EOS_ERR_COMPOSITION   16

static KOKKOS_INLINE_FUNCTION int eos_input_to_cgs(struct eos_input * vars)
{
  vars->rho *= UNIT_DENSITY_IN_CGS;
  vars->eps *= UNIT_SPECEGY_IN_CGS;
  return 0;
}

static KOKKOS_INLINE_FUNCTION int eos_output_from_cgs(struct eos_output * vars)
{
  vars->press  /= UNIT_PRESSURE_IN_CGS;
  vars->csound /= UNIT_VEL_IN_CGS;
  return 0;
}

static KOKKOS_INLINE_FUNCTION int eos_validate(const HelmTable *tab, struct eos_input const * vars, struct eos_input * vars_adj, int * bitmask)
{
  *bitmask = EOS_ERR_VALID;
  memcpy(vars_adj, vars, sizeof(*vars));

#ifdef EOS_HELMHOLTZ
  if(vars->Ye < 0)
  {
    BITMASK_SET_FLAG(*bitmask, EOS_ERR_COMPOSITION);
    vars_adj->Ye = 0;
  }
  if(vars->Ye > 1)
  {
    BITMASK_SET_FLAG(*bitmask, EOS_ERR_COMPOSITION);
    vars_adj->Ye = 1;
  }
  if(vars->Abar < 1)
  {
    BITMASK_SET_FLAG(*bitmask, EOS_ERR_COMPOSITION);
    vars_adj->Abar = 1;
  }

  double rho_ye_min, rho_ye_max;
  helm_get_rhoye_range(tab, &rho_ye_min, &rho_ye_max);
  if(vars->rho * vars_adj->Ye < rho_ye_min)
  {
    BITMASK_SET_FLAG(*bitmask, EOS_ERR_RHO_LT_RHOMIN);
    vars_adj->rho = rho_ye_min / vars_adj->Ye;
  }
  if(vars->rho * vars_adj->Ye > rho_ye_max)
  {
    BITMASK_SET_FLAG(*bitmask, EOS_ERR_RHO_GT_RHOMAX);
    vars_adj->rho = rho_ye_max / vars_adj->Ye;
  }

  /* check energy range by evaluating at table T boundaries */
  HelmInput hin;
  HelmResult hout;
  hin.rho  = vars_adj->rho;
  hin.abar = vars_adj->Abar;
  hin.ye   = vars_adj->Ye;

  double tmin, tmax;
  helm_get_temp_range(tab, &tmin, &tmax);

  hin.temp = tmin;
  if (helm_eos_from_temp(tab, &hin, &hout)) return 1;
  double eps_min = hout.etot;

  hin.temp = tmax;
  if (helm_eos_from_temp(tab, &hin, &hout)) return 1;
  double eps_max = hout.etot;

  if(vars->eps < eps_min)
  {
    BITMASK_SET_FLAG(*bitmask, EOS_ERR_EPS_LT_EPSMIN);
    vars_adj->eps = eps_min;
  }
  if(vars->eps > eps_max)
  {
    BITMASK_SET_FLAG(*bitmask, EOS_ERR_EPS_GT_EPSMAX);
    vars_adj->eps = eps_max;
  }
#endif

  return 0;
}

static KOKKOS_INLINE_FUNCTION int eos_compute_from_valid(const HelmTable *tab, struct eos_input const * in, struct eos_output * out)
{
#ifdef EOS_HELMHOLTZ
  /* use Newton iteration to invert for T given (rho, eps, abar, ye) */
  double temp_guess = in->temp;
  double tmin, tmax;
  helm_get_temp_range(tab, &tmin, &tmax);
  if (temp_guess < tmin || temp_guess > tmax) {
    /* initial guess from gamma=5/3 electron gas */
    temp_guess = (2.0/3.0) * in->Abar * in->eps * helm_constants::me / helm_constants::kerg;
  }

  HelmResult hout;
  int ierr = helm_eos_from_energy(tab, in->rho, in->eps,
                                  in->Abar, in->Ye, temp_guess, &hout);
  if (ierr) {
    printf("%s:%d unexpected EOS failure!\n", __FILE__, __LINE__);
    return 1;
  }

  out->press  = hout.ptot;
  out->csound = hout.csound;
  out->temp   = hout.temp;
#ifdef EOS_PROVIDES_ENTROPY
  out->entropy = hout.stot;
#endif
#ifdef EOS_PROVIDES_CV
  out->cv = hout.cv;
#endif
#endif

  return 0;
}

KOKKOS_INLINE_FUNCTION
int eos_compute_P(const HelmTable *tab, struct eos_input const * in_, struct eos_output * out_)
{
  struct eos_input in, in_adj;
  memcpy(&in, in_, sizeof(in));
#ifdef EOS_USES_CGS
  eos_input_to_cgs(&in);
#endif

  int bitmask = 0;
  int ierr = eos_validate(tab, &in, &in_adj, &bitmask);
  if(ierr) {endrun(920903);}
  if(bitmask != EOS_ERR_VALID)
  {
    /* One call carrying the same flags, rather than the chain of partial writes
       this used to assemble. The body is compiled for the device now, where
       stderr does not exist; and even on the host, concurrent work items
       interleaved the pieces into unreadable lines. The condition and the
       information are unchanged. */
    printf("EOS ERROR:%s%s%s%s%s\n",
           BITMASK_CHECK_FLAG(bitmask, EOS_ERR_COMPOSITION)   ? "/invalid composition" : "",
           BITMASK_CHECK_FLAG(bitmask, EOS_ERR_RHO_LT_RHOMIN) ? "/density too low"     : "",
           BITMASK_CHECK_FLAG(bitmask, EOS_ERR_RHO_GT_RHOMAX) ? "/density too large"   : "",
           BITMASK_CHECK_FLAG(bitmask, EOS_ERR_EPS_LT_EPSMIN) ? "/temperature too low" : "",
           BITMASK_CHECK_FLAG(bitmask, EOS_ERR_EPS_GT_EPSMAX) ? "/temperature too high": "");

#ifdef EOS_USES_CGS
    char const * unit_dens = "g/cm^3";
    char const * unit_ene  = "erg/g";
#else
    char const * unit_dens = "";
    char const * unit_ene  = "";
#endif

    printf("  rho  = %.19e %s\n", in.rho, unit_dens);
    printf("  eps  = %.19e %s\n", in.eps, unit_ene);
#ifdef EOS_CARRIES_YE
    printf("  Ye   = %.19e\n", in.Ye);
#endif
#ifdef EOS_CARRIES_ABAR
    printf("  Abar = %.19e\n", in.Abar);
#endif
    printf("Using 0th order extrapolation\n");
    memcpy(&in, &in_adj, sizeof(in));
  }
  struct eos_output out;
  ierr = eos_compute_from_valid(tab, &in, &out);
  if (ierr) {
    /* EOS evaluation failed even after clamping — use a fallback ideal gas estimate
       rather than crashing. This can happen at extreme conditions during initialization. */
    double gamma = 5.0/3.0;
    out.press = (gamma - 1.0) * in.rho * in.eps;
    out.csound = sqrt(gamma * out.press / in.rho);
    out.temp = in.temp > 0 ? in.temp : 1.0e6;
#ifdef EOS_PROVIDES_ENTROPY
    out.entropy = 0;
#endif
#ifdef EOS_PROVIDES_CV
    out.cv = in.eps / (out.temp + 1.0e-30);
#endif
    ierr = 0;
  }
#ifdef EOS_USES_CGS
  ierr = eos_output_from_cgs(&out);
  if(ierr) {endrun(920904);}
#endif
  memcpy(out_, &out, sizeof(out));

  return 0;
}

#endif /* EOS_HELMHOLTZ */

KOKKOS_INLINE_FUNCTION
void set_eos_pressure_impl(int i, struct particle_data *pp, struct gas_cell_data *cell,
                           const struct EosTableView *eos_tables)
{
#ifdef HYDRO_MULTIFLUID_DM
    if(pp[i].FluidType == FLUID_DM) { set_dark_eos_pressure(i, pp, cell); return; }
#endif
    double soundspeed, press=0, temp=0, mu_meanwt=1, gamma_eos_index = GAMMA_DEFAULT; soundspeed=0;
    if(All.Time > All.TimeBegin) {gamma_eos_index = cell[i].gamma_eos_value();} /* can only safely set this after initial startup, not on first call before proper cooling pass */
    cell[i].Gamma = gamma_eos_index;
    press = (gamma_eos_index-1) * cell[i].InternalEnergyPred * cell[i].density_for_energy();

#ifdef COOLING
#ifdef GIZMO_TRACK_ELECTRON_STATE
    double ne_battery_save = 1.0; /* electron fraction (per H), used to populate n_e_cell below; default = fully ionized for pre-init */
#endif
    {
        /* the composition is seeded at initialization and owned by the cooling solve thereafter, so it is
           read here rather than re-established: the first timestep calls this routine several times, and
           overwriting the cache would discard the solve that has already run */
        double ne=1, nh0=0, nhp=0, rho_fortemp=cell[i].Density*All.cf_a3inv, u0=cell[i].InternalEnergyPred;
        temp = ThermalProperties(u0, rho_fortemp, i, &mu_meanwt, &ne, &nh0, &nhp, pp, cell);
        cell[i].Gamma = cell[i].gamma_eos_value();
#ifdef GIZMO_TRACK_ELECTRON_STATE
        ne_battery_save = ne;
#endif
    }
#else
    cell[i].MeanMolecularWeight = MEAN_MOLECULAR_WEIGHT_DEFAULT; /* no chemistry solved here, so the fallback composition */
    temp = cell[i].gas_temperature_from_u(cell[i].InternalEnergyPred);
#endif
    /* this is the boundary that keeps Temperature fresh: it is the temperature of the energy the
       cell holds right now. anything wanting another energy derives it with gas_temperature_from_u */
    cell[i].Temperature = temp;

#ifdef GIZMO_TRACK_ELECTRON_STATE
    /* electron number density cache; consumed by the Biermann battery (grad n_e)
       and by the 2-T plasma integrator (u_e <-> T_e mapping). */
#ifdef COOLING
    cell[i].n_e_cell = ne_battery_save * cell[i].nHcgs();
#else
    cell[i].n_e_cell = cell[i].nHcgs(); /* no chemistry tracked: assume fully ionized */
#endif
    /* electron-temperature cache. Battery-only (no TWO_TEMPERATURE_PLASMA):
       T_e == T_gas every step. Under TWO_TEMPERATURE_PLASMA: this is the
       one-time seed of u_e_cell, taken from the electron temperature a
       snapshot restart read back, or from LTE (T_e = TwoTemp_InitialTeOverTgas
       * T_gas) on a fresh start, where T_e_cell is still zero; after the first
       cooling step the integrator owns u_e_cell + T_e_cell and we leave them
       alone here. */
#ifdef TWO_TEMPERATURE_PLASMA
    if(!(cell[i].u_e_cell > 0)) {
        const double T_e_init = (cell[i].T_e_cell > 0) ? cell[i].T_e_cell : (temp * All.TwoTemp_InitialTeOverTgas);
        const double rho_phys = cell[i].Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS;
        const double u_e_phys = 1.5 * cell[i].n_e_cell * BOLTZMANN_CGS * T_e_init / rho_phys; /* erg/g */
        cell[i].T_e_cell = T_e_init;
        cell[i].u_e_cell = u_e_phys / UNIT_SPECEGY_IN_CGS;
    }
#else
    cell[i].T_e_cell = temp;
#endif
#endif

#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & (2|4|8))
    /* Reset Tier-2 EMF accumulator each step. Each enabled bit's per-cell
       builder (radiative below, dust further below) ADDS its contribution. */
    cell[i].E_battery_T2_cell = {};
#endif

#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 2)
    /* Tier-2 radiative-ionization battery EMF (Durrive & Langer 2015 / Harrison 1973):
         alpha[k] = (sigma_HI[k] n_HI + sigma_HeI[k] n_HeI + sigma_HeII[k] n_HeII)
                    / (n_e * e * c)
         E_RI = sum_k alpha[k] * F_rad[k]                             [statvolt/cm = G]
       Sum over photoionizing bands; gradient pass + curl in hydro_toplevel turn this
       into dB/dt|_RI. Requires RT_EVOLVE_FLUX (for cell.Rad_Flux) and RT_CHEM_PHOTOION
       (for cell.HI / sigma_HI / etc.); precompiler_logic.h enforces this.

       F_rad units: cell.Rad_Flux[k] is stored as F_phys * V_phys in code units. We use
       vol_inv = Density*cf_a3inv/Mass = 1/V_phys_code, then UNIT_FLUX_IN_CGS to get
       physical cgs flux. (cf. gravtree.cc:351, where vol_inv is computed identically.) */
    {
        Vec3<MyDouble> E_RI = {};
        double n_e_cgs = 0;
#if (MHD_BATTERY_MECHANISMS & 1)
        n_e_cgs = cell[i].n_e_cell;
#else
#ifdef COOLING
        n_e_cgs = ne_battery_save * cell[i].nHcgs();
#else
        n_e_cgs = cell[i].nHcgs();
#endif
#endif
        if((n_e_cgs > 0) && (pp[i].Mass > 0) && (cell[i].Density > 0)) {
            const double nH = cell[i].nHcgs();
            const double n_HI = cell[i].HI * nH;
#ifdef RT_CHEM_PHOTOION_HE
            const double He_per_H = (1.0 - HYDROGEN_MASSFRAC) / (4.0 * HYDROGEN_MASSFRAC);
            const double n_HeI  = cell[i].HeI  * nH * He_per_H;
            const double n_HeII = cell[i].HeII * nH * He_per_H;
#endif
            const double inv_neec = 1.0 / (n_e_cgs * ELECTRONCHARGE_CGS * C_LIGHT_CGS);
            const double vol_inv_code = cell[i].Density * All.cf_a3inv / pp[i].Mass;
            const double rad_flux_to_cgs = vol_inv_code * UNIT_FLUX_IN_CGS;

            for(int k=0; k<N_RT_FREQ_BINS; k++) {
                double sig_x_n = All.rt_ion_sigma_HI[k] * n_HI;
#ifdef RT_CHEM_PHOTOION_HE
                sig_x_n += All.rt_ion_sigma_HeI[k]  * n_HeI;
                sig_x_n += All.rt_ion_sigma_HeII[k] * n_HeII;
#endif
                if(sig_x_n > 0) {
                    const double alpha_k = sig_x_n * inv_neec;
                    for(int kd=0; kd<3; kd++) {
                        E_RI[kd] += alpha_k * cell[i].Rad_Flux[k][kd] * rad_flux_to_cgs;
                    }
                }
            }
        }
        cell[i].E_battery_T2_cell += E_RI;
    }
#endif

#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & (4|8))
    /* Tier-2 dust battery EMF (Soliman, Hopkins & Squire 2025).
       Aggregator in solids/dust_battery_functions.h dispatches on the bitmask:
       bit 8 -> battery_E_dust_explicit (SHS25 Eq. 9 with sh25_alpha_coeffs +
                                          cell.J_dust_cell summed from grains);
       bit 4 -> battery_E_dust_TVA      (SHS25 Eq. 17 mostly-neutral TVA form,
                                          driven by dust-vs-gas radiation-pressure
                                          differential acceleration). */
    cell[i].E_battery_T2_cell += dust_battery_assemble_E_cell(i, cell, pp);
#endif

#ifdef EOS_SUBSTELLAR_ISM
    press = cell[i].density_for_energy() * BOLTZMANN_CGS * temp / UNIT_ENERGY_IN_CGS / (mu_meanwt * PROTONMASS_CGS / UNIT_MASS_IN_CGS);
#endif

#ifdef GALSF_EFFECTIVE_EQS
    if(cell[i].Density*All.cf_a3inv >= All.PhysDensThresh) {press = All.FactorForSofterEQS * press + (1 - All.FactorForSofterEQS)  * (gamma_eos_index-1) * cell[i].Density * All.InitGasU;}
#endif

#ifdef EOS_HELMHOLTZ
    /* Tabulated EOS: a Newton inversion over table lookups, run wherever this
       body runs. It used to be fenced off from the device pass because the
       tables were host-only; they are not any more, so there is nothing left to
       fence and no configuration in which this branch is skipped. */
    struct eos_input eos_in;
    struct eos_output eos_out;
    eos_in.rho  = cell[i].Density;
    eos_in.eps  = cell[i].InternalEnergyPred;
    eos_in.Ye   = cell[i].Ye;
    eos_in.Abar = cell[i].Abar;
    eos_in.temp = cell[i].Temperature;
    /* A null table means eos_init failed. That already asked for a controlled
       stop, but the request only drains at the next phase boundary, so this code
       still runs -- and dereferencing the table would turn a clean shutdown into
       a segfault, on the device with nothing to read afterwards. Nothing here can
       recover the physics; the only job is to reach the drain intact. The view
       itself is checked before it is followed: every caller today passes the
       address of a stack view, but a guard that dereferences the thing it is
       guarding is not a guard. */
    if(!eos_tables || !eos_tables->helm) {endrun(920905);}
    else {
    /* eos_compute_P clamps out-of-range input and falls back to an ideal gas if
       the solve still fails, so it has no failure return; the check is kept
       because that is a property of the routine rather than of this call. */
    int ierr = eos_compute_P(eos_tables->helm, &eos_in, &eos_out);
    if(ierr) {endrun(920901);}
    press      = eos_out.press;
    soundspeed = eos_out.csound;
    cell[i].Temperature = eos_out.temp;
    }
#endif /* EOS_HELMHOLTZ */

#if defined(EOS_TILLOTSON) || defined(EOS_ANEOS)
    /* Per-particle solid-EOS dispatch keyed on CompositionType. With a
       single solid-EOS flag enabled, the switch reduces to one case and
       is bit-identical to pre-17c. With multiple solid-EOS flags enabled
       (post-17d), eos_branch_of() partitions the CompositionType ID
       space; see eos/composition_registry.h. */
    switch(eos_branch_of(cell[i].CompositionType)) {
#ifdef EOS_TILLOTSON
        case EOS_BRANCH_TILLOTSON:
            press = cell[i].calculate_tillotson_eos();
            soundspeed = cell[i].SoundSpeed;
            break;
#endif
#ifdef EOS_ANEOS
        case EOS_BRANCH_ANEOS: {
            /* Same reasoning as the Helmholtz branch above: a null descriptor
               array means the tables never loaded, the stop is already requested,
               and reading through it would replace a clean shutdown with a fault.
               The view is checked before it is followed, for the same reason. */
            if(!eos_tables || !eos_tables->aneos) {endrun(920906); break;}
            int aneos_mat = aneos_subindex(cell[i].CompositionType);
            double aneos_rho_cgs = cell[i].Density * UNIT_DENSITY_IN_CGS;
            double aneos_u_cgs   = cell[i].InternalEnergyPred * UNIT_SPECEGY_IN_CGS;
            double aneos_T_guess = cell[i].Temperature;
            double aneos_P, aneos_cs, aneos_S, aneos_cv, aneos_grun;
            int aneos_phase;
            int aneos_ierr = aneos_compute_P(eos_tables->aneos, aneos_mat, aneos_rho_cgs, aneos_u_cgs, &aneos_T_guess,
                          &aneos_P, &aneos_cs, &aneos_S, &aneos_cv, &aneos_grun, &aneos_phase);
            /* The cell's CompositionType names a material whose table was never
               loaded: a setup error, and the same for every cell of that type, so
               there is nothing to recover to. endrun requests a controlled stop on
               the host and records the error for the host to see on the device --
               where the assert this replaces would have been a trap, and where an
               ideal-gas substitute would be silently wrong physics. */
            if(aneos_ierr) {endrun(920902); break;}
            press      = aneos_P / UNIT_PRESSURE_IN_CGS;
            soundspeed = aneos_cs / UNIT_VEL_IN_CGS;
            cell[i].Temperature = aneos_T_guess;
            cell[i].PhaseID = aneos_phase;
            break;
        }
#endif
        default:
            break;
    }
#endif

#ifdef EOS_MHD_CORE_BAROTROPIC
    press = 0.04*cell[i].Density*sqrt(1.+pow(cell[i].Density/1.47705e8 ,4./3.));
#endif
#ifdef EOS_ENFORCE_ADIABAT
    press = EOS_ENFORCE_ADIABAT * pow(cell[i].Density, gamma_eos_index);
#endif
#if defined(EOS_ENFORCE_ADIABAT) || defined(EOS_MHD_CORE_BAROTROPIC)
#ifdef TURB_DRIVING
    cell[i].EgyDiss += (cell[i].InternalEnergy - press / (cell[i].Density * (gamma_eos_index-1.)));
#endif
    cell[i].InternalEnergy = cell[i].InternalEnergyPred = press / (cell[i].Density * (gamma_eos_index-1.));
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
    /* ForceSoftening_KernelRadius(p) (gravity/forcetree.cc) is just
     * `return P[p].ForceSoftening;` — inline directly via pp[] so this
     * KOKKOS_INLINE_FUNCTION doesn't depend on the host-only forward decl
     * being visible (cooling.cc deliberately includes eos_functions.h
     * BEFORE proto.h for nvcc execution-space precedence — see
     * cooling.cc:15-20 comment). KERNEL_FAC_FROM_FORCESOFT_TO_PLUMMER macro
     * comes from mesh/kernel.h, included at top of this file under the same
     * gate. Phase D fix 2026-05-21 config 118 EOS_TRUELOVE_PRESSURE. */
    double h_eff = DMAX(pp[i].Get_Particle_Size(), KERNEL_FAC_FROM_FORCESOFT_TO_PLUMMER*pp[i].ForceSoftening);
    double NJeans = 4;
    double xJeans = (NJeans * NJeans / gamma_eos_index) * All.G * h_eff*h_eff * cell[i].Density * cell[i].Density /All.cf_atime;
    if(xJeans>press) press=xJeans;
#endif

#if defined(HYDRO_GENERATE_TARGET_MESH)
    /* Device-callable: the two user-edit bodies are KOKKOS_INLINE_FUNCTION earlier in
       this header, so this branch runs on either pass and no longer needs the
       host-only guard. */
    press = return_user_desired_target_pressure(i, pp, cell) * (cell[i].Density / return_user_desired_target_density(i, pp, cell));
    cell[i].InternalEnergy = cell[i].InternalEnergyPred = return_user_desired_target_pressure(i, pp, cell) / ((gamma_eos_index-1) * cell[i].Density);
#endif

#ifdef EOS_GENERAL
    if(soundspeed == 0) {double rho_for_soundspeed = cell[i].density_for_energy(); /* every pressure term assembled above scales with the density, so P/rho keeps a finite limit as rho->0: evaluate that limit rather than forming 0/0 */
        if(rho_for_soundspeed > 0) {cell[i].SoundSpeed = sqrt(gamma_eos_index * press / rho_for_soundspeed);} else {cell[i].SoundSpeed = sqrt(gamma_eos_index * (gamma_eos_index-1.) * cell[i].InternalEnergyPred);}
    } else {cell[i].SoundSpeed = soundspeed;}
#endif

    cell[i].Pressure = press;
}
