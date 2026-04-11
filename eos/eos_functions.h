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

/* return an estimate of the Hydrogen molecular fraction of gas */
KOKKOS_INLINE_FUNCTION double Get_Gas_Molecular_Mass_Fraction(int i, double temperature, double neutral_fraction, double free_electron_ratio, double urad_from_uvb_in_G0, struct particle_data *pp, struct gas_cell_data *cell)
{
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
    double surface_density_Msun_pc2_infty = 0.05 * evaluate_NH_from_GradRho(pp[i].GradRho,pp[i].KernelRadius,cell[i].Density,pp[i].NumNgb,1,i) * UNIT_SURFDEN_IN_CGS / 0.000208854; // approximate column density with Sobolev or Treecol methods as appropriate; converts to M_solar/pc^2
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


#if (SINGLE_STAR_SINK_FORMATION & 256) || (COOL_MOLECFRAC == 2) /* estimate f_H2 with Krumholz & Gnedin 2010 fitting function, assuming simple scalings of radiation field, clumping, and other factors with basic gas properties so function only of surface density and metallicity, truncated at low values (or else it gives non-sensical answers) */
    double clumping_factor=1, fH2_kg=0, tau_fmol = (0.1 + pp[i].Metallicity[0]/All.SolarAbundances[0]) * evaluate_NH_from_GradRho(pp[i].GradRho,pp[i].KernelRadius,cell[i].Density,pp[i].NumNgb,1,i) * 434.78 * UNIT_SURFDEN_IN_CGS; // convert units for surface density. also limit to Z>=0.1, where their fits were actually good, or else get unphysically low molecular fractions
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
    return calculate_mean_molecular_weight(&(ChimesGasVars[target]), &ChimesGlobalVars);
#elif defined(COOLING)
    double X=HYDROGEN_MASSFRAC, Y=1.-X, Z=0, fmol;
#ifdef METALS
    if(target >= 0)
    {
        Z = DMIN(0.25,pp[target].Metallicity[0]); if(NUM_METAL_SPECIES>=10) {Y = DMIN(0.35,pp[target].Metallicity[1]);}
        X = 1. - (Y+Z);
    }
#endif
    fmol = Get_Gas_Molecular_Mass_Fraction(target, T_guess, *xH0, *ne_guess, urad_from_uvb_in_G0, pp, cell); /* use our simple subroutine to estimate this, ignoring UVB and with clumping factor=1 */
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
    double ne=1, nh0=0, nHe0, nHepp, nhp, nHeII, temperature, mu_meanwt=1, rho=cell[i].Density*All.cf_a3inv, u0=cell[i].InternalEnergyPred;
    temperature = ThermalProperties(u0, rho, i, &mu_meanwt, &ne, &nh0, &nhp, &nHe0, &nHeII, &nHepp, pp, cell); // get thermodynamic properties
    double f_ion = DMIN(DMAX(DMAX(DMAX(1-nh0, nhp), ne/1.2), 1.e-8), 1.); // account for different measures above (assuming primordial composition)
    if((!isfinite(f_ion)) || (f_ion<0)) {f_ion=0;}
    return f_ion;
#endif
#endif
    return 1;
}
