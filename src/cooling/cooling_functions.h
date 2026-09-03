/* cooling_functions.h — GPU-callable cooling/ionization functions.
 *
 * These functions were extracted from cooling.cc to enable cross-TU device
 * calls (required for density postloop GPU port and cooling scatter GPU port).
 * They take the cooling tables as data, through the view in cooling_tables.h,
 * so they can be compiled into a kernel from any translation unit.
 *
 * Functions included (dependency order, leaves first):
 *   return_local_gammamultiplier_impl
 *   return_uvb_shieldfac
 *   find_abundances_and_rates_impl
 *   convert_temp_to_u_impl (EOS_SUBSTELLAR_ISM only)
 *   convert_u_to_temp_impl
 *   ThermalProperties
 *
 * All are KOKKOS_INLINE_FUNCTION for device callability from any TU.
 * The originals in cooling.cc are replaced with #include of this header.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef COOLING_FUNCTIONS_H
#define COOLING_FUNCTIONS_H

#include "cooling_tables.h"
#include "../eos/eos_functions.h"
#include "../declarations/multifluid_helpers.h"

#ifdef COOLING
#if !defined(CHIMES)

#ifndef NUM_SPECIES_IN_EOS
#define NUM_SPECIES_IN_EOS 5
#endif

/* Table access: the functions below take the tables as data, through the view
   declared in cooling_tables.h, and never name the global instance. That is what
   lets them be compiled into a kernel from any translation unit -- a device
   symbol cannot be shared across translation units in this build, so a function
   that reached for the instance by name would read storage nobody fills, and do
   it silently. Each entry point here has a matching host wrapper in cooling.cc,
   which owns the tables, so callers on the host are unaffected. */


/* ================================================================
   return_local_gammamultiplier — local UVB multiplier from RT
   ================================================================ */
KOKKOS_INLINE_FUNCTION
double return_local_gammamultiplier_impl(int target, struct gas_cell_data *cell, const struct PhysicsTablesView *tables)
{
#if defined(GALSF_FB_FIRE_RT_LONGRANGE) && !defined(CHIMES)
    if((target >= 0) && (tables->cooling->gJH0 > 0))
    {
        double local_gammamultiplier = cell[target].Rad_Flux_EUV * 2.29e-10;
        local_gammamultiplier = 1.0 + local_gammamultiplier / tables->cooling->gJH0;
        if(!isfinite(local_gammamultiplier)) {local_gammamultiplier=1;}
        return DMAX(1., DMIN(2./tables->cooling->gJH0, DMIN(1.e20, local_gammamultiplier)));
    }
#endif
    return 1;
}


/* ================================================================
   return_uvb_shieldfac — UVB self-shielding attenuation
   ================================================================ */
KOKKOS_INLINE_FUNCTION
double return_uvb_shieldfac(int target, double gamma_12, double nHcgs, double logT, struct gas_cell_data *cell)
{
#ifdef GALSF_EFFECTIVE_EQS
    return 1;
#endif
#if ((GALSF_FB_FIRE_STELLAREVOLUTION > 2) || !defined(GALSF_FB_FIRE_STELLAREVOLUTION)) && defined(GALSF_FB_FIRE_RT_HIIHEATING)
    if(target>=0) {if(cell[target].DelayTimeHII > 0) {return 1;}}
#endif
    double NH_SS_z, NH_SS = 0.0123;
    if(gamma_12>0) {NH_SS_z = NH_SS*pow(gamma_12,0.66)*pow(10.,0.173*(logT-4.));} else {NH_SS_z = NH_SS*pow(10.,0.173*(logT-4.));}
    double q_SS = nHcgs/NH_SS_z;
#if defined(GALSF_FB_FIRE_STELLAREVOLUTION) && (GALSF_FB_FIRE_STELLAREVOLUTION <= 2)
    return 1./(1.+q_SS*(1.+q_SS/2.*(1.+q_SS/3.*(1.+q_SS/4.*(1.+q_SS/5.*(1.+q_SS/6.*q_SS))))));
#endif
    return 0.98 / pow(1 + pow(q_SS,1.64), 2.28) + 0.02 / pow(1 + q_SS*(1.+1.e-4*nHcgs*nHcgs*nHcgs*nHcgs), 0.84);
}


/* ================================================================
   find_abundances_and_rates — ionization equilibrium solver
   ================================================================ */
KOKKOS_INLINE_FUNCTION
double find_abundances_and_rates_impl(double logT, double rho, int target, double shieldfac, int return_cooling_mode,
                                 double *ne_guess, double *nH0_guess, double *nHp_guess, double *nHe0_guess, double *nHep_guess, double *nHepp_guess,
                                 double *mu_guess, double *LambdaExc_return, double *LambdaIon_return, double *LambdaRec_return, double *LambdaFF_return,
                                 struct particle_data *pp, struct gas_cell_data *cell, const struct PhysicsTablesView *tables)
{
    int j, jp, niter;
    double Tlow, Thi, flow, fhi, t, gJH0ne, gJHe0ne, gJHepne, logT_input, rho_input, ne_input, neold, nenew;
    double bH0, bHep, bff, aHp, aHep, aHepp, ad, geH0, geHe0, geHep, EPSILON_SMALL=1.e-40;
    double n_elec, nH0, nHe0, nHp, nHep, nHepp;
    logT_input = logT; rho_input = rho; ne_input = *ne_guess;
    if(!isfinite(logT)) {logT=tables->cooling->Tmin;}
    if(!isfinite(rho)) {logT=tables->cooling->Tmin;}

    if(logT <= tables->cooling->Tmin)
    {
        nH0 = 1.0; nHe0 = yhelium(target, pp); nHp = 0; nHep = 0; nHepp = 0; n_elec = 1.e-22;
        *nH0_guess=nH0; *nHe0_guess=nHe0; *nHp_guess=nHp; *nHep_guess=nHep; *nHepp_guess=nHepp; *ne_guess=n_elec;
        *mu_guess=Get_Gas_Mean_Molecular_Weight_mu(pow(10.,logT), rho, nH0_guess, ne_guess, 0, target, pp, cell);
        return 0;
    }
    if(logT >= tables->cooling->Tmax)
    {
        nH0 = 0; nHe0 = 0; nHp = 1.0; nHep = 0; nHepp = yhelium(target, pp); n_elec = nHp + 2.0 * nHepp;
        *nH0_guess=nH0; *nHe0_guess=nHe0; *nHp_guess=nHp; *nHep_guess=nHep; *nHepp_guess=nHepp; *ne_guess=n_elec;
        *mu_guess=Get_Gas_Mean_Molecular_Weight_mu(pow(10.,logT), rho, nH0_guess, ne_guess, 1.e3, target, pp, cell);
        return 0;
    }

    t = (logT - tables->cooling->Tmin) / tables->cooling->deltaT;
    j = (int) t;
    if(j<0) {j=0;}
    if(j>NCOOLTAB){
        PRINT_WARNING("j>=NCOOLTAB : j=%d t %g Tlow %g Thi %g logT %g Tmin %g deltaT %g \n",j,t,tables->cooling->Tmin+tables->cooling->deltaT*j,tables->cooling->Tmin+tables->cooling->deltaT*(j+1),logT,tables->cooling->Tmin,tables->cooling->deltaT);
        j=NCOOLTAB;
    }
    jp = j + 1;
    if(jp > NCOOLTAB) {jp=NCOOLTAB;}
    Tlow = tables->cooling->Tmin + tables->cooling->deltaT * j;
    Thi = Tlow + tables->cooling->deltaT;
    fhi = t - j;
    flow = 1 - fhi;
    if(*ne_guess == 0)
    {
        *ne_guess = 1.0;
        if(logT < 3.8) {*ne_guess = 0.1;}
        if(logT < 2) {*ne_guess = 1.e-10;}
    }
    double local_gammamultiplier = return_local_gammamultiplier_impl(target, cell, tables);
    double nHcgs = HYDROGEN_MASSFRAC * rho / PROTONMASS_CGS;
    if(shieldfac < 0) {shieldfac = return_uvb_shieldfac(target, local_gammamultiplier*tables->cooling->gJH0/1.0e-12, nHcgs, logT, cell);}
    n_elec = *ne_guess; if(!isfinite(n_elec)) {n_elec=1;}
    neold = n_elec; niter = 0;
    double dt = 0, fac_noneq_cgs = 0, necgs = n_elec * nHcgs, ne_lower=0, ne_upper=2.;
    int bisection_mode=0;
    if(target >= 0) {dt = get_particle_timestep_in_physical(target, pp);}
#ifdef TRANSPORT_SUBCYCLE_COOLING
    dt *= All.Transport_Subcycle_dt_fraction;
#endif
    fac_noneq_cgs = (dt * UNIT_TIME_IN_CGS) * (necgs + 1.e-30*nHcgs);

#if defined(RT_CHEM_PHOTOION)
    double c_light_ne=0;
    if(target >= 0)
    {
        nH0 = cell[target].HI;
#ifdef RT_CHEM_PHOTOION_HE
        nHe0 = cell[target].HeI; nHep = cell[target].HeII;
#endif
    }
#endif

    do
    {
        niter++;

        aHp = flow * tables->cooling->AlphaHp[j] + fhi * tables->cooling->AlphaHp[jp];
        aHep = flow * tables->cooling->AlphaHep[j] + fhi * tables->cooling->AlphaHep[jp];
        aHepp = flow * tables->cooling->AlphaHepp[j] + fhi * tables->cooling->AlphaHepp[jp];
        ad = flow * tables->cooling->Alphad[j] + fhi * tables->cooling->Alphad[jp];
        geH0 = flow * tables->cooling->GammaeH0[j] + fhi * tables->cooling->GammaeH0[jp];
        geH0 = DMAX(geH0, EPSILON_SMALL);
        geHe0 = flow * tables->cooling->GammaeHe0[j] + fhi * tables->cooling->GammaeHe0[jp];
        geHe0 = DMAX(geHe0, EPSILON_SMALL);
        geHep = flow * tables->cooling->GammaeHep[j] + fhi * tables->cooling->GammaeHep[jp];
        geHep = DMAX(geHep, EPSILON_SMALL);
        fac_noneq_cgs = (dt * UNIT_TIME_IN_CGS) * (necgs + 1.e-30*nHcgs);
        if(necgs <= 1.e-25 || tables->cooling->J_UV == 0)
        {
            gJH0ne = gJHe0ne = gJHepne = 0;
        }
        else
        {
            gJH0ne = tables->cooling->gJH0 * local_gammamultiplier / necgs * shieldfac;
            gJH0ne = DMAX(gJH0ne, EPSILON_SMALL); if(!isfinite(gJH0ne)) {gJH0ne=0;}
            gJHe0ne = tables->cooling->gJHe0 * local_gammamultiplier / necgs * shieldfac;
            gJHe0ne = DMAX(gJHe0ne, EPSILON_SMALL); if(!isfinite(gJHe0ne)) {gJHe0ne=0;}
            gJHepne = tables->cooling->gJHep * local_gammamultiplier / necgs * shieldfac;
            gJHepne = DMAX(gJHepne, EPSILON_SMALL); if(!isfinite(gJHepne)) {gJHepne=0;}
        }
#if defined(RT_DISABLE_UV_BACKGROUND)
        gJH0ne = gJHe0ne = gJHepne = 0;
#endif
#if defined(RT_CHEM_PHOTOION)
        if(target >= 0)
        {
            int k;
            c_light_ne = C_LIGHT_CGS / ((MIN_REAL_NUMBER + necgs) * UNIT_LENGTH_IN_CGS);
            double gJH0ne_0=tables->cooling->gJH0 * local_gammamultiplier / (MIN_REAL_NUMBER + necgs), gJHe0ne_0=tables->cooling->gJHe0 * local_gammamultiplier / (MIN_REAL_NUMBER + necgs), gJHepne_0=tables->cooling->gJHep * local_gammamultiplier / (MIN_REAL_NUMBER + necgs);
            gJH0ne = DMAX(gJH0ne, EPSILON_SMALL); if(!isfinite(gJH0ne)) {gJH0ne=0;}
            gJHe0ne = DMAX(gJHe0ne, EPSILON_SMALL); if(!isfinite(gJHe0ne)) {gJHe0ne=0;}
            gJHepne = DMAX(gJHepne, EPSILON_SMALL); if(!isfinite(gJHepne)) {gJHepne=0;}
#if defined(RT_DISABLE_UV_BACKGROUND)
            gJH0ne_0=gJHe0ne_0=gJHepne_0=MAX_REAL_NUMBER;
#endif
            for(k = 0; k < N_RT_FREQ_BINS; k++)
            {
                if(RT_BAND_IS_IONIZING(k))
                {
                    double n_gamma_tot = cell[target].rt_photon_number_density(k);
#ifdef RT_INFRARED
                    n_gamma_tot += rt_irband_egydensity_in_band(target,All.RHD_bins_nu_min_ev[k],All.RHD_bins_nu_max_ev[k], cell) / (DMAX(All.rt_nu_eff_eV[RT_FREQ_BIN_H0],cell[target].Radiation_Temperature/2959.81)*ELECTRONVOLT_IN_ERGS/UNIT_ENERGY_IN_CGS);
#endif
                    double c_ne_time_n_photons_vol = c_light_ne * n_gamma_tot;
                    double cross_section_ion, dummy, thold=1.0e20;
#ifdef GALSF
                    if(All.ComovingIntegrationOn) {thold=1.0e10;}
#endif
                    if(All.rt_ion_G_HI[k] > 0)
                    {
                        cross_section_ion = nH0 * All.rt_ion_sigma_HI[k];
                        dummy = All.rt_ion_sigma_HI[k] * c_ne_time_n_photons_vol;
                        gJH0ne += dummy;
                    }
#if 1
                    if(All.rt_ion_G_HeI[k] > 0)
                    {
                        cross_section_ion = nHe0 * All.rt_ion_sigma_HeI[k];
                        dummy = All.rt_ion_sigma_HeI[k] * c_ne_time_n_photons_vol;
                        gJHe0ne += dummy;
                    }
                    if(All.rt_ion_G_HeII[k] > 0)
                    {
                        cross_section_ion = nHep * All.rt_ion_sigma_HeII[k];
                        dummy = All.rt_ion_sigma_HeII[k] * c_ne_time_n_photons_vol;
                        gJHepne += dummy;
                    }
#endif
                }
            }
        }
#endif


        nH0 = aHp / (MIN_REAL_NUMBER + aHp + geH0 + gJH0ne);
#if defined(RT_CHEM_PHOTOION)
        if(target >= 0) {nH0 = (cell[target].HI + fac_noneq_cgs * aHp) / (1 + fac_noneq_cgs * (aHp + geH0 + gJH0ne));}
#endif
        nHp = 1.0 - nH0;

        if( ((gJHe0ne + geHe0) <= MIN_REAL_NUMBER) || (aHepp <= MIN_REAL_NUMBER) )
        {
            nHep = 0.0;
            nHepp = 0.0;
            nHe0 = yhelium(target, pp);
        }
        else
        {
            nHep = yhelium(target, pp) / (1.0 + (aHep + ad) / (geHe0 + gJHe0ne) + (geHep + gJHepne) / aHepp);
            nHe0 = nHep * (aHep + ad) / (geHe0 + gJHe0ne);
            nHepp = nHep * (geHep + gJHepne) / aHepp;
        }
#if defined(RT_CHEM_PHOTOION) && defined(RT_CHEM_PHOTOION_HE)
        if(target >= 0)
        {
            double yHe = yhelium(target, pp);
            nHep = cell[target].HeII + yHe * fac_noneq_cgs * (geHe0 + gJHe0ne) - cell[target].HeIII * (fac_noneq_cgs*(geHe0 + gJHe0ne - aHepp) / (1.0 + fac_noneq_cgs*aHepp));
            nHep /= 1.0 + fac_noneq_cgs*(geHe0 + gJHe0ne + aHep + ad + geHep + gJHepne) + (fac_noneq_cgs*(geHe0 + gJHe0ne - aHepp) / (1.0 + fac_noneq_cgs*aHepp)) * fac_noneq_cgs*(geHep + gJHepne);
            if(nHep < 0) {nHep=0;}
            if(nHep > yHe) {nHep=yHe;}
            nHepp = (cell[target].HeIII + cell[target].HeII * fac_noneq_cgs*(geHep + gJHepne)) / (1. + fac_noneq_cgs*aHepp);
            if(nHepp < 0) {nHepp=0;}
            if(nHepp > yHe-nHep) {nHepp=yHe-nHep;}
            nHe0 = yHe - (nHep + nHepp);
        }
#endif
        if(!isfinite(n_elec)) {printf("target=%d niter=%d logT=%g n_elec/old=%g/%g nHp/nHep/nHepp=%g/%g/%g nHcgs=%g yHe=%g dt=%g shieldfac/local_gammamult=%g/%g aHp/aHep/aHepp=%g/%g/%g geH0/geHe0/geHep=%g/%g/%g gJH0ne/gJHe0ne/gJHepne=%g/%g/%g \n",target,niter,logT,n_elec,neold,nHp,nHep,nHepp,nHcgs,yhelium(target, pp),dt,shieldfac,local_gammamultiplier,aHp,aHep,aHepp,geH0,geHe0,geHep,gJH0ne,gJHe0ne,gJHepne);}

        double error_old = fabs(n_elec - neold);
        neold = n_elec;
        n_elec = nHp + nHep + 2 * nHepp;
#ifdef COOL_LOW_TEMPERATURES
        double temp = pow(10.,logT);
        n_elec += return_electron_fraction_from_heavy_ions(target, temp, rho, n_elec, pp, cell);
#ifdef SIMPLE_STEADYSTATE_CHEMISTRY
        if(target >= 0) {
            n_elec += return_electron_fraction_from_alkali(target, temp, pp, cell);
            n_elec += return_electron_fraction_from_Cplus(target, temp, neold, shieldfac, pp, cell);
            n_elec += return_electron_fraction_from_Oplus(target, nHp, pp, cell);
            n_elec += return_electron_fraction_from_molecular_ions(target, temp, pp, cell);
        }
#endif
#endif

        if(n_elec > neold) {ne_lower = DMAX(neold, ne_lower);}
        if(n_elec < neold) {ne_upper = DMIN(neold, ne_upper);}

        double nenew_tolmin = DMIN(1.0e-14, 0.01 * 1.e-3/nHcgs);
        if(bisection_mode) {
            if(n_elec < neold) {nenew = 0.5*(ne_lower + neold); ne_upper=neold;}
            else {nenew = 0.5*(ne_upper + neold); ne_lower = neold;}
        } else {
            nenew = 0.5 * (n_elec + neold);
            if(niter>30 && (fabs(n_elec - neold) > 0.6 * error_old) && (nenew > nenew_tolmin)) { bisection_mode = 1;}
        }

        n_elec = nenew;
        if(!isfinite(n_elec)) {n_elec=1;}
        necgs = n_elec * nHcgs;

        double dneTHhold = DMAX(n_elec*0.01 , nenew_tolmin);
        if(fabs(n_elec - neold) < dneTHhold) break;

        if(niter > (MAXITER - 10)) {printf("n_elec= %g/%g/%g yh=%g nHcgs=%g niter=%d\n", n_elec,neold,nenew, yhelium(target, pp), nHcgs, niter);}
    }
    while(niter < MAXITER);

    if(niter >= MAXITER) {printf("failed to converge in find_abundances_and_rates(): logT_input=%g  rho_input=%g  ne_input=%g target=%d ID=%ld shieldfac=%g cooling_return=%d", logT_input, rho_input, ne_input, target, (long)(long long)target, shieldfac, return_cooling_mode); endrun(13);}

    bH0 = flow * tables->cooling->BetaH0[j] + fhi * tables->cooling->BetaH0[jp];
    bHep = flow * tables->cooling->BetaHep[j] + fhi * tables->cooling->BetaHep[jp];
    bff = flow * tables->cooling->Betaff[j] + fhi * tables->cooling->Betaff[jp];
    *nH0_guess=nH0; *nHe0_guess=nHe0; *nHp_guess=nHp; *nHep_guess=nHep; *nHepp_guess=nHepp; *ne_guess=n_elec;
    *mu_guess=Get_Gas_Mean_Molecular_Weight_mu(pow(10.,logT), rho, nH0_guess, ne_guess, sqrt(shieldfac)*(tables->cooling->gJH0/2.29e-10), target, pp, cell);
    if(target >= 0)
    {
#if defined(OUTPUT_MOLECULAR_FRACTION)
        cell[target].MolecularMassFraction = Get_Gas_Molecular_Mass_Fraction(target, pow(10.,logT), nH0, n_elec, sqrt(shieldfac)*(tables->cooling->gJH0/2.29e-10), pp, cell);
#endif
    }

    if(return_cooling_mode==1)
    {
        double LambdaExcH0 = bH0 * n_elec * nH0;
        double LambdaExcHep = bHep * n_elec * nHep;
        double LambdaExc = LambdaExcH0 + LambdaExcHep;

        double LambdaIonH0 = 2.18e-11 * geH0 * n_elec * nH0;
        double LambdaIonHe0 = 3.94e-11 * geHe0 * n_elec * nHe0;
        double LambdaIonHep = 8.72e-11 * geHep * n_elec * nHep;
        double LambdaIon = LambdaIonH0 + LambdaIonHe0 + LambdaIonHep;

        double T_lin = pow(10.0, logT);
        double LambdaRecHp = 1.036e-16 * T_lin * n_elec * (aHp * nHp);
        double LambdaRecHep = 1.036e-16 * T_lin * n_elec * (aHep * nHep);
        double LambdaRecHepp = 1.036e-16 * T_lin * n_elec * (aHepp * nHepp);
        double LambdaRecHepd = 6.526e-11 * ad * n_elec * nHep;
        double LambdaRec = LambdaRecHp + LambdaRecHep + LambdaRecHepp + LambdaRecHepd;

        double LambdaFF = bff * (nHp + nHep + 4 * nHepp) * n_elec;
        double Lambda = LambdaExc + LambdaIon + LambdaRec + LambdaFF;

        *LambdaExc_return = LambdaExc; *LambdaIon_return = LambdaIon; *LambdaRec_return = LambdaRec; *LambdaFF_return = LambdaFF;
        return Lambda;
    }
    return 0;
}


/* ================================================================
   convert_temp_to_u — temperature to internal energy (EOS_SUBSTELLAR_ISM)
   ================================================================ */
#if defined(EOS_SUBSTELLAR_ISM)
KOKKOS_INLINE_FUNCTION
double convert_temp_to_u_impl(double temp, double rho, int target, double *cv, double *ne, double *nH0, double *nHp, double *nHe0, double *nHep, double *nHepp, double *mu, struct particle_data *pp, struct gas_cell_data *cell, const struct PhysicsTablesView *tables) {
    double dummy;
    find_abundances_and_rates_impl(log10(temp), rho, target, -1, 0, ne, nH0, nHp, nHe0, nHep, nHepp, mu, &dummy, &dummy, &dummy,&dummy, pp, cell, tables);
    double X = HYDROGEN_MASSFRAC, Y = 1. - X, Z = 0, fmol = 0;
#ifdef METALS
    if (target >= 0) {
        Z = DMIN(0.25, pp[target].Metallicity[0]);
        if (NUM_METAL_SPECIES >= 10) {
            Y = DMIN(0.35, pp[target].Metallicity[1]);
        }
        X = 1. - (Y + Z);
    }
#endif
    double urad_from_uvb_in_G0 = MIN_REAL_NUMBER;
    if(target >= 0) { fmol = Get_Gas_Molecular_Mass_Fraction(target, temp, *nH0, *ne, urad_from_uvb_in_G0, pp, cell); }

    double E_i[NUM_SPECIES_IN_EOS] = {0}, cv_i[NUM_SPECIES_IN_EOS] = {0}, m_i[NUM_SPECIES_IN_EOS] = {0}, N_i[NUM_SPECIES_IN_EOS] = {0};
    double cv_mono = 1.5 * BOLTZMANN_CGS;
    cv_i[1] = cv_i[2] = cv_i[3] = cv_i[4] = cv_mono;
    E_i[1] = E_i[2] = E_i[3] = E_i[4] = cv_mono * temp;
#ifndef EOS_SUBSTELLAR_ISM
    E_i[0] = cv_mono * temp;
    cv_i[0] = 7. / 5 * BOLTZMANN_CGS;
#else
    if (fmol > MIN_REAL_NUMBER) {
        double z[3];
        hydrogen_molecule_partitionfunc(temp, z);
        E_i[0] = z[0];
        cv_i[0] = z[1];
    }
#endif
    m_i[0] = 2.;
    N_i[0] = 0.5 * X * fmol;
    m_i[1] = 1.;
    N_i[1] = X * (1 - fmol);
    m_i[2] = 4.;
    N_i[2] = 0.25 * Y;
    m_i[3] = ELECTRONMASS_CGS / PROTONMASS_CGS;
    N_i[3] = *ne * X;
    m_i[4] = 16. + 12. * fmol;
    N_i[4] = Z / (16. + 12. * fmol);

    int k;
    double sum_E = 0., sum_cv = 0., sum_M = 0.;
    for (k = 0; k < NUM_SPECIES_IN_EOS; k++) {
        sum_cv += N_i[k] * cv_i[k];
        sum_E += N_i[k] * E_i[k];
        sum_M += N_i[k] * m_i[k];
    }
    *cv = sum_cv / sum_M / PROTONMASS_CGS;
    return sum_E / sum_M / PROTONMASS_CGS;
}
#endif /* EOS_SUBSTELLAR_ISM */


/* ================================================================
   convert_u_to_temp — internal energy to temperature
   ================================================================ */
#if defined(EOS_SUBSTELLAR_ISM)
KOKKOS_INLINE_FUNCTION
double convert_u_to_temp_impl(double u, double rho, int target, double *ne, double *nH0, double *nHp, double *nHe0, double *nHep, double *nHepp, double *mu, struct particle_data *pp, struct gas_cell_data *cell, const struct PhysicsTablesView *tables) {
    double dT = 1e100, dT_old = 1e100, du=1e100, du_old=1e100, temp = 0.9 * u * PROTONMASS_CGS / BOLTZMANN_CGS, cv, u_from_temp;
    double temp_min_0 = DMAX(DMIN(1.e-3,pow(10.,tables->cooling->Tmin)), 0.1*temp), temp_max_0=DMIN(DMAX(1.e12,pow(10.,tables->cooling->Tmax)),temp*10), temp_min=temp_min_0, temp_max=temp_max_0;
    if(target >= 0) { temp = cell[target].gas_temperature_from_u(u / UNIT_SPECEGY_IN_CGS); } /* same relation, taken from the cache that owns it */
    else            { temp = u * PROTONMASS_CGS / BOLTZMANN_CGS; }
    temp = DMIN(DMAX(temp,temp_min),temp_max);
    const double tolerance = 1e-4;
    double dummy;
    int iter = 0, bisection = 0;
    do {
        u_from_temp = convert_temp_to_u_impl(temp, rho, target, &cv, ne, nH0, nHp, nHe0, nHep, nHepp, mu, pp, cell, tables);
        du_old = du;
        du = u_from_temp - u;
        if(du > 0 && temp <= temp_min){return temp_min;}
        if(du < 0 && temp >= temp_max){return temp_max;}
        if(du > 0){temp_max = DMIN(temp, temp_max);} else {temp_min = DMAX(temp, temp_min);}
        dT_old = dT;
        if(iter==0){
            dT = -du / cv;
        } else {
            dT = -du * dT_old / (du-du_old);
        }
        if (fabs(du) > 0.5 * fabs(du_old) || bisection) {
            bisection = 1;
            dT = sqrt(temp_min*temp_max) - temp;
        }
        if(dT==0){break;}
        temp += dT;
        temp = DMAX(DMIN(temp,temp_max),temp_min);
        iter++;
    } while (fabs(du) > tolerance * u && iter < MAXITER);
    if (iter >= MAXITER) {
        PRINT_WARNING("Particle ID=%lld failed to converge in convert_u_to_temp. u=%g du=%g T=%g dT=%g\n", (long long)target, u, du, temp, dT); endrun(91743);
    }
    return DMAX(DMIN(temp,temp_max_0),temp_min_0);
}

#else /* !EOS_SUBSTELLAR_ISM: default convert_u_to_temp */

KOKKOS_INLINE_FUNCTION
double convert_u_to_temp_impl(double u, double rho, int target, double *ne_guess, double *nH0_guess, double *nHp_guess, double *nHe0_guess, double *nHep_guess, double *nHepp_guess, double *mu_guess, struct particle_data *pp, struct gas_cell_data *cell, const struct PhysicsTablesView *tables)
{
    int iter = 0;
    double temp, temp_old, temp_old_old = 0, temp_new, prefac_fun_old, prefac_fun, fac, err_old, err_new, T_bracket_errneg = 0, T_bracket_errpos = 0, T_bracket_min = 0, T_bracket_max = 1.e20, bracket_sign = 0, Lambda_filler = 0;
    double u_input = u, rho_input = rho, temp_guess;
    double T_0 = u * PROTONMASS_CGS / BOLTZMANN_CGS;
    double gamma_minus1 = (target >= 0 ? cell[target].gamma_eos_value() : GAMMA_DEFAULT) - 1.;
    temp_guess = gamma_minus1 * T_0;
    *mu_guess = Get_Gas_Mean_Molecular_Weight_mu(temp_guess, rho, nH0_guess, ne_guess, 0., target, pp, cell);
    prefac_fun = gamma_minus1 * (*mu_guess);
    err_new = prefac_fun - temp_guess / T_0;
    if(err_new < 0) {T_bracket_errneg = temp_guess;} else {T_bracket_errpos = temp_guess;}
    temp = prefac_fun * T_0;

    do
    {
        prefac_fun_old = prefac_fun;
        err_old = err_new;
        find_abundances_and_rates_impl(log10(temp), rho, target, -1, 0, ne_guess, nH0_guess, nHp_guess, nHe0_guess, nHep_guess, nHepp_guess, mu_guess, &Lambda_filler, &Lambda_filler, &Lambda_filler, &Lambda_filler, pp, cell, tables);
        gamma_minus1 = (target >= 0 ? cell[target].gamma_eos_value() : GAMMA_DEFAULT) - 1.;
        prefac_fun = gamma_minus1 * (*mu_guess);
        temp_old = temp;
        temp_new = prefac_fun * T_0;
        err_new = (temp_new - temp_old) / T_0;
        if(T_bracket_errpos == 0) {if(err_new > 0) {T_bracket_errpos = temp_old;} else {T_bracket_errneg = temp_old;}}
        if(T_bracket_errneg == 0) {if(err_new < 0) {T_bracket_errneg = temp_old;} else {T_bracket_errpos = temp_old;}}
        if(T_bracket_errneg > 0 && T_bracket_errpos > 0)
        {
            if(bracket_sign == 0) {if(T_bracket_errpos > T_bracket_errneg) {bracket_sign=1;} else {bracket_sign=-1;}}
            if(err_new > 0) {
                if(bracket_sign > 0) {T_bracket_errpos = DMIN(T_bracket_errpos, temp_old);} else {T_bracket_errpos = DMAX(T_bracket_errpos, temp_old);}
            } else {
                if(bracket_sign > 0) {T_bracket_errneg = DMAX(T_bracket_errneg, temp_old);} else {T_bracket_errneg = DMIN(T_bracket_errneg, temp_old);}
            }
            if(bracket_sign > 0) {T_bracket_max=T_bracket_errpos; T_bracket_min=T_bracket_errneg;} else {T_bracket_max=T_bracket_errneg; T_bracket_min=T_bracket_errpos;}
        }

        if((fabs(prefac_fun-prefac_fun_old) < 1.e-4) && (fabs(temp_new-temp_old)/(temp_new+temp_old) < 1.e-4)) {break;}
        fac = (prefac_fun-prefac_fun_old)*T_0 / (temp_old-temp_old_old);

        if(fac > 1) {fac = 1;}
        if(fac > 0.9) {fac=0.9;}
        if(fac < -9999.) {fac=-9999.;}

        temp = temp_old + (temp_new - temp_old) * 1./(1. - fac);
        if(temp < 0.5*temp_old) {temp = 0.5*temp_old;}
        if(temp > 3.0*temp_old) {temp = 3.0*temp_old;}

        if(T_bracket_errneg > 0 && T_bracket_errpos > 0)
        {
            if(temp >= T_bracket_max || temp <= T_bracket_min) {temp = sqrt(T_bracket_min*T_bracket_max);}
        }
        temp_old_old = temp_old;
        iter++;
        if(iter > (MAXITER - 10)) {printf("-> temp_next/new/old/oldold=%g/%g/%g/%g ne=%g mu=%g rho=%g iter=%d target=%d err_new/prev=%g/%g gamma_minus_1_mu_new/prev=%g/%g Brackets: Error_bracket_positive=%g Error_bracket_negative=%g T_bracket_Min/Max=%g/%g fac_for_SecantDT=%g \n", temp,temp_new,temp_old,temp_old_old,*ne_guess, (*mu_guess) ,rho,iter,target,err_new,err_old,prefac_fun,prefac_fun_old,T_bracket_errpos,T_bracket_errneg,T_bracket_min,T_bracket_max,fac);}
    }
    while(
#if defined(RT_INFRARED)
        (fabs(temp - temp_old) > 1e-3 * temp) && iter < MAXITER);
#else
          ((fabs(temp - temp_old) > 0.25 * temp) ||
           ((fabs(temp - temp_old) > 0.1 * temp) && (temp > 20.)) ||
           ((fabs(temp - temp_old) > 0.05 * temp) && (temp > 200.)) ||
           ((fabs(temp - temp_old) > 0.01 * temp) && (temp > 200.) && (iter<100)) ||
           ((fabs(temp - temp_old) > 1.0e-3 * temp) && (temp > 200.) && (iter<10))) && iter < MAXITER);
#endif
    if(iter >= MAXITER) {printf("failed to converge in convert_u_to_temp(): u_input= %g rho_input=%g n_elec_input=%g target=%d ID=%ld\n", u_input, rho_input, *ne_guess, target, (long)(long long)target); endrun(12);}

    if(temp<=0) temp=pow(10.0,tables->cooling->Tmin);
    if(log10(temp)<tables->cooling->Tmin) temp=pow(10.0,tables->cooling->Tmin);
    return temp;
}
#endif /* EOS_SUBSTELLAR_ISM vs default */


/* ThermalProperties now lives in eos/eos_functions.h, beside yhelium and the EOS body
   that calls it. It reads saved cell state and needs nothing from this header; keeping
   it here forced eos_functions.h to reach it by external linkage, which a device TU
   outside cooling.cc cannot resolve. This header includes eos_functions.h, so every
   consumer of this file still sees it. */


#endif /* !CHIMES */
#endif /* COOLING — end of cooling-specific functions */


/* The host wrapper `set_eos_pressure` stays in eos/eos.cc and is not inlined here;
   the reason it and its body must remain separate is recorded beside the definition
   there. The device-callable `set_eos_pressure_impl` is a different thing and IS
   called from the post-cooling device kernel, so none of this says the equation of
   state is pinned to the host. */


#endif /* COOLING_FUNCTIONS_H */
