#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../allvars.h"
#include "../proto.h"

#include "./cooling.h"


/*
 * This file contains the routines for optically-thin cooling (generally aimed towards simulations of the ISM, 
 *   galaxy formation, and cosmology). A wide range of heating/cooling processes are included, including 
 *   free-free, metal-line, Compton, collisional, photo-ionization and recombination, and more. Some of these 
 *   are controlled by individual modules that need to be enabled or disabled explicitly.
 *
 * This file was originally part of the GADGET3 code developed by
 *   Volker Springel (volker.springel@h-its.org). The code has been modified heavily by 
 *   Phil Hopkins (phopkins@caltech.edu) for GIZMO; essentially everything has been re-written at this point */


#ifdef COOLING

#define NH_SS 0.0123 /* CAFG: H number density above which we assume no ionizing bkg (proper cm^-3) */
#define YHELIUM_0 ((1-HYDROGEN_MASSFRAC)/(4*HYDROGEN_MASSFRAC)) /* helium number fraction to use by default for primordial gas assumptions */

/* these are variables of the cooling tables. they are static but this shouldnt be a problem for shared-memory structure because
    they are only defined once in a global operation, then locked for particle-by-particle operations */
#define NCOOLTAB  2000 /* defines size of cooling table */
static double Tmin = 0.0, Tmax = 9.0, deltaT; /* minimum/maximum temp, in log10(T/K) and temperature gridding: will be appropriately set in make_cooling_tables subroutine below */
static double *BetaH0, *BetaHep, *Betaff, *AlphaHp, *AlphaHep, *Alphad, *AlphaHepp, *GammaeH0, *GammaeHe0, *GammaeHep; // UV background parameters
#ifdef COOL_METAL_LINES_BY_SPECIES
/* if this is enabled, the cooling table files should be in a folder named 'spcool_tables' in the run directory.
 cooling tables can be downloaded at: https://dl.dropbox.com/u/16659252/spcool_tables.tgz */
static float *SpCoolTable0, *SpCoolTable1;
#endif
/* these are constants of the UV background at a given redshift: they are interpolated from TREECOOL but then not modified particle-by-particle */
static double J_UV = 0, gJH0 = 0, gJHep = 0, gJHe0 = 0, epsH0 = 0, epsHep = 0, epsHe0 = 0;







/* this is just a simple loop to do the particle cooling. this is now openmp-parallelized, since the cooling iteration can be a non-negligible cost */
void cooling_parent_routine(void)
{
    NextParticle = FirstActiveParticle;
/*
#ifdef _OPENMP
#pragma omp parallel // static variables above causing problems with openmp shared memory (getting swapped) on some compilers: demote for now
#endif
*/
    {
        while(1)
        {
            int i, exitFlag = 0;
/*
#ifdef _OPENMP
#pragma omp critical(_nexport_)
#endif
*/
            {
                if(NextParticle<0) {exitFlag = 1;} else {i=NextParticle; NextParticle=NextActiveParticle[NextParticle];}
            }
            if(exitFlag) {break;}

            /* here apply any conditional statements about whether we should or should not enter the cooling loop */
            if(P[i].Type != 0) {continue;} /* only gas cools */
            if(P[i].Mass <= 0) {continue;} /* only non-zero mass particles cool */
#ifdef GALSF_EFFECTIVE_EQS
            if((SphP[i].Density*All.cf_a3inv > All.PhysDensThresh) && ((All.ComovingIntegrationOn==0) || (SphP[i].Density>=All.OverDensThresh))) {continue;} /* no cooling for effective-eos star-forming particles */
#endif
#ifdef GALSF_TURNOFF_COOLING_WINDS
            if(SphP[i].DelayTimeCoolingSNe > 0) {continue;} /* no cooling for particles marked in delayed cooling */
#endif
            do_the_cooling_for_particle(i);
        } /* while bracket */
    } /* omp bracket */
}





/* subroutine which actually sends the particle data to the cooling routine and updates the entropies */
void do_the_cooling_for_particle(int i)
{
    double unew;
    double dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval;
    double dtime = dt / All.cf_hubble_a; /*  the actual time-step */
    if((P[i].TimeBin)&&(dt>0)&&(P[i].Mass>0)&&(P[i].Type==0))  // upon start-up, need to protect against dt==0 //
    {
        
        double uold = DMAX(All.MinEgySpec, SphP[i].InternalEnergy);
#ifdef GALSF_FB_HII_HEATING
        double u_to_temp_fac = 0.59 * PROTONMASS / BOLTZMANN * GAMMA_MINUS1 * All.UnitEnergy_in_cgs / All.UnitMass_in_g;
        double uion = HIIRegion_Temp / u_to_temp_fac;
        if(SphP[i].DelayTimeHII > 0) if(uold<uion) uold=uion; /* u_old should be >= ionized temp if used here */
#endif // GALSF_FB_HII_HEATING
        
#ifndef COOLING_OPERATOR_SPLIT
        /* do some prep operations on the hydro-step determined heating/cooling rates before passing to the cooling subroutine */
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        /* calculate the contribution to the energy change from the mass fluxes in the gravitation field */
        double grav_acc; int k;
        for(k = 0; k < 3; k++)
        {
            grav_acc = All.cf_a2inv * P[i].GravAccel[k];
#ifdef PMGRID
            grav_acc += All.cf_a2inv * P[i].GravPM[k];
#endif
            SphP[i].DtInternalEnergy -= SphP[i].GravWorkTerm[k] * All.cf_atime * grav_acc;
        }
#endif
        /* limit the magnitude of the hydro dtinternalenergy */
        double du = SphP[i].DtInternalEnergy * dtime;
        if(du < -0.5*SphP[i].InternalEnergy) {SphP[i].DtInternalEnergy = -0.5*SphP[i].InternalEnergy / dtime;}
        if(du >  50.*SphP[i].InternalEnergy) {SphP[i].DtInternalEnergy =  50.*SphP[i].InternalEnergy / dtime;}
        /* and convert to cgs before use in the cooling sub-routine */
        SphP[i].DtInternalEnergy *= All.HubbleParam * All.UnitEnergy_in_cgs / (All.UnitMass_in_g * All.UnitTime_in_s) * (PROTONMASS/HYDROGEN_MASSFRAC);
#endif
        
        
#ifndef RT_COOLING_PHOTOHEATING_OLDFORMAT
        /* Call the actual COOLING subroutine! */
        unew = DoCooling(uold, SphP[i].Density * All.cf_a3inv, dtime, SphP[i].Ne, i);
#else
        double fac_entr_to_u = pow(SphP[i].Density * All.cf_a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;
        unew = uold + dt * fac_entr_to_u * (rt_DoHeating(i, dt) + rt_DoCooling(i, dt));
#endif
        
        
#ifdef GALSF_FB_HII_HEATING
        /* set internal energy to minimum level if marked as ionized by stars */
        if(SphP[i].DelayTimeHII > 0)
        {
            if(unew<uion) {unew=uion; if(SphP[i].DtInternalEnergy<0) SphP[i].DtInternalEnergy=0;}
            SphP[i].Ne = 1.0 + 2.0*yhelium(i); /* fully ionized */
        }
#endif // GALSF_FB_HII_HEATING
        
        
#if defined(BH_THERMALFEEDBACK)
        if(SphP[i].Injected_BH_Energy)
		{
            unew += SphP[i].Injected_BH_Energy / P[i].Mass;
            SphP[i].Injected_BH_Energy = 0;
		}
#endif
        

#if defined(COSMIC_RAYS) && !defined(COSMIC_RAYS_DISABLE_COOLING)
        /* cosmic ray interactions affecting the -thermal- temperature of the gas are included in the actual cooling/heating functions; 
            they are solved implicitly above. however we need to account for energy losses of the actual cosmic ray fluid, here. The 
            timescale for this is reasonably long, so we can treat it semi-explicitly, as we do here.
            -- We use the estimate for combined hadronic + Coulomb losses from Volk 1996, Ensslin 1997, as updated in Guo & Oh 2008: */
        double ne_cgs = ((0.78 + 0.22*SphP[i].Ne*HYDROGEN_MASSFRAC) / PROTONMASS) * (SphP[i].Density * All.cf_a3inv * All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam);
        double CR_coolingrate_perunitenergy = -7.51e-16 * ne_cgs * (All.UnitTime_in_s / All.HubbleParam); // converts cgs to code units //
        if(dtime > 0)
        {
            double q_CR_cool = exp(CR_coolingrate_perunitenergy * dtime);
            if(CR_coolingrate_perunitenergy * dtime < -20.) {q_CR_cool = 0;}
            SphP[i].CosmicRayEnergyPred *= q_CR_cool; SphP[i].CosmicRayEnergy *= q_CR_cool;
#ifdef COSMIC_RAYS_M1
            int kCR; for(kCR=0;kCR<3;kCR++) {SphP[i].CosmicRayFlux[kCR] *= q_CR_cool; SphP[i].CosmicRayFluxPred[kCR] *= q_CR_cool;}
#endif
        }
#endif
        
        
        /* InternalEnergy, InternalEnergyPred, Pressure, ne are now immediately updated; however, if COOLING_OPERATOR_SPLIT
         is set, then DtInternalEnergy carries information from the hydro loop which is only half-stepped here, so is -not- updated. 
         if the flag is not set (default), then the full hydro-heating is accounted for in the cooling loop, so it should be re-zeroed here */
        SphP[i].InternalEnergy = unew;
        SphP[i].InternalEnergyPred = SphP[i].InternalEnergy;
        SphP[i].Pressure = get_pressure(i);
#ifndef COOLING_OPERATOR_SPLIT
        SphP[i].DtInternalEnergy = 0;
#endif
        
        
#ifdef GALSF_FB_HII_HEATING
        /* count off time which has passed since ionization 'clock' */
        if(SphP[i].DelayTimeHII > 0) SphP[i].DelayTimeHII -= dtime;
        if(SphP[i].DelayTimeHII < 0) SphP[i].DelayTimeHII = 0;
#endif // GALSF_FB_HII_HEATING
        
    } // closes if((P[i].TimeBin)&&(dt>0)&&(P[i].Mass>0)&&(P[i].Type==0)) check
}






/* returns new internal energy per unit mass. 
 * Arguments are passed in code units, density is proper density.
 */
double DoCooling(double u_old, double rho, double dt, double ne_guess, int target)
{
    double u, du, u_lower, u_upper, ratefact, LambdaNet;
    int iter=0, iter_upper=0, iter_lower=0;
    
#ifdef GRACKLE
#ifndef COOLING_OPERATOR_SPLIT
    /* because grackle uses a pre-defined set of libraries, we can't properly incorporate the hydro heating
     into the cooling subroutine. instead, we will use the approximate treatment below to split the step */
    du = dt * SphP[target].DtInternalEnergy / (All.HubbleParam * All.UnitEnergy_in_cgs / (All.UnitMass_in_g * All.UnitTime_in_s) * (PROTONMASS/HYDROGEN_MASSFRAC));
    u_old += 0.5*du;
    u = CallGrackle(u_old, rho, dt, ne_guess, target, 0);
    /* now we attempt to correct for what the solution would have been if we had included the remaining half-step heating
     term in the full implicit solution. The term "r" below represents the exact solution if the cooling function has
     the form d(u-u0)/dt ~ -a*(u-u0)  around some u0 which is close to the "ufinal" returned by the cooling routine,
     to which we then add the heating term from hydro and compute the solution over a full timestep */
    double r=u/u_old; if(r>1) {r=1/r;} if(fabs(r-1)>1.e-4) {r=(r-1)/log(r);} r=DMAX(0,DMIN(r,1));
    du *= 0.5*r; if(du<-0.5*u) {du=-0.5*u;} u+=du;
#else
    /* with full operator splitting we just call grackle normally. note this is usually fine,
     but can lead to artificial noise at high densities and low temperatures, especially if something
     like artificial pressure (but not temperature) floors are used such that the temperature gets
     'contaminated' by the pressure terms */
    u = CallGrackle(u_old, rho, dt, ne_guess, target, 0);
#endif
    return DMAX(u,All.MinEgySpec);
#endif
    
    rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;	/* convert to physical cgs units */
    u_old *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
    dt *= All.UnitTime_in_s / All.HubbleParam;
    double nHcgs = HYDROGEN_MASSFRAC * rho / PROTONMASS;	/* hydrogen number dens in cgs units */
    ratefact = nHcgs * nHcgs / rho;
    
    u = u_old; u_lower = u; u_upper = u; /* initialize values */
    LambdaNet = CoolingRateFromU(u, rho, ne_guess, target);

    /* bracketing */
    if(u - u_old - ratefact * LambdaNet * dt < 0)	/* heating */
    {
        u_upper *= sqrt(1.1); u_lower /= sqrt(1.1);
        while((iter_upper<MAXITER)&&(u_upper - u_old - ratefact * CoolingRateFromU(u_upper, rho, ne_guess, target) * dt < 0))
        {
            u_upper *= 1.1; u_lower *= 1.1; iter_upper++;
        }
        
    }

    if(u - u_old - ratefact * LambdaNet * dt > 0) /* cooling */
    {
        u_lower /= sqrt(1.1); u_upper *= sqrt(1.1);
        while((iter_lower<MAXITER)&&(u_lower - u_old - ratefact * CoolingRateFromU(u_lower, rho, ne_guess, target) * dt > 0))
        {
            u_upper /= 1.1; u_lower /= 1.1; iter_lower++;
        }
    }

    /* core iteration to convergence */
    do
    {
        u = 0.5 * (u_lower + u_upper);
        LambdaNet = CoolingRateFromU(u, rho, ne_guess, target);
        if(u - u_old - ratefact * LambdaNet * dt > 0) {u_upper = u;} else {u_lower = u;}
        du = u_upper - u_lower;
        iter++;
        if(iter >= (MAXITER - 10)) {printf("u=%g u_old=%g u_upper=%g u_lower=%g ne_guess=%g dt=%g iter=%d \n", u,u_old,u_upper,u_lower,ne_guess,dt,iter);}
    }
    while(((fabs(du/u) > 3.0e-2)||((fabs(du/u) > 3.0e-4)&&(iter < 10))) && (iter < MAXITER)); /* iteration condition */
    /* crash condition */
    if(iter >= MAXITER) {printf("failed to converge in DoCooling(): u_in=%g rho_in=%g dt=%g ne_in=%g target=%d \n",u_old,rho,dt,ne_guess,target); endrun(10);}
    
    /* safe return */
    return u * All.UnitDensity_in_cgs / All.UnitPressure_in_cgs;	/* in internal units */
}




/* returns cooling time. 
 * NOTE: If we actually have heating, a cooling time of 0 is returned.
 */
double GetCoolingTime(double u_old, double rho, double ne_guess, int target)
{
#if defined(GRACKLE) && !defined(GALSF_EFFECTIVE_EQS)
    double LambdaNet = CallGrackle(u_old, rho, 0.0, ne_guess, target, 1);
    if(LambdaNet >= 0) LambdaNet = 0.0;
    return LambdaNet * All.HubbleParam / All.UnitTime_in_s;
#else
    rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;	/* convert to physical cgs units */
    u_old *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
    double nHcgs = HYDROGEN_MASSFRAC * rho / PROTONMASS;	/* hydrogen number dens in cgs units */
    double LambdaNet = CoolingRateFromU(u_old, rho, ne_guess, target);
    if(LambdaNet >= 0) {return 0;} /* net heating due to UV background */
    return u_old / (-(nHcgs * nHcgs / rho) * LambdaNet) * All.HubbleParam / All.UnitTime_in_s;
#endif
}


/* returns new internal energy per unit mass. 
 * Arguments are passed in code units, density is proper density.
 */
double DoInstabilityCooling(double m_old, double u, double rho, double dt, double fac, double ne_guess, int target)
{
    if(fac <= 0) {return 0.01*m_old;} /* the hot phase is actually colder than the cold reservoir! */
    double m, dm, m_lower, m_upper, ratefact, LambdaNet;
    int iter = 0;
    
    rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;	/* convert to physical cgs units */
    u *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
    dt *= All.UnitTime_in_s / All.HubbleParam;
    fac *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;
    double nHcgs = HYDROGEN_MASSFRAC * rho / PROTONMASS;	/* hydrogen number dens in cgs units */
    ratefact = nHcgs * nHcgs / rho * fac;
    m = m_old; m_lower = m; m_upper = m;
    LambdaNet = CoolingRateFromU(u, rho, ne_guess, target);
    
    /* bracketing */
    if(m - m_old - m * m / m_old * ratefact * LambdaNet * dt < 0)	/* heating */
    {
        m_upper *= sqrt(1.1); m_lower /= sqrt(1.1);
        while(m_upper - m_old - m_upper * m_upper / m_old * ratefact * CoolingRateFromU(u, rho * m_upper / m_old, ne_guess, target) * dt < 0)
        {
            m_upper *= 1.1; m_lower *= 1.1;
        }
    }
    if(m - m_old - m_old * ratefact * LambdaNet * dt > 0)
    {
        m_lower /= sqrt(1.1); m_upper *= sqrt(1.1);
        while(m_lower - m_old - m_lower * m_lower / m_old * ratefact * CoolingRateFromU(u, rho * m_lower / m_old, ne_guess, target) * dt > 0)
        {
            m_upper /= 1.1; m_lower /= 1.1;
        }
    }
    
    do
    {
        m = 0.5 * (m_lower + m_upper);
        LambdaNet = CoolingRateFromU(u, rho * m / m_old, ne_guess, target);
        if(m - m_old - m * m / m_old * ratefact * LambdaNet * dt > 0) {m_upper = m;} else {m_lower = m;}
        dm = m_upper - m_lower;
        iter++;
        if(iter >= (MAXITER - 10)) {printf("->m= %g\n", m);}
    }
    while(fabs(dm / m) > 1.0e-6 && iter < MAXITER);
    if(iter >= MAXITER) {printf("failed to converge in DoInstabilityCooling(): m_in=%g u_in=%g rho=%g dt=%g fac=%g ne_in=%g target=%d \n",m_old,u,rho,dt,fac,ne_guess,target); endrun(11);}
    return m;
}



void cool_test(void)
{
#if !defined(COOL_METAL_LINES_BY_SPECIES)
    double uin, rhoin, tempin, muin, nein;
    uin = 6.01329e+09;
    rhoin = 7.85767e-29;
    tempin = 2034.0025;
    muin = 0.691955;
    nein = (1 + 4 * YHELIUM_0) / muin - (1 + YHELIUM_0);
    double dtin=1.0e-7;
    double uout,uint,nH0_guess,nHp_guess,nHe0_guess,nHep_guess,nHepp_guess;
    int i,target;
    for(i=0;i<20;i++) {
        rhoin=SphP[i].Density;
        nein=SphP[i].Ne;
        target=i;
        uin=SphP[i].InternalEnergy;
        uout=DoCooling(uin,rhoin,dtin,nein,target);
        printf("%d %d : ne: %g %g \n",ThisTask,target,SphP[i].Ne,nein);
        nein=SphP[i].Ne;
        rhoin *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;    /* convert to physical cgs units */
        uint = uin*All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
        tempin=convert_u_to_temp(uint, rhoin, target, &nein, &nH0_guess, &nHp_guess, &nHe0_guess, &nHep_guess, &nHepp_guess);
        printf("%d %d : in: : %g %g %g \n",ThisTask,target,uin,rhoin,nein);
        printf("%d %d : out: %g %g %g %g %g \n",ThisTask,target,tempin,
               CoolingRate(log10(tempin),rhoin,nein,target),CoolingRateFromU(uint,rhoin,nein,target),uout,nein);
        fflush(stdout);
    }
#endif
}



double get_mu(double T_guess, double rho, double *ne_guess, int target)
{
    double X=HYDROGEN_MASSFRAC, Y=1.-X, Z=0, fmol;
    
#ifdef METALS
    if(target >= 0)
    {
        Z = DMIN(0.25,P[target].Metallicity[0]);
        if(NUM_METAL_SPECIES>=10) {Y = DMIN(0.35,P[target].Metallicity[1]);}
        X = 1. - (Y+Z);
    }
#endif
    
    double T_mol = 100.; // temperature below which gas at a given density becomes molecular, from Glover+Clark 2012
    if(rho > 0) {T_mol *= (rho/PROTONMASS) / 100.;}
    if(T_mol>8000.) {T_mol=8000.;}
    T_mol = T_guess / T_mol;
    fmol = 1. / (1. + T_mol*T_mol);
    
    return 1. / ( X*(1-0.5*fmol) + Y/4. + *ne_guess*HYDROGEN_MASSFRAC + Z/(16.+12.*fmol) ); // since our ne is defined in some routines with He, should multiply by universal
}


double yhelium(int target)
{
#ifdef COOL_METAL_LINES_BY_SPECIES
    if(target >= 0) {double ytmp=DMIN(0.5,P[target].Metallicity[1]); return 0.25*ytmp/(1.-ytmp);} else {return YHELIUM_0;}
#else
    return YHELIUM_0;
#endif
}



/* this function determines the electron fraction, and hence the mean molecular weight. With it arrives at a self-consistent temperature.
 * Ionization abundances and the rates for the emission are also computed */
double convert_u_to_temp(double u, double rho, int target, double *ne_guess, double *nH0_guess, double *nHp_guess, double *nHe0_guess, double *nHep_guess, double *nHepp_guess)
{
    int iter = 0;
    double temp, temp_old, temp_new, max = 0, ne_old, mu;
    double u_input = u, rho_input = rho, temp_guess = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS;
    mu = get_mu(temp_guess, rho, ne_guess, target);
    temp = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS * mu;
    
    do
    {
        ne_old = *ne_guess;
        find_abundances_and_rates(log10(temp), rho, target, -1, 0, ne_guess, nH0_guess, nHp_guess, nHe0_guess, nHep_guess, nHepp_guess);
        temp_old = temp;
        
        mu = get_mu(temp, rho, ne_guess, target);
        temp_new = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS * mu;
        
        max = DMAX(max, temp_new * mu * HYDROGEN_MASSFRAC * fabs((*ne_guess - ne_old) / (temp_new - temp_old + 1.0)));
        temp = temp_old + (temp_new - temp_old) / (1 + max);
        iter++;
        if(iter > (MAXITER - 10)) {printf("-> temp=%g/%g/%g ne=%g/%g mu=%g rho=%g max=%g iter=%d target=%d \n", temp,temp_new,temp_old,*ne_guess,ne_old, mu ,rho,max,iter,target);}
    }
    while(
          ((fabs(temp - temp_old) > 0.25 * temp) ||
           ((fabs(temp - temp_old) > 0.1 * temp) && (temp > 20.)) ||
           ((fabs(temp - temp_old) > 0.05 * temp) && (temp > 200.)) ||
           ((fabs(temp - temp_old) > 0.01 * temp) && (temp > 200.) && (iter<100)) ||
           ((fabs(temp - temp_old) > 1.0e-3 * temp) && (temp > 200.) && (iter<10))) && iter < MAXITER);
    
    if(iter >= MAXITER) {printf("failed to converge in convert_u_to_temp(): u_input= %g rho_input=%g n_elec_input=%g target=%d\n", u_input, rho_input, *ne_guess, target); endrun(12);}

    if(temp<=0) temp=pow(10.0,Tmin);
    if(log10(temp)<Tmin) temp=pow(10.0,Tmin);
    return temp;
}



/* this function computes the actual ionization states, relative abundances, and returns the ionization/recombination rates if needed */
double find_abundances_and_rates(double logT, double rho, int target, double shieldfac, int return_cooling_mode,
                                 double *ne_guess, double *nH0_guess, double *nHp_guess, double *nHe0_guess, double *nHep_guess, double *nHepp_guess)
{
    int j, niter;
    double Tlow, Thi, flow, fhi, t, gJH0ne, gJHe0ne, gJHepne, logT_input, rho_input, ne_input, neold, nenew;
    double bH0, bHep, bff, aHp, aHep, aHepp, ad, geH0, geHe0, geHep;
    double n_elec, nH0, nHe0, nHp, nHep, nHepp; /* ionization states */
    logT_input = logT; rho_input = rho; ne_input = *ne_guess; /* save inputs (in case of failed convergence below) */
    if(isnan(logT)) logT=Tmin;    /* nan trap (just in case) */
    
    if(logT <= Tmin)		/* everything neutral */
    {
        nH0 = 1.0; nHe0 = yhelium(target); nHp = 0; nHep = 0; nHepp = 0; n_elec = 0;
        *nH0_guess=nH0; *nHe0_guess=nHe0; *nHp_guess=nHp; *nHep_guess=nHep; *nHepp_guess=nHepp; *ne_guess=n_elec;
        return 0;
    }
    if(logT >= Tmax)		/* everything is ionized */
    {
        nH0 = 0; nHe0 = 0; nHp = 1.0; nHep = 0; nHepp = yhelium(target); n_elec = nHp + 2.0 * nHepp;
        *nH0_guess=nH0; *nHe0_guess=nHe0; *nHp_guess=nHp; *nHep_guess=nHep; *nHepp_guess=nHepp; *ne_guess=n_elec;
        return 0;
    }

    /* initialize quantities needed for iteration below */
    t = (logT - Tmin) / deltaT;
    j = (int) t;
    if(j<0){j=0;}
    if(j>NCOOLTAB){
#ifndef IO_REDUCED_MODE
        printf("warning: j>NCOOLTAB : j=%d t %g Tlow %g Thi %g logT %g Tmin %g deltaT %g \n",j,t,Tmin+deltaT*j,Tmin+deltaT*(j+1),logT,Tmin,deltaT);fflush(stdout);
#endif
        j=NCOOLTAB;
    }
    Tlow = Tmin + deltaT * j;
    Thi = Tlow + deltaT;
    fhi = t - j;
    flow = 1 - fhi;
    if(*ne_guess == 0) /* no guess provided, try to start from something sensible */
    {
        *ne_guess = 1.0;
        if(logT < 3.8) {*ne_guess = 0.1;}
        if(logT < 2) {*ne_guess = 1.e-10;}
    }

    /* account for non-local UV background */
    double local_gammamultiplier=1;
#ifdef GALSF_FB_LOCAL_UV_HEATING
    if((target >= 0) && (gJH0 > 0))
    {
        local_gammamultiplier = SphP[target].RadFluxEUV * 2.29e-10; // converts to GammaHI for typical SED (rad_uv normalized to Habing)
        local_gammamultiplier = 1 + local_gammamultiplier/gJH0;
    }
#endif
    /* CAFG: this is the density that we should use for UV background threshold */
    double nHcgs = HYDROGEN_MASSFRAC * rho / PROTONMASS;	/* hydrogen number dens in cgs units */
    if(shieldfac < 0)
    {
        double NH_SS_z;
        if(gJH0>0)
            NH_SS_z = NH_SS*pow(local_gammamultiplier*gJH0/1.0e-12,0.66)*pow(10.,0.173*(logT-4.));
        else
            NH_SS_z = NH_SS*pow(10.,0.173*(logT-4.));
        double q_SS = nHcgs/NH_SS_z;
        shieldfac = 1./(1.+q_SS*(1.+q_SS/2.*(1.+q_SS/3.*(1.+q_SS/4.*(1.+q_SS/5.*(1.+q_SS/6.*q_SS))))));
#ifdef COOL_LOW_TEMPERATURES
        if(logT < Tmin+1) shieldfac *= (logT-Tmin); // make cutoff towards Tmin more continuous //
#endif
#ifdef GALSF_EFFECTIVE_EQS
        shieldfac = 1; // self-shielding is implicit in the sub-grid model already //
#endif
    }
    n_elec = *ne_guess; neold = n_elec; niter = 0;
    double necgs = n_elec * nHcgs; /* more initialized quantities */
    
#if defined(RT_CHEM_PHOTOION)
    double c_light_ne=0, Sigma_particle=0, abs_per_kappa_dt=0;
    if(target >= 0)
    {
        double L_particle = Get_Particle_Size(target)*All.cf_atime; // particle effective size/slab thickness
        double dt = (P[target].TimeBin ? (1 << P[target].TimeBin) : 0) * All.Timebase_interval / All.cf_hubble_a; // dtime [code units]
        double cx_to_kappa = HYDROGEN_MASSFRAC / PROTONMASS * All.UnitMass_in_g / All.HubbleParam; // pre-factor for converting cross sections into opacities
        Sigma_particle = cx_to_kappa * P[target].Mass / (M_PI*L_particle*L_particle); // effective surface density through particle
        abs_per_kappa_dt = cx_to_kappa * RT_SPEEDOFLIGHT_REDUCTION * (C/All.UnitVelocity_in_cm_per_s) * (SphP[target].Density*All.cf_a3inv) * dt; // fractional absorption over timestep
        nH0 = SphP[target].HI; // need to initialize a value for the iteration below
#ifdef RT_CHEM_PHOTOION_HE
        nHe0 = SphP[target].HeI; nHep = SphP[target].HeII; // need to intialize a value for the iteration below
#endif
    }
#endif
    
    /* evaluate number densities iteratively (cf KWH eqns 33-38) in units of nH */
    do
    {
        niter++;
        
        aHp = flow * AlphaHp[j] + fhi * AlphaHp[j + 1];
        aHep = flow * AlphaHep[j] + fhi * AlphaHep[j + 1];
        aHepp = flow * AlphaHepp[j] + fhi * AlphaHepp[j + 1];
        ad = flow * Alphad[j] + fhi * Alphad[j + 1];
        geH0 = flow * GammaeH0[j] + fhi * GammaeH0[j + 1];
        geHe0 = flow * GammaeHe0[j] + fhi * GammaeHe0[j + 1];
        geHep = flow * GammaeHep[j] + fhi * GammaeHep[j + 1];
#ifdef COOL_LOW_TEMPERATURES
        // make cutoff towards Tmin more continuous //
        if(logT < Tmin+1) {
            geH0 *= (logT-Tmin);
            geHe0 *= (logT-Tmin);
            geHep *= (logT-Tmin);
        }
#endif
        
        if(necgs <= 1.e-25 || J_UV == 0)
        {
            gJH0ne = gJHe0ne = gJHepne = 0;
        }
        else
        {
            /* account for self-shielding in calculating UV background effects */
            gJH0ne = gJH0 * local_gammamultiplier / necgs * shieldfac; // check units, should be = c_light * n_photons_vol * rt_sigma_HI[0] / necgs;
            gJHe0ne = gJHe0 * local_gammamultiplier / necgs * shieldfac;
            gJHepne = gJHep * local_gammamultiplier / necgs * shieldfac;
        }
#if defined(RT_DISABLE_UV_BACKGROUND)
        gJH0ne = gJHe0ne = gJHepne = 0;
#endif
#if defined(RT_CHEM_PHOTOION)
        /* add in photons from explicit radiative transfer (on top of assumed background) */
        if((necgs > 1.e-25)&&(target >= 0))
        {
            int k;
            c_light_ne = C / ((MIN_REAL_NUMBER + necgs) * All.UnitLength_in_cm / All.HubbleParam); // want physical cgs units for quantities below
            double gJH0ne_0=gJH0 * local_gammamultiplier / necgs, gJHe0ne_0=gJHe0 * local_gammamultiplier / necgs, gJHepne_0=gJHep * local_gammamultiplier / necgs; // need a baseline, so we don't over-shoot below
#if defined(RT_DISABLE_UV_BACKGROUND)
            gJH0ne_0=gJHe0ne_0=gJHepne_0=MAX_REAL_NUMBER;
#endif
            for(k = 0; k < N_RT_FREQ_BINS; k++)
            {
                if((k==RT_FREQ_BIN_H0)||(k==RT_FREQ_BIN_He0)||(k==RT_FREQ_BIN_He1)||(k==RT_FREQ_BIN_He2))
                {
                    double c_ne_time_n_photons_vol = c_light_ne * rt_return_photon_number_density(target,k); // gives photon flux
                    double cross_section_ion, dummy, thold=1.0e20;
#ifdef GALSF
                    if(All.ComovingIntegrationOn) {thold=1.0e10;}
#endif
                    if(G_HI[k] > 0)
                    {
                        cross_section_ion = nH0 * rt_sigma_HI[k];
                        dummy = rt_sigma_HI[k] * c_ne_time_n_photons_vol * slab_averaging_function(cross_section_ion * Sigma_particle); // egy per photon x cross section x photon flux (w attenuation factors)
                        // * slab_averaging_function(cross_section_ion * abs_per_kappa_dt);
                        if(dummy > thold*gJH0ne_0) {dummy = thold*gJH0ne_0;}
                        gJH0ne += dummy;
                    }
#ifdef RT_CHEM_PHOTOION_HE
                    if(G_HeI[k] > 0)
                    {
                        cross_section_ion = nHe0 * rt_sigma_HeI[k];
                        dummy = rt_sigma_HeI[k] * c_ne_time_n_photons_vol * slab_averaging_function(cross_section_ion * Sigma_particle); // egy per photon x cross section x photon flux (w attenuation factors)
                        // * slab_averaging_function(cross_section_ion * abs_per_kappa_dt);
                        if(dummy > thold*gJHe0ne_0) {dummy = thold*gJHe0ne_0;}
                        gJHe0ne += dummy;
                    }
                    if(G_HeII[k] > 0)
                    {
                        cross_section_ion = nHep * rt_sigma_HeII[k];
                        dummy = rt_sigma_HeII[k] * c_ne_time_n_photons_vol * slab_averaging_function(cross_section_ion * Sigma_particle); // egy per photon x cross section x photon flux (w attenuation factors)
                        // * slab_averaging_function(cross_section_ion * abs_per_kappa_dt);
                        if(dummy > thold*gJHepne_0) {dummy = thold*gJHepne_0;}
                        gJHepne += dummy;
                    }
#endif
                }
            }
        }
#endif
        
        
        nH0 = aHp / (aHp + geH0 + gJH0ne);	/* eqn (33) */
        nHp = 1.0 - nH0;		/* eqn (34) */
        
        if((gJHe0ne + geHe0) <= 1.0e-60)	/* no ionization at all */
        {
            nHep = 0.0;
            nHepp = 0.0;
            nHe0 = yhelium(target);
        }
        else
        {
            nHep = yhelium(target) / (1.0 + (aHep + ad) / (geHe0 + gJHe0ne) + (geHep + gJHepne) / aHepp);	/* eqn (35) */
            nHe0 = nHep * (aHep + ad) / (geHe0 + gJHe0ne);	/* eqn (36) */
            nHepp = nHep * (geHep + gJHepne) / aHepp;	/* eqn (37) */
        }
        
        neold = n_elec;
        n_elec = nHp + nHep + 2 * nHepp;	/* eqn (38) */
        necgs = n_elec * nHcgs;
        
        if(J_UV == 0) break;
        
        nenew = 0.5 * (n_elec + neold);
        n_elec = nenew;
        necgs = n_elec * nHcgs;
        
        double dneTHhold = DMAX(n_elec*0.01 , 1.0e-4);
        if(fabs(n_elec - neold) < dneTHhold) break;
        
        if(niter > (MAXITER - 10)) {printf("n_elec= %g/%g/%g yh=%g nHcgs=%g niter=%d\n", n_elec,neold,nenew, yhelium(target), nHcgs, niter);}
    }
    while(niter < MAXITER);
    
    if(niter >= MAXITER) {printf("failed to converge in find_abundances_and_rates(): logT_input=%g  rho_input=%g  ne_input=%g target=%d shieldfac=%g cooling_return=%d", logT_input, rho_input, ne_input, target, shieldfac, return_cooling_mode); endrun(13);}
    
    bH0 = flow * BetaH0[j] + fhi * BetaH0[j + 1];
    bHep = flow * BetaHep[j] + fhi * BetaHep[j + 1];
    bff = flow * Betaff[j] + fhi * Betaff[j + 1];
    if(target >= 0)
    {
        SphP[target].Ne = n_elec;
#ifdef RT_CHEM_PHOTOION
        SphP[target].HI = nH0;
        SphP[target].HII = nHp;
#ifdef RT_CHEM_PHOTOION_HE
        SphP[target].HeI = nHe0;
        SphP[target].HeII = nHep;
        SphP[target].HeIII = nHepp;
#endif
#endif
    }
    *nH0_guess=nH0; *nHe0_guess=nHe0; *nHp_guess=nHp; *nHep_guess=nHep; *nHepp_guess=nHepp; *ne_guess=n_elec; /* write to send back */
    
    /* now check if we want to return the ionization/recombination heating/cooling rates calculated with all the above quantities */
    if(return_cooling_mode==1)
    {
        /* Compute cooling and heating rate (cf KWH Table 1) in units of nH**2 */
        double LambdaExcH0 = bH0 * n_elec * nH0;
        double LambdaExcHep = bHep * n_elec * nHep;
        double LambdaExc = LambdaExcH0 + LambdaExcHep;	/* collisional excitation */
        
        double LambdaIonH0 = 2.18e-11 * geH0 * n_elec * nH0;
        double LambdaIonHe0 = 3.94e-11 * geHe0 * n_elec * nHe0;
        double LambdaIonHep = 8.72e-11 * geHep * n_elec * nHep;
        double LambdaIon = LambdaIonH0 + LambdaIonHe0 + LambdaIonHep;	/* collisional ionization */
        
        double T_lin = pow(10.0, logT);
        double LambdaRecHp = 1.036e-16 * T_lin * n_elec * (aHp * nHp);
        double LambdaRecHep = 1.036e-16 * T_lin * n_elec * (aHep * nHep);
        double LambdaRecHepp = 1.036e-16 * T_lin * n_elec * (aHepp * nHepp);
        double LambdaRecHepd = 6.526e-11 * ad * n_elec * nHep;
        double LambdaRec = LambdaRecHp + LambdaRecHep + LambdaRecHepp + LambdaRecHepd; /* recombination */
        
        double LambdaFF = bff * (nHp + nHep + 4 * nHepp) * n_elec; /* free-free (Bremsstrahlung) */
        
        double Lambda = LambdaExc + LambdaIon + LambdaRec + LambdaFF; /* sum all of the above */
        return Lambda; /* send it back */
    }
    return 0;
} // end of find_abundances_and_rates() //




/*  this function first computes the self-consistent temperature and abundance ratios, and then it calculates (heating rate-cooling rate)/n_h^2 in cgs units */
double CoolingRateFromU(double u, double rho, double ne_guess, int target)
{
    double nH0_guess, nHp_guess, nHe0_guess, nHep_guess, nHepp_guess;
    double temp = convert_u_to_temp(u, rho, target, &ne_guess, &nH0_guess, &nHp_guess, &nHe0_guess, &nHep_guess, &nHepp_guess);
    return CoolingRate(log10(temp), rho, ne_guess, target);
}


/*  this function computes the self-consistent temperature and electron fraction */ 
double ThermalProperties(double u, double rho, int target, double *mu_guess, double *ne_guess, double *nH0_guess, double *nHp_guess, double *nHe0_guess, double *nHep_guess, double *nHepp_guess)
{
    double temp;
    rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;	/* convert to physical cgs units */
    u *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
    temp = convert_u_to_temp(u, rho, target, ne_guess, nH0_guess, nHp_guess, nHe0_guess, nHep_guess, nHepp_guess);
#ifdef GALSF_FB_HII_HEATING
    if(target >= 0) {if(SphP[target].DelayTimeHII > 0) {SphP[target].Ne = 1.0 + 2.0*yhelium(target);}} /* fully ionized */
#endif
    *mu_guess = get_mu(temp, rho, ne_guess, target);
    return temp;
}




extern FILE *fd;




/*  Calculates (heating rate-cooling rate)/n_h^2 in cgs units 
 */
double CoolingRate(double logT, double rho, double n_elec_guess, int target)
{
    double n_elec=n_elec_guess, nH0, nHe0, nHp, nHep, nHepp; /* ionization states [computed below] */
    double Lambda, Heat, LambdaExc, LambdaIon, LambdaRec, LambdaFF, LambdaCmptn, LambdaExcH0, LambdaExcHep, LambdaIonH0, LambdaIonHe0, LambdaIonHep;
    double LambdaRecHp, LambdaRecHep, LambdaRecHepp, LambdaRecHepd, redshift, T, NH_SS_z, shieldfac, LambdaMol, LambdaMetal;
    double nHcgs = HYDROGEN_MASSFRAC * rho / PROTONMASS;	/* hydrogen number dens in cgs units */
    LambdaMol=0; LambdaMetal=0; LambdaCmptn=0; NH_SS_z=NH_SS;
    if(logT <= Tmin) {logT = Tmin + 0.5 * deltaT;}	/* floor at Tmin */
#ifdef COOL_METAL_LINES_BY_SPECIES
    double *Z;
    if(target>=0)
    {
        Z = P[target].Metallicity;
    } else {
        /* initialize dummy values here so the function doesn't crash, if called when there isn't a target particle */
        int k;
        double Zsol[NUM_METAL_SPECIES];
        for(k=0;k<NUM_METAL_SPECIES;k++) Zsol[k]=All.SolarAbundances[k];
        Z = Zsol;
    }
#endif
    double local_gammamultiplier=1;
#ifdef GALSF_FB_LOCAL_UV_HEATING
    if((target >= 0) && (gJH0 > 0))
    {
        local_gammamultiplier = SphP[target].RadFluxEUV * 2.29e-10; // converts to GammaHI for typical SED (rad_uv normalized to Habing)
        local_gammamultiplier = 1 + local_gammamultiplier/gJH0;
    }
#endif
    
    /* CAFG: if density exceeds NH_SS, ignore ionizing background. */
    if(J_UV != 0)
        NH_SS_z=NH_SS*pow(local_gammamultiplier*gJH0/1.0e-12,0.66)*pow(10.,0.173*(logT-4.));
    else
        NH_SS_z=NH_SS*pow(10.,0.173*(logT-4.));
    double q_SS = nHcgs/NH_SS_z;
    shieldfac = 1./(1.+q_SS*(1.+q_SS/2.*(1.+q_SS/3.*(1.+q_SS/4.*(1.+q_SS/5.*(1.+q_SS/6.*q_SS))))));
#ifdef GALSF_EFFECTIVE_EQS
    shieldfac = 1; // self-shielding is implicit in the sub-grid model already //
#endif
    
#ifdef BH_COMPTON_HEATING
    double AGN_LambdaPre,AGN_T_Compton;
    AGN_T_Compton = 2.0e7; /* approximate from Sazonov et al. */
    if(target < 0) {
        AGN_LambdaPre = 0;
    } else {
        AGN_LambdaPre = SphP[target].RadFluxAGN * (3.9/2.0) * All.UnitMass_in_g/(All.UnitLength_in_cm*All.UnitLength_in_cm)*All.HubbleParam*All.cf_a2inv; /* proper units */
#ifdef SINGLE_STAR_FORMATION
        /* here we are hijacking this module to approximate dust heating/cooling */
        /* assuming heating/cooling balance defines the target temperature: */
        AGN_T_Compton = pow( 1.0e4 + AGN_LambdaPre / 5.67e-5 , 0.25); // (sigma*T^4 = Flux_incident)
        if(AGN_T_Compton < Tmin) {AGN_T_Compton=Tmin;}
#else
        /* now have incident flux, need to convert to relevant pre-factor for heating rate */
        AGN_LambdaPre *= 6.652e-25; /* sigma_T for absorption */
        AGN_LambdaPre *= (4.*1.381e-16)/(9.109e-28*2.998e10*2.998e10); /* times 4*k_B/(me*c^2) */
#endif
    }
#endif

#if defined(RT_CHEM_PHOTOION) || defined(RT_PHOTOELECTRIC)
    double Sigma_particle = 0, abs_per_kappa_dt = 0, cx_to_kappa = 0;
    if(target >= 0)
    {
        double L_particle = Get_Particle_Size(target)*All.cf_atime; // particle effective size/slab thickness
        double dt = (P[target].TimeBin ? (1 << P[target].TimeBin) : 0) * All.Timebase_interval / All.cf_hubble_a; // dtime [code units]
        Sigma_particle = P[target].Mass / (M_PI*L_particle*L_particle); // effective surface density through particle
        abs_per_kappa_dt = RT_SPEEDOFLIGHT_REDUCTION * (C/All.UnitVelocity_in_cm_per_s) * (SphP[target].Density*All.cf_a3inv) * dt; // fractional absorption over timestep
        cx_to_kappa = HYDROGEN_MASSFRAC / PROTONMASS * All.UnitMass_in_g / All.HubbleParam; // pre-factor for converting cross sections into opacities
    }
#endif

    
    T = pow(10.0, logT);
    if(logT < Tmax)
    {
        /* get ionization states for H and He with associated ionization, collision, recombination, and free-free heating/cooling */
        Lambda = find_abundances_and_rates(logT, rho, target, shieldfac, 1, &n_elec, &nH0, &nHp, &nHe0, &nHep, &nHepp);
        
#ifdef COOL_METAL_LINES_BY_SPECIES
        /* can restrict to low-densities where not self-shielded, but let shieldfac (in ne) take care of this self-consistently */
        if((J_UV != 0)&&(logT > Tmin+0.5*deltaT)&&(logT > 4.00))
        {
            /* cooling rates tabulated for each species from Wiersma, Schaye, & Smith tables (2008) */
            LambdaMetal = GetCoolingRateWSpecies(nHcgs, logT, Z); //* nHcgs*nHcgs;
            /* tables normalized so ne*ni/(nH*nH) included already, so just multiply by nH^2 */
            /* (sorry, -- dont -- multiply by nH^2 here b/c that's how everything is normalized in this function) */
            LambdaMetal *= n_elec;
            /* (modified now to correct out tabulated ne so that calculated ne can be inserted; ni not used b/c it should vary species-to-species */
            Lambda += LambdaMetal;
        }
#endif
        
#ifdef COOL_LOW_TEMPERATURES
        if((logT <= 5.2)&&(logT > Tmin+0.5*deltaT))
        {
            /* approx to cooling function for solar metallicity and nH=1 cm^(-3) -- want to do something
             much better, definitely, but for now use this just to get some idea of system with cooling to very low-temp */
            LambdaMol = 2.8958629e-26/(pow(T/125.21547,-4.9201887)+pow(T/1349.8649,-1.7287826)+pow(T/6450.0636,-0.30749082));//*nHcgs*nHcgs;
            LambdaMol *= (1-shieldfac);
	        LambdaMol *= 1./(1. + nHcgs/700.); // above the critical density, cooling rate suppressed by ~1/n; use critical density of CO[J(1-0)] as a proxy for this
            double LambdaDust = 0;
#ifdef COOL_METAL_LINES_BY_SPECIES
            LambdaMol *= (1+Z[0]/All.SolarAbundances[0])*(0.001 + 0.1*nHcgs/(1.0+nHcgs)
                            + 0.09*nHcgs/(1.0+0.1*nHcgs)
                            + (Z[0]/All.SolarAbundances[0])*(Z[0]/All.SolarAbundances[0])/(1.0+nHcgs));
            /* add dust cooling as well */
            double Tdust = 30.;
#if defined(SINGLE_STAR_FORMATION) && defined(BH_COMPTON_HEATING)
            Tdust = AGN_T_Compton;
#endif
#ifdef RT_INFRARED
            if(target >= 0) {Tdust = SphP[target].Dust_Temperature;}
#endif
            if(T > Tdust) {LambdaDust = 1.116e-32 * (T-Tdust)*sqrt(T)*(1.-0.8*exp(-75./T)) * (Z[0]/All.SolarAbundances[0]);}  // Meijerink & Spaans 2005; Hollenbach & McKee 1979,1989 //
#endif
            Lambda += LambdaMol + LambdaDust;
        }
#endif
        
        
        if(All.ComovingIntegrationOn)
        {
            redshift = 1 / All.Time - 1;
            LambdaCmptn = 5.65e-36 * n_elec * (T - 2.73 * (1. + redshift)) * pow(1. + redshift, 4.) / nHcgs;
            Lambda += LambdaCmptn;
        }
        else {LambdaCmptn = 0;}

#if defined(BH_COMPTON_HEATING) && !defined(SINGLE_STAR_FORMATION)
        if(T > AGN_T_Compton)
        {
            LambdaCmptn = AGN_LambdaPre * (T - AGN_T_Compton) * n_elec/nHcgs;
            if(LambdaCmptn > 2.19e-21/sqrt(T/1.0e8)) LambdaCmptn=2.19e-21/sqrt(T/1.0e8);
            Lambda += LambdaCmptn;
        }
#endif
        
        
        Heat = 0;  /* Now, collect heating terms */


        if(J_UV != 0) {Heat += local_gammamultiplier * (nH0 * epsH0 + nHe0 * epsHe0 + nHep * epsHep) / nHcgs * shieldfac;} // shieldfac allows for self-shielding from background
#if defined(RT_DISABLE_UV_BACKGROUND)
        Heat = 0;
#endif
#if defined(RT_CHEM_PHOTOION)
        /* add in photons from explicit radiative transfer (on top of assumed background) */
        if((target >= 0) && (nHcgs > MIN_REAL_NUMBER))
        {
            int k; double c_light_nH = C / (nHcgs * All.UnitLength_in_cm / All.HubbleParam) * All.UnitEnergy_in_cgs / All.HubbleParam; // want physical cgs units for quantities below
            for(k = 0; k < N_RT_FREQ_BINS; k++)
            {
                if((k==RT_FREQ_BIN_H0)||(k==RT_FREQ_BIN_He0)||(k==RT_FREQ_BIN_He1)||(k==RT_FREQ_BIN_He2))
                {
                    double c_nH_time_n_photons_vol = c_light_nH * rt_return_photon_number_density(target,k); // gives photon flux
                    double cross_section_ion, kappa_ion, dummy;
                    if(G_HI[k] > 0)
                    {
                        cross_section_ion = nH0 * rt_sigma_HI[k];
                        kappa_ion = cx_to_kappa * cross_section_ion;
                        dummy = G_HI[k] * cross_section_ion * c_nH_time_n_photons_vol * slab_averaging_function(kappa_ion * Sigma_particle); // egy per photon x cross section x photon flux (w attenuation factors)
                        // * slab_averaging_function(kappa_ion * abs_per_kappa_dt);
                        Heat += dummy;
                    }
                    if(G_HeI[k] > 0)
                    {
                        cross_section_ion = nHe0 * rt_sigma_HeI[k];
                        kappa_ion = cx_to_kappa * cross_section_ion;
                        dummy = G_HeI[k] * cross_section_ion * c_nH_time_n_photons_vol * slab_averaging_function(kappa_ion * Sigma_particle);
                        // * slab_averaging_function(kappa_ion * abs_per_kappa_dt);
                        Heat += dummy;
                    }
                    if(G_HeII[k] > 0)
                    {
                        cross_section_ion = nHep * rt_sigma_HeII[k];
                        kappa_ion = cx_to_kappa * cross_section_ion;
                        dummy = G_HeII[k] * cross_section_ion * c_nH_time_n_photons_vol * slab_averaging_function(kappa_ion*Sigma_particle);
                        // * slab_averaging_function(kappa_ion * abs_per_kappa_dt);
                        Heat += dummy;
                    }
                }
            }
        }
#endif
        

#if defined(COSMIC_RAYS) && !defined(COSMIC_RAYS_DISABLE_COOLING)
        /* cosmic ray heating, from Guo & Oh 2008: this scales proportional to the electron number density and
         cosmic ray energy density, both of which we quickly evaluate here (make sure we convert to the correct per-atom units)
         - note that only 1/6 of the hadronic cooling is thermalized, according to their calculation, while all the Coulomb losses heat */
        if(SphP[target].CosmicRayEnergyPred > 0)
        {
            Heat += 1.0e-16 * (0.98 + 1.65*n_elec*HYDROGEN_MASSFRAC) / nHcgs *
                ((SphP[target].CosmicRayEnergyPred / P[target].Mass * SphP[target].Density * All.cf_a3inv) *
                 (All.UnitPressure_in_cgs * All.HubbleParam * All.HubbleParam));
        }
#else 
#ifdef COOL_LOW_TEMPERATURES
        /* if COSMIC_RAYS is not enabled, but low-temperature cooling is on, we account for the CRs as a heating source using
         a more approximate expression (assuming the mean background of the Milky Way clouds) */
        if(logT <= 5.2) {Heat += 1.0e-16 * (0.98 + 1.65*n_elec*HYDROGEN_MASSFRAC) / (1.e-2 + nHcgs) * 9.0e-12;} // multiplied by background of ~5eV/cm^3 (Goldsmith & Langer (1978),  van Dishoeck & Black (1986) //
#endif
#endif
      
        
#if defined(COOL_METAL_LINES_BY_SPECIES) && defined(COOL_LOW_TEMPERATURES)
        /* Dust collisional heating */
        double Tdust = 30.;
#if defined(SINGLE_STAR_FORMATION) && defined(BH_COMPTON_HEATING)
        Tdust = AGN_T_Compton;
#endif
#ifdef RT_INFRARED
        if(target >= 0) {Tdust = SphP[target].Dust_Temperature;}
#endif
        if(T < Tdust) {Heat += 1.116e-32 * (Tdust-T)*sqrt(T)*(1.-0.8*exp(-75./T)) * (Z[0]/All.SolarAbundances[0]);} // Meijerink & Spaans 2005; Hollenbach & McKee 1979,1989 //
#endif
        
#if defined(BH_COMPTON_HEATING) && !defined(SINGLE_STAR_FORMATION)
        /* Compton heating from AGN */
        if(T < AGN_T_Compton) Heat += AGN_LambdaPre * (AGN_T_Compton - T) / nHcgs; /* note this is independent of the free electron fraction */
#endif
        
#if defined(GALSF_FB_LOCAL_UV_HEATING) || defined(RT_PHOTOELECTRIC)
        /* Photoelectric heating following Bakes & Thielens 1994 (also Wolfire 1995); now with 'update' from Wolfire 2005 for PAH [fudge factor 0.5 below] */
        if((target >= 0) && (T < 1.0e6))
        {
#ifdef GALSF_FB_LOCAL_UV_HEATING
            double photoelec = SphP[target].RadFluxUV;
#endif
#ifdef RT_PHOTOELECTRIC
            double photoelec = SphP[target].E_gamma[RT_FREQ_BIN_PHOTOELECTRIC] * (SphP[target].Density*All.cf_a3inv/P[target].Mass) * All.UnitPressure_in_cgs * All.HubbleParam*All.HubbleParam / 3.9e-14; // convert to Habing field //
            if(photoelec > 0)
            {
                photoelec *= slab_averaging_function(SphP[target].Kappa_RT[RT_FREQ_BIN_PHOTOELECTRIC] * Sigma_particle);
                // * slab_averaging_function(SphP[target].Kappa_RT[RT_FREQ_BIN_PHOTOELECTRIC] * abs_per_kappa_dt);
                if(photoelec > 1.0e4) {photoelec = 1.e4;}
            }
#endif
            if(photoelec > 0)
            {
                double LambdaPElec = 1.3e-24 * photoelec / nHcgs * P[target].Metallicity[0]/All.SolarAbundances[0];
                double x_photoelec = photoelec * sqrt(T) / (0.5 * (1.0e-12+n_elec) * nHcgs);
                LambdaPElec *= 0.049/(1+pow(x_photoelec/1925.,0.73)) + 0.037*pow(T/1.0e4,0.7)/(1+x_photoelec/5000.);
                Heat += LambdaPElec;
            }
        }
#endif
    }
  else				/* here we're outside of tabulated rates, T>Tmax K */
    {
      /* at high T (fully ionized); only free-free and Compton cooling are present.  Assumes no heating. */

      Heat = 0;
      LambdaExcH0 = LambdaExcHep = LambdaIonH0 = LambdaIonHe0 = LambdaIonHep = LambdaRecHp = LambdaRecHep = LambdaRecHepp = LambdaRecHepd = 0;

      /* very hot: H and He both fully ionized */
      nHp = 1.0;
      nHep = 0;
      nHepp = yhelium(target);
      n_elec = nHp + 2.0 * nHepp;
        
      LambdaFF = 1.42e-27 * sqrt(T) * (1.1 + 0.34 * exp(-(5.5 - logT) * (5.5 - logT) / 3)) * (nHp + 4 * nHepp) * n_elec;

      if(All.ComovingIntegrationOn)
      {
          redshift = 1 / All.Time - 1; /* add inverse Compton cooling off the microwave background */
          LambdaCmptn = 5.65e-36 * n_elec * (T - 2.73 * (1. + redshift)) * pow(1. + redshift, 4.) / nHcgs;
      }
      else {LambdaCmptn = 0;}
#if defined(BH_COMPTON_HEATING) && !defined(SINGLE_STAR_FORMATION)
        /* Relativistic compton cooling from an AGN source */
        LambdaCmptn += AGN_LambdaPre * (T - AGN_T_Compton) * (T/1.5e9)/(1-exp(-T/1.5e9)) * n_elec/nHcgs;
#endif
        
      Lambda = LambdaFF + LambdaCmptn;

      /* per CAFG's calculations, we should note that at very high temperatures, the rate-limiting step may be
         the Coulomb collisions moving energy from protons to e-; which if slow will prevent efficient e- cooling */
      if(Lambda > 2.19e-21/sqrt(T/1.0e8)) Lambda=2.19e-21/sqrt(T/1.0e8);
    }
    
    
    double Q = Heat - Lambda;
#ifdef COOL_LOW_TEMPERATURES
    /* if we are in the optically thick limit, we need to modify the cooling/heating rates according to the appropriate limits; 
        this flag does so by using a simple approximation. we consider the element as if it were a slab, with a column density 
        calculated from the simulation properties and the Sobolev approximation. we then assume it develops an equilibrium internal 
        temperature structure on a radiative diffusion timescale much faster than the dynamical time, and so the surface radiation 
        from a photosphere can be simply related to the local density by the optical depth to infinity. the equations here follow 
        Rafikov, 2007 (ApJ, 662, 642): 
            denergy/dt/dArea = sigma*T^4 / fc(tau)
            fc(tau) = tau^eta + 1/tau (taking chi, phi~1; the second term describes the optically thin limit, which is calculated above 
                more accurately anyways - that was just Kirchoff's Law; so we only need to worry about the first term)
            eta = 4*(gamma-1) / [gamma*(1+alpha+beta*(gamma-1)/gamma)], where gamma=real polytropic index, and alpha/beta follow
                an opacity law kappa=kappa_0 * P^alpha * T^beta. for almost all the regimes of interest, however, eta~1, which is also 
                what is obtained for a convectively stable slab. so we will use this.
            now, this gives sigma*T^4/tau * Area_eff / nHcgs as the 'effective' cooling rate in our units of Heat or Lambda above. 
                the nHcgs just puts it in the same volumetric terms. The Area_eff must be defined as ~m_particle/surface_density
                to have the same meaning for a slab as assumed in Rafikov (and to integrate correctly over all particles in the slab, 
                if/when the slab is resolved). We estimate this in our usual fashion with the Sobolev-type column density
            tau = kappa * surface_density; we estimate kappa ~ 5 cm^2/g * (0.001+Z/Z_solar), as the frequency-integrated kappa for warm 
                dust radiation (~150K), weighted by the dust-to-gas ratio (with a floor for molecular absorption). we could make this 
                temperature-dependent, though, fairly easily - for this particular problem it won't make much difference
        This rate then acts as an upper limit to the net heating/cooling calculated above (restricts absolute value)
     */
    if( (nHcgs > 0.1) && (target >= 0) )  /* don't bother at very low densities, since youre not optically thick, and protect from target=-1 with GALSF_EFFECTIVE_EQS */
    {
        double surface_density = evaluate_NH_from_GradRho(SphP[target].Gradients.Density,PPP[target].Hsml,SphP[target].Density,PPP[target].NumNgb,1);
        surface_density *= All.UnitDensity_in_cgs * All.UnitLength_in_cm * All.HubbleParam; // converts to cgs
        double effective_area = 2.3 * PROTONMASS / surface_density; // since cooling rate is ultimately per-particle, need a particle-weight here
        double kappa_eff; // effective kappa, accounting for metal abundance, temperature, and density //
        if(T < 1500.)
        {
            if(T < 150.) {kappa_eff=0.0027*T*sqrt(T);} else {kappa_eff=5.;}
            kappa_eff *= P[target].Metallicity[0]/All.SolarAbundances[0];
            if(kappa_eff < 0.1) {kappa_eff=0.1;}
        } else {
            /* this is an approximate result for high-temperature opacities, but provides a pretty good fit from 1.5e3 - 1.0e9 K */
            double k_electron = 0.2 * (1. + HYDROGEN_MASSFRAC); //0.167 * n_elec; /* Thompson scattering (non-relativistic) */
            double k_molecular = 0.1 * P[target].Metallicity[0]; /* molecular line opacities */
            double k_Hminus = 1.1e-25 * sqrt(P[target].Metallicity[0] * rho) * pow(T,7.7); /* negative H- ion opacity */
            double k_Kramers = 4.0e25 * (1.+HYDROGEN_MASSFRAC) * (P[target].Metallicity[0]+0.001) * rho / (T*T*T*sqrt(T)); /* free-free, bound-free, bound-bound transitions */
            double k_radiative = k_molecular + 1./(1./k_Hminus + 1./(k_electron+k_Kramers)); /* approximate interpolation between the above opacities */
            double k_conductive = 2.6e-7 * n_elec * T*T/(rho*rho); //*(1+pow(rho/1.e6,0.67) /* e- thermal conductivity can dominate at low-T, high-rho, here it as expressed as opacity */
            kappa_eff = 1./(1./k_radiative + 1./k_conductive); /* effective opacity including both heat carriers (this is exact) */
        }
        double tau_eff = kappa_eff * surface_density;
        double Lambda_Thick_BlackBody = 5.67e-5 * (T*T*T*T) * effective_area / ((1.+tau_eff) * nHcgs);
        if(Q > 0) {if(Q > Lambda_Thick_BlackBody) {Q=Lambda_Thick_BlackBody;}} else {if(Q < -Lambda_Thick_BlackBody) {Q=-Lambda_Thick_BlackBody;}}
    }
#endif
    
#ifndef COOLING_OPERATOR_SPLIT
    /* add the hydro energy change directly: this represents an additional heating/cooling term, to be accounted for 
        in the semi-implicit solution determined here. this is more accurate when tcool << tdynamical */
    if(target >= 0) Q += SphP[target].DtInternalEnergy / nHcgs;
#endif
    
  return Q;
} // ends CoolingRate





void InitCoolMemory(void)
{
    BetaH0 = (double *) mymalloc("BetaH0", (NCOOLTAB + 1) * sizeof(double));
    BetaHep = (double *) mymalloc("BetaHep", (NCOOLTAB + 1) * sizeof(double));
    AlphaHp = (double *) mymalloc("AlphaHp", (NCOOLTAB + 1) * sizeof(double));
    AlphaHep = (double *) mymalloc("AlphaHep", (NCOOLTAB + 1) * sizeof(double));
    Alphad = (double *) mymalloc("Alphad", (NCOOLTAB + 1) * sizeof(double));
    AlphaHepp = (double *) mymalloc("AlphaHepp", (NCOOLTAB + 1) * sizeof(double));
    GammaeH0 = (double *) mymalloc("GammaeH0", (NCOOLTAB + 1) * sizeof(double));
    GammaeHe0 = (double *) mymalloc("GammaeHe0", (NCOOLTAB + 1) * sizeof(double));
    GammaeHep = (double *) mymalloc("GammaeHep", (NCOOLTAB + 1) * sizeof(double));
    Betaff = (double *) mymalloc("Betaff", (NCOOLTAB + 1) * sizeof(double));
    
#ifdef COOL_METAL_LINES_BY_SPECIES
    long i_nH=41; long i_T=176; long kspecies=(long)NUM_METAL_SPECIES-1;
#ifdef GALSF_FB_RPROCESS_ENRICHMENT
    //kspecies -= 1;
    kspecies -= NUM_RPROCESS_SPECIES;
#endif
    SpCoolTable0 = (float *) mymalloc("SpCoolTable0",(kspecies*i_nH*i_T)*sizeof(float));
    if(All.ComovingIntegrationOn)
        SpCoolTable1 = (float *) mymalloc("SpCoolTable1",(kspecies*i_nH*i_T)*sizeof(float));
#endif
}



void MakeCoolingTable(void)
     /* Set up interpolation tables in T for cooling rates given in KWH, ApJS, 105, 19 
        Hydrogen, Helium III recombination rates and collisional ionization cross-sections are updated */
{
    int i;
    double T,Tfact;
    
    if(All.MinGasTemp > 0.0)
        Tmin = log10(All.MinGasTemp); // Tmin = log10(0.1 * All.MinGasTemp);
    else
        Tmin = 1.0;
    
    deltaT = (Tmax - Tmin) / NCOOLTAB;
    double ethmin = pow(10.0, Tmin) * (1. + YHELIUM_0) / ((1. + 4. * YHELIUM_0) * (PROTONMASS / BOLTZMANN) * GAMMA_MINUS1); /* minimum internal energy for neutral gas */
    /* minimum internal energy for neutral gas */
    for(i = 0; i <= NCOOLTAB; i++)
    {
        BetaH0[i] = BetaHep[i] = Betaff[i] = AlphaHp[i] = AlphaHep[i] = AlphaHepp[i] = Alphad[i] = GammaeH0[i] = GammaeHe0[i] = GammaeHep[i] = 0;
        T = pow(10.0, Tmin + deltaT * i);
        Tfact = 1.0 / (1 + sqrt(T / 1.0e5));
        
        if(118348 / T < 70) BetaH0[i] = 7.5e-19 * exp(-118348 / T) * Tfact;
        if(473638 / T < 70) BetaHep[i] = 5.54e-17 * pow(T, -0.397) * exp(-473638 / T) * Tfact;
        
        Betaff[i] = 1.43e-27 * sqrt(T) * (1.1 + 0.34 * exp(-(5.5 - log10(T)) * (5.5 - log10(T)) / 3));
        //AlphaHp[i] = 8.4e-11 * pow(T / 1000, -0.2) / (1. + pow(T / 1.0e6, 0.7)) / sqrt(T);	/* old Cen92 fit */
        //AlphaHep[i] = 1.5e-10 * pow(T, -0.6353); /* old Cen92 fit */
        //AlphaHepp[i] = 4. * AlphaHp[i];	/* old Cen92 fit */
        AlphaHp[i] = 7.982e-11 / ( sqrt(T/3.148) * pow((1.0+sqrt(T/3.148)), 0.252) * pow((1.0+sqrt(T/7.036e5)), 1.748) ); /* Verner & Ferland (1996) [more accurate than Cen92] */
        AlphaHep[i]= 9.356e-10 / ( sqrt(T/4.266e-2) * pow((1.0+sqrt(T/4.266e-2)), 0.2108) * pow((1.0+sqrt(T/3.676e7)), 1.7892) ); /* Verner & Ferland (1996) [more accurate than Cen92] */
        AlphaHepp[i] = 2. * 7.982e-11 / ( sqrt(T/(4.*3.148)) * pow((1.0+sqrt(T/(4.*3.148))), 0.252) * pow((1.0+sqrt(T/(4.*7.036e5))), 1.748) ); /* Verner & Ferland (1996) : ~ Z*alphaHp[1,T/Z^2] */
        
        if(470000 / T < 70) Alphad[i] = 1.9e-3 * pow(T, -1.5) * exp(-470000 / T) * (1. + 0.3 * exp(-94000 / T));
        if(157809.1 / T < 70) GammaeH0[i] = 5.85e-11 * sqrt(T) * exp(-157809.1 / T) * Tfact;
        if(285335.4 / T < 70) GammaeHe0[i] = 2.38e-11 * sqrt(T) * exp(-285335.4 / T) * Tfact;
        if(631515.0 / T < 70) GammaeHep[i] = 5.68e-12 * sqrt(T) * exp(-631515.0 / T) * Tfact;
        
    }
}


#ifdef COOL_METAL_LINES_BY_SPECIES

void LoadMultiSpeciesTables(void)
{
    if(All.ComovingIntegrationOn) {
        int i;
        double z;
        if(All.Time==All.TimeBegin) {
            All.SpeciesTableInUse=48;
            ReadMultiSpeciesTables(All.SpeciesTableInUse);
        }
        z=log10(1/All.Time)*48;
        i=(int)z;
        if(i<48) {
            if(i<All.SpeciesTableInUse) {
                All.SpeciesTableInUse=i;
                ReadMultiSpeciesTables(All.SpeciesTableInUse);
            }}
    } else {
        if(All.Time==All.TimeBegin) ReadMultiSpeciesTables(0);
    }
}

void ReadMultiSpeciesTables(int iT)
{
    /* read table w n,T for each species */
    long i_nH=41; long i_Temp=176; long kspecies=(long)NUM_METAL_SPECIES-1; long i,j,k,r;
#ifdef GALSF_FB_RPROCESS_ENRICHMENT
    //kspecies -= 1;
    kspecies -= NUM_RPROCESS_SPECIES;
#endif
    /* int i_He=7;  int l; */
    FILE *fdcool; char *fname;
    
    fname=GetMultiSpeciesFilename(iT,0);
    if(ThisTask == 0) printf("Opening Cooling Table %s \n",fname);
    if(!(fdcool = fopen(fname, "r"))) {
        printf(" Cannot read species cooling table in file `%s'\n", fname); endrun(456);}
    for(i=0;i<kspecies;i++) {
        for(j=0;j<i_nH;j++) {
            for(k=0;k<i_Temp;k++) {
                r=fread(&SpCoolTable0[i*i_nH*i_Temp + j*i_Temp + k],sizeof(float),1,fdcool);
                if(r!=1) {printf(" Reached Cooling EOF! \n"); 
                }
            }}}
    fclose(fdcool);
    /*
     GetMultiSpeciesFilename(iT,&fname,1);
     if(!(fdcool = fopen(fname, "r"))) {
     printf(" Cannot read species (He) cooling table in file `%s'\n", fname); endrun(456);}
     for(i=0;i<2;i++)
     for(j=0;j<i_nH;j++)
     for(k=0;k<i_Temp;k++)
     for(l=0;l<i_He;l++)
     fread(&SpCoolTable0_He[i][j][k][l],sizeof(float),1,fdcool);
     fclose(fdcool);
     */
    if (All.ComovingIntegrationOn && i<48) {
        fname=GetMultiSpeciesFilename(iT+1,0);
        if(ThisTask == 0) printf("Opening (z+) Cooling Table %s \n",fname);
        if(!(fdcool = fopen(fname, "r"))) {
            printf(" Cannot read species 1 cooling table in file `%s'\n", fname); endrun(456);}
        for(i=0;i<kspecies;i++) {
            for(j=0;j<i_nH;j++) {
                for(k=0;k<i_Temp;k++) {
                    r=fread(&SpCoolTable1[i*i_nH*i_Temp + j*i_Temp + k],sizeof(float),1,fdcool);
                    if(r!=1) {printf(" Reached Cooling EOF! \n");
                    }
                }}}
        fclose(fdcool);
        /*
         GetMultiSpeciesFilename(iT+1,&fname,1);
         if(!(fdcool = fopen(fname, "r"))) {
         printf(" Cannot read species 1 (He) cooling table in file `%s'\n", fname); endrun(456);}
         for(i=0;i<2;i++)
         for(j=0;j<i_nH;j++)
         for(k=0;k<i_Temp;k++)
         for(l=0;l<i_He;l++)
         fread(&SpCoolTable1_He[i][j][k][l],sizeof(float),1,fdcool);
         fclose(fdcool);
         */
    }
}

char *GetMultiSpeciesFilename(int i, int hk)
{
    static char fname[100];
    if(i<0) i=0; if(i>48) i=48;
    if(hk==0) {
        sprintf(fname,"./spcool_tables/spcool_%d",i);
    } else {
        sprintf(fname,"./spcool_tables/spcool_He_%d",i);
    }
    return fname;
}

#endif



/* table input (from file TREECOOL) for ionizing parameters */
/* NOTE: we've switched to using the updated TREECOOL from CAFG, june11 version */

#define JAMPL	1.0		/* amplitude factor relative to input table */
#define TABLESIZE 250		/* Max # of lines in TREECOOL */

static float inlogz[TABLESIZE];
static float gH0[TABLESIZE], gHe[TABLESIZE], gHep[TABLESIZE];
static float eH0[TABLESIZE], eHe[TABLESIZE], eHep[TABLESIZE];
static int nheattab;		/* length of table */

void ReadIonizeParams(char *fname)
{
    int i;
    FILE *fdcool;
    
    if(!(fdcool = fopen(fname, "r")))
    {
        printf(" Cannot read ionization table in file `%s'\n", fname);
        endrun(456);
    }
    
    for(i = 0; i < TABLESIZE; i++)
        gH0[i] = 0;
    
    for(i = 0; i < TABLESIZE; i++)
        if(fscanf(fdcool, "%g %g %g %g %g %g %g",
                  &inlogz[i], &gH0[i], &gHe[i], &gHep[i], &eH0[i], &eHe[i], &eHep[i]) == EOF)
            break;
    
    fclose(fdcool);
    
    /*  nheattab is the number of entries in the table */
    
    for(i = 0, nheattab = 0; i < TABLESIZE; i++)
        if(gH0[i] != 0.0)
            nheattab++;
        else
            break;
    
    if(ThisTask == 0)
        printf("\n\nread ionization table with %d entries in file `%s'.\n\n", nheattab, fname);
}


void IonizeParams(void)
{
    IonizeParamsTable();
}



void IonizeParamsTable(void)
{
    int i, ilow;
    double logz, dzlow, dzhi;
    double redshift;
    
    if(All.ComovingIntegrationOn)
        redshift = 1 / All.Time - 1;
    else
    {
        /* in non-cosmological mode, still use, but adopt z=0 background */
        redshift = 0;
        /*
         gJHe0 = gJHep = gJH0 = epsHe0 = epsHep = epsH0 = J_UV = 0;
         return;
         */
    }
    
    logz = log10(redshift + 1.0);
    ilow = 0;
    for(i = 0; i < nheattab; i++)
    {
        if(inlogz[i] < logz)
            ilow = i;
        else
            break;
    }
    
    dzlow = logz - inlogz[ilow];
    dzhi = inlogz[ilow + 1] - logz;
    
    if(logz > inlogz[nheattab - 1] || gH0[ilow] == 0 || gH0[ilow + 1] == 0 || nheattab == 0)
    {
        gJHe0 = gJHep = gJH0 = 0;
        epsHe0 = epsHep = epsH0 = 0;
        J_UV = 0;
        return;
    }
    else
        J_UV = 1.e-21;		/* irrelevant as long as it's not 0 */
    
    gJH0 = JAMPL * pow(10., (dzhi * log10(gH0[ilow]) + dzlow * log10(gH0[ilow + 1])) / (dzlow + dzhi));
    gJHe0 = JAMPL * pow(10., (dzhi * log10(gHe[ilow]) + dzlow * log10(gHe[ilow + 1])) / (dzlow + dzhi));
    gJHep = JAMPL * pow(10., (dzhi * log10(gHep[ilow]) + dzlow * log10(gHep[ilow + 1])) / (dzlow + dzhi));
    epsH0 = JAMPL * pow(10., (dzhi * log10(eH0[ilow]) + dzlow * log10(eH0[ilow + 1])) / (dzlow + dzhi));
    epsHe0 = JAMPL * pow(10., (dzhi * log10(eHe[ilow]) + dzlow * log10(eHe[ilow + 1])) / (dzlow + dzhi));
    epsHep = JAMPL * pow(10., (dzhi * log10(eHep[ilow]) + dzlow * log10(eHep[ilow + 1])) / (dzlow + dzhi));
    
    return;
}


void SetZeroIonization(void)
{
    gJHe0 = gJHep = gJH0 = 0;
    epsHe0 = epsHep = epsH0 = 0;
    J_UV = 0;
}


void IonizeParamsFunction(void)
{
    int i, nint;
    double a0, planck, ev, e0_H, e0_He, e0_Hep;
    double gint, eint, t, tinv, fac, eps;
    double at, beta, s;
    double pi;
    
#define UVALPHA         1.0
    double Jold = -1.0;
    double redshift;
    
    J_UV = 0.;
    gJHe0 = gJHep = gJH0 = 0.;
    epsHe0 = epsHep = epsH0 = 0.;
    
    
    if(All.ComovingIntegrationOn)	/* analytically compute params from power law J_nu */
    {
        redshift = 1 / All.Time - 1;
        
        if(redshift >= 6)
            J_UV = 0.;
        else
        {
            if(redshift >= 3)
                J_UV = 4e-22 / (1 + redshift);
            else
            {
                if(redshift >= 2)
                    J_UV = 1e-22;
                else
                    J_UV = 1.e-22 * pow(3.0 / (1 + redshift), -3.0);
            }
        }
        
        if(J_UV == Jold)
            return;
        
        
        Jold = J_UV;
        
        if(J_UV == 0)
            return;
        
        
        a0 = 6.30e-18;
        planck = 6.6262e-27;
        ev = 1.6022e-12;
        e0_H = 13.6058 * ev;
        e0_He = 24.59 * ev;
        e0_Hep = 54.4232 * ev;
        
        gint = 0.0;
        eint = 0.0;
        nint = 5000;
        at = 1. / ((double) nint);
        
        for(i = 1; i <= nint; i++)
        {
            t = (double) i;
            t = (t - 0.5) * at;
            tinv = 1. / t;
            eps = sqrt(tinv - 1.);
            fac = exp(4. - 4. * atan(eps) / eps) / (1. - exp(-2. * M_PI / eps)) * pow(t, UVALPHA + 3.);
            gint += fac * at;
            eint += fac * (tinv - 1.) * at;
        }
        
        gJH0 = a0 * gint / planck;
        epsH0 = a0 * eint * (e0_H / planck);
        gJHep = gJH0 * pow(e0_H / e0_Hep, UVALPHA) / 4.0;
        epsHep = epsH0 * pow((e0_H / e0_Hep), UVALPHA - 1.) / 4.0;
        
        at = 7.83e-18;
        beta = 1.66;
        s = 2.05;
        
        gJHe0 = (at / planck) * pow((e0_H / e0_He), UVALPHA) *
        (beta / (UVALPHA + s) + (1. - beta) / (UVALPHA + s + 1));
        epsHe0 = (e0_He / planck) * at * pow(e0_H / e0_He, UVALPHA) *
        (beta / (UVALPHA + s - 1) + (1 - 2 * beta) / (UVALPHA + s) - (1 - beta) / (UVALPHA + s + 1));
        
        pi = M_PI;
        gJH0 *= 4. * pi * J_UV;
        gJHep *= 4. * pi * J_UV;
        gJHe0 *= 4. * pi * J_UV;
        epsH0 *= 4. * pi * J_UV;
        epsHep *= 4. * pi * J_UV;
        epsHe0 *= 4. * pi * J_UV;
    }
}





void InitCool(void)
{
    if(ThisTask == 0)
        printf("Initializing cooling ...\n");
    
    All.Time = All.TimeBegin;
    set_cosmo_factors_for_current_time();
    
#ifdef GRACKLE
    InitGrackle();
#endif
    
    InitCoolMemory();
    MakeCoolingTable();
    ReadIonizeParams("TREECOOL");
    IonizeParams();
#ifdef COOL_METAL_LINES_BY_SPECIES
    LoadMultiSpeciesTables();
#endif
}




#ifdef COOL_METAL_LINES_BY_SPECIES
double GetCoolingRateWSpecies(double nHcgs, double logT, double *Z)
{
    double ne_over_nh_tbl=1, Lambda=0;
    int k, N_species_active = NUM_METAL_SPECIES-1;
#ifdef GALSF_FB_RPROCESS_ENRICHMENT
    N_species_active -= NUM_RPROCESS_SPECIES;
#endif
    
    /* pre-calculate the indices for density and temperature, then we just need to call the tables by species */
    int ixmax=40, iymax=175;
    int ix0, iy0, ix1, iy1;
    double dx, dy, dz, mdz;
    long i_T=iymax+1, inHT=i_T*(ixmax+1);
    if(All.ComovingIntegrationOn && All.SpeciesTableInUse<48) {dz=log10(1/All.Time)*48; dz=dz-(int)dz; mdz=1-dz;} else {dz=0; mdz=1;}
    
    dx = (log10(nHcgs)-(-8.0))/(0.0-(-8.0))*ixmax;
    dy = (logT-2.0)/(9.0-2.0)*iymax;
    if(dx<0) {dx=0;} else {if(dx>ixmax) {dx=ixmax;}}
    ix0=(int)dx; ix1=ix0+1; if(ix1>ixmax) {ix1=ixmax;}
    dx=dx-ix0;
    if(dy<0) {dy=0;} else {if(dy>iymax) {dy=iymax;}}
    iy0=(int)dy; iy1=iy0+1; if(iy1>iymax) {iy1=iymax;}
    dy=dy-iy0;
    long index_x0y0=iy0+ix0*i_T, index_x0y1=iy1+ix0*i_T, index_x1y0=iy0+ix1*i_T, index_x1y1=iy1+ix1*i_T;
    
    ne_over_nh_tbl = GetLambdaSpecies(0,index_x0y0,index_x0y1,index_x1y0,index_x1y1,dx,dy,dz,mdz);
    if(ne_over_nh_tbl > 0)
    {
        double zfac = 0.0127 / All.SolarAbundances[0];
        for (k=1; k<N_species_active; k++)
        {
            long k_index = k * inHT;
            Lambda += GetLambdaSpecies(k_index,index_x0y0,index_x0y1,index_x1y0,index_x1y1,dx,dy,dz,mdz) * Z[k+1]/(All.SolarAbundances[k+1]*zfac);
        }
        Lambda /= ne_over_nh_tbl;
    }
    return Lambda;
}


double GetLambdaSpecies(long k_index, long index_x0y0, long index_x0y1, long index_x1y0, long index_x1y1, double dx, double dy, double dz, double mdz)
{
    long x0y0 = index_x0y0 + k_index;
    long x0y1 = index_x0y1 + k_index;
    long x1y0 = index_x1y0 + k_index;
    long x1y1 = index_x1y1 + k_index;
    double i1, i2, j1, j2, w1, w2, u1;
    i1 = SpCoolTable0[x0y0];
    i2 = SpCoolTable0[x0y1];
    j1 = SpCoolTable0[x1y0];
    j2 = SpCoolTable0[x1y1];
    if(dz > 0)
    {
        i1 = mdz * i1 + dz * SpCoolTable1[x0y0];
        i2 = mdz * i2 + dz * SpCoolTable1[x0y1];
        j1 = mdz * j1 + dz * SpCoolTable1[x1y0];
        j2 = mdz * j2 + dz * SpCoolTable1[x1y1];
    }
    w1 = i1*(1-dy) + i2*dy;
    w2 = j1*(1-dy) + j2*dy;
    u1 = w1*(1-dx) + w2*dx;
    return u1;
}

#endif // COOL_METAL_LINES_BY_SPECIES



#ifdef GALSF_FB_LOCAL_UV_HEATING
void selfshield_local_incident_uv_flux(void)
{
    /* include local self-shielding with the following */
    int i;
    double GradRho = 0;
    double sigma_eff_0 = All.UnitDensity_in_cgs * All.UnitLength_in_cm * All.HubbleParam;
    double code_flux_to_physical = sigma_eff_0 * All.cf_a2inv; // convert code flux [units=(L/M)_physical * Mcode/(Rcode*Rcode)] to physical cgs units
    
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if(P[i].Type==0)
        {
            if((SphP[i].RadFluxUV>0) && (PPP[i].Hsml>0) && (SphP[i].Density>0) && (P[i].Mass>0) && (All.Time>0))
            {
                SphP[i].RadFluxUV *= code_flux_to_physical; // convert to cgs
                SphP[i].RadFluxEUV *= code_flux_to_physical; // convert to cgs
                
                GradRho = sigma_eff_0 * evaluate_NH_from_GradRho(P[i].GradRho,PPP[i].Hsml,SphP[i].Density,PPP[i].NumNgb,1); // in CGS
                double tau_nuv = KAPPA_UV * GradRho * (1.0e-3 + P[i].Metallicity[0]/All.SolarAbundances[0]); // optical depth: this part is attenuated by dust //
                double tau_euv = 3.7e6 * GradRho; // optical depth: 912 angstrom kappa_euv: opacity from neutral gas //
                SphP[i].RadFluxUV *= exp(-tau_nuv); // attenuate
                SphP[i].RadFluxEUV *= 0.01 + 0.99/(1.0 + 0.8*tau_euv + 0.85*tau_euv*tau_euv); // attenuate (for clumpy medium with 1% scattering) //
                
                SphP[i].RadFluxUV *= 1276.19; // convert to Habing units (normalize strength to local MW field)
                SphP[i].RadFluxEUV *= 1276.19; // convert to Habing units (normalize strength to local MW field)
            } else {
                SphP[i].RadFluxUV = 0;
                SphP[i].RadFluxEUV = 0;
            }}}
}
#endif // GALSF_FB_LOCAL_UV_HEATING


#endif
