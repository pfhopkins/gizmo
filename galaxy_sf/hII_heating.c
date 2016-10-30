#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "../allvars.h"
#include "../proto.h"

/* Routines for simple photo-ionization heating feedback model */
/*
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */


#if defined(GALSF_FB_HII_HEATING) || (defined(RT_CHEM_PHOTOION) && defined(GALSF))

double particle_ionizing_luminosity_in_cgs(long i)
{
    double lm_ssp=0;
    if(P[i].Type != 5)
    {
        /* use updated SB99 tracks: including rotation, new mass-loss tracks, etc. */
        double star_age = evaluate_stellar_age_Gyr(P[i].StellarAge);
        if(star_age >= 0.02) return 0; // skip since old stars don't contribute
        if(star_age < 0.0035) {lm_ssp=500.;} else {
            double log_age=log10(star_age/0.0035);
            lm_ssp=470.*pow(10.,-2.24*log_age-4.2*log_age*log_age) + 60.*pow(10.,-3.6*log_age);
        }
        lm_ssp *= calculate_relative_light_to_mass_ratio_from_imf(i);
#ifdef SINGLE_STAR_FORMATION
        /* use effective temperature as a function of stellar mass and size to get ionizing photon production */
        double l_sol = bh_lum_bol(0,P[i].Mass,i) * (All.UnitEnergy_in_cgs / (All.UnitTime_in_s * SOLAR_LUM)); // L/Lsun
        double m_sol = P[i].Mass * (All.UnitMass_in_g / (All.HubbleParam * SOLAR_MASS)); // M/Msun
        double r_sol = pow(m_sol, 0.738); // R/Rsun
        double T_eff = 5780. * pow(l_sol/(r_sol*r_sol), 0.25); // ZAMS effective temperature
        double x0 = 157800./T_eff; // h*nu/kT for nu>13.6 eV
        double fion = 0.0; // fraction of blackbody emitted above x0
        if(x0 < 30.) {double q=18./(x0*x0) + 1./(8. + x0 + 20.*exp(-x0/10.)); fion = exp(-1./q);} // accurate to <10% for a Planck spectrum to x0>30, well into vanishing flux //
        lm_ssp = fion * l_sol / m_sol; // just needs to be multiplied by the actual stellar luminosity to get luminosity to mass ratio
#endif
    } // (P[i].Type != 5)
    
#ifdef BH_HII_HEATING
    /* AGN template: light-to-mass ratio L(>13.6ev)/Mparticle in Lsun/Msun, above is dNion/dt = 5.5e54 s^-1 (Lbol/1e45 erg/s) */
    if(P[i].Type == 5) {lm_ssp = 1.741e6 * bh_lum_bol(P[i].BH_Mdot,P[i].Mass,i) / (P[i].Mass*All.UnitTime_in_Megayears/All.HubbleParam * (C / All.UnitVelocity_in_cm_per_s) * (C / All.UnitVelocity_in_cm_per_s));}
#endif
    
    lm_ssp *= (1.95*P[i].Mass*All.UnitMass_in_g/All.HubbleParam); // convert to luminosity from L/M
    if(lm_ssp <= 0) {lm_ssp=0;} // trap for negative values (shouldnt happen)
    if(isnan(lm_ssp)) {lm_ssp=0;} // trap for nans (if stellar age routine cant evaluate non-zero value)
    return lm_ssp;
}


#endif


#if defined(GALSF_FB_HII_HEATING)

/* this version of the HII routine only communicates with
     particles on the same processor */

void HII_heating_singledomain(void)
{
#ifdef RT_CHEM_PHOTOION
  return; // the work here is done in the actual RT routines if this switch is enabled //
#endif
  if(All.HIIRegion_fLum_Coupled<=0) return;
  if(All.Time<=0) return;


  MyDouble *pos;
  int startnode, numngb, j, n, i;
  int NITER_HIIFB, MAX_N_ITERATIONS_HIIFB;
  int jnearest,already_ionized,do_ionize,dummy;
  MyFloat h_i, dt, rho;
  double dx, dy, dz, h_i2, r2, r, u;
  double u_to_temp_fac,mionizable,mionized,RHII,RHIIMAX,R_search,rnearest;
  double stellum,uion,prob,rho_j,prandom;
  double m_available,m_effective,RHII_initial,RHIImultiplier;
  MAX_N_ITERATIONS_HIIFB = 5; NITER_HIIFB = 0;
  //MAX_N_ITERATIONS_HIIFB = 10;

  double total_l_ionizing,total_m_ionizing,total_m_ionizable,total_m_ionized,avg_RHII;
  total_l_ionizing=total_m_ionized=avg_RHII=total_m_ionizable=total_m_ionizing=0;
  double totMPI_m_ionizing,totMPI_l_ionizing,totMPI_m_ionized,totMPI_m_ionizable,totMPI_avg_RHII;
  totMPI_m_ionizing=totMPI_l_ionizing=totMPI_m_ionized=totMPI_m_ionizable=totMPI_avg_RHII=0;


    Ngblist = (int *) mymalloc("Ngblist",NumPart * sizeof(int));
    
  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
  {
#ifdef BH_HII_HEATING
     if((P[i].Type==5)||(((P[i].Type == 4)||((All.ComovingIntegrationOn==0)&&((P[i].Type == 2)||(P[i].Type==3))))))
#else
     if((P[i].Type == 4)||((All.ComovingIntegrationOn==0)&&((P[i].Type == 2)||(P[i].Type==3))))
#endif
     {
#ifndef WAKEUP
         dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval / All.cf_hubble_a;
#else
         dt = P[i].dt_step * All.Timebase_interval / All.cf_hubble_a;
#endif
         if(dt<=0) continue; // don't keep going with this loop
         
         stellum = All.HIIRegion_fLum_Coupled * particle_ionizing_luminosity_in_cgs(i);
         if(stellum <= 0) continue;
         pos = P[i].Pos;
         rho = P[i].DensAroundStar;
         h_i = PPP[i].Hsml;
         total_m_ionizing += 1;//P[i].Mass;
         total_l_ionizing += stellum;
        
         RHII = 4.67e-9*pow(stellum,0.333)*
                pow(rho*All.cf_a3inv*All.UnitDensity_in_cgs*All.HubbleParam*All.HubbleParam,-0.66667);
         RHII /= All.cf_atime*All.UnitLength_in_cm/All.HubbleParam;
         // crude estimate of where flux falls below cosmic background
         RHIIMAX=240.0*pow(stellum,0.5)/(All.cf_atime*All.UnitLength_in_cm/All.HubbleParam);
         //if(RHIIMAX>20.0*h_i) RHIIMAX=20.0*h_i;
         //RHIIMAX *= 50;
         //if(RHIIMAX<10.0*h_i) RHIIMAX=10.0*h_i;
         
         if(RHIIMAX < h_i) RHIIMAX=h_i;
         if(RHIIMAX > 5.0*h_i) RHIIMAX=5.*h_i;
         /*
         if(All.ComovingIntegrationOn)
         {
             double fac = DMIN(50./(3.*All.cf_atime), 100.); // want to search larger radii at high-z, where the background is weaker
             RHIIMAX *= fac;
             fac = DMIN(1.0/All.cf_atime , 10.0);
             if(RHIIMAX < fac*h_i) RHIIMAX = fac*h_i;
             fac = DMIN(10.0/All.cf_atime , 30.0);
             if(RHIIMAX > fac*h_i) RHIIMAX = fac*h_i;
         } else {
             RHIIMAX *= 20;
             if(RHIIMAX < 2.0*h_i) RHIIMAX = 2.0*h_i;
             if(RHIIMAX > 10.0*h_i) RHIIMAX = 10.0*h_i;
         }
         */
         
         mionizable=NORM_COEFF*rho*RHII*RHII*RHII;
         if(RHII>RHIIMAX) RHII=RHIIMAX;
         if(RHII<0.5*h_i) RHII=0.5*h_i;
         RHII_initial=RHII;

     prandom = get_random_number(P[i].ID + 7); // pre-calc the (eventually) needed random number
     // guesstimate if this is even close to being interesting for the particle masses of interest
     if(prandom < 2.0*mionizable/P[i].Mass) // prandom > this, won't be able to ionize anything interesting
     {
         mionized=0.0;
         total_m_ionizable += mionizable;
         h_i2=h_i*h_i;
         u_to_temp_fac = PROTONMASS / BOLTZMANN * GAMMA_MINUS1 * All.UnitEnergy_in_cgs / All.UnitMass_in_g;
         uion = HIIRegion_Temp / u_to_temp_fac;
         startnode = All.MaxPart;     /* root node */
         jnearest=-1; rnearest=1.0e10; dummy=0; NITER_HIIFB=0;

       do {
         jnearest=-1; rnearest=1.0e10;
         R_search = RHII;
         if(h_i>R_search) R_search=h_i;
         numngb = ngb_treefind_variable_threads(pos, R_search, -1, &startnode, 0, &dummy, &dummy, &dummy, Ngblist);
         if(numngb>0)
         {
         for(n = 0; n < numngb; n++)
         {
         j = Ngblist[n];
         if(P[j].Type == 0 && P[j].Mass > 0)
          {
            dx = pos[0] - P[j].Pos[0];
            dy = pos[1] - P[j].Pos[1];
            dz = pos[2] - P[j].Pos[2];
#ifdef PERIODIC               /*  now find the closest image in the given box size  */
              NEAREST_XYZ(dx,dy,dz,1);
#endif
            r2 = dx * dx + dy * dy + dz * dz;
            r=sqrt(r2);

            /* check whether the particle is already ionized */
            already_ionized = 0;
              rho_j = Particle_density_for_energy_i(j);
              if(SphP[j].InternalEnergy<SphP[j].InternalEnergyPred) {u=SphP[j].InternalEnergy;} else {u=SphP[j].InternalEnergyPred;}
              
            if((SphP[j].DelayTimeHII > 0)||(u>uion)) already_ionized=1;

            /* now, if inside RHII and mionized<mionizeable and not already ionized, can be ionized! */
            do_ionize=0; prob=0;
            if((r<=RHII)&&(already_ionized==0)&&(mionized<mionizable)) 
            {
               m_effective = P[j].Mass*(SphP[j].Density/rho);
               // weight by density b/c of how the recomination rate in each particle scales            
              m_available = mionizable-mionized;
              if(m_effective<=m_available) {
                do_ionize=1;
                prob = 1.001;
              } else {
                prob = m_available/m_effective; // determine randomly if ionized
                if(prandom < prob) do_ionize=1;
              } // if(m_effective<=m_available) {
               if(do_ionize==1) 
               {
                   SphP[j].InternalEnergy = uion;
                   SphP[j].InternalEnergyPred = SphP[j].InternalEnergy;
                 SphP[j].DelayTimeHII = dt;
                 already_ionized = 1;
               } // if(do_ionize==1) 
              mionized += prob*m_effective;
            } // if((r<=RHII)&&(already_ionized==0)&&(mionized<mionizable)) 

            /* if nearest un-ionized particle, mark as such */
            if((r<rnearest)&&(already_ionized==0)) 
            {
              rnearest = r;
              jnearest = j;
            }
          } // if(P[j].Type == 0 && P[j].Mass > 0)
         } // for(n = 0; n < numngb; n++)
         } // if(numngb>0)


          // if still have photons and jnearest is un-ionized 
          if((mionized<mionizable)&&(jnearest>=0))
          {
            j=jnearest;
            m_effective = P[j].Mass*(SphP[j].Density/rho);
            m_available = mionizable-mionized;
            prob=m_available/m_effective;
            do_ionize=0;
            if(prandom < prob) do_ionize=1;
            if(do_ionize==1)
            {
                SphP[j].InternalEnergy = uion;
                SphP[j].InternalEnergyPred = SphP[j].InternalEnergy;
              SphP[j].DelayTimeHII = dt;
            } // if(do_ionize==1)
            mionized += prob*m_effective;
          } // if((mionized<mionizable)&&(jnearest>=0))


          /* now check if we have ionized sufficient material, and if not, 
               iterate with larger regions until we do */
            RHIImultiplier=1.10;
            if(mionized < 0.95*mionizable) 
            {
               /* ok, this guy did not find enough gas to ionize, it needs to expand its search */
               if((RHII >= 30.0*RHII_initial)||(RHII>=RHIIMAX)||(NITER_HIIFB >= MAX_N_ITERATIONS_HIIFB))
               {
                 /* we're done looping, this is just too big an HII region */
                 mionized = 1.001*mionizable;
               } else {
                /* in this case we're allowed to keep expanding RHII */
                 if(mionized <= 0) 
                 {
                   RHIImultiplier = 2.0;
                 } else {
                   RHIImultiplier=pow(mionized/mionizable,-0.333);
                   if(RHIImultiplier>5.0) RHIImultiplier=5.0;
                   if(RHIImultiplier<1.26) RHIImultiplier=1.26;
                 } // if(mionized <= 0) 
               RHII *= RHIImultiplier;
               if(RHII>1.26*RHIIMAX) RHII=1.26*RHIIMAX;
               startnode=All.MaxPart; // this will trigger the while loop to continue
             } // if((RHII >= 5.0*RHII_initial)||(RHII>=RHIIMAX)||(NITER_HIIFB >= MAX_N_ITERATIONS_HIIFB))
            } // if(mionized < 0.95*mionizable) 
       NITER_HIIFB++;
       } while(startnode >= 0);
     total_m_ionized += mionized;
     avg_RHII += RHII;
     } // if(prandom < 2.0*mionizable/P[i].Mass)
  } // if((P[i].Type == 4)||(P[i].Type == 2)||(P[i].Type == 3))
  } // for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
  myfree(Ngblist);


   MPI_Reduce(&total_m_ionizing, &totMPI_m_ionizing, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&total_l_ionizing, &totMPI_l_ionizing, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&total_m_ionized, &totMPI_m_ionized, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&avg_RHII, &totMPI_avg_RHII, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   if(ThisTask == 0)
   {
     if(totMPI_m_ionizing>0)
       {
           totMPI_avg_RHII /= totMPI_m_ionizing;
#ifndef IO_REDUCED_MODE
           printf("HII PhotoHeating: Time=%g: %g sources with L_tot/erg=%g ; M_ionized=%g ; <R_HII>=%g \n",
                All.Time,totMPI_m_ionizing,totMPI_l_ionizing,totMPI_m_ionized,totMPI_avg_RHII);
           fflush(stdout);
#endif
           fprintf(FdHIIHeating, "%lg %g %g %g %g \n",
                   All.Time,totMPI_m_ionizing,totMPI_l_ionizing,totMPI_m_ionized,totMPI_avg_RHII);
     }
#ifdef IO_REDUCED_MODE
       if(All.HighestActiveTimeBin == All.HighestOccupiedTimeBin)
#endif
       {fflush(FdHIIHeating);}
   } // ThisTask == 0

//  CPU_Step[CPU_HIIHEATING] += measure_time();

} // void HII_heating_singledomain(void)


#endif // GALSF_FB_HII_HEATING

