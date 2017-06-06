#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "../allvars.h"
#include "../proto.h"
#include "../kernel.h"
#ifdef PTHREADS_NUM_THREADS
#include <pthread.h>
#endif

/* Routines for mechanical feedback/enrichment models: stellar winds, supernovae, etc */

/*
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */


#if defined(GALSF_FB_SNE_HEATING) || defined(GALSF_FB_GASRETURN)

/* in case you're wondering, here are some conventions that may be useful for solar abundances
 All.SolarAbundances[0]=0.02;        // all metals (by mass); present photospheric abundances from Asplund et al. 2009 (Z=0.0134, proto-solar=0.0142) in notes;
 //   also Anders+Grevesse 1989 (older, but hugely-cited compilation; their Z=0.0201, proto-solar=0.0213)
 // note that the 'all metals' above is the only one where solar enters with any direct role; for everything else,
 //  'solar' is totally arbitrary (the enrichment routines, etc, don't know what solar is 'supposed' to be): so these
 //  are here purely for convenience and to initialize non-zero metallicities
 //
 All.SolarAbundances[1]=0.28;    // He  (10.93 in units where log[H]=12, so photospheric mass fraction -> Y=0.2485 [Hydrogen X=0.7381]; Anders+Grevesse Y=0.2485, X=0.7314)
 All.SolarAbundances[2]=3.26e-3; // C   (8.43 -> 2.38e-3, AG=3.18e-3)
 All.SolarAbundances[3]=1.32e-3; // N   (7.83 -> 0.70e-3, AG=1.15e-3)
 All.SolarAbundances[4]=8.65e-3; // O   (8.69 -> 5.79e-3, AG=9.97e-3)
 All.SolarAbundances[5]=2.22e-3; // Ne  (7.93 -> 1.26e-3, AG=1.72e-3)
 All.SolarAbundances[6]=9.31e-4; // Mg  (7.60 -> 7.14e-4, AG=6.75e-4)
 All.SolarAbundances[7]=1.08e-3; // Si  (7.51 -> 6.71e-4, AG=7.30e-4)
 All.SolarAbundances[8]=6.44e-4; // S   (7.12 -> 3.12e-4, AG=3.80e-4)
 All.SolarAbundances[9]=1.01e-4; // Ca  (6.34 -> 0.65e-4, AG=0.67e-4)
 All.SolarAbundances[10]=1.73e-3; // Fe (7.50 -> 1.31e-3, AG=1.92e-3)
 */


#ifdef PTHREADS_NUM_THREADS
extern pthread_mutex_t mutex_nexport;
extern pthread_mutex_t mutex_partnodedrift;
#define LOCK_NEXPORT     pthread_mutex_lock(&mutex_nexport);
#define UNLOCK_NEXPORT   pthread_mutex_unlock(&mutex_nexport);
#else
#define LOCK_NEXPORT
#define UNLOCK_NEXPORT
#endif


#define SNeIIBW_Radius_Factor 1.0 // (optional) boost cooling radius for resolution-check


struct kernel_addFB
{
    double dp[3];
    double r;
    double wk, dwk;
    double hinv, hinv3, hinv4;
};


struct addFBdata_in
{
    MyDouble Pos[3];
    MyDouble Vel[3];
    MyFloat Hsml;
    MyFloat V_i;
    MyFloat SNe_v_ejecta;
    MyDouble Msne;
    MyDouble unit_mom_SNe;
    MyFloat Area_weighted_sum[AREA_WEIGHTED_SUM_ELEMENTS];
#ifdef METALS
    MyDouble yields[NUM_METAL_SPECIES];
#endif
    int NodeList[NODELISTLENGTH];
}
*AddFBDataIn, *AddFBDataGet;


struct addFBdata_out
{
    MyFloat Area_weighted_sum[AREA_WEIGHTED_SUM_ELEMENTS];
    MyFloat M_coupled;
}
*AddFBDataResult, *AddFBDataOut;



void particle2in_addFB(struct addFBdata_in *in, int i, int feedback_type);
void particle2in_addFB_SNe(struct addFBdata_in *in, int i);
void particle2in_addFB_winds(struct addFBdata_in *in, int i);
void particle2in_addFB_Rprocess(struct addFBdata_in *in, int i);
void particle2in_addFB_wt(struct addFBdata_in *in, int i);
void out2particle_addFB(struct addFBdata_out *out, int i, int mode, int feedback_type);

//
// key variable passed throughout is 'feedback type' -- tells the routine which mode we're in:
//     -1 == evaluate weighting factors for feedback
//      0 == SNe (Type I or II)
//      1 == stellar winds
//      2 == rprocess events (NS-NS mergers)
//
void particle2in_addFB(struct addFBdata_in *in, int i, int feedback_type)
{
    if(feedback_type==-2) particle2in_addFB_wt(in,i);
    if(feedback_type==-1) particle2in_addFB_wt(in,i);
    if(feedback_type==0) particle2in_addFB_SNe(in,i);
    if(feedback_type==1) particle2in_addFB_winds(in,i);
    if(feedback_type==2) particle2in_addFB_Rprocess(in,i);
}


void particle2in_addFB_Rprocess(struct addFBdata_in *in, int i)
{
#ifdef GALSF_FB_RPROCESS_ENRICHMENT
    /*
    model 0    tmin = 3e7 yr, rate = 1e-5
    model 1    tmin = 3e6 yr, rate = 1e-5
    model 2    tmin = 3e7 yr, rate = 1e-6
    model 3    tmin = 3e6 yr, rate = 1e-6
    */
    if(P[i].RProcessEvent_ThisTimeStep<=0)
    {
        in->Msne = 0;
        return;
    }
    int k;
    for(k = 0; k < 3; k++)
    {
        in->Pos[k] = P[i].Pos[k];
        in->Vel[k] = P[i].Vel[k];
    }
    for(k=0;k<NUM_METAL_SPECIES;k++) in->yields[k]=0.0;
    /* this is the important operation for setting the yields for each different model */
    double star_age = evaluate_stellar_age_Gyr(P[i].StellarAge);
    double p = get_random_number(P[i].ID + 8);
    double pcrit,tcrit;
    for(k=0;k<NUM_RPROCESS_SPECIES;k++)
    {
        in->yields[NUM_METAL_SPECIES-NUM_RPROCESS_SPECIES+k] = 0.0;
        if(k==0) {tcrit=0.03; pcrit=0.3333333333;}  // age > 3e7 yr, rate = 1e-5
        if(k==1) {tcrit=0.003; pcrit=0.3333333333;} // age > 3e6 yr, rate = 1e-5
        if(k==2) {tcrit=0.03; pcrit=0.03333333333;}  // age > 3e7 yr, rate = 1e-6
        if(k==3) {tcrit=0.003; pcrit=0.03333333333;}   // age > 3e6 yr, rate = 1e-6

        if((star_age>=tcrit)&&(p<=pcrit)&&(P[i].RProcessEvent_ThisTimeStep>0))
        {
            in->yields[NUM_METAL_SPECIES-NUM_RPROCESS_SPECIES+k] = 1.0; // absolute unit is irrelevant, so use 1.0 //
        }
    }
    in->Hsml = PPP[i].Hsml;
    double heff=PPP[i].Hsml / PPP[i].NumNgb; in->V_i=heff*heff*heff;
    in->Msne = 0.01 * (double)P[i].RProcessEvent_ThisTimeStep / ((double)((All.UnitMass_in_g/All.HubbleParam)/SOLAR_MASS)); // mass ejected ~0.01*M_sun; only here for bookkeeping //
    in->unit_mom_SNe = 0;
    in->SNe_v_ejecta = 0.;
    for(k=0;k<AREA_WEIGHTED_SUM_ELEMENTS;k++) {in->Area_weighted_sum[k] = P[i].Area_weighted_sum[k];}
#endif
}


void particle2in_addFB_wt(struct addFBdata_in *in, int i)
{
    int k;
    for(k = 0; k < 3; k++)
    {
        in->Pos[k] = P[i].Pos[k];
        in->Vel[k] = P[i].Vel[k];
    }
    in->Hsml = PPP[i].Hsml;
    double heff=PPP[i].Hsml / PPP[i].NumNgb; in->V_i=heff*heff*heff;
    in->Msne = P[i].Mass;
    in->unit_mom_SNe = 1;
    in->SNe_v_ejecta = 500.;
    for(k=0;k<AREA_WEIGHTED_SUM_ELEMENTS;k++) {in->Area_weighted_sum[k] = P[i].Area_weighted_sum[k];}
#ifdef GALSF_TURNOFF_COOLING_WINDS
    /* calculate the 'blast radius' and 'cooling turnoff time' used by this model */
    double n0 = P[i].DensAroundStar*All.cf_a3inv*All.UnitDensity_in_cgs * All.HubbleParam*All.HubbleParam / PROTONMASS;
    n0 = SNeIIBW_Radius_Factor * 0.087*pow(n0,-0.36) / (All.UnitLength_in_cm/All.HubbleParam/3.086e21*All.cf_atime);
    in->Hsml = DMAX(PPP[i].Hsml/2.,DMIN(n0,5.*PPP[i].Hsml));
#endif
}



void particle2in_addFB_SNe(struct addFBdata_in *in, int i)
{
    int k;
    double unitmass_in_msun,agemax,n_sn_0,star_age;
    double Msne,Esne51,unit_egy_SNe,unit_mom_SNe,SNe_v_ejecta;
#ifdef METALS
    double yields[NUM_METAL_SPECIES];
#endif
    
    agemax=0.03753; /* in Gyr */
    unitmass_in_msun=(All.UnitMass_in_g/All.HubbleParam)/SOLAR_MASS;
    
    n_sn_0 = P[i].SNe_ThisTimeStep;
    if((n_sn_0<=0)||(P[i].DensAroundStar<=0))
    {
        in->Msne = 0;
        return;
    }
    
    star_age = evaluate_stellar_age_Gyr(P[i].StellarAge);
    Esne51 = All.SNeIIEnergyFrac*n_sn_0;
    unit_egy_SNe = 1.0e51/(All.UnitEnergy_in_cgs/All.HubbleParam);
    Msne = 10.5; //average ejecta mass for single event (normalized to give total mass loss correctly)
#ifdef SINGLE_STAR_FORMATION
    Msne = P[i].Mass; // conserve mass and destroy the star completely
#endif
    if(star_age > agemax) Msne=1.4; // SnIa
    
#ifdef METALS
    if(NUM_METAL_SPECIES>=10) {
        // All, then He,C,N,O,Ne,Mg,Si,S,Ca,Fe
        if(star_age > agemax) {
            /* SNIa */ /* from Iwamoto et al. 1999; 'W7' models */
            yields[0]=1.4;/* total metal mass */
            yields[1]=0.0;/*He*/ yields[2]=0.049;/*C*/ yields[3]=1.2e-6;/*N*/ yields[4]=0.143;/*O*/
            yields[5]=0.0045;/*Ne*/ yields[6]=0.0086;/*Mg*/ yields[7]=0.156;/*Si*/
            yields[8]=0.087;/*S*/ yields[9]=0.012;/*Ca*/ yields[10]=0.743;/*Fe*/
        } else {
            /* SNII (IMF-averaged... may not be the best approx on short timescales...) */
            // Woosley & Weaver 1995 'B' models, integrated (log-log interp) over Salpeter IMF, from 10-50 rescaled from 0.1-100 solar //
            //yields[0]=1.70;/*Z*/
            //yields[1]=4.03;/*He*/ yields[2]=0.117;/*C*/ yields[3]=0.0399;/*N*/ yields[4]=1.06;/*O*/
            //yields[5]=0.169;/*Ne*/ yields[6]=0.0596;/*Mg*/ yields[7]=0.0924;/*Si*/
            //yields[8]=0.0408;/*S*/ yields[9]=0.00492;/*Ca*/ yields[10]=0.0842;/*Fe*/
            /* note that the Mg yield here, and perhaps some other products, is systematically low compared to observations (by ~0.4 dex) */
            // Nomoto 2006 (arXiv:0605725) //
            yields[0]=2.0;/*Z*/
            yields[1]=3.87;/*He*/ yields[2]=0.133;/*C*/ yields[3]=0.0479;/*N*/ yields[4]=1.17;/*O*/
            yields[5]=0.30;/*Ne*/ yields[6]=0.0987;/*Mg*/ yields[7]=0.0933;/*Si*/
            yields[8]=0.0397;/*S*/ yields[9]=0.00458;/*Ca*/ yields[10]=0.0741;/*Fe*/
            
            if(P[i].Metallicity[0]<0.033)
            {
                yields[3] *= P[i].Metallicity[0]/All.SolarAbundances[0]; // N scaling is strongly dependent on initial metallicity of the star //
            } else {
                yields[3] *= 1.65;
            }
            yields[0] += yields[3]-0.0479; // correct total metal mass for this correction //
        }
    }
    if(NUM_METAL_SPECIES==3 || NUM_METAL_SPECIES==4)
    {
        if(star_age > agemax) {
            yields[0]=1.4; yields[1]=0.0086; yields[2]=0.743; // All Z, Mg, Fe in total mass (SnIa)
        } else {
            yields[0]=2.0; yields[1]=0.12; yields[2]=0.0741; // SnII (per-SNe IMF-weighted averages)
        }
    }
    if(NUM_METAL_SPECIES==1) {if(star_age > agemax) {yields[0]=1.4;} else {yields[0]=2.0;}}
#ifdef GALSF_FB_RPROCESS_ENRICHMENT
    for(k=1;k<=NUM_RPROCESS_SPECIES;k++)
        yields[NUM_METAL_SPECIES-k] = 0.0; // R-process tracker
#endif
    for(k=0;k<NUM_METAL_SPECIES;k++) yields[k]=yields[k]/Msne;
    /* this is almost always irrelevant, but we can add a check to allow for larger abundances in the progenitor stars */
    for(k=0;k<NUM_METAL_SPECIES;k++)
    {
        yields[k]=yields[k]*(1.-P[i].Metallicity[0]) + (P[i].Metallicity[k]-All.SolarAbundances[k]);
    }
    if(star_age > agemax) {if(NUM_METAL_SPECIES>=10) {yields[1]=0.0;}} // no He yield for Ia SNe //
    for(k=0;k<NUM_METAL_SPECIES;k++) {if(yields[k]<0) yields[k]=0.0; if(yields[k]>1) yields[k]=1; in->yields[k]=yields[k];}
#endif
    
    Msne *= n_sn_0/unitmass_in_msun;
    SNe_v_ejecta = sqrt(2.0*Esne51*unit_egy_SNe/Msne); // v_ej
    unit_mom_SNe = Msne * SNe_v_ejecta; // total SNe ejecta momentum = M_ej*v_ej
    
    for(k = 0; k < 3; k++)
    {
        in->Pos[k] = P[i].Pos[k];
        in->Vel[k] = P[i].Vel[k];
    }
    in->Hsml = PPP[i].Hsml;
    double heff=PPP[i].Hsml / PPP[i].NumNgb; in->V_i=heff*heff*heff;
    in->Msne = Msne;
    in->SNe_v_ejecta = SNe_v_ejecta;
    in->unit_mom_SNe = unit_mom_SNe;
    for(k=0;k<AREA_WEIGHTED_SUM_ELEMENTS;k++) {in->Area_weighted_sum[k] = P[i].Area_weighted_sum[k];}
#ifdef GALSF_TURNOFF_COOLING_WINDS
    /* calculate the 'blast radius' and 'cooling turnoff time' used by this model */
    double n0 = P[i].DensAroundStar*All.cf_a3inv*All.UnitDensity_in_cgs * All.HubbleParam*All.HubbleParam / PROTONMASS;
    n0 = SNeIIBW_Radius_Factor * 0.087*pow(n0,-0.36) / (All.UnitLength_in_cm/All.HubbleParam/3.086e21*All.cf_atime);
    in->Hsml = DMAX(PPP[i].Hsml/2.,DMIN(n0,5.*PPP[i].Hsml));
    in->unit_mom_SNe = 0;
#endif
}



void particle2in_addFB_winds(struct addFBdata_in *in, int i)
{
#ifdef GALSF_FB_GASRETURN
    int k;
    double star_age,T_corr,GasSpecEnergy,M_wind,E_wind,wind_velocity,wind_momentum;
#ifdef METALS
    double yields[NUM_METAL_SPECIES];
#endif
    
    if((P[i].MassReturn_ThisTimeStep<=0)||(P[i].DensAroundStar<=0))
    {
        in->Msne = 0;
        return;
    }
    
    GasSpecEnergy = All.AGBGasEnergy * 3.0e7*(1.0/GAMMA_MINUS1)*(BOLTZMANN/PROTONMASS) * All.UnitMass_in_g/All.UnitEnergy_in_cgs;
    
    star_age = evaluate_stellar_age_Gyr(P[i].StellarAge);
    /* wind internal energy (hot winds from O-stars, slow from AGB winds) */
    if(star_age <= 0.1)
        T_corr=0.0013 + 16.0/(1+pow(star_age/0.0025,1.4)+pow(star_age/0.01,5.0));
    else
        T_corr=0.0013;
    
    /* define the metal yields */
#ifdef METALS
    /* assume track initial metallicity; turn on COOL_METAL_LINES_BY_SPECIES for more detailed tracking of light elements */
    for(k=0;k<NUM_METAL_SPECIES;k++) yields[k]=P[i].Metallicity[k];
    if(NUM_METAL_SPECIES>=10)
    {
        /* All, then He,C,N,O,Ne,Mg,Si,S,Ca,Fe ;; follow AGB/O star yields in more detail for the light elements */
        /*   the interesting species are He & CNO */
        // below is based on a compilation of van den Hoek & Groenewegen 1997, Marigo 2001, Izzard 2004 //
        yields[1]=0.36; /*He*/ yields[2]=0.016; /*C*/ yields[3]=0.0041; /*N*/ yields[4]=0.0118; /*O*/
        if(P[i].Metallicity[0]<0.033)
        {
            yields[4] *= P[i].Metallicity[0]/All.SolarAbundances[0]; // O scaling is strongly dependent on initial metallicity of the star //
        } else {
            yields[4] *= 1.65;
        }
        for(k=1;k<=4;k++)
        {
            yields[k] = yields[k]*(1.-P[i].Metallicity[0]) + (P[i].Metallicity[k]-All.SolarAbundances[k]);
            if(yields[k]<0) yields[k]=0;
            if(yields[k]>1) yields[k]=1;
        }
        yields[0]=0.0; for(k=2;k<=10;k++) yields[0]+=yields[k];
#ifdef GALSF_FB_RPROCESS_ENRICHMENT
        for(k=1;k<=NUM_RPROCESS_SPECIES;k++)
            yields[NUM_METAL_SPECIES-k] = 0.0; // R-process tracker
#endif
    } else {
        for(k=0;k<NUM_METAL_SPECIES;k++) {yields[k]=0.0;}
        yields[0]=0.032;
    }
    for(k=0;k<NUM_METAL_SPECIES;k++) in->yields[k]=yields[k];
#endif
    
    M_wind = P[i].Mass * P[i].MassReturn_ThisTimeStep;
    E_wind = GasSpecEnergy*T_corr*M_wind;
    wind_velocity = sqrt(2.0*E_wind/M_wind);
#ifdef SINGLE_STAR_FORMATION
    double m_msun = P[i].Mass * All.UnitMass_in_g / (All.HubbleParam * SOLAR_MASS); 
    wind_velocity = 616.e5 * sqrt((1.+0.1125*m_msun)/(1.+0.0125*m_msun)) * pow(m_msun,0.131); // same as scaling below from size-mass relation+eddington factor
    wind_velocity /= All.UnitVelocity_in_cm_per_s; // put into code units
#endif
    wind_momentum = M_wind*wind_velocity;
    
    for(k = 0; k < 3; k++)
    {
        in->Pos[k] = P[i].Pos[k];
        in->Vel[k] = P[i].Vel[k];
    }
    in->Hsml = PPP[i].Hsml;
    double heff=PPP[i].Hsml / PPP[i].NumNgb; in->V_i=heff*heff*heff;
    in->Msne = M_wind;
    in->SNe_v_ejecta = wind_velocity;
    in->unit_mom_SNe = wind_momentum;
    for(k=0;k<AREA_WEIGHTED_SUM_ELEMENTS;k++) {in->Area_weighted_sum[k] = P[i].Area_weighted_sum[k];}
#endif // GALSF_FB_GASRETURN //
}



void out2particle_addFB(struct addFBdata_out *out, int i, int mode, int feedback_type)
{
    if(feedback_type < 0)
    {
        int k=0, kmin=0, kmax=7;
        if(feedback_type == -1) {kmin=kmax; kmax=AREA_WEIGHTED_SUM_ELEMENTS;}
#ifdef USE_ORIGINAL_FIRE2_SNE_COUPLING_SCHEME
        kmin=0; kmax=AREA_WEIGHTED_SUM_ELEMENTS;
#endif
        for(k=kmin;k<kmax;k++) {ASSIGN_ADD(P[i].Area_weighted_sum[k], out->Area_weighted_sum[k], mode);}
    } else {
        P[i].Mass -= out->M_coupled;
        if((P[i].Mass<0)||(isnan(P[i].Mass))) {P[i].Mass=0;}
    }
}





void mechanical_fb_calc(int feedback_type)
{
    int j, k, ngrp, ndone, ndone_flag;
    int recvTask, place;
    int save_NextParticle;
    long long n_exported = 0;
    
    /* allocate buffers to arrange communication */
    long long NTaskTimesNumPart;
    NTaskTimesNumPart = maxThreads * NumPart;
    Ngblist = (int *) mymalloc("Ngblist", NTaskTimesNumPart * sizeof(int));
    size_t MyBufferSize = All.BufferSize;
    All.BunchSize = (int) ((MyBufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
                                             sizeof(struct addFBdata_in) +
                                             sizeof(struct addFBdata_out) +
                                             sizemax(sizeof(struct addFBdata_in),sizeof(struct addFBdata_out))));
    DataIndexTable = (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));
    
    NextParticle = FirstActiveParticle;	/* begin with this index */
    do
    {
        
        BufferFullFlag = 0;
        Nexport = 0;
        save_NextParticle = NextParticle;
        
        for(j = 0; j < NTask; j++)
        {
            Send_count[j] = 0;
            Exportflag[j] = -1;
        }
        
        /* do local particles and prepare export list */
#ifdef PTHREADS_NUM_THREADS
        pthread_t mythreads[PTHREADS_NUM_THREADS - 1];
        int threadid[PTHREADS_NUM_THREADS - 1];
        pthread_attr_t attr;
        
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
        pthread_mutex_init(&mutex_nexport, NULL);
        pthread_mutex_init(&mutex_partnodedrift, NULL);
        
        TimerFlag = 0;
        
        for(j = 0; j < PTHREADS_NUM_THREADS - 1; j++)
        {
            threadid[j] = j + 1;
            pthread_create(&mythreads[j], &attr, addFB_evaluate_primary, &threadid[j]);
        }
#endif
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
            int mainthreadid = omp_get_thread_num();
#else
            int mainthreadid = 0;
#endif
            addFB_evaluate_primary(&mainthreadid, feedback_type);	/* do local particles and prepare export list */
        }
        
#ifdef PTHREADS_NUM_THREADS
        for(j = 0; j < PTHREADS_NUM_THREADS - 1; j++)
            pthread_join(mythreads[j], NULL);
#endif
        
        
        if(BufferFullFlag)
        {
            int last_nextparticle = NextParticle;
            
            NextParticle = save_NextParticle;
            
            while(NextParticle >= 0)
            {
                if(NextParticle == last_nextparticle)
                    break;
                
                if(ProcessedFlag[NextParticle] != 1)
                    break;
                
                ProcessedFlag[NextParticle] = 2;
                
                NextParticle = NextActiveParticle[NextParticle];
            }
            
            if(NextParticle == save_NextParticle)
            {
                /* in this case, the buffer is too small to process even a single particle */
                endrun(116608);
            }
            
            int new_export = 0;
            
            for(j = 0, k = 0; j < Nexport; j++)
                if(ProcessedFlag[DataIndexTable[j].Index] != 2)
                {
                    if(k < j + 1)
                        k = j + 1;
                    
                    for(; k < Nexport; k++)
                        if(ProcessedFlag[DataIndexTable[k].Index] == 2)
                        {
                            int old_index = DataIndexTable[j].Index;
                            
                            DataIndexTable[j] = DataIndexTable[k];
                            DataNodeList[j] = DataNodeList[k];
                            DataIndexTable[j].IndexGet = j;
                            new_export++;
                            
                            DataIndexTable[k].Index = old_index;
                            k++;
                            break;
                        }
                }
                else
                    new_export++;
            
            Nexport = new_export;
            
        }
        
        n_exported += Nexport;
        
        for(j = 0; j < NTask; j++)
            Send_count[j] = 0;
        for(j = 0; j < Nexport; j++)
            Send_count[DataIndexTable[j].Task]++;
        
        MYSORT_DATAINDEX(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare);
        MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);
        
        for(j = 0, Nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; j < NTask; j++)
        {
            Nimport += Recv_count[j];
            
            if(j > 0)
            {
                Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
                Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
            }
        }
        
        AddFBDataGet = (struct addFBdata_in *) mymalloc("AddFBDataGet", Nimport * sizeof(struct addFBdata_in));
        AddFBDataIn = (struct addFBdata_in *) mymalloc("AddFBDataIn", Nexport * sizeof(struct addFBdata_in));
        
        /* prepare particle data for export */
        
        for(j = 0; j < Nexport; j++)
        {
            place = DataIndexTable[j].Index;
            particle2in_addFB(&AddFBDataIn[j], place, feedback_type);
            memcpy(AddFBDataIn[j].NodeList,
                   DataNodeList[DataIndexTable[j].IndexGet].NodeList, NODELISTLENGTH * sizeof(int));
        }
        
        /* exchange particle data */
        int TAG_TO_USE = TAG_FBLOOP_1A;
        if(feedback_type==-2) {TAG_TO_USE = TAG_FBLOOP_5A;}
        if(feedback_type==-1) {TAG_TO_USE = TAG_FBLOOP_1A;}
        if(feedback_type== 0) {TAG_TO_USE = TAG_FBLOOP_2A;}
        if(feedback_type== 1) {TAG_TO_USE = TAG_FBLOOP_3A;}
        if(feedback_type== 2) {TAG_TO_USE = TAG_FBLOOP_4A;}
        if(feedback_type== 3) {TAG_TO_USE = TAG_FBLOOP_5A;}
        for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
        {
            recvTask = ThisTask ^ ngrp;
            
            if(recvTask < NTask)
            {
                if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
                {
                    /* get the particles */
                    MPI_Sendrecv(&AddFBDataIn[Send_offset[recvTask]],
                                 Send_count[recvTask] * sizeof(struct addFBdata_in), MPI_BYTE,
                                 recvTask, TAG_TO_USE,
                                 &AddFBDataGet[Recv_offset[recvTask]],
                                 Recv_count[recvTask] * sizeof(struct addFBdata_in), MPI_BYTE,
                                 recvTask, TAG_TO_USE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
        
        myfree(AddFBDataIn);
        AddFBDataResult =
        (struct addFBdata_out *) mymalloc("AddFBDataResult", Nimport * sizeof(struct addFBdata_out));
        AddFBDataOut =
        (struct addFBdata_out *) mymalloc("AddFBDataOut", Nexport * sizeof(struct addFBdata_out));
        
        /* now do the particles that were sent to us */
        NextJ = 0;
#ifdef PTHREADS_NUM_THREADS
        for(j = 0; j < PTHREADS_NUM_THREADS - 1; j++)
            pthread_create(&mythreads[j], &attr, addFB_evaluate_secondary, &threadid[j]);
#endif
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
            int mainthreadid = omp_get_thread_num();
#else
            int mainthreadid = 0;
#endif
            addFB_evaluate_secondary(&mainthreadid, feedback_type);
        }
        
#ifdef PTHREADS_NUM_THREADS
        for(j = 0; j < PTHREADS_NUM_THREADS - 1; j++)
            pthread_join(mythreads[j], NULL);
        
        pthread_mutex_destroy(&mutex_partnodedrift);
        pthread_mutex_destroy(&mutex_nexport);
        pthread_attr_destroy(&attr);
#endif
        
        if(NextParticle < 0)
            ndone_flag = 1;
        else
            ndone_flag = 0;
        
        MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        /* get the result */
        TAG_TO_USE = TAG_FBLOOP_1B;
        if(feedback_type==-2) {TAG_TO_USE = TAG_FBLOOP_5B;}
        if(feedback_type==-1) {TAG_TO_USE = TAG_FBLOOP_1B;}
        if(feedback_type== 0) {TAG_TO_USE = TAG_FBLOOP_2B;}
        if(feedback_type== 1) {TAG_TO_USE = TAG_FBLOOP_3B;}
        if(feedback_type== 2) {TAG_TO_USE = TAG_FBLOOP_4B;}
        if(feedback_type== 3) {TAG_TO_USE = TAG_FBLOOP_5B;}
        for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
        {
            recvTask = ThisTask ^ ngrp;
            if(recvTask < NTask)
            {
                if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
                {
                    /* send the results */
                    MPI_Sendrecv(&AddFBDataResult[Recv_offset[recvTask]],
                                 Recv_count[recvTask] * sizeof(struct addFBdata_out),
                                 MPI_BYTE, recvTask, TAG_TO_USE,
                                 &AddFBDataOut[Send_offset[recvTask]],
                                 Send_count[recvTask] * sizeof(struct addFBdata_out),
                                 MPI_BYTE, recvTask, TAG_TO_USE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
        
        /* add the result to the local particles */
        for(j = 0; j < Nexport; j++)
        {
            place = DataIndexTable[j].Index;
            out2particle_addFB(&AddFBDataOut[j], place, 1, feedback_type);
        }
        myfree(AddFBDataOut);
        myfree(AddFBDataResult);
        myfree(AddFBDataGet);
    }
    while(ndone < NTask);
    
    myfree(DataNodeList);
    myfree(DataIndexTable);
    myfree(Ngblist);
    
    /* do final operations on results */
    /* (not needed here, since the entire operation is from central particle to the gas) */
    /*
     for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
     if(P[i].Type == 0)
     {
     }
     */
}




#ifdef USE_ORIGINAL_FIRE2_SNE_COUPLING_SCHEME

int addFB_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex,
                   int *ngblist, int feedback_type)
{
    int startnode, numngb_inbox, listindex = 0;
    int j, k, n;
    double u,r2,h2;
    double v_ejecta_max,kernel_zero,wk,dM,dP;
    double E_coupled,dP_sum,dP_boost_sum;
    
    struct kernel_addFB kernel;
    struct addFBdata_in local;
    struct addFBdata_out out;
    memset(&out, 0, sizeof(struct addFBdata_out));
    
    v_ejecta_max = 5000.0 * 1.0e5/ All.UnitVelocity_in_cm_per_s;
    // 'speed limit' to prevent numerically problematic kicks at low resolution //
    kernel_main(0.0,1.0,1.0,&kernel_zero,&wk,-1);
    
    /* Load the data for the particle injecting feedback */
    if(mode == 0)
        particle2in_addFB(&local, target, feedback_type);
    else
        local = AddFBDataGet[target];
    
    if(local.Msne<=0) return 0; // no SNe for the master particle! nothing to do here //
    if(local.Hsml<=0) return 0; // zero-extent kernel, no particles //
    h2 = local.Hsml*local.Hsml;
    kernel_hinv(local.Hsml, &kernel.hinv, &kernel.hinv3, &kernel.hinv4);
    
    // some units (just used below, but handy to define for clarity) //
    double unitlength_in_kpc=All.UnitLength_in_cm/All.HubbleParam/3.086e21*All.cf_atime;
    double density_to_n=All.cf_a3inv*All.UnitDensity_in_cgs * All.HubbleParam*All.HubbleParam / PROTONMASS;
    double unit_egy_SNe = 1.0e51/(All.UnitEnergy_in_cgs/All.HubbleParam);
#ifdef GALSF_TURNOFF_COOLING_WINDS
    double pressure_to_p4 = (1/All.cf_afac1)*density_to_n*(All.UnitEnergy_in_cgs/All.UnitMass_in_g) / 1.0e4;
#endif
    
#if defined(COSMIC_RAYS) && defined(GALSF_FB_SNE_HEATING)
    // account for energy going into CRs, so we don't 'double count' //
    double CR_energy_to_inject = 0;
    if((local.SNe_v_ejecta > 2.0e8 / All.UnitVelocity_in_cm_per_s) && (feedback_type == 0))
    {
        local.SNe_v_ejecta *= sqrt(1-All.CosmicRay_SNeFraction);
        CR_energy_to_inject = (All.CosmicRay_SNeFraction/(1.-All.CosmicRay_SNeFraction)) * 0.5 * local.Msne * local.SNe_v_ejecta * local.SNe_v_ejecta;
    }
#endif
    
    // now define quantities that will be used below //
    double Esne51;
    Esne51 = 0.5*local.SNe_v_ejecta*local.SNe_v_ejecta*local.Msne / unit_egy_SNe;
    double RsneKPC, RsneKPC_0;//, RsneMAX;
    RsneKPC=0.; //RsneMAX=local.Hsml;
    RsneKPC_0=(0.0284/unitlength_in_kpc) * pow(1+Esne51,0.286); //Cioffi: weak external pressure
    double r2max_phys = 2.0/unitlength_in_kpc; // no super-long-range effects allowed! (of course this is arbitrary in code units) //
    r2max_phys *= r2max_phys;
    
    
    
    /* Now start the actual FB computation for this particle */
    if(mode == 0)
    {
        startnode = All.MaxPart;	/* root node */
    }
    else
    {
        startnode = AddFBDataGet[target].NodeList[0];
        startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }
    
    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb_inbox = ngb_treefind_pairs_threads(local.Pos, local.Hsml, target, &startnode, mode, exportflag, exportnodecount, exportindex, ngblist);
            
            if(numngb_inbox < 0)
                return -1;
            
            E_coupled = dP_sum = dP_boost_sum = 0;
            for(n = 0; n < numngb_inbox; n++)
            {
                j = ngblist[n];
                if(P[j].Type != 0) continue; // require a gas particle //
                if(P[j].Mass <= 0) continue; // require the particle has mass //
                
                for(k=0; k<3; k++) {kernel.dp[k] = local.Pos[k] - P[j].Pos[k];}
#ifdef PERIODIC
                NEAREST_XYZ(kernel.dp[0],kernel.dp[1],kernel.dp[2],1); // find the closest image in the given box size  //
#endif
                r2=0; for(k=0;k<3;k++) {r2 += kernel.dp[k]*kernel.dp[k];}
                if(r2<=0) continue; // same particle //
                
                double h2j = PPP[j].Hsml * PPP[j].Hsml;
                if((r2>h2)&&(r2>h2j)) continue; // outside kernel (in both 'directions') //
                if(r2 > r2max_phys) continue; // outside long-range cutoff //
                // calculate kernel quantities //
                kernel.r = sqrt(r2);
                if(kernel.r <= 0) continue;
                u = kernel.r * kernel.hinv;
                double hinv_j = 1./PPP[j].Hsml;
                double hinv3_j = hinv_j*hinv_j*hinv_j;
                double wk_j = 0, dwk_j = 0, u_j = kernel.r * hinv_j, hinv4_j = hinv_j*hinv3_j, V_j = P[j].Mass / SphP[j].Density;
                kernel_main(u, kernel.hinv3, kernel.hinv4, &kernel.wk, &kernel.dwk, 1);
                kernel_main(u_j, hinv3_j, hinv4_j, &wk_j, &dwk_j, 1);
                if(local.V_i<0 || isnan(local.V_i)) {local.V_i=0;}
                if(V_j<0 || isnan(V_j)) {V_j=0;}
                double sph_area = fabs(local.V_i*local.V_i*kernel.dwk + V_j*V_j*dwk_j); // effective face area //
                wk = 0.5 * (1 - 1/sqrt(1 + sph_area / (M_PI*kernel.r*kernel.r))); // corresponding geometric weight //
                
                if((wk <= 0)||(isnan(wk))) continue; // no point in going further, there's no physical weight here
                
                double wk_vec[AREA_WEIGHTED_SUM_ELEMENTS] = {0};
                wk_vec[0] = wk;
#ifndef GALSF_FB_SNE_NONISOTROPIZED
                if(kernel.dp[0]>0) {wk_vec[1]=wk*kernel.dp[0]/kernel.r; wk_vec[2]=0;} else {wk_vec[1]=0; wk_vec[2]=wk*kernel.dp[0]/kernel.r;}
                if(kernel.dp[1]>0) {wk_vec[3]=wk*kernel.dp[1]/kernel.r; wk_vec[4]=0;} else {wk_vec[3]=0; wk_vec[4]=wk*kernel.dp[1]/kernel.r;}
                if(kernel.dp[2]>0) {wk_vec[5]=wk*kernel.dp[2]/kernel.r; wk_vec[6]=0;} else {wk_vec[5]=0; wk_vec[6]=wk*kernel.dp[2]/kernel.r;}
#endif
                
                // if feedback_type==-1, this is a pre-calc loop to get the relevant weights for coupling //
                if(feedback_type < 0)
                {
                    for(k=0;k<AREA_WEIGHTED_SUM_ELEMENTS;k++) out.Area_weighted_sum[k] += wk_vec[k];
                    continue;
                }
                // NOW do the actual feedback calculation //
                double wk_norm = 1. / (MIN_REAL_NUMBER + fabs(local.Area_weighted_sum[0])); // normalization for scalar weight sum
                wk *= wk_norm; // this way wk matches the value summed above for the weighting //
                
                if((wk <= 0)||(isnan(wk))) continue;
                
                /* define initial mass and ejecta velocity in this 'cone' */
                double v_bw[3]={0}, e_shock=0;
                double pnorm = 0;
                double pvec[3]={0};
                for(k=0; k<3; k++)
                {
#ifdef GALSF_FB_SNE_NONISOTROPIZED
                    pvec[k] = -wk * kernel.dp[k] / kernel.r;
#else
                    double q;
                    q = 0;
                    
#if !(EXPAND_PREPROCESSOR_(GALSF_FB_SNE_HEATING) == 1) // check whether a numerical value is assigned
#if (GALSF_FB_SNE_HEATING == 0) // code for symmetrized but non-isotropic
//#define DO_SYMMETRIZED_SNE_HEATING_ONLY_BUT_NOT_FULLY_ISOTROPIC
#define DO_FULLY_ISOTROPIZED_SNE_HEATING
#endif
#endif
                    
//#ifdef DO_SYMMETRIZED_SNE_HEATING_ONLY_BUT_NOT_FULLY_ISOTROPIC
#ifndef DO_FULLY_ISOTROPIZED_SNE_HEATING
                    q = 0; int i1=2*k+1, i2=i1+1;
                    double q_i1 = fabs(local.Area_weighted_sum[i1]);
                    double q_i2 = fabs(local.Area_weighted_sum[i2]);
                    if((q_i1>MIN_REAL_NUMBER)&&(q_i2>MIN_REAL_NUMBER))
                    {
                        double rr = q_i2/q_i1;
                        double rr2 = rr * rr;
                        if(wk_vec[i1] != 0)
                        {
                            q += wk_norm * wk_vec[i1] * sqrt(0.5*(1.0+rr2));
                        } else {
                            q += wk_norm * wk_vec[i2] * sqrt(0.5*(1.0+1.0/rr2));
                        }
                    } else {
                        q += wk_norm * (wk_vec[i1] + wk_vec[i2]);
                    }
                    pvec[k] = -q;
#else
                    if(k==0) {q=wk_vec[1]/(MIN_REAL_NUMBER+fabs(local.Area_weighted_sum[1])) + wk_vec[2]/(MIN_REAL_NUMBER+fabs(local.Area_weighted_sum[2]));}
                    if(k==1) {q=wk_vec[3]/(MIN_REAL_NUMBER+fabs(local.Area_weighted_sum[3])) + wk_vec[4]/(MIN_REAL_NUMBER+fabs(local.Area_weighted_sum[4]));}
                    if(k==2) {q=wk_vec[5]/(MIN_REAL_NUMBER+fabs(local.Area_weighted_sum[5])) + wk_vec[6]/(MIN_REAL_NUMBER+fabs(local.Area_weighted_sum[6]));}
                    pvec[k] = -q/4.; // factor of 4 accounts for our normalization of each directional component below to be =P (given by properly integrating over a unit sphere)
#endif

#endif
                    pnorm += pvec[k]*pvec[k];
                }
                pnorm = sqrt(pnorm);

                wk = pnorm; // this (vector norm) is the new 'weight function' for our purposes
                dM = wk * local.Msne;

                /* now, add contribution from relative star-gas particle motion to shock energy */
                for(k=0;k<3;k++)
                {
                    v_bw[k] = local.SNe_v_ejecta*pvec[k]/pnorm + (local.Vel[k]-P[j].Vel[k])/All.cf_atime;
                    e_shock += v_bw[k]*v_bw[k];
                }
                double mj_preshock = P[j].Mass;
                double dM_ejecta_in = dM;
                double massratio_ejecta = dM_ejecta_in / (dM_ejecta_in + P[j].Mass);
                double mu_j = P[j].Mass / (dM + P[j].Mass); 
                e_shock *= pnorm * 0.5*local.Msne * mu_j;
                
                
                
                if((wk <= 0)||(isnan(wk))) continue;
                
#ifndef GALSF_TURNOFF_COOLING_WINDS
                RsneKPC = RsneKPC_0;
                double n0 = SphP[j].Density*density_to_n;
                /* this is tedious, but is a fast approximation (essentially a lookup table) for the -0.429 power above */
                if(n0 < 1.e-3) {RsneKPC *= 19.4;} else {
                    if(n0 < 1.e-2) {RsneKPC *= 1.9 + 23./(1.+333.*n0);} else {
                        if(n0 < 1.e-1) {RsneKPC *= 0.7 + 8.4/(1.+33.3*n0);} else {
                            if(n0 < 1) {RsneKPC *= 0.08 + 3.1/(1.+2.5*n0);} else {
                                if(n0 < 10) {RsneKPC *= 0.1 + 1.14/(1.+0.333*n0);} else {
                                    if(n0 < 100) {RsneKPC *= 0.035 + 0.43/(1.+0.0333*n0);} else {
                                        if(n0 < 1000) {RsneKPC *= 0.017 + 0.154/(1.+0.00333*n0);} else {
                                            if(n0 < 1.e4) {RsneKPC *= 0.006 + 0.057/(1.+0.000333*n0);} else {
                                                RsneKPC *= pow(n0, -0.429); }}}}}}}}
                

                /* below expression is again just as good a fit to the simulations, and much faster to evaluate */
                double z0 = P[j].Metallicity[0]/All.SolarAbundances[0];
                if(z0 < 0.01)
                {
                    RsneKPC *= 2.0;
                } else {
                    if(z0 < 1)
                    {
                        RsneKPC *= 0.93 + 0.0615 / (0.05 + 0.8*z0);
                    } else {
                        RsneKPC *= 0.8 + 0.4 / (1 + z0);
                    }
                }
                /* calculates cooling radius given density and metallicity in this annulus into which the ejecta propagate */
                
                /* if coupling radius > R_cooling, account for thermal energy loss in the post-shock medium:
                 from Thornton et al. thermal energy scales as R^(-6.5) for R>R_cool */
                double r_eff_ij = sqrt(r2) - Get_Particle_Size(j);
                if(r_eff_ij > RsneKPC) {e_shock *= RsneKPC*RsneKPC*RsneKPC/(r_eff_ij*r_eff_ij*r_eff_ij);}

#endif
                
                /* now we have the proper energy to couple */
                E_coupled += e_shock;
                out.M_coupled += dM_ejecta_in;
                
                /* inject actual mass from mass return */
                if(P[j].Hsml<=0) {if(SphP[j].Density>0){SphP[j].Density*=(1+dM_ejecta_in/P[j].Mass);} else {SphP[j].Density=dM_ejecta_in*kernel.hinv3;}} else {SphP[j].Density+=kernel_zero*dM_ejecta_in*hinv3_j;}
                SphP[j].Density *= 1 + dM_ejecta_in/P[j].Mass; // inject mass at constant particle volume //
                P[j].Mass += dM_ejecta_in;
                out.M_coupled += dM_ejecta_in;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
                SphP[j].MassTrue += dM_ejecta_in;
#endif
#ifdef METALS
                /* inject metals */
                for(k=0;k<NUM_METAL_SPECIES;k++) {P[j].Metallicity[k]=(1-massratio_ejecta)*P[j].Metallicity[k] + massratio_ejecta*local.yields[k];}
                if(feedback_type == 2) continue; // for r-process, nothing left here to bother coupling //
#endif
#if defined(COSMIC_RAYS) && defined(GALSF_FB_SNE_HEATING)
                /* inject cosmic rays */
                SphP[j].CosmicRayEnergy += pnorm * CR_energy_to_inject;
                SphP[j].CosmicRayEnergyPred += pnorm * CR_energy_to_inject;
#endif
                
                /* inject the post-shock energy and momentum (convert to specific units as needed first) */
                e_shock *= 1 / P[j].Mass;
                SphP[j].InternalEnergy += e_shock;
                SphP[j].InternalEnergyPred += e_shock;
#ifdef GALSF_TURNOFF_COOLING_WINDS
                /* if the sub-grid 'cooling turnoff' model is enabled, turn off cooling for the 'blastwave timescale' */
                dP = 7.08 * pow(Esne51*SphP[j].Density*density_to_n,0.34) * pow(SphP[j].Pressure*pressure_to_p4,-0.70) / (All.UnitTime_in_Megayears/All.HubbleParam);
                if(dP>SphP[j].DelayTimeCoolingSNe) SphP[j].DelayTimeCoolingSNe=dP;
#else
                /* inject momentum */
                double m_ej_input = pnorm * local.Msne;
                /* appropriate factor for the ejecta being energy-conserving inside the cooling radius (or Hsml, if thats smaller) */
                double m_cooling = 4.18879*pnorm*SphP[j].Density*RsneKPC*RsneKPC*RsneKPC;
                /* apply limiter for energy conservation */
                double mom_boost_fac = 1 + sqrt(DMIN(mj_preshock , m_cooling) / m_ej_input);

                /* save summation values for outputs */
                dP = local.unit_mom_SNe / P[j].Mass * pnorm;
                dP_sum += dP;
                dP_boost_sum += dP * mom_boost_fac;

                /* actually do the injection */
                double q0 = All.cf_atime * (pnorm*local.Msne/P[j].Mass) * mom_boost_fac;
                for(k=0; k<3; k++)
                {
                    double q = q0 * v_bw[k];
                    P[j].Vel[k] += q;
                    SphP[j].VelPred[k] += q;
                }
#endif
                
#ifdef PM_HIRES_REGION_CLIPPING
                dP=0; for(k=0;k<3;k++) dP+=P[j].Vel[k]*P[j].Vel[k]; dP=sqrt(dP);
                if(dP>5.e9*All.cf_atime/All.UnitVelocity_in_cm_per_s) P[j].Mass=0;
                if(dP>1.e9*All.cf_atime/All.UnitVelocity_in_cm_per_s) for(k=0;k<3;k++) P[j].Vel[k]*=(1.e9*All.cf_atime/All.UnitVelocity_in_cm_per_s)/dP;
#endif
                
            } // for(n = 0; n < numngb; n++)
        } // while(startnode >= 0)
        
        if(mode == 1)
        {
            listindex++;
            if(listindex < NODELISTLENGTH)
            {
                startnode = AddFBDataGet[target].NodeList[listindex];
                if(startnode >= 0)
                    startnode = Nodes[startnode].u.d.nextnode;	/* open it */
            }
        } // if(mode == 1)
    } // while(startnode >= 0)
    
    /* Now collect the result at the right place */
    if(mode == 0)
        out2particle_addFB(&out, target, 0, feedback_type);
    else
        AddFBDataResult[target] = out;
    
    return 0;
} // int addFB_evaluate



#else // un-protected [updated, more fixed energy-injecting SNe scheme]


int addFB_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex,
                   int *ngblist, int feedback_type)
{
    int startnode, numngb_inbox, listindex = 0;
    int j, k, n;
    double u,r2,h2;
    double v_ejecta_max,kernel_zero,wk,dM_ejecta_in,dP;
    double E_coupled,dP_sum,dP_boost_sum;
    
    struct kernel_addFB kernel;
    struct addFBdata_in local;
    struct addFBdata_out out;
    memset(&out, 0, sizeof(struct addFBdata_out));
    
    v_ejecta_max = 5000.0 * 1.0e5/ All.UnitVelocity_in_cm_per_s;
    // 'speed limit' to prevent numerically problematic kicks at low resolution //
    kernel_main(0.0,1.0,1.0,&kernel_zero,&wk,-1);
    
    /* Load the data for the particle injecting feedback */
    if(mode == 0)
        particle2in_addFB(&local, target, feedback_type);
    else
        local = AddFBDataGet[target];
    
    if(local.Msne<=0) return 0; // no SNe for the master particle! nothing to do here //
    if(local.Hsml<=0) return 0; // zero-extent kernel, no particles //
    h2 = local.Hsml*local.Hsml;
    kernel_hinv(local.Hsml, &kernel.hinv, &kernel.hinv3, &kernel.hinv4);
    
    // some units (just used below, but handy to define for clarity) //
    double unitlength_in_kpc=All.UnitLength_in_cm/All.HubbleParam/3.086e21*All.cf_atime;
    double density_to_n=All.cf_a3inv*All.UnitDensity_in_cgs * All.HubbleParam*All.HubbleParam / PROTONMASS;
    double unit_egy_SNe = 1.0e51/(All.UnitEnergy_in_cgs/All.HubbleParam);
#ifdef GALSF_TURNOFF_COOLING_WINDS
    double pressure_to_p4 = (1/All.cf_afac1)*density_to_n*(All.UnitEnergy_in_cgs/All.UnitMass_in_g) / 1.0e4;
#endif
    
    // now define quantities that will be used below //
    double psi_cool=1, psi_egycon=1, v_ejecta_eff=local.SNe_v_ejecta;
    double wk_norm = 1. / (MIN_REAL_NUMBER + fabs(local.Area_weighted_sum[0])); // normalization for scalar weight sum
    double pnorm_sum = 1./(MIN_REAL_NUMBER + fabs(local.Area_weighted_sum[10])); // re-normalization after second pass for normalized "pnorm" (should be close to ~1)
    if(local.Area_weighted_sum[0] > MIN_REAL_NUMBER)
    {
        if(feedback_type >= 0)
        {
            double vba_2_eff = wk_norm * local.Area_weighted_sum[7]; // phi term for energy: weighted mass-deposited KE for ejecta neighbors
            v_ejecta_eff = sqrt(local.SNe_v_ejecta*local.SNe_v_ejecta + vba_2_eff); // account for all terms to get the revised KE term here
        }
        if(feedback_type == 0)
        {
            double beta_egycon = sqrt(pnorm_sum / local.Msne) * (1./v_ejecta_eff) * local.Area_weighted_sum[8]; // beta term for re-normalization for energy [can be positive or negative]
            double beta_cool = pnorm_sum * local.Area_weighted_sum[9]; // beta term if all particles in terminal-momentum-limit
            if(All.ComovingIntegrationOn) {if(fabs(beta_cool) < fabs(beta_egycon)) {beta_egycon = beta_cool;}}
            psi_egycon = sqrt(1. + beta_egycon*beta_egycon) - beta_egycon; // exact solution for energy equation for constant psi
            if(beta_egycon > 20.) {psi_egycon = 1./(2.*beta_egycon);} // replace with series expansion to avoid roundoff error at high beta
            if(beta_cool > 0.5) {psi_cool = 1./(2.*beta_cool);} // for cooling limit, only need upper limit to psi, all else will use less energy
        }
    }
    
#if defined(COSMIC_RAYS) && defined(GALSF_FB_SNE_HEATING)
    // account for energy going into CRs, so we don't 'double count' //
    double CR_energy_to_inject = 0;
    if((v_ejecta_eff > 2.0e8 / All.UnitVelocity_in_cm_per_s) && (feedback_type == 0))
    {
        v_ejecta_eff *= sqrt(1-All.CosmicRay_SNeFraction);
        CR_energy_to_inject = (All.CosmicRay_SNeFraction/(1.-All.CosmicRay_SNeFraction)) * 0.5 * local.Msne * v_ejecta_eff * v_ejecta_eff;
    }
#endif
    
    double Energy_injected_codeunits = 0.5 * local.Msne * v_ejecta_eff * v_ejecta_eff;
    double Esne51 = Energy_injected_codeunits / unit_egy_SNe;
    double RsneKPC = 0., RsneKPC_3 = 0., m_cooling = 0., v_cooling = 2.1e7 / All.UnitVelocity_in_cm_per_s;
    double RsneKPC_0 = (0.0284/unitlength_in_kpc);
    if(feedback_type == 0) // check for SNe specifically
    {
        RsneKPC_0 *= pow(1+Esne51,0.286); //SNe: using scaling from Cioffi with weak external pressure
    } else {
        RsneKPC_0 *= pow(Esne51,0.286); // ensures smooth conservation for winds and tracers as mass-loading goes to vanishingly small values
    }
    double r2max_phys = 2.0/unitlength_in_kpc; // no super-long-range effects allowed! (of course this is arbitrary in code units) //
    if(local.Hsml >= r2max_phys) {psi_egycon=DMIN(psi_egycon,1); psi_cool=DMIN(psi_cool,1);}
    r2max_phys *= r2max_phys;
    
    
    
    /* Now start the actual FB computation for this particle */
    if(mode == 0)
    {
        startnode = All.MaxPart;	/* root node */
    }
    else
    {
        startnode = AddFBDataGet[target].NodeList[0];
        startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }
    
    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb_inbox = ngb_treefind_pairs_threads(local.Pos, local.Hsml, target, &startnode, mode, exportflag, exportnodecount, exportindex, ngblist);
            
            if(numngb_inbox < 0)
                return -1;
            
            E_coupled = dP_sum = dP_boost_sum = 0;
            for(n = 0; n < numngb_inbox; n++)
            {
                j = ngblist[n];
                if(P[j].Type != 0) continue; // require a gas particle //
                if(P[j].Mass <= 0) continue; // require the particle has mass //
                
                for(k=0; k<3; k++) {kernel.dp[k] = local.Pos[k] - P[j].Pos[k];}
#ifdef PERIODIC
                NEAREST_XYZ(kernel.dp[0],kernel.dp[1],kernel.dp[2],1); // find the closest image in the given box size  //
#endif
                r2=0; for(k=0;k<3;k++) {r2 += kernel.dp[k]*kernel.dp[k];}
                if(r2<=0) continue; // same particle //
                
                double h2j = PPP[j].Hsml * PPP[j].Hsml;
                if((r2>h2)&&(r2>h2j)) continue; // outside kernel (in both 'directions') //
                if(r2 > r2max_phys) continue; // outside long-range cutoff //
                // calculate kernel quantities //
                kernel.r = sqrt(r2);
                if(kernel.r <= 0) continue;
                u = kernel.r * kernel.hinv;
                double hinv_j = 1./PPP[j].Hsml;
                double hinv3_j = hinv_j*hinv_j*hinv_j;
                double wk_j = 0, dwk_j = 0, u_j = kernel.r * hinv_j, hinv4_j = hinv_j*hinv3_j, V_j = P[j].Mass / SphP[j].Density;
                kernel_main(u, kernel.hinv3, kernel.hinv4, &kernel.wk, &kernel.dwk, 1);
                kernel_main(u_j, hinv3_j, hinv4_j, &wk_j, &dwk_j, 1);
                if(local.V_i<0 || isnan(local.V_i)) {local.V_i=0;}
                if(V_j<0 || isnan(V_j)) {V_j=0;}
                double sph_area = fabs(local.V_i*local.V_i*kernel.dwk + V_j*V_j*dwk_j); // effective face area //
                wk = 0.5 * (1 - 1/sqrt(1 + sph_area / (M_PI*kernel.r*kernel.r))); // corresponding geometric weight //
                
                if((wk <= 0)||(isnan(wk))) continue; // no point in going further, there's no physical weight here
                
                double wk_vec[AREA_WEIGHTED_SUM_ELEMENTS] = {0};
                wk_vec[0] = wk;
                if(kernel.dp[0]>0) {wk_vec[1]=wk*kernel.dp[0]/kernel.r; wk_vec[2]=0;} else {wk_vec[1]=0; wk_vec[2]=wk*kernel.dp[0]/kernel.r;}
                if(kernel.dp[1]>0) {wk_vec[3]=wk*kernel.dp[1]/kernel.r; wk_vec[4]=0;} else {wk_vec[3]=0; wk_vec[4]=wk*kernel.dp[1]/kernel.r;}
                if(kernel.dp[2]>0) {wk_vec[5]=wk*kernel.dp[2]/kernel.r; wk_vec[6]=0;} else {wk_vec[5]=0; wk_vec[6]=wk*kernel.dp[2]/kernel.r;}

#ifndef GALSF_TURNOFF_COOLING_WINDS
                RsneKPC = RsneKPC_0;
                /* calculate cooling radius given density and metallicity in this annulus into which the ejecta propagate */
                if(feedback_type != 2)
                {
                    double e0 = Esne51;
                    if(feedback_type < 0) {e0=1;}
                    if(feedback_type == 0) {e0+=1;}
                    double n0 = SphP[j].Density*density_to_n;
                    if(n0 < 0.001) {n0=0.001;}
                    double z0 = P[j].Metallicity[0]/All.SolarAbundances[0], z0_term = 1.;
                    if(z0 < 0.01) {z0 = 0.01;}
                    if(z0 < 1.) {z0_term = z0*sqrt(z0);} else {z0_term = z0;}
                    double nz_dep  = pow(n0 * z0_term , 0.14);;
                    v_cooling = 2.10e7 * DMAX(nz_dep,0.5) / All.UnitVelocity_in_cm_per_s;
                    m_cooling = 4.56e36 * e0 / (nz_dep*nz_dep * All.UnitMass_in_g/All.HubbleParam);
                    RsneKPC = pow( 0.238732 * m_cooling/SphP[j].Density , 1./3. );
                }
                RsneKPC_3 = RsneKPC*RsneKPC*RsneKPC;
#endif
                
                // if feedback_type==-1, this is a pre-calc loop to get the relevant weights for coupling //
                if(feedback_type < 0)
                {
                    if(feedback_type==-1) // the Area_weighted_sum quantities are computed on loop=-2; these quantities must be computed on loop=-1 (after Area_weighted_sums are computed)
                    {
                        /* calculate the corrected momentum vectors that we will actually use in the coupling proper */
                        double pnorm=0, pvec[3]={0}, vel_ba_2=0, cos_vel_ba_pcoupled=0;
                        for(k=0;k<3;k++)
                        {
                            double q = 0; int i1=2*k+1, i2=i1+1;
                            double q_i1 = fabs(local.Area_weighted_sum[i1]);
                            double q_i2 = fabs(local.Area_weighted_sum[i2]);
                            if((q_i1>MIN_REAL_NUMBER)&&(q_i2>MIN_REAL_NUMBER))
                            {
                                double rr = q_i2/q_i1;
                                double rr2 = rr * rr;
                                if(wk_vec[i1] != 0)
                                {
                                    q += wk_norm * wk_vec[i1] * sqrt(0.5*(1.0+rr2));
                                } else {
                                    q += wk_norm * wk_vec[i2] * sqrt(0.5*(1.0+1.0/rr2));
                                }
                            } else {
                                q += wk_norm * (wk_vec[i1] + wk_vec[i2]);
                            }
                            pvec[k] = -q;
                            pnorm += pvec[k]*pvec[k];
                        }
                        pnorm = sqrt(pnorm);
                        /* now calculate the additional weights that are needed for energy terms */
                        for(k=0;k<3;k++)
                        {
                            double v_ba = (P[j].Vel[k] - local.Vel[k]) / All.cf_atime; // relative gas-star velocity //
                            vel_ba_2 += v_ba*v_ba; // magnitude of velocity vector (for corrected post-shock energies to distribute)
                            cos_vel_ba_pcoupled += v_ba * pvec[k]/pnorm; // direction of ejecta [after correction loop]
                        }
                        wk_vec[7] = wk * vel_ba_2; // phi_0 term : residual KE term from mass-coupling for {small, second-order} energy correction
                        wk_vec[8] = sqrt(pnorm * P[j].Mass) * cos_vel_ba_pcoupled; // beta_0 term : cross-term for momentum coupling effect on energy-coupling
                        wk_vec[9] = pnorm * cos_vel_ba_pcoupled / v_cooling; // calculate the beta term as if all particles hit terminal: more accurate result in that limit
                        wk_vec[10] = pnorm; // normalization (so that we can divide by its sum to properly normalize the beta_egy and beta_cool quantities)
                    }
                    for(k=0;k<AREA_WEIGHTED_SUM_ELEMENTS;k++) {out.Area_weighted_sum[k] += wk_vec[k];}
                    continue;
                }
                // NOW do the actual feedback calculation //
                wk *= wk_norm; // this way wk matches the value summed above for the weighting //
                
                if((wk <= 0)||(isnan(wk))) continue;
                
                /* define initial mass and ejecta velocity in this 'cone' */
                double pnorm = 0, pvec[3] = {0};
                for(k=0; k<3; k++)
                {
                    double q = 0; int i1=2*k+1, i2=i1+1;
		            double q_i1 = fabs(local.Area_weighted_sum[i1]);
		            double q_i2 = fabs(local.Area_weighted_sum[i2]);
		            if((q_i1>MIN_REAL_NUMBER)&&(q_i2>MIN_REAL_NUMBER))
                    {
                        double rr = q_i2/q_i1;
                        double rr2 = rr * rr;
                        if(wk_vec[i1] != 0)
                        {
                            q += wk_norm * wk_vec[i1] * sqrt(0.5*(1.0+rr2));
                        } else {
                            q += wk_norm * wk_vec[i2] * sqrt(0.5*(1.0+1.0/rr2));
                        }
                    } else {
                        q += wk_norm * (wk_vec[i1] + wk_vec[i2]);
                    }
                    pvec[k] = -q;
                    pnorm += pvec[k]*pvec[k];
                }
                pnorm = sqrt(pnorm); // this (vector norm) is the new 'weight function' for our purposes
                dM_ejecta_in = pnorm * local.Msne;
                double mj_preshock = P[j].Mass;
                double massratio_ejecta = dM_ejecta_in / (dM_ejecta_in + P[j].Mass);
                
                /* inject actual mass from mass return */
                if(P[j].Hsml<=0) {if(SphP[j].Density>0){SphP[j].Density*=(1+dM_ejecta_in/P[j].Mass);} else {SphP[j].Density=dM_ejecta_in*kernel.hinv3;}} else {SphP[j].Density+=kernel_zero*dM_ejecta_in*hinv3_j;}
                SphP[j].Density *= 1 + dM_ejecta_in/P[j].Mass; // inject mass at constant particle volume //
                P[j].Mass += dM_ejecta_in;
                out.M_coupled += dM_ejecta_in;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
                SphP[j].MassTrue += dM_ejecta_in;
#endif
#ifdef METALS
                /* inject metals */
                for(k=0;k<NUM_METAL_SPECIES;k++) {P[j].Metallicity[k]=(1-massratio_ejecta)*P[j].Metallicity[k] + massratio_ejecta*local.yields[k];}
                if(feedback_type == 2) continue; // for r-process, nothing left here to bother coupling //
#endif
#if defined(COSMIC_RAYS) && defined(GALSF_FB_SNE_HEATING)
                /* inject cosmic rays */
                SphP[j].CosmicRayEnergy += pnorm * CR_energy_to_inject;
                SphP[j].CosmicRayEnergyPred += pnorm * CR_energy_to_inject;
#endif
                
                /* inject the post-shock energy and momentum (convert to specific units as needed first) */
#ifdef GALSF_TURNOFF_COOLING_WINDS
                /* if the sub-grid 'cooling turnoff' model is enabled, turn off cooling for the 'blastwave timescale' */
                dP = 7.08 * pow(Esne51*SphP[j].Density*density_to_n,0.34) * pow(SphP[j].Pressure*pressure_to_p4,-0.70) / (All.UnitTime_in_Megayears/All.HubbleParam);
                if(dP>SphP[j].DelayTimeCoolingSNe) SphP[j].DelayTimeCoolingSNe=dP;
#else
                /* inject momentum: account for ejecta being energy-conserving inside the cooling radius (or Hsml, if thats smaller) */
                double wk_m_cooling = pnorm * m_cooling; // effective cooling mass for this particle
                double boost_max = sqrt(1 + wk_m_cooling / dM_ejecta_in); // terminal momentum boost-factor
                double boost_egycon = sqrt(1 + mj_preshock / dM_ejecta_in); // energy-conserving limit for coupling through neighbors
                double mom_boost_fac = 1;
                if(feedback_type == 0)
                {
                    boost_max *= psi_cool; // appropriately re-weight boost to avoid energy conservation errors [cooling-limit]
                    boost_egycon *= psi_egycon; // appropriately re-weight boost to avoid energy conservation errors [energy-conserving-limit]
                    if((wk_m_cooling < mj_preshock) || (boost_max < boost_egycon)) {mom_boost_fac = boost_max;} else {mom_boost_fac = boost_egycon;} // limit to cooling case if egy-conserving exceeds terminal boost, or coupled mass short of cooling mass
                    if(mom_boost_fac < 1) {mom_boost_fac=1;} // impose lower limit of initial ejecta momentum
                } else {
                    mom_boost_fac = DMIN(boost_egycon , boost_max); // simply take minimum - nothing fancy for winds
                }
                
                /* save summation values for outputs */
                dP = local.unit_mom_SNe / P[j].Mass * pnorm;
                dP_sum += dP;
                dP_boost_sum += dP * mom_boost_fac;

                /* actually do the injection */
                double mom_prefactor =  mom_boost_fac * massratio_ejecta * (All.cf_atime*v_ejecta_eff) / pnorm; // this gives the appropriately-normalized tap-able momentum from the energy-conserving solution
                double KE_initial = 0, KE_final = 0;
                for(k=0; k<3; k++)
                {
                    double d_vel = mom_prefactor * pvec[k] + massratio_ejecta*(local.Vel[k] - P[j].Vel[k]); // local.Vel term from extra momentum of moving star, P[j].Vel term from going from momentum to velocity boost with added mass
                    KE_initial += P[j].Vel[k]*P[j].Vel[k];
                    P[j].Vel[k] += d_vel;
                    SphP[j].VelPred[k] += d_vel;
                    KE_final += P[j].Vel[k]*P[j].Vel[k];
                }
                /* now calculate the residual energy and add it as thermal */
                KE_initial *= 0.5 * mj_preshock * All.cf_a2inv;
                KE_final *= 0.5 * P[j].Mass * All.cf_a2inv;
                double E_sne_initial = pnorm * Energy_injected_codeunits;
                double d_Egy_internal = KE_initial + E_sne_initial - KE_final;
                if(d_Egy_internal < 0.5*E_sne_initial) {d_Egy_internal = 0.5*E_sne_initial;}
                /* if coupling radius > R_cooling, account for thermal energy loss in the post-shock medium: from Thornton et al. thermal energy scales as R^(-6.5) for R>R_cool */
                double r_eff_ij = kernel.r - Get_Particle_Size(j);
                if(r_eff_ij > RsneKPC) {d_Egy_internal *= RsneKPC_3 / (r_eff_ij*r_eff_ij*r_eff_ij);}
                d_Egy_internal /= P[j].Mass; // convert to specific internal energy, finally //
                if(d_Egy_internal > 0) {SphP[j].InternalEnergy += d_Egy_internal; SphP[j].InternalEnergyPred += d_Egy_internal; E_coupled += d_Egy_internal;}

#ifdef PM_HIRES_REGION_CLIPPING
                double dP=0; for(k=0;k<3;k++) dP+=P[j].Vel[k]*P[j].Vel[k]; dP=sqrt(dP);
                if(dP>5.e9*All.cf_atime/All.UnitVelocity_in_cm_per_s) P[j].Mass=0;
                if(dP>1.e9*All.cf_atime/All.UnitVelocity_in_cm_per_s) for(k=0;k<3;k++) P[j].Vel[k]*=(1.e9*All.cf_atime/All.UnitVelocity_in_cm_per_s)/dP;
#endif

#endif
                
            } // for(n = 0; n < numngb; n++)
        } // while(startnode >= 0)
        
        if(mode == 1)
        {
            listindex++;
            if(listindex < NODELISTLENGTH)
            {
                startnode = AddFBDataGet[target].NodeList[listindex];
                if(startnode >= 0)
                    startnode = Nodes[startnode].u.d.nextnode;	/* open it */
            }
        } // if(mode == 1)
    } // while(startnode >= 0)
    
    /* Now collect the result at the right place */
    if(mode == 0)
        out2particle_addFB(&out, target, 0, feedback_type);
    else
        AddFBDataResult[target] = out;
    
    return 0;
} // int addFB_evaluate

#endif // USE_ORIGINAL_FIRE2_SNE_COUPLING_SCHEME else


int addFB_evaluate_active_check(int i, int feedback_type);
int addFB_evaluate_active_check(int i, int feedback_type)
{
    if(P[i].Type <= 1) return 0;
    if(P[i].Mass <= 0) return 0;
    if(PPP[i].Hsml <= 0) return 0;
    if(PPP[i].NumNgb <= 0) return 0;
#ifdef GALSF_FB_SNE_HEATING
    if(P[i].SNe_ThisTimeStep>0) {if(feedback_type<0 || feedback_type==0) return 1;}
#endif
#ifdef GALSF_FB_GASRETURN
    if(P[i].MassReturn_ThisTimeStep>0) {if(feedback_type<0 || feedback_type==1) return 1;}
#endif
#ifdef GALSF_FB_RPROCESS_ENRICHMENT
    if(P[i].RProcessEvent_ThisTimeStep>0) {if(feedback_type<0 || feedback_type==2) return 1;}
#endif
    return 0;
}


void *addFB_evaluate_primary(void *p, int feedback_type)
{
#define CONDITION_FOR_EVALUATION if(addFB_evaluate_active_check(i,feedback_type)==1)
#define EVALUATION_CALL addFB_evaluate(i, 0, exportflag, exportnodecount, exportindex, ngblist, feedback_type)
#include "../system/code_block_primary_loop_evaluation.h"
#undef CONDITION_FOR_EVALUATION
#undef EVALUATION_CALL
}
void *addFB_evaluate_secondary(void *p, int feedback_type)
{
#define EVALUATION_CALL addFB_evaluate(j, 1, &dummy, &dummy, &dummy, ngblist, feedback_type);
#include "../system/code_block_secondary_loop_evaluation.h"
#undef EVALUATION_CALL
}







void determine_where_SNe_occur()
{
    int i;
    double dt,star_age,agemin,agebrk,agemax,RSNe,p,n_sn_0,RSNeFac;
    double npossible,nhosttotal,ntotal,ptotal,dtmean,rmean;
    npossible=nhosttotal=ntotal=ptotal=dtmean=rmean=0;
    double mpi_npossible,mpi_nhosttotal,mpi_ntotal,mpi_ptotal,mpi_dtmean,mpi_rmean;
    mpi_npossible=mpi_nhosttotal=mpi_ntotal=mpi_ptotal=mpi_dtmean=mpi_rmean=0;
#ifdef GALSF_FB_GASRETURN
    double D_RETURN_FRAC = 0.01; // fraction of particle mass to return on a recycling step //
#ifdef SINGLE_STAR_FORMATION
    D_RETURN_FRAC = 1.0e-7; // needs to be much smaller to have quasi-continuous winds on these scales //
#endif
#endif
    
    if(All.Time<=0) return;
    
    // basic variables we will use //
    agemin=0.003401; agebrk=0.01037; agemax=0.03753; // in Gyr //
    // converts rate to code units //
    RSNeFac=(All.UnitTime_in_Megayears/All.HubbleParam) * (All.UnitMass_in_g/All.HubbleParam)/SOLAR_MASS;
    
    // loop over particles //
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        P[i].SNe_ThisTimeStep=0;
#ifdef GALSF_FB_GASRETURN
        P[i].MassReturn_ThisTimeStep=0;
#endif
#ifdef GALSF_FB_RPROCESS_ENRICHMENT
        P[i].RProcessEvent_ThisTimeStep=0;
#endif
        if(All.ComovingIntegrationOn) if(P[i].Type != 4) continue;
        if(All.ComovingIntegrationOn==0) if((P[i].Type<2)||(P[i].Type>4)) continue;
        if(P[i].Mass<=0) continue;
        
        // get particle timestep //
#ifndef WAKEUP
        dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval / All.cf_hubble_a; // dloga to dt_physical
#else
        dt = P[i].dt_step * All.Timebase_interval / All.cf_hubble_a; //
#endif
        if(dt<=0) continue;
        
        // now use the simple tabulated SNe rates //
        RSNe=0.0;
        star_age = evaluate_stellar_age_Gyr(P[i].StellarAge);
        if(star_age<=0) continue;
        npossible++;
        
        // SNe component //
        if(All.SNeIIEnergyFrac>0)
        {
#ifdef SINGLE_STAR_FORMATION
	    /* here we are determining SNe for individual stars, so it happens deterministically at the end of their lives */
            double m_sol = P[i].Mass * (All.UnitMass_in_g / (All.HubbleParam * SOLAR_MASS)); // M/Msun 
	        P[i].SNe_ThisTimeStep = 0;
	        if(m_sol > 8.) // minimum mass for SNe
	        { 
	            double l_sol = bh_lum_bol(0,P[i].Mass,i) * (All.UnitEnergy_in_cgs / (All.UnitTime_in_s * SOLAR_LUM)); // L/Lsun
	            double lifetime = 9.6 * (m_sol/l_sol); // standard lifetime (in Gyr): this gives first SNe at 3Myr
		        if(star_age >= lifetime) {P[i].SNe_ThisTimeStep = 1; ntotal++; nhosttotal++;}
	        }
#else 
	    /* here we are determining an expected SNe rate, so SNe occur stochastically but with an age dependence in the population */
            if(star_age > agemin)
            {
                if((star_age>=agemin)&&(star_age<=agebrk))
                    RSNe = 5.408e-4; // NSNe/Myr *if* each SNe had exactly 10^51 ergs; really from the energy curve //
                if((star_age>=agebrk)&&(star_age<=agemax))
                    RSNe = 2.516e-4; // this is for a 1 Msun population //
                // add contribution from Type-Ia //
                if(star_age>agemax)
                    RSNe = 5.3e-8 + 1.6e-5*exp(-0.5*((star_age-0.05)/0.01)*((star_age-0.05)/0.01));
                // delayed population (constant rate)  +  prompt population (gaussian) //
                p = dt * (RSNe*RSNeFac) * P[i].Mass;
                double renorm = calculate_relative_light_to_mass_ratio_from_imf(i);
#ifdef GALSF_SFR_IMF_SAMPLING
                if(star_age < agemax)
                {
                    p *= renorm; // multiplies by number of O-stars relative to expectation
                } else {
                    if(P[i].IMF_NumMassiveStars > 0) {p += renorm*dt*(2.516e-4*RSNeFac)*P[i].Mass;} // account for residual O-stars
                }
#else
                if(star_age < agemax) {p *= renorm;}
#endif
                ptotal += p;
                n_sn_0 = (float)floor(p);
                p -= n_sn_0;
                if(get_random_number(P[i].ID + 6) < p) n_sn_0++;
#ifdef GALSF_SFR_IMF_SAMPLING
                // limit to number of O-stars for SNe //
                if((star_age < agemax) && (P[i].IMF_NumMassiveStars < n_sn_0)) {n_sn_0 = P[i].IMF_NumMassiveStars;}
                // lose an O-star for every SNe //
                if(P[i].IMF_NumMassiveStars > 0) {P[i].IMF_NumMassiveStars = DMAX(0 , P[i].IMF_NumMassiveStars - n_sn_0);}
#endif
                P[i].SNe_ThisTimeStep = n_sn_0;
                
                rmean += RSNe;
                ntotal += n_sn_0;
                if(n_sn_0>0) nhosttotal++;
            } // if(star_age > agemin) //
#endif
        }
        dtmean += dt;
        
#ifdef GALSF_FB_GASRETURN
        // Stellar Winds component //
        if(All.GasReturnFraction>0)
        {
#ifdef SINGLE_STAR_FORMATION
	        /* use a standard scaling from e.g. Castor, Abbot, & Klein */
	        double L_sol = bh_lum_bol(0, P[i].Mass, i) * All.UnitEnergy_in_cgs / (All.UnitTime_in_s * SOLAR_LUM); // L in solar
	        double M_sol = P[i].Mass * All.UnitMass_in_g / (All.HubbleParam * SOLAR_MASS); // M in solar
	        double gam = DMIN(0.5,3.2e-5*L_sol/M_sol); // Eddington factor (~L/Ledd for winds), capped at 1/2 for sanity reasons
	        double alpha = 0.5 + 0.4/(1. + 16./M_sol); // approximate scaling for alpha factor with stellar type (weak effect)
	        double q0 = (1.-alpha)*gam / (1.-gam); double k0=1./30.; //k is a normalization factor in the model
	        double mdot = 2.338 * alpha * pow(L_sol,7./8.) * pow(M_sol,0.1845) * (1./q0) * pow(q0*k0,1./alpha); // in Msun/Gyr
	        p = mdot / M_sol; // mass fraction returned per Gyr
            p *= All.GasReturnFraction * (dt*0.001*All.UnitTime_in_Megayears/All.HubbleParam); // fraction of particle mass expected to return in the timestep //
            p = 1.0 - exp(-p); // need to account for p>1 cases //
#else
            p=0.0;
            double ZZ = P[i].Metallicity[0]/All.SolarAbundances[0];
            if(ZZ>3) {ZZ=3;}
            if(ZZ<0.01) {ZZ=0.01;}
            if(star_age<=0.001){p=11.6846;} else {
                if(star_age<=0.0035){p=11.6846*ZZ*
                    pow(10.,1.838*(0.79+log10(ZZ))*(log10(star_age)-(-3.00)));} else {
                        if(star_age<=0.1){p=72.1215*pow(star_age/0.0035,-3.25)+0.0103;} else {
                            p=1.03*pow(star_age,-1.1)/(12.9-log(star_age));
                        }}}
            if(star_age < 0.1) {p *= calculate_relative_light_to_mass_ratio_from_imf(i);} // late-time independent of massive stars
            p *= All.GasReturnFraction * (dt*0.001*All.UnitTime_in_Megayears/All.HubbleParam); // fraction of particle mass expected to return in the timestep //
            p = 1.0 - exp(-p); // need to account for p>1 cases //
            p *= 1.4 * 0.291175; // to give expected return fraction from stellar winds alone (~17%) //

            /* // updated fit from M Grudic. More accurate for early times. 
               //     Needs to add the above call for later times (t >~ 0.02-0.1 Gyr) since late-time AGB loss is not strongly
               //     metallicity-dependent (as fit below only includes line-driven winds).
            double f1 = 4.68 * pow(ZZ, 0.87); // fit for fractional mass loss in first 1.5Myr
            double f3 = 0.44 * pow(ZZ, 0.77); // fit fractional mass loss from 20Myr onward
            if(star_age<=0.0015){p = f1;} else {
                if(star_age<=0.004){p = f1 * pow(star_age/0.0015,2.1);} else {
                    if(star_age<=0.02){p = f1 * 7.844 * pow(star_age/0.004, 0.621335*log(0.1275*f3/f1));} else {
                        p = f3 * pow(star_age/0.02, -1.1);
                    }}}
             if(star_age < 0.1) {p *= calculate_relative_light_to_mass_ratio_from_imf(i);} // late-time independent of massive stars
             p *= All.GasReturnFraction * (dt*0.001*All.UnitTime_in_Megayears/All.HubbleParam); // fraction of particle mass expected to return in the timestep //
             p = 1.0 - exp(-p); // need to account for p>1 cases //
            */
#endif
	        P[i].MassReturn_ThisTimeStep = 0; // zero mass return out
	        double n_wind_0 = (double)floor(p/D_RETURN_FRAC); // if p >> return frac, should have > 1 event, so we inject the correct wind mass
            p -= n_wind_0*D_RETURN_FRAC;
	        P[i].MassReturn_ThisTimeStep += n_wind_0*D_RETURN_FRAC; // add this in, then determine if there is a 'remainder' to be added as well
            if(get_random_number(P[i].ID + 5) < p/D_RETURN_FRAC) {P[i].MassReturn_ThisTimeStep += D_RETURN_FRAC;}
        }
#endif
        
#ifdef GALSF_FB_RPROCESS_ENRICHMENT
        p=0.0;
        /* we'll use the maximum rate here, then in the -yields- setting, 'cut' each down to its sub-population */
        if(star_age>=0.003) // rate is zero at <3e6 yr
        {
            p = 3.0e-5 / (1000.*star_age); // Nsne/Myr for a 1 Msun population
            p *= dt * RSNeFac * P[i].Mass;
            n_sn_0 = (float)floor(p);
            p -= n_sn_0;
            if(get_random_number(P[i].ID + 7) < p) n_sn_0++;
            P[i].RProcessEvent_ThisTimeStep = n_sn_0;
        }
#endif
        
    } // for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i]) //
    
    
    MPI_Reduce(&dtmean, &mpi_dtmean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&rmean, &mpi_rmean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ptotal, &mpi_ptotal, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nhosttotal, &mpi_nhosttotal, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ntotal, &mpi_ntotal, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&npossible, &mpi_npossible, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask == 0)
    {
#ifdef IO_REDUCED_MODE
        if(mpi_ntotal > 0 && mpi_nhosttotal > 0 && mpi_dtmean > 0)
#endif
        if(mpi_npossible>0)
        {
            mpi_dtmean /= mpi_npossible;
            mpi_rmean /= mpi_npossible;
            fprintf(FdSneIIHeating, "%lg %g %g %g %g %g %g \n",
                    All.Time,mpi_npossible,mpi_nhosttotal,mpi_ntotal,mpi_ptotal,mpi_dtmean,mpi_rmean);
        }
#ifdef IO_REDUCED_MODE
        if(All.HighestActiveTimeBin == All.HighestOccupiedTimeBin)
#endif
        {fflush(FdSneIIHeating);}
} // if(ThisTask == 0) //
    
} // void determine_where_SNe_occur() //



#ifdef GALSF_GASOLINE_RADHEATING
/* this routine copies the 'heating' term from young stars included in the Stinson+ 2013 GASOLINE model:
 the integrated luminosity from all stars with age<4 Myr is coupled directly as a heating term to the
 gas, with some efficiency parameter */
void luminosity_heating_gasoline(void)
{
    double Gasoline_LumHeating_Efficiency = 0.1;
    // coupling 'efficiency' (currently hard-coded to 10% to match their model, can change) //
    double dt,star_age,dE,dE_j,wtsum,h;
    int startnode, numngb, i, j, n, dummy=0;
    
    if(All.Time<=0) return;
    Ngblist = (int *) mymalloc("Ngblist",NumPart * sizeof(int));
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if((P[i].Type == 4)||((All.ComovingIntegrationOn==0)&&((P[i].Type == 2)||(P[i].Type==3))))
        {
#ifndef WAKEUP
            dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval / All.cf_hubble_a;
#else
            dt = P[i].dt_step * All.Timebase_interval / All.cf_hubble_a;
#endif
            star_age = evaluate_stellar_age_Gyr(P[i].StellarAge);
            if((star_age < 0.004)&&(dt>0)&&(P[i].Mass>0))
            {
                dE = evaluate_l_over_m_ssp(star_age) * calculate_relative_light_to_mass_ratio_from_imf(i);
                dE *= Gasoline_LumHeating_Efficiency;
                dE *= (4.0/2.0) * (P[i].Mass*All.UnitMass_in_g/All.HubbleParam); // L in CGS
                dE *= (dt*All.UnitTime_in_s/All.HubbleParam) / (All.UnitEnergy_in_cgs/All.HubbleParam); // dE in code units
                
                h = 1.0*P[i].Hsml;
                numngb=ngb_treefind_variable_threads(P[i].Pos,h,-1,&startnode,0,&dummy,&dummy,&dummy,Ngblist);
                wtsum = 0.0;
                if(numngb>0)
                {
                    for(n = 0; n < numngb; n++)
                    {
                        j = Ngblist[n];
                        if(P[j].Type == 0 && P[j].Mass > 0)
                        {
                            wtsum += 1.0;
                        }
                    }
                    if(wtsum>0)
                    {
                        dE /= wtsum;
                        for(n = 0; n < numngb; n++)
                        {
                            j = Ngblist[n];
                            if(P[j].Type == 0 && P[j].Mass > 0)
                            {
                                dE_j = dE / P[j].Mass;
                                SphP[j].InternalEnergy += dE_j;
                                SphP[j].InternalEnergyPred += dE_j;
                            }
                        }
                    } // if(wtsum>0)
                } // if(numngb>0)
            } // if((star_age < 0.004)&&(dt>0)&&(P[i].Mass>0))
        } // if((P[i].Type == 4)||(P[i].Type == 2)||(P[i].Type == 3))
    } // for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    myfree(Ngblist);
} // void luminosity_heating_gasoline(void)
#endif


#endif /* GALSF_FB_SNE_HEATING */

