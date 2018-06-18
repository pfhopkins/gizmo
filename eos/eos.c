#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "../allvars.h"
#include "../proto.h"

/* Routines for gas equation-of-state terms (collects things like calculation of gas pressure)
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */


/* this pair of functions: 'return_user_desired_target_density' and 'return_user_desired_target_pressure' should be used
 together with 'HYDRO_GENERATE_TARGET_MESH'. This will attempt to move the mesh and mass
 towards the 'target' pressure profile. Use this to build your ICs.
 The 'desired' pressure and density as a function of particle properties (most commonly, position) should be provided in the function below */
double return_user_desired_target_density(int i)
{
    return 1; // uniform density everywhere -- will try to generate a glass //
    /*
     // this example would initialize a constant-density (density=rho_0) spherical cloud (radius=r_cloud) with a smooth density 'edge' (width=interp_width) surrounded by an ambient medium of density =rho_0/rho_contrast //
     double dx=P[i].Pos[0]-boxHalf_X, dy=P[i].Pos[1]-boxHalf_Y, dz=P[i].Pos[2]-boxHalf_Z, r=sqrt(dx*dx+dy*dy+dz*dz);
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
     double dx=P[i].Pos[0]-boxHalf_X, dy=P[i].Pos[1]-boxHalf_Y, dz=P[i].Pos[2]-boxHalf_Z, r=sqrt(dx*dx+dy*dy+dz*dz);
     double b = 2.; return 2.*M_PI/fabs((3.-b)*(1.-b)) * pow(return_user_desired_target_density(i),2) * r*r;
     */
}




/* return the pressure of particle i */
double get_pressure(int i)
{
    MyFloat press = GAMMA_MINUS1 * SphP[i].InternalEnergyPred * Particle_density_for_energy_i(i); /* ideal gas EOS (will get over-written it more complex EOS assumed) */
    
#ifdef GALSF_EFFECTIVE_EQS
    /* modify pressure to 'interpolate' between effective EOS and isothermal, with the Springel & Hernquist 2003 'effective' EOS */
    if(SphP[i].Density*All.cf_a3inv >= All.PhysDensThresh)
        press = All.FactorForSofterEQS * press + (1 - All.FactorForSofterEQS) * All.cf_afac1 * GAMMA_MINUS1 * SphP[i].Density * All.InitGasU;
#endif
    
    
#ifdef EOS_HELMHOLTZ
    /* pass the necessary quantities to wrappers for the Timms EOS */
    struct eos_input eos_in;
    struct eos_output eos_out;
    eos_in.rho  = SphP[i].Density;
    eos_in.eps  = SphP[i].InternalEnergyPred;
    eos_in.Ye   = SphP[i].Ye;
    eos_in.Abar = SphP[i].Abar;
    eos_in.temp = SphP[i].Temperature;
    int ierr = eos_compute(&eos_in, &eos_out);
    assert(!ierr);
    press              = eos_out.press;
    SphP[i].SoundSpeed = eos_out.csound;
    SphP[i].Temperature= eos_out.temp;
#endif

    
#ifdef EOS_TILLOTSON
    press = calculate_eos_tillotson(i);
#endif
    
#ifdef EOS_ENFORCE_ADIABAT
    press = EOS_ENFORCE_ADIABAT * pow(SphP[i].Density, GAMMA);
#endif
    
    
#ifdef COSMIC_RAYS
    press += Get_Particle_CosmicRayPressure(i);
    /* we will also compute the CR contribution to the effective soundspeed here */
    if((P[i].Mass > 0) && (SphP[i].Density>0) && (SphP[i].CosmicRayEnergyPred > 0))
    {
        SphP[i].SoundSpeed = sqrt(GAMMA*GAMMA_MINUS1 * SphP[i].InternalEnergyPred + GAMMA_COSMICRAY*GAMMA_COSMICRAY_MINUS1 * SphP[i].CosmicRayEnergyPred / P[i].Mass);
    } else {
        SphP[i].SoundSpeed = sqrt(GAMMA*GAMMA_MINUS1 * SphP[i].InternalEnergyPred);
    }
#ifdef COSMIC_RAYS_ALFVEN
    press += (GAMMA_ALFVEN_CRS-1) * SphP[i].Density * (SphP[i].CosmicRayAlfvenEnergy[0]+SphP[i].CosmicRayAlfvenEnergy[1]);
    SphP[i].SoundSpeed = sqrt(SphP[i].SoundSpeed*SphP[i].SoundSpeed + GAMMA_ALFVEN_CRS*(GAMMA_ALFVEN_CRS-1)*(SphP[i].CosmicRayAlfvenEnergy[0]+SphP[i].CosmicRayAlfvenEnergy[1])/P[i].Mass);
#endif
#endif
    
    
#if defined(EOS_TRUELOVE_PRESSURE) || defined(TRUELOVE_CRITERION_PRESSURE)
    /* add an artificial pressure term to suppress fragmentation at/below the explicit resolution scale */
    double h_eff = DMAX(Get_Particle_Size(i), All.ForceSoftening[0]/2.8); /* need to include latter to account for inter-particle spacing << grav soft cases */
    /* standard finite-volume formulation of this (note there is some geometric ambiguity about whether there should be a "pi" in the equation below, but this 
        can be completely folded into the (already arbitrary) definition of NJeans, so we just use the latter parameter */
    double NJeans = 4; // set so that resolution = lambda_Jeans/NJeans -- fragmentation with Jeans/Toomre scales below this will be artificially suppressed now
    double xJeans = (NJeans * NJeans / GAMMA) * All.G * h_eff*h_eff * SphP[i].Density * SphP[i].Density * All.cf_afac1/All.cf_atime;
    if(xJeans>press) press=xJeans;
    SphP[i].SoundSpeed = sqrt(GAMMA * press / Particle_density_for_energy_i(i));
#endif
    
    
#if defined(HYDRO_GENERATE_TARGET_MESH)
    press = return_user_desired_target_pressure(i) * (SphP[i].Density / return_user_desired_target_density(i)); // define pressure by reference to 'desired' fluid quantities //
    SphP[i].InternalEnergy = SphP[i].InternalEnergyPred = press / (GAMMA_MINUS1 * SphP[i].Density);
#endif
    
    return press;
}




/* trivial function to check if particle falls below the minimum allowed temperature */
void check_particle_for_temperature_minimum(int i)
{
    if(All.MinEgySpec)
    {
        if(SphP[i].InternalEnergy < All.MinEgySpec)
        {
            SphP[i].InternalEnergy = All.MinEgySpec;
            SphP[i].DtInternalEnergy = 0;
        }
    }
}



double INLINE_FUNC Particle_density_for_energy_i(int i)
{
#ifdef HYDRO_PRESSURE_SPH
    return SphP[i].EgyWtDensity;
#endif
    return SphP[i].Density;
}




double INLINE_FUNC Particle_effective_soundspeed_i(int i)
{
#ifdef EOS_GENERAL
    return SphP[i].SoundSpeed;
#endif
    /* if nothing above triggers, then we resort to good old-fashioned ideal gas */
    return sqrt(GAMMA * SphP[i].Pressure / Particle_density_for_energy_i(i));
}



#ifdef COSMIC_RAYS
double INLINE_FUNC Get_Particle_CosmicRayPressure(int i)
{
    if((P[i].Mass > 0) && (SphP[i].Density>0) && (SphP[i].CosmicRayEnergyPred > 0))
    {
        return GAMMA_COSMICRAY_MINUS1 * (SphP[i].CosmicRayEnergyPred * SphP[i].Density) / P[i].Mass; // cosmic ray pressure = (4/3-1) * e_cr = 1/3 * (E_cr/Vol) //
    } else {
        return 0;
    }
}


double Get_CosmicRayGradientLength(int i)
{
    /* now we need the cosmic ray pressure or energy density scale length, defined as :
        L = (e_cr + p_cr) / |gradient_p_cr| = cr_enthalpy / |gradient(p_cr)| */
    double CRPressureGradMag = 0.0;
    int k; for(k=0;k<3;k++) {CRPressureGradMag += SphP[i].Gradients.CosmicRayPressure[k]*SphP[i].Gradients.CosmicRayPressure[k];}
    /* limit the scale length: if too sharp, need a slope limiter at around the particle size */
    double L_gradient_min = Get_Particle_Size(i) * All.cf_atime;
    /* limit this scale length; if the gradient is too shallow, there is no information beyond a few smoothing lengths, so we can't let streaming go that far */
    double L_gradient_max = DMAX(200.*L_gradient_min, 100.0*PPP[i].Hsml*All.cf_atime);

    /* also, physically, cosmic rays cannot stream/diffuse with a faster coefficient than ~v_max*L_mean_free_path, where L_mean_free_path ~ 2.e20 * (cm^-3/n) */
    double nH_cgs = SphP[i].Density * All.cf_a3inv * ( All.UnitDensity_in_cgs * All.HubbleParam*All.HubbleParam ) / PROTONMASS ;
    double L_mean_free_path = (3.e25 / nH_cgs) / (All.UnitLength_in_cm / All.HubbleParam);
    L_gradient_max = DMIN(L_gradient_max, L_mean_free_path);
    
    double CRPressureGradScaleLength = Get_Particle_CosmicRayPressure(i) / sqrt(1.0e-33 + CRPressureGradMag) * All.cf_atime;
    if(CRPressureGradScaleLength > 0) {CRPressureGradScaleLength = 1.0/(1.0/CRPressureGradScaleLength + 1.0/L_gradient_max);} else {CRPressureGradScaleLength=0;}
    CRPressureGradScaleLength = sqrt(L_gradient_min*L_gradient_min + CRPressureGradScaleLength*CRPressureGradScaleLength);
    return CRPressureGradScaleLength; /* this is returned in -physical- units */
}

double Get_CosmicRayStreamingVelocity(int i)
{
    /* in the weak-field (high-beta) case, the streaming velocity is approximately the sound speed */
    double v_streaming = sqrt(GAMMA*GAMMA_MINUS1 * SphP[i].InternalEnergyPred); // thermal ion sound speed //
#ifdef MAGNETIC
    /* in the strong-field (low-beta) case, it's actually the Alfven velocity: interpolate between these */
    double vA_2 = 0.0; double cs_stream = v_streaming;
    int k; for(k=0;k<3;k++) {vA_2 += Get_Particle_BField(i,k)*Get_Particle_BField(i,k);}
    vA_2 *= All.cf_afac1 / (All.cf_atime * SphP[i].Density);
    v_streaming = DMIN(1.0e6*cs_stream, sqrt(cs_stream*cs_stream + vA_2));
#endif
    v_streaming *= All.cf_afac3; // converts to physical units and rescales according to chosen coefficient //
    return v_streaming;
}


double CosmicRay_Update_DriftKick(int i, double dt_entr, int mode)
{
    /* routine to do the drift/kick operations for CRs: mode=0 is kick, mode=1 is drift */
    if(dt_entr <= 0) {return 0;} // no update
    int k; double eCR, u0;
    if(mode==0) {eCR=SphP[i].CosmicRayEnergy; u0=SphP[i].InternalEnergy;} else {eCR=SphP[i].CosmicRayEnergyPred; u0=SphP[i].InternalEnergyPred;} // initial energy
    if(eCR < 0) {eCR=0;} // limit to physical values
    
#if defined(COSMIC_RAYS_M1) && !defined(COSMIC_RAYS_ALFVEN)
    // this is the exact solution for the CR flux-update equation over a finite timestep dt: it needs to be solved this way [implicitly] as opposed to explicitly for dt because in the limit of dt_cr_dimless being large, the problem exactly approaches the diffusive solution
    double DtCosmicRayFlux[3]={0}, flux[3]={0}, CR_veff[3]={0}, CR_vmag=0, q_cr = 0, cr_speed = COSMIC_RAYS_M1;// * (C/All.UnitVelocity_in_cm_per_s);
    cr_speed = DMAX( All.cf_afac3*SphP[i].MaxSignalVel , DMIN(COSMIC_RAYS_M1 , fabs(SphP[i].CosmicRayDiffusionCoeff)/(Get_Particle_Size(i)*All.cf_atime)));// * (C/All.UnitVelocity_in_cm_per_s);
    for(k=0;k<3;k++) {DtCosmicRayFlux[k] = -fabs(SphP[i].CosmicRayDiffusionCoeff) * (P[i].Mass/SphP[i].Density) * (SphP[i].Gradients.CosmicRayPressure[k]/GAMMA_COSMICRAY_MINUS1);}
#ifdef MAGNETIC // do projection onto field lines
    double B0[3]={0}, Bmag2=0, DtCRDotBhat=0;
    for(k=0;k<3;k++)
    {
        if(mode==0) {B0[k]=SphP[i].B[k];} else {B0[k]=SphP[i].BPred[k];}
        DtCRDotBhat += DtCosmicRayFlux[k] * B0[k]; Bmag2 += B0[k]*B0[k];
    }
    if(Bmag2 > 0) {for(k=0;k<3;k++) {DtCosmicRayFlux[k] = DtCRDotBhat * B0[k] / Bmag2;}}
#endif
    double dt_cr_dimless = dt_entr * cr_speed*cr_speed * GAMMA_COSMICRAY_MINUS1 / (MIN_REAL_NUMBER + fabs(SphP[i].CosmicRayDiffusionCoeff));
    if((dt_cr_dimless > 0)&&(dt_cr_dimless < 20.)) {q_cr = exp(-dt_cr_dimless);} // factor for CR interpolation
    if(mode==0) {for(k=0;k<3;k++) {flux[k]=SphP[i].CosmicRayFlux[k];}} else {for(k=0;k<3;k++) {flux[k]=SphP[i].CosmicRayFluxPred[k];}}
    for(k=0;k<3;k++) {flux[k] = q_cr*flux[k] + (1.-q_cr)*DtCosmicRayFlux[k];} // updated flux
    for(k=0;k<3;k++) {CR_veff[k]=flux[k]/(eCR+MIN_REAL_NUMBER); CR_vmag+=CR_veff[k]*CR_veff[k];} // effective streaming speed
    if((CR_vmag <= 0) || (isnan(CR_vmag))) // check for valid numbers
    {
        for(k=0;k<3;k++) {flux[k]=0; for(k=0;k<3;k++) {CR_veff[k]=0;}} // zero if invalid
    } else {
        CR_vmag = sqrt(CR_vmag);
        if(CR_vmag > cr_speed) {for(k=0;k<3;k++) {flux[k]*=cr_speed/CR_vmag; CR_veff[k]*=cr_speed/CR_vmag;}} // limit flux to free-streaming speed [as with RT]
    }
    if(mode==0) {for(k=0;k<3;k++) {SphP[i].CosmicRayFlux[k]=flux[k];}} else {for(k=0;k<3;k++) {SphP[i].CosmicRayFluxPred[k]=flux[k];}}
#endif
    
    
    // now update all scalar fields (CR energies and Alfvenic energies, if those are followed) from fluxes and adiabatic terms //
    int q_whichupdate, q_N_updates = 1;
#ifdef COSMIC_RAYS_ALFVEN
    q_N_updates = 3; // update the Alfvenic energy terms from (0) advection with gas [already solved], (1) their fluxes, and (2) their adiabatic terms. this should be basically identical to the CR term.
#endif
    for(q_whichupdate=0; q_whichupdate<q_N_updates; q_whichupdate++)
    {
        // first update the CR energies from fluxes. since this is positive-definite, some additional care is needed //
        double dCR_dt = SphP[i].DtCosmicRayEnergy, gamma_eff = GAMMA_COSMICRAY, eCR_tmp = eCR;
#ifdef COSMIC_RAYS_ALFVEN
        if(q_whichupdate>0) {dCR_dt=SphP[i].DtCosmicRayAlfvenEnergy[q_whichupdate-1]; gamma_eff=GAMMA_ALFVEN_CRS; if(mode==0) {eCR_tmp=SphP[i].CosmicRayAlfvenEnergy[q_whichupdate-1];} else {eCR_tmp=SphP[i].CosmicRayAlfvenEnergyPred[q_whichupdate-1];}}
#endif
        double dCR = dCR_dt*dt_entr, dCRmax = 1.e10*(eCR_tmp+MIN_REAL_NUMBER);
#if defined(GALSF) && !defined(COSMIC_RAYS_ALFVEN)
        dCRmax = DMAX(2.0*eCR_tmp , 0.1*u0*P[i].Mass);
#endif
        if(dCR > dCRmax) {dCR=dCRmax;} // don't allow excessively large values
        if(dCR < -eCR_tmp) {dCR=-eCR_tmp;} // don't allow it to go negative
        eCR_tmp += dCR; if((eCR_tmp<0)||(isnan(eCR_tmp))) {eCR_tmp=0;} // check against energy going negative or nan
        if(q_whichupdate==0) {if(mode==0) {SphP[i].CosmicRayEnergy=eCR_tmp;} else {SphP[i].CosmicRayEnergyPred=eCR_tmp;}} // updated energy
#ifdef COSMIC_RAYS_ALFVEN
        if(q_whichupdate>0) {if(mode==0) {SphP[i].CosmicRayAlfvenEnergy[q_whichupdate-1]=eCR_tmp;} else {SphP[i].CosmicRayAlfvenEnergyPred[q_whichupdate-1]=eCR_tmp;}} // updated energy
#endif
        double eCR_0 = eCR_tmp; // save this value for below
        
        // now need to account for the adiabatic heating/cooling of the 'fluid', here, with gamma=gamma_eff //
        double d_div = (-(gamma_eff-1.) * P[i].Particle_DivVel*All.cf_a2inv) * dt_entr;
        if(All.ComovingIntegrationOn) {d_div += (-3.*(gamma_eff-1.) * All.cf_hubble_a) * dt_entr;} /* adiabatic term from Hubble expansion (needed for cosmological integrations */
        double dCR_div = DMIN(eCR_tmp*d_div , 0.5*u0*P[i].Mass); // limit so don't take away all the gas internal energy [to negative values]
        if(dCR_div + eCR_tmp < 0) {dCR_div = -eCR_tmp;} // check against energy going negative
        eCR_tmp += dCR_div; if((eCR_tmp<0)||(isnan(eCR_tmp))) {eCR_tmp=0;} // check against energy going negative or nan
        dCR_div = eCR_tmp - eCR_0; // actual change that is going to be applied
        if(q_whichupdate==0) {if(mode==0) {SphP[i].CosmicRayEnergy += dCR_div; SphP[i].InternalEnergy -= dCR_div/P[i].Mass;} else {SphP[i].CosmicRayEnergyPred += dCR_div; SphP[i].InternalEnergyPred -= dCR_div/P[i].Mass;}}
#ifdef COSMIC_RAYS_ALFVEN
        if(q_whichupdate>0) {if(mode==0) {SphP[i].CosmicRayAlfvenEnergy[q_whichupdate-1] += dCR_div; SphP[i].InternalEnergy -= dCR_div/P[i].Mass;} else {SphP[i].CosmicRayAlfvenEnergyPred[q_whichupdate-1] += dCR_div; SphP[i].InternalEnergyPred -= dCR_div/P[i].Mass;}}
#endif
    }
    
    
#ifdef COSMIC_RAYS_ALFVEN
    double Z_charge_CR = 1, E_CRs_Gev = 1, keffective_over_rLinv = 1; // charge and energy and resonant Alfven wavenumber (in gyro units) of the CR population we're evolving

    // ok, the updates from [0] advection w gas, [1] fluxes, [2] adiabatic, [-] catastrophic (in cooling.c) are all set, just need exchange terms b/t CR and Alfven //
    double bhat[3], Bmag=0, Bmag_Gauss, clight_code=C/All.UnitVelocity_in_cm_per_s, Omega_gyro, eA[2], vA_code, vA2_c2, E_B, fac, K0, flux_G, fac_Omega, flux[3], f_CR, f_CR_dot_B, CR_vmag, cs_thermal, r_turb_driving, G_ion_neutral=0, G_turb_plus_linear_landau=0, G_nonlinear_landau_prefix=0;
    for(k=0;k<3;k++) {if(mode==0) {bhat[k]=SphP[i].B[k];} else {bhat[k]=SphP[i].BPred[k];}} // grab whichever B field we need for our mode
    if(mode==0) {eCR=SphP[i].CosmicRayEnergy; u0=SphP[i].InternalEnergy;} else {eCR=SphP[i].CosmicRayEnergyPred; u0=SphP[i].InternalEnergyPred;} // initial energy
    for(k=0;k<2;k++) {if(mode==0) {eA[k]=SphP[i].CosmicRayAlfvenEnergy[k];} else {eA[k]=SphP[i].CosmicRayAlfvenEnergyPred[k];}} // Alfven energy
    if(mode==0) {for(k=0;k<3;k++) {flux[k]=SphP[i].CosmicRayFlux[k];}} else {for(k=0;k<3;k++) {flux[k]=SphP[i].CosmicRayFluxPred[k];}} // load flux
    f_CR=0; f_CR_dot_B=0; for(k=0;k<3;k++) {f_CR+=flux[k]*flux[k]; f_CR_dot_B+=bhat[k]*flux[k];} // compute the magnitude of the flux density
    f_CR=sqrt(f_CR); if(f_CR_dot_B<0) {f_CR*=-1;} // initialize the flux density variable from the previous timestep, appropriately signed with respect to the b-field
    for(k=0;k<3;k++) {Bmag+=bhat[k]*bhat[k];} // compute magnitude
    Bmag = sqrt(Bmag); for(k=0;k<3;k++) {bhat[k]/=(MIN_REAL_NUMBER+Bmag);} // now it's bhat we have here
    Bmag *= SphP[i].Density/P[i].Mass * All.cf_a2inv; // convert to actual B in physical units
    Bmag_Gauss = Bmag * sqrt(4.*M_PI*All.UnitPressure_in_cgs*All.HubbleParam*All.HubbleParam); // turn it into Gauss
    Omega_gyro = 8987.34 * Bmag_Gauss * (Z_charge_CR/E_CRs_Gev) * (All.UnitTime_in_s/All.HubbleParam); // gyro frequency of the CR population we're evolving
    vA_code = sqrt( Bmag*Bmag / (SphP[i].Density*All.cf_a3inv) ); // Alfven speed^2 in code units [recall B units such that there is no 4pi here]
    cs_thermal = sqrt(GAMMA*(GAMMA-1.) * u0); // thermal sound speed at appropriate drift-time [in code units, physical]
    vA2_c2 = vA_code*vA_code / (clight_code*clight_code); // Alfven speed vs c_light
    E_B = MIN_REAL_NUMBER + 0.5*Bmag*Bmag * (P[i].Mass/(SphP[i].Density*All.cf_a3inv)); // B-field energy (energy density times volume, for ratios with energies above)
    fac_Omega = (3.*M_PI/16.) * Omega_gyro * (1.+2.*vA2_c2); // factor which will be used heavily below
    
    // non-adiabatic exchange terms between eA (Alfven) and thermal gas -- e.g. 'streaming losses' [Sa+/- in the Thomas+Pfrommer notation]
#ifdef COOLING
    /* ion-neutral damping: need thermodynamic information (neutral fractions, etc) to compute self-consistently */
    double ne=SphP[i].Ne, nh0=0, nHe0, nHepp, nhp, nHeII, temperature, mu_meanwt=1, rho=SphP[i].Density*All.cf_a3inv, rho_cgs=rho*All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;
    temperature = ThermalProperties(u0, rho, i, &mu_meanwt, &ne, &nh0, &nhp, &nHe0, &nHeII, &nHepp); // get thermodynamic properties
#ifdef GALSF_FB_HII_HEATING
    if(SphP[i].DelayTimeHII>0) {nh0=0;} // account for our effective ionization model here
#endif
    G_ion_neutral = 0.785e-12 * (rho_cgs/PROTONMASS) * nh0 * sqrt(temperature) * (All.UnitTime_in_s/All.HubbleParam); // need to get thermodynamic quantities [neutral fraction, temperature in Kelvin] to compute here -- // G_ion_neutral = (xiH + xiHe); // xiH = nH * siH * sqrt[(32/9pi) *kB*T*mH/(mi*(mi+mH))]
#endif
    if(Z_charge_CR > 1) {G_ion_neutral /= sqrt(2.*Z_charge_CR);}
    /* turbulent (anisotropic and linear landau) damping terms: need to know the turbulent driving scale: assume a cascade with a driving length equal to the pressure gradient scale length */
    r_turb_driving = 0; for(k=0;k<3;k++) {r_turb_driving += SphP[i].Gradients.Pressure[k]*SphP[i].Gradients.Pressure[k];} // compute gradient magnitude
    r_turb_driving = DMAX( SphP[i].Pressure / (MIN_REAL_NUMBER + sqrt(r_turb_driving)) , Get_Particle_Size(i) ) * All.cf_atime; // maximum of gradient scale length or resolution scale
    G_turb_plus_linear_landau = (vA_code + sqrt(M_PI/16.)*cs_thermal) * sqrt(Omega_gyro / (r_turb_driving * clight_code)); //G_turb_plus_linear_landau = ((vA_code + sqrt(M_PI/16.)*cs_thermal) / clight_code) * Omega_gyro * sqrt(r_gyro / r_turb_driving);
    double A_coeff = G_ion_neutral + G_turb_plus_linear_landau; fac = A_coeff * dt_entr; if((fac>20.)||(!isfinite(fac))) {fac=20.;} // get dimensionless time factor, limit to give non-nan values
    fac = exp(fac); // exponentiate factor needed below for evolution, limit to give non-nan values
    /* non-linear landau damping */
    G_nonlinear_landau_prefix = keffective_over_rLinv * sqrt(M_PI/8.) * (1./E_B) * Omega_gyro * (cs_thermal / clight_code);
    double facloss_p = 1. / (fac + (G_nonlinear_landau_prefix * eA[0] / (MIN_REAL_NUMBER + A_coeff)) * (fac-1.));
    double facloss_m = 1. / (fac + (G_nonlinear_landau_prefix * eA[1] / (MIN_REAL_NUMBER + A_coeff)) * (fac-1.));
    eCR += (1.-facloss_p)*eA[0] + (1.-facloss_m)*eA[1]; eA[0] *= facloss_p; eA[1] *= facloss_m;
    
    // exchange term between eCR and eA, strictly lossy for eA [Alfven] terms [the loss terms proportional to (e_cr+P_cr), in the Thomas+Pfrommer notation]
    fac = -fac_Omega * vA2_c2 * GAMMA_COSMICRAY * (eCR/E_B) * dt_entr; if((fac<-log(100.))||(!isfinite(fac))) {fac=-log(100.);} // prop to eA_pm = loss term, so solve that exactly, then add to ECR (gain term)
    eCR += (1.-exp(fac)) * (eA[0]+eA[1]); eA[0] *= exp(fac); eA[1] *= exp(fac);
    // exchange term between eCR and eA, need to check which 'side' is lossy [the loss terms proportional to f_cr, in the Thomas+Pfrommer notation]
    fac = fac_Omega * vA2_c2 * (fabs(f_CR)/(MIN_REAL_NUMBER + E_B*vA_code)) * dt_entr; if((fac>log(100.))||(!isfinite(fac))) {fac=log(100.);}  // limit factor
    double fac_CR_limiter = 0.90, fac_lim; if(f_CR > 0) {fac_lim = log(1 + fac_CR_limiter*eCR/(MIN_REAL_NUMBER+eA[0]));} else {fac_lim = log(1 + fac_CR_limiter*eCR/(MIN_REAL_NUMBER+eA[1]));}
    if(fac > fac_lim) {fac = fac_lim;} // ok this should ensure change in CR energy is not able to make it negative //
    if(f_CR < 0) {fac *= -1;} // normalize term correctly
    eCR += (1.-exp(fac))*eA[0] + (1.-exp(-fac))*eA[1]; eA[0] *= exp(fac); eA[1] *=exp(-fac);
    // assign the updated values
    if(mode==0) {SphP[i].CosmicRayEnergy=eCR;} else {SphP[i].CosmicRayEnergyPred=eCR;} // CR energy
    for(k=0;k<2;k++) {if(mode==0) {SphP[i].CosmicRayAlfvenEnergy[k]=eA[k];} else {SphP[i].CosmicRayAlfvenEnergyPred[k]=eA[k];}} // Alfven energy

    // ok now compute updates for the flux variable: this is split in two parts, one for change-of-basis for b-orientation, the second the usual split advection-diffusion update
    fac=0; for(k=0;k<3;k++) {fac += bhat[k] * (bhat[0]*SphP[i].Gradients.Velocity[k][0] + bhat[1]*SphP[i].Gradients.Velocity[k][1] + bhat[2]*SphP[i].Gradients.Velocity[k][2]);}
    if(All.ComovingIntegrationOn) {fac += All.cf_hubble_a;} // adds cosmological/hubble flow term here [not included in peculiar velocity gradient]
    fac *= -All.cf_a2inv*dt_entr; if(!isfinite(fac)) {fac=0;} else {if(fac>2.) {fac=2.;} else {if(fac<-2.) {fac=-2.;}}} // limit factor for change here, should be small given Courant factor
    f_CR *= exp(fac); // update flux term accordingly, before next step //
    // now the advection-diffusion update --- this is the important step //
    K0 = (fac_Omega/(clight_code*clight_code)) * ((eA[0]+eA[1])/E_B); // 1/effective diffusion coefficient
    fac = (COSMIC_RAYS_ALFVEN*COSMIC_RAYS_ALFVEN) * K0 * dt_entr; // d_tau [dimensionless time unit]
    if((fac > 20.)||(!isfinite(fac))) {fac = 20.;} // limit to prevent nan's or infinities
    flux_G=0; for(k=0;k<3;k++) {flux_G += bhat[k] * SphP[i].Gradients.CosmicRayPressure[k] * (P[i].Mass/SphP[i].Density) * (1./(MIN_REAL_NUMBER + K0));} // b.gradient[P] -- flux source term
    flux_G += GAMMA_COSMICRAY * ((eA[1]-eA[0])/(MIN_REAL_NUMBER + eA[0]+eA[1])) * vA_code * eCR; // add secondary source term from streaming
    f_CR = flux_G + (f_CR-flux_G) * exp(-fac); // now compute the actual solution
    CR_vmag=fabs(f_CR)/(MIN_REAL_NUMBER+eCR); if((CR_vmag<=0)||(!isfinite(CR_vmag))) {f_CR=0;} else {if(CR_vmag>COSMIC_RAYS_ALFVEN) {f_CR*=COSMIC_RAYS_ALFVEN/CR_vmag;}} // limit flux to maximal-streaming speed
    for(k=0;k<3;k++) {flux[k]=f_CR*bhat[k];} // assign directionality from b-field
    if(mode==0) {for(k=0;k<3;k++) {SphP[i].CosmicRayFlux[k]=flux[k];}} else {for(k=0;k<3;k++) {SphP[i].CosmicRayFluxPred[k]=flux[k];}} // assign to flux vector
#endif

    return 1;
}
#endif


