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
#ifdef SPHEQ_DENSITY_INDEPENDENT_SPH
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
#endif




