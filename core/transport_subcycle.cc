/* transport_subcycle.cc - implementation of transport (RT + CR) subcycling
 *
 * Flux exchange uses the standard code_block_xchange pattern with a transport-only
 * core function. This correctly handles cross-domain interactions via MPI.
 * The tree walk is repeated each sub-step (future optimization: cache neighbor lists).
 *
 * Written by Mike Grudic with Claude for GIZMO.
 */

#include <mpi.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"

#ifdef TRANSPORT_SUBCYCLE

#include "transport_subcycle.h"

/* ======================================================================================== */
/* ===== CODE_BLOCK_XCHANGE SETUP FOR TRANSPORT FLUX EXCHANGE ============================= */
/* ======================================================================================== */

/* kernel struct — same as hydro_toplevel.cc */
struct kernel_hydra
{
    Vec3<double> dp;
    double r, vsig, sound_i, sound_j;
    Vec3<double> dv; double vdotr2;
    double wk_i, wk_j, dwk_i, dwk_j;
    double h_i, h_j, dwk_ij, rho_ij_inv;
    double spec_egy_u_i;
#ifdef HYDRO_SPH
    double p_over_rho2_i;
#endif
#ifdef MAGNETIC
    double b2_i, b2_j;
    double alfven2_i, alfven2_j;
#ifdef HYDRO_SPH
    double mf_i, mf_j;
#endif
#endif
};


/* define the code_block_xchange names */
#define CORE_FUNCTION_NAME transport_flux_evaluate
#define INPUTFUNCTION_NAME particle2in_transport
#define OUTPUTFUNCTION_NAME out2particle_transport
#define CONDITIONFUNCTION_FOR_EVALUATION if((P[i].Type==0)&&(P[i].Mass>0))
#include "../system/code_block_xchange_initialize.h"


/* ===== INPUT STRUCT ===== */
static struct INPUT_STRUCT_NAME
{
    MyDouble Pos[3];
    Vec3<MyFloat> Vel;
    MyFloat Rad_E_gamma[N_RT_FREQ_BINS];
    MyFloat Rad_Kappa[N_RT_FREQ_BINS];
    MyFloat RT_DiffusionCoeff[N_RT_FREQ_BINS];
    MyDouble Mass;
    MyFloat Density;
    MyFloat KernelRadius;
    MyFloat ConditionNumber;
    MyFloat SoundSpeed;
#if defined(RT_EVOLVE_FLUX) || defined(HYDRO_SPH)
    SymmetricTensor2<MyDouble> ET[N_RT_FREQ_BINS];
#endif
#ifdef RT_EVOLVE_FLUX
    Vec3<MyFloat> Rad_Flux[N_RT_FREQ_BINS];
#endif
#ifdef RT_COMPGRAD_EDDINGTON_TENSOR
    struct { Vec3<MyDouble> Rad_E_gamma_ET[N_RT_FREQ_BINS]; } Gradients;
#endif
#ifdef RT_INFRARED
    MyFloat Radiation_Temperature;
#endif
#ifdef MAGNETIC
    Vec3<MyFloat> BPred;
#endif
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    Vec3<MyFloat> ParticleVel;
#endif
    SymmetricTensor2<MyDouble> NV_T;  /* gradient matrix for face area computation */
    int NodeList[NODELISTLENGTH];     /* required by code_block_xchange framework */
}
*DATAIN_NAME, *DATAGET_NAME;

/* ===== OUTPUT STRUCT ===== */
static struct OUTPUT_STRUCT_NAME
{
#if defined(RT_EVOLVE_ENERGY)
    MyFloat Dt_Rad_E_gamma[N_RT_FREQ_BINS];
#endif
#if defined(RT_EVOLVE_FLUX)
    Vec3<MyFloat> Dt_Rad_Flux[N_RT_FREQ_BINS];
#endif
#if defined(RT_INFRARED)
    MyFloat Dt_Rad_E_gamma_T_weighted_IR;
#endif
}
*DATARESULT_NAME, *DATAOUT_NAME;


/* ===== PACK INPUT ===== */
void particle2in_transport(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration)
{
    for(int k=0; k<3; k++) {in->Pos[k] = P[i].Pos[k];}
    in->Vel = CellP[i].VelPred;
    in->Mass = P[i].Mass;
    in->Density = CellP[i].Density;
    in->KernelRadius = P[i].KernelRadius;
    in->ConditionNumber = CellP[i].ConditionNumber;
    in->SoundSpeed = CellP[i].effective_soundspeed();
    for(int kf=0; kf<N_RT_FREQ_BINS; kf++) {
        in->Rad_E_gamma[kf] = CellP[i].Rad_E_gamma_Pred[kf];
        in->Rad_Kappa[kf] = CellP[i].Rad_Kappa[kf];
        in->RT_DiffusionCoeff[kf] = rt_diffusion_coefficient(i, kf, &CellP[i]);
#if defined(RT_EVOLVE_FLUX) || defined(HYDRO_SPH)
        in->ET[kf] = CellP[i].ET[kf];
#endif
#ifdef RT_EVOLVE_FLUX
        in->Rad_Flux[kf] = CellP[i].Rad_Flux_Pred[kf];
#endif
#ifdef RT_COMPGRAD_EDDINGTON_TENSOR
        in->Gradients.Rad_E_gamma_ET[kf] = CellP[i].Gradients.Rad_E_gamma_ET[kf];
#endif
    }
#ifdef RT_INFRARED
    in->Radiation_Temperature = CellP[i].Radiation_Temperature;
#endif
#ifdef MAGNETIC
    in->BPred = CellP[i].BPred;
#endif
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    in->ParticleVel = P[i].Vel;
#endif
    in->NV_T = CellP[i].NV_T;
}


/* ===== UNPACK OUTPUT ===== */
void out2particle_transport(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration)
{
    int k;
#if defined(RT_EVOLVE_ENERGY)
    for(k=0; k<N_RT_FREQ_BINS; k++) CellP[i].Dt_Rad_E_gamma[k] += out->Dt_Rad_E_gamma[k];
#endif
#if defined(RT_EVOLVE_FLUX)
    for(k=0; k<N_RT_FREQ_BINS; k++) CellP[i].Dt_Rad_Flux[k] += out->Dt_Rad_Flux[k];
#endif
#if defined(RT_INFRARED)
    CellP[i].Dt_Rad_E_gamma_T_weighted_IR += out->Dt_Rad_E_gamma_T_weighted_IR;
#endif
}


/* ===== CORE FUNCTION: transport flux evaluation ===== */
/* This is included from a separate file to keep this manageable */
#include "transport_flux_evaluate.h"


/* ===== INITIAL OPERATIONS (zero rate arrays) ===== */
void transport_flux_initial_operations(void)
{
    int k;
    for(int i : ActiveParticleList) {
        if(P[i].Type != 0 || P[i].Mass <= 0) continue;
#if defined(RT_SOLVER_EXPLICIT) && defined(RT_EVOLVE_ENERGY)
        for(k=0; k<N_RT_FREQ_BINS; k++) CellP[i].Dt_Rad_E_gamma[k] = 0;
#endif
#if defined(RT_SOLVER_EXPLICIT) && defined(RT_EVOLVE_FLUX)
        for(k=0; k<N_RT_FREQ_BINS; k++) CellP[i].Dt_Rad_Flux[k] = {};
#endif
#if defined(RT_SOLVER_EXPLICIT) && defined(RT_INFRARED)
        CellP[i].Dt_Rad_E_gamma_T_weighted_IR = 0;
#endif
    }
}


/* ===== PUBLIC ENTRY POINT ===== */
void transport_subcycle_exchange_fluxes(void)
{
    CPU_Step[CPU_MISC] += measure_time();
    transport_flux_initial_operations();
    #include "../system/code_block_xchange_perform_ops_malloc.h"
    #include "../system/code_block_xchange_perform_ops.h"
    #include "../system/code_block_xchange_perform_ops_demalloc.h"

    /* add back the radiation pressure work terms saved during the hydro pass.
       The flux exchange zeroed Dt_Rad_E_gamma and recomputed only transport fluxes;
       the work terms must be re-added so each sub-step's kick includes them. */
#if defined(RT_RAD_PRESSURE_FORCES) && defined(RT_EVOLVE_ENERGY)
    for(int i : ActiveParticleList) {
        if(P[i].Type != 0 || P[i].Mass <= 0) continue;
        for(int kf=0; kf<N_RT_FREQ_BINS; kf++)
            CellP[i].Dt_Rad_E_gamma[kf] += CellP[i].Dt_Rad_E_gamma_Work[kf];
    }
#endif

    CPU_Step[CPU_RTNONFLUXOPS] += measure_time();
}
#include "../system/code_block_xchange_finalize.h"


/* ======================================================================================== */
/* ===== KICK ============================================================================= */
/* ======================================================================================== */

void transport_subcycle_kick(void)
{
    double sub_frac = All.Transport_Subcycle_dt_fraction;
    for(int idx : ActiveParticleList) {
        if(P[idx].Type != 0 || P[idx].Mass <= 0) continue;
        int i = idx;
        double sub_dt = get_particle_timestep_in_physical(i) * sub_frac;
        if(sub_dt <= 0) continue;

#ifdef RADTRANSFER
        /* rt_update_driftkick adds IR gas heating to DtInternalEnergy as a rate.
           Since we call it N times, the rate would accumulate N-fold. Save/restore
           DtInternalEnergy each sub-step to prevent accumulation, but keep the final
           sub-step's contribution so the cooling function sees the correct IR heating rate. */
        rt_update_driftkick(i, sub_dt, 0, &P[i], &CellP[i]); /* kick: advance conserved variables */
        rt_eddington_update_calculation(i, &CellP[i]);
        /* update opacities — Rad_Kappa (especially IR band) depends on Dust_Temperature and
           Radiation_Temperature which were just modified by rt_update_driftkick. Stale opacities
           cause the transport flux computation in subsequent sub-steps to use the wrong diffusion rate. */
        for(int kf = 0; kf < N_RT_FREQ_BINS; kf++) {CellP[i].Rad_Kappa[kf] = rt_kappa(i, kf, &P[i], &CellP[i]);}
        /* sync Pred = conserved for the next sub-step's flux exchange */
        for(int kf = 0; kf < N_RT_FREQ_BINS; kf++) {
            CellP[i].Rad_E_gamma_Pred[kf] = CellP[i].Rad_E_gamma[kf];
#ifdef RT_EVOLVE_FLUX
            CellP[i].Rad_Flux_Pred[kf] = CellP[i].Rad_Flux[kf];
#endif
#ifdef RT_EVOLVE_INTENSITIES
            for(int ka = 0; ka < N_RT_INTENSITY_BINS; ka++)
                CellP[i].Rad_Intensity_Pred[kf][ka] = CellP[i].Rad_Intensity[kf][ka];
#endif
        }
#endif

#ifdef COSMIC_RAY_FLUID
        CosmicRay_Update_DriftKick(i, sub_dt, 0, &P[i], &CellP[i]);
        for(int kb = 0; kb < N_CR_PARTICLE_BINS; kb++) {
            CellP[i].CosmicRayEnergyPred[kb] = CellP[i].CosmicRayEnergy[kb];
            CellP[i].CosmicRayFluxPred[kb] = CellP[i].CosmicRayFlux[kb];
        }
#endif
    }
}


#endif // TRANSPORT_SUBCYCLE
