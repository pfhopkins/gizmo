/* hydro_functions.h — GPU-callable hydro force kernel functions (AthenaK pattern).
 *
 * Contains: struct hydro_evaluate_data_in_, struct hydro_evaluate_data_out_,
 *   hydro_accumulate_neighbor().
 *
 * The per-neighbor-pair flux computation extracted from hydro_evaluate.h.
 * Includes the Riemann solver, face reconstruction, and all physics sub-modules
 * via their existing inline include files.
 *
 * For GPU builds: the GPU TU includes this with KOKKOS_INLINE_FUNCTION as
 * __device__ __host__ inline; all sub-includes (reimann.h, hydro_core_meshless.h,
 * etc.) are compiled into device code.
 *
 * NOTE: This header does NOT handle j-particle writes (j_is_active_for_fluxes path).
 * Conservation is maintained because the symmetric neighbor list visits each pair
 * from both sides — each particle processes all fluxes as the i-side.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef HYDRO_FUNCTIONS_H
#define HYDRO_FUNCTIONS_H

/* Requires: allvars.h, proto.h, kernel.h, reimann.h included before this header.
   Also requires: Conserved_var_Riemann, kernel_hydra, INPUT_STRUCT_NAME, OUTPUT_STRUCT_NAME
   defined (from hydro_toplevel.cc via code_block_xchange_initialize.h).
   This header is #include'd from within hydro_toplevel.cc or the GPU TU after those
   struct definitions are in scope. */

/* Portable atomic operations: Kokkos atomics on GPU, direct write on CPU.
   On CPU the caller ensures thread safety via OpenMP atomics at a higher level. */
#ifdef OPENMP_GPU_OFFLOAD
#define HYDRO_ATOMIC_ADD(ptr, val) Kokkos::atomic_add(ptr, val)
#define HYDRO_ATOMIC_STORE(ptr, val) Kokkos::atomic_store(ptr, val)
#else
#define HYDRO_ATOMIC_ADD(ptr, val) (*(ptr) += (val))
#define HYDRO_ATOMIC_STORE(ptr, val) (*(ptr) = (val))
#endif


/* Per-neighbor-pair hydro flux accumulation for particle i.
 * This is the inner body of hydro_force_evaluate() from hydro_evaluate.h,
 * without the j-particle write-back (j_is_active_for_fluxes path).
 *
 * Arguments mirror those available in the original loop:
 *   local   — input data for particle i
 *   out     — output accumulator for particle i
 *   kernel  — kernel workspace (dp, r, wk, etc.)
 *   Fluxes  — Riemann flux workspace (zeroed per pair)
 *   j       — neighbor particle index
 *   dt_hydrostep_i — particle i's timestep
 *   P, CellP — particle arrays
 *
 * The function computes: kernel evaluation, face reconstruction, Riemann solve,
 * and flux accumulation into 'out'. All sub-includes (hydro_core_meshless.h,
 * reimann.h, conduction.h, viscosity.h, etc.) are included inline. */
KOKKOS_INLINE_FUNCTION
void hydro_accumulate_neighbor(
    struct INPUT_STRUCT_NAME *local_ptr_arg,
    struct OUTPUT_STRUCT_NAME *out_ptr_arg,
    struct kernel_hydra *kernel_ptr_arg,
    struct Conserved_var_Riemann *Fluxes_ptr_arg,
    int j, double dt_hydrostep_i,
    struct particle_data *P, struct gas_cell_data *CellP,
    int *TimeBinActive_arr, int *NeedToWakeup_flag)
{
    /* Sub-includes (hydro_core_meshless.h, reimann.h, conduction.h, etc.) expect
       local, out, kernel, Fluxes as value types with . access. Create references
       from the pointer arguments, then #define the names so all sub-includes work. */
    auto &local = *local_ptr_arg;
    auto &out = *out_ptr_arg;
    auto &kernel = *kernel_ptr_arg;
    auto &Fluxes = *Fluxes_ptr_arg;

    int k;
    if(P[j].Mass <= 0) return;
    if(CellP[j].Density <= 0) return;
#ifdef GALSF_SUBGRID_WINDS
    if(CellP[j].DelayTime > 0) return;
#endif

    double dt_hydrostep_j = get_particle_timestep_in_physical(j, P);
    double dt_hydrostep = DMAX(dt_hydrostep_i, dt_hydrostep_j);
    double FluxCorrectionFactor_to_i = 1;
    int j_is_active_for_fluxes = 0; /* GPU: always 0 — each side accumulates its own fluxes */

    kernel.dp = local.Pos - P[j].Pos;
    nearest_xyz(kernel.dp);
    double r2 = kernel.dp.norm_sq();
    kernel.h_j = P[j].KernelRadius;

    if((r2 >= kernel.h_i * kernel.h_i) && (r2 >= kernel.h_j * kernel.h_j)) return;
    if(r2 <= 0) return;

    kernel.r = sqrt(r2);
    double rinv = 1.0 / kernel.r;
    double rinv_soft = 1.0 / sqrt(r2 + 0.0001 * kernel.h_i * kernel.h_i);

    Vec3<MyDouble> VelPred_j = CellP[j].VelPred;
    NGB_SHEARBOX_BOUNDARY_VELCORR_(local.Pos, P[j].Pos, VelPred_j, -1);
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    Vec3<MyDouble> ParticleVel_j = CellP[j].VelPred;
    NGB_SHEARBOX_BOUNDARY_VELCORR_(local.Pos, P[j].Pos, ParticleVel_j, -1);
#endif
    kernel.dv = local.Vel - VelPred_j;
    kernel.rho_ij_inv = 2.0 / (local.Density + CellP[j].Density);
    double Particle_Size_j = P[j].Get_Particle_Size() * All.cf_atime;
    double Particle_Size_i = local.Mass / local.Density;
    Particle_Size_i = pow(Particle_Size_i, 1.0/NUMDIMS) * All.cf_atime;

    kernel.sound_j = CellP[j].effective_soundspeed();
    kernel.vsig = kernel.sound_i + kernel.sound_j;

#ifdef COSMIC_RAY_FLUID
    double CosmicRayPressure_j[N_CR_PARTICLE_BINS];
    for(k=0; k<N_CR_PARTICLE_BINS; k++) {CosmicRayPressure_j[k] = Get_Gas_CosmicRayPressure(j, k, CellP);}
#endif
#ifdef MAGNETIC
    Vec3<double> BPred_j = CellP[j].Bfield();
    NGB_SHEARBOX_BOUNDARY_BCORR_(local.Pos, P[j].Pos, BPred_j, -1);
#ifdef DIVBCLEANING_DEDNER
    double PhiPred_j = CellP[j].PhiPred / P[j].Mass;
#endif
#endif

    /* kernel evaluation at h_i and h_j */
    double hinv_i, hinv3_i, hinv4_i, hinv_j, hinv3_j, hinv4_j, u;
    kernel_hinv(kernel.h_i, &hinv_i, &hinv3_i, &hinv4_i);
    if(kernel.r < kernel.h_i) {
        u = kernel.r * hinv_i;
        kernel_main(u, hinv3_i, hinv4_i, &kernel.wk_i, &kernel.dwk_i, 0);
    } else { kernel.wk_i = kernel.dwk_i = 0; }
    kernel_hinv(kernel.h_j, &hinv_j, &hinv3_j, &hinv4_j);
    if(kernel.r < kernel.h_j) {
        u = kernel.r * hinv_j;
        kernel_main(u, hinv3_j, hinv4_j, &kernel.wk_j, &kernel.dwk_j, 0);
    } else { kernel.wk_j = kernel.dwk_j = 0; }
    kernel.dwk_ij = 0.5 * (kernel.dwk_i + kernel.dwk_j);

    /* Variables expected by sub-includes that are normally in the enclosing scope */
    double cnumcrit2 = ((double)CONDITION_NUMBER_DANGER)*((double)CONDITION_NUMBER_DANGER) - local.ConditionNumber * local.ConditionNumber;
#ifdef MAGNETIC
    double fac_magnetic_pressure = 1.0 / (All.cf_atime * All.cf_atime);
#endif
#ifdef HYDRO_MESHLESS_FINITE_MASS
    double epsilon_entropic_eos_big = 0.5;
    double epsilon_entropic_eos_small = 1.e-3;
#if defined(FORCE_ENTROPIC_EOS_BELOW)
    epsilon_entropic_eos_small = FORCE_ENTROPIC_EOS_BELOW;
#endif
#endif

    double V_i = local.Mass / local.Density;
    double V_j = P[j].Mass / CellP[j].Density;
    double Face_Area_Norm = 0;
    Vec3<double> Face_Area_Vec;
    double face_vel_i = 0, face_vel_j = 0;
    double v_hll = 0, k_hll = 0, b_hll = 1;
    int kernel_mode = 0;

    memset(&Fluxes, 0, sizeof(struct Conserved_var_Riemann));

#ifndef HYDRO_SPH
    struct Input_vec_Riemann Riemann_vec;
    struct Riemann_outputs Riemann_out;
    memset(&Riemann_vec, 0, sizeof(struct Input_vec_Riemann));
    memset(&Riemann_out, 0, sizeof(struct Riemann_outputs));
    double face_area_dot_vel = 0;
#endif

    /* MHD signal velocity setup (bhat, bhat_mag used by B_dot_grad_weights in core solver) */
#ifdef MAGNETIC
    Vec3<double> bhat = 0.5 * (local.BPred + BPred_j) * All.cf_a2inv;
    double bhat_mag = bhat.norm_sq();
    if(bhat_mag>0) {bhat_mag=sqrt(bhat_mag); bhat /= bhat_mag;}
#define B_dot_grad_weights(grad_i,grad_j) {if(bhat_mag<=0) {b_hll=1;} else {double q_tmp_sum=0,b_tmp_sum=0; for(k=0;k<3;k++) {\
                                           double q_tmp=0.5*(grad_i[k]+grad_j[k]); q_tmp_sum+=q_tmp*q_tmp; b_tmp_sum+=bhat[k]*q_tmp;}\
                                           if(q_tmp_sum>0) {b_hll=b_tmp_sum*b_tmp_sum/q_tmp_sum;} else {b_hll=1;}}}
#endif

    /* HLL correction macros (normally in hydro_evaluate.h, needed by conduction/viscosity sub-includes) */
#ifndef MAGNETIC
    v_hll = 0.5*fabs(face_vel_i-face_vel_j) + DMAX(kernel.sound_i,kernel.sound_j);
#ifndef B_dot_grad_weights
#define B_dot_grad_weights(grad_i,grad_j) {b_hll=1;}
#endif
#ifndef HLL_DIFFUSION_COMPROMISE_FACTOR
#define HLL_DIFFUSION_COMPROMISE_FACTOR 1.5
#endif
#else
#ifndef HLL_DIFFUSION_COMPROMISE_FACTOR
#define HLL_DIFFUSION_COMPROMISE_FACTOR 1.1
#endif
#endif
#ifndef HLL_correction
#define HLL_correction(ui,uj,wt,kappa) (k_hll = v_hll * (wt) * kernel.r * All.cf_atime / fabs(kappa),\
                                        k_hll = (0.2 + k_hll) / (0.2 + k_hll + k_hll*k_hll),\
                                        -1.0*k_hll*Face_Area_Norm*v_hll*((ui)-(uj)))
#endif
#ifndef HLL_DIFFUSION_OVERSHOOT_FACTOR
#if !defined(MAGNETIC) || defined(GALSF) || defined(COOLING) || defined(SINK_PARTICLES)
#define HLL_DIFFUSION_OVERSHOOT_FACTOR  0.005
#else
#define HLL_DIFFUSION_OVERSHOOT_FACTOR  1.0
#endif
#endif

#ifdef EOS_ELASTIC
    double tensile_correction_factor = 0;
#endif

    /* ---- Core hydro solver (face reconstruction + Riemann) ---- */
#ifdef HYDRO_SPH
#include "hydro_core_sph.h"
#else
#include "hydro_core_meshless.h"
#endif

    /* ---- Physics sub-modules ---- */
#ifdef EOS_ELASTIC
#include "../solids/elastic_stress_tensor_force.h"
#endif
#ifdef MHD_NON_IDEAL
#include "nonideal_mhd.h"
#endif
#ifdef CONDUCTION
#include "conduction.h"
#endif
#ifdef VISCOSITY
#include "viscosity.h"
#endif
#ifdef TURB_DIFFUSION
#include "../turb/turbulent_diffusion.h"
#endif
#ifdef CHIMES_TURB_DIFF_IONS
#include "../turb/chimes_turbulent_ion_diffusion.h"
#endif
#ifdef COSMIC_RAY_FLUID
#include "../eos/cosmic_ray_fluid/cosmic_ray_diffusion.h"
#endif
#ifdef RT_SOLVER_EXPLICIT
#if defined(RT_OTVET) || defined(RT_FLUXLIMITEDDIFFUSION) || defined(RT_M1)
#include "../radiation/rt_diffusion_explicit.h"
#endif
#if defined(RT_LOCALRAYGRID) || defined(RT_FIRE_FIX_SPECTRAL_SHAPE)
#include "../radiation/rt_direct_ray_transport.h"
#endif
#endif

    /* ---- Flux assignment to particle i ---- */
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    {
        double dmass_holder = Fluxes.rho * dt_hydrostep_i, dmass_limiter;
        if(dmass_holder > 0) {dmass_limiter = P[j].Mass;} else {dmass_limiter = local.Mass;}
        dmass_limiter *= 0.1;
        if(fabs(dmass_holder) > dmass_limiter) {dmass_holder *= dmass_limiter / fabs(dmass_holder);}
        /* Timestep-conditional mass flux for machine-accurate conservation */
        double dt_hydrostep_j_loc = get_particle_timestep_in_physical(j, P);
        if(local.dt_hydrostep_i < dt_hydrostep_j_loc) {
            out.dMass += FluxCorrectionFactor_to_i * dmass_holder;
        } else if(local.dt_hydrostep_i == dt_hydrostep_j_loc) {
            out.dMass += FluxCorrectionFactor_to_i * 0.5 * dmass_holder;
        }
        out.DtMass += FluxCorrectionFactor_to_i * Fluxes.rho;
        Vec3<double> gravwork = kernel.dp * Fluxes.rho;
        out.GravWorkTerm += gravwork * FluxCorrectionFactor_to_i;
#ifdef METALS
        if(Fluxes.rho > 0) { for(k=0;k<NUM_METAL_SPECIES;k++) {out.Dyield[k] += FluxCorrectionFactor_to_i * (P[j].Metallicity[k] - local.Metallicity[k]) * dmass_holder;} }
#endif
    }
#endif
    out.Acc += FluxCorrectionFactor_to_i * Fluxes.v;
    out.DtInternalEnergy += FluxCorrectionFactor_to_i * Fluxes.p;
#ifdef MAGNETIC
#ifndef HYDRO_SPH
    out.Face_Area += Face_Area_Vec;
#endif
#ifndef FREEZE_HYDRO
    out.DtB += FluxCorrectionFactor_to_i * Fluxes.B;
    out.divB += Fluxes.B_normal_corrected;
#if defined(DIVBCLEANING_DEDNER) && defined(HYDRO_MESHLESS_FINITE_VOLUME)
    out.DtPhi += FluxCorrectionFactor_to_i * Fluxes.phi;
#endif
#ifdef HYDRO_SPH
    out.DtInternalEnergy += FluxCorrectionFactor_to_i * dot(magfluxv, local.Vel) / All.cf_atime;
    out.DtInternalEnergy += FluxCorrectionFactor_to_i * resistivity_heatflux;
#else
    {
        double wt_face_sum = Face_Area_Norm * (-face_area_dot_vel + face_vel_i);
        out.DtInternalEnergy += FluxCorrectionFactor_to_i * 0.5 * kernel.b2_i * All.cf_a2inv * All.cf_a2inv * wt_face_sum;
    }
#ifdef DIVBCLEANING_DEDNER
    for(k=0; k<3; k++) {
        out.DtB_PhiCorr[k] += FluxCorrectionFactor_to_i * Riemann_out.phi_normal_db * Face_Area_Vec[k];
        out.DtB[k] += FluxCorrectionFactor_to_i * Riemann_out.phi_normal_mean * Face_Area_Vec[k];
        out.DtInternalEnergy += FluxCorrectionFactor_to_i * Riemann_out.phi_normal_mean * Face_Area_Vec[k] * local.BPred[k] * All.cf_a2inv;
    }
#endif
#ifdef MHD_NON_IDEAL
    out.DtInternalEnergy += FluxCorrectionFactor_to_i * dot(local.BPred, bflux_from_nonideal_effects) * All.cf_a2inv;
#endif
#endif
#endif
#endif

    /* ---- J-particle writes (Kokkos atomics for thread safety) ---- */

#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    /* MFV mass conservation: machine-accurate two-sided mass exchange.
       With j_is_active_for_fluxes=0 (dead code, always 0 in modern GIZMO),
       FluxCorrectionFactor_to_j=1, and we use the timestep-conditional logic. */
    {
        double dmass_holder = Fluxes.rho * dt_hydrostep_i, dmass_limiter;
        if(dmass_holder > 0) {dmass_limiter = P[j].Mass;} else {dmass_limiter = local.Mass;}
        dmass_limiter *= 0.1;
        if(fabs(dmass_holder) > dmass_limiter) {dmass_holder *= dmass_limiter / fabs(dmass_holder);}
        double dt_hydrostep_j_loc = get_particle_timestep_in_physical(j, P);
        if(local.dt_hydrostep_i < dt_hydrostep_j_loc) {
            /* i has shorter timestep: both sides get full flux */
            HYDRO_ATOMIC_ADD(&CellP[j].dMass, (MyDouble)(-dmass_holder));
        } else if(local.dt_hydrostep_i == dt_hydrostep_j_loc) {
            /* equal timesteps, j_is_active_for_fluxes=0: half flux */
            HYDRO_ATOMIC_ADD(&CellP[j].dMass, (MyDouble)(-0.5 * dmass_holder));
        }
        /* if dt_i > dt_j: j will handle this pair when it's active */
    }
#endif

    /* Signal velocity for timestepping */
    if(kernel.vsig > out.MaxSignalVel) {out.MaxSignalVel = kernel.vsig;}

    /* Wakeup check: if j is inactive and signal velocity exceeds threshold,
       flag it for wakeup. Uses Kokkos atomics for thread safety. */
    if(TimeBinActive_arr && !(TimeBinActive_arr[P[j].TimeBin]))
    {
        if(kernel.vsig > WAKEUP * CellP[j].MaxSignalVel) {
            short int wakeup_val = (short int)(local.TimeBin + 1);
            HYDRO_ATOMIC_STORE(&P[j].wakeup, wakeup_val);
            if(NeedToWakeup_flag) HYDRO_ATOMIC_STORE(NeedToWakeup_flag, 1);
        }
    }
}


#endif /* HYDRO_FUNCTIONS_H */
