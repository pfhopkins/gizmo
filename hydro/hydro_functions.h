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
 * NOTE: Conservation is maintained because the symmetric neighbor list visits
 * each pair from both sides — each particle processes all fluxes as the i-side
 * (the legacy j_is_active_for_fluxes / symmetric-dispatch path has been removed).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef HYDRO_FUNCTIONS_H
#define HYDRO_FUNCTIONS_H

#ifdef COSMIC_RAY_FLUID
#include "../eos/cosmic_ray_fluid/cosmic_ray_functions.h"
#endif

#include "hydro_pair_types.h"
#include "compute_finitevol_faces_functions.h"
#include "conduction_functions.h"
#include "viscosity_functions.h"
#include "nonideal_mhd_functions.h"
#include "../solids/elastic_stress_tensor_force_functions.h"
#include "../turb/turbulent_diffusion_functions.h"
#include "../turb/chimes_turbulent_ion_diffusion_functions.h"

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
 * This is the inner body of hydro_force_evaluate() from hydro_evaluate.h.
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
    /* BPred_j declared unconditionally so nonideal_mhd can take it as a plain
       arg; zero-init when MAGNETIC is off (function body guards with MHD_NON_IDEAL
       which implies MAGNETIC, so the zero is never actually consumed). */
    Vec3<double> BPred_j = {};
#ifdef MAGNETIC
    BPred_j = CellP[j].Bfield();
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
    double fac_mu = 1.0 / All.cf_atime;
    double fac_vsic_fix = All.cf_hubble_a; /* needed by SPH viscosity limiter */
#ifdef MAGNETIC
    double fac_magnetic_pressure = 1.0 / All.cf_atime; /* B*B*fac = pressure units */
#if defined(HYDRO_SPH)
    Vec3<double> magfluxv = {}; double resistivity_heatflux = 0;
    kernel.mf_i = local.Mass * fac_magnetic_pressure / (local.Density * local.Density);
    kernel.mf_j = local.Mass * fac_magnetic_pressure;
    double mm_i[3][3], mm_j[3][3];
    for(k = 0; k < 3; k++) {
        for(int k2 = 0; k2 < 3; k2++) mm_i[k][k2] = local.BPred[k] * local.BPred[k2];
    }
    for(k = 0; k < 3; k++) mm_i[k][k] -= 0.5 * kernel.b2_i;
#endif
#endif
#ifdef HYDRO_MESHLESS_FINITE_MASS
    double epsilon_entropic_eos_big = 0.5;
    double epsilon_entropic_eos_small = 1.e-3;
#if defined(FORCE_ENTROPIC_EOS_BELOW)
    epsilon_entropic_eos_small = FORCE_ENTROPIC_EOS_BELOW;
#elif !defined(SELFGRAVITY_OFF)
    epsilon_entropic_eos_small = 1.e-2; epsilon_entropic_eos_big = 0.6;
#endif
#endif

#if defined(HYDRO_SPH)
#ifdef HYDRO_PRESSURE_SPH
    kernel.p_over_rho2_i = local.Pressure / (local.EgyWtRho*local.EgyWtRho);
#else
    kernel.p_over_rho2_i = local.Pressure / (local.Density*local.Density);
#endif
#endif
#if defined(RT_SOLVER_EXPLICIT) && (N_RT_FREQ_BINS > 0)
    double tau_c_i[N_RT_FREQ_BINS]; for(k=0;k<N_RT_FREQ_BINS;k++) {tau_c_i[k] = Particle_Size_i * local.Rad_Kappa[k]*local.Density*All.cf_a3inv;}
#endif

    /* MHD signal velocity: full fast magnetosonic wave speed computation */
#ifdef MAGNETIC
    kernel.b2_j = BPred_j.norm_sq();
    kernel.alfven2_j = kernel.b2_j * fac_magnetic_pressure / CellP[j].Density;
    kernel.alfven2_j = DMIN(kernel.alfven2_j, 1000. * kernel.sound_j*kernel.sound_j);
    double vcsa2_j = kernel.sound_j*kernel.sound_j + kernel.alfven2_j;
    double vcsa2_i = kernel.sound_i*kernel.sound_i + kernel.alfven2_i;
    double Bpro2_j = dot(BPred_j, kernel.dp) / kernel.r; Bpro2_j *= Bpro2_j;
    double magneticspeed_j = sqrt(0.5 * (vcsa2_j + sqrt(DMAX((vcsa2_j*vcsa2_j -
            4 * kernel.sound_j*kernel.sound_j * Bpro2_j*fac_magnetic_pressure/CellP[j].Density), 0))));
    double Bpro2_i = dot(local.BPred, kernel.dp) / kernel.r; Bpro2_i *= Bpro2_i;
    double magneticspeed_i = sqrt(0.5 * (vcsa2_i + sqrt(DMAX((vcsa2_i*vcsa2_i -
            4 * kernel.sound_i*kernel.sound_i * Bpro2_i*fac_magnetic_pressure/local.Density), 0))));
    kernel.vsig = magneticspeed_i + magneticspeed_j;
    Bpro2_i /= kernel.b2_i; Bpro2_j /= kernel.b2_j;
#endif

    /* relative velocity along separation vector + hubble correction */
    kernel.vdotr2 = dot(kernel.dp, kernel.dv);
    if(All.ComovingIntegrationOn) {kernel.vdotr2 += All.cf_hubble_a2 * r2;}
    if(kernel.vdotr2 < 0)
    {
#if defined(HYDRO_SPH) || defined(HYDRO_MESHLESS_FINITE_VOLUME)
        kernel.vsig -= 3 * fac_mu * kernel.vdotr2 * rinv;
#else
        kernel.vsig -= fac_mu * kernel.vdotr2 * rinv;
#endif
    }

#ifdef ENERGY_ENTROPY_SWITCH_IS_ACTIVE
    double KE = kernel.dv.norm_sq();
    if(KE > out.MaxKineticEnergyNgb) {out.MaxKineticEnergyNgb = KE;}
#endif
    /* mdot_estimated is set inside the MFM/MFV Riemann core when TURB_DIFF_METALS
       is enabled; declared unconditionally so turb-diffusion functions can take
       it as a plain argument (value is 0 when not set). */
    double mdot_estimated = 0;
    /* tensile_correction_factor: only non-trivial under Tillotson/elastic/ANEOS;
       declared unconditionally so the elastic function can take it as a plain arg. */
    double tensile_correction_factor = 0;
#if defined(EOS_TILLOTSON) || defined(EOS_ELASTIC) || defined(EOS_ANEOS)
    tensile_correction_factor = get_negative_pressure_tensilecorrfac(kernel.r, kernel.h_i, kernel.h_j);
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

    /* ---- Core hydro solver (face reconstruction + Riemann) ---- */
#ifdef HYDRO_SPH
#include "hydro_core_sph.h"
#else
#include "hydro_core_meshless.h"
#endif

#ifdef FREEZE_HYDRO
    memset(&Fluxes, 0, sizeof(struct Conserved_var_Riemann));
#endif

    /* SPH face setup (needs to come AFTER core solver for SPH mode) */
#ifdef HYDRO_SPH
    face_vel_i = dot(local.Vel, kernel.dp) / (kernel.r * All.cf_atime);
    face_vel_j = dot(VelPred_j, kernel.dp) / (kernel.r * All.cf_atime);
    Face_Area_Norm = local.Mass * P[j].Mass * fabs(kernel.dwk_i+kernel.dwk_j) / (local.Density * CellP[j].Density) * All.cf_atime*All.cf_atime;
    Face_Area_Vec = kernel.dp * (Face_Area_Norm / kernel.r);
#endif

    /* HLL diffusion setup: bhat, v_hll, B_dot_grad_weights.
       MUST come AFTER core solver because face_vel_i/j are computed there.
       bhat/bhat_mag are declared unconditionally so per-pair physics functions
       can take them as plain arguments without #ifdef MAGNETIC in the signature. */
    Vec3<double> bhat = {};
    double bhat_mag = 0;
#ifdef MAGNETIC
    bhat = 0.5 * (local.BPred + BPred_j) * All.cf_a2inv;
    bhat_mag = bhat.norm_sq();
    if(bhat_mag>0) {bhat_mag=sqrt(bhat_mag); bhat /= bhat_mag;}
    v_hll = 0.5*fabs(face_vel_i-face_vel_j) + DMAX(magneticspeed_i,magneticspeed_j);
#define B_dot_grad_weights(grad_i,grad_j) {if(bhat_mag<=0) {b_hll=1;} else {double q_tmp_sum=0,b_tmp_sum=0; for(k=0;k<3;k++) {\
                                           double q_tmp=0.5*(grad_i[k]+grad_j[k]); q_tmp_sum+=q_tmp*q_tmp; b_tmp_sum+=bhat[k]*q_tmp;}\
                                           if((b_tmp_sum!=0)&&(q_tmp_sum>0)) {b_hll=fabs(b_tmp_sum)/sqrt(q_tmp_sum); b_hll*=b_hll;} else {b_hll=0;}}}
#ifndef HLL_DIFFUSION_COMPROMISE_FACTOR
#define HLL_DIFFUSION_COMPROMISE_FACTOR 1.1
#endif
#else
    v_hll = 0.5*fabs(face_vel_i-face_vel_j) + DMAX(kernel.sound_i,kernel.sound_j);
#ifndef B_dot_grad_weights
#define B_dot_grad_weights(grad_i,grad_j) {b_hll=1;}
#endif
#ifndef HLL_DIFFUSION_COMPROMISE_FACTOR
#define HLL_DIFFUSION_COMPROMISE_FACTOR 1.5
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

    /* ---- Physics sub-modules ---- */
    elastic_stress_tensor_force_compute_pair(local, P[j], CellP[j], VelPred_j, kernel, rinv,
                                             Face_Area_Vec, Face_Area_Norm,
                                             tensile_correction_factor, dt_hydrostep, Fluxes);
    nonideal_mhd_compute_pair(local, P[j], CellP[j], BPred_j, kernel, rinv,
                              Face_Area_Vec, Face_Area_Norm, v_hll, bhat, bhat_mag,
                              dt_hydrostep, Fluxes);
    /* Per-pair physics sub-modules. Functions guard their bodies with the
       relevant #ifdef so callers invoke unconditionally (no-op when disabled). */
    double face_density_for_diffusion = 0;
#if defined(SAVE_FACE_DENSITY) && !defined(HYDRO_SPH)
    face_density_for_diffusion = Riemann_out.Face_Density;
#endif
    conduction_compute_pair(local, P[j], CellP[j], kernel, rinv, Face_Area_Vec, Face_Area_Norm,
                            v_hll, bhat, bhat_mag, dt_hydrostep, Fluxes);
    viscosity_compute_pair(local, P[j], CellP[j], VelPred_j, kernel, rinv,
                           Face_Area_Vec, Face_Area_Norm, v_hll, bhat, bhat_mag,
                           dt_hydrostep, Fluxes);
    turb_diff_metals_compute_pair(local, P[j], CellP[j], kernel, rinv, Face_Area_Vec, Face_Area_Norm,
                                  face_density_for_diffusion, v_hll, dt_hydrostep, mdot_estimated, out);
    chimes_turb_diff_ions_compute_pair(local, P[j], CellP[j], kernel, rinv, Face_Area_Vec, Face_Area_Norm,
                                       face_density_for_diffusion, v_hll, dt_hydrostep, mdot_estimated, out);
#ifdef COSMIC_RAY_FLUID
#include "../eos/cosmic_ray_fluid/cosmic_ray_diffusion.h"
#endif
#if defined(RT_SOLVER_EXPLICIT) && (N_RT_FREQ_BINS > 0)
#if defined(RT_EVOLVE_INTENSITIES)
#include "../radiation/rt_direct_ray_transport.h"
#else
#include "../radiation/rt_diffusion_explicit.h"
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
            out.dMass += dmass_holder;
        } else if(local.dt_hydrostep_i == dt_hydrostep_j_loc) {
            out.dMass += 0.5 * dmass_holder;
        }
        out.DtMass += Fluxes.rho;
        Vec3<double> gravwork = kernel.dp * Fluxes.rho;
        out.GravWorkTerm += gravwork;
#ifdef METALS
        if(Fluxes.rho > 0) { for(k=0;k<NUM_METAL_SPECIES;k++) {out.Dyield[k] += (P[j].Metallicity[k] - local.Metallicity[k]) * dmass_holder;} }
#endif
    }
#endif
    out.Acc += Fluxes.v;
    out.DtInternalEnergy += Fluxes.p;
#ifdef MAGNETIC
#ifndef HYDRO_SPH
    out.Face_Area += Face_Area_Vec;
#endif
#ifndef FREEZE_HYDRO
    out.DtB += Fluxes.B;
    out.divB += Fluxes.B_normal_corrected;
#if defined(DIVBCLEANING_DEDNER) && defined(HYDRO_MESHLESS_FINITE_VOLUME)
    out.DtPhi += Fluxes.phi;
#endif
#ifdef HYDRO_SPH
    out.DtInternalEnergy += dot(magfluxv, local.Vel) / All.cf_atime;
    out.DtInternalEnergy += resistivity_heatflux;
#else
    {
        double wt_face_sum = Face_Area_Norm * (-face_area_dot_vel + face_vel_i);
        out.DtInternalEnergy += 0.5 * kernel.b2_i * All.cf_a2inv * All.cf_a2inv * wt_face_sum;
    }
#ifdef DIVBCLEANING_DEDNER
    for(k=0; k<3; k++) {
        out.DtB_PhiCorr[k] += Riemann_out.phi_normal_db * Face_Area_Vec[k];
        out.DtB[k] += Riemann_out.phi_normal_mean * Face_Area_Vec[k];
        out.DtInternalEnergy += Riemann_out.phi_normal_mean * Face_Area_Vec[k] * local.BPred[k] * All.cf_a2inv;
    }
#endif
#ifdef MHD_NON_IDEAL
    out.DtInternalEnergy += dot(local.BPred, bflux_from_nonideal_effects) * All.cf_a2inv;
#endif
#endif
#endif
#endif

    /* ---- J-particle writes (Kokkos atomics for thread safety) ---- */

#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    /* MFV mass conservation: machine-accurate two-sided mass exchange, using
       timestep-conditional logic to avoid double-counting when both i and j
       evaluate the same pair. */
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
            /* equal timesteps: each side applies half so the pair sums to full */
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
