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
#include "hydro_core_meshless_functions.h"
#include "hydro_core_sph_functions.h"
#include "conduction_functions.h"
#include "viscosity_functions.h"
#include "nonideal_mhd_functions.h"
#include "../solids/elastic_stress_tensor_force_functions.h"
#include "../solids/elastic_physics_functions.h"  /* get_negative_pressure_tensilecorrfac (KOKKOS_INLINE) for hydro_accumulate_neighbor device pass */
#include "../turb/turbulent_diffusion_functions.h"
#include "../turb/chimes_turbulent_ion_diffusion_functions.h"
#include "../eos/cosmic_ray_fluid/cosmic_ray_diffusion_functions.h"
#include "../radiation/rt_direct_ray_transport_functions.h"
#include "../radiation/rt_diffusion_explicit_functions.h"

/* Requires: allvars.h, proto.h, kernel.h, reimann.h included before this header.
   Also requires Conserved_var_Riemann, kernel_hydra, and caller-provided
   INPUT_STRUCT_NAME / OUTPUT_STRUCT_NAME aliases to be in scope. */

/* Portable atomic operations: Kokkos atomics on GPU, direct write on CPU.
   On CPU the caller ensures thread safety via OpenMP atomics at a higher level. */
#define HYDRO_ATOMIC_ADD(ptr, val) Kokkos::atomic_add(ptr, val)
#define HYDRO_ATOMIC_STORE(ptr, val) Kokkos::atomic_store(ptr, val)
#define HYDRO_ATOMIC_MAX(ptr, val) Kokkos::atomic_max(ptr, val)


/* Per-neighbor-pair hydro flux accumulation for particle i: kernel evaluation,
 * face reconstruction, Riemann solve, and flux accumulation into 'out'. */
KOKKOS_INLINE_FUNCTION
void hydro_accumulate_neighbor(
    struct INPUT_STRUCT_NAME &local,
    struct OUTPUT_STRUCT_NAME &out,
    struct kernel_hydra &kernel,
    struct Conserved_var_Riemann &Fluxes,
    int j, double dt_hydrostep_i,
    struct particle_data *P, struct gas_cell_data *CellP,
    int *TimeBinActive_arr, int *NeedToWakeup_flag,
    bool allow_j_writes)
{
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
    double PhiPred_j = 0;
#ifdef MAGNETIC
    BPred_j = CellP[j].Bfield();
    NGB_SHEARBOX_BOUNDARY_BCORR_(local.Pos, P[j].Pos, BPred_j, -1);
#ifdef DIVBCLEANING_DEDNER
    PhiPred_j = CellP[j].PhiPred / P[j].Mass;
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
    /* fac_magnetic_pressure lifted out of #ifdef MAGNETIC so hydro_core functions
       can take it as a plain arg; value is 0 when MAGNETIC is off (unused there). */
    double fac_magnetic_pressure = 0;
    /* magfluxv / resistivity_heatflux lifted out of MAGNETIC+HYDRO_SPH so the
       hydro_core_sph function can take them as output args. Used only under
       that combination; zero-initialised otherwise. */
    Vec3<double> magfluxv = {};
    double resistivity_heatflux = 0;
#ifdef MAGNETIC
    fac_magnetic_pressure = 1.0 / All.cf_atime;
#if defined(HYDRO_SPH)
    kernel.mf_i = local.Mass * fac_magnetic_pressure / (local.Density * local.Density);
    kernel.mf_j = local.Mass * fac_magnetic_pressure;
#endif
#endif
    /* Entropic-energy-equation thresholds: declared unconditionally so the
       hydro_core_meshless function can take them as plain args. Values only
       active under HYDRO_MESHLESS_FINITE_MASS; harmless defaults otherwise. */
    double epsilon_entropic_eos_big = 0.5;
    double epsilon_entropic_eos_small = 1.e-3;
#ifdef HYDRO_MESHLESS_FINITE_MASS
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
    /* magneticspeed_i/j declared unconditionally so hydro_core_sph can take
       them as plain args; zero under !MAGNETIC (unused on that path). */
    double magneticspeed_i = 0, magneticspeed_j = 0;
#ifdef MAGNETIC
    kernel.b2_j = BPred_j.norm_sq();
    kernel.alfven2_j = kernel.b2_j * fac_magnetic_pressure / CellP[j].Density;
    kernel.alfven2_j = DMIN(kernel.alfven2_j, 1000. * kernel.sound_j*kernel.sound_j);
    double vcsa2_j = kernel.sound_j*kernel.sound_j + kernel.alfven2_j;
    double vcsa2_i = kernel.sound_i*kernel.sound_i + kernel.alfven2_i;
    double Bpro2_j = dot(BPred_j, kernel.dp) / kernel.r; Bpro2_j *= Bpro2_j;
    magneticspeed_j = sqrt(0.5 * (vcsa2_j + sqrt(DMAX((vcsa2_j*vcsa2_j -
            4 * kernel.sound_j*kernel.sound_j * Bpro2_j*fac_magnetic_pressure/CellP[j].Density), 0))));
    double Bpro2_i = dot(local.BPred, kernel.dp) / kernel.r; Bpro2_i *= Bpro2_i;
    magneticspeed_i = sqrt(0.5 * (vcsa2_i + sqrt(DMAX((vcsa2_i*vcsa2_i -
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
    double v_hll = 0;
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
    hydro_core_sph_compute_pair(local, j, P, CellP, VelPred_j, BPred_j, PhiPred_j,
                                kernel, r2, V_j,
                                magneticspeed_i, magneticspeed_j,
                                fac_mu, fac_vsic_fix, tensile_correction_factor,
                                dt_hydrostep, magfluxv, resistivity_heatflux, Fluxes);
#else
    {
        const double *cr_pressure_j_ptr = nullptr;
#ifdef COSMIC_RAY_FLUID
        cr_pressure_j_ptr = CosmicRayPressure_j;
#endif
        Vec3<MyDouble> particle_vel_j_for_core = {};
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        particle_vel_j_for_core = ParticleVel_j;
#endif
        hydro_core_meshless_compute_pair(local, j, P, CellP, VelPred_j, particle_vel_j_for_core,
                                         BPred_j, PhiPred_j, kernel, rinv, r2, V_i, V_j,
                                         Particle_Size_i, Particle_Size_j, cnumcrit2,
                                         fac_magnetic_pressure, tensile_correction_factor,
                                         epsilon_entropic_eos_big, epsilon_entropic_eos_small,
                                         cr_pressure_j_ptr,
                                         Face_Area_Vec, Face_Area_Norm,
                                         face_vel_i, face_vel_j, face_area_dot_vel,
                                         mdot_estimated, Riemann_vec, Riemann_out, Fluxes, out);
    }
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
#else
    v_hll = 0.5*fabs(face_vel_i-face_vel_j) + DMAX(kernel.sound_i,kernel.sound_j);
#endif

    /* ---- Physics sub-modules ---- */
    elastic_stress_tensor_force_compute_pair(local, P[j], CellP[j], VelPred_j, kernel, rinv,
                                             Face_Area_Vec, Face_Area_Norm,
                                             tensile_correction_factor, dt_hydrostep, Fluxes);
    Vec3<double> bflux_from_nonideal_effects = {};
    /* nonideal_mhd_compute_pair handles classical non-ideal MHD only
       (Ohmic / Hall / ambipolar). Battery sources are now applied as
       cell-centered host source terms in hydro_toplevel.cc. */
    nonideal_mhd_compute_pair(local, P[j], CellP[j], BPred_j, kernel, rinv,
                              Face_Area_Vec, Face_Area_Norm, v_hll, bhat, bhat_mag,
                              dt_hydrostep, Fluxes, bflux_from_nonideal_effects);
    /* Per-pair physics sub-modules. Functions guard their bodies with the
       relevant #ifdef so callers invoke unconditionally (no-op when disabled). */
    double face_density_for_diffusion = 0;
#if defined(SAVE_FACE_DENSITY) && !defined(HYDRO_SPH)
    face_density_for_diffusion = Riemann_out.Face_Density;
#endif
#if defined(TWO_TEMPERATURE_PLASMA) && (TWO_TEMPERATURE_PLASMA & 4) && defined(CONDUCTION)
    /* 2-T plasma bit 2: capture the conduction-only contribution to Fluxes.p
       so the cooling step can route it to u_e (electron heat) instead of u_total.
       Pre/post difference avoids modifying conduction_compute_pair's signature. */
    const MyDouble Fluxes_p_pre_conduction_2T = Fluxes.p;
#endif
    conduction_compute_pair(local, P[j], CellP[j], kernel, rinv, Face_Area_Vec, Face_Area_Norm,
                            v_hll, bhat, bhat_mag, dt_hydrostep, Fluxes);
#if defined(TWO_TEMPERATURE_PLASMA) && (TWO_TEMPERATURE_PLASMA & 4) && defined(CONDUCTION)
    out.DtInternalEnergy_FromConduction += (Fluxes.p - Fluxes_p_pre_conduction_2T);
#endif
    viscosity_compute_pair(local, P[j], CellP[j], VelPred_j, kernel, rinv,
                           Face_Area_Vec, Face_Area_Norm, v_hll, bhat, bhat_mag,
                           dt_hydrostep, Fluxes);
    turb_diff_metals_compute_pair(local, P[j], CellP[j], kernel, rinv, Face_Area_Vec, Face_Area_Norm,
                                  face_density_for_diffusion, v_hll, dt_hydrostep, mdot_estimated, out);
    chimes_turb_diff_ions_compute_pair(local, P[j], CellP[j], kernel, rinv, Face_Area_Vec, Face_Area_Norm,
                                       face_density_for_diffusion, v_hll, dt_hydrostep, mdot_estimated, out);
    {
        /* Riemann_out / face_area_dot_vel exist only under !HYDRO_SPH; local
           shims default to zero so the function signature stays the same. */
        double riemann_S_M_val = 0, face_area_dot_vel_val = 0;
#ifndef HYDRO_SPH
        riemann_S_M_val = Riemann_out.S_M;
        face_area_dot_vel_val = face_area_dot_vel;
#endif
        const double *cr_pressure_j_ptr = nullptr;
#ifdef COSMIC_RAY_FLUID
        cr_pressure_j_ptr = CosmicRayPressure_j;
#endif
        cosmic_ray_diffusion_compute_pair(local, j, P, CellP, VelPred_j, kernel,
                                          Face_Area_Vec, Face_Area_Norm, V_i, V_j,
                                          Particle_Size_i, Particle_Size_j,
                                          face_vel_i, face_vel_j,
                                          face_area_dot_vel_val, riemann_S_M_val,
                                          cr_pressure_j_ptr, bhat, dt_hydrostep, Fluxes, out);
    }
    {
        const double *tau_c_i_ptr = nullptr;
#if defined(RT_SOLVER_EXPLICIT) && (N_RT_FREQ_BINS > 0)
        tau_c_i_ptr = tau_c_i;
#endif
        Vec3<MyDouble> particle_vel_j_for_rt = {};
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        particle_vel_j_for_rt = ParticleVel_j;
#endif
        rt_direct_ray_transport_compute_pair(local, P[j], CellP[j], VelPred_j, particle_vel_j_for_rt,
                                             Face_Area_Vec, Face_Area_Norm, V_i, V_j, Particle_Size_j,
                                             tau_c_i_ptr, dt_hydrostep, out);
        rt_diffusion_explicit_compute_pair(local, j, P, CellP, VelPred_j, particle_vel_j_for_rt,
                                           kernel, rinv, Face_Area_Vec, Face_Area_Norm,
                                           V_i, V_j, Particle_Size_i, Particle_Size_j,
                                           face_vel_i, face_vel_j, tau_c_i_ptr, dt_hydrostep, out);
    }

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
#if defined(MHD_NON_IDEAL) || defined(MHD_BATTERY_MECHANISMS)
    out.DtInternalEnergy += dot(local.BPred, bflux_from_nonideal_effects) * All.cf_a2inv;
#endif
#endif
#endif
#endif

    /* ---- J-particle writes (Kokkos atomics for thread safety) ----
     *
     * allow_j_writes gates the entire j-side block. Production passes
     * true (legacy behavior). The runner-Spec oracle "brute" pass
     * (Mode B + oracle, diagnostic-only) passes false so the oracle
     * dry-run does not mutate production CellP[j].dMass / P[j].wakeup
     * state — without this gate the brute pass would double-apply
     * j-side writes that the main pass already applied, corrupting
     * the simulation. Same shape as sink_feed's oracle_dry_run pattern. */
    /* Signal velocity for timestepping (i-side accum — MUST update every pair
     * regardless of allow_j_writes. Without this update outside the gate,
     * the oracle brute pass (allow_j_writes=false) leaves out.MaxSignalVel
     * stuck at the kernel.sound_i seed → uniform 2× mismatch against the
     * tree pass. Bug introduced by commit 8a (allow_j_writes gate) which
     * accidentally swallowed this i-side update; surfaced by the commit 8
     * runtime matrix oracle pass on 2026-05-20. */
    if(kernel.vsig > out.MaxSignalVel) {out.MaxSignalVel = kernel.vsig;}

    if(allow_j_writes) {

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

    /* Wakeup check: if j is inactive and signal velocity exceeds threshold,
       flag it for wakeup. Uses Kokkos atomics for thread safety. */
    if(TimeBinActive_arr && !(TimeBinActive_arr[P[j].TimeBin]))
    {
        if(kernel.vsig > WAKEUP * CellP[j].MaxSignalVel) {
            short int wakeup_val = (short int)(local.TimeBin + 1);
            HYDRO_ATOMIC_MAX(&P[j].wakeup, wakeup_val);
            if(NeedToWakeup_flag) HYDRO_ATOMIC_STORE(NeedToWakeup_flag, 1);
        }
    }

    } /* end allow_j_writes */
}


#endif /* HYDRO_FUNCTIONS_H */
