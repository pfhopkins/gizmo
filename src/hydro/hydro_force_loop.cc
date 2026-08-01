/* hydro/hydro_force_loop.cc — host hooks + toplevel for HydroForceSpec.
 *
 * Ports the legacy GPU walker hydro_evaluate_gpu (hydro/density_gpu.cc,
 * still alive for core/transport_subcycle.cc) to the runner Spec contract,
 * plus rewrites the toplevel hydro_force() (formerly in
 * hydro/hydro_toplevel.cc) to dispatch via run_neighbor_loop<HydroForceSpec>.
 * Final consumer of the hydro corridor's shared symmetric gas topology
 * (see hydro_corridor.h).
 *
 * Owns:
 *   - GHOST_WRITEBACK_BUNDLE(hydro_force) manifest (PARTICLE_MAX(wakeup) +
 *     MFV GAS_ADD(dMass))
 *   - DeviceContext extension lifecycle (UVM int[TIMEBINS] for
 *     TimeBinActive, UVM int for need_wakeup, oracle_dry_run)
 *   - Ghost lifecycle (ghost_writeback_begin pre-zeros ghost-region
 *     wakeup + MFV dMass before bundle snapshot; ghost_writeback_end
 *     calls bundle's reverse-comm; ghost_write_detector_begin/end forward
 *     to the mesh helpers)
 *   - apply_active_writeback (the body from the retired out2particle_hydra
 *     in hydro/hydro_toplevel.cc:115-295, with the MaxSignalVel floor
 *     restored via fmax(out->MaxSignalVel, effective_soundspeed()))
 *   - merge_accum + compare_accum (field-wise per #ifdef, mirroring
 *     the GradientsSpec pattern)
 *   - hydro_force() toplevel
 *
 * Written by Philip F. Hopkins (phopkins@caltech.edu) for GIZMO. */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"  /* MUST precede allvars.h: installs device-pass `#define All AllDeviceMirror` redirect before cell_data.h is parsed */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../system/gpu_particles_arena.h"
#include "../mesh/kernel.h"                   /* MUST precede hydro_force_loop.h
                                                * — kernel.h has no include guards */
#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/neighbor_list.h"            /* gizmo_sym_* globals */
#include "../mesh/ghost_symlist_lifecycle.h"  /* gizmo_hydro_cleanup_symlist_and_ghosts */
#include "../mesh/ghost_writeback.h"
#include "../mesh/ghost_writeback_ops.h"      /* GHOST_WRITEBACK_BUNDLE_* macros */
#include "hydro_corridor.h"
#include "hydro_force_loop.h"


/* External symbol exposed by the legacy hydro_force toplevel that the new
 * toplevel below replaces. NeedToWakeupParticles_local is the global
 * call-level event flag; cleanup_device_context OR's the device-side
 * accumulator into this. */
extern int NeedToWakeupParticles_local;

extern void hydro_force_initial_operations_preloop(void);
extern void hydro_final_operations_and_cleanup(void);

/* ============================================================================
 * Ghost-writeback manifest.
 *
 * Mode A imported-ghost lifecycle: the runner's snapshot-diff scaffold
 * (mesh/ghost_writeback.cc) generates snapshot / change-predicate / pack /
 * exchange / apply / cleanup machinery from each manifest line. ONE LINE per
 * (op, field) under the appropriate physics flag #ifdef.
 *
 * Mode B (single-rank local OR multi-rank remote): the runner skips
 * ghost-writeback hooks because Mode B doesn't import ghosts — j-side writes
 * land directly on the j-owning rank via request-driven peer evaluation
 * (same architecture as sink_feed multi-rank, sinks/sink_feed_loop.cc:178).
 *
 * Operations available: see mesh/ghost_writeback_ops.h.
 * ========================================================================== */

GHOST_WRITEBACK_BUNDLE_BEGIN(hydro_force)
    GHOST_WRITEBACK_PARTICLE_MAX(wakeup)
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    GHOST_WRITEBACK_GAS_ADD(dMass)
#endif
GHOST_WRITEBACK_BUNDLE_END(hydro_force)

/* ============================================================================
 * DeviceContext lifecycle.
 *
 * populate_device_context: allocate UVM TimeBinActive_uvm[TIMEBINS] + the
 *   single-int need_wakeup_uvm; copy host TimeBinActive[] into the UVM
 *   array; init need_wakeup to 0; oracle_dry_run = false.
 *
 * cleanup_device_context: OR the device-set wakeup flag into the global
 *   NeedToWakeupParticles_local (the AGS pattern — call-level event
 *   propagation lives here, not in apply_active_writeback or the
 *   toplevel caller). Free both UVM allocations.
 *
 * set_oracle_brute_pass: flip oracle_dry_run for the brute oracle pass; the
 *   pair_kernel reads this through ActiveData and gates `allow_j_writes`
 *   in hydro_accumulate_neighbor. Brute pass also sees
 *   need_wakeup_ptr = nullptr via the ternary in load_active so no spurious
 *   wakeup events accumulate from the diagnostic pass.
 * ========================================================================== */

void HydroForceSpec::populate_device_context(const neighbor_loop_args& args,
                                              DeviceContext& ctx)
{
    ctx.TimeBinActive_uvm = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(TIMEBINS * sizeof(int));
    for(int k = 0; k < TIMEBINS; k++) { ctx.TimeBinActive_uvm[k] = TimeBinActive[k]; }
    ctx.need_wakeup_uvm = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sizeof(int));
    *ctx.need_wakeup_uvm = 0;
    ctx.wakeup_dirty_base = WakeupDirty;   /* global UVM sidecar base; kernel marks WakeupDirty[j] on wakeup */
    ctx.oracle_dry_run = false;

#if defined(GALSF_ISMDUSTCHEM_MODEL) && (defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME)))
    /* Host-precompute per-active ISMDustChem passive-scalar diffusion values.
     * `return_ismdustchem_species_of_interest_for_diffusion_and_yields` reads
     * All.* grain tables that are not mirrored to All_dev — host-only. Pattern
     * matches legacy density_gpu.cc:169-188. Gate matches DeviceContext field
     * gating + load_active write reachability. */
    const int n_ismdc = NUM_ISMDUSTCHEM_PASSIVE_SCALARS;
    if(n_ismdc > 0 && args.num_active > 0) {
        double *ismdc_host = new double[(size_t)args.num_active * n_ismdc];
        for(int aa = 0; aa < args.num_active; aa++) {
            int ii = args.active_list[aa];
            for(int ki = 0; ki < n_ismdc; ki++) {
                ismdc_host[aa * n_ismdc + ki] =
                    return_ismdustchem_species_of_interest_for_diffusion_and_yields(
                        ii, ISMDUSTCHEM_SPECIES_OFFSET_IN_METALLICITY + ki, 0, CellP);
            }
        }
        ctx.ismdc_uvm = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
            (size_t)args.num_active * n_ismdc * sizeof(double));
        memcpy(ctx.ismdc_uvm, ismdc_host, (size_t)args.num_active * n_ismdc * sizeof(double));
        delete[] ismdc_host;
    } else {
        ctx.ismdc_uvm = nullptr;
    }
#endif
}

void HydroForceSpec::cleanup_device_context(const neighbor_loop_args& /*args*/,
                                             DeviceContext& ctx)
{
#if defined(GALSF_ISMDUSTCHEM_MODEL) && (defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME)))
    /* Only the owning ctx frees ismdc_uvm. Oracle shallow-copy contexts
     * (run_mode_b_local_with_oracle) share this pointer read-only and MUST
     * NOT free it; the runner constructs them without invoking
     * cleanup_device_context, so this branch is safe. */
    if(ctx.ismdc_uvm) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(ctx.ismdc_uvm);
        ctx.ismdc_uvm = nullptr;
    }
#endif
    if(ctx.need_wakeup_uvm) {
        if(*ctx.need_wakeup_uvm) { NeedToWakeupParticles_local = 1; }
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(ctx.need_wakeup_uvm);
        ctx.need_wakeup_uvm = nullptr;
    }
    if(ctx.TimeBinActive_uvm) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(ctx.TimeBinActive_uvm);
        ctx.TimeBinActive_uvm = nullptr;
    }
}

void HydroForceSpec::set_oracle_brute_pass(DeviceContext& ctx, bool on)
{
    ctx.oracle_dry_run = on;
}

/* ============================================================================
 * Ghost-region pre-zero + lifecycle hooks (Mode A only).
 *
 * ghost_writeback_begin zeros ONLY the ghost region (j in
 * [num_local, num_local+num_ghosts)) — NOT home-rank — because home-rank
 * accumulators are zeroed by hydro_force_initial_operations_preloop()
 * upstream. Then call gpu_particles_arena_invalidate() so the next kernel
 * acquire re-reads the zeroed host state, and engage the bundle's snapshot.
 *
 * ghost_writeback_end engages the bundle's snapshot-diff + reverse-comm
 * + apply machinery. The runner gates these hooks on
 * nlr_path_uses_imported_ghosts(plan.path), which is true only for Mode A —
 * Mode B paths skip this entirely (j-writes already on the right rank).
 * ========================================================================== */

/* ghost_write_detector_begin/end: runner default (loop_name = "hydro_force"). */

void HydroForceSpec::ghost_writeback_begin(const neighbor_loop_args& /*args*/,
                                            const NeighborLoopPlan& /*plan*/)
{
    int num_ghosts = ghost_get_num_ghosts();
    int num_local  = ghost_get_num_local();
    if(num_ghosts > 0) {
        for(int g = 0; g < num_ghosts; g++) {
            int j = num_local + g;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
            CellP[j].dMass = 0;
#endif
            P[j].wakeup = 0;
        }
        gpu_particles_arena_invalidate();
    }
    ghost_writeback_begin_bundle(hydro_force_ghost_writeback_bundle_ptr());
}

void HydroForceSpec::ghost_writeback_end(const neighbor_loop_args& /*args*/,
                                          const NeighborLoopPlan& /*plan*/)
{
    ghost_writeback_end_bundle(hydro_force_ghost_writeback_bundle_ptr());
}

/* ============================================================================
 * apply_active_writeback — replays the legacy out2particle_hydra body
 * (hydro/hydro_toplevel.cc:115-295 in the pre-port code). With the addition
 * of a MaxSignalVel floor at end: legacy seeded `out.MaxSignalVel =
 * kernel.sound_i` BEFORE the pair loop, so a zero-neighbor row ended with
 * MaxSignalVel = sound_i (not zero, which would crash CFL). The runner's
 * zero_accum can't see sound_i, so the seed happens here at writeback,
 * via fmax(accum.MaxSignalVel, CellP[i].effective_soundspeed()).
 *
 * `out->Var += ...` semantics retained: legacy zeroes the targets at
 * hydro_force_initial_operations_preloop() entry, so per-active += correctly
 * accumulates the runner's per-active reduced contribution.
 * ========================================================================== */

void HydroForceSpec::apply_active_writeback(const neighbor_loop_args& /*args*/,
                                             int /*active_slot*/, int i,
                                             const AccumData& accum_in)
{
    /* Legacy used non-const out2particle_hydra(struct hydro_data_out *out, ...);
     * the body only reads accum but uses non-const access for some macro
     * helpers. const_cast to reuse the legacy body verbatim. */
    struct hydro_data_out *out = const_cast<struct hydro_data_out*>(&accum_in);
    int k;

#if defined(MAGNETIC) && defined(MHD_BATTERY_MECHANISMS)
    {
        Vec3<double> dBdt_battery_total = {};
        const double dBdt_phys_to_code =
            UNIT_TIME_IN_CGS / All.UnitMagneticField_in_gauss / DMAX(All.cf_a2inv, MIN_REAL_NUMBER);
#if (MHD_BATTERY_MECHANISMS & 1)
        {
            const double n_e_cgs = CellP[i].n_e();
            const double T_e_cgs = CellP[i].T_e();
            if((n_e_cgs > 0) && (T_e_cgs > 0)) {
                const Vec3<double> g_Te_code = CellP[i].Gradients.ElectronTemperature;
                const Vec3<double> g_ne_code = CellP[i].Gradients.ElectronNumberDensity;
                const double inv_L_cgs = 1.0 / (UNIT_LENGTH_IN_CGS * All.cf_atime);
                const Vec3<double> g_Te_phys = g_Te_code * inv_L_cgs;
                const Vec3<double> g_ne_phys = g_ne_code * inv_L_cgs;
                const double prefac = C_LIGHT_CGS * BOLTZMANN_CGS
                    / (ELECTRONCHARGE_CGS * DMAX(n_e_cgs, MIN_REAL_NUMBER));
                const Vec3<double> dBdt_Bier_phys_cgs = prefac * cross(g_Te_phys, g_ne_phys);
                dBdt_battery_total += dBdt_Bier_phys_cgs * dBdt_phys_to_code;
            }
        }
#endif
#if (MHD_BATTERY_MECHANISMS & (2|4|8))
        {
            const double inv_L_cgs = 1.0 / (UNIT_LENGTH_IN_CGS * All.cf_atime);
            Mat3<double> gradE_phys_cgs;
            for(int kr = 0; kr < 3; kr++) {
                for(int kc = 0; kc < 3; kc++) {
                    gradE_phys_cgs[kr][kc] = ((double)CellP[i].Gradients.E_battery_T2[kr][kc]) * inv_L_cgs;
                }
            }
            const Vec3<double> curl_E_phys_cgs = gradE_phys_cgs.curl();
            const Vec3<double> dBdt_T2_phys_cgs = -C_LIGHT_CGS * curl_E_phys_cgs;
            dBdt_battery_total += dBdt_T2_phys_cgs * dBdt_phys_to_code;
        }
#endif
        const double dt_code   = get_particle_timestep_in_physical(i);
        const double V_code    = P[i].Mass / DMAX(CellP[i].Density, MIN_REAL_NUMBER);
        const Vec3<double> B_phys_codeunits = CellP[i].Bfield() * All.cf_a2inv;
        const Vec3<double> dB_step          = dBdt_battery_total * dt_code;
        const double dEmag_cell = fabs(dot(B_phys_codeunits, dB_step)) * V_code;
        const double E_internal_cell = P[i].Mass * CellP[i].InternalEnergyPred;
        const double P_thermal_phys  = CellP[i].Pressure * All.cf_a3inv;
        const double P_mag_phys      = 0.5 * B_phys_codeunits.norm_sq();
        const double E_pressure_cell = (P_thermal_phys + P_mag_phys) * V_code;
        const double eps = 0.1;
        const double allowed = eps * DMIN(DMAX(E_internal_cell, MIN_REAL_NUMBER),
                                          DMAX(E_pressure_cell, MIN_REAL_NUMBER));
        if((dEmag_cell > allowed) && (dEmag_cell > 0)) {
            const double scale = allowed / dEmag_cell;
            dBdt_battery_total *= scale;
        }
        out->DtB += dBdt_battery_total;
    }
#endif

    CellP[i].HydroAccel       += out->Acc;
    CellP[i].DtInternalEnergy += out->DtInternalEnergy;
#if defined(TWO_TEMPERATURE_PLASMA) && (TWO_TEMPERATURE_PLASMA & 4) && defined(CONDUCTION)
    CellP[i].DtInternalEnergy_FromConduction += out->DtInternalEnergy_FromConduction;
#endif
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    CellP[i].DtMass       += out->DtMass;
    CellP[i].dMass        += out->dMass;
    CellP[i].GravWorkTerm += out->GravWorkTerm;
#endif
    /* MaxSignalVel — legacy seeded `out.MaxSignalVel = kernel.sound_i`
     * before the pair loop. Replicate by flooring with effective_soundspeed
     * so zero-neighbor actives don't end with MaxSignalVel = 0. */
    {
        double max_sig = fmax((double)out->MaxSignalVel, CellP[i].effective_soundspeed());
        if(CellP[i].MaxSignalVel < max_sig) { CellP[i].MaxSignalVel = (MyFloat)max_sig; }
    }
#ifdef OUTPUT_SHOCK_MACH_NUMBER
    if(CellP[i].ShockMachNumber < out->MaxShockMachNumber) { CellP[i].ShockMachNumber = out->MaxShockMachNumber; }
#endif
#ifdef ENERGY_ENTROPY_SWITCH_IS_ACTIVE
    if(CellP[i].MaxKineticEnergyNgb < out->MaxKineticEnergyNgb) { CellP[i].MaxKineticEnergyNgb = out->MaxKineticEnergyNgb; }
#endif
#if defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME))
    for(k = 0; k < NUM_METAL_SPECIES; k++) { CellP[i].Dyield[k] += out->Dyield[k]; }
#endif
#ifdef CHIMES_TURB_DIFF_IONS
    for(k = 0; k < ChimesGlobalVars.totalNumberOfSpecies; k++) {
        CellP[i].ChimesNIons[k] = DMAX(CellP[i].ChimesNIons[k] + out->ChimesIonsYield[k],
                                       0.5 * CellP[i].ChimesNIons[k]);
    }
#endif
#if defined(RT_SOLVER_EXPLICIT)
#if defined(RT_EVOLVE_ENERGY)
    for(k = 0; k < N_RT_FREQ_BINS; k++) { CellP[i].Dt_Rad_E_gamma[k] += out->Dt_Rad_E_gamma[k]; }
#endif
#if defined(RT_EVOLVE_FLUX)
    for(k = 0; k < N_RT_FREQ_BINS; k++) { CellP[i].Dt_Rad_Flux[k] += out->Dt_Rad_Flux[k]; }
#endif
#if defined(RT_INFRARED)
    CellP[i].Dt_Rad_E_gamma_T_weighted_IR += out->Dt_Rad_E_gamma_T_weighted_IR;
#endif
#if defined(RT_EVOLVE_INTENSITIES)
    for(k = 0; k < N_RT_FREQ_BINS; k++) {
        for(int k_dir = 0; k_dir < N_RT_INTENSITY_BINS; k_dir++) {
            CellP[i].Dt_Rad_Intensity[k][k_dir] += out->Dt_Rad_Intensity[k][k_dir];
        }
    }
#endif
#endif
#if defined(MAGNETIC)
    CellP[i].DtB       += out->DtB;
    CellP[i].Face_Area += out->Face_Area;
    CellP[i].divB      += out->divB;
#if defined(DIVBCLEANING_DEDNER)
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    CellP[i].DtPhi += out->DtPhi;
#endif
    CellP[i].DtB_PhiCorr += out->DtB_PhiCorr;
#endif
#endif
#ifdef COSMIC_RAY_FLUID
    CellP[i].Face_DivVel_ForAdOps += out->Face_DivVel_ForAdOps;
#if defined(CRFLUID_INJECTION_AT_SHOCKS)
    CellP[i].DtCREgyNewInjectionFromShocks += out->DtCREgyNewInjectionFromShocks;
#endif
    for(k = 0; k < N_CR_PARTICLE_BINS; k++) {
        CellP[i].DtCosmicRayEnergy[k] += out->DtCosmicRayEnergy[k];
#if defined(CRFLUID_EVOLVE_SPECTRUM)
        CellP[i].DtCosmicRay_Number_in_Bin[k] += out->DtCosmicRay_Number_in_Bin[k];
#endif
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
        for(int kAlf = 0; kAlf < 2; kAlf++) { CellP[i].DtCosmicRayAlfvenEnergy[k][kAlf] += out->DtCosmicRayAlfvenEnergy[k][kAlf]; }
#endif
    }
#endif
}

/* ============================================================================
 * merge_accum — peer-rank accum reduction (Mode B remote). Additive for
 * Acc / DtInternalEnergy / DtMass / dMass / GravWorkTerm / Dyield /
 * Dt_Rad_* / DtB / Face_Area / divB / CR fields. MAX for MaxSignalVel /
 * MaxShockMachNumber / MaxKineticEnergyNgb. Mirrors apply_active_writeback's
 * #ifdef coverage exactly so the runner's "merge across peers then apply"
 * fold composes correctly.
 * ========================================================================== */

void HydroForceSpec::merge_accum(AccumData& dst, const AccumData& src)
{
#define MERGE_ADD_VEC3(field) do { for(int kv = 0; kv < 3; kv++) dst.field[kv] += src.field[kv]; } while(0)
#define MERGE_ADD(field)      do { dst.field += src.field; } while(0)
#define MERGE_MAX(field)      do { if(src.field > dst.field) dst.field = src.field; } while(0)

    for(int kv = 0; kv < 3; kv++) { dst.Acc[kv] += src.Acc[kv]; }
    MERGE_ADD(DtInternalEnergy);
#if defined(TWO_TEMPERATURE_PLASMA) && (TWO_TEMPERATURE_PLASMA & 4) && defined(CONDUCTION)
    MERGE_ADD(DtInternalEnergy_FromConduction);
#endif
    MERGE_MAX(MaxSignalVel);
#ifdef OUTPUT_SHOCK_MACH_NUMBER
    MERGE_MAX(MaxShockMachNumber);
#endif
#ifdef ENERGY_ENTROPY_SWITCH_IS_ACTIVE
    MERGE_MAX(MaxKineticEnergyNgb);
#endif
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    MERGE_ADD(DtMass);
    MERGE_ADD(dMass);
    for(int kv = 0; kv < 3; kv++) { dst.GravWorkTerm[kv] += src.GravWorkTerm[kv]; }
#endif
#if defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME))
    for(int k = 0; k < NUM_METAL_SPECIES; k++) { dst.Dyield[k] += src.Dyield[k]; }
#endif
#ifdef CHIMES_TURB_DIFF_IONS
    for(int k = 0; k < ChimesGlobalVars.totalNumberOfSpecies; k++) { dst.ChimesIonsYield[k] += src.ChimesIonsYield[k]; }
#endif
#if defined(RT_SOLVER_EXPLICIT)
#if defined(RT_EVOLVE_ENERGY)
    for(int k = 0; k < N_RT_FREQ_BINS; k++) { dst.Dt_Rad_E_gamma[k] += src.Dt_Rad_E_gamma[k]; }
#endif
#if defined(RT_EVOLVE_FLUX)
    for(int k = 0; k < N_RT_FREQ_BINS; k++) { for(int kv = 0; kv < 3; kv++) { dst.Dt_Rad_Flux[k][kv] += src.Dt_Rad_Flux[k][kv]; } }
#endif
#if defined(RT_INFRARED)
    MERGE_ADD(Dt_Rad_E_gamma_T_weighted_IR);
#endif
#if defined(RT_EVOLVE_INTENSITIES)
    for(int k = 0; k < N_RT_FREQ_BINS; k++) {
        for(int k_dir = 0; k_dir < N_RT_INTENSITY_BINS; k_dir++) { dst.Dt_Rad_Intensity[k][k_dir] += src.Dt_Rad_Intensity[k][k_dir]; }
    }
#endif
#endif
#if defined(MAGNETIC)
    for(int kv = 0; kv < 3; kv++) { dst.Face_Area[kv] += src.Face_Area[kv]; }
    for(int kv = 0; kv < 3; kv++) { dst.DtB[kv] += src.DtB[kv]; }
    MERGE_ADD(divB);
#if defined(DIVBCLEANING_DEDNER)
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    MERGE_ADD(DtPhi);
#endif
    for(int kv = 0; kv < 3; kv++) { dst.DtB_PhiCorr[kv] += src.DtB_PhiCorr[kv]; }
#endif
#endif
#ifdef COSMIC_RAY_FLUID
    MERGE_ADD(Face_DivVel_ForAdOps);
#if defined(CRFLUID_INJECTION_AT_SHOCKS)
    MERGE_ADD(DtCREgyNewInjectionFromShocks);
#endif
    for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {
        dst.DtCosmicRayEnergy[k] += src.DtCosmicRayEnergy[k];
#if defined(CRFLUID_EVOLVE_SPECTRUM)
        dst.DtCosmicRay_Number_in_Bin[k] += src.DtCosmicRay_Number_in_Bin[k];
#endif
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
        for(int kAlf = 0; kAlf < 2; kAlf++) { dst.DtCosmicRayAlfvenEnergy[k][kAlf] += src.DtCosmicRayAlfvenEnergy[k][kAlf]; }
#endif
    }
#endif

#undef MERGE_ADD_VEC3
#undef MERGE_ADD
#undef MERGE_MAX
}

/* ============================================================================
 * compare_accum — oracle gate. Field-wise max-of-relative-residuals with
 * an absolute-difference floor of 1e-12: sum-to-zero cancellation fields
 * legitimately end at ~1e-22 in some configs, and a pure relative residual
 * mis-flags those as O(1) "mismatches". Coverage mirrors merge_accum
 * exactly.
 * ========================================================================== */

double HydroForceSpec::compare_accum(const AccumData& a, const AccumData& b)
{
    double worst = 0.0;
    auto rel = [](double x, double y) {
        constexpr double abs_tol = 1e-12;
        double d = fabs(x - y);
        if(d < abs_tol) return 0.0;
        double m = fmax(fmax(fabs(x), fabs(y)), 1e-30);
        return d / m;
    };
#define CHECK(field) do { double r = rel((double)a.field, (double)b.field); \
                          if(r > worst) worst = r; } while(0)

    for(int kv = 0; kv < 3; kv++) { CHECK(Acc[kv]); }
    CHECK(DtInternalEnergy);
#if defined(TWO_TEMPERATURE_PLASMA) && (TWO_TEMPERATURE_PLASMA & 4) && defined(CONDUCTION)
    CHECK(DtInternalEnergy_FromConduction);
#endif
    CHECK(MaxSignalVel);
#ifdef OUTPUT_SHOCK_MACH_NUMBER
    CHECK(MaxShockMachNumber);
#endif
#ifdef ENERGY_ENTROPY_SWITCH_IS_ACTIVE
    CHECK(MaxKineticEnergyNgb);
#endif
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    CHECK(DtMass);
    CHECK(dMass);
    for(int kv = 0; kv < 3; kv++) { CHECK(GravWorkTerm[kv]); }
#endif
#if defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME))
    for(int k = 0; k < NUM_METAL_SPECIES; k++) { CHECK(Dyield[k]); }
#endif
#ifdef CHIMES_TURB_DIFF_IONS
    for(int k = 0; k < ChimesGlobalVars.totalNumberOfSpecies; k++) { CHECK(ChimesIonsYield[k]); }
#endif
#if defined(RT_SOLVER_EXPLICIT)
#if defined(RT_EVOLVE_ENERGY)
    for(int k = 0; k < N_RT_FREQ_BINS; k++) { CHECK(Dt_Rad_E_gamma[k]); }
#endif
#if defined(RT_EVOLVE_FLUX)
    for(int k = 0; k < N_RT_FREQ_BINS; k++) { for(int kv = 0; kv < 3; kv++) { CHECK(Dt_Rad_Flux[k][kv]); } }
#endif
#if defined(RT_INFRARED)
    CHECK(Dt_Rad_E_gamma_T_weighted_IR);
#endif
#if defined(RT_EVOLVE_INTENSITIES)
    for(int k = 0; k < N_RT_FREQ_BINS; k++) {
        for(int k_dir = 0; k_dir < N_RT_INTENSITY_BINS; k_dir++) { CHECK(Dt_Rad_Intensity[k][k_dir]); }
    }
#endif
#endif
#if defined(MAGNETIC)
    for(int kv = 0; kv < 3; kv++) { CHECK(Face_Area[kv]); }
    for(int kv = 0; kv < 3; kv++) { CHECK(DtB[kv]); }
    CHECK(divB);
#if defined(DIVBCLEANING_DEDNER)
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    CHECK(DtPhi);
#endif
    for(int kv = 0; kv < 3; kv++) { CHECK(DtB_PhiCorr[kv]); }
#endif
#endif
#ifdef COSMIC_RAY_FLUID
    CHECK(Face_DivVel_ForAdOps);
#if defined(CRFLUID_INJECTION_AT_SHOCKS)
    CHECK(DtCREgyNewInjectionFromShocks);
#endif
    for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {
        CHECK(DtCosmicRayEnergy[k]);
#if defined(CRFLUID_EVOLVE_SPECTRUM)
        CHECK(DtCosmicRay_Number_in_Bin[k]);
#endif
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
        for(int kAlf = 0; kAlf < 2; kAlf++) { CHECK(DtCosmicRayAlfvenEnergy[k][kAlf]); }
#endif
    }
#endif

#undef CHECK
    return worst;
}

/* ============================================================================
 * Toplevel: hydro_force
 *
 * Sequence:
 *   1. hydro_force_initial_operations_preloop()  (verbatim host pre-processing)
 *   2. corridor ghost-value refresh, then consume the corridor CSR when
 *      published (Mode A, any rank count — see hydro_corridor.h) OR reuse
 *      gizmo_sym_active_indices (built upstream in gradients toplevel)
 *   3. run_neighbor_loop<HydroForceSpec>(args)  (runner handles ghost
 *      lifecycle bundle + apply_active_writeback + cleanup_device_context's
 *      need_wakeup OR-into-global)
 *   4. hydro_final_operations_and_cleanup()  (verbatim host post-processing)
 *   5. gizmo_hydro_cleanup_symlist_and_ghosts()  (corridor CSR teardown)
 *
 * Empty-globally case: if no active gas anywhere and corridor didn't
 * pre-build a CSR, skip the runner call but still run pre/post + cleanup
 * — same shape as gradients toplevel's nothing_to_do path. The legacy
 * walker also called these even on empty steps.
 * ========================================================================== */

void hydro_force(void)
{
    CPU_Step[CPU_MISC] += measure_time();
    double t00_truestart = my_second();
    double t_preloop_start = my_second();
    hydro_force_initial_operations_preloop();
    double t_preloop = timediff(t_preloop_start, my_second());
    (void)t_preloop;

    /* Corridor ghost refresh FIRST, view fetch AFTER: owner-side values
     * changed since the last corridor refresh (gradients results, MG
     * gradient correction, dynamic-diffusion calc), and intervening loops'
     * own runner import+cleanup may have torn the corridor pool down — the
     * refresh re-imports and republishes the CSR view, so a fetch before it
     * would be stale. No-op on Mode B / unpublished corridor. */
    gizmo_hydro_corridor_refresh_ghost_values("pre_hydro_force");
    const nlr_external_csr        *corridor_csr  = gizmo_hydro_corridor_external_csr();
    const GizmoHydroCorridorMode   corridor_mode = gizmo_hydro_corridor_get_mode();

    const bool nothing_to_do =
        (corridor_csr == nullptr) && (gizmo_sym_num_active_global <= 0);

    if(!nothing_to_do) {
        neighbor_loop_args args = nlr_default_args();
        if(corridor_csr != nullptr) {
            args.active_list       = corridor_csr->active_indices;
            args.num_active        = corridor_csr->num_active;
            args.external_csr      = corridor_csr;
            args.dispatch_override = NlrForceMode::A;
        } else {
            /* Mode B (request-driven, no corridor CSR): corridor-built
             * active list. Mode A always publishes a view when there is
             * active gas; reaching here in Mode A is a corridor sequencing
             * bug — fail loudly, never quietly rebuild (dual path retired). */
            if(corridor_mode == GizmoHydroCorridorMode::MODE_A) {
                printf("FATAL: hydro_force in Mode A with active gas but no published corridor CSR on task %d.\n", ThisTask);
                fflush(stdout);
                endrun(7317);
            }
            args.active_list = gizmo_sym_active_indices;
            args.num_active  = gizmo_sym_num_active;
            if(corridor_mode == GizmoHydroCorridorMode::MODE_B) {
                args.dispatch_override = NlrForceMode::B;
            }
        }
        run_neighbor_loop<HydroForceSpec>(args);
    }

    double t_postloop_start = my_second();
    hydro_final_operations_and_cleanup();
    double t_postloop = timediff(t_postloop_start, my_second());
    (void)t_postloop;

    gizmo_hydro_cleanup_symlist_and_ghosts();

    double t1 = my_second(); cpu_chain_sync(t1);
    (void)t1; (void)t00_truestart;
}
