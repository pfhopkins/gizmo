/* hydro/hydro_force_loop.h — HydroForceSpec for the neighbor-loop runner.
 *
 * Ports the legacy GPU walker `hydro_evaluate_gpu` (hydro/density_gpu.cc:
 * 106-481) to the runner Spec contract. Closes the Wave 5 hydro corridor
 * (cellcorrections + gradients + hydro_force all on the runner). Pair body
 * `hydro_accumulate_neighbor` is reused verbatim from
 * hydro/hydro_functions.h — physics unchanged. The j-side write block
 * (P[j].wakeup atomic-max, MFV CellP[j].dMass atomic-add, NeedToWakeup_flag
 * store) is gated by `allow_j_writes` (commit 8a, a2cb965a) so the runner's
 * Mode B + oracle brute pass can dry-run without mutating production state.
 *
 * Ghost writeback (Mode A imported-ghost path): runner's snapshot-diff
 * bundle handles wakeup + MFV dMass reverse-comm via PARTICLE_MAX(wakeup)
 * + GAS_ADD(dMass) declared in hydro_force_loop.cc. Pre-zero of ghost-region
 * P[j].wakeup + CellP[j].dMass happens in `ghost_writeback_begin` (ghost
 * region only — home-rank zeroing stays in hydro_force_initial_operations_
 * preloop). Mode B path uses request-driven peer-to-peer routing; j-writes
 * land on the j-owning rank directly inside its local pair_kernel — no
 * reverse-comm needed, same architecture as sink_feed multi-rank.
 *
 * Written by Philip F. Hopkins (phopkins@caltech.edu) for GIZMO. */

#ifndef HYDRO_FORCE_LOOP_H
#define HYDRO_FORCE_LOOP_H

/* Kokkos_Core.hpp MUST precede declarations/allvars.h: allvars.h pulls in
 * declarations/macros.h which #defines `terminate(...)`; that macro mangles
 * std::terminate inside Kokkos transitive <exception> include if allvars.h
 * comes first. Same convention as sinks/sink_feed_loop.h / sink_swk_loop.h. */
#include <Kokkos_Core.hpp>
#include "../declarations/allvars.h"
#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/mode_b_local_walker.h"      /* MODE_B_SEARCH_*, MODE_B_RADIUS_* */
#include "../core/timestep_functions.h"       /* get_particle_timestep_in_physical */
#include "hydro_structs.h"                    /* hydro_data_in / hydro_data_out */

/* reimann.h provides Input_vec_Riemann / Riemann_outputs / Riemann_solver
 * used inside hydro_core_meshless_functions.h (transitively included by
 * hydro_functions.h). Match the legacy GPU TU's gating
 * (hydro/density_gpu.cc:48-50) — SPH builds use a different solver path
 * defined inline in hydro_core_sph_functions.h. */
#ifndef HYDRO_SPH
#include "reimann.h"
#endif

/* hydro_functions.h is template-style on INPUT_STRUCT_NAME / OUTPUT_STRUCT_NAME
 * (see hydro/density_gpu.cc:51-56). Define them BEFORE the include so the
 * `hydro_accumulate_neighbor` declaration resolves with our struct types.
 * Each TU that includes hydro_force_loop.h establishes its own definition. */
#define INPUT_STRUCT_NAME hydro_data_in
#define OUTPUT_STRUCT_NAME hydro_data_out
#include "hydro_functions.h"
#undef INPUT_STRUCT_NAME
#undef OUTPUT_STRUCT_NAME

/* NOTE: caller translation units must include "../mesh/kernel.h" BEFORE
 * this header. kernel.h has no include guards; defines static inline
 * kernel_main / kernel_hinv used by the pair body. Matches the
 * cellcorrections_loop.h / gradients_loop.h / sink_env1_loop.h convention. */

/* No KOKKOS_INLINE_FUNCTION fallback here — Kokkos_Core.hpp is included
 * unconditionally above. This Spec carries device-callable pair-kernel
 * accessors; misordered includes must compile-fail loudly, not silently
 * resolve to host-only `inline`. Same convention as neighbor_loop_runner.h. */

/* ============================================================================
 * Per-pair physics types.
 * ========================================================================== */

/* ActiveData = hydro_data_in `local` plus per-active runner geometry (pos /
 * h_search — required by Mode B remote walker). Pointer fields that the
 * pair body dereferences on the j-owning rank (TimeBinActive_ptr,
 * need_wakeup_ptr) live in NeighborData below — they MUST be peer-ctx
 * pointers in Mode B remote, not serialized origin pointers, or the peer
 * segfaults on dereference. oracle_dry_run lives in NeighborData too so the
 * j-write gate is derived consistently from the evaluating rank's ctx. */
struct HydroForceActiveData
{
    /* Mode B remote walker requirements */
    Vec3<double> pos;
    double       h_search;

    /* Per-active input — the per-particle hydra block */
    struct hydro_data_in local;
};

/* NeighborData carries the integer j plus the j-owning rank's pointers /
 * gates. hydro_accumulate_neighbor indexes P[j] / CellP[j] for ~30 fields
 * AND dereferences TimeBinActive_ptr / need_wakeup_ptr on the j-owning rank.
 * load_neighbor populates these from the evaluating rank's ctx (peer ctx on
 * Mode B remote, origin ctx everywhere else), so the same Spec compiles
 * cleanly for all dispatch paths. */
struct HydroForceNeighborData
{
    int                   j;
    struct particle_data *P;
    struct gas_cell_data *CellP;
    int                  *TimeBinActive_ptr;    /* evaluating ctx UVM */
    int                  *need_wakeup_ptr;      /* evaluating ctx UVM; nullptr under oracle dry-run */
    bool                  oracle_dry_run;       /* gates allow_j_writes */
};

/* DeviceContext extension. UVM pointers backing the per-call scratch the
 * pair body atomically writes (need_wakeup_uvm) or reads (TimeBinActive_uvm),
 * plus the oracle_dry_run flag set by the runner's set_oracle_brute_pass
 * before the brute oracle pass. Trivially copyable — captured by value into
 * Kokkos device lambdas (static_assert in runner). */
struct HydroForceDeviceContext : NeighborLoopDeviceContextBase
{
    int  *TimeBinActive_uvm;   /* UVM int[TIMEBINS]; populate copies from host */
    int  *need_wakeup_uvm;     /* UVM int[1]; populate inits to 0 */
    bool  oracle_dry_run;      /* set true by set_oracle_brute_pass */
#if defined(GALSF_ISMDUSTCHEM_MODEL) && (defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME)))
    /* UVM double[num_active * NUM_ISMDUSTCHEM_PASSIVE_SCALARS]; host-precomputed
     * per-active passive scalars (legacy density_gpu.cc:169-188). nullptr when
     * NUM_ISMDUSTCHEM_PASSIVE_SCALARS == 0 || num_active == 0. Owning context
     * frees it in cleanup_device_context; oracle shallow-copy ctx must NOT
     * free (read-only alias of the same UVM pointer). Gate matches
     * load_active's actual write reachability (hydro_data_in::Metallicity is
     * gated by TURB_DIFF_METALS || METALS+MFV — see hydro_structs.h:129). */
    double *ismdc_uvm;
#endif
};

/* ============================================================================
 * HydroForceSpec — NeighborLoopSpec contract.
 * ========================================================================== */
struct HydroForceSpec
{
    static constexpr const char *loop_name = "hydro_force";

    /* Symmetric gas-gas topology (matches gizmo_sym_neighbor_list + the
     * legacy hydro_evaluate_gpu CSR consumer). */
    static constexpr int                     search_mode        = MODE_B_SEARCH_SYMMETRIC;
    static constexpr unsigned int            neighbor_type_mask = (1u << 0);
    static constexpr mode_b_radius_policy_t  radius_policy      = MODE_B_RADIUS_DEFAULT;

    static constexpr WritePattern   write_pattern              = WritePattern::ActiveReduceOnly;
    static constexpr SidxCacheKind  sidx_cache_kind            = SidxCacheKind::GasOnly;
    static constexpr double         accum_tolerance            = 1e-10;
    static constexpr bool           uses_ghost_writeback       = true;
    static constexpr bool           uses_ghost_write_detector  = true;

    /* Broad active predicate matching the legacy GPU walker's row source
     * `gizmo_sym_active_indices` (Type==0 && Mass>0). */
    static bool is_active(int i) {
        if(P[i].Type != 0) return false;
        if(P[i].Mass <= 0) return false;
        return true;
    }

    using CallScalars   = NlrCommonScalars;
    using ActiveData    = HydroForceActiveData;
    using AccumData     = struct hydro_data_out;
    using NeighborData  = HydroForceNeighborData;

    using ScatterData    = NoScatter;
    using IdentityFields = NoIdentity;
    using IterControl    = NotIterative;
    using DeviceContext  = HydroForceDeviceContext;

    /* Pre-arena per-active search radius. */
    static double search_radius(const neighbor_loop_args& /*args*/,
                                 int /*active_slot*/, int i)
    {
        return (double)P[i].KernelRadius;
    }

    /* Per-call scalars: only the common cosmology block. The pair body reads
     * All.* directly (bare All.* is fine under the All-mirror refactor;
     * see feedback_bare_all_in_pair_body_ok.md). */
    static CallScalars populate_call_scalars(const neighbor_loop_args& /*args*/)
    {
        return nlr_common_scalars_from_all();
    }

    /* DeviceContext lifecycle (body in hydro_force_loop.cc) */
    static void populate_device_context(const neighbor_loop_args& args,
                                         DeviceContext& ctx);
    static void cleanup_device_context (const neighbor_loop_args& args,
                                         DeviceContext& ctx);
    static void set_oracle_brute_pass  (DeviceContext& ctx, bool on);

    /* Ghost lifecycle hooks. Mode A snapshot-diff lifecycle:
     *   ghost_write_detector_begin -> ghost_writeback_begin -> kernel ->
     *   ghost_writeback_end (snapshot-diff reverse-comm) ->
     *   ghost_write_detector_end.
     * Mode B skips these (no imported ghosts; j-writes land on owner rank
     * directly via request-driven P2P). Bodies in hydro_force_loop.cc. */
    static void ghost_write_detector_begin(const struct neighbor_loop_args&,
                                             const struct NeighborLoopPlan&);
    static void ghost_write_detector_end  (const struct neighbor_loop_args&,
                                             const struct NeighborLoopPlan&);
    static void ghost_writeback_begin     (const struct neighbor_loop_args&,
                                             const struct NeighborLoopPlan&);
    static void ghost_writeback_end       (const struct neighbor_loop_args&,
                                             const struct NeighborLoopPlan&);

    /* ----- Device hooks (KOKKOS_INLINE_FUNCTION) ----- */

    KOKKOS_INLINE_FUNCTION
    static void zero_accum(AccumData& accum)
    {
        for(size_t b = 0; b < sizeof(accum); b++) ((char*)&accum)[b] = 0;
    }

    /* Build ActiveData per active i. Byte-exact reconstruction of the
     * particle2in-equivalent block from hydro/density_gpu.cc:209-357. */
    KOKKOS_INLINE_FUNCTION
    static ActiveData load_active(const DeviceContext& ctx,
                                   int active_slot, int i,
                                   double h_search,
                                   const CallScalars& /*scalars*/)
    {
        ActiveData active;
        for(size_t b = 0; b < sizeof(active); b++) ((char*)&active)[b] = 0;

        struct hydro_data_in &local = active.local;
        struct particle_data *kp    = ctx.P;
        struct gas_cell_data *kc    = ctx.CellP;

        active.pos             = kp[i].Pos;
        active.h_search        = h_search;

        local.Pos = kp[i].Pos;
        local.Vel = kc[i].VelPred;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        local.ParticleVel = kc[i].VelPred;
#endif
        local.KernelRadius      = kp[i].KernelRadius;
        local.Mass              = kp[i].Mass;
        local.Density           = kc[i].Density;
        local.Pressure          = kc[i].Pressure;
        local.ConditionNumber   = kc[i].ConditionNumber;
#ifdef MHD_CONSTRAINED_GRADIENT
        if(kc[i].FlagForConstrainedGradients == 0) { local.ConditionNumber *= -1; }
#endif
        local.FaceClosureError    = kc[i].FaceClosureError;
        local.InternalEnergyPred  = kc[i].InternalEnergyPred;
        local.SoundSpeed          = kc[i].effective_soundspeed();
        local.dt_hydrostep_i      = get_particle_timestep_in_physical(i, kp);
        local.DrkernNgbFactor     = kp[i].DrkernNgbFactor;
        local.Gradients.Density   = kc[i].Gradients.Density;
        local.Gradients.Pressure  = kc[i].Gradients.Pressure;
        local.Gradients.Velocity  = kc[i].Gradients.Velocity;
        local.NV_T                = kc[i].NV_T;
        local.TimeBin             = kp[i].TimeBin;
#ifdef MAGNETIC
        local.BPred       = kc[i].Bfield();
        local.Gradients.B = kc[i].Gradients.B;
#ifdef MHD_BATTERY_MECHANISMS
#if (MHD_BATTERY_MECHANISMS & 1)
        local.Gradients.ElectronNumberDensity = kc[i].Gradients.ElectronNumberDensity;
        local.Gradients.ElectronTemperature   = kc[i].Gradients.ElectronTemperature;
#endif
#endif
#ifdef DIVBCLEANING_DEDNER
        local.PhiPred       = kc[i].PhiPred / kp[i].Mass;
        local.Gradients.Phi = kc[i].Gradients.Phi;
#endif
#ifdef MHD_MODIFIED_GRADIENT
        local.MG_cgcoeff = kc[i].MG_cgcoeff;
#endif
#endif
#ifdef DOGRAD_INTERNAL_ENERGY
        local.Gradients.InternalEnergy = kc[i].Gradients.InternalEnergy;
#endif
#ifdef DOGRAD_SOUNDSPEED
        local.Gradients.SoundSpeed = kc[i].Gradients.SoundSpeed;
#endif
#if defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME))
        for(int k = 0; k < NUM_METAL_SPECIES; k++) { local.Metallicity[k] = kp[i].Metallicity[k]; }
#endif
#ifdef COSMIC_RAY_FLUID
        for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {
            local.CosmicRayPressure[k] = Get_Gas_CosmicRayPressure(i, k, kc);
        }
#endif
#ifdef CONDUCTION
        local.Kappa_Conduction = kc[i].Kappa_Conduction;
#endif
#ifdef VISCOSITY
        local.Eta_ShearViscosity = kc[i].Eta_ShearViscosity;
        local.Zeta_BulkViscosity = kc[i].Zeta_BulkViscosity;
#endif
#ifdef TURB_DIFFUSION
        local.TD_DiffCoeff = kc[i].TD_DiffCoeff;
#endif
#if defined(KERNEL_CRK_FACES)
        for(int k = 0; k < 16; k++) { local.Tensor_CRK_Face_Corrections[k] = kc[i].Tensor_CRK_Face_Corrections[k]; }
#endif
#ifdef HYDRO_SPH
        local.DrkernHydroSumFactor = kc[i].DrkernHydroSumFactor;
#if defined(SPHAV_CD10_VISCOSITY_SWITCH)
        local.alpha = kc[i].alpha_limiter * kc[i].alpha;
#else
        local.alpha = kc[i].alpha_limiter;
#endif
#endif
#ifdef HYDRO_PRESSURE_SPH
        local.EgyWtRho = kc[i].EgyWtDensity;
#endif
#ifdef MHD_NON_IDEAL
        local.Eta_MHD_OhmicResistivity_Coeff  = kc[i].Eta_MHD_OhmicResistivity_Coeff;
        local.Eta_MHD_HallEffect_Coeff        = kc[i].Eta_MHD_HallEffect_Coeff;
        local.Eta_MHD_AmbiPolarDiffusion_Coeff = kc[i].Eta_MHD_AmbiPolarDiffusion_Coeff;
#endif
#if defined(SPH_TP12_ARTIFICIAL_RESISTIVITY)
        local.Balpha = kc[i].Balpha;
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
        for(int k = 0; k < NUM_METAL_SPECIES; k++) { local.Gradients.Metallicity[k] = kc[i].Gradients.Metallicity[k]; }
#endif
        /* Host-precomputed per-active passive scalars (legacy
         * density_gpu.cc:299-301 pattern). Indexed by active_slot so the same
         * load works on the origin rank for Mode A and for Mode B local +
         * remote (the origin rank populates ActiveData.local before remote
         * shipping, so the peer never needs the UVM array). The outer
         * compound gate matches hydro_data_in::Metallicity field gating
         * (hydro_structs.h:129) — legacy had this under bare
         * GALSF_ISMDUSTCHEM_MODEL, which was broken for METALS+MFM+ISMDC
         * without TURB_DIFF_METALS; fixed in this commit (also fixed at
         * legacy density_gpu.cc:299-301). */
#if defined(GALSF_ISMDUSTCHEM_MODEL) && (defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME)))
        if(ctx.ismdc_uvm) {
            const int n_ismdc = NUM_ISMDUSTCHEM_PASSIVE_SCALARS;
            for(int ki = 0; ki < n_ismdc; ki++) {
                local.Metallicity[ISMDUSTCHEM_SPECIES_OFFSET_IN_METALLICITY + ki] =
                    ctx.ismdc_uvm[active_slot * n_ismdc + ki];
            }
        }
#endif
#ifdef CHIMES_TURB_DIFF_IONS
        for(int k = 0; k < ChimesGlobalVars.totalNumberOfSpecies; k++) { local.ChimesNIons[k] = kc[i].ChimesNIons[k]; }
#endif
#if defined(RT_SOLVER_EXPLICIT) && defined(RT_COMPGRAD_EDDINGTON_TENSOR) && (N_RT_FREQ_BINS > 0)
        for(int k = 0; k < N_RT_FREQ_BINS; k++) { local.Gradients.Rad_E_gamma_ET[k] = kc[i].Gradients.Rad_E_gamma_ET[k]; }
#endif
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
        for(int k = 0; k < N_RT_FREQ_BINS; k++) {
            for(int k2 = 0; k2 < 3; k2++) { local.Gradients.Rad_E_gamma_Grad[k][k2] = kc[i].Gradients.Rad_E_gamma_Grad[k][k2]; }
            for(int k2 = 0; k2 < 3; k2++) { for(int k3 = 0; k3 < 3; k3++) { local.Gradients.Rad_Flux_Grad[k][k2][k3] = kc[i].Gradients.Rad_Flux_Grad[k][k2][k3]; } }
        }
#endif
#ifdef RT_SOLVER_EXPLICIT
        for(int k = 0; k < N_RT_FREQ_BINS; k++) {
            local.Rad_E_gamma[k]      = kc[i].Rad_E_gamma_Pred[k];
            local.Rad_Kappa[k]        = kc[i].Rad_Kappa[k];
            local.RT_DiffusionCoeff[k] = rt_diffusion_coefficient(i, k, kc);
#if defined(RT_EVOLVE_FLUX) || defined(HYDRO_SPH)
            local.ET[k] = kc[i].ET[k];
#endif
#ifdef RT_EVOLVE_FLUX
            local.Rad_Flux[k] = kc[i].Rad_Flux_Pred[k];
#endif
#if defined(RT_EVOLVE_INTENSITIES)
            for(int k2 = 0; k2 < N_RT_INTENSITY_BINS; k2++) { local.Rad_Intensity_Pred[k][k2] = kc[i].Rad_Intensity_Pred[k][k2]; }
#endif
        }
#ifdef RT_INFRARED
        local.Radiation_Temperature = kc[i].Radiation_Temperature;
#endif
#endif
#ifdef COSMIC_RAY_FLUID
        for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {
            local.CosmicRayDiffusionCoeff[k] = kc[i].CosmicRayDiffusionCoeff[k];
            local.CosmicRayFlux[k]           = kc[i].CosmicRayFluxPred[k];
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
            for(int k2 = 0; k2 < 2; k2++) { local.CosmicRayAlfvenEnergy[k][k2] = kc[i].CosmicRayAlfvenEnergyPred[k][k2]; }
#endif
#ifdef CRFLUID_EVOLVE_SPECTRUM
            local.CR_number_to_energy_ratio[k] = kc[i].CosmicRay_Number_in_Bin[k] / (kc[i].CosmicRayEnergy[k] + MIN_REAL_NUMBER);
            local.CR_number_to_energy_ratio[k] *= kc[i].Flux_Number_to_Energy_Correction_Factor[k];
#endif
        }
#endif
#if defined(EOS_ELASTIC) || defined(EOS_ANEOS)
        local.CompositionType = kc[i].CompositionType;
#endif
#ifdef EOS_ELASTIC
        for(int k = 0; k < 3; k++) { for(int k2 = 0; k2 < 3; k2++) { local.Elastic_Stress_Tensor[k][k2] = kc[i].Elastic_Stress_Tensor_Pred[k][k2]; } }
#endif
#if defined(EOS_DAMAGE_POROSITY) && ((EOS_DAMAGE_POROSITY) & 1)
        local.Damage = kc[i].Damage;
#endif
#ifdef GALSF_SUBGRID_WINDS
        local.DelayTime = kc[i].DelayTime;
#endif
        return active;
    }

    KOKKOS_INLINE_FUNCTION
    static NeighborData load_neighbor(const DeviceContext& ctx,
                                       int j,
                                       const IdentitySidecar& /*id*/,
                                       const ActiveData& /*active*/)
    {
        /* All ctx-derived pointer / gate fields populated here from the
         * EVALUATING rank's ctx. In Mode B remote, evaluation runs on the
         * peer with peer_ctx; in Mode A and Mode B local it runs on origin
         * with origin_ctx. Either way the dereference target is on the same
         * rank as the dereferencing kernel, so the segfault-on-shipped-
         * origin-pointer (run 5 crash) cannot recur. */
        NeighborData nb;
        nb.j                  = j;
        nb.P                  = ctx.P;
        nb.CellP              = ctx.CellP;
        nb.TimeBinActive_ptr  = ctx.TimeBinActive_uvm;
        nb.oracle_dry_run     = ctx.oracle_dry_run;
        nb.need_wakeup_ptr    = ctx.oracle_dry_run ? nullptr : ctx.need_wakeup_uvm;
        return nb;
    }

    /* The physics — forwards to the unchanged inline pair body in
     * hydro_functions.h. Pre-pair seed: MaxSignalVel >= sound_i — enforced
     * by taking max each pair (legacy seeds once before the for-loop;
     * since runner zero_accum'd accum before the first pair, this idempotent
     * fmax produces the same result with negligible overhead). MAGNETIC
     * b2_i / alfven2_i kernel setup mirrors density_gpu.cc:567-573. */
    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& active,
                             const NeighborData& neighbor,
                             AccumData& accum,
                             NoScatter& /*scatter*/)
    {
        struct kernel_hydra kernel;
        for(size_t b = 0; b < sizeof(kernel); b++) ((char*)&kernel)[b] = 0;
        kernel.h_i           = active.local.KernelRadius;
        kernel.sound_i       = active.local.SoundSpeed;
        kernel.spec_egy_u_i  = active.local.InternalEnergyPred;
#ifdef MAGNETIC
        {
            double fac_magnetic_pressure_loc = 1.0 / All.cf_atime;
            kernel.b2_i      = active.local.BPred.norm_sq();
            kernel.alfven2_i = kernel.b2_i * fac_magnetic_pressure_loc / active.local.Density;
            kernel.alfven2_i = DMIN(kernel.alfven2_i, 1000. * kernel.sound_i * kernel.sound_i);
        }
#endif

        if(accum.MaxSignalVel < kernel.sound_i) { accum.MaxSignalVel = kernel.sound_i; }

        struct Conserved_var_Riemann Fluxes;
        for(size_t b = 0; b < sizeof(Fluxes); b++) ((char*)&Fluxes)[b] = 0;

        hydro_accumulate_neighbor(
            const_cast<struct hydro_data_in&>(active.local),
            accum, kernel, Fluxes, neighbor.j,
            active.local.dt_hydrostep_i,
            neighbor.P, neighbor.CellP,
            neighbor.TimeBinActive_ptr,
            neighbor.need_wakeup_ptr,
            /*allow_j_writes=*/!neighbor.oracle_dry_run);
    }

    /* Host writebacks (post-dispatch). Body in hydro_force_loop.cc.
     * Replays the per-active scatter from legacy out2particle_hydra
     * (hydro/hydro_toplevel.cc:115-295 in pre-port code). Enforces
     * MaxSignalVel >= effective_soundspeed() so zero-neighbor actives
     * don't end with MaxSignalVel=0 (CFL crash). */
    static void apply_active_writeback(const neighbor_loop_args& args,
                                        int active_slot, int i,
                                        const AccumData& accum);

    /* Peer-rank accum merge (Mode B remote). Body in hydro_force_loop.cc:
     * additive for most fields, MAX for MaxSignalVel + MaxShockMachNumber +
     * MaxKineticEnergyNgb. */
    static void merge_accum(AccumData& local_accum, const AccumData& peer_accum);

    /* Oracle gate. Body in hydro_force_loop.cc, same shape as gradients
     * commit 7 — field-wise per-#ifdef coverage + 1e-12 abs-tol floor for
     * sum-to-zero cancellation noise. */
    static double compare_accum(const AccumData& local, const AccumData& oracle);
};

#endif /* HYDRO_FORCE_LOOP_H */
