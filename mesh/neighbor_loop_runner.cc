/* mesh/neighbor_loop_runner.cc — generic NeighborLoopSpec runner.
 *
 * 3c.1 scope: WritePattern::ActiveReduceOnly, RemoteEvalMode::RemoteComputesAccum,
 * non-iterative, NoIdentity, NoScatter — sufficient for SinkEnv1Spec migration.
 *
 * Mode A (GPU NGL pipeline): IMPLEMENTED. Stages caller-supplied radii and
 * per-call CallScalars host-side pre-arena, runs gpu_particles_arena_acquire
 * + gpu_ngb_list_build, then a tiny Kokkos parallel_for that calls
 * Spec::load_active to fill ActiveData[] in UVM (same device epoch as the
 * legacy lambda's q-packing), then the parametric pair-kernel parallel_for
 * calling Spec::load_neighbor + Spec::pair_kernel. Both launches go through
 * gizmo_gpu_kernel_launch (project's parallel_for + fence + check_last_error).
 *
 * Mode B / Brute paths: NOT YET IMPLEMENTED — abort with endrun if dispatch
 * lands on them. Wired in 3c.2 (Mode B local) and 3c.3 (Mode B remote /
 * Brute oracle).
 *
 * Architecture binding contract:
 *   ~/.claude/memory/reference_neighbor_loop_contract.md
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO.
 */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <type_traits>
#include <Kokkos_Core.hpp>

#include "../declarations/allvars.h"
#include "../declarations/gpu_all_mirror.h"          /* GIZMO_GPU_ENSURE_ALL_FRESH, GPU_ALL_SYNC_FUNC */
#include "../core/proto.h"
#include "../system/gpu_particles_arena.h"
#include "../declarations/gpu_dispatch_templates.h"  /* gizmo_gpu_kernel_launch */

#include "neighbor_loop_runner.h"
#include "gpu_neighbor_list.h"
#include "kernel.h"  /* MUST precede sink_env1_spec.h (kernel_main, NEAREST_XYZ) */

/* Spec instantiations supported in 3c.1. Each #include declares one Spec
 * type whose explicit template instantiation appears at the bottom of
 * this file. */
#include "../sinks/sink_env1_spec.h"

/* ============================================================================
 * Env-gate stub functions (3c.1: dispatch always selects Mode A; these
 * stubs always return false / sane defaults. Real wiring lands in 3c.2.)
 * ========================================================================== */

int gizmo_nlr_default_modeb_threshold_sum(void) { return 0; }
int gizmo_nlr_default_modeb_threshold_max(void) { return 0; }

bool gizmo_nlr_force_mode_b_global(void) {
    /* 3c.1 stub: always false. Real impl reads GIZMO_NLR_FORCE_MODEB. */
    return false;
}
bool gizmo_nlr_force_mode_a_global(void) {
    /* 3c.1 stub: always false. Real impl reads GIZMO_NLR_FORCE_MODEA. */
    return false;
}
bool gizmo_nlr_oracle_enabled_global(void) {
    /* 3c.1 stub: always false. Real impl reads GIZMO_NLR_ORACLE in 3c.2. */
    return false;
}
bool gizmo_nlr_oracle_enabled_for(const char * /*loop_name*/) {
    /* 3c.1 stub: always false. Real impl reads GIZMO_<LOOP>_ORACLE in 3c.2. */
    return false;
}

/* ============================================================================
 * SIDX cache kind resolver (private to runner; see neighbor_loop_runner.h
 * SidxCacheKind enum doc).
 * ========================================================================== */

static gpu_spatial_index_t* nlr_resolve_sidx_cache(SidxCacheKind k,
                                                    const char *loop_name)
{
    switch(k) {
        case SidxCacheKind::AllTypes:
            return gpu_step_sidx_alltypes_ptr();
    }
    /* Unreachable in 3c.1 — kept exhaustive for compiler warnings on enum
     * additions. Other kinds land alongside their first caller. */
    fprintf(stderr, "neighbor_loop_runner: SidxCacheKind=%d not implemented "
            "for loop '%s' (3c.1 supports only AllTypes)\n",
            (int)k, loop_name ? loop_name : "?");
    fflush(stderr);
    endrun(81030);
    return nullptr;
}

/* ============================================================================
 * run_mode_a<Spec> — generic Mode A path through the GPU NGL pipeline.
 *
 * Three-epoch staging contract (see neighbor_loop_runner.h doc):
 *   (1) host pre-arena: Spec::search_radius_host  → radii_uvm[num_active]
 *   (2) host pre-arena: Spec::populate_call_scalars_host → CallScalars cs
 *   (3) device post-NGL-build: Spec::load_active → d_actives[num_active]
 *
 * Both Kokkos launches go through gizmo_gpu_kernel_launch which wraps
 * parallel_for + fence + check_last_error (declarations/gpu_dispatch_templates.h).
 * No additional explicit fence is needed before host-side d_accums readback
 * — the launch helper already fenced.
 * ========================================================================== */

template <typename Spec>
static void run_mode_a(const neighbor_loop_args& args)
{
    using ActiveData   = typename Spec::ActiveData;
    using AccumData    = typename Spec::AccumData;
    using ScatterData  = typename Spec::ScatterData;
    using NeighborData = typename Spec::NeighborData;
    using CallScalars  = typename Spec::CallScalars;
    using DeviceCtx    = NeighborLoopDeviceContextBase;

    const int N = args.num_active;
    if(N <= 0) {
        /* Defensive: caller already early-outs in collective dispatch when
         * global_num_active == 0, but local-zero with global-positive is
         * legitimate (peer rank with no actives still must enter the
         * collective in Mode B; Mode A has no collective work, just no-op). */
        return;
    }

    /* (1) Host, pre-arena: stage caller-supplied radii from external state. */
    double *radii_uvm = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
        N * sizeof(double));
    for(int aa = 0; aa < N; aa++) {
        radii_uvm[aa] = Spec::search_radius_host(args, aa, args.active_list[aa]);
    }

    /* (2) Host, pre-arena: capture per-call scalar globals into a POD. */
    CallScalars cs = Spec::populate_call_scalars_host(args);

    /* Arena + freshness (matches sinks/sink_environment_gpu.cc:76,86). The
     * caller is responsible for the args.CellP=NULL-when-no-gas decision;
     * runner does not read All.TotN_gas. */
    GIZMO_GPU_ENSURE_ALL_FRESH(neighbor_loop_runner);
    gpu_particles_arena_set_site(Spec::loop_name);
    gpu_particles_arena_acquire(args.num_total, args.P, args.CellP);
    struct particle_data *P_gpu    = gpu_particles_arena_P();
    struct gas_cell_data *CellP_gpu = (args.CellP != nullptr)
                                        ? gpu_particles_arena_CellP() : nullptr;

    /* NGL build using pre-arena radii. SIDX cache resolved from spec. */
    gpu_neighbor_list_t gnl;
    gpu_spatial_index_t *sidx = nlr_resolve_sidx_cache(Spec::sidx_cache_kind,
                                                       Spec::loop_name);
    gpu_ngb_list_build(P_gpu, args.num_total,
                       args.active_list, N,
                       Spec::search_mode,
                       (int)Spec::neighbor_type_mask,
                       &gnl, sidx,
                       1.0, radii_uvm, NULL, Spec::loop_name);

    /* UVM-allocate ActiveData[] and AccumData[] arrays. */
    ActiveData *d_actives = (ActiveData *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(N * sizeof(ActiveData));
    AccumData *d_accums = (AccumData *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(N * sizeof(AccumData));

    /* Build DeviceContext (base only in 3c.1; future specs may extend). */
    DeviceCtx ctx;
    ctx.P         = P_gpu;
    ctx.CellP     = CellP_gpu;
    ctx.num_total = args.num_total;

    /* (3) Device, post-NGL-build: stage ActiveData. Same device epoch as
     * the legacy GPU lambda's q-packing. d_active is gnl-resident. */
    int *d_active_idx = gnl.d_active;
    {
        gizmo_gpu_kernel_launch("nlr_stage_active", N, KOKKOS_LAMBDA(int aa) {
            d_actives[aa] = Spec::load_active(ctx, aa, d_active_idx[aa],
                                              radii_uvm[aa], cs);
        });
    }

    /* Pair-kernel launch — generic over Spec. */
    {
        int *offsets   = gnl.offsets;
        int *neighbors = gnl.neighbors;
        gizmo_gpu_kernel_launch(Spec::loop_name, N, KOKKOS_LAMBDA(int aa) {
            Spec::zero_accum(d_accums[aa]);
            const ActiveData& a = d_actives[aa];
            ScatterData s{};                     /* NoScatter for ActiveReduceOnly */
            int start = offsets[aa], end = offsets[aa + 1];
            for(int nn = start; nn < end; nn++) {
                int j = neighbors[nn];
                IdentitySidecar id{};            /* NoIdentity */
                NeighborData nb = Spec::load_neighbor(ctx, j, id, a);
                Spec::pair_kernel(a, nb, d_accums[aa], s);
            }
        });
    }
    /* Both launches fenced internally by gizmo_gpu_kernel_launch. UVM is
     * coherent → host can read d_accums directly. */

    /* (4) Host writeback via spec. */
    for(int aa = 0; aa < N; aa++) {
        Spec::apply_active_writeback(args, aa, args.active_list[aa], d_accums[aa]);
    }

    /* Cleanup. SIDX cache pointer passed so the free leaves cached storage
     * intact for sink_feed/sink_swk reuse (matches existing
     * sink_environment_gpu.cc:261 idiom). */
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_accums);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_actives);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(radii_uvm);
    gpu_ngb_list_free(&gnl, sidx);
    gpu_particles_arena_mark_clean_after_scatter(Spec::loop_name);
}

/* ============================================================================
 * Public entry: run_neighbor_loop<Spec>
 * ========================================================================== */

template <typename Spec>
void run_neighbor_loop(const neighbor_loop_args& args)
{
    /* ---- Compile-time spec consistency ---- */

    /* WritePattern ↔ AccumData/ScatterData consistency. */
    static_assert(Spec::write_pattern != WritePattern::ActiveReduceOnly ||
                  std::is_same<typename Spec::ScatterData, NoScatter>::value,
        "ActiveReduceOnly requires ScatterData == NoScatter");
    static_assert(Spec::write_pattern != WritePattern::NeighborScatter ||
                  std::is_same<typename Spec::AccumData, NoAccum>::value,
        "NeighborScatter requires AccumData == NoAccum");

    /* POD/device-copy contract — captured by value into Kokkos lambdas
     * and/or staged into UVM arrays. Trivially-copyable is the binding
     * requirement. */
    static_assert(std::is_trivially_copyable<typename Spec::CallScalars>::value,
        "Spec::CallScalars must be trivially-copyable (lambda capture by value)");
    static_assert(std::is_trivially_copyable<typename Spec::ActiveData>::value,
        "Spec::ActiveData must be trivially-copyable (UVM-staged)");
    static_assert(std::is_trivially_copyable<typename Spec::NeighborData>::value,
        "Spec::NeighborData must be trivially-copyable (built per-pair on device)");
    static_assert(std::is_trivially_copyable<typename Spec::AccumData>::value,
        "Spec::AccumData must be trivially-copyable (UVM-staged)");

    /* ---- Dispatch ---- */

    /* 3c.1: dispatch always selects Mode A. Mode B / Brute paths abort
     * loudly (force-mode unsupported feature → abort per design doc).
     * 3c.2 wires the real dispatch (Allreduce sum/max thresholds + force
     * envs + oracle compare). */
    if(gizmo_nlr_force_mode_b_global()) {
        fprintf(stderr, "neighbor_loop_runner: GIZMO_NLR_FORCE_MODEB=1 set "
                "but Mode B is not yet implemented for loop '%s' in 3c.1 "
                "(lands in 3c.2). Run without the force flag.\n",
                Spec::loop_name);
        fflush(stderr);
        endrun(81031);
    }
    if(gizmo_nlr_oracle_enabled_global() ||
       gizmo_nlr_oracle_enabled_for(Spec::loop_name)) {
        fprintf(stderr, "neighbor_loop_runner: GIZMO_NLR_*_ORACLE=1 set but "
                "Brute oracle is not yet implemented for loop '%s' in 3c.1 "
                "(lands in 3c.2). Run without the oracle flag (the legacy "
                "GIZMO_MODE_B_SINK_ENV1 SPIKE path's oracle is still active).\n",
                Spec::loop_name);
        fflush(stderr);
        endrun(81032);
    }

    run_mode_a<Spec>(args);
}

/* ============================================================================
 * Explicit template instantiations — one per migrated caller.
 *
 * Forgetting an instantiation = clean linker error at the call site. Each
 * Spec must opt in here and is implicitly acknowledging the
 * Spec::sidx_cache_kind = AllTypes (the only kind implemented in 3c.1)
 * via its constexpr declaration; cf. nlr_resolve_sidx_cache.
 * ========================================================================== */

template void run_neighbor_loop<SinkEnv1Spec>(const neighbor_loop_args&);

/* Per-TU GPU All-mirror sync function (paired with GIZMO_GPU_ENSURE_ALL_FRESH
 * in run_mode_a). Same idiom as sink_environment_gpu.cc:388. */
GPU_ALL_SYNC_FUNC(neighbor_loop_runner)
