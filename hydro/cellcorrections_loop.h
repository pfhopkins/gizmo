/* hydro/cellcorrections_loop.h — first-pass cell volume corrections.
 *
 * Defines CellcorrectionsSpec for the neighbor-loop runner. Computes the
 * second-order volume correction Volume_1[i] = sum_j Volume_0[j]^2 wk(r, h_j)
 * for active gas particles. Pure i-side accumulate: no j-side writes, no
 * ghost-writeback, no iteration. Same topology as gradients / hydro_force
 * (symmetric gas-gas, GasOnly cache, per-active P[i].KernelRadius) — this
 * Spec is the first WAVE 5 corridor consumer, lands ahead of GradientsSpec.
 *
 * Commit 5a scope: Spec parity with the legacy host walker in
 * hydro/density.cc:62-162; runner builds its own CSR (no external CSR
 * injection yet — commit 5b adds that + hoists corridor CSR begin).
 *
 * Written by Philip F. Hopkins (phopkins@caltech.edu) for GIZMO. */

#ifndef CELLCORRECTIONS_LOOP_H
#define CELLCORRECTIONS_LOOP_H

#include "../declarations/allvars.h"

#ifdef HYDRO_VOLUME_CORRECTIONS

#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/mode_b_local_walker.h"      /* MODE_B_SEARCH_*, MODE_B_RADIUS_* */
/* NOTE: caller translation units must include "../mesh/kernel.h" BEFORE
 * this header. kernel.h has no include guards (defines static inline
 * kernel_main / kernel_hinv used by the pair body below); double-include
 * triggers redefinition errors. The runner and cellcorrections_loop.cc
 * both include kernel.h first per this convention (sink_env1_loop.h
 * follows the same pattern). */

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

/* Forward-decl from hydro/gradients.cc — the GasGrad_isactive predicate
 * gates gradients-eligible cells; cellcorrections must match it
 * (legacy semantic: the cell must pass the same active-gas filter that
 * gradients applies, since Volume_1 is consumed by gradients). */
int GasGrad_isactive(int i, struct particle_data *pp, struct gas_cell_data *cell);

/* ============================================================================
 * Per-pair physics types (file scope; PascalCase).
 * ========================================================================== */

struct CellcorrectionsActiveData {
    Vec3<double> pos;       /* P[i].Pos — only needed to compute dp = i - j */
    double       h_search;  /* per-active radius (P[i].KernelRadius) — required
                             * by runner's Mode B remote walker contract
                             * (mesh/neighbor_loop_runner.cc:927) */
};

struct CellcorrectionsAccum {
    double volume_1;    /* sum_j Volume_0[j]^2 wk(r, h_j) */
};

struct CellcorrectionsNeighborData {
    Vec3<double> pos;           /* P[j].Pos */
    double       kernel_radius; /* P[j].KernelRadius — pair kernel uses j's radius */
    double       volume_0;      /* CellP[j].Volume_0 — the contribution weight */
};

/* ============================================================================
 * Inline pair body — single source of truth for cellcorrections physics.
 * Pure i-side accumulate using j's kernel radius for the contribution
 * filter and weight (matches legacy hydro/density.cc:130-149). The early
 * return on r2 >= h_j*h_j is the legacy filter; we still get called for
 * such pairs because MODE_B_SEARCH_SYMMETRIC reaches max(h_i, h_j).
 * ========================================================================== */
KOKKOS_INLINE_FUNCTION
static void cellcorrections_pair_kernel(const CellcorrectionsActiveData &active,
                                        const CellcorrectionsNeighborData &nb,
                                        CellcorrectionsAccum &out)
{
    Vec3<double> dp;
    dp[0] = active.pos[0] - nb.pos[0];
    dp[1] = active.pos[1] - nb.pos[1];
    dp[2] = active.pos[2] - nb.pos[2];
    nearest_xyz(dp);
    double r2 = dp.norm_sq();
    double h_j = nb.kernel_radius;
    if(r2 >= h_j * h_j) return;
    double hinv, hinv3, hinv4, wk = 0, dwk = 0;
    kernel_hinv(h_j, &hinv, &hinv3, &hinv4);
    double u = sqrt(r2) * hinv;
    kernel_main(u, hinv3, hinv4, &wk, &dwk, -1);
    out.volume_1 += nb.volume_0 * nb.volume_0 * wk;
}

/* ============================================================================
 * CellcorrectionsSpec — the NeighborLoopSpec contract.
 * ========================================================================== */

struct CellcorrectionsSpec {
    /* Identity */
    static constexpr const char *loop_name = "cellcorrections";

    /* Search policy. Symmetric so the pair contributes whenever r < h_j
     * (j's kernel) even if r > h_i; gas-only neighbor_type_mask matches
     * legacy gpu_ngb_list_build's gas-only j-type filter. */
    static constexpr int                     search_mode        = MODE_B_SEARCH_SYMMETRIC;
    static constexpr unsigned int            neighbor_type_mask = (1u << 0);
    static constexpr mode_b_radius_policy_t  radius_policy      = MODE_B_RADIUS_DEFAULT;

    /* Writeback policy. Pure i-side accumulate; no ghost-writeback. */
    static constexpr WritePattern   write_pattern    = WritePattern::ActiveReduceOnly;
    static constexpr SidxCacheKind  sidx_cache_kind  = SidxCacheKind::GasOnly;
    static constexpr double         accum_tolerance  = 1e-10;
    static constexpr bool           uses_ghost_writeback       = false;
    static constexpr bool           uses_ghost_write_detector  = false;

    /* Active-particle predicate. Matches the legacy filter
     * hydro/density.cc:81-83 (gas + Mass>0 + GasGrad_isactive +
     * KernelRadius>0). The toplevel passes this to nlr_build_active_list. */
    static bool is_active(int i) {
        if(P[i].Type != 0)              return false;
        if(P[i].Mass <= 0)              return false;
        if(P[i].KernelRadius <= 0)      return false;
        if(!GasGrad_isactive(i, P, CellP)) return false;
        return true;
    }

    /* Per-pair physics types */
    using CallScalars   = NlrCommonScalars;   /* nothing loop-specific needed */
    using ActiveData    = CellcorrectionsActiveData;
    using AccumData     = CellcorrectionsAccum;
    using NeighborData  = CellcorrectionsNeighborData;

    using ScatterData    = NoScatter;
    using IdentityFields = NoIdentity;
    using IterControl    = NotIterative;
    using DeviceContext  = NeighborLoopDeviceContextBase;

    /* Per-active host hook: pre-arena search radius from P[i].KernelRadius. */
    static double search_radius(const neighbor_loop_args& /*args*/,
                                 int /*active_slot*/, int i)
    {
        return (double)P[i].KernelRadius;
    }

    /* Per-call scalars: only the common cosmology block. The pair body
     * doesn't actually read these (geometry only), but the runner contract
     * still requires populate_call_scalars to be present. */
    static CallScalars populate_call_scalars(const neighbor_loop_args& /*args*/)
    {
        return nlr_common_scalars_from_all();
    }

    /* Device hooks (KOKKOS_INLINE_FUNCTION — host+device callable). */

    KOKKOS_INLINE_FUNCTION
    static void zero_accum(AccumData& accum)
    {
        accum.volume_1 = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    static ActiveData load_active(const NeighborLoopDeviceContextBase& ctx,
                                   int /*active_slot*/, int i,
                                   double h_search,
                                   const CallScalars& /*scalars*/)
    {
        ActiveData active;
        active.pos      = ctx.P[i].Pos;
        active.h_search = h_search;
        return active;
    }

    KOKKOS_INLINE_FUNCTION
    static NeighborData load_neighbor(const NeighborLoopDeviceContextBase& ctx,
                                       int j,
                                       const IdentitySidecar& /*id*/,
                                       const ActiveData& /*active*/)
    {
        NeighborData nb;
        nb.pos           = ctx.P[j].Pos;
        nb.kernel_radius = (double)ctx.P[j].KernelRadius;
        nb.volume_0      = (ctx.CellP && ctx.P[j].Type == 0)
                              ? (double)ctx.CellP[j].Volume_0
                              : 0.0;
        return nb;
    }

    /* The physics — forwards to the inline pair body above. */
    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& active,
                             const NeighborData& neighbor,
                             AccumData& accum,
                             NoScatter& /*scatter*/)
    {
        cellcorrections_pair_kernel(active, neighbor, accum);
    }

    /* Host writebacks (post-dispatch) — body in cellcorrections_loop.cc */
    static void apply_active_writeback(const neighbor_loop_args& args,
                                        int active_slot, int i,
                                        const AccumData& accum);

    /* Per-field merge of a peer-rank accum into local accum (Mode B remote). */
    static void merge_accum(AccumData& local, const AccumData& peer)
    {
        local.volume_1 += peer.volume_1;
    }

    /* Oracle comparison (env-gated by runner). Body in .cc. */
    static double compare_accum(const AccumData& local, const AccumData& oracle);
};

#endif /* HYDRO_VOLUME_CORRECTIONS */
#endif /* CELLCORRECTIONS_LOOP_H */
