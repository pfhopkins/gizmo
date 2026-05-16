/* mesh/ghost_writeback_ops.h — physics-facing manifest macros for the
 * ghost-writeback scaffold (mesh/ghost_writeback.{h,cc}, Pass B.iv).
 *
 * The science-author surface is the manifest itself:
 *
 *   GHOST_WRITEBACK_BUNDLE_BEGIN(my_loop)
 *   #ifdef SOME_PHYSICS_FLAG
 *       GHOST_WRITEBACK_PARTICLE_MIN(MyField)
 *   #endif
 *   GHOST_WRITEBACK_BUNDLE_END(my_loop)
 *
 * One bundle block per loop, one line per field-op. The macros generate
 * the callback context, snapshot/changed/pack/apply/cleanup functions,
 * and a static descriptor — none of which the loop author writes by hand.
 *
 * A `GHOST_WRITEBACK_BUNDLE_END(my_loop)` block defines the accessor
 * `my_loop_ghost_writeback_bundle_ptr()` returning the bundle. Spec hooks
 * call ghost_writeback_begin_bundle / _end_bundle with this pointer.
 *
 * Operations available:
 *   GHOST_WRITEBACK_PARTICLE_MIN(field)         home P[j].field = min(home, ghost)        [B.iv]
 *   GHOST_WRITEBACK_PARTICLE_MAX(field)         home P[j].field = max(home, ghost)        [3d.1]
 *   GHOST_WRITEBACK_GAS_ADD(field)              snapshot-diff additive on CellP[j].field  [3d.1]
 *   GHOST_WRITEBACK_PARTICLE_ADD(field)         snapshot-diff additive on P[j].field      [3e.2]
 *   GHOST_WRITEBACK_PARTICLE_ADD_ARRAY(f, N)    snapshot-diff additive on P[j].field[N]   [3e.2]
 *   GHOST_WRITEBACK_PARTICLE_ADD_VEC3(field)    snapshot-diff additive on P[j].field (Vec3<T>) [3e.2]
 *   GHOST_WRITEBACK_GAS_MAX(field)              home CellP[j].field = max(home, ghost)    [3e.2]
 *
 * Semantics summary:
 *   ADD ops: pack(delta = post - snap), apply(home += delta).
 *   MAX ops: pack(value = post) only when (post > snap); apply uses strict
 *            (value > home) — NaN-safe (NaN never overwrites finite home,
 *            finite never overwrites a sticky-NaN home).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO.
 */
#ifndef GHOST_WRITEBACK_OPS_H
#define GHOST_WRITEBACK_OPS_H

#include <cstdlib>
#include <type_traits>

#include "../declarations/allvars.h"
#include "ghost_writeback.h"

/* ============================================================================
 * gw_detail — engine-internal namespace; the manifest macros below dispatch
 * here. Physics authors do not reference gw_detail directly.
 * ========================================================================== */
namespace gw_detail {

/* ParticleMinOp<FieldT, MemPtr> — generic min-reduce reverse-comm for any
 * scalar field on particle_data.
 *
 * Each unique (FieldT, MemPtr) pair instantiates one anonymous descriptor;
 * the manifest macro adds a pointer to this descriptor's callback to the
 * bundle's array. A single Ctx instance per (FieldT, MemPtr) holds the
 * snapshot allocation across one begin/end pair. */
template <typename FieldT, FieldT particle_data::*MemPtr>
struct ParticleMinOp {
    struct Ctx {
        FieldT *snap;
        int     num_ghosts;
    };
    static Ctx s_ctx;

    /* Delta record — encodes home-rank target index and the new (post-
     * kernel) field value. */
    struct Delta {
        int    home_index;
        FieldT value;
    };

    static void snapshot_fn(void *vctx, int num_ghosts, int num_local) {
        Ctx *c = static_cast<Ctx*>(vctx);
        c->num_ghosts = num_ghosts;
        if (num_ghosts <= 0) { c->snap = nullptr; return; }
        c->snap = (FieldT*) malloc(num_ghosts * sizeof(FieldT));
        for (int g = 0; g < num_ghosts; g++) {
            c->snap[g] = P[num_local + g].*MemPtr;
        }
    }

    static int delta_for_ghost_fn(void *vctx, int g, int num_local) {
        Ctx *c = static_cast<Ctx*>(vctx);
        if (!c->snap) return 0;
        return (P[num_local + g].*MemPtr < c->snap[g]) ? 1 : 0;
    }

    static void pack_fn(void *vctx, int g, int num_local, void *out_delta) {
        (void)vctx;
        Delta *d = static_cast<Delta*>(out_delta);
        int *home_index = ghost_get_home_index();
        d->home_index = home_index[g];
        d->value      = P[num_local + g].*MemPtr;
    }

    static void apply_fn(void *vctx, const void *in_delta) {
        (void)vctx;
        const Delta *d = static_cast<const Delta*>(in_delta);
        if (d->value < P[d->home_index].*MemPtr) {
            P[d->home_index].*MemPtr = d->value;
        }
    }

    static void cleanup_fn(void *vctx) {
        Ctx *c = static_cast<Ctx*>(vctx);
        if (c->snap) { free(c->snap); c->snap = nullptr; }
        c->num_ghosts = 0;
    }

    static const ghost_writeback_callback callback;
};

/* Out-of-line static member definitions. */
template <typename FieldT, FieldT particle_data::*MemPtr>
typename ParticleMinOp<FieldT, MemPtr>::Ctx
ParticleMinOp<FieldT, MemPtr>::s_ctx{nullptr, 0};

template <typename FieldT, FieldT particle_data::*MemPtr>
const ghost_writeback_callback ParticleMinOp<FieldT, MemPtr>::callback = {
    sizeof(typename ParticleMinOp<FieldT, MemPtr>::Delta),
    & ParticleMinOp<FieldT, MemPtr>::snapshot_fn,
    & ParticleMinOp<FieldT, MemPtr>::delta_for_ghost_fn,
    & ParticleMinOp<FieldT, MemPtr>::pack_fn,
    & ParticleMinOp<FieldT, MemPtr>::apply_fn,
    & ParticleMinOp<FieldT, MemPtr>::cleanup_fn,
    & ParticleMinOp<FieldT, MemPtr>::s_ctx,
};

/* ParticleMaxOp — symmetric MAX variant of ParticleMinOp. Home keeps the
 * larger of (home, ghost) per delta record. Used by sink_feed for
 * P[j].SwallowID (D1/S-SYNC max-ID-wins tiebreak across ranks). */
template <typename FieldT, FieldT particle_data::*MemPtr>
struct ParticleMaxOp {
    static_assert(std::is_arithmetic<FieldT>::value || std::is_integral<FieldT>::value,
                  "GHOST_WRITEBACK_PARTICLE_MAX requires a comparable scalar field");

    struct Ctx {
        FieldT *snap;
        int     num_ghosts;
    };
    static Ctx s_ctx;

    struct Delta {
        int    home_index;
        FieldT value;
    };

    static void snapshot_fn(void *vctx, int num_ghosts, int num_local) {
        Ctx *c = static_cast<Ctx*>(vctx);
        c->num_ghosts = num_ghosts;
        if (num_ghosts <= 0) { c->snap = nullptr; return; }
        c->snap = (FieldT*) malloc(num_ghosts * sizeof(FieldT));
        for (int g = 0; g < num_ghosts; g++) {
            c->snap[g] = P[num_local + g].*MemPtr;
        }
    }

    static int delta_for_ghost_fn(void *vctx, int g, int num_local) {
        Ctx *c = static_cast<Ctx*>(vctx);
        if (!c->snap) return 0;
        return (P[num_local + g].*MemPtr > c->snap[g]) ? 1 : 0;
    }

    static void pack_fn(void *vctx, int g, int num_local, void *out_delta) {
        (void)vctx;
        Delta *d = static_cast<Delta*>(out_delta);
        int *home_index = ghost_get_home_index();
        d->home_index = home_index[g];
        d->value      = P[num_local + g].*MemPtr;
    }

    static void apply_fn(void *vctx, const void *in_delta) {
        (void)vctx;
        const Delta *d = static_cast<const Delta*>(in_delta);
        if (d->value > P[d->home_index].*MemPtr) {
            P[d->home_index].*MemPtr = d->value;
        }
    }

    static void cleanup_fn(void *vctx) {
        Ctx *c = static_cast<Ctx*>(vctx);
        if (c->snap) { free(c->snap); c->snap = nullptr; }
        c->num_ghosts = 0;
    }

    static const ghost_writeback_callback callback;
};

template <typename FieldT, FieldT particle_data::*MemPtr>
typename ParticleMaxOp<FieldT, MemPtr>::Ctx
ParticleMaxOp<FieldT, MemPtr>::s_ctx{nullptr, 0};

template <typename FieldT, FieldT particle_data::*MemPtr>
const ghost_writeback_callback ParticleMaxOp<FieldT, MemPtr>::callback = {
    sizeof(typename ParticleMaxOp<FieldT, MemPtr>::Delta),
    & ParticleMaxOp<FieldT, MemPtr>::snapshot_fn,
    & ParticleMaxOp<FieldT, MemPtr>::delta_for_ghost_fn,
    & ParticleMaxOp<FieldT, MemPtr>::pack_fn,
    & ParticleMaxOp<FieldT, MemPtr>::apply_fn,
    & ParticleMaxOp<FieldT, MemPtr>::cleanup_fn,
    & ParticleMaxOp<FieldT, MemPtr>::s_ctx,
};

/* GasAddOp — snapshot-diff additive reverse-comm for any arithmetic field
 * on gas_cell_data. Snapshot pre-kernel; pack post-kernel delta = post -
 * snapshot; home applies +=. Used by sink_feed for
 * CellP[j].Injected_Sink_Energy (additive across multi-rank contributors). */
template <typename FieldT, FieldT gas_cell_data::*MemPtr>
struct GasAddOp {
    static_assert(std::is_arithmetic<FieldT>::value,
                  "GHOST_WRITEBACK_GAS_ADD requires an arithmetic scalar field");

    struct Ctx {
        FieldT *snap;
        int     num_ghosts;
    };
    static Ctx s_ctx;

    struct Delta {
        int    home_index;
        FieldT delta;
    };

    static void snapshot_fn(void *vctx, int num_ghosts, int num_local) {
        Ctx *c = static_cast<Ctx*>(vctx);
        c->num_ghosts = num_ghosts;
        if (num_ghosts <= 0) { c->snap = nullptr; return; }
        c->snap = (FieldT*) malloc(num_ghosts * sizeof(FieldT));
        for (int g = 0; g < num_ghosts; g++) {
            c->snap[g] = CellP[num_local + g].*MemPtr;
        }
    }

    static int delta_for_ghost_fn(void *vctx, int g, int num_local) {
        Ctx *c = static_cast<Ctx*>(vctx);
        if (!c->snap) return 0;
        return (CellP[num_local + g].*MemPtr != c->snap[g]) ? 1 : 0;
    }

    static void pack_fn(void *vctx, int g, int num_local, void *out_delta) {
        Ctx *c = static_cast<Ctx*>(vctx);
        Delta *d = static_cast<Delta*>(out_delta);
        int *home_index = ghost_get_home_index();
        d->home_index = home_index[g];
        d->delta      = CellP[num_local + g].*MemPtr - c->snap[g];
    }

    static void apply_fn(void *vctx, const void *in_delta) {
        (void)vctx;
        const Delta *d = static_cast<const Delta*>(in_delta);
        CellP[d->home_index].*MemPtr += d->delta;
    }

    static void cleanup_fn(void *vctx) {
        Ctx *c = static_cast<Ctx*>(vctx);
        if (c->snap) { free(c->snap); c->snap = nullptr; }
        c->num_ghosts = 0;
    }

    static const ghost_writeback_callback callback;
};

template <typename FieldT, FieldT gas_cell_data::*MemPtr>
typename GasAddOp<FieldT, MemPtr>::Ctx
GasAddOp<FieldT, MemPtr>::s_ctx{nullptr, 0};

template <typename FieldT, FieldT gas_cell_data::*MemPtr>
const ghost_writeback_callback GasAddOp<FieldT, MemPtr>::callback = {
    sizeof(typename GasAddOp<FieldT, MemPtr>::Delta),
    & GasAddOp<FieldT, MemPtr>::snapshot_fn,
    & GasAddOp<FieldT, MemPtr>::delta_for_ghost_fn,
    & GasAddOp<FieldT, MemPtr>::pack_fn,
    & GasAddOp<FieldT, MemPtr>::apply_fn,
    & GasAddOp<FieldT, MemPtr>::cleanup_fn,
    & GasAddOp<FieldT, MemPtr>::s_ctx,
};

/* ParticleAddOp — snapshot-diff additive reverse-comm for any scalar field on
 * particle_data. Sibling of GasAddOp; differs only in P[] vs CellP[] array.
 * Used by thermalfb for P[j].Mass (per-rank post-kernel mass deltas summed
 * additively at home). */
template <typename FieldT, FieldT particle_data::*MemPtr>
struct ParticleAddOp {
    static_assert(std::is_arithmetic<FieldT>::value,
                  "GHOST_WRITEBACK_PARTICLE_ADD requires an arithmetic scalar field");

    struct Ctx {
        FieldT *snap;
        int     num_ghosts;
    };
    static Ctx s_ctx;

    struct Delta {
        int    home_index;
        FieldT delta;
    };

    static void snapshot_fn(void *vctx, int num_ghosts, int num_local) {
        Ctx *c = static_cast<Ctx*>(vctx);
        c->num_ghosts = num_ghosts;
        if (num_ghosts <= 0) { c->snap = nullptr; return; }
        c->snap = (FieldT*) malloc(num_ghosts * sizeof(FieldT));
        for (int g = 0; g < num_ghosts; g++) {
            c->snap[g] = P[num_local + g].*MemPtr;
        }
    }

    static int delta_for_ghost_fn(void *vctx, int g, int num_local) {
        Ctx *c = static_cast<Ctx*>(vctx);
        if (!c->snap) return 0;
        return (P[num_local + g].*MemPtr != c->snap[g]) ? 1 : 0;
    }

    static void pack_fn(void *vctx, int g, int num_local, void *out_delta) {
        Ctx *c = static_cast<Ctx*>(vctx);
        Delta *d = static_cast<Delta*>(out_delta);
        int *home_index = ghost_get_home_index();
        d->home_index = home_index[g];
        d->delta      = P[num_local + g].*MemPtr - c->snap[g];
    }

    static void apply_fn(void *vctx, const void *in_delta) {
        (void)vctx;
        const Delta *d = static_cast<const Delta*>(in_delta);
        P[d->home_index].*MemPtr += d->delta;
    }

    static void cleanup_fn(void *vctx) {
        Ctx *c = static_cast<Ctx*>(vctx);
        if (c->snap) { free(c->snap); c->snap = nullptr; }
        c->num_ghosts = 0;
    }

    static const ghost_writeback_callback callback;
};

template <typename FieldT, FieldT particle_data::*MemPtr>
typename ParticleAddOp<FieldT, MemPtr>::Ctx
ParticleAddOp<FieldT, MemPtr>::s_ctx{nullptr, 0};

template <typename FieldT, FieldT particle_data::*MemPtr>
const ghost_writeback_callback ParticleAddOp<FieldT, MemPtr>::callback = {
    sizeof(typename ParticleAddOp<FieldT, MemPtr>::Delta),
    & ParticleAddOp<FieldT, MemPtr>::snapshot_fn,
    & ParticleAddOp<FieldT, MemPtr>::delta_for_ghost_fn,
    & ParticleAddOp<FieldT, MemPtr>::pack_fn,
    & ParticleAddOp<FieldT, MemPtr>::apply_fn,
    & ParticleAddOp<FieldT, MemPtr>::cleanup_fn,
    & ParticleAddOp<FieldT, MemPtr>::s_ctx,
};

/* ParticleAddArrayOp<ElemT, N, ArrayMemPtr> — snapshot-diff additive reverse-
 * comm for a fixed-size array field on particle_data. Covers both 3-component
 * vectors (P[j].dp[3]) and METALS arrays (P[j].Metallicity[NUM_METAL_SPECIES])
 * via a single generic. Predicate fires when ANY element differs from snapshot.
 *
 * Template form: pass element type + N + pointer-to-member-array as separate
 * params. The pointer-to-member-array type spelled out is
 *   ElemT (particle_data::*ArrayMemPtr)[N]
 * which `decltype(particle_data::field)` resolves to for a declared `ElemT
 * field[N]`. The macro hides this. */
template <typename ElemT, int N, ElemT (particle_data::*ArrayMemPtr)[N]>
struct ParticleAddArrayOp {
    static_assert(std::is_arithmetic<ElemT>::value,
                  "GHOST_WRITEBACK_PARTICLE_ADD_ARRAY requires an arithmetic element type");
    static_assert(N > 0, "GHOST_WRITEBACK_PARTICLE_ADD_ARRAY requires N > 0");

    struct Snap { ElemT v[N]; };

    struct Ctx {
        Snap *snap;
        int   num_ghosts;
    };
    static Ctx s_ctx;

    struct Delta {
        int   home_index;
        ElemT delta[N];
    };

    static void snapshot_fn(void *vctx, int num_ghosts, int num_local) {
        Ctx *c = static_cast<Ctx*>(vctx);
        c->num_ghosts = num_ghosts;
        if (num_ghosts <= 0) { c->snap = nullptr; return; }
        c->snap = (Snap*) malloc(num_ghosts * sizeof(Snap));
        for (int g = 0; g < num_ghosts; g++) {
            for (int k = 0; k < N; k++) {
                c->snap[g].v[k] = (P[num_local + g].*ArrayMemPtr)[k];
            }
        }
    }

    static int delta_for_ghost_fn(void *vctx, int g, int num_local) {
        Ctx *c = static_cast<Ctx*>(vctx);
        if (!c->snap) return 0;
        for (int k = 0; k < N; k++) {
            if ((P[num_local + g].*ArrayMemPtr)[k] != c->snap[g].v[k]) return 1;
        }
        return 0;
    }

    static void pack_fn(void *vctx, int g, int num_local, void *out_delta) {
        Ctx *c = static_cast<Ctx*>(vctx);
        Delta *d = static_cast<Delta*>(out_delta);
        int *home_index = ghost_get_home_index();
        d->home_index = home_index[g];
        for (int k = 0; k < N; k++) {
            d->delta[k] = (P[num_local + g].*ArrayMemPtr)[k] - c->snap[g].v[k];
        }
    }

    static void apply_fn(void *vctx, const void *in_delta) {
        (void)vctx;
        const Delta *d = static_cast<const Delta*>(in_delta);
        for (int k = 0; k < N; k++) {
            (P[d->home_index].*ArrayMemPtr)[k] += d->delta[k];
        }
    }

    static void cleanup_fn(void *vctx) {
        Ctx *c = static_cast<Ctx*>(vctx);
        if (c->snap) { free(c->snap); c->snap = nullptr; }
        c->num_ghosts = 0;
    }

    static const ghost_writeback_callback callback;
};

template <typename ElemT, int N, ElemT (particle_data::*ArrayMemPtr)[N]>
typename ParticleAddArrayOp<ElemT, N, ArrayMemPtr>::Ctx
ParticleAddArrayOp<ElemT, N, ArrayMemPtr>::s_ctx{nullptr, 0};

template <typename ElemT, int N, ElemT (particle_data::*ArrayMemPtr)[N]>
const ghost_writeback_callback ParticleAddArrayOp<ElemT, N, ArrayMemPtr>::callback = {
    sizeof(typename ParticleAddArrayOp<ElemT, N, ArrayMemPtr>::Delta),
    & ParticleAddArrayOp<ElemT, N, ArrayMemPtr>::snapshot_fn,
    & ParticleAddArrayOp<ElemT, N, ArrayMemPtr>::delta_for_ghost_fn,
    & ParticleAddArrayOp<ElemT, N, ArrayMemPtr>::pack_fn,
    & ParticleAddArrayOp<ElemT, N, ArrayMemPtr>::apply_fn,
    & ParticleAddArrayOp<ElemT, N, ArrayMemPtr>::cleanup_fn,
    & ParticleAddArrayOp<ElemT, N, ArrayMemPtr>::s_ctx,
};

/* ParticleAddVec3Op<ElemT, MemPtr> — snapshot-diff additive reverse-comm for
 * a Vec3<ElemT> field on particle_data. Sibling of ParticleAddArrayOp; this
 * variant exists because Vec3<T> is a struct (with `T data[3]` inside), not a
 * raw T[3] array, so pointer-to-member-array deduction does not bind to it.
 * Apply uses Vec3<T>::operator+= componentwise. Used by thermalfb for
 * P[j].dp (per-rank momentum-injection deltas summed additively at home). */
template <typename ElemT, Vec3<ElemT> particle_data::*MemPtr>
struct ParticleAddVec3Op {
    static_assert(std::is_arithmetic<ElemT>::value,
                  "GHOST_WRITEBACK_PARTICLE_ADD_VEC3 requires an arithmetic element type");

    struct Ctx {
        Vec3<ElemT> *snap;
        int          num_ghosts;
    };
    static Ctx s_ctx;

    struct Delta {
        int         home_index;
        Vec3<ElemT> delta;
    };

    static void snapshot_fn(void *vctx, int num_ghosts, int num_local) {
        Ctx *c = static_cast<Ctx*>(vctx);
        c->num_ghosts = num_ghosts;
        if (num_ghosts <= 0) { c->snap = nullptr; return; }
        c->snap = (Vec3<ElemT>*) malloc(num_ghosts * sizeof(Vec3<ElemT>));
        for (int g = 0; g < num_ghosts; g++) {
            c->snap[g] = P[num_local + g].*MemPtr;
        }
    }

    static int delta_for_ghost_fn(void *vctx, int g, int num_local) {
        Ctx *c = static_cast<Ctx*>(vctx);
        if (!c->snap) return 0;
        const Vec3<ElemT>& now = P[num_local + g].*MemPtr;
        const Vec3<ElemT>& was = c->snap[g];
        return (now[0] != was[0] || now[1] != was[1] || now[2] != was[2]) ? 1 : 0;
    }

    static void pack_fn(void *vctx, int g, int num_local, void *out_delta) {
        Ctx *c = static_cast<Ctx*>(vctx);
        Delta *d = static_cast<Delta*>(out_delta);
        int *home_index = ghost_get_home_index();
        d->home_index = home_index[g];
        const Vec3<ElemT>& now = P[num_local + g].*MemPtr;
        const Vec3<ElemT>& was = c->snap[g];
        d->delta[0] = now[0] - was[0];
        d->delta[1] = now[1] - was[1];
        d->delta[2] = now[2] - was[2];
    }

    static void apply_fn(void *vctx, const void *in_delta) {
        (void)vctx;
        const Delta *d = static_cast<const Delta*>(in_delta);
        P[d->home_index].*MemPtr += d->delta;
    }

    static void cleanup_fn(void *vctx) {
        Ctx *c = static_cast<Ctx*>(vctx);
        if (c->snap) { free(c->snap); c->snap = nullptr; }
        c->num_ghosts = 0;
    }

    static const ghost_writeback_callback callback;
};

template <typename ElemT, Vec3<ElemT> particle_data::*MemPtr>
typename ParticleAddVec3Op<ElemT, MemPtr>::Ctx
ParticleAddVec3Op<ElemT, MemPtr>::s_ctx{nullptr, 0};

template <typename ElemT, Vec3<ElemT> particle_data::*MemPtr>
const ghost_writeback_callback ParticleAddVec3Op<ElemT, MemPtr>::callback = {
    sizeof(typename ParticleAddVec3Op<ElemT, MemPtr>::Delta),
    & ParticleAddVec3Op<ElemT, MemPtr>::snapshot_fn,
    & ParticleAddVec3Op<ElemT, MemPtr>::delta_for_ghost_fn,
    & ParticleAddVec3Op<ElemT, MemPtr>::pack_fn,
    & ParticleAddVec3Op<ElemT, MemPtr>::apply_fn,
    & ParticleAddVec3Op<ElemT, MemPtr>::cleanup_fn,
    & ParticleAddVec3Op<ElemT, MemPtr>::s_ctx,
};

/* GasMaxOp — value-pack max-reduce reverse-comm for any scalar field on
 * gas_cell_data. Mirrors device-side atomic_max semantics; the ADD analogue
 * (snapshot-diff additive) is NOT equivalent across ranks (two ranks each
 * producing post>snap would over-shoot home additively).
 *
 * Predicate: post > snap. Pack: value = post (NOT delta). Apply: home =
 * max(home, value). Strict (>) comparison is NaN-safe — NaN never overwrites
 * a finite home, finite never overwrites a sticky-NaN home. Used by thermalfb
 * for CellP[j].DelayTimeCoolingSNe (legacy code used additive merge here; that
 * cross-rank inconsistency is intentionally fixed in this port). */
template <typename FieldT, FieldT gas_cell_data::*MemPtr>
struct GasMaxOp {
    static_assert(std::is_arithmetic<FieldT>::value,
                  "GHOST_WRITEBACK_GAS_MAX requires an arithmetic scalar field");

    struct Ctx {
        FieldT *snap;
        int     num_ghosts;
    };
    static Ctx s_ctx;

    struct Delta {
        int    home_index;
        FieldT value;
    };

    static void snapshot_fn(void *vctx, int num_ghosts, int num_local) {
        Ctx *c = static_cast<Ctx*>(vctx);
        c->num_ghosts = num_ghosts;
        if (num_ghosts <= 0) { c->snap = nullptr; return; }
        c->snap = (FieldT*) malloc(num_ghosts * sizeof(FieldT));
        for (int g = 0; g < num_ghosts; g++) {
            c->snap[g] = CellP[num_local + g].*MemPtr;
        }
    }

    static int delta_for_ghost_fn(void *vctx, int g, int num_local) {
        Ctx *c = static_cast<Ctx*>(vctx);
        if (!c->snap) return 0;
        /* Strict >. NaN-safe: NaN > anything is false, anything > NaN is
         * false; we ship only when the post-kernel value is strictly greater
         * than the pre-kernel snapshot. */
        return (CellP[num_local + g].*MemPtr > c->snap[g]) ? 1 : 0;
    }

    static void pack_fn(void *vctx, int g, int num_local, void *out_delta) {
        (void)vctx;
        Delta *d = static_cast<Delta*>(out_delta);
        int *home_index = ghost_get_home_index();
        d->home_index = home_index[g];
        d->value      = CellP[num_local + g].*MemPtr;  /* post, not delta */
    }

    static void apply_fn(void *vctx, const void *in_delta) {
        (void)vctx;
        const Delta *d = static_cast<const Delta*>(in_delta);
        /* Strict >: NaN-safe (see snapshot_fn / delta_for_ghost_fn). */
        if (d->value > CellP[d->home_index].*MemPtr) {
            CellP[d->home_index].*MemPtr = d->value;
        }
    }

    static void cleanup_fn(void *vctx) {
        Ctx *c = static_cast<Ctx*>(vctx);
        if (c->snap) { free(c->snap); c->snap = nullptr; }
        c->num_ghosts = 0;
    }

    static const ghost_writeback_callback callback;
};

template <typename FieldT, FieldT gas_cell_data::*MemPtr>
typename GasMaxOp<FieldT, MemPtr>::Ctx
GasMaxOp<FieldT, MemPtr>::s_ctx{nullptr, 0};

template <typename FieldT, FieldT gas_cell_data::*MemPtr>
const ghost_writeback_callback GasMaxOp<FieldT, MemPtr>::callback = {
    sizeof(typename GasMaxOp<FieldT, MemPtr>::Delta),
    & GasMaxOp<FieldT, MemPtr>::snapshot_fn,
    & GasMaxOp<FieldT, MemPtr>::delta_for_ghost_fn,
    & GasMaxOp<FieldT, MemPtr>::pack_fn,
    & GasMaxOp<FieldT, MemPtr>::apply_fn,
    & GasMaxOp<FieldT, MemPtr>::cleanup_fn,
    & GasMaxOp<FieldT, MemPtr>::s_ctx,
};

/* Small helper trait — extract ElemT from Vec3<ElemT>. Used by the
 * GHOST_WRITEBACK_PARTICLE_ADD_VEC3 macro so the science author writes only
 * the field name (not the element type) at the manifest site. */
template <typename V> struct vec3_elem; /* primary undefined */
template <typename T> struct vec3_elem<Vec3<T>> { using type = T; };

} /* namespace gw_detail */

/* ============================================================================
 * Manifest macros (physics-facing).
 * ========================================================================== */

/* One line per field-op inside the manifest block. The macro yields a
 * pointer-to-callback expression with a trailing comma — the BUNDLE_END
 * macro closes the array. */
#define GHOST_WRITEBACK_PARTICLE_MIN(field)                                   \
    & ::gw_detail::ParticleMinOp<                                             \
        decltype(particle_data::field), &particle_data::field                 \
      >::callback,

#define GHOST_WRITEBACK_PARTICLE_MAX(field)                                   \
    & ::gw_detail::ParticleMaxOp<                                             \
        decltype(particle_data::field), &particle_data::field                 \
      >::callback,

#define GHOST_WRITEBACK_GAS_ADD(field)                                        \
    & ::gw_detail::GasAddOp<                                                  \
        decltype(gas_cell_data::field), &gas_cell_data::field                 \
      >::callback,

#define GHOST_WRITEBACK_PARTICLE_ADD(field)                                   \
    & ::gw_detail::ParticleAddOp<                                             \
        decltype(particle_data::field), &particle_data::field                 \
      >::callback,

/* Fixed-size array variant. `field` is e.g. `dp` (declared as
 * `MyFloat dp[3]` in particle_data) or `Metallicity` (declared as
 * `MyFloat Metallicity[NUM_METAL_SPECIES]`). `N` is the array length spelled
 * the same way as the declaration (3, NUM_METAL_SPECIES, etc.). The macro
 * extracts the element type with std::remove_extent_t. */
#define GHOST_WRITEBACK_PARTICLE_ADD_ARRAY(field, N)                          \
    & ::gw_detail::ParticleAddArrayOp<                                        \
        std::remove_extent_t<decltype(particle_data::field)>,                 \
        (N), &particle_data::field                                            \
      >::callback,

/* Vec3<T> variant. `field` is declared as `Vec3<ElemT> field` on particle_data
 * (e.g. `Vec3<MyDouble> dp`). The macro extracts ElemT via vec3_elem<>. */
#define GHOST_WRITEBACK_PARTICLE_ADD_VEC3(field)                              \
    & ::gw_detail::ParticleAddVec3Op<                                         \
        typename ::gw_detail::vec3_elem<decltype(particle_data::field)>::type,\
        &particle_data::field                                                 \
      >::callback,

#define GHOST_WRITEBACK_GAS_MAX(field)                                        \
    & ::gw_detail::GasMaxOp<                                                  \
        decltype(gas_cell_data::field), &gas_cell_data::field                 \
      >::callback,

/* Bundle assembly. BUNDLE_BEGIN(loop) opens an anonymous-namespace block
 * and starts the callback-pointer array. BUNDLE_END(loop) closes the
 * array (with a sentinel so it is always non-empty even when no flags are
 * active), defines the bundle (with loop_name = "loop" for the rank-0
 * traffic print in _end_bundle), and emits the accessor function. */
#define GHOST_WRITEBACK_BUNDLE_BEGIN(LOOP)                                    \
    namespace LOOP##_writeback_detail {                                       \
        static const struct ghost_writeback_callback *const raw_cbs[] = {

#define GHOST_WRITEBACK_BUNDLE_END(LOOP)                                      \
            nullptr  /* sentinel — array always has size >= 1 */              \
        };                                                                    \
        /* Runtime static-init counts non-null entries in raw_cbs (the        \
         * sentinel is excluded). Bundle is read-only after init. */          \
        static int count_real_cbs() {                                         \
            int n = 0;                                                        \
            for (auto p : raw_cbs) if (p) ++n;                                \
            return n;                                                         \
        }                                                                     \
        static const struct ghost_writeback_bundle bundle = {                 \
            raw_cbs, count_real_cbs(), #LOOP                                  \
        };                                                                    \
    }                                                                         \
    static inline const struct ghost_writeback_bundle *                       \
    LOOP##_ghost_writeback_bundle_ptr(void) {                                 \
        return & LOOP##_writeback_detail::bundle;                             \
    }

#endif /* GHOST_WRITEBACK_OPS_H */
