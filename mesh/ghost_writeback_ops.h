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
 * Operations available in B.iv:
 *   GHOST_WRITEBACK_PARTICLE_MIN(field)   home P[j].field = min(home, ghost)
 *
 * Future ops (NOT yet shipped — add when a migrated caller actually needs
 * them, then implement + test on the same Vista validation matrix):
 *   GHOST_WRITEBACK_PARTICLE_MAX(field)   home P[j].field = max(home, ghost)
 *   GHOST_WRITEBACK_PARTICLE_ADD(field)   delta = post - snapshot
 *   GHOST_WRITEBACK_GAS_ADD(field)        same as _ADD but on CellP[j]
 *   ...vector variants, snapshot-diff variants, etc.
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

/* Bundle assembly. BUNDLE_BEGIN(loop) opens an anonymous-namespace block
 * and starts the callback-pointer array. BUNDLE_END(loop) closes the
 * array (with a sentinel so it is always non-empty even when no flags are
 * active), defines the bundle, and emits the accessor function. */
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
            raw_cbs, count_real_cbs()                                         \
        };                                                                    \
    }                                                                         \
    static inline const struct ghost_writeback_bundle *                       \
    LOOP##_ghost_writeback_bundle_ptr(void) {                                 \
        return & LOOP##_writeback_detail::bundle;                             \
    }

#endif /* GHOST_WRITEBACK_OPS_H */
