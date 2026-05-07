/* ghost_exchange_spec.h — public spec struct + GHOST_TYPE_* / NGB_SEARCH_*
 * macros for callers that build their own spec literal in-line and call
 * ghost_exchange_run() (declared in core/proto.h).
 *
 * The spec literal at the caller is the single source of truth for that
 * loop's (supply_type_mask, search_mode, query list). Editing those there
 * is the only edit needed to flip the call's physics. ghost_exchange.cc
 * has no per-caller knowledge.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO. */

#ifndef GHOST_EXCHANGE_SPEC_H
#define GHOST_EXCHANGE_SPEC_H

#include "neighbor_list.h"   /* NGB_SEARCH_ONEWAY, NGB_SEARCH_SYMMETRIC */

/* Type bitmask values matching P[].Type encoding. Bit k set ⇔ Type k included.
 * Keep these names literal: Type 0 is usually cells/fluids, Type 5 is often
 * sinks, but Types 1-4 are intentionally physics/config dependent. */
#define GHOST_TYPE_0     (1u << 0)
#define GHOST_TYPE_1     (1u << 1)
#define GHOST_TYPE_2     (1u << 2)
#define GHOST_TYPE_3     (1u << 3)
#define GHOST_TYPE_4     (1u << 4)
#define GHOST_TYPE_5     (1u << 5)
#define GHOST_TYPE_ALL   ((1u << 6) - 1u)

struct ghost_exchange_spec_t {
    /* Legacy: filters ActiveParticleList in tile-overlap impl AND in the
     * request-driven impl when n_queries < 0. New explicit-query callers
     * leave 0. */
    unsigned int request_type_mask;

    /* Bitmask of Types eligible to be SUPPLIED as ghosts (the search pool). */
    unsigned int supply_type_mask;

    /* NGB_SEARCH_ONEWAY (r_ij < h_i) or NGB_SEARCH_SYMMETRIC (r_ij < max(h_i,h_j)). */
    int          search_mode;

    /* Multiplier on per-query h before the predicate. */
    double       safety_factor;

    /* For diagnostic prints + the request-driven caller allowlist. Must
     * match a name in ghost_request_driven_caller_safe() inside
     * ghost_exchange.cc to actually go through request-driven. Otherwise
     * the call falls back to legacy tile-overlap. */
    const char  *caller_name;

    /* Caller-owned explicit query list. n_queries >= 0 → use these. n_queries
     * < 0 (sentinel) → use legacy ActiveParticleList scan filtered by
     * request_type_mask. query_h is the RAW h, NOT pre-multiplied by
     * safety_factor (the impl applies it). */
    int                  n_queries;
    const double       (*query_pos)[3];
    const double        *query_h;
};

#endif /* GHOST_EXCHANGE_SPEC_H */
