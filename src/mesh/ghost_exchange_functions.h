/* ghost_exchange_functions.h -- shared pure-geometry neighbour-accept predicate.
 *
 * gx_pair_accept() is the SINGLE source of truth for the per-pair geometric
 * neighbour test used by ghost discovery: minimum-image separation vs the
 * search radius (ONEWAY r<h_q ; SYMMETRIC r<max(h_q,h_j)).  It is the exact
 * predicate the receiver BVH leaf walk applies (ghost_exchange.cc
 * gx_walk_local_bvh) and is meant to be reused verbatim by the bounded
 * fine-tree receiver walk so all walkers accept identical pairs.
 *
 * SCOPE: geometry ONLY.  No supply-type mask, no source/destination
 * bookkeeping, no dedup, no matched[] write -- those are caller eligibility
 * concerns and stay outside the helper (different callers share this accept
 * but differ on supply eligibility).
 *
 * Host- and device-callable.  GX_EXCHANGE_INLINE READS (never defines/mutates)
 * KOKKOS_INLINE_FUNCTION: a GPU TU that included Kokkos upstream gets the real
 * device decoration; a plain-mpicxx host TU (ghost_exchange.cc) gets `inline`
 * and never pulls <Kokkos_Core.hpp>.  No globals, no Kokkos types in the body.
 *
 * Perf note: gx_pair_accept recomputes h_q*h_q per candidate (the pre-extraction
 * BVH walker hoisted it once per query).  h_q and search_mode are loop-invariant
 * across the leaf scan, so the inlined square should hoist via LICM; revisit only
 * if profiling shows a hot-loop cost.
 */
#ifndef GIZMO_GHOST_EXCHANGE_FUNCTIONS_H
#define GIZMO_GHOST_EXCHANGE_FUNCTIONS_H

#include "neighbor_list.h"   /* NGB_SEARCH_ONEWAY / NGB_SEARCH_SYMMETRIC */

#ifdef KOKKOS_INLINE_FUNCTION
#define GX_EXCHANGE_INLINE KOKKOS_INLINE_FUNCTION
#else
#define GX_EXCHANGE_INLINE inline
#endif

/* Geometric neighbour accept.  Returns 1 if j is a neighbour of the query
 * under the minimum-image metric, else 0.
 *   pos_q[3]        query position
 *   h_q             query search radius
 *   xj,yj,zj        neighbour position
 *   h_j             neighbour supply radius (ignored for ONEWAY)
 *   search_mode     NGB_SEARCH_ONEWAY or NGB_SEARCH_SYMMETRIC
 *   periodic_flags  per-axis 0/1 wrap flag
 *   box_sizes       per-axis box length (used only where periodic_flags set) */
GX_EXCHANGE_INLINE
int gx_pair_accept(const double pos_q[3], double h_q,
                   double xj, double yj, double zj, double h_j,
                   int search_mode,
                   const int periodic_flags[3],
                   const double box_sizes[3])
{
    double dx = pos_q[0] - xj;
    double dy = pos_q[1] - yj;
    double dz = pos_q[2] - zj;
    if(periodic_flags[0]) { double b=box_sizes[0], h=0.5*b; if(dx>h) dx-=b; else if(dx<-h) dx+=b; }
    if(periodic_flags[1]) { double b=box_sizes[1], h=0.5*b; if(dy>h) dy-=b; else if(dy<-h) dy+=b; }
    if(periodic_flags[2]) { double b=box_sizes[2], h=0.5*b; if(dz>h) dz-=b; else if(dz<-h) dz+=b; }
    double r2 = dx*dx + dy*dy + dz*dz;
    double thresh2;
    if(search_mode == NGB_SEARCH_ONEWAY) {
        thresh2 = h_q * h_q;
    } else {
        double h_max = (h_q > h_j) ? h_q : h_j;
        thresh2 = h_max * h_max;
    }
    return r2 < thresh2;
}

#endif /* GIZMO_GHOST_EXCHANGE_FUNCTIONS_H */
