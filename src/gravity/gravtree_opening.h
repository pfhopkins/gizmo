#ifndef GRAVTREE_OPENING_H
#define GRAVTREE_OPENING_H

/* Acceptance-geometry decision for the gravity tree walk (Stage 2 of the per-node decision).
 *
 * The per-node decision is two-stage. The CALLER runs Stage 1 first — the node-structure and
 * LET-sentinel prechecks that must precede geometry:
 *   - empty node (mass <= 0)            -> skip to sibling
 *   - single-particle node, local       -> open to its leaf (exact force)
 *   - single-particle node, foreign      -> continue to Stage 2 (treat as a normal node)
 * then calls the Stage-2 predicate below for a real aggregate node to choose: skip it (PM
 * short-range cull), open it, or accept its multipole.
 *
 * This file is the single home for the multipole-acceptance geometry shared by the GPU primary
 * walk and (later) the LET cell predicate. It is foreign-blind: the caller applies the LET policy
 * (OPEN && foreign-multipole -> ACCEPT; SKIP/ACCEPT honored as-is). Wrap policy lives in the
 * gravity_box_* helpers; this file is decision logic only. Within Stage 2 the only SKIP (PM cull)
 * precedes every OPEN, so foreign-blind + the caller transform is exactly walk-equivalent.
 *
 * Distances are supplied by the caller so the walk computes the center-of-mass distance ONCE and
 * uses it for both the opening decision and the accepted-node force (single source). A point
 * wrapper (below) and a later cell/AABB wrapper feed the same core. */

#include "gravity_box_distance.h"

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

typedef enum {
    GRAV_SKIP_NODE = 0,       /* PM short-range cull: contribute nothing (walk: -> sibling) */
    GRAV_ACCEPT_MULTIPOLE,    /* use the node's own moment (walk: multipole -> sibling) */
    GRAV_OPEN_NODE            /* must descend (walk: -> nextnode); foreign => caller maps to ACCEPT */
} gravtree_open_t;

/* Stage-2 acceptance geometry for a real aggregate node. Foreign-blind. Mirrors the node-branch
 * of gpu_gravtree_walk_one (gpu_gravtree.cc) from the PM cull through the relative criterion.
 *   r2_com        : |node center-of-mass - target|^2, caller-wrapped (the SAME r2 used for the force)
 *   cen_dx/dy/dz  : node geometric center - target, RAW (wrapped here via gravity_box_*)
 *   t_soft        : target softening (walk's `soft`)
 *   t_h           : target interaction radius (walk's `h`; non-NEIGHBORS softening-open)
 *   t_aold        : target a_old * ErrTolForceAcc (relative criterion)
 *   t_type        : target Type (sink-direct gate)
 *   len/mass/msoft: node size / mass / max softening
 *   rcut/rcut2    : PM short-range cutoff and its square
 *   n_sink        : sinks under the node (sink-direct gate; 0 if unused)
 *   is_first_step : hybrid opening — relative criterion suppressed on step 0 */
KOKKOS_INLINE_FUNCTION
gravtree_open_t gravtree_open_decision_from_distances(
    double r2_com,
    double cen_dx, double cen_dy, double cen_dz,
    double t_soft, double t_h, double t_aold, int t_type,
    double len, double mass, double msoft,
    double rcut, double rcut2, int n_sink, int is_first_step)
{
    (void) t_h; (void) t_type; (void) n_sink;
    (void) rcut; (void) rcut2; (void) is_first_step;

#ifdef PMGRID
    if(r2_com > rcut2)
    {
        double eff_dist = rcut + 0.5 * len;
        double dcx = gravity_box_long_abs_x(cen_dx, cen_dy, cen_dz, -1);
        double dcy = gravity_box_long_abs_y(cen_dx, cen_dy, cen_dz, -1);
        double dcz = gravity_box_long_abs_z(cen_dx, cen_dy, cen_dz, -1);
        if(dcx > eff_dist || dcy > eff_dist || dcz > eff_dist) { return GRAV_SKIP_NODE; }
    }
#endif

#ifdef NEIGHBORS_MUST_BE_COMPUTED_EXPLICITLY_IN_FORCETREE
    {
        double dcx = cen_dx, dcy = cen_dy, dcz = cen_dz;
        gravity_box_nearest_image(dcx, dcy, dcz, -1);
        double dist_to_center2 = dcx*dcx + dcy*dcy + dcz*dcz;
        double soft_max = (t_soft > msoft) ? t_soft : msoft;
        double dist_to_open = soft_max + len * 1.73205 / 2.0;
        if(dist_to_center2 < dist_to_open * dist_to_open) { return GRAV_OPEN_NODE; }
    }
#else
    if(t_h < msoft && r2_com < msoft * msoft) { return GRAV_OPEN_NODE; }
#endif

    if(All.ErrTolTheta)
    {
        if(len * len > r2_com * All.ErrTolTheta * All.ErrTolTheta) { return GRAV_OPEN_NODE; }
    }
#ifndef GRAVITY_HYBRID_OPENING_CRIT
    else
#else
    /* hybrid: Barnes-Hut on step 0 (no a_old yet), then ALSO apply the relative criterion later. */
    if(!is_first_step)
#endif
    {
        if((r2_com < (t_soft + 0.6*len)*(t_soft + 0.6*len)) ||
           (r2_com < (msoft  + 0.6*len)*(msoft  + 0.6*len))) { return GRAV_OPEN_NODE; }
        if(mass * len * len > r2_com * r2_com * t_aold) { return GRAV_OPEN_NODE; }
        double dcx = gravity_box_long_abs_x(cen_dx, cen_dy, cen_dz, -1);
        double dcy = gravity_box_long_abs_y(cen_dx, cen_dy, cen_dz, -1);
        double dcz = gravity_box_long_abs_z(cen_dx, cen_dy, cen_dz, -1);
        if(dcx < 0.60*len && dcy < 0.60*len && dcz < 0.60*len) { return GRAV_OPEN_NODE; }
#if (defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES)) && defined(SINGLE_STAR_DIRECT_GRAVITY_RADIUS)
        if(t_type == 5 && n_sink > 0)
        {
            double r_direct = (double)SINGLE_STAR_DIRECT_GRAVITY_RADIUS / UNIT_LENGTH_IN_AU + 0.6*len;
            if(r2_com < r_direct * r_direct) { return GRAV_OPEN_NODE; }
        }
#endif
    }

    return GRAV_ACCEPT_MULTIPOLE;
}

/* Point-target wrapper: computes the wrapped center-of-mass distance + raw center deltas, then
 * defers to the core. NOT used by the hot walk (which already owns its wrapped dr/r2); for tests
 * and the later cell/AABB (LET) variant.
 *   p{x,y,z} : target position;  s{x,y,z} : node center-of-mass;  c{x,y,z} : node geometric center */
KOKKOS_INLINE_FUNCTION
gravtree_open_t gravtree_open_decision_point(
    double px, double py, double pz,
    double sx, double sy, double sz,
    double cx, double cy, double cz,
    double t_soft, double t_h, double t_aold, int t_type,
    double len, double mass, double msoft,
    double rcut, double rcut2, int n_sink, int is_first_step)
{
    double drx = sx - px, dry = sy - py, drz = sz - pz;
    gravity_box_nearest_image(drx, dry, drz, -1);
    double r2_com = drx*drx + dry*dry + drz*drz;
    double cen_dx = cx - px, cen_dy = cy - py, cen_dz = cz - pz;
    return gravtree_open_decision_from_distances(
        r2_com, cen_dx, cen_dy, cen_dz,
        t_soft, t_h, t_aold, t_type, len, mass, msoft,
        rcut, rcut2, n_sink, is_first_step);
}

/* Conservative per-axis distance from a point to a receiver-cover axis-aligned box [lo,hi]^3,
 * under the gravity wrap policy. The displacement to the box center is wrapped jointly (one
 * nearest-image call, so shearing-box axis coupling enters through that wrap), then reduced to
 * each axis half-width interval (zero once inside). The three returned distances are non-negative
 * and <= boxHalf, so feeding them as the core's center deltas leaves its internal gravity_box_*
 * wrap idempotent. Using the MINIMUM distance over the cover makes every distance-decreasing open
 * test fire whenever any target in the cover would fire it, and the lone distance-increasing test
 * (PM cull) skip only when the whole cover is beyond the cutoff -- a conservative over-include of
 * what the walk needs. The per-axis half-width reduction is exact for open / non-shearing periodic
 * boxes; under a shearing box (or a very large cover) it stays conservative -- it over-includes,
 * but is not necessarily tight. */
KOKKOS_INLINE_FUNCTION
void gravtree_cover_axis_min_dist(double px, double py, double pz,
                                  const double cover_min[3], const double cover_max[3],
                                  double &ax, double &ay, double &az)
{
    double mx = 0.5*(cover_min[0]+cover_max[0]), wx = 0.5*(cover_max[0]-cover_min[0]);
    double my = 0.5*(cover_min[1]+cover_max[1]), wy = 0.5*(cover_max[1]-cover_min[1]);
    double mz = 0.5*(cover_min[2]+cover_max[2]), wz = 0.5*(cover_max[2]-cover_min[2]);
    double dx = px - mx, dy = py - my, dz = pz - mz;
    gravity_box_nearest_image(dx, dy, dz, -1);
    double adx = (dx < 0.0) ? -dx : dx;
    double ady = (dy < 0.0) ? -dy : dy;
    double adz = (dz < 0.0) ? -dz : dz;
    ax = (adx > wx) ? adx - wx : 0.0;
    ay = (ady > wy) ? ady - wy : 0.0;
    az = (adz > wz) ? adz - wz : 0.0;
}

/* Cell/AABB-target opening decision over a receiver cover (the point wrapper above is the
 * degenerate cover_min==cover_max==target). Only the distance/parameter reduction differs from
 * the point variant -- the same Stage-2 core makes the decision. Distances are the conservative
 * minimum over the cover (CoM distance for the force-side geometry, geometric-center distance for
 * the PM/inside-node side checks); softening, OldAcc and sink-target inputs are the cover's
 * worst-case (max softening for the relative softening-open, min softening for the node-softening
 * open, min OldAcc for the relative criterion, sink-target if any cover particle is a sink). The
 * caller still owns the structure phase (empty subtree, single-particle) ahead of this. */
KOKKOS_INLINE_FUNCTION
gravtree_open_t gravtree_open_decision_cell(
    double cx, double cy, double cz,
    double sx, double sy, double sz,
    double len, double mass, double msoft, int n_sink,
    const double cover_min[3], const double cover_max[3],
    double t_soft_max, double t_soft_min, double t_aold_min, int cover_has_sink,
    double rcut, double rcut2, int is_first_step)
{
    double sax, say, saz;
    gravtree_cover_axis_min_dist(sx, sy, sz, cover_min, cover_max, sax, say, saz);
    double r2_com = sax*sax + say*say + saz*saz;
    double cax, cay, caz;
    gravtree_cover_axis_min_dist(cx, cy, cz, cover_min, cover_max, cax, cay, caz);
    return gravtree_open_decision_from_distances(
        r2_com, cax, cay, caz,
        t_soft_max, t_soft_min, t_aold_min, cover_has_sink ? 5 : 0,
        len, mass, msoft, rcut, rcut2, n_sink, is_first_step);
}

#endif /* GRAVTREE_OPENING_H */
