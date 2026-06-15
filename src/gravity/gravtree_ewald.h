#ifndef GRAVTREE_EWALD_H
#define GRAVTREE_EWALD_H

/* Ewald periodic-image correction trilinear interpolation — the single home for the table lookup
 * shared by the CPU walk (forcetree.cc: ewald_pot_corr + force_treeevaluate_ewald_correction) and
 * the GPU walks (gpu_gravtree.cc: the primary-pass potential correction + gpu_ewald_walk_one force).
 *
 * Weights-once / apply-N-tables: grav_ewald_interp_setup turns |dr| into the octant index + the 8
 * trilinear weights ONCE; grav_ewald_interp_apply dots those against ONE flat table. The force
 * callers reuse one set of weights across the three force tables (fcorrx/y/z) and apply the
 * per-component odd-force signs OUTSIDE this helper (the index is built from |dr|). Tables,
 * fac_intp, sign handling, and which interactions are corrected all stay at the callers.
 *
 * Dependency-light: needs only MyFloat, GIZMO_EWALD_EN (forcetree.h), and the
 * KOKKOS_INLINE_FUNCTION fallback. No Vec3, no All, no table ownership. Include AFTER
 * declarations/allvars.h + gravity/forcetree.h. Gated on BOX_PERIODIC, matching the Ewald-table
 * declarations in forcetree.cc — non-periodic builds see no names.
 *
 * The corner order + the 8-term expression order are kept identical to the legacy inline forms
 * (no loop), so the unification is bit-exact. */

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#ifdef BOX_PERIODIC

struct grav_ewald_interp_weights {
    int    i, j, k;
    double f1, f2, f3, f4, f5, f6, f7, f8;
};

/* |dr| -> clamped octant index (i,j,k) + the 8 trilinear weights. abs only; no sign. */
KOKKOS_INLINE_FUNCTION grav_ewald_interp_weights
grav_ewald_interp_setup(double dx, double dy, double dz, double fac_intp)
{
    const int EN = GIZMO_EWALD_EN;
    if(dx < 0) dx = -dx;
    if(dy < 0) dy = -dy;
    if(dz < 0) dz = -dz;
    grav_ewald_interp_weights w;
    double u = dx * fac_intp; w.i = (int) u; if(w.i >= EN) w.i = EN - 1; u -= w.i;
    double v = dy * fac_intp; w.j = (int) v; if(w.j >= EN) w.j = EN - 1; v -= w.j;
    double t = dz * fac_intp; w.k = (int) t; if(w.k >= EN) w.k = EN - 1; t -= w.k;
    w.f1 = (1 - u) * (1 - v) * (1 - t);
    w.f2 = (1 - u) * (1 - v) * (t);
    w.f3 = (1 - u) * (v)     * (1 - t);
    w.f4 = (1 - u) * (v)     * (t);
    w.f5 = (u)     * (1 - v) * (1 - t);
    w.f6 = (u)     * (1 - v) * (t);
    w.f7 = (u)     * (v)     * (1 - t);
    w.f8 = (u)     * (v)     * (t);
    return w;
}

/* 8-corner trilinear dot for one flat table (octant stride = GIZMO_EWALD_EN+1). Same corner order
 * and 8-term expression order as the legacy inline forms. */
KOKKOS_INLINE_FUNCTION double
grav_ewald_interp_apply(const MyFloat *table, const grav_ewald_interp_weights &w)
{
    const int stride1 = GIZMO_EWALD_EN + 1;
    const int stride2 = stride1 * stride1;
    const int i = w.i, j = w.j, k = w.k;
#define GRAV_EWALD_IDX3(ii,jj,kk) ((ii)*stride2 + (jj)*stride1 + (kk))
    double r = table[GRAV_EWALD_IDX3(i,   j,   k  )] * w.f1 +
               table[GRAV_EWALD_IDX3(i,   j,   k+1)] * w.f2 +
               table[GRAV_EWALD_IDX3(i,   j+1, k  )] * w.f3 +
               table[GRAV_EWALD_IDX3(i,   j+1, k+1)] * w.f4 +
               table[GRAV_EWALD_IDX3(i+1, j,   k  )] * w.f5 +
               table[GRAV_EWALD_IDX3(i+1, j,   k+1)] * w.f6 +
               table[GRAV_EWALD_IDX3(i+1, j+1, k  )] * w.f7 +
               table[GRAV_EWALD_IDX3(i+1, j+1, k+1)] * w.f8;
#undef GRAV_EWALD_IDX3
    return r;
}

#endif /* BOX_PERIODIC */

#endif /* GRAVTREE_EWALD_H */
