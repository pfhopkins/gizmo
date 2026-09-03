/* timestep_functions.h — Canonical KOKKOS_INLINE_FUNCTION implementations of
 * timestep utility functions.  Single source of truth for both CPU and GPU.
 *
 * timestep_dilation_factor returns the dilation factor frozen for this particle when its
 * timestep was assigned (core/timestep.cc, get_timestep), so a step's physical landing time
 * cannot mutate while it is being taken.  Identical on CPU and GPU.  The live evaluations,
 * for timestep assignment and for tree nodes, are host-only and live in core/timestep.cc.
 *
 * Include order: after allvars.h, proto.h. */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

KOKKOS_INLINE_FUNCTION
double timestep_dilation_factor(int i, struct particle_data *pp)
{
#ifdef USE_TIMESTEP_DILATION_FOR_ZOOMS
    if(i < 0) {return 1;}
    return pp[i].TimestepDilationFactor;
#else
    (void)i; (void)pp; return 1;
#endif
}

KOKKOS_INLINE_FUNCTION
double unit_integertime_in_physical(int i, struct particle_data *pp)
{
    return (All.Timebase_interval / All.cf_hubble_a) * timestep_dilation_factor(i, pp);
}

KOKKOS_INLINE_FUNCTION
double get_physical_timestep_from_timebin(int bin, int i, struct particle_data *pp)
{
    return GET_INTEGERTIME_FROM_TIMEBIN(bin) * unit_integertime_in_physical(i, pp);
}

KOKKOS_INLINE_FUNCTION
double get_particle_timestep_in_physical(int i, struct particle_data *pp)
{
    return pp[i].integertime_step() * unit_integertime_in_physical(i, pp);
}

/* --- live dilation factors -------------------------------------------------
 * The two nuclear-zoom helpers were file-scope statics in core/timestep.cc and read
 * nothing but All; the node factor below reads All and the node array passed to it.
 * They live here because the node dilation factor is needed wherever a drift factor
 * is, on host and device alike. core/timestep.cc provides their externally-visible
 * host symbols through its non-inline re-include of this header. */

#if defined(USE_TIMESTEP_DILATION_FOR_ZOOMS) && defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM)
/* smallest physical distance from pos to any of the refinement centers */
KOKKOS_INLINE_FUNCTION
double distance_to_nearest_refinement_center(Vec3<double> pos)
{
    double rmin = MAX_REAL_NUMBER;
    for(int j = 0; j < SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM; j++)
    {
        Vec3<double> p0 = All.SpecialParticle_Position_ForRefinement[j];
        Vec3<double> dp = All.cf_atime * (pos - p0);
        double r = dp.norm(); if(r < rmin) {rmin = r;}
    }
    return rmin;
}

/* dilation amplitude a >= 1 at distance r from the refinement center: unity far away, saturating
   at amax on approach */
KOKKOS_INLINE_FUNCTION
double nuclear_zoom_dilation_amplitude(double r)
{
    double fac_amax = 100.;
#ifdef SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_SPECIALBOUNDARIES
#if (SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_SPECIALBOUNDARIES >= 3)
    fac_amax = 1.e6;
#endif
#endif
    double amax = fac_amax;
    double r_amax = fac_amax * All.ForceSoftening[3]; // modify as needed
    double index = 1;
    if(r < 1.e-10 || isnan(r) || isfinite(r)==0) {r = 1.e-10;}
    return 1. + 1. / (1./amax + pow(r / r_amax, index));
}
#endif


/* live dilation factor at the center of mass of tree node 'no', for drifting the node itself. Nodes
   carry no particle type, so the stars-only restriction is particle-only and does not apply here.

   Nodes also carry no sink distance: Min_Distance_to_Sink is a per-particle result of the gravity
   walk, and there is no node-level equivalent to feed the weighted-motion smoothing. So under
   SPECIAL_POINT_WEIGHTED_MOTION (without the nuclear-zoom term, which does work for nodes) a node
   drifts undilated while the particles it summarises drift at the smoothing weight, leaving its
   center of mass inconsistent with them. Giving nodes that term means carrying a sink distance
   through the tree moments. The weighted-motion module is still in development; this needs
   resolving before it is relied on. */
KOKKOS_INLINE_FUNCTION
double return_node_timestep_dilation_factor_P(int no, const struct NODE *nodes)
{
#if !defined(USE_TIMESTEP_DILATION_FOR_ZOOMS) || defined(DILATION_FOR_STELLAR_KINEMATICS_ONLY)
    (void)no; (void)nodes; return 1;
#else

    if(All.Time <= All.TimeBegin) {return 1;}
    if(no < 0) {return 1;}

    double a = 1;

#if defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM)
    Vec3<double> pos_node; pos_node = nodes[no].u.d.s;
    a = nuclear_zoom_dilation_amplitude(distance_to_nearest_refinement_center(pos_node));
#else
    (void)nodes;
#endif

    return 1. / a;
#endif
}


/* --- drift and gravitational-kick time factors -----------------------------
 * A drift or kick factor is the time integral over the step, times the particle's
 * or node's timestep dilation. For a non-cosmological run the integral is just
 * the elapsed code time; for a cosmological one it is a lookup in the tables
 * built once by init_drift_table(), which is called only when
 * ComovingIntegrationOn is set -- so on a non-cosmological run the tables are
 * never filled and the pointers below are null.
 *
 * The view carries whatever a caller needs to evaluate a factor without reaching
 * a global: on the host it is filled from the tables themselves, in device code
 * from their shared-memory mirror. */
struct DriftKickTableView
{
    const double *drift;        /*!< DriftTable, or null on a non-cosmological run */
    const double *gravkick;     /*!< GravKickTable, or null on a non-cosmological run */
    double logTimeBegin;        /*!< log of the scale factor the tables start at */
    double logTimeMax;          /*!< log of the scale factor they end at */
    double timebase_interval;   /*!< code time per unit of integer time */
    int comoving;               /*!< nonzero if the tables are live and must be used */
};

/* Assembles a view. Both the host entry points and the device-side mirror build
   their view here, so the meaning of each field is fixed in one place. */
KOKKOS_INLINE_FUNCTION
struct DriftKickTableView drift_kick_table_view(const double *drift, const double *gravkick,
                                                double logTimeBegin, double logTimeMax,
                                                double timebase_interval, int comoving)
{
    struct DriftKickTableView view;
    view.drift = comoving ? drift : NULL;          /* never read when not comoving, and never built */
    view.gravkick = comoving ? gravkick : NULL;
    view.logTimeBegin = logTimeBegin;
    view.logTimeMax = logTimeMax;
    view.timebase_interval = timebase_interval;
    view.comoving = comoving;
    return view;
}

/* The one interpolator. Returns the time integral over [time0, time1] with no
   dilation applied; callers multiply in the factor for the particle or node
   they are drifting. */
KOKKOS_INLINE_FUNCTION
double drift_kick_table_factor(const double *table, integertime time0, integertime time1,
                               const struct DriftKickTableView *view)
{
    if(!view->comoving) {return (time1 - time0) * view->timebase_interval;}

    double logTimeBegin = view->logTimeBegin, logTimeMax = view->logTimeMax;
    double a1 = logTimeBegin + time0 * view->timebase_interval;
    double a2 = logTimeBegin + time1 * view->timebase_interval;
    double u1, u2, df1, df2; int i1, i2;

    if(logTimeMax > logTimeBegin)
        u1 = (a1 - logTimeBegin) / (logTimeMax - logTimeBegin) * DRIFT_TABLE_LENGTH;
    else
        u1 = 0;
    i1 = (int) u1;
    if(i1 >= DRIFT_TABLE_LENGTH)
        i1 = DRIFT_TABLE_LENGTH - 1;

    if(i1 <= 1)
        df1 = u1 * table[0];
    else
        df1 = table[i1 - 1] + (table[i1] - table[i1 - 1]) * (u1 - i1);

    if(logTimeMax > logTimeBegin)
        u2 = (a2 - logTimeBegin) / (logTimeMax - logTimeBegin) * DRIFT_TABLE_LENGTH;
    else
        u2 = 0;
    i2 = (int) u2;
    if(i2 >= DRIFT_TABLE_LENGTH)
        i2 = DRIFT_TABLE_LENGTH - 1;

    if(i2 <= 1)
        df2 = u2 * table[0];
    else
        df2 = table[i2 - 1] + (table[i2] - table[i2 - 1]) * (u2 - i2);

    return df2 - df1;
}

/*! Cosmological prefactor for a drift step between time0 and time1, dilated by
 *  'dilation'. The value returned is \f[ \int_{a_0}^{a_1} \frac{{\rm d}a}{H(a)} \f]
 */
KOKKOS_INLINE_FUNCTION
double get_drift_factor_impl(integertime time0, integertime time1, double dilation,
                             const struct DriftKickTableView *view)
{
    return drift_kick_table_factor(view->drift, time0, time1, view) * dilation;
}

KOKKOS_INLINE_FUNCTION
double get_gravkick_factor_impl(integertime time0, integertime time1, double dilation,
                                const struct DriftKickTableView *view)
{
    return drift_kick_table_factor(view->gravkick, time0, time1, view) * dilation;
}
