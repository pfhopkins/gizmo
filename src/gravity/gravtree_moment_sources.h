#ifndef GRAVTREE_MOMENT_SOURCES_H
#define GRAVTREE_MOMENT_SOURCES_H

/*! \file gravtree_moment_sources.h
 *  \brief Host-only single source of truth for the per-particle gravity-tree
 *         source INPUT physics gates: RT source luminosity, sink bolometric
 *         luminosity + emission-angle vector, and cosmic-ray injection rate.
 *
 *  These per-particle quantities feed the GPU gravity walk and the node-moment
 *  build.  They are precomputed on the host because the underlying helpers
 *  (rt_get_source_luminosity, sink_lum_bol, cr_get_source_injection_rate) are
 *  not device-callable.  The gates used to be duplicated in each precompute
 *  venue (the primary GPU walk in gpu_gravtree.cc, the node-moment refresh in
 *  gpu_moment_refresh.cc, and the LET leaf synthesis in let_pack.cc); this
 *  header is the one place that physics lives.
 *
 *  Host-only: plain static inline, NO Kokkos / no device assumptions / no
 *  storage ownership.  It uses the exact gates the live precomputes use, and
 *  sets a validity flag per section (rt_active / bh_active) so the caller writes
 *  only ACTIVE entries into its already-bulk-zeroed arrays — matching the legacy
 *  per-venue write pattern (no extra UVM writes).  Pure refactor — no physics
 *  change.  The array payload fields are left untouched when their section is
 *  inactive; the caller must gate its copy-out on the corresponding flag.
 *
 *  Include this AFTER allvars.h and proto.h (which declare the field types and
 *  rt_get_source_luminosity[_chimes] / cr_get_source_injection_rate).  proto.h
 *  and sinks/sink.h have no include guards, so we do not re-include them here;
 *  sink_lum_bol() is forward-declared below to keep this header self-contained.
 */

#ifdef SINK_PHOTONMOMENTUM
double sink_lum_bol(double mdot, double mass, long pindex);   /* canonical decl in sinks/sink.h (unguarded) */
#endif

/* Per-particle source inputs.  Field types match the venue storage / wire types
 * (MyFloat / Vec3<MyFloat> / double) so the venue copy-out is a plain assignment
 * with no extra conversion.  rt_active / bh_active flag whether the array
 * payloads were filled; the caller copies those out only when the flag is set
 * (otherwise the payload fields are indeterminate and must not be read). */
struct gravtree_source_inputs_t {
#ifdef RT_USE_GRAVTREE
    int     rt_active;                       /* nonzero if the particle is an active RT source */
    MyFloat src_lum[N_RT_FREQ_BINS];         /* valid only when rt_active */
#ifdef CHIMES_STELLAR_FLUXES
    double  src_lum_G0[CHIMES_LOCAL_UV_NBINS];  /* valid only when rt_active */
    double  src_lum_ion[CHIMES_LOCAL_UV_NBINS]; /* valid only when rt_active */
#endif
#endif
#ifdef SINK_PHOTONMOMENTUM
    int           bh_active;                 /* nonzero if a live emitting sink (gates bh_lum/bh_angle) */
    MyFloat       bh_lum;                    /* bolometric sink luminosity; valid only when bh_active */
    Vec3<MyFloat> bh_angle;                  /* emission-angle vector;        valid only when bh_active */
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    MyFloat cr_inject;                       /* cosmic-ray source injection rate (0 for non-sources) */
#endif
};

/* Fill *out using the EXACT gates the live gravity precomputes use.  Only the
 * cheap validity scalars are initialised unconditionally; array payloads are
 * written only inside an active section, and the caller mirrors that by copying
 * them out only when the flag is set.  No physics change relative to the legacy
 * per-venue loops. */
static inline void
gravtree_fill_particle_source_inputs(int p, struct particle_data *P_arr,
                                     struct gas_cell_data *CellP_arr,
                                     struct gravtree_source_inputs_t *out)
{
#ifdef RT_USE_GRAVTREE
    out->rt_active = 0;
    if(P_arr[p].Mass > 0) {
        double lum[N_RT_FREQ_BINS];
#ifdef CHIMES_STELLAR_FLUXES
        double lum_G0[CHIMES_LOCAL_UV_NBINS], lum_ion[CHIMES_LOCAL_UV_NBINS];
        out->rt_active = rt_get_source_luminosity_chimes(p, 1, lum, lum_G0, lum_ion, P_arr, CellP_arr);
#else
        out->rt_active = rt_get_source_luminosity(p, 1, lum, P_arr, CellP_arr);
#endif
        if(out->rt_active) {
            int kf;
            for(kf = 0; kf < N_RT_FREQ_BINS; kf++) {out->src_lum[kf] = (MyFloat) lum[kf];}
#ifdef CHIMES_STELLAR_FLUXES
            for(kf = 0; kf < CHIMES_LOCAL_UV_NBINS; kf++) {
                out->src_lum_G0[kf]  = lum_G0[kf];
                out->src_lum_ion[kf] = lum_ion[kf];
            }
#endif
        }
    }
#endif /* RT_USE_GRAVTREE */

#ifdef SINK_PHOTONMOMENTUM
    out->bh_active = 0;
    if(P_arr[p].Type == 5 && P_arr[p].Mass > 0 &&
       P_arr[p].DensityAroundParticle > 0 && P_arr[p].Sink_Mdot > 0) {
        out->bh_active = 1;     /* gated in regardless of the luminosity value (which may be 0) */
        out->bh_lum    = (MyFloat) sink_lum_bol(P_arr[p].Sink_Mdot, P_arr[p].Sink_Mass, p);
#if defined(SINK_FOLLOW_ACCRETED_ANGMOM)
        out->bh_angle = P_arr[p].Sink_Specific_AngMom;
#else
        out->bh_angle = P_arr[p].GradRho;
#endif
    }
#endif /* SINK_PHOTONMOMENTUM */

#ifdef COSMIC_RAY_SUBGRID_LEBRON
    /* nonzero only for Type 4/5 sources; gas + degenerate (Mass<=0) slots return 0.
     * Mirror the legacy CPU tree, which evaluates every particle ungated. */
    out->cr_inject = 0;
    if(P_arr[p].Mass > 0) {
        out->cr_inject = (MyFloat) cr_get_source_injection_rate(p, P_arr, CellP_arr);
    }
#endif /* COSMIC_RAY_SUBGRID_LEBRON */
}

#endif /* GRAVTREE_MOMENT_SOURCES_H */
