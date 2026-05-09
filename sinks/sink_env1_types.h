/* sinks/sink_env1_types.h — neutral payload types shared across sink_env1
 * runner consumers (Spec ActiveData embedding, SSOT pair kernel signature).
 *
 * No GPU dependency. Only basic typedefs (Vec3, MyIDType) and the SINK_GRAV*
 * macros from declarations/allvars.h are needed to resolve the conditional
 * fields. Anyone who needs sink_env1_query_t includes THIS header.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO.
 */
#ifndef SINK_ENV1_TYPES_H
#define SINK_ENV1_TYPES_H

#include "../declarations/allvars.h"

#ifdef SINK_PARTICLES

/* Compact active-side query payload. Trivially copyable for byte-level
 * MPI transfer. Conditional fields wrapped in the same #ifdefs as the
 * pair body so payload size matches per build config.
 *
 * h_search MUST come from nl_radii[a] (the radius the existing GPU path
 * was given), not from P[ii].KernelRadius — they happen to be equal in
 * the current code but the pair body is contractually keyed off the
 * caller-supplied radius.
 */
struct sink_env1_query_t {
    Vec3<double>  pos;          /* P[ii].Pos */
    Vec3<double>  vel;          /* P[ii].Vel */
    MyIDType      id;           /* P[ii].ID — for self-skip predicate */
    double        h_search;     /* nl_radii[a] — pair body's h_i */
    double        ags_h;        /* AGS_KernelRadius if defined, else sink_radius_grav */
#if defined(SINK_GRAVCAPTURE_GAS) || (SINK_GRAVACCRETION == 8)
    double        mass;         /* active sink mass (boundedness/Bondi paths) */
#endif
#if defined(SINK_GRAVCAPTURE_FIXEDSINKRADIUS)
    double        sink_radius;
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
    Vec3<double>  sink_angmom;  /* Sink_Specific_AngMom */
#endif
    int           origin_local_idx;  /* index into requester's nl_outs[] */
    int           origin_rank;       /* requester's MPI rank */
};

#endif /* SINK_PARTICLES */

#endif /* SINK_ENV1_TYPES_H */
