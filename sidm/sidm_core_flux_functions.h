/* sidm_core_flux_functions.h -- per-pair SIDM scattering decision + kick.
 *
 * Replaces the fragment sidm/sidm_core_flux_computation.h. Body guarded by
 * DM_SIDM so the caller (ags_rkern.cc AGSForce_evaluate) can invoke
 * unconditionally.
 *
 * SIDM scatter is a genuine two-sided physics event: when a scatter occurs,
 * both i and j receive momentum kicks. The i-side update goes into the caller
 * out struct directly; the j-side delta is returned in SidmScatterResult and
 * the caller applies it atomically (OMP atomic today on the CPU tree-walk;
 * will switch to Kokkos::atomic in the B2 AGSForce GPU port).
 *
 * RNG: currently uses the host-side GSL generator via gsl_rng_uniform and
 * calculate_interact_kick. The B2 port will migrate these to the counter-based
 * GPU RNG (declarations/gpu_rng.h). For now the structural port preserves the
 * exact GSL stream so behaviour is bit-identical to the fragment.
 *
 * Requires allvars.h / proto.h and sidm/sidm_core.h for
 * prob_of_interaction / prob_of_grain_interaction / calculate_interact_kick.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef SIDM_CORE_FLUX_FUNCTIONS_H
#define SIDM_CORE_FLUX_FUNCTIONS_H

#include "../declarations/allvars.h"
#include "../declarations/gpu_rng.h"
#include "sidm_helper_functions.h"
#ifdef GRAIN_COLLISIONS
#include "../solids/grain_helper_functions.h"
#endif

struct SidmScatterResult {
    int scattered;          /* 1 if a scatter occurred this pair */
    Vec3<double> dv_sidm;   /* to add to P[j].Vel[k] (atomic); P[j].dp[k] += dv_sidm[k]*P[j].Mass */
    int set_wakeup_j;       /* 1 if P[j].wakeup should be set to -1 */
};

template <typename LocalT, typename KernelT, typename OutT>
KOKKOS_INLINE_FUNCTION
SidmScatterResult sidm_core_flux_compute_pair(
    const LocalT &local,
    int j,
    struct particle_data *P,
    const KernelT &kernel,
    OutT &out,
    const MyDouble *geofactor_table)
{
    SidmScatterResult r;
    r.scattered = 0;
    r.dv_sidm = {0, 0, 0};
    r.set_wakeup_j = 0;
#ifdef DM_SIDM
    double Pj_dtime = get_particle_timestep_in_physical(j);
    if(!( ((1 << local.Type) & (DM_SIDM)) && ((1 << P[j].Type) & (DM_SIDM))
          && (local.ID != P[j].ID) && (local.dtime <= Pj_dtime) )) {
        return r;
    }
    /* ensure each pair is computed only once */
    if((local.dtime == Pj_dtime) && (local.ID > P[j].ID)) { return r; }

    double h_si = 0.5 * (kernel.h_i + kernel.h_j);
    double m_si = 0.5 * (local.Mass + P[j].Mass);
    Vec3<double> dv_local = kernel.dv;
#ifdef GRAIN_COLLISIONS
    double prob = prob_of_grain_interaction_tab(local.Grain_CrossSection_PerUnitMass, local.Mass, kernel.r, h_si, dv_local, local.dtime, j, P, geofactor_table);
#else
    double prob = prob_of_interaction_tab(m_si, kernel.r, h_si, dv_local, local.dtime, geofactor_table);
#endif
    if(prob > 0.2) { out.dtime_sidm = DMIN(out.dtime_sidm, local.dtime * (0.2 / prob)); }

    /* counter-based RNG, symmetric (i,j) key: both sides of the pair see the
       same stream. Counter = Ti_Current << 8 | tag_nibble (0 = threshold,
       1 = scatter direction — see declarations/gpu_rng.h). */
    uint64_t rng_key = (uint64_t)local.ID ^ (uint64_t)P[j].ID;
    uint64_t rng_ctr_threshold = ((uint64_t)All.Ti_Current << 8) | 0;
    uint64_t rng_ctr_direction = ((uint64_t)All.Ti_Current << 8) | 1;
    if(gizmo_gpu_rand_double(rng_key, rng_ctr_threshold) >= prob) { return r; }

    /* scatter happens */
    r.scattered = 1;
    if(!(TimeBinActive[P[j].TimeBin])) {
        if(WAKEUP * local.dtime < Pj_dtime) { r.set_wakeup_j = 1; }
    }
    Vec3<double> kick;
    calculate_interact_kick_rng(dv_local, kick, m_si, rng_key, rng_ctr_direction);
    for(int k=0; k<3; k++) {
        double dv_sidm = (local.Mass / m_si) * kick[k];
        out.sidm_kick[k] -= (P[j].Mass / m_si) * kick[k];
        r.dv_sidm[k] = dv_sidm;
    }
    out.si_count++;
#else
    (void)local; (void)j; (void)P; (void)kernel; (void)out;
#endif /* DM_SIDM */
    return r;
}

#endif /* SIDM_CORE_FLUX_FUNCTIONS_H */
