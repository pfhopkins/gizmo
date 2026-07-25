/* sidm_core_flux_functions.h -- per-pair SIDM scattering decision + kick.
 *
 * KOKKOS_INLINE_FUNCTION; guarded by DM_SIDM so the sole caller
 * (AgsForceSpec::pair_kernel in gravity/ags_force_loop.h) can invoke
 * unconditionally.
 *
 * SIDM scatter is a genuine two-sided physics event: when a scatter occurs,
 * both i and j receive momentum kicks. The i-side update goes into the caller
 * out struct (merged via AccumData); the j-side delta is returned in
 * SidmScatterResult and applied by the caller via Kokkos::atomic_add on
 * P[j].Vel / P[j].dp (and atomic_max on P[j].wakeup).
 *
 * RNG: counter-based gizmo_gpu_rand_double keyed on (local.ID ^ P[j].ID),
 * with a per-loop salt XOR-mixed into the counter (see gpu_rng.h
 * gizmo_loop_rng_salt). Identical statistics and identical streams on CPU
 * and GPU; no GSL.
 *
 * Helpers (prob_of_interaction_tab, prob_of_grain_interaction_tab,
 * calculate_interact_kick_rng) live in sidm/sidm_helper_functions.h and
 * solids/grain_helper_functions.h.
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
#ifdef GRAIN_EVOLUTION
#include "../solids/grain_evolution_functions.h"
#endif

struct SidmScatterResult {
    int scattered;          /* 1 if a scatter occurred this pair */
    Vec3<double> dv_sidm;   /* to add to P[j].Vel[k] (atomic); P[j].dp[k] += dv_sidm[k]*P[j].Mass */
    int set_wakeup_j;       /* 1 if P[j].wakeup should be set to -1 */
};

/* `rng_salt` is REQUIRED — a per-loop FNV-1a hash from gizmo_loop_rng_salt()
 * (see declarations/gpu_rng.h). It is XOR-mixed into the counter so two
 * different loops drawing SIDM scatter on the same (Ti_Current, ID_i^ID_j)
 * pair pull independent streams. No default value — every caller must
 * supply a named loop salt. Today's sole caller is the ags_force runner
 * (AGS_FORCE_RNG_SALT in gravity/ags_force_loop.h). */
template <typename LocalT, typename KernelT, typename OutT>
KOKKOS_INLINE_FUNCTION
SidmScatterResult sidm_core_flux_compute_pair(
    const LocalT &local,
    int j,
    struct particle_data *P,
    const KernelT &kernel,
    OutT &out,
    const MyDouble *geofactor_table,
    const int *timebin_active,
    uint64_t rng_salt)
{
    SidmScatterResult r;
    r.scattered = 0;
    r.dv_sidm = {0, 0, 0};
    r.set_wakeup_j = 0;
#ifdef DM_SIDM
    double Pj_dtime = get_particle_timestep_in_physical(j, P);
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

    /* Counter-based RNG, symmetric (i,j) key: both sides of the pair see the
       same stream. Timestep, loop-domain salt, and draw tag are mixed into
       one counter; the tag only needs to distinguish the two SIDM draws
       (0 = threshold, 1 = scatter direction). Ti_Current is left-shifted
       to keep timestep in the high half of the counter; the salt and tag
       occupy the rest. */
    uint64_t rng_key = (uint64_t)local.ID ^ (uint64_t)P[j].ID;
    uint64_t ti      = ((uint64_t)All.Ti_Current) << 32;
    uint64_t rng_ctr_threshold = ti ^ rng_salt ^ UINT64_C(0);
    uint64_t rng_ctr_direction = ti ^ rng_salt ^ UINT64_C(1);
    if(gizmo_gpu_rand_double(rng_key, rng_ctr_threshold) >= prob) { return r; }

    /* scatter happens */
    r.scattered = 1;
    if(!(timebin_active[P[j].TimeBin])) {
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

#if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & 7)
    /* Pairwise grain-evolution outcomes (COAG/FRAG/SHAT). The SIDM scatter
     * machinery already handles "does a collision fire" via the prob
     * gate; this resolver decides what HAPPENS to grain mass + size when
     * a collision fires, by comparing |dv| to per-species thresholds. */
    grain_evolution_resolve_pairwise(local, j, P, dv_local, out);
#endif
#else
    (void)local; (void)j; (void)P; (void)kernel; (void)out;
    (void)geofactor_table; (void)timebin_active; (void)rng_salt;
#endif /* DM_SIDM */
    return r;
}

#endif /* SIDM_CORE_FLUX_FUNCTIONS_H */
