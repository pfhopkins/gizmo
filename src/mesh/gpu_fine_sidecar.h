#ifndef GPU_FINE_SIDECAR_H
#define GPU_FINE_SIDECAR_H

/* Device-resident sidecar for the bounded fine-tree receiver walk (Candidate L,
 * OPEN_topleaf_router_design.md §45).  Mirrors the host receiver's exact supply
 * substrate to the device so the Step-3 device walk never reads scattered host
 * P[]/managed arrays:
 *   d_supply_{x,y,z}  DOUBLE  supply positions (built from P[pool[p]].Pos)
 *   d_supply_h        DOUBLE  supply reach = gx_policy_scaled_h(j) (NO SIDX slack;
 *                             NEVER the host float compact_xyzh)
 *   d_supply_type     int     supply particle Type (for the supply-mask reduction)
 *   d_j_to_pool       int     [NumPart]  absolute P-index -> pool slot / -1
 *   d_fine_band       DOUBLE  [nnodes*FINEBAND_NTYPES]  per-node per-type opener band
 *
 * PASSIVE / ORACLE-ONLY at S2a: the arrays are allocated + uploaded + verifiable,
 * with NO consumer yet (the device walk is S2b).  Broadcast stays authoritative.
 * The host owner (ghost_exchange.cc) stages every host buffer with its own SSOT
 * (gx_policy_scaled_h, P[]), so the device reach == the host receiver reach by
 * construction; this TU only owns the device memory + deep_copy + readback.
 *
 * Temporary validation scaffolding (teardown ledger §39). */

#ifdef __cplusplus
extern "C" {
#endif

/* Freshness/layout key.  ANY field mismatch at consume time => the sidecar is
 * UNAVAILABLE and the caller falls back to authoritative broadcast (fail-closed).
 * MaxPart / nnodes are named explicitly because d_fine_band indexes
 * (no-MaxPart)*FINEBAND_NTYPES+t and d_j_to_pool is [NumPart]; a layout ambiguity
 * would mis-index the band. */
struct gx_fine_sidecar_key_t {
    int          numpart;          /* NumPart at build */
    int          maxpart;          /* All.MaxPart */
    int          numnodestree;     /* Numnodestree */
    int          fb_maxpart;       /* g_fineband.maxpart (band index base) */
    int          fb_nnodes;        /* g_fineband.nnodes  (band index range) */
    int          num_pool;         /* supply pool size */
    unsigned int eligible_mask;    /* supply-cache eligible type mask */
    int          radius_policy;    /* caller reach policy */
    double       j_scale;          /* caller j_radius_scale */
    double       safety;           /* ghost safety factor */
    long         treebuild_gen;    /* force_treebuild_generation() */
    long         hmax_refresh_gen; /* force_hmax_refresh_generation() */
    long long    ti;               /* All.Ti_Current */
    long long    pool_ti;          /* supply-cache epoch (g_fineband.pool_ti / ti_when_built) */
    /* Gravity-SoA node-geometry drift freshness.  Step 3 owns
     * gpu_gravity_soa_ensure_drifted(All.Ti_Current) + a soa_drift_ti stamp; until
     * then this stays a sentinel that never certifies, so the sidecar fails closed
     * to broadcast (no device walk may trust un-drift-certified geometry). */
    long long    soa_drift_ti;
};

/* Sentinel meaning "SoA geometry drift NOT yet certified" (fails the valid check). */
#define GX_FINE_SIDECAR_SOA_DRIFT_UNCERTIFIED (-1LL)

/* (Re)allocate device arrays to fit and deep_copy the host-staged buffers into
 * them; store `key`.  All host pointers are HostSpace/plain host memory owned by
 * the caller.  Returns 0 on success, <0 on allocation/argument failure (leaves the
 * sidecar invalid). */
int gpu_fine_sidecar_upload(const double *sx, const double *sy, const double *sz,
                            const double *sh, const int *stype, int num_pool,
                            const int *j_to_pool, int numpart,
                            const double *band, long band_len,
                            const struct gx_fine_sidecar_key_t *key);

/* Free all device arrays; mark invalid. */
void gpu_fine_sidecar_free(void);

/* 1 iff the sidecar is populated AND every key field matches `want` AND the
 * SoA-drift stamp certifies `want->soa_drift_ti` (not the uncertified sentinel). */
int gpu_fine_sidecar_is_valid(const struct gx_fine_sidecar_key_t *want);

/* Diagnostic readback: copy the device arrays back into caller host buffers, each
 * sized as uploaded (num_pool / numpart / band_len).  Any size disagreement with
 * the stored key => <0 (no copy).  Returns 0 on success.  For the gated S2a verify
 * only. */
int gpu_fine_sidecar_readback(double *sx, double *sy, double *sz, double *sh,
                              int *stype, int num_pool,
                              int *j_to_pool, int numpart,
                              double *band, long band_len);

#ifdef __cplusplus
}
#endif

#endif /* GPU_FINE_SIDECAR_H */
