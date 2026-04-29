/*! \file let_pack.cc
 *  \brief Step 13 Phase 9.1c-e -- Locally Essential Tree (LET) pack, exchange,
 *         and unpack.  Replaces the iterative gravity export loop with a
 *         one-shot subtree exchange.
 *
 *  Flow per gravity_tree() invocation (after force_treebuild and
 *  force_exchange_pseudodata):
 *
 *    let_run_exchange()
 *      1. let_compute_local_payload  -- our bbox + worst-case opening bounds
 *      2. let_exchange_payloads       -- MPI_Allgather payloads to all ranks
 *      3. let_pack_for_rank(R) for each remote R
 *           recurse our local tree from each topnode; ship every node
 *           let_node_essential_for_rank() flags as "could be opened by some
 *           particle in R".  At leaves, synthesize single-particle NODEs
 *           covering all #ifdef payloads (mirrors force_update_node_recursive
 *           single-particle accumulation, forcetree.cc:752-861).  Edge
 *           pointers (sibling/nextnode that exit the shipped subtree) are
 *           encoded via LET_EDGE_SENTINEL so the unpack can rewrite them to
 *           the local topleaf continuation.
 *      4. let_exchange_nodes          -- MPI_Alltoall counts + MPI_Alltoallv payloads
 *      5. let_unpack_and_install      -- copy NODE+extNODE bytes into the
 *           foreign slot range [MaxPart+MaxNodes, MaxPart+MaxNodes+Numforeignnodes);
 *           build remote_id->local_foreign_idx translation table; remap
 *           intra-subtree pointers; rewrite each affected local topleaf's
 *           u.d.nextnode to point at the foreign subtree root.
 *
 *  Buffer-overflow policy (Phase 9.0): if Numforeignnodes would exceed
 *  MaxForeignNodes, endrun() with the LETAllocFactor restart message.
 *  Future option (b) -- graceful shrink + temporary fallback to legacy export
 *  -- documented but not implemented unless practical memory limits require.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../system/gpu_particles_arena.h"  /* gpu_particles_arena_invalidate */
#include "let_data.h"
#include "gpu_pseudo_update.h"  /* gpu_scatter_foreign_to_soa */

#ifdef OPENMP_GPU_OFFLOAD

/* Sentinel encoding for out-of-subtree sibling pointers:
 *
 *   v == LET_EDGE_SENTINEL_BASE                    -- legacy / nextnode of non-essential nodes:
 *                                                     resolves to topleaf_sibling (plain exit).
 *   v <  LET_EDGE_SENTINEL_BASE  (encoded)          -- sibling pointer exits this recursion level.
 *                                                     The target node index is:
 *                                                       orig_sib = LET_EDGE_SENTINEL_BASE - v
 *                                                     Receiver looks up orig_sib in the wire map.
 *                                                     If found  → slot_base + wire_map[orig_sib]
 *                                                     If absent → topleaf_sibling (genuine exit).
 *
 * Rule: the "last child sibling" sentinel in pack_recurse and the top-level loop must encode the
 * sib_terminator so the receiver can follow the sibling chain across recursion levels.
 * Never-followed nextnode sentinels (non-essential nodes, synthesized leaves) stay as
 * LET_EDGE_SENTINEL_BASE (plain), since they're never decoded.
 */
#define LET_EDGE_SENTINEL_BASE         (-1000000)
#define LET_EDGE_SENTINEL_ENCODE(orig) (LET_EDGE_SENTINEL_BASE - (orig))
#define LET_EDGE_SENTINEL_IS_ENCODED(v)  ((v) < LET_EDGE_SENTINEL_BASE)
#define LET_EDGE_SENTINEL_DECODE(v)      (LET_EDGE_SENTINEL_BASE - (v))

/* ----------------------------------------------------------------------
 * Phase 9.5: active-only LET helpers
 *
 * Each rank computes a per-topleaf bitmap (one bit per topleaf, NTopleaves
 * total) where bit tl == 1 iff topleaf tl is owned by ThisTask AND at least
 * one particle in ActiveParticleList lives in tl.  Bitmaps are MPI_Allgathered
 * so every rank sees every other rank's active-topleaf set.  In the pack
 * loop, sender S short-circuits packing for receiver R when R's bitmap is
 * all zero (R has no active particles -> R does not need any LET data).
 *
 * Particle -> topleaf lookup uses Father[i] -> ancestor chain until the
 * first BITFLAG_TOPLEVEL node, which is necessarily a topleaf (only
 * topleaves contain particles; internal topnodes have only topnode children).
 * ---------------------------------------------------------------------- */
static inline int let_bitmap_word_count(int n_topleaves)
{
    return (n_topleaves + 63) / 64;
}

static inline int let_bitmap_test(const uint64_t *b, int tl)
{
    return (int) ((b[tl >> 6] >> (tl & 63)) & 1ULL);
}

static inline int let_bitmap_any_set(const uint64_t *b, int n_words)
{
    if(!b) return 1;  /* NULL bitmap -> conservative: assume some bits set */
    for(int w = 0; w < n_words; w++) if(b[w]) return 1;
    return 0;
}

extern "C" void let_compute_local_active_bitmap(uint64_t *bitmap, int n_words)
{
    memset(bitmap, 0, (size_t)n_words * sizeof(uint64_t));
    if(NTopleaves <= 0) return;

    /* Build inverse lookup: Nodes[no] -> topleaf index (or -1) for MY topleaves only. */
    int *my_tl_lookup = (int *) mymalloc("LET_my_tl_lookup", (size_t)MaxNodes * sizeof(int));
    for(int j = 0; j < MaxNodes; j++) my_tl_lookup[j] = -1;
    for(int t = 0; t < NTopleaves; t++)
    {
        if(DomainTask[t] != ThisTask) continue;
        int no = DomainNodeIndex[t];
        if(no >= All.MaxPart && no < All.MaxPart + MaxNodes)
            my_tl_lookup[no - All.MaxPart] = t;
    }

    /* Fallback: if ActiveParticleList is empty (e.g. tree rebuilt before
     * make_list_of_active_particles ran for the current step), set all MY
     * topleaves' bits.  Preserves Phase 9.4 conservative behavior. */
    if(ActiveParticleList.empty())
    {
        for(int t = 0; t < NTopleaves; t++)
        {
            if(DomainTask[t] != ThisTask) continue;
            bitmap[t >> 6] |= (1ULL << (t & 63));
        }
        myfree(my_tl_lookup);
        return;
    }

    for(size_t k = 0; k < ActiveParticleList.size(); k++)
    {
        int i = ActiveParticleList[k];
        if(i < 0 || i >= NumPart) continue;
        if(P[i].Mass <= 0) continue;
        int no = Father[i];
        int guard = 0;
        while(no >= All.MaxPart && no < All.MaxPart + MaxNodes && guard++ < 1024)
        {
            int tl = my_tl_lookup[no - All.MaxPart];
            if(tl >= 0)
            {
                bitmap[tl >> 6] |= (1ULL << (tl & 63));
                break;
            }
            no = Nodes[no].u.d.father;
        }
    }
    myfree(my_tl_lookup);
}

/* ----------------------------------------------------------------------
 * Step 1: per-rank-payload computation
 * ---------------------------------------------------------------------- */
extern "C" void let_compute_local_payload(struct LETPerRankPayload *out,
                                          const uint64_t *active_bitmap,
                                          int bitmap_n_words)
{
    /* bbox: union of OUR topleaf bboxes (each topleaf's [center-len/2, center+len/2]).
     * This is tighter than the union of particle positions and matches what the
     * walk's min_dist check effectively bounds.  If active_bitmap is non-NULL,
     * restrict to ACTIVE topleaves only (Phase 9.5 tight mode). */
    out->bbox_min[0] = out->bbox_min[1] = out->bbox_min[2] = DBL_MAX;
    out->bbox_max[0] = out->bbox_max[1] = out->bbox_max[2] = -DBL_MAX;
    int found_any = 0;
    for(int i = 0; i < NTopleaves; i++)
    {
        if(DomainTask[i] != ThisTask) continue;
        if(active_bitmap && !let_bitmap_test(active_bitmap, i)) continue;
        int no = DomainNodeIndex[i];
        if(no < All.MaxPart || no >= All.MaxPart + MaxNodes) continue;
        double cx = (double) Nodes[no].center[0];
        double cy = (double) Nodes[no].center[1];
        double cz = (double) Nodes[no].center[2];
        double half = 0.5 * (double) Nodes[no].len;
        if(cx - half < out->bbox_min[0]) out->bbox_min[0] = cx - half;
        if(cy - half < out->bbox_min[1]) out->bbox_min[1] = cy - half;
        if(cz - half < out->bbox_min[2]) out->bbox_min[2] = cz - half;
        if(cx + half > out->bbox_max[0]) out->bbox_max[0] = cx + half;
        if(cy + half > out->bbox_max[1]) out->bbox_max[1] = cy + half;
        if(cz + half > out->bbox_max[2]) out->bbox_max[2] = cz + half;
        found_any = 1;
    }
    if(!found_any)
    {
        /* No local topleaves.  Use degenerate bbox at origin; min_dist will
         * be large for all nodes, so essential check returns 0 for everything,
         * meaning we ship nothing -- correct behavior for a rank with no
         * particles. */
        for(int k = 0; k < 3; k++) {out->bbox_min[k] = out->bbox_max[k] = 0.0;}
    }

    /* Per-particle bounds.  If active_bitmap is non-NULL, scan only
     * ActiveParticleList (Phase 9.5 tight mode).  Otherwise scan all NumPart
     * (Phase 9.4 conservative mode) -- safer for RT/TREECOL which may run
     * on non-active particles. */
    out->min_OldAcc = DBL_MAX;
    for(int t = 0; t < 6; t++) out->max_soft_by_type[t] = 0.0;
    out->has_sink = 0;

    int n_iter = (active_bitmap && !ActiveParticleList.empty()) ? (int) ActiveParticleList.size() : NumPart;
    for(int kk = 0; kk < n_iter; kk++)
    {
        int i = (active_bitmap && !ActiveParticleList.empty()) ? ActiveParticleList[kk] : kk;
        if(i < 0 || i >= NumPart) continue;
        if(P[i].Mass <= 0) continue;
        int t = P[i].Type;
        if(t < 0 || t > 5) continue;

        /* min(OldAcc) -- relative-criterion worst case.  Skip zeros (uninitialised
         * particles or first-step where OldAcc isn't yet set) to avoid biasing the
         * min toward 0, which would force us to ship every node.  If ALL particles
         * have OldAcc==0 (first step), out->min_OldAcc stays DBL_MAX and the
         * relative check below will skip; we fall back to BH+softening only,
         * which is conservative. */
        double oa = (double) P[i].OldAcc;
        if(oa > 0 && oa < out->min_OldAcc) out->min_OldAcc = oa;

        /* max softening kernel radius per type */
        double soft = (double) ForceSoftening_KernelRadius(i);
        if(soft > out->max_soft_by_type[t]) out->max_soft_by_type[t] = soft;

        if(t == 5) out->has_sink = 1;
    }
    if(out->min_OldAcc == DBL_MAX) out->min_OldAcc = 0.0;  /* no positive OldAcc; relative check disabled */

    /* Use relative criterion?  Standard GIZMO: relative is on iff All.ErrTolTheta == 0.
     * Under GRAVITY_HYBRID_OPENING_CRIT it's also suppressed at first step. */
#ifdef GRAVITY_HYBRID_OPENING_CRIT
    out->use_rel_crit = (All.Ti_Current > 0 || RestartFlag == 1) ? 1 : 0;
#else
    out->use_rel_crit = (All.ErrTolTheta == 0) ? 1 : 0;
#endif
}

/* ----------------------------------------------------------------------
 * Step 2: payload exchange (Allgather)
 * ---------------------------------------------------------------------- */
extern "C" int let_exchange_payloads(const struct LETPerRankPayload *local,
                                      struct LETPerRankPayload *all_ranks)
{
    return MPI_Allgather(local, sizeof(struct LETPerRankPayload), MPI_BYTE,
                         all_ranks, sizeof(struct LETPerRankPayload), MPI_BYTE,
                         MPI_COMM_WORLD);
}

/* ----------------------------------------------------------------------
 * Essential-node check (worst-case opening criterion for "any particle in R")
 *
 * Returns 1 if SOME particle in R might open this node (must ship + recurse).
 * Returns 0 if NO particle in R will open this node (ship as multipole-only OR
 * skip).  See handoff_step13_phase9_walk_audit.md for the 20-criterion derivation.
 * ---------------------------------------------------------------------- */
static int let_node_essential_for_rank(double cx, double cy, double cz,
                                       double len, double mass, double maxsoft,
                                       const struct LETPerRankPayload *p)
{
    double r2 = let_point_to_bbox_dist_sq(p->bbox_min, p->bbox_max, cx, cy, cz);

    /* BH criterion: open if len^2 > r^2 * theta^2 */
    if(All.ErrTolTheta != 0)
    {
        double theta2 = All.ErrTolTheta * All.ErrTolTheta;
        if(len * len > r2 * theta2) return 1;
    }

    /* Relative criterion: open if M*len^2 > r^4 * OldAcc * ErrTolForceAcc / G_factor.
     * GIZMO's relative form uses aold = ErrTolForceAcc * OldAcc.  Smaller aold ->
     * more permissive opening, so over-include with min(OldAcc) over R. */
    if(p->use_rel_crit)
    {
        if(p->min_OldAcc > 0)
        {
            double aold_min = p->min_OldAcc * All.ErrTolForceAcc;
            double lhs = mass * len * len;
            double rhs = r2 * r2 * aold_min;
            if(lhs > rhs) return 1;
        }
        else
        {
            /* No positive OldAcc info -- conservative: always essential under relative crit */
            return 1;
        }
    }

    /* Softening criterion (target-side): open if r < (target_soft + 0.6 * len).
     * Use max softening over types in R for max permissiveness. */
    double max_soft_R = 0.0;
    for(int t = 0; t < 6; t++)
    {
        if(p->max_soft_by_type[t] > max_soft_R) max_soft_R = p->max_soft_by_type[t];
    }
    double soft_check = max_soft_R + 0.6 * len;
    if(r2 < soft_check * soft_check) return 1;

    /* Node-side maxsoft criterion: open if r < (node_maxsoft + 0.6 * len) */
    double node_soft_check = maxsoft + 0.6 * len;
    if(r2 < node_soft_check * node_soft_check) return 1;

#ifdef PMGRID
    /* PM short-range: walk skips nodes outside rcut.  For LET, ship if within
     * rcut + 0.5*len (walk would touch).  Beyond that, walk skips -- don't ship. */
    double rcut = (double) All.Rcut[0];
#ifdef PM_PLACEHIGHRESREGION
    if((double) All.Rcut[1] > rcut) rcut = (double) All.Rcut[1];
#endif
    double pm_check = rcut + 0.5 * len;
    if(r2 >= pm_check * pm_check) return 0;  /* outside PM short-range; walk would skip */
#endif

#if (defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES)) && defined(SINGLE_STAR_DIRECT_GRAVITY_RADIUS)
    if(p->has_sink)
    {
        double ssdgr_check = SINGLE_STAR_DIRECT_GRAVITY_RADIUS + 0.6 * len;
        if(r2 < ssdgr_check * ssdgr_check) return 1;
    }
#endif

    return 0;
}

/* ----------------------------------------------------------------------
 * Single-particle leaf NODE/extNODE synthesis.
 *
 * Mirrors the single-particle accumulation in force_update_node_recursive
 * (forcetree.cc:752-861) under the simplification mass = particle.Mass
 * (so all mass-weighted divisions degenerate to identity).  Sets BITFLAG_
 * MULTIPLEPARTICLES so the receiver's walk uses the synthesized multipole
 * directly (exact for single particle) instead of skipping to a non-existent
 * source-side particle.
 *
 * Edge pointers (sibling/nextnode) start as LET_EDGE_SENTINEL_BASE; the
 * surrounding pack_recurse caller updates them based on the position of
 * this leaf in the iteration chain.
 * ---------------------------------------------------------------------- */
static void let_synthesize_particle_leaf(int p_idx, int sib_terminator_sentinel,
                                          struct LETNodeWire *w)
{
    memset(w, 0, sizeof(struct LETNodeWire));

    w->remote_id = -1 - p_idx;  /* negative encoding distinguishes synthesized particle leaves
                                  * from real internal nodes (whose remote_id >= MaxPart).  Unpack
                                  * uses remote_id < 0 to recognize synthesized leaves -- they
                                  * still get a foreign slot and remap entry, but their pointer
                                  * fields are simpler (no inbound references except from parent). */

    struct particle_data *pa = &P[p_idx];
    MyFloat mass = (MyFloat) pa->Mass;
    Vec3<MyFloat> pos = {(MyFloat) pa->Pos[0], (MyFloat) pa->Pos[1], (MyFloat) pa->Pos[2]};
    Vec3<MyFloat> vel = {(MyFloat) pa->Vel[0], (MyFloat) pa->Vel[1], (MyFloat) pa->Vel[2]};

    w->node.center = pos;
    w->node.len = 0;     /* zero size -- never opens, always treated as multipole */
    w->node.u.d.s = pos;
    w->node.u.d.mass = mass;
    /* CRITICAL: BITFLAG_MULTIPLEPARTICLES=1 forces the walk to use the multipole
     * (exact for single particle).  Without it, walk skips and tries to access
     * the source-side particle, which doesn't exist on the receiver. */
    w->node.u.d.bitflags = (1u << BITFLAG_MULTIPLEPARTICLES);
    w->node.u.d.sibling = sib_terminator_sentinel;
    w->node.u.d.nextnode = sib_terminator_sentinel;
    w->node.u.d.father = -1;  /* foreign nodes have no father in OUR tree */
    w->node.GravCost = 0;
    w->node.Ti_current = All.Ti_Current;
    w->node.N_part = 1;
    w->node.maxsoft = (MyFloat) ForceSoftening_KernelRadius(p_idx);
#ifdef SINGLE_STAR_SINK_DYNAMICS
    if(pa->Type == 5 && pa->KernelRadius > w->node.maxsoft) w->node.maxsoft = (MyFloat) pa->KernelRadius;
#endif

    /* Optional payload fields, mirroring forcetree.cc:752-861 single-particle contribution */

#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    if(pa->Type == 0) w->node.gasmass = mass;
#if defined(SINK_ALPHADISK_ACCRETION) && defined(RT_USE_TREECOL_FOR_NH)
    if(pa->Type == 5) w->node.gasmass = (MyFloat) P[p_idx].Sink_Mass_Reservoir;
#endif
#endif

#ifdef COSMIC_RAY_SUBGRID_LEBRON
    w->node.cr_injection = (MyFloat) cr_get_source_injection_rate(p_idx, P, CellP);
#endif

#ifdef RT_USE_GRAVTREE
    {
        double lum[N_RT_FREQ_BINS];
#ifdef CHIMES_STELLAR_FLUXES
        double chimes_lum_G0[CHIMES_LOCAL_UV_NBINS];
        double chimes_lum_ion[CHIMES_LOCAL_UV_NBINS];
        int active_check = rt_get_source_luminosity_chimes(p_idx, 1, lum, chimes_lum_G0, chimes_lum_ion, P, CellP);
#else
        int active_check = rt_get_source_luminosity(p_idx, 1, lum, P, CellP);
#endif
        if(active_check)
        {
            for(int k = 0; k < N_RT_FREQ_BINS; k++) w->node.stellar_lum[k] = (MyFloat) lum[k];
#ifdef CHIMES_STELLAR_FLUXES
            for(int k = 0; k < CHIMES_LOCAL_UV_NBINS; k++)
            {
                w->node.chimes_stellar_lum_G0[k] = chimes_lum_G0[k];
                w->node.chimes_stellar_lum_ion[k] = chimes_lum_ion[k];
            }
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
            /* For a single particle, luminosity-weighted position == particle position;
             * after the divide-by-l_tot the result is just pos. */
            w->node.rt_source_lum_s = pos;
            w->extnode.rt_source_lum_vs = vel;
            w->extnode.rt_source_lum_dp = {};
#endif
        }
        /* If !active_check: stellar_lum stays zero (memset above) -- correct */
    }
#endif

#ifdef SINK_PHOTONMOMENTUM
    if(pa->Type == 5)
    {
        if(pa->Mass > 0 && pa->DensityAroundParticle > 0 && pa->Sink_Mdot > 0)
        {
            double BHLum = sink_lum_bol(pa->Sink_Mdot, pa->Sink_Mass, p_idx);
            w->node.sink_lum = (MyFloat) BHLum;
            /* sink_lum_grad after div-by-sink_lum = the unweighted vector for single particle */
#if defined(SINK_FOLLOW_ACCRETED_ANGMOM)
            w->node.sink_lum_grad[0] = (MyFloat) pa->Sink_Specific_AngMom[0];
            w->node.sink_lum_grad[1] = (MyFloat) pa->Sink_Specific_AngMom[1];
            w->node.sink_lum_grad[2] = (MyFloat) pa->Sink_Specific_AngMom[2];
#else
            w->node.sink_lum_grad[0] = (MyFloat) pa->GradRho[0];
            w->node.sink_lum_grad[1] = (MyFloat) pa->GradRho[1];
            w->node.sink_lum_grad[2] = (MyFloat) pa->GradRho[2];
#endif
        }
        /* else: sink_lum stays 0 (memset); walk's sink_lum > 0 check filters out */
    }
#endif

#ifdef SINK_CALC_DISTANCES
    if(pa->Type == SPECIAL_POINT_TYPE_FOR_NODE_DISTANCES)
    {
        w->node.sink_mass = mass;
        w->node.sink_pos = pos;  /* pos*mass / mass = pos */
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
        w->node.N_SINK = 1;
        w->node.sink_vel = vel;
#ifdef SPECIAL_POINT_MOTION
        w->node.sink_acc[0] = (MyFloat) pa->Acc_Total_PrevStep[0];
        w->node.sink_acc[1] = (MyFloat) pa->Acc_Total_PrevStep[1];
        w->node.sink_acc[2] = (MyFloat) pa->Acc_Total_PrevStep[2];
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
        w->node.MaxFeedbackVel = (MyFloat) pa->MaxFeedbackVel;
#endif
#endif
    }
#endif

#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    for(int k = 0; k < 6; k++)
        w->node.tidal_tensorps_prevstep.data[k] = (MyFloat) pa->tidal_tensorps_prevstep.data[k];
#endif

#ifdef DM_SCALARFIELD_SCREENING
    if(pa->Type != 0)
    {
        w->node.mass_dm = mass;
        w->node.s_dm = pos;
        w->extnode.vs_dm = vel;
        w->extnode.dp_dm = {};
    }
#endif

    /* extnode core fields */
    w->extnode.dp = {};
    w->extnode.vs = vel;
    w->extnode.vmax = (MyFloat) fmax(fabs(pa->Vel[0]), fmax(fabs(pa->Vel[1]), fabs(pa->Vel[2])));
    if(pa->Type == 0)
    {
        double htmp = (double) pa->KernelRadius;
        if(htmp > (double) All.MaxKernelRadius) htmp = (double) All.MaxKernelRadius;
        w->extnode.hmax = (MyFloat) htmp;
        w->extnode.divVmax = (MyFloat) pa->Particle_DivVel;
    }
    else
    {
        w->extnode.hmax = 0;
        w->extnode.divVmax = 0;
    }
    w->extnode.Ti_lastkicked = All.Ti_Current;
    w->extnode.Flag = 0;
}

/* ----------------------------------------------------------------------
 * Step 3: pack -- recursive walk producing LETNodeWire array for one rank
 *
 * Buffer growth: amortized doubling.  Caller passes (buf, count, capacity);
 * pack functions grow buf in place via realloc.
 * ---------------------------------------------------------------------- */
static void grow_wire_buf(struct LETNodeWire **buf, int needed, int *capacity)
{
    if(needed <= *capacity) return;
    int new_cap = (*capacity == 0) ? 1024 : *capacity;
    while(new_cap < needed) new_cap *= 2;
    struct LETNodeWire *nb = (struct LETNodeWire *) realloc(*buf, (size_t)new_cap * sizeof(struct LETNodeWire));
    if(!nb)
    {
        printf("LET pack: realloc failed (cap=%d, sizeof=%zu, total=%g MB)\n",
               new_cap, sizeof(struct LETNodeWire),
               (double)new_cap * sizeof(struct LETNodeWire) / (1024.0*1024.0));
        endrun(914010);
    }
    *buf = nb;
    *capacity = new_cap;
}

static void grow_hdr_buf(struct LETSubtreeHeader **buf, int needed, int *capacity)
{
    if(needed <= *capacity) return;
    int new_cap = (*capacity == 0) ? 16 : *capacity;
    while(new_cap < needed) new_cap *= 2;
    struct LETSubtreeHeader *nb = (struct LETSubtreeHeader *) realloc(*buf, (size_t)new_cap * sizeof(struct LETSubtreeHeader));
    if(!nb)
    {
        printf("LET pack: hdr realloc failed (cap=%d)\n", new_cap);
        endrun(914011);
    }
    *buf = nb;
    *capacity = new_cap;
}

static void pack_recurse(int no, int sib_terminator,
                          const struct LETPerRankPayload *payload,
                          int subtree_root_topleaf_no,  /* unused; kept for future per-subtree edge encoding */
                          struct LETNodeWire **buf, int *count, int *capacity)
{
    /* Bounds: must be a local internal node */
    if(no < All.MaxPart || no >= All.MaxPart + MaxNodes) return;

    /* Check essential-for-R BEFORE shipping.  If not essential, the walk will
     * close at the parent; we don't need to ship this node either (parent's
     * multipole already covers it). */
    double cx = (double) Nodes[no].center[0];
    double cy = (double) Nodes[no].center[1];
    double cz = (double) Nodes[no].center[2];
    double len = (double) Nodes[no].len;
    double mass = (double) Nodes[no].u.d.mass;
    double maxsoft = (double) Nodes[no].maxsoft;
    int is_essential = let_node_essential_for_rank(cx, cy, cz, len, mass, maxsoft, payload);

    /* Always ship the node (parent expects it).  If not essential, ship as
     * multipole-only (no recursion).  If essential, ship + recurse to children. */
    grow_wire_buf(buf, *count + 1, capacity);
    int my_idx = (*count)++;
    struct LETNodeWire *w = &(*buf)[my_idx];
    w->remote_id = no;
    w->_pad0 = 0;
    w->node = Nodes[no];     /* full struct copy including all #ifdef payloads */
    w->extnode = Extnodes[no];
    /* Edge pointers: tentatively encode as sentinel; updated below if children land within S_R */
    w->node.u.d.sibling = LET_EDGE_SENTINEL_BASE;
    w->node.u.d.nextnode = LET_EDGE_SENTINEL_BASE;
    w->node.u.d.father = -1;  /* foreign nodes have no local father */

    if(!is_essential)
    {
        /* Multipole-only: receiver's walk will close on this node (their criterion
         * matches our worst-case bound), so they never descend.  No recursion. */
        return;
    }

    /* Single-particle leaf in source tree: bitflag=0 means walk would skip
     * to nextnode (a particle).  We can't ship the particle, so override
     * bitflag to MULTIPLEPARTICLES so receiver uses our multipole.  Exact
     * for single particle. */
    if(!(Nodes[no].u.d.bitflags & (1u << BITFLAG_MULTIPLEPARTICLES)))
    {
        w->node.u.d.bitflags |= (1u << BITFLAG_MULTIPLEPARTICLES);
        return;  /* no children to recurse into */
    }

    /* Multi-particle internal node: enumerate children via nextnode/sibling chain.
     * For each child:
     *   - particle (< MaxPart): synthesize a leaf wire
     *   - internal node: recurse
     *   - pseudo (>= MaxPart+MaxNodes+MaxForeignNodes): skip (R has its own access via S->R LET pack)
     *   - foreign (in [MaxPart+MaxNodes, +MaxForeignNodes)): shouldn't appear during pack (we run before unpack)
     *
     * After processing all children, link them into a sibling chain in the WIRE buffer
     * by setting our nextnode to first-child wire-idx, and each child's sibling to
     * next-child wire-idx (last child's sibling = sentinel = "exit subtree"). */
    int first_child_wire_idx = -1;
    int last_child_wire_idx = -1;
    int child = Nodes[no].u.d.nextnode;
    while(child != sib_terminator && child >= 0)
    {
        int next_child;
        int child_wire_idx = -1;

        if(child < All.MaxPart)
        {
            /* Particle leaf -- synthesize */
            grow_wire_buf(buf, *count + 1, capacity);
            child_wire_idx = (*count)++;
            let_synthesize_particle_leaf(child, LET_EDGE_SENTINEL_BASE, &(*buf)[child_wire_idx]);
            next_child = Nextnode[child];  /* particle's next walk target */
        }
        else if(child < All.MaxPart + MaxNodes)
        {
            /* Local internal node -- recurse */
            int child_sib = Nodes[child].u.d.sibling;
            child_wire_idx = *count;
            pack_recurse(child, child_sib, payload, subtree_root_topleaf_no, buf, count, capacity);
            /* If pack_recurse added zero entries (skipped), child_wire_idx == old count;
             * we need to detect that and not link. */
            if(*count == child_wire_idx) child_wire_idx = -1;  /* nothing added */
            next_child = child_sib;
        }
        else if(child < All.MaxPart + MaxNodes + MaxForeignNodes)
        {
            /* Foreign node -- shouldn't happen during pack */
            next_child = Nodes[child].u.d.sibling;
            child_wire_idx = -1;
        }
        else
        {
            /* Pseudo-particle -- skip */
            next_child = Nextnode[child - MaxNodes - MaxForeignNodes];
            child_wire_idx = -1;
        }

        /* Link this child into the sibling chain */
        if(child_wire_idx >= 0)
        {
            if(first_child_wire_idx < 0) first_child_wire_idx = child_wire_idx;
            if(last_child_wire_idx >= 0)
            {
                /* Update prior-last child's sibling to point to this child's wire idx
                 * (we'll convert wire idx -> remote_id in a moment) */
                (*buf)[last_child_wire_idx].node.u.d.sibling = child_wire_idx;
            }
            last_child_wire_idx = child_wire_idx;
        }

        child = next_child;
    }

    /* Set our nextnode to first child (or sentinel if no children shipped) */
    /* (re-fetch &(*buf)[my_idx] -- realloc inside grow_wire_buf may have moved) */
    if(first_child_wire_idx >= 0)
    {
        (*buf)[my_idx].node.u.d.nextnode = first_child_wire_idx;
        if(last_child_wire_idx >= 0)
        {
            /* Last child's sibling encodes sib_terminator so the receiver can find the
             * wire copy of the next sibling at this level (if it was shipped) or fall back
             * to topleaf_sibling (if it wasn't).  See LET_EDGE_SENTINEL_ENCODE. */
            (*buf)[last_child_wire_idx].node.u.d.sibling = LET_EDGE_SENTINEL_ENCODE(sib_terminator);
        }
    }
    /* else: no children shipped (e.g., all were particles outside essential range);
     *       nextnode stays sentinel; receiver will treat as multipole-leaf. */
}

extern "C" int let_pack_for_rank(int R,
                                  const struct LETPerRankPayload *all_ranks,
                                  struct LETNodeWire **out_buf,
                                  int *out_capacity,
                                  struct LETSubtreeHeader **out_hdr_buf,
                                  int *out_hdr_capacity,
                                  int *out_hdr_count,
                                  const uint64_t *receiver_active_bitmap,
                                  int bitmap_n_words)
{
    int count = 0;
    int hdr_count = 0;
    /* Phase 9.5: short-circuit when receiver R has zero active particles.
     * R cannot use any LET nodes we'd ship; skip the entire local-tree walk. */
    if(receiver_active_bitmap && !let_bitmap_any_set(receiver_active_bitmap, bitmap_n_words))
    {
        *out_hdr_count = 0;
        return 0;
    }
    /* Walk the LOCAL tree from each topnode that's NOT in R's domain.  Topnodes
     * in R's domain are R's own data; they won't help R (and would create a
     * self-reference if shipped). */
    /* Iterate via DomainNodeIndex over OUR topleaves, then use the topleaf's
     * subtree (root via nextnode / sibling chain) as the pack starting point.
     * Actually we need to ship from ROOT downward filtering by essential, since
     * R needs to traverse from the root to find what to multipole-vs-open. */

    /* Strategy: walk our local topnode tree from root (Nodes[All.MaxPart]).  For
     * each topleaf owned by US (so R doesn't already have it as pseudo), we
     * enter pack_recurse from that topleaf with sib_terminator = topleaf.sibling. */

    /* NOTE: simpler -- pack each of our local topleaves' subtrees independently.
     * R will see each shipped subtree as the foreign-content for that topleaf
     * (the unpack step rewrites Nodes[topleaf_in_R].u.d.nextnode = subtree_root). */
    for(int i = 0; i < NTopleaves; i++)
    {
        if(DomainTask[i] != ThisTask) continue;
        int topleaf_no = DomainNodeIndex[i];
        if(topleaf_no < All.MaxPart || topleaf_no >= All.MaxPart + MaxNodes) continue;

        int subtree_root = Nodes[topleaf_no].u.d.nextnode;
        if(subtree_root < 0) continue;  /* empty topleaf */
        if(subtree_root == Nodes[topleaf_no].u.d.sibling) continue;  /* topleaf has no descendants */

        int sib_term = Nodes[topleaf_no].u.d.sibling;

        /* Record wire offset BEFORE this topleaf's pack, so we can emit a
         * subtree header covering [wire_offset, *count) on the way out. */
        int wire_offset_before = count;

        /* Walk the topleaf's children via the sibling chain starting from subtree_root.
         * Track first/last wire indices so we can link consecutive children's sibling
         * pointers — without this, the receiver's walk would follow child1.sibling =
         * LET_EDGE_SENTINEL_BASE and exit after child1, missing child2..childN. */
        int first_child_wire_idx = -1;
        int last_child_wire_idx  = -1;
        int child = subtree_root;
        while(child != sib_term && child >= 0)
        {
            int next_child;
            int child_wire_idx = -1;
            if(child < All.MaxPart)
            {
                /* Particle directly under topleaf -- synthesize leaf */
                grow_wire_buf(out_buf, count + 1, out_capacity);
                child_wire_idx = count;
                let_synthesize_particle_leaf(child, LET_EDGE_SENTINEL_BASE, &(*out_buf)[count]);
                count++;
                next_child = Nextnode[child];
            }
            else if(child < All.MaxPart + MaxNodes)
            {
                int child_sib = Nodes[child].u.d.sibling;
                child_wire_idx = count;
                pack_recurse(child, child_sib, &all_ranks[R], topleaf_no, out_buf, &count, out_capacity);
                if(count == child_wire_idx) child_wire_idx = -1;  /* pack_recurse added nothing */
                next_child = child_sib;
            }
            else
            {
                next_child = Nextnode[child - MaxNodes - MaxForeignNodes];
            }
            /* Link this child into the top-level sibling chain */
            if(child_wire_idx >= 0)
            {
                if(first_child_wire_idx < 0) first_child_wire_idx = child_wire_idx;
                if(last_child_wire_idx >= 0)
                    (*out_buf)[last_child_wire_idx].node.u.d.sibling = child_wire_idx;
                last_child_wire_idx = child_wire_idx;
            }
            child = next_child;
        }
        /* Last top-level child: encode sib_term (= topleaf.sibling) so receiver resolves
         * via the wire map (falls back to topleaf_sibling since sib_term is never packed). */
        if(last_child_wire_idx >= 0)
            (*out_buf)[last_child_wire_idx].node.u.d.sibling = LET_EDGE_SENTINEL_ENCODE(sib_term);

        /* Emit subtree header if this topleaf shipped any nodes */
        int subtree_count = count - wire_offset_before;
        if(subtree_count > 0)
        {
            grow_hdr_buf(out_hdr_buf, hdr_count + 1, out_hdr_capacity);
            (*out_hdr_buf)[hdr_count].topleaf_idx = i;
            (*out_hdr_buf)[hdr_count].wire_offset = wire_offset_before;
            (*out_hdr_buf)[hdr_count].count       = subtree_count;
            (*out_hdr_buf)[hdr_count]._pad0       = 0;
            hdr_count++;
        }
    }
    *out_hdr_count = hdr_count;
    return count;
}

/* ----------------------------------------------------------------------
 * Step 4 + 5: MPI exchange + install in one scope.
 *
 * GIZMO mymalloc is a strict LIFO stack.  An earlier draft returned
 * flat_recv / flat_hdr_recv to the caller while freeing intermediates
 * in this function -- that left the recv buffers mid-stack and
 * triggered "Wrong call of myfree(): not the last allocated block!"
 * the moment any caller tried to free anything below them.  Solution:
 * keep the unpack call inside this scope, so all temporaries can be
 * released in strict reverse-alloc order before returning.
 * ---------------------------------------------------------------------- */
extern "C" int let_exchange_nodes(struct LETNodeWire **send_buf_per_rank,
                                   const int *send_count_per_rank,
                                   struct LETSubtreeHeader **send_hdr_per_rank,
                                   const int *send_hdr_count_per_rank)
{
    /* Phase 1: exchange node-counts AND header-counts */
    int *send_counts_int = (int *) mymalloc("LET_send_counts",     NTask * sizeof(int));
    int *recv_counts_int = (int *) mymalloc("LET_recv_counts",     NTask * sizeof(int));
    int *send_hdr_counts = (int *) mymalloc("LET_send_hdr_counts", NTask * sizeof(int));
    int *recv_hdr_counts = (int *) mymalloc("LET_recv_hdr_counts", NTask * sizeof(int));
    for(int r = 0; r < NTask; r++) {
        send_counts_int[r] = send_count_per_rank[r];
        send_hdr_counts[r] = send_hdr_count_per_rank[r];
    }
    MPI_Alltoall(send_counts_int, 1, MPI_INT, recv_counts_int, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Alltoall(send_hdr_counts, 1, MPI_INT, recv_hdr_counts, 1, MPI_INT, MPI_COMM_WORLD);

    int total_send = 0, total_recv = 0;
    int total_hdr_send = 0, total_hdr_recv = 0;
    for(int r = 0; r < NTask; r++) {
        total_send += send_counts_int[r]; total_recv += recv_counts_int[r];
        total_hdr_send += send_hdr_counts[r]; total_hdr_recv += recv_hdr_counts[r];
    }

    /* Allocate offsets/bytes for both exchanges, then flat_send / flat_recv
     * for both data streams.  All temporaries are freed in strict reverse
     * order at the end of this function. */
    int *send_offsets     = (int *) mymalloc("LET_send_offsets",     NTask * sizeof(int));
    int *recv_offsets     = (int *) mymalloc("LET_recv_offsets",     NTask * sizeof(int));
    int *send_bytes       = (int *) mymalloc("LET_send_bytes",       NTask * sizeof(int));
    int *recv_bytes       = (int *) mymalloc("LET_recv_bytes",       NTask * sizeof(int));
    int *send_hdr_offsets = (int *) mymalloc("LET_send_hdr_offsets", NTask * sizeof(int));
    int *recv_hdr_offsets = (int *) mymalloc("LET_recv_hdr_offsets", NTask * sizeof(int));
    int *send_hdr_bytes   = (int *) mymalloc("LET_send_hdr_bytes",   NTask * sizeof(int));
    int *recv_hdr_bytes   = (int *) mymalloc("LET_recv_hdr_bytes",   NTask * sizeof(int));

    int s_off = 0, r_off = 0, hs_off = 0, hr_off = 0;
    for(int r = 0; r < NTask; r++)
    {
        send_offsets[r]     = s_off  * sizeof(struct LETNodeWire);
        recv_offsets[r]     = r_off  * sizeof(struct LETNodeWire);
        send_bytes[r]       = send_counts_int[r] * sizeof(struct LETNodeWire);
        recv_bytes[r]       = recv_counts_int[r] * sizeof(struct LETNodeWire);
        send_hdr_offsets[r] = hs_off * sizeof(struct LETSubtreeHeader);
        recv_hdr_offsets[r] = hr_off * sizeof(struct LETSubtreeHeader);
        send_hdr_bytes[r]   = send_hdr_counts[r] * sizeof(struct LETSubtreeHeader);
        recv_hdr_bytes[r]   = recv_hdr_counts[r] * sizeof(struct LETSubtreeHeader);
        s_off  += send_counts_int[r];
        r_off  += recv_counts_int[r];
        hs_off += send_hdr_counts[r];
        hr_off += recv_hdr_counts[r];
    }

    struct LETNodeWire *flat_send = (struct LETNodeWire *) mymalloc("LET_flat_send",
        (size_t)total_send * sizeof(struct LETNodeWire) + 1);
    struct LETNodeWire *flat_recv = (struct LETNodeWire *) mymalloc("LET_flat_recv",
        (size_t)total_recv * sizeof(struct LETNodeWire) + 1);
    struct LETSubtreeHeader *flat_hdr_send = (struct LETSubtreeHeader *) mymalloc("LET_flat_hdr_send",
        (size_t)total_hdr_send * sizeof(struct LETSubtreeHeader) + 1);
    struct LETSubtreeHeader *flat_hdr_recv = (struct LETSubtreeHeader *) mymalloc("LET_flat_hdr_recv",
        (size_t)total_hdr_recv * sizeof(struct LETSubtreeHeader) + 1);

    /* Concatenate per-rank send buffers */
    int s_pos = 0, hs_pos = 0;
    for(int r = 0; r < NTask; r++)
    {
        if(send_counts_int[r] > 0 && send_buf_per_rank[r])
        {
            memcpy(flat_send + s_pos, send_buf_per_rank[r],
                   (size_t)send_counts_int[r] * sizeof(struct LETNodeWire));
        }
        if(send_hdr_counts[r] > 0 && send_hdr_per_rank[r])
        {
            memcpy(flat_hdr_send + hs_pos, send_hdr_per_rank[r],
                   (size_t)send_hdr_counts[r] * sizeof(struct LETSubtreeHeader));
        }
        s_pos  += send_counts_int[r];
        hs_pos += send_hdr_counts[r];
    }

    /* MPI exchanges (parallel for nodes + headers) */
    MPI_Alltoallv(flat_send,     send_bytes,     send_offsets,     MPI_BYTE,
                  flat_recv,     recv_bytes,     recv_offsets,     MPI_BYTE,
                  MPI_COMM_WORLD);
    MPI_Alltoallv(flat_hdr_send, send_hdr_bytes, send_hdr_offsets, MPI_BYTE,
                  flat_hdr_recv, recv_hdr_bytes, recv_hdr_offsets, MPI_BYTE,
                  MPI_COMM_WORLD);

    /* Install foreign tree contents while flat_recv / flat_hdr_recv are
     * still alive on the mymalloc stack. */
    let_unpack_and_install(flat_recv, recv_counts_int, total_recv,
                            flat_hdr_recv, recv_hdr_counts, total_hdr_recv);

    /* Free everything in strict reverse-alloc order */
    myfree(flat_hdr_recv);
    myfree(flat_hdr_send);
    myfree(flat_recv);
    myfree(flat_send);
    myfree(recv_hdr_bytes);
    myfree(send_hdr_bytes);
    myfree(recv_hdr_offsets);
    myfree(send_hdr_offsets);
    myfree(recv_bytes);
    myfree(send_bytes);
    myfree(recv_offsets);
    myfree(send_offsets);
    myfree(recv_hdr_counts);
    myfree(send_hdr_counts);
    myfree(recv_counts_int);
    myfree(send_counts_int);
    return 0;
}

/* ----------------------------------------------------------------------
 * Step 5: unpack received nodes into Nodes_base[] foreign range; remap
 *         pointers via remote_id->local_foreign_idx translation table;
 *         redirect each affected local topleaf's u.d.nextnode.
 *
 * Each sender's wire data is self-contained: shipped node indices reference
 * indices WITHIN this sender's wire buffer (we converted them in pack to
 * wire-indices, with sentinels for subtree-edge pointers).  So unpack per
 * sender:
 *   1. Allocate consecutive foreign slots for this sender's nodes.
 *   2. Build per-sender translation: wire_idx -> local_foreign_slot.
 *      (NOTE: the wire layout uses wire-local indices in u.d.{sibling,nextnode},
 *       NOT the original source-rank Nodes_base indices; pack stored wire
 *       offsets directly to simplify this step.)
 *   3. For each unpacked node, remap its sibling/nextnode/father using the
 *      translation; sentinel -> source-side topleaf's sibling on receiver.
 *      WAIT: we need to know WHICH local topleaf this subtree corresponds to.
 *
 * Fixme/limitation:  Per-subtree topleaf attribution requires extra metadata
 * we haven't carried in the wire format yet.  For Phase 9.1 first-cut, we
 * store the topleaf-correspondence by linking the foreign subtree's ROOT to
 * the local topleaf via DomainNodeIndex[] -- the foreign root's _pad0 field
 * carries the LOCAL topleaf node index that this subtree belongs to.  Pack
 * sets this when starting a new subtree-root entry.
 *
 * To keep this manageable: pack stores per-subtree-root the local topleaf_no
 * in the FIRST wire entry's _pad0 field, with a flag in remote_id high bits
 * to signal "I am a subtree root."  Unpack walks per-sender, identifies subtree
 * roots, and uses _pad0 as the topleaf-correspondence target.
 * ---------------------------------------------------------------------- */

/* Phase 9.1e_v2: install nodes per-sender into contiguous foreign slots,
 * remap intra-sender wire indices to absolute Node indices, resolve
 * subtree-edge sentinels via per-subtree topleaf, and redirect each affected
 * local topleaf's u.d.nextnode at the foreign subtree root.
 *
 * Key invariants:
 *   - Within sender r's flat node payload of length recv_count_per_rank[r],
 *     each LETNodeWire's u.d.{sibling,nextnode} value V is either:
 *       * V in [0, recv_count_per_rank[r])  -> intra-sender wire index;
 *         remap to slot_base_r + V.
 *       * V == LET_EDGE_SENTINEL_BASE       -> subtree-edge: redirect to
 *         the LOCAL topleaf's u.d.sibling (lookup via the subtree header
 *         that owns this node's wire range).
 *       * V == -1 (legacy "end of walk")    -> leave as-is.
 *       * other negative                    -> leave as-is (defensive).
 *   - Each subtree header h carries topleaf_idx referring to the SHARED
 *     DomainNodeIndex[]/DomainTask[] partition (same on every rank).  The
 *     receiver redirects Nodes[DomainNodeIndex[h.topleaf_idx]].u.d.nextnode
 *     -> slot_base_r + h.wire_offset.
 */
extern "C" int let_unpack_and_install(const struct LETNodeWire *recv_buf,
                                       const int *recv_count_per_rank,
                                       int recv_count_total,
                                       const struct LETSubtreeHeader *recv_hdr_buf,
                                       const int *recv_hdr_count_per_rank,
                                       int recv_hdr_count_total)
{
    if(recv_count_total == 0) return 0;

    /* Capacity check */
    if(Numforeignnodes + recv_count_total > MaxForeignNodes)
    {
        if(ThisTask == 0)
        {
            printf("ERROR: LET unpack overflow.  Numforeignnodes=%d + recv=%d > MaxForeignNodes=%d.\n"
                   "       Increase All.LETAllocFactor (currently %g) in the parameter file.\n"
                   "       (Future option (b) -- graceful shrink + fallback to legacy export -- "
                   "is documented in handoff_step13_phase9_locked.md but not implemented.)\n",
                   Numforeignnodes, recv_count_total, MaxForeignNodes, All.LETAllocFactor);
        }
        endrun(914020);
    }

    int node_off = 0;     /* running offset into recv_buf (per-sender) */
    int hdr_off  = 0;     /* running offset into recv_hdr_buf (per-sender) */

    for(int r = 0; r < NTask; r++)
    {
        int rcount = recv_count_per_rank[r];
        int hcount = recv_hdr_count_per_rank[r];
        if(rcount == 0)
        {
            /* skip — no nodes from this sender; should also have no headers */
            hdr_off += hcount;
            continue;
        }

        int slot_base = All.MaxPart + MaxNodes + Numforeignnodes;

        /* Build orig_to_wire[]: maps sender-side node index -> wire index (0-based within this
         * sender's rcount-wide buffer).  Used to resolve LET_EDGE_SENTINEL_ENCODE sentinels
         * where the encoded sibling may point to another packed node in the same sender's buffer.
         * Array is indexed as (remote_id - All.MaxPart); remote_id in [All.MaxPart, All.MaxPart+MaxNodes).
         * Synthesized particle leaves (remote_id < 0) are not in the map -- their nextnode sentinels
         * are never followed (len=0 leaf always closed by BH criterion). */
        int *orig_to_wire = (int *) mymalloc("orig_to_wire", MaxNodes * sizeof(int));
        for(int jj = 0; jj < MaxNodes; jj++) orig_to_wire[jj] = -1;
        for(int j = 0; j < rcount; j++)
        {
            int rid = recv_buf[node_off + j].remote_id;
            if(rid >= All.MaxPart && rid < All.MaxPart + MaxNodes)
                orig_to_wire[rid - All.MaxPart] = j;
        }

        /* Pass 1: install nodes (raw byte copy), remap intra-sender wire indices,
         * resolve both plain sentinels (LET_EDGE_SENTINEL_BASE) and encoded sentinels
         * (LET_EDGE_SENTINEL_ENCODE(orig_sib)) in sibling/nextnode.  We do NOT yet
         * know topleaf_sibling per node (that requires the header lookup), so encoded
         * sentinels that fall back (orig_sib not in map) are temporarily marked with
         * LET_EDGE_SENTINEL_BASE and resolved in Pass 2. */
        for(int j = 0; j < rcount; j++)
        {
            int abs_idx = slot_base + j;
            Nodes[abs_idx]    = recv_buf[node_off + j].node;
            Extnodes[abs_idx] = recv_buf[node_off + j].extnode;

            int sib  = Nodes[abs_idx].u.d.sibling;
            int next = Nodes[abs_idx].u.d.nextnode;

            /* Remap intra-buffer wire indices first */
            if(sib  >= 0 && sib  < rcount) { Nodes[abs_idx].u.d.sibling  = slot_base + sib; sib = Nodes[abs_idx].u.d.sibling; }
            if(next >= 0 && next < rcount) { Nodes[abs_idx].u.d.nextnode = slot_base + next; next = Nodes[abs_idx].u.d.nextnode; }

            /* Resolve encoded sentinel for sibling */
            if(LET_EDGE_SENTINEL_IS_ENCODED(sib))
            {
                int orig_sib = LET_EDGE_SENTINEL_DECODE(sib);
                if(orig_sib >= All.MaxPart && orig_sib < All.MaxPart + MaxNodes && orig_to_wire[orig_sib - All.MaxPart] >= 0)
                    Nodes[abs_idx].u.d.sibling = slot_base + orig_to_wire[orig_sib - All.MaxPart];
                else
                    Nodes[abs_idx].u.d.sibling = LET_EDGE_SENTINEL_BASE; /* resolved in Pass 2 */
            }
            /* Resolve encoded sentinel for nextnode (non-essential / synth leaves -- usually
             * never followed, but resolve for correctness) */
            if(LET_EDGE_SENTINEL_IS_ENCODED(next))
            {
                int orig_next = LET_EDGE_SENTINEL_DECODE(next);
                if(orig_next >= All.MaxPart && orig_next < All.MaxPart + MaxNodes && orig_to_wire[orig_next - All.MaxPart] >= 0)
                    Nodes[abs_idx].u.d.nextnode = slot_base + orig_to_wire[orig_next - All.MaxPart];
                else
                    Nodes[abs_idx].u.d.nextnode = LET_EDGE_SENTINEL_BASE;
            }

            Nodes[abs_idx].u.d.father = -1;  /* foreign nodes have no local father */
        }
        myfree(orig_to_wire);

        /* Pass 2: resolve remaining plain sentinels (LET_EDGE_SENTINEL_BASE) to topleaf_sibling,
         * and redirect each affected local topleaf's u.d.nextnode at the foreign subtree root. */
        for(int hh = 0; hh < hcount; hh++)
        {
            const struct LETSubtreeHeader *h = &recv_hdr_buf[hdr_off + hh];
            int topleaf_idx = h->topleaf_idx;
            int wire_off    = h->wire_offset;
            int wire_cnt    = h->count;

            /* Defensive bounds */
            if(topleaf_idx < 0 || topleaf_idx >= NTopleaves) continue;
            if(wire_off < 0 || wire_cnt <= 0 || wire_off + wire_cnt > rcount) continue;

            int local_topleaf_no = DomainNodeIndex[topleaf_idx];
            if(local_topleaf_no < All.MaxPart || local_topleaf_no >= All.MaxPart + MaxNodes) continue;

            int topleaf_sibling = Nodes[local_topleaf_no].u.d.sibling;
            int subtree_root    = slot_base + wire_off;

            /* Resolve remaining plain sentinels (fallbacks + non-essential nextnode) */
            for(int j = wire_off; j < wire_off + wire_cnt; j++)
            {
                int abs_idx = slot_base + j;
                if(Nodes[abs_idx].u.d.sibling  == LET_EDGE_SENTINEL_BASE) Nodes[abs_idx].u.d.sibling  = topleaf_sibling;
                if(Nodes[abs_idx].u.d.nextnode == LET_EDGE_SENTINEL_BASE) Nodes[abs_idx].u.d.nextnode = topleaf_sibling;
            }

            /* Redirect local topleaf at the foreign subtree root (AoS + SoA). */
            int old_nn = Nodes[local_topleaf_no].u.d.nextnode;
            Nodes[local_topleaf_no].u.d.nextnode = subtree_root;
            gpu_set_soa_nextnode(local_topleaf_no, subtree_root);
        }
        /* Pass 3: AoS -> SoA scatter for the foreign-node range we just
         * installed.  GPU walk reads node fields via SoA only; without this
         * the foreign nodes would have garbage SoA entries. */
        gpu_scatter_foreign_to_soa(slot_base, rcount);

        Numforeignnodes += rcount;
        node_off += rcount;
        hdr_off  += hcount;
    }

    return 0;
}

/* ----------------------------------------------------------------------
 * Top-level orchestrator
 * ---------------------------------------------------------------------- */
extern "C" int let_run_exchange(void)
{
    /* Defensive no-op if no foreign-node headroom was allocated.  GPU builds
     * reject LETAllocFactor<=0 during parameter validation because the legacy
     * gravity export fallback is retired there. */
    if(MaxForeignNodes <= 0) return 0;

    /* let_synthesize_particle_leaf and let_compute_local_payload read
     * P/CellP and may transitively invoke RT/sink/CR helpers that mutate
     * cached fields in CellP.  Invalidate the GPU particles arena up front
     * so the next gpu_particles_arena_acquire re-seeds from host. */
    gpu_particles_arena_invalidate();

    /* Reset foreign count -- fresh LET each tree-build cycle */
    Numforeignnodes = 0;

    /* Phase 9.5: compute local active-topleaf bitmap and Allgather across
     * ranks.  Then pass MY bitmap into the payload (for tighter bbox/bounds)
     * and EACH RECEIVER R's bitmap into let_pack_for_rank(R) (for short-
     * circuit when R is fully inactive this step). */
    int bitmap_n_words = let_bitmap_word_count(NTopleaves);
    if(bitmap_n_words < 1) bitmap_n_words = 1;
    uint64_t *my_active_bitmap = (uint64_t *) mymalloc("LET_my_active_bitmap",
        (size_t) bitmap_n_words * sizeof(uint64_t));
    let_compute_local_active_bitmap(my_active_bitmap, bitmap_n_words);

    uint64_t *all_active_bitmaps = (uint64_t *) mymalloc("LET_all_active_bitmaps",
        (size_t) NTask * (size_t) bitmap_n_words * sizeof(uint64_t));
    MPI_Allgather(my_active_bitmap, bitmap_n_words, MPI_UINT64_T,
                  all_active_bitmaps, bitmap_n_words, MPI_UINT64_T,
                  MPI_COMM_WORLD);

    struct LETPerRankPayload my_payload;
    /* Phase 9.5 Step C: pass MY active bitmap to tighten bbox + scan only
     * active particles.  May regress TREECOL self-shielding for non-active
     * particles -- gated by the bitmap pointer (NULL = Phase 9.4 conservative). */
    let_compute_local_payload(&my_payload, my_active_bitmap, bitmap_n_words);

    struct LETPerRankPayload *all_payloads =
        (struct LETPerRankPayload *) mymalloc("LET_payloads", NTask * sizeof(struct LETPerRankPayload));
    let_exchange_payloads(&my_payload, all_payloads);

    /* Pack per remote rank.  Per-rank send buffers grown via realloc.  */
    struct LETNodeWire **send_per_rank = (struct LETNodeWire **) mymalloc("LET_send_perrank",
        NTask * sizeof(struct LETNodeWire *));
    struct LETSubtreeHeader **send_hdr_per_rank = (struct LETSubtreeHeader **) mymalloc("LET_send_hdr_perrank",
        NTask * sizeof(struct LETSubtreeHeader *));
    int *send_count     = (int *) mymalloc("LET_send_count",     NTask * sizeof(int));
    int *send_hdr_count = (int *) mymalloc("LET_send_hdr_count", NTask * sizeof(int));
    for(int r = 0; r < NTask; r++) {
        send_per_rank[r] = NULL; send_count[r] = 0;
        send_hdr_per_rank[r] = NULL; send_hdr_count[r] = 0;
    }

    for(int r = 0; r < NTask; r++)
    {
        if(r == ThisTask) {send_count[r] = 0; send_hdr_count[r] = 0; continue;}
        int cap = 0, hcap = 0, hcnt = 0;
        const uint64_t *r_bitmap = all_active_bitmaps + (size_t) r * (size_t) bitmap_n_words;
        send_count[r] = let_pack_for_rank(r, all_payloads,
                                           &send_per_rank[r], &cap,
                                           &send_hdr_per_rank[r], &hcap, &hcnt,
                                           r_bitmap, bitmap_n_words);
        send_hdr_count[r] = hcnt;
    }

    /* Exchange + install (let_exchange_nodes inlines let_unpack_and_install
     * to keep mymalloc LIFO discipline correct). */
    let_exchange_nodes(send_per_rank, send_count,
                       send_hdr_per_rank, send_hdr_count);

    /* Free per-rank send buffers (allocated via realloc, not mymalloc) */
    for(int r = 0; r < NTask; r++) {
        if(send_per_rank[r]) free(send_per_rank[r]);
        if(send_hdr_per_rank[r]) free(send_hdr_per_rank[r]);
    }

    /* Cleanup */
    myfree(send_hdr_count);
    myfree(send_count);
    myfree(send_hdr_per_rank);
    myfree(send_per_rank);
    myfree(all_payloads);
    myfree(all_active_bitmaps);
    myfree(my_active_bitmap);
    return 0;
}

#endif /* OPENMP_GPU_OFFLOAD */
