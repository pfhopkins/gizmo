/* gpu_force_drift.cc — Step 13 Phase 7.a
 *
 * GPU pre-walk drift kernel: replaces the CPU host loop in
 * gpu_gravtree_walk_primary that called force_drift_node + mark_dirty for
 * every node with stale Ti_current.  After Phase 7.a:
 *   - mutates Nodes[]/Extnodes[] (UVM AoS) directly inside a Kokkos kernel,
 *   - mirrors the same fields into the SoA used by the GPU walk,
 *   - has zero CPU-side AoS->SoA reseed (the entire dirty_[]/seed_dirty_/
 *     seed_node_/mark_dirty machinery is retired by this commit).
 *
 * Drift factor: trivial closed-form for non-cosmological; cosmological reads
 * a SharedSpace mirror of DriftTable[] populated lazily from host on each
 * top-level call.  The mirror is 1000 doubles (DRIFT_TABLE_LENGTH) so the
 * refresh cost is in the noise.
 *
 * Timestep-dilation policy: GPU drift currently supports configurations where
 * return_timestep_dilation_factor(no, 1) is provably == 1, which is every
 * combination except SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM and
 * SPECIAL_POINT_WEIGHTED_MOTION (both of which need a per-node dilation cache
 * we don't yet build on the GPU side).  Those configs trigger a #error here;
 * see Phase 7 closeout doc for the explicit deferred-work item.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef OPENMP_GPU_OFFLOAD
#include <Kokkos_Core.hpp>
#endif

#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../system/gpu_particles_arena.h"
#include "gpu_gravity_tree.h"
#include "forcetree.h"

#if defined(OPENMP_GPU_OFFLOAD)

/* --- compile-time gate on supported dilation configurations -------------- */
#if defined(USE_TIMESTEP_DILATION_FOR_ZOOMS) \
    && !defined(DILATION_FOR_STELLAR_KINEMATICS_ONLY) \
    && (defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM) || defined(SPECIAL_POINT_WEIGHTED_MOTION))
#error "GPU drift kernel (Phase 7.a) does not yet implement node-indexed timestep dilation.  Either enable DILATION_FOR_STELLAR_KINEMATICS_ONLY (which makes mode=1 dilation a no-op), disable USE_TIMESTEP_DILATION_FOR_ZOOMS, or implement the per-node dilation cache.  See handoff_step13_phase7_closeout.md."
#endif

/* --- DriftTable mirror (cosmological only) ------------------------------- */
static double *drift_table_dev_ = NULL;   /* SharedSpace, length DRIFT_TABLE_LENGTH */
static double  drift_table_logTimeBegin_ = 0.0;
static double  drift_table_logTimeMax_   = 0.0;

static void drift_table_refresh_(void)
{
    if(!drift_table_dev_) {
        drift_table_dev_ = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
                                DRIFT_TABLE_LENGTH * sizeof(double));
        if(!drift_table_dev_) {
            printf("gpu_force_drift: drift_table_dev_ alloc failed\n");
            endrun(929701);
        }
    }
    for(int i = 0; i < DRIFT_TABLE_LENGTH; i++) {drift_table_dev_[i] = DriftTable[i];}
    drift_table_logTimeBegin_ = log(All.TimeBegin);
    drift_table_logTimeMax_   = log(All.TimeMax);
}

/* --- inline drift-factor helper, GPU-callable ---------------------------- */
KOKKOS_INLINE_FUNCTION
double gpu_node_drift_factor_(integertime time0, integertime time1,
                              const double *table,
                              double logTBegin, double logTMax,
                              int comoving, double timebase_interval)
{
    if(!comoving) {
        return (double)(time1 - time0) * timebase_interval;
    }
    /* cosmological: linear interpolation in DriftTable. */
    double a1 = logTBegin + (double)time0 * timebase_interval;
    double a2 = logTBegin + (double)time1 * timebase_interval;
    double span = (logTMax > logTBegin) ? (logTMax - logTBegin) : 1.0;
    double u1 = (logTMax > logTBegin) ? (a1 - logTBegin) / span * (double)DRIFT_TABLE_LENGTH : 0.0;
    double u2 = (logTMax > logTBegin) ? (a2 - logTBegin) / span * (double)DRIFT_TABLE_LENGTH : 0.0;
    int i1 = (int) u1; if(i1 >= DRIFT_TABLE_LENGTH) {i1 = DRIFT_TABLE_LENGTH - 1;}
    int i2 = (int) u2; if(i2 >= DRIFT_TABLE_LENGTH) {i2 = DRIFT_TABLE_LENGTH - 1;}
    double df1 = (i1 <= 1) ? u1 * table[0]
                           : table[i1 - 1] + (table[i1] - table[i1 - 1]) * (u1 - (double)i1);
    double df2 = (i2 <= 1) ? u2 * table[0]
                           : table[i2 - 1] + (table[i2] - table[i2 - 1]) * (u2 - (double)i2);
    return df2 - df1;
}

/* --- dispatcher ---------------------------------------------------------- */

extern "C" int gpu_force_drift_nodes(integertime time1)
{
    GIZMO_GPU_ENSURE_ALL_FRESH(force_drift);

    if(Numnodestree <= 0) {return 0;}

    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!soa || !soa->len || !soa->s || !soa->node_vs || !soa->bitflags
            || !soa->hmax || !soa->vmax) {
        printf("gpu_force_drift_nodes: SoA mirrors not ready\n");
        return 1;
    }

    /* Refresh DriftTable mirror unconditionally (cheap; 8 KB).  Recomputes
     * logTimeBegin / logTimeMax from current All.* in case TimeMax was bumped
     * by a runtime restart. */
    drift_table_refresh_();

    int      MaxPart        = All.MaxPart;
    int      n_local_nodes  = Numnodestree;
    int      n_foreign_nodes = Numforeignnodes;       /* Phase 9.2b */
    int      maxNodes_snap  = MaxNodes;               /* Phase 9.2b: foreign-range base */
    int      n_nodes        = n_local_nodes + n_foreign_nodes;
    integertime ti_target   = time1;
    int      comoving       = (All.ComovingIntegrationOn ? 1 : 0);
    double   timebase_int   = All.Timebase_interval;
    double   logTBegin      = drift_table_logTimeBegin_;
    double   logTMax        = drift_table_logTimeMax_;
    const double *dt_table  = drift_table_dev_;

    /* Use the SHIFTED pointers (Nodes = Nodes_base - MaxPart, Extnodes = Extnodes_base - MaxPart)
     * so that Nodes_uvm[MaxPart + k] == Nodes_base[k] for all k in [0, Numnodestree). */
    struct NODE     *Nodes_uvm    = Nodes;
    struct extNODE  *Extnodes_uvm = Extnodes;

    /* SoA mirror handles. */
    MyFloat           *len_soa     = soa->len;
    Vec3<MyGravFloat> *s_soa       = soa->s;
    Vec3<MyGravFloat> *vs_soa      = soa->node_vs;
    MyGravFloat       *hmax_soa    = soa->hmax;
    unsigned int      *bitflags_soa = soa->bitflags;
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    Vec3<MyGravFloat> *rt_s_soa    = soa->rt_source_lum_s;
    Vec3<MyGravFloat> *rt_vs_soa   = soa->rt_source_lum_vs;
#endif
#ifdef DM_SCALARFIELD_SCREENING
    Vec3<MyGravFloat> *s_dm_soa    = soa->s_dm;
    Vec3<MyGravFloat> *vs_dm_soa   = soa->vs_dm;
#endif

    Kokkos::parallel_for("gpu_force_drift_nodes", n_nodes, KOKKOS_LAMBDA(int kk) {
        /* Phase 9.2b: kk in [0, n_local_nodes) drives local nodes; kk in
         * [n_local_nodes, n_local_nodes + n_foreign_nodes) drives foreign
         * nodes installed by LET unpack (slot index in [0, Numforeignnodes)).
         * SoA index `k` differs from iteration index for foreign nodes:
         *   local:    k_soa = kk          ; no = MaxPart + kk
         *   foreign:  k_soa = maxNodes_snap + (kk - n_local_nodes)
         *             no    = MaxPart + maxNodes_snap + (kk - n_local_nodes) */
        int k, no;
        if(kk < n_local_nodes) {
            k  = kk;
            no = MaxPart + kk;
        } else {
            int slot = kk - n_local_nodes;
            k  = maxNodes_snap + slot;
            no = MaxPart + maxNodes_snap + slot;
        }
        if(Nodes_uvm[no].Ti_current == ti_target) {return;}

        /* dilation factor — see compile-time gate at file top. */
        double dilation = 1.0;

        /* Drift factor matches CPU get_drift_factor(.., .., no, 1). */
        double dt_drift = gpu_node_drift_factor_(Nodes_uvm[no].Ti_current,
                                                 ti_target, dt_table,
                                                 logTBegin, logTMax, comoving,
                                                 timebase_int) * dilation;
        double dt_drift_hmax = dt_drift;

        /* If node has been kicked, fold dp into vs and clear dp. */
        if(Nodes_uvm[no].u.d.bitflags & (1u << BITFLAG_NODEHASBEENKICKED)) {
            double mass = (double) Nodes_uvm[no].u.d.mass;
            double fac  = (mass > 0) ? (1.0 / mass) : 0.0;

#ifdef RT_SEPARATELY_TRACK_LUMPOS
            double l_tot = 0.0;
            for(int b = 0; b < N_RT_FREQ_BINS; b++) {l_tot += (double)Nodes_uvm[no].stellar_lum[b];}
            double fac_lum = (l_tot > 0) ? (1.0 / l_tot) : 0.0;
#endif
#ifdef DM_SCALARFIELD_SCREENING
            double mass_dm = (double) Nodes_uvm[no].mass_dm;
            double fac_dm  = (mass_dm > 0) ? (1.0 / mass_dm) : 0.0;
#endif

            for(int j = 0; j < 3; j++) {
                Extnodes_uvm[no].vs[j] = (MyFloat)((double)Extnodes_uvm[no].vs[j] + fac * (double)Extnodes_uvm[no].dp[j]);
                Extnodes_uvm[no].dp[j] = 0;
#ifdef RT_SEPARATELY_TRACK_LUMPOS
                Extnodes_uvm[no].rt_source_lum_vs[j] = (MyFloat)((double)Extnodes_uvm[no].rt_source_lum_vs[j]
                                                       + fac_lum * (double)Extnodes_uvm[no].rt_source_lum_dp[j]);
                Extnodes_uvm[no].rt_source_lum_dp[j] = 0;
#endif
#ifdef DM_SCALARFIELD_SCREENING
                Extnodes_uvm[no].vs_dm[j] = (MyFloat)((double)Extnodes_uvm[no].vs_dm[j] + fac_dm * (double)Extnodes_uvm[no].dp_dm[j]);
                Extnodes_uvm[no].dp_dm[j] = 0;
#endif
            }
            Nodes_uvm[no].u.d.bitflags &= (~(1u << BITFLAG_NODEHASBEENKICKED));
        }

        /* Apply drift to s, len, hmax. */
        for(int j = 0; j < 3; j++) {
            Nodes_uvm[no].u.d.s[j] = (MyFloat)((double)Nodes_uvm[no].u.d.s[j] + (double)Extnodes_uvm[no].vs[j] * dt_drift);
#ifdef DM_SCALARFIELD_SCREENING
            Nodes_uvm[no].s_dm[j]  = (MyFloat)((double)Nodes_uvm[no].s_dm[j]  + (double)Extnodes_uvm[no].vs_dm[j] * dt_drift);
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
            Nodes_uvm[no].rt_source_lum_s[j] = (MyFloat)((double)Nodes_uvm[no].rt_source_lum_s[j]
                                                + (double)Extnodes_uvm[no].rt_source_lum_vs[j] * dt_drift);
#endif
        }
        Nodes_uvm[no].len = (MyFloat)((double)Nodes_uvm[no].len + 2.0 * (double)Extnodes_uvm[no].vmax * dt_drift);

        if(Extnodes_uvm[no].hmax > 0) {
            double exp_arg = (double)Extnodes_uvm[no].divVmax * dt_drift_hmax / (double)NUMDIMS;
            if(exp_arg < -1.0) {exp_arg = -1.0;}
            if(exp_arg >  1.0) {exp_arg =  1.0;}
            Extnodes_uvm[no].hmax = (MyFloat)((double)Extnodes_uvm[no].hmax * exp(exp_arg));
        }

        Nodes_uvm[no].Ti_current = ti_target;

        /* SoA mirror update: only the fields the walk reads.  Vec3 narrowing
         * cast for mixed-precision builds (MyGravFloat=float, MyFloat=double). */
        len_soa[k]  = Nodes_uvm[no].len;
        s_soa[k]    = { (MyGravFloat)Nodes_uvm[no].u.d.s[0],
                        (MyGravFloat)Nodes_uvm[no].u.d.s[1],
                        (MyGravFloat)Nodes_uvm[no].u.d.s[2] };
        vs_soa[k]   = { (MyGravFloat)Extnodes_uvm[no].vs[0],
                        (MyGravFloat)Extnodes_uvm[no].vs[1],
                        (MyGravFloat)Extnodes_uvm[no].vs[2] };
        hmax_soa[k] = (MyGravFloat)Extnodes_uvm[no].hmax;
        bitflags_soa[k] = Nodes_uvm[no].u.d.bitflags;
#ifdef RT_SEPARATELY_TRACK_LUMPOS
        rt_s_soa[k]  = { (MyGravFloat)Nodes_uvm[no].rt_source_lum_s[0],
                         (MyGravFloat)Nodes_uvm[no].rt_source_lum_s[1],
                         (MyGravFloat)Nodes_uvm[no].rt_source_lum_s[2] };
        rt_vs_soa[k] = { (MyGravFloat)Extnodes_uvm[no].rt_source_lum_vs[0],
                         (MyGravFloat)Extnodes_uvm[no].rt_source_lum_vs[1],
                         (MyGravFloat)Extnodes_uvm[no].rt_source_lum_vs[2] };
#endif
#ifdef DM_SCALARFIELD_SCREENING
        s_dm_soa[k]  = { (MyGravFloat)Nodes_uvm[no].s_dm[0],
                         (MyGravFloat)Nodes_uvm[no].s_dm[1],
                         (MyGravFloat)Nodes_uvm[no].s_dm[2] };
        vs_dm_soa[k] = { (MyGravFloat)Extnodes_uvm[no].vs_dm[0],
                         (MyGravFloat)Extnodes_uvm[no].vs_dm[1],
                         (MyGravFloat)Extnodes_uvm[no].vs_dm[2] };
#endif
    });
    Kokkos::fence();
    return 0;
}

extern "C" void gpu_force_drift_release(void)
{
    if(drift_table_dev_) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(drift_table_dev_);
        drift_table_dev_ = NULL;
    }
}

GPU_ALL_SYNC_FUNC(force_drift)
GPU_ALL_SYNC_FUNC(gpu_assign_gravcost)


/* =========================================================================== *
 * Phase 7.b: GPU gravity-cost assignment.                                      *
 *                                                                              *
 * Replaces the host loop at gravtree.cc:487-495 that walks the Father chain    *
 * for every local particle and accumulates Nodes[].GravCost into               *
 * P[i].GravCost[takelevel].  The same loop is also responsible for zeroing     *
 * P[i].GravCost[takelevel] before the accumulation (host did this at line 145; *
 * on GPU builds the GPU kernel does it inline).                                 *
 *                                                                              *
 * All inputs are UVM: Father[] since Phase 6.6, Nodes[] since Phase 6.8d,     *
 * P[] via gpu_particles_arena (SharedSpace since Phase 1).  No atomics needed  *
 * since each thread owns a private P[i].GravCost[takelevel] slot.              *
 * =========================================================================== */
extern "C" void gpu_assign_gravcost(int takelevel)
{
    if(NumPart <= 0) return;
    GIZMO_GPU_ENSURE_ALL_FRESH(gpu_assign_gravcost);

    /* Re-seed arena from host P[] if hydro/kicks/predict invalidated it between
     * gravity calls.  gpu_particles_arena_P() returns NULL when arena_valid_==0,
     * which caused a NULL-deref crash on the third gravity_tree() call (after hydro
     * ran and fired gpu_particles_arena_invalidate()).  Acquiring here is cheap when
     * already valid (fast path in arena code), and correct when stale. */
    gpu_particles_arena_acquire(NumPart, P, CellP);

    struct particle_data *Pp = gpu_particles_arena_P();
    int                  *Fa = Father;   /* UVM since Phase 6.6 */
    struct NODE          *No = Nodes;    /* UVM shifted ptr since Phase 6.8d */

    int _mp = All.MaxPart, _mn = MaxNodes;
    Kokkos::parallel_for("gpu_assign_gravcost", NumPart,
        KOKKOS_LAMBDA(const int i) {
            float gc = 0.0f;
            int no = Fa[i];
            while(no >= 0) {
                if(no < _mp || no >= _mp + _mn) {break;}  /* guard against corrupt father */
                double nm = (double)No[no].u.d.mass;
                if(nm > 0) {
                    gc += (float)(No[no].GravCost * (double)Pp[i].Mass / nm);
                }
                no = No[no].u.d.father;
            }
            Pp[i].GravCost[takelevel] = gc;
        });
    Kokkos::fence();

    /* Copy GravCost results from the arena (Pp) back to host P[].  The arena is a
     * separate SharedSpace allocation from P, so the GPU's writes to Pp[i].GravCost
     * do not automatically update P[i].GravCost; the host loop below does that. */
    for(int i = 0; i < NumPart; i++) {P[i].GravCost[takelevel] = Pp[i].GravCost[takelevel];}
}

#else /* !OPENMP_GPU_OFFLOAD */

/* Non-GPU builds: stubs so the header prototype resolves. */
extern "C" int  gpu_force_drift_nodes(integertime t) { (void)t; return 0; }
extern "C" void gpu_force_drift_release(void) {}
extern "C" void gpu_assign_gravcost(int tl) { (void)tl; }

#endif /* OPENMP_GPU_OFFLOAD */
