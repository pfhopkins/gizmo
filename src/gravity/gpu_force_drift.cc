/* gpu_force_drift.cc
 *
 * GPU pre-walk drift kernel: replaces the CPU host loop in
 * gpu_gravtree_walk_primary that called force_drift_node + mark_dirty for
 * every node with stale Ti_current.  This kernel:
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
 * Timestep-dilation policy: GPU drift supports all configurations.  For
 * SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM and SPECIAL_POINT_WEIGHTED_MOTION the
 * per-node dilation_dev cache (populated by gpu_force_update_tree) is used;
 * all other configs are provably dilation==1 and skip the cache lookup.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../declarations/gpu_error_check.h"
#include "../system/gpu_particles_arena.h"
#include "gpu_gravity_tree.h"
#include "forcetree.h"


/* USE_TIMESTEP_DILATION_FOR_ZOOMS: node-indexed dilation is supported via a
 * per-call host-pre-compute cache. The dispatcher below allocates a SharedSpace
 * `dilation_dev` array of length n_local_nodes + n_foreign_nodes, populates it
 * on the host using return_node_timestep_dilation_factor(no), then the GPU
 * drift kernel reads the cached value. */

/* --- DriftTable mirror (cosmological only) ------------------------------- */
static double *drift_table_dev_ = NULL;   /* SharedSpace, length DRIFT_TABLE_LENGTH */
static double  drift_table_logTimeBegin_ = 0.0;
static double  drift_table_logTimeMax_   = 0.0;

static int drift_table_refresh_(void)
{
    if(!drift_table_dev_) {
        drift_table_dev_ = (double *) gizmo_gpu_alloc_shared(DRIFT_TABLE_LENGTH * sizeof(double), NULL);
        if(!drift_table_dev_) {
            printf("gpu_force_drift: drift_table_dev_ alloc failed\n");
            endrun(929701);
            return 1;   /* soft bad-stop: caller skips the drift kernel; drains at next poll */
        }
    }
    for(int i = 0; i < DRIFT_TABLE_LENGTH; i++) {drift_table_dev_[i] = DriftTable[i];}
    drift_table_logTimeBegin_ = log(All.TimeBegin);
    drift_table_logTimeMax_   = log(All.TimeMax);
    return 0;
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
    GIZMO_GPU_ENSURE_ALL_FRESH();

    if(Numnodestree <= 0) {return 0;}

    /* This sweep skips any node already at time1 and therefore also skips writing that
     * node's SoA mirror. A host lazy drift advances Nodes[].Ti_current without touching
     * the mirror, so if the host has already drifted to time1 the sweep would leave the
     * walk reading pre-drift geometry. The routing keeps the two apart -- a step whose
     * tree update runs on the host also walks on the host -- and this is the check that
     * the two never disagree. The caller treats a nonzero return as a controlled stop
     * taken by every rank at the next poll, so no rank exits a collective alone. */
    if(force_host_lazy_drift_ti() == time1) {
        printf("gpu_force_drift_nodes: task %d already drifted nodes on the host at this time; the device node mirror cannot be brought up to date by a sweep that skips them\n", ThisTask);
        return 1;
    }

    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!soa || !soa->len || !soa->s || !soa->node_vs || !soa->bitflags
            || !soa->hmax || !soa->vmax) {
        printf("gpu_force_drift_nodes: SoA mirrors not ready\n");
        return 1;
    }

    /* Refresh DriftTable mirror unconditionally (cheap; 8 KB).  Recomputes
     * logTimeBegin / logTimeMax from current All.* in case TimeMax was bumped
     * by a runtime restart. */
    if(drift_table_refresh_() != 0) {return 1;}   /* soft bad-stop propagated: no kernel launch on NULL drift table */

    int      tree_base        = All.TreeNodeIndexBase;
    int      n_local_nodes  = Numnodestree;
    int      n_foreign_nodes = Numforeignnodes;
    int      maxNodes_snap  = MaxNodes;               /* foreign-range base */
    int      n_nodes        = n_local_nodes + n_foreign_nodes;

#ifdef USE_TIMESTEP_DILATION_FOR_ZOOMS
    /* Pre-compute per-node dilation factors on the host (cheap O(n_nodes) loop
     * with one All.SpecialParticle_Position_ForRefinement comparison per node
     * for SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM). The GPU kernel then just reads
     * dilation_dev[kk] -- no GPU-side host-only field access required. */
    double *dilation_dev = (double *) gizmo_gpu_alloc_shared((size_t)n_nodes * sizeof(double), NULL);
    if(!dilation_dev) {printf("gpu_force_drift_nodes: dilation_dev alloc failed\n"); endrun(929702); return 1;}   /* soft bad-stop: skip the dilation omp loop on NULL; drains at next poll */
#pragma omp parallel for schedule(static)
    for(int kk = 0; kk < n_nodes; kk++) {
        int no_kk;
        if(kk < n_local_nodes) no_kk = tree_base + kk;
        else                   no_kk = tree_base + maxNodes_snap + (kk - n_local_nodes);
        dilation_dev[kk] = return_node_timestep_dilation_factor(no_kk);
    }
#endif


    integertime ti_target   = time1;
    int      comoving       = (All.ComovingIntegrationOn ? 1 : 0);
    double   timebase_int   = All.Timebase_interval;
    double   logTBegin      = drift_table_logTimeBegin_;
    double   logTMax        = drift_table_logTimeMax_;
    const double *dt_table  = drift_table_dev_;

    /* Use the SHIFTED pointers (Nodes = Nodes_base - tree_base, Extnodes = Extnodes_base - tree_base)
     * so that Nodes_uvm[tree_base + k] == Nodes_base[k] for all k in [0, Numnodestree). */
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
        /* kk in [0, n_local_nodes) drives local nodes; kk in
         * [n_local_nodes, n_local_nodes + n_foreign_nodes) drives foreign
         * nodes installed by LET unpack (slot index in [0, Numforeignnodes)).
         * SoA index `k` differs from iteration index for foreign nodes:
         *   local:    k_soa = kk          ; no = tree_base + kk
         *   foreign:  k_soa = maxNodes_snap + (kk - n_local_nodes)
         *             no    = tree_base + maxNodes_snap + (kk - n_local_nodes) */
        int k, no;
        if(kk < n_local_nodes) {
            k  = kk;
            no = tree_base + kk;
        } else {
            int slot = kk - n_local_nodes;
            k  = maxNodes_snap + slot;
            no = tree_base + maxNodes_snap + slot;
        }
        if(Nodes_uvm[no].Ti_current == ti_target) {return;}

        /* Per-node dilation factor (host-pre-computed, see dispatcher above). */
#ifdef USE_TIMESTEP_DILATION_FOR_ZOOMS
        double dilation = dilation_dev[kk];
#else
        double dilation = 1.0;
#endif

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
        Nodes_uvm[no].len = (MyFloat)((double)Nodes_uvm[no].len
                                      + TREE_DRIFT_VELOCITY_PREFAC * (double)Extnodes_uvm[no].vmax * dt_drift);

        {
            double exp_arg = (double)Extnodes_uvm[no].divVmax * dt_drift_hmax / (double)NUMDIMS;
            if(exp_arg < -1.0) {exp_arg = -1.0;}
            if(exp_arg >  1.0) {exp_arg =  1.0;}
            double decay_fac = exp(exp_arg);
            if(Extnodes_uvm[no].hmax > 0) {
                Extnodes_uvm[no].hmax = (MyFloat)((double)Extnodes_uvm[no].hmax * decay_fac);
            }
            /* Mode B per-type bands: upward-only inflate.
             * Bands include static-ish sources (P[j].ForceSoftening) that
             * don't shrink under drift, so decaying below the actual FS value
             * would under-bound the node-prune. force_update_hmax() re-grows
             * bands per-particle each call; we just must not shrink them
             * here. Scalar `hmax` keeps its legacy bidirectional decay above.
             * Without this guard, expansion regions (positive divVmax) would
             * fail to track via the upward branch. */
            if(decay_fac > 1.0) {
                for(int t = 0; t < 6; t++) {
                    if(Extnodes_uvm[no].hmax_per_type[t] > 0) {
                        Extnodes_uvm[no].hmax_per_type[t] = (MyFloat)((double)Extnodes_uvm[no].hmax_per_type[t] * decay_fac);
                    }
                }
            }
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
    gizmo_gpu_check_last_error("gpu_force_drift_nodes", n_nodes);
#ifdef USE_TIMESTEP_DILATION_FOR_ZOOMS
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(dilation_dev);
#endif
    /* The sweep drifted every node's AoS + SoA geometry to time1 → record
     * certification so a consumer can confirm the tree is current at time1
     * without re-sweeping. */
    gpu_gravity_soa_mark_drift_certified(time1);
    return 0;
}

extern "C" void gpu_force_drift_release(void)
{
    if(drift_table_dev_) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(drift_table_dev_);
        drift_table_dev_ = NULL;
    }
}


