/* Source-reach map: exact routing of SYMMETRIC queries to remote top leaves (see source_reach_map.h). */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <new>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../core/timestep_functions.h"   /* aggregate_radius_bound */
#include "../gravity/forcetree.h"         /* force_hmax_per_type_particle_radius, tree_lifetime_drift */
#include "ghost_exchange_functions.h"     /* gx_extended_overlap_wrap_and_test, gx_extended_wrapped_separation */
#include "ghost_symlist_lifecycle.h"      /* gizmo_ghost_safety_factor */
#include "mode_b_local_walker.h"          /* ModeBTopleafMap */
#include "source_reach_map.h"

namespace {

/* What the owner of top leaf T tells the owner of top leaf U: the members of T that may reach U's cube before the tree
 * is rebuilt -- a box of their positions at the build, as offsets from T's (fixed) centre rounded outward, and their
 * largest radius per type at the build, rounded up (-1 = no such member). */
struct SourceReachEntry {
    int t_leaf, u_leaf;
    float lo[3], hi[3];
    float reach[6];
};

/* One of this rank's top leaves, as a query's leaf U: its cube at the build and the lists other ranks sent for it. */
struct OwnLeaf {
    int leaf;
    float half_len;
    long long recv_first, recv_end;
};

int g_valid = 0;
/* Demand gate.  Every decision is rank-uniform (taken from vote results); only the local timings differ per rank. */
int g_wanted = 0;                 /* the map is to be built */
integertime g_tree_ti = -1;       /* time of the last tree build or moment refresh; -1 once a due build was tried */
int g_tree_usable = 0;            /* that build's usable flag */
double g_t_tree0 = -1, g_t_lastbuild_end = -1;   /* local wall clock: start of the current tree build, start of the last map build call */
double g_tb_local = 0, g_b_local = -1;           /* local: last tree build time; map build time summed over the last mapped tree (-1: never built) */
double g_E_tree = 0;              /* corridor imbalance (slowest - fastest rank) summed over the current tree so far */
int g_n_tree = 0;                 /* the symmetric discoveries in that sum */
int g_map_this_tree = 0;          /* the current tree has a map */
double g_E_off_last = 0;          /* that sum over the last tree, or tree part, without a map */
int g_n_off_last = 0;             /* its discoveries */
double g_E_nogain = -1;           /* the unmapped imbalance at which the map last proved not worth its cost */
double g_B_all = -1;              /* the map's build cost, slowest rank (-1: never built) */
double g_e_fresh = -1;            /* corridor imbalance of the first discovery on the current map (-1: none yet) */
double g_E_decay = 0;             /* imbalance above that, summed since: the map's decay, to set against a rebuild */
integertime g_build_ti = 0;
double g_lifetime_drift = 0.0;    /* undilated drift factor over the tree's expected lifetime, from the build */
double g_scale_max = 1.0;         /* the largest j-side reach scale the lists cover */
std::vector<float> g_widen_allowed, g_growth_allowed;   /* per top leaf: motion and growth (divVmax x drift) the build allowed for */
std::vector<unsigned char> g_stale;                      /* per top leaf */
std::vector<float> g_top_enclose;                        /* per top-level node: half side of a cube on its centre holding every leaf below at the build */
std::vector<SourceReachEntry> g_recv;                   /* lists for this rank's leaves U, sorted by (U, T) */
std::vector<float> g_covered;                            /* per particle slot: its radius at the build, rounded down; -1 = not a member then */
std::vector<OwnLeaf> g_own;                              /* sorted by leaf */
const ModeBTopleafMap *g_map = nullptr;
integertime g_now_ti = -1;
double g_now_drift = 0.0;

/* Room the lists leave for a member's radius to rise other than by drift (the density update sets a new one at the time of the
 * build): a member is listed as if its radius were this factor larger, so a rise within it keeps the map valid.  Larger lists
 * cost envelopes; a rise beyond it turns the member's leaf to the node bound until the next build.  The value is a measured
 * performance choice, not a bound: on the fire and nuclear-zoom test problems all but ~1 in 50,000 post-density radii stay
 * within 10% of their value at the build (most within 1%), and 1.25 bought fewer stale leaves at more envelopes. */
const double SOURCE_REACH_RADIUS_MARGIN = 1.1;

inline float float_at_or_above(double x) {float f = (float) x; return ((double) f >= x) ? f : nextafterf(f, INFINITY);}
inline float float_at_or_below(double x) {float f = (float) x; return ((double) f <= x) ? f : nextafterf(f, -INFINITY);}
/* A node bound another thread of the post-density update may be raising. */
template <class T> inline double load_relaxed(const T *p) {T v; __atomic_load(p, &v, __ATOMIC_RELAXED); return (double) v;}

/* The radius a member of radius h may reach after growth `growth` (divVmax times drift), or -1 if it cannot be bounded. */
inline double grown_radius(double h, double growth)
{
    int invalid = 0;
    const double r = aggregate_radius_bound(h, growth, 1.0, (double) All.MaxKernelRadius, &invalid);
    return invalid ? -1.0 : r;
}

inline int own_slot(int leaf)
{
    int a = 0, b = (int) g_own.size();
    while(a < b) {const int m = (a + b) / 2; if(g_own[m].leaf < leaf) {a = m + 1;} else {b = m;}}
    return (a < (int) g_own.size() && g_own[a].leaf == leaf) ? a : -1;
}

/* The entry (t, u) in a range sorted by the key `first` then `second`, or nullptr. */
inline const SourceReachEntry *find_entry(const std::vector<SourceReachEntry> &v, long long first, long long end, int key1, int key2, int by_u)
{
    while(first < end) {
        const long long m = (first + end) / 2;
        const int k1 = by_u ? v[m].u_leaf : v[m].t_leaf, k2 = by_u ? v[m].t_leaf : v[m].u_leaf;
        if(k1 < key1 || (k1 == key1 && k2 < key2)) {first = m + 1;} else {end = m;}
    }
    if(first < (long long) v.size()) {
        const SourceReachEntry &e = v[first];
        if((by_u ? e.u_leaf : e.t_leaf) == key1 && (by_u ? e.t_leaf : e.u_leaf) == key2) {return &e;}
    }
    return nullptr;
}

/* Every other rank's top leaf whose cube meets the box (centre c, half widths hw) widened by R: f(leaf, node).  Walks
 * the replicated top-level tree only, never below a top leaf. */
template <class F>
void for_remote_leaves_near(const double c[3], const double hw[3], double R, F &&f)
{
    const int base = All.TreeNodeIndexBase;
    int no = base;
    while(no >= 0) {
        const struct NODE *nop = &Nodes[no];
        /* not the node's own length: a leaf the owner grew need not lie inside this rank's copy of its ancestors */
        const double h = (double) g_top_enclose[no - base];
        const int near = gx_extended_overlap_wrap_and_test((double) nop->center[0] - c[0], (double) nop->center[1] - c[1],
                                                           (double) nop->center[2] - c[2], h + hw[0], h + hw[1], h + hw[2], R);
        const int leaf = g_map->topleaf_of(no, base);
        if(leaf >= 0) {
            if(near && DomainTask[leaf] != ThisTask) {f(leaf, no);}
            no = nop->u.d.sibling;
        } else {
            no = near ? nop->u.d.nextnode : nop->u.d.sibling;
        }
    }
}

/* Build this rank's lists for its own leaves T.  Returns 0 on success; 1 if a member's reach cannot be bounded or an
 * allocation failed; 2 if a member is not at the current time (a moment refresh mid-step: the next tree build can succeed). */
int build_own_lists(std::vector<std::vector<SourceReachEntry>> &per_thread)
{
    const int tree_slots = All.TreeParticleSlots, local_end = All.TreeNodeIndexBase + MaxNodes;
    int failed = 0, behind = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) reduction(|:failed, behind)
#endif
    for(int s = 0; s < (int) g_own.size(); s++) {
#ifdef _OPENMP
        std::vector<SourceReachEntry> &out = per_thread[omp_get_thread_num()];
#else
        std::vector<SourceReachEntry> &out = per_thread[0];
#endif
        try {   /* an exception may not leave the parallel region: a failed allocation turns the map off */
        OwnLeaf &own = g_own[s];
        const int t = own.leaf, t_node = DomainNodeIndex[t];
        const double widen = (double) g_widen_allowed[t], growth = (double) g_growth_allowed[t];
        /* the members: every particle of the leaf's subtree with a parent and mass, inserted ones included; each must be at
           the current time (true after a build's full drift; a moment refresh mid-step may find one behind, and then the
           map stays off until the next build) */
        std::vector<double> mpos, mreach, mtest;
        std::vector<int> mtype;
        double t_reach_max = 0;
        const int stop = Nodes[t_node].u.d.sibling;
        int p = Nodes[t_node].u.d.nextnode;
        while(p >= 0 && p != stop) {
            if(p < tree_slots) {
                if(Father[p] >= 0 && P[p].Mass > 0) {
                    if(P[p].Ti_current != All.Ti_Current) {behind = 1;}
                    const double h = SOURCE_REACH_RADIUS_MARGIN * force_hmax_per_type_particle_radius(p);
                    g_covered[p] = float_at_or_below(h);
                    const double g = grown_radius(h, growth);
                    if(!(g >= 0)) {failed = 1;}
                    const double test = g_scale_max * g + widen;
                    for(int k = 0; k < 3; k++) {mpos.push_back((double) P[p].Pos[k]);}
                    mreach.push_back(h); mtest.push_back(test); mtype.push_back((int) P[p].Type);
                    if(test > t_reach_max) {t_reach_max = test;}
                }
                p = Nextnode[p];
            } else if(p >= All.TreeNodeIndexBase && p < local_end) {
                p = Nodes[p].u.d.nextnode;
            } else {
                break;   /* an own leaf holds only local nodes and particles */
            }
        }
        const int nm = (int) mreach.size();
        const double tc[3] = {(double) Nodes[t_node].center[0], (double) Nodes[t_node].center[1], (double) Nodes[t_node].center[2]};
        const double th = 0.5 * (double) Nodes[t_node].len, thw[3] = {th, th, th};
        if(nm == 0) {continue;}
        /* member offsets from the leaf's centre, under the box wrap */
        std::vector<double> moff(3 * (size_t) nm);
        for(int m = 0; m < nm; m++) {
            double d[3] = {mpos[3 * m] - tc[0], mpos[3 * m + 1] - tc[1], mpos[3 * m + 2] - tc[2]};
            NEAREST_XYZ(d[0], d[1], d[2], 1);
            for(int k = 0; k < 3; k++) {moff[3 * m + k] = d[k];}
        }
        for_remote_leaves_near(tc, thw, t_reach_max, [&](int u, int u_node) {
            const double uc[3] = {(double) Nodes[u_node].center[0], (double) Nodes[u_node].center[1], (double) Nodes[u_node].center[2]};
            const double uh = 0.5 * (double) Nodes[u_node].len;
            SourceReachEntry e;
            e.t_leaf = t; e.u_leaf = u;
            double lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY}, rr[6] = {-1, -1, -1, -1, -1, -1};
            int any = 0;
            for(int m = 0; m < nm; m++) {
                if(!gx_extended_overlap_wrap_and_test(uc[0] - mpos[3 * m], uc[1] - mpos[3 * m + 1], uc[2] - mpos[3 * m + 2], uh, uh, uh, mtest[m])) {continue;}
                any = 1;
                for(int k = 0; k < 3; k++) {lo[k] = DMIN(lo[k], moff[3 * m + k]); hi[k] = DMAX(hi[k], moff[3 * m + k]);}
                if(mreach[m] > rr[mtype[m]]) {rr[mtype[m]] = mreach[m];}
            }
            if(!any) {return;}
            for(int k = 0; k < 3; k++) {e.lo[k] = float_at_or_below(lo[k]); e.hi[k] = float_at_or_above(hi[k]);}
            for(int k = 0; k < 6; k++) {e.reach[k] = (rr[k] >= 0) ? float_at_or_above(rr[k]) : -1.0f;}
            out.push_back(e);
        });
        } catch(...) {failed = 1;}
    }
    return failed ? 1 : (behind ? 2 : 0);
}

inline bool by_u_then_t(const SourceReachEntry &a, const SourceReachEntry &b) {return a.u_leaf < b.u_leaf || (a.u_leaf == b.u_leaf && a.t_leaf < b.t_leaf);}

void release_all(void)
{
    g_valid = 0;
    std::vector<SourceReachEntry>().swap(g_recv);
    std::vector<OwnLeaf>().swap(g_own);
    std::vector<float>().swap(g_covered);
    std::vector<float>().swap(g_widen_allowed);
    std::vector<float>().swap(g_growth_allowed);
    std::vector<unsigned char>().swap(g_stale);
    std::vector<float>().swap(g_top_enclose);
    g_map = nullptr;
}

/* A build that did not complete, from the vote all ranks agree on: 1 = not possible on this tree (a partial tree, members
 * behind the current time), retried at the next tree; 0 = it failed doing the work (allocation, a reach that cannot be
 * bounded, an exchange too large, or a configuration where it can never work): its time is charged as the map's cost and
 * the map is off until the corridor imbalance again exceeds both that cost and the imbalance it was wanted at. */
void build_failed(int vote, double t_call)
{
    g_tree_ti = -1;   /* not repeated before the next tree */
    if(vote == 0 && g_wanted) {
        const double t_failed = MPI_Wtime() - t_call;
        g_b_local = g_map_this_tree ? g_b_local + t_failed : t_failed;   /* every attempt of this tree, as for a successful build */
        g_wanted = 0; g_E_nogain = g_E_off_last;
        if(ThisTask == 0) {printf("source-reach map: off (its build failed)\n"); fflush(stdout);}
    }
    release_all();
}

/* A time in microseconds for the gate's 64-bit vote. */
inline long long vote_us(double t) {return (t > 0) ? (long long) (t * 1.0e6) : 0;}

} /* namespace */

/* a tree build starts here, or the tree is freed: only a completed build or moment refresh re-arms the build at a discovery */
void source_reach_map_invalidate(void) {g_valid = 0; g_tree_ti = -1; g_t_tree0 = MPI_Wtime();}


void source_reach_map_build(int usable)
{
    g_valid = 0;
    g_now_ti = -1;
    const double t_call = MPI_Wtime();
    if(g_t_tree0 > g_t_lastbuild_end) {   /* a new tree (not a moment refresh): close the last tree's account */
        g_tb_local = t_call - g_t_tree0;
        if(g_map_this_tree) {
            /* saved = the imbalance the same discoveries had without the map, at the unmapped rate, minus what they had with it */
            const double saved = ((g_n_off_last > 0) ? g_E_off_last / g_n_off_last : 0.0) * g_n_tree - g_E_tree;
            if(!(saved > g_B_all)) {   /* the map did not save more than it cost: drop it */
                g_wanted = 0; g_E_nogain = g_E_off_last;
                if(ThisTask == 0) {printf("source-reach map: off (saved %g s, cost %g s per tree)\n", saved, g_B_all); fflush(stdout);}
            }
        } else {g_E_off_last = g_E_tree; g_n_off_last = g_n_tree;}
        g_E_tree = 0; g_n_tree = 0; g_map_this_tree = 0;
    }
    g_t_lastbuild_end = t_call;   /* this tree's account is open: a later call before the next tree build is a moment refresh */
    g_tree_ti = All.Ti_Current; g_tree_usable = usable;
    if(!g_wanted) {release_all(); return;}   /* nothing built (rank-uniform) */
    /* each vote: 2 = go on, 1 = not possible on this tree, 0 = failed (build_failed); MPI_MIN takes the worst */
    int ok = (usable && NTopleaves > 0 && All.TreeNodeIndexBase > 0 && Nodes != NULL && Nextnode != NULL) ? 2 : 1;
    if(NTask <= 1) {ok = 0;}   /* no other rank to route to */
#if defined(BOX_SHEARING) && (BOX_SHEARING > 1)
    ok = 0;   /* the shear offset moves the radial wrap over the tree's lifetime, which no particle motion bound covers: node bound */
#endif
    int ok_all = 0;
    MPI_Allreduce(&ok, &ok_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(ok_all < 2) {build_failed(ok_all, t_call); return;}

    g_map = &mode_b_topleaf_map_current();
    g_build_ti = All.Ti_Current;
    tree_lifetime_ensure();   /* collective, rank-uniform here: computed by the LET build, or now */
    g_lifetime_drift = tree_lifetime_drift();
    g_scale_max = gizmo_ghost_safety_factor() * gizmo_ghost_safety_factor();
    std::vector<int> send_counts(NTask, 0), recv_counts(NTask, 0), send_offsets(NTask, 0), recv_offsets(NTask, 0);
    std::vector<SourceReachEntry> outgoing;
    try {
        g_widen_allowed.assign(NTopleaves, 0.0f);
        g_growth_allowed.assign(NTopleaves, 0.0f);
        g_stale.assign(NTopleaves, 0);
        g_covered.assign(All.TreeParticleSlots > 0 ? All.TreeParticleSlots : 0, -1.0f);
        g_own.clear();
        for(int t = 0; t < NTopleaves; t++) {
            const int no = DomainNodeIndex[t];
            /* every rank holds the owner's speed and growth bounds here, bit-identical, just exchanged with the top-leaf moments,
               so every rank rounds them to the same allowance the owner's lists were tested with (or less) */
            g_widen_allowed[t] = float_at_or_below(TREE_NODE_WIDENING_DELTA((double) Extnodes[no].vmax, g_lifetime_drift));
            g_growth_allowed[t] = float_at_or_below((double) Extnodes[no].divVmax * g_lifetime_drift);
            if(DomainTask[t] == ThisTask) {
                OwnLeaf own; own.leaf = t; own.half_len = float_at_or_below(0.5 * (double) Nodes[no].len);
                own.recv_first = own.recv_end = 0;
                g_own.push_back(own);
            }
        }
        /* pruning sizes of the replicated top-level tree, from the leaves up (children sit above their father in Nodes) */
        const int base = All.TreeNodeIndexBase;
        g_top_enclose.assign(NTopnodes, 0.0f);
        for(int k = 0; k < NTopnodes; k++) {g_top_enclose[k] = float_at_or_above(0.5 * (double) Nodes[base + k].len);}
        for(int k = NTopnodes - 1; k >= 1; k--) {
            const int f = Nodes[base + k].u.d.father - base;
            if(f < 0 || f >= NTopnodes) {continue;}
            double d = 0;
            for(int c = 0; c < 3; c++) {d = DMAX(d, fabs((double) Nodes[base + k].center[c] - (double) Nodes[base + f].center[c]));}
            const double need = d + (double) g_top_enclose[k];
            if(need > (double) g_top_enclose[f]) {g_top_enclose[f] = float_at_or_above(need);}
        }
#ifdef _OPENMP
        std::vector<std::vector<SourceReachEntry>> per_thread(omp_get_max_threads());
#else
        std::vector<std::vector<SourceReachEntry>> per_thread(1);
#endif
        const int lists = build_own_lists(per_thread);
        if(lists == 1) {ok = 0;} else if(lists == 2 && ok > 1) {ok = 1;}
        size_t total = 0;
        for(auto &v : per_thread) {total += v.size();}
        std::vector<SourceReachEntry> sent; sent.reserve(total);
        for(auto &v : per_thread) {sent.insert(sent.end(), v.begin(), v.end()); std::vector<SourceReachEntry>().swap(v);}
        /* to each owner of a U, its entries, grouped by destination */
        std::vector<long long> per_dest(NTask, 0);
        for(const auto &e : sent) {per_dest[DomainTask[e.u_leaf]]++;}
        long long off = 0;
        for(int r = 0; r < NTask; r++) {
            if(per_dest[r] * (long long) sizeof(SourceReachEntry) > (long long) INT_MAX || off * (long long) sizeof(SourceReachEntry) > (long long) INT_MAX) {ok = 0;}
            send_counts[r] = (int) (per_dest[r] * (long long) sizeof(SourceReachEntry));
            send_offsets[r] = (int) (off * (long long) sizeof(SourceReachEntry));
            off += per_dest[r];
        }
        outgoing.resize(sent.size());
        std::vector<long long> cursor(NTask, 0);
        for(int r = 1; r < NTask; r++) {cursor[r] = cursor[r - 1] + per_dest[r - 1];}
        for(const auto &e : sent) {outgoing[cursor[DomainTask[e.u_leaf]]++] = e;}
    } catch(const std::bad_alloc &) {ok = 0;}

    /* every rank agrees before any data moves: one rank that could not build its lists turns the map off everywhere */
    MPI_Allreduce(&ok, &ok_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(ok_all < 2) {build_failed(ok_all, t_call); return;}
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    long long recv_bytes = 0;
    for(int r = 0; r < NTask; r++) {
        recv_offsets[r] = (int) recv_bytes;
        recv_bytes += recv_counts[r];
        if(recv_bytes > (long long) INT_MAX) {ok = 0; break;}
    }
    try {if(ok) {g_recv.resize((size_t) (recv_bytes / (long long) sizeof(SourceReachEntry)));}} catch(const std::bad_alloc &) {ok = 0;}
    MPI_Allreduce(&ok, &ok_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(ok_all < 2) {build_failed(ok_all, t_call); return;}
    MPI_Alltoallv(outgoing.data(), send_counts.data(), send_offsets.data(), MPI_BYTE,
                  g_recv.data(), recv_counts.data(), recv_offsets.data(), MPI_BYTE, MPI_COMM_WORLD);
    std::vector<SourceReachEntry>().swap(outgoing);
    {
        std::sort(g_recv.begin(), g_recv.end(), by_u_then_t);
        long long a = 0;
        for(auto &own : g_own) {
            while(a < (long long) g_recv.size() && g_recv[a].u_leaf < own.leaf) {a++;}
            own.recv_first = a;
            while(a < (long long) g_recv.size() && g_recv[a].u_leaf == own.leaf) {a++;}
            own.recv_end = a;
        }
    }
    g_valid = 1;
    g_e_fresh = -1; g_E_decay = 0;
    const double t_build = MPI_Wtime() - t_call;
    g_b_local = g_map_this_tree ? g_b_local + t_build : t_build;   /* every map build of this tree (moment refreshes too) */
    g_map_this_tree = 1;
}

void source_reach_map_build_if_due(void)
{
    if(!g_wanted || g_valid || g_tree_ti != All.Ti_Current) {return;}   /* rank-uniform */
    g_E_off_last = g_E_tree; g_n_off_last = g_n_tree; g_E_tree = 0; g_n_tree = 0;   /* the tree so far ran without a map */
    source_reach_map_build(g_tree_usable);
    g_tree_ti = -1;   /* tried once for this tree: a failed build is not repeated before the next one */
}

void source_reach_map_gate_fill(long long *v, double t_discovery)
{
    v[0] = -vote_us(t_discovery);                            /* -> slowest rank */
    v[1] = vote_us(t_discovery);                             /* -> fastest rank */
    v[2] = -vote_us(g_tb_local);                             /* -> slowest tree build */
    v[3] = (g_b_local < 0) ? 1 : -vote_us(g_b_local);        /* -> slowest map build; 1 = never built */
}

void source_reach_map_gate_apply(const long long *a, int symmetric)
{
    if(!symmetric) {return;}
    const double tmax = -a[0] * 1.0e-6, tmin = a[1] * 1.0e-6, tb = -a[2] * 1.0e-6;
    g_B_all = (a[3] > 0) ? -1.0 : -a[3] * 1.0e-6;
    g_E_tree += DMAX(tmax - tmin, 0.0); g_n_tree++;
    if(g_map_this_tree && g_valid) {
        /* a map decays on a long-lived tree (members past what it allowed for fall back to the node bound); once the imbalance it has
           let back exceeds what a tree build and a map build cost, the tree is rebuilt, and the map with it (rank-uniform) */
        const double e = DMAX(tmax - tmin, 0.0);
        if(g_e_fresh < 0) {g_e_fresh = e;}
        g_E_decay += DMAX(e - g_e_fresh, 0.0);
        const double rebuild_cost = tb + DMAX(g_B_all, 0.0);
        if(g_E_decay > rebuild_cost && !TreeReconstructFlag) {
            TreeReconstructFlag = 1;
            if(ThisTask == 0) {printf("source-reach map: tree rebuild requested (map decay %g s > tree + map build %g s)\n", g_E_decay, rebuild_cost); fflush(stdout);}
            g_E_decay = 0;
        }
    }
    if(g_wanted || g_map_this_tree) {return;}
    /* the map's measured cost per tree, or before it was ever built the tree build it is part of */
    const double cost = (g_B_all >= 0) ? g_B_all : tb;
    if(g_E_tree > cost && g_E_tree > g_E_nogain) {
        g_wanted = 1;
        if(ThisTask == 0) {printf("source-reach map: on (corridor imbalance %g s this tree > cost %g s)\n", g_E_tree, cost); fflush(stdout);}
    }
}

void source_reach_map_prepare(void)
{
    if(!g_valid || g_now_ti == All.Ti_Current) {return;}
    g_now_drift = (All.Ti_Current > g_build_ti) ? get_drift_factor_undilated(g_build_ti, All.Ti_Current) : 0.0;
    g_now_ti = All.Ti_Current;
}

int source_reach_map_query_leaf(const double pos[3])
{
    if(!g_valid) {return -1;}
    const int leaf = domain_topleaf_of_position(pos);
    if(leaf < 0 || leaf >= NTopleaves || DomainTask[leaf] != ThisTask) {return -1;}
    const int slot = own_slot(leaf);
    if(slot < 0) {return -1;}
    /* the lists were tested against the cube at the build; the key clamps a position outside the extent to a boundary
       leaf, so the position is checked against that cube */
    const struct NODE *nop = &Nodes[DomainNodeIndex[leaf]];
    const double hw = (double) g_own[slot].half_len;
    double s[3];
    gx_extended_wrapped_separation((double) nop->center[0] - pos[0], (double) nop->center[1] - pos[1], (double) nop->center[2] - pos[2], hw, &s[0], &s[1], &s[2]);
    if(!(s[0] <= hw && s[1] <= hw && s[2] <= hw)) {return -1;}
    return slot;
}

int source_reach_map_verdict(int query_leaf, int leaf, int leaf_node, const double pos[3], double h_q,
                             unsigned int type_mask, double j_reach_scale)
{
    if(!g_valid || g_now_ti != All.Ti_Current || query_leaf < 0 || leaf < 0 || leaf >= NTopleaves) {return -1;}
    if(!(j_reach_scale <= g_scale_max) || g_stale[leaf]) {return -1;}
    const struct NODE *nop = &Nodes[leaf_node];
    const double hn = 0.5 * (double) nop->len;
    /* r < h_q: the query's own radius against the leaf's box */
    if(gx_extended_overlap_wrap_and_test((double) nop->center[0] - pos[0], (double) nop->center[1] - pos[1], (double) nop->center[2] - pos[2], hn, hn, hn, h_q)) {return 1;}
    /* r < s h_j: only while the leaf has moved and grown no more than its lists allowed for */
    const double widen = TREE_NODE_WIDENING_DELTA((double) Extnodes[leaf_node].vmax, g_now_drift);
    const double growth = (double) Extnodes[leaf_node].divVmax * g_now_drift;
    if(!(widen <= (double) g_widen_allowed[leaf]) || !(growth <= (double) g_growth_allowed[leaf])) {return -1;}
    const OwnLeaf &own = g_own[query_leaf];
    const SourceReachEntry *e = find_entry(g_recv, own.recv_first, own.recv_end, own.leaf, leaf, 1);
    if(!e) {return 0;}
    double reach = -1;
    for(int t = 0; t < 6; t++) {if((type_mask & (1u << t)) && (double) e->reach[t] > reach) {reach = (double) e->reach[t];}}
    if(reach < 0) {return 0;}
    const double grown = grown_radius(reach, growth);
    if(!(grown >= 0)) {return -1;}
    double cc[3], hw[3];
    for(int k = 0; k < 3; k++) {
        cc[k] = (double) nop->center[k] + 0.5 * ((double) e->lo[k] + (double) e->hi[k]) - pos[k];
        hw[k] = 0.5 * ((double) e->hi[k] - (double) e->lo[k]) + widen;
        hw[k] += 1.0e-12 * (fabs((double) nop->center[k]) + hw[k]);   /* the box is re-formed in double from float offsets: round outward */
    }
    return gx_extended_overlap_wrap_and_test(cc[0], cc[1], cc[2], hw[0], hw[1], hw[2], j_reach_scale * grown) ? 1 : 0;
}

int source_reach_map_mark_stale(int leaf)
{
    if(leaf < 0 || leaf >= (int) g_stale.size()) {return 0;}
    return (__atomic_exchange_n(&g_stale[leaf], (unsigned char) 1, __ATOMIC_RELAXED) == 0) ? 1 : 0;
}

int source_reach_map_is_stale(int leaf)
{
    return (leaf >= 0 && leaf < (int) g_stale.size()) ? (int) __atomic_load_n(&g_stale[leaf], __ATOMIC_RELAXED) : 0;
}

int source_reach_map_member_covered(int i, double reach, double growth_rate)
{
    if(!g_valid || g_now_ti != All.Ti_Current) {return 1;}   /* nothing will rely on the map before it is rebuilt */
    if(i < 0 || i >= (int) g_covered.size()) {return 0;}
    const double covered = (double) g_covered[i];
    if(!(covered >= 0)) {return 0;}   /* not a member at the build */
    if(reach <= covered) {return 1;}
    /* Grown by drift alone it may reach covered grown at its own rate, which the leaf's divVmax bounds; the map already allows
       for that, both where the member is listed and in the test that left it out. */
    const double grown = grown_radius(covered, growth_rate * g_now_drift);
    return (grown >= 0 && reach <= grown) ? 1 : 0;
}

void source_reach_map_swap_slots(int i, int j)
{
    if(i < 0 || j < 0 || i >= (int) g_covered.size() || j >= (int) g_covered.size()) {return;}
    const float c = g_covered[i]; g_covered[i] = g_covered[j]; g_covered[j] = c;
}

void source_reach_map_clear_slot(int i)
{
    if(i >= 0 && i < (int) g_covered.size()) {g_covered[i] = -1.0f;}
}
