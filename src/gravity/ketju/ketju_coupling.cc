/* KETJU regularized integrator coupling for GIZMO
 *
 * Implements the coupling between GIZMO's gravity tree and the MSTAR
 * regularized N-body integrator. Based on the GADGET4-KETJU architecture
 * (Mannerkoski+ 2022) but adapted for GIZMO's data structures and extended
 * to keep particles individually visible in the tree for radiation physics.
 *
 * Architecture:
 *   1. limit_timesteps(): force chain particles to shared timebin
 *   2. find_regions(): detect chain regions around massive particles
 *   3. run_integration(): collect particles, subtract tree force, run MSTAR
 *   4. set_final_velocities(): swap in true velocities after drift
 *   5. finish_step(): clear region data at end of step
 */

#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <climits>
#include <algorithm>
#include <cctype>
#include <vector>
#include <set>
#include <unordered_map>
#include <queue>
#include <map>
#include <utility>
#include <cfloat>
#include <queue>
#include <functional>
#include <array>

#include "../../declarations/allvars.h"
#include "../../core/proto.h"
#include "../../mesh/kernel.h"
#include "ketju_coupling.h"
#include "../../galaxy_sf/single_star_record.h"
#define KETJU_TREE_PAIR_KERNEL
#include "../../gravity/gravtree_force_kernel.h"
#include "ketju_tree.h"
#include "../../mesh/gpu_neighbor_list.h"   /* gizmo_motion_bound_raise */
#include "../../system/gpu_particles_arena.h"
#include "ketju_mstar_kokkos.h"
#include "ketju_star_mergers.h"

#ifdef KETJU_REGULARIZATION

/* KETJU phases all charge the misc bucket (no KETJU slots in the CPU table) */
#define CPU_KETJULIMIT   CPU_MISC
#define CPU_KETJUFIND    CPU_MISC
#define CPU_KETJUEXCH    CPU_MISC
#define CPU_KETJUMSTAR   CPU_MISC
#define CPU_KETJUSCATTER CPU_MISC
#define CPU_KETJUFINISH  CPU_MISC
#ifndef STELLA_OWNS_STAR
#define STELLA_OWNS_STAR(p) ((p).Type == 4)
#endif


/* every rank's chain spheres, for the LET sender: a node overlapping a receiver's sphere is shipped with its children,
   so that receiver's chain members see their partners as leaves (as the local walk's sphere opening does).
   KjLetSpheres holds {rank, centre x, y, z, radius} per chain. */
static std::vector<double> KjLetSpheres;
static void ketju_member_step_end_init(void);
static inline int ketju_no_live_chains(void);
void ketju_let_gather_spheres(void)
{
    ketju_member_step_end_init();
    if(ketju_no_live_chains()) {KjLetSpheres.clear(); return;}
    std::vector<double> loc; std::set<MyIDType> seen;
    for(int i = 0; i < NumPart; i++) {
        if(P[i].KetjuChainID == 0 || !(P[i].KetjuOpenRadius > 0) || !seen.insert(P[i].KetjuChainID).second) {continue;}
        const double *sph = ketju_member(i).chain_sphere;
        loc.push_back((double)ThisTask); for(int k = 0; k < 3; k++) {loc.push_back(sph[k]);}
        loc.push_back(3.0 * sph[3]); /* the walk opens around each member, reach twice the sphere radius */
    }
    int n = (int)loc.size(); std::vector<int> cnt(NTask), off(NTask + 1, 0);
    MPI_Allgather(&n, 1, MPI_INT, cnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for(int t = 0; t < NTask; t++) {off[t + 1] = off[t] + cnt[t];}
    KjLetSpheres.assign(off[NTask], 0.0);
    MPI_Allgatherv(loc.data(), n, MPI_DOUBLE, KjLetSpheres.data(), cnt.data(), off.data(), MPI_DOUBLE, MPI_COMM_WORLD);
}
int ketju_let_sphere_opens(int rank, double cx, double cy, double cz, double len)
{
    for(size_t k = 0; k + 5 <= KjLetSpheres.size(); k += 5) {
        if((int)KjLetSpheres[k] != rank) {continue;}
        double dx = cx - KjLetSpheres[k + 1], dy = cy - KjLetSpheres[k + 2], dz = cz - KjLetSpheres[k + 3];
        gravity_box_nearest_image(dx, dy, dz, -1);
        if(ketju_node_overlaps_chain_sphere(dx, dy, dz, KjLetSpheres[k + 4], len)) {return 1;}
    }
    return 0;
}

struct ketju_member_state &ketju_member(int i) {return SingleStars.ketju[single_star_slot(P[i].StarIdx)];}
/* a star's compact record, made at its first chain step; without IMF sampling it lives as long as the chain step */
static struct ketju_member_state &ketju_member_open(int i)
{
    if(P[i].StarIdx <= 0) {struct single_star_record r; memset(&r, 0, sizeof(r)); P[i].StarIdx = single_star_record_new(&r);}
    return ketju_member(i);
}
static void ketju_member_close(int i)
{
    if(P[i].StarIdx <= 0) {return;}
#ifdef GALSF_SINGLE_STARS_FROM_IMF
    memset(&ketju_member(i), 0, sizeof(struct ketju_member_state));
#else
    single_star_record_free(P[i].StarIdx); P[i].StarIdx = 0;
#endif
}


/* start-up state of the KETJU fields */
void ketju_init_particle(int i)
{
    P[i].KetjuIntegrated = 0;
    P[i].KetjuChainID = 0;
#ifndef GALSF_SINGLE_STARS_FROM_IMF
    P[i].StarIdx = 0;
#endif
    P[i].KetjuOpenRadius = 0;
}

/* softening and AGS zeta of particle p as the gravity walk uses them: [0] as target, [1] as leaf secondary */
void tree_pair_softening(int p, double *soft, double *zeta)
{
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL) || defined(GALSF_MERGER_STARCLUSTER_PARTICLES)
    soft[0] = ForceSoftening_KernelRadius(p);
#else
    soft[0] = All.ForceSoftening[P[p].Type];
#endif
    zeta[0] = 0;
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL)
    grav_target_select_soft_and_zeta(P[p].Type, P[p].AGS_zeta, soft[0], zeta[0]);
#endif
    soft[1] = P[p].ForceSoftening; zeta[1] = 0;
#if defined(ADAPTIVE_GRAVSOFT_FORGAS)
    if(P[p].Type == 0) {zeta[1] = P[p].AGS_zeta;}
#elif defined(ADAPTIVE_GRAVSOFT_FORALL)
    zeta[1] = P[p].AGS_zeta;
#endif
}

#if defined(KETJU_MERGE_STARS)
#define KJ_STAR_COLLISIONS /* Type-4 stars merge on contact (R1 + R2) inside MSTAR */
#endif

/* MSTAR is integrated in natural N-body units (pc, Msun, km/s) whatever the host units: its GBS convergence degrades
 * badly for e.g. kpc / 1e10 Msun states. Converted in place around ketju_run_integrator only; identity for pc/Msun/km/s hosts. */
#define KJ_LEN_TO_NAT  (All.UnitLength_in_cm / 3.085678e18)      /* host length unit in pc */
#define KJ_MASS_TO_NAT (All.UnitMass_in_g / 1.989e33)            /* host mass unit in Msun */
#define KJ_VEL_TO_NAT  (All.UnitVelocity_in_cm_per_s / 1.0e5)    /* host velocity unit in km/s */
#define KJ_TIME_TO_NAT (KJ_LEN_TO_NAT / KJ_VEL_TO_NAT)           /* host time unit in pc/(km/s) */
static void kj_state_change_units(struct ketju_system_physical_state *ps, int n, int to_natural)
{
    double fL = KJ_LEN_TO_NAT, fM = KJ_MASS_TO_NAT, fV = KJ_VEL_TO_NAT;
    if(!to_natural) {fL = 1./fL; fM = 1./fM; fV = 1./fV;}
    for(int i = 0; i < n; i++) {
        ps->mass[i] *= fM;
        for(int j = 0; j < 3; j++) {ps->pos[i][j] *= fL; ps->vel[i][j] *= fV;}
    }
    ps->time *= fL / fV;
}

#ifdef TEST_KETJU_MSTAR_PAIR
/* one orbit of an e = 0.9 pair through the wrapper and the unit conversion: it must come back to its start */
void ketju_test_mstar_pair(void)
{
    struct ketju_system *S = (struct ketju_system *)calloc(1, sizeof(struct ketju_system));
    kj_mstar_create_system(S, 0, 2, MPI_COMM_SELF);
    S->constants->G = All.G * KJ_LEN_TO_NAT * KJ_VEL_TO_NAT * KJ_VEL_TO_NAT / KJ_MASS_TO_NAT;
    S->constants->c = 2.9979e10 / All.UnitVelocity_in_cm_per_s * KJ_VEL_TO_NAT;
    S->options->gbs_relative_tolerance = 1e-12;
    const double m = 1.989e33 / All.UnitMass_in_g, a = All.KetjuRegionRadius, e = 0.9, GM = All.G * 2 * m, ra = a * (1 + e), va = sqrt(GM * (1 - e) / (a * (1 + e))), T = 2 * M_PI * a * sqrt(a / GM);
    struct ketju_system_physical_state *ps = S->physical_state;
    ps->time = 0; ps->mass[0] = ps->mass[1] = m;
    ps->pos[0][0] = 0.5 * ra; ps->pos[1][0] = -0.5 * ra; ps->vel[0][1] = 0.5 * va; ps->vel[1][1] = -0.5 * va;
    kj_state_change_units(ps, 2, 1);
    kj_mstar_run_integrator(S, T * KJ_TIME_TO_NAT);
    kj_state_change_units(ps, 2, 0);
    double dr[3], dv[3], r2 = 0, v2 = 0; for(int k = 0; k < 3; k++) {dr[k] = ps->pos[0][k] - ps->pos[1][k]; dv[k] = ps->vel[0][k] - ps->vel[1][k]; r2 += dr[k] * dr[k]; v2 += dv[k] * dv[k];}
    double r = sqrt(r2), af = 1 / (2 / r - v2 / GM), rv = dr[0] * dv[0] + dr[1] * dv[1] + dr[2] * dv[2], l2 = r2 * v2 - rv * rv, ef = sqrt(fmax(0., 1 - l2 / (GM * af)));
    if(ThisTask == 0) {printf("KETJU_MSTAR_PAIR da/a %.3e de %.3e\n", af / a - 1, ef - e);}
    kj_mstar_free_system(S); free(S);
}
#endif

/* Cost tracking for load balancing (keyed by first center ID per region) */
static std::unordered_map<MyIDType, double> region_previous_cost;

/* ============================================================
 *  Lightweight particle data for MPI communication
 * ============================================================ */
/* IMF siblings share P.ID and differ in ID_child_number: KETJU keys particles by both */
struct KetjuUID {MyIDType id, child; bool operator==(const KetjuUID &o) const {return id == o.id && child == o.child;}};
struct KetjuUIDHash {size_t operator()(const KetjuUID &k) const {return std::hash<MyIDType>()(k.id * 0x9E3779B97F4A7C15ULL ^ k.child);}};

struct ketju_mpi_particle {
    MyIDType ID;
    MyIDType Child; /* P.ID_child_number */
    int Type;
    int Task;
    int Index;
    double Mass;
    double Pos[3];
    double Vel[3];
    double Soft[2], Zeta[2]; /* softening/zeta as the tree uses them ([0] target, [1] secondary); fixed for stars under KETJU */
#ifdef KJ_STAR_COLLISIONS
    double CollRadius; /* contact radius, physical code length (0: never collides) */
    double BirthTime, FormDens; /* StellarAge and physical formation density, for the collision log */
#endif
};

/* ============================================================
 *  Per-particle extra data tracked alongside integrator arrays
 * ============================================================ */
struct ketju_extra_data {
    MyIDType ID;
    MyIDType Child;
    int Task;
    int Index;
    int Type;
#ifdef KJ_STAR_COLLISIONS
    double BirthTime, FormDens;
#endif
};

/* ============================================================
 *  MPI task group for parallel KETJU computation
 *  Based on public GADGET4-KETJU (Mannerkoski+ 2023)
 * ============================================================ */
/* task-group communicators by (tag, task list): every rank asks for every region's groups, so the cache is rank-uniform */
static std::map<std::pair<int, std::vector<int>>, std::pair<MPI_Group, MPI_Comm>> KetjuCommCache;
static void ketju_comm_cache_flush(void)
{
    for(auto &e : KetjuCommCache) {if(e.second.second != MPI_COMM_NULL) {MPI_Comm_free(&e.second.second);} MPI_Group_free(&e.second.first);}
    KetjuCommCache.clear();
}

struct KetjuTaskGroup {
    MPI_Group group;
    MPI_Comm comm;
    int rank;      /* rank within this group, MPI_UNDEFINED if not a member */
    int size;
    int root;      /* rank of root within the group */
    int root_sim;  /* rank of root in MPI_COMM_WORLD */

    KetjuTaskGroup() : group(MPI_GROUP_NULL), comm(MPI_COMM_NULL), rank(MPI_UNDEFINED), size(0), root(MPI_UNDEFINED), root_sim(MPI_UNDEFINED) {}

    void init(const std::vector<int> &task_sim_indices, int tag = 0) {
        auto key = std::make_pair(tag, task_sim_indices);
        auto it = KetjuCommCache.find(key);
        if(it == KetjuCommCache.end()) {
            MPI_Group world_group, g; MPI_Comm c = MPI_COMM_NULL; int me;
            MPI_Comm_group(MPI_COMM_WORLD, &world_group);
            MPI_Group_incl(world_group, task_sim_indices.size(), task_sim_indices.data(), &g);
            MPI_Group_rank(g, &me);
            if(me != MPI_UNDEFINED) {MPI_Comm_create_group(MPI_COMM_WORLD, g, tag, &c);}
            MPI_Group_free(&world_group);
            it = KetjuCommCache.emplace(key, std::make_pair(g, c)).first;
        }
        group = it->second.first; comm = it->second.second;
        MPI_Group_size(group, &size);
        MPI_Group_rank(group, &rank);
        root = 0;
        root_sim = task_sim_indices[0];
    }

    void free_comms() {
        comm = MPI_COMM_NULL; group = MPI_GROUP_NULL; /* owned by KetjuCommCache */
        rank = MPI_UNDEFINED; size = 0; root = MPI_UNDEFINED; root_sim = MPI_UNDEFINED;
    }

    /* Set both groups to have the same root task (from intersection) */
    void set_common_root(KetjuTaskGroup &other) {
        if(group == MPI_GROUP_NULL || other.group == MPI_GROUP_NULL) return;
        MPI_Group world_group, intersection;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        MPI_Group_intersection(group, other.group, &intersection);
        if(intersection != MPI_GROUP_EMPTY) {
            int root_inter = 0;
            MPI_Group_translate_ranks(intersection, 1, &root_inter, group, &root);
            MPI_Group_translate_ranks(intersection, 1, &root_inter, other.group, &other.root);
            MPI_Group_translate_ranks(group, 1, &root, world_group, &root_sim);
            MPI_Group_translate_ranks(other.group, 1, &other.root, world_group, &other.root_sim);
        }
        MPI_Group_free(&intersection);
        MPI_Group_free(&world_group);
    }

    int is_member() const { return rank != MPI_UNDEFINED; }
    int is_root() const { return rank == root; }
};

/* Compute info for scheduling regions across tasks */
struct region_compute_info {
    int compute_sequence_position; /* which sequential run this region belongs to */
    int first_task_index;          /* first MPI task in compute group */
    int final_task_index;          /* last MPI task in compute group */
};

/* ============================================================
 *  A single KETJU region
 * ============================================================ */
struct KetjuRegion {
    std::vector<ketju_mpi_particle> centers;      /* BH/massive-star region centers */
    std::vector<ketju_mpi_particle> all_particles; /* all chain members (gathered on root) */
    std::set<int> local_member_indices;            /* local P[] indices in this region */
    int total_particle_count;
    int num_pn_particles;                          /* count of PN-enabled (SMBH) particles */

    /* MPI task groups for parallel computation */
    KetjuTaskGroup affected_tasks;  /* tasks that hold particles in this region */
    KetjuTaskGroup compute_tasks;   /* tasks that run the integrator */
    region_compute_info compute_info;
    std::vector<int> affected_sim_indices;       /* world ranks of affected tasks */
    std::vector<int> particle_counts_on_affected; /* particle count per affected task */

    /* integrator state (only allocated on compute root) */
    struct ketju_system *integrator;
    std::vector<ketju_extra_data> extra_data;
    std::vector<double> true_vel;  /* MSTAR end-of-step CoM-relative velocities before the second negative half-kick (compute tasks) */
    std::vector<double> coll_radius; /* MSTAR contact radii in its natural units (compute tasks) */

    /* CoM frame data (stored at setup, used in scatter) */
    double com_pos[3];
    double com_vel[3];
    integertime ti_step;     /* integer timestep for this region */

    KetjuRegion() : total_particle_count(0), num_pn_particles(0), integrator(NULL), ti_step(0) {
        for(int j = 0; j < 3; j++) { com_pos[j] = 0; com_vel[j] = 0; }
        compute_info.compute_sequence_position = 0;
        compute_info.first_task_index = 0;
        compute_info.final_task_index = 0;
    }
    ~KetjuRegion() {
        if(integrator) { kj_mstar_free_system(integrator); free(integrator); }
        affected_tasks.free_comms();
        compute_tasks.free_comms();
    }
};

/* ============================================================
 *  Loop scheduling for parallel N² force computation
 *  Distributes the upper triangle of the N×N pair loop across
 *  num_proc tasks. Returns the start index for edge_index-th block.
 *  Based on public GADGET4-KETJU (Mannerkoski+ 2023).
 * ============================================================ */
static int loop_scheduling_block_edge(int Nloop, int num_proc, int edge_index)
{
    if(Nloop <= 0 || num_proc <= 0) return 0;
    if(num_proc >= Nloop) num_proc = Nloop;
    if(edge_index <= 0) return 0;
    if(edge_index >= num_proc) return Nloop - 1;
    double P = (double)num_proc, N = (double)Nloop;
    return (int)ceil(N - 0.5 - sqrt(N * (N - 1) * (P - edge_index) / P + 0.25));
}

/* ============================================================
 *  Module-level state (persists within a timestep)
 * ============================================================ */
static std::vector<KetjuRegion> ActiveRegions;
static int KetjuNumMultiRegions = 0; /* regions with >= 2 members this step (rank-uniform) */
static long long KetjuMemberStepEnd = -1; /* latest step end of any integrated chain member (rank-uniform); -1 = not yet known */
static int KetjuCandidateFlag = 1; static integertime KetjuCandidateTi = -1; /* any active chain candidate at this time, from the gravity call's reduction */
static void ketju_member_step_end_init(void) /* first call, also after a restart: from the flagged members */
{
    if(KetjuMemberStepEnd >= 0) {return;}
    if(RestartFlag != 1) {KetjuMemberStepEnd = 0; return;} /* no chain before the first step */
    long long e = 0;
    for(int i = 0; i < NumPart; i++) {if(P[i].KetjuIntegrated && P[i].KetjuChainID != 0) {e = std::max(e, (long long)P[i].Ti_begstep + (long long)P[i].dt_step);}}
    MPI_Allreduce(&e, &KetjuMemberStepEnd, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
}
static inline int ketju_no_live_chains(void) {return KetjuMemberStepEnd == 0 || (long long)All.Ti_Current > KetjuMemberStepEnd;}

/* Wall-clock laps into the KETJU cpu.txt categories; WallclockTime is advanced by the same amount so the
 * enclosing GIZMO timer (gravity, drift) does not count it again. The destructor laps into `rest`. */
struct KjClock {
    double t; int rest;
    explicit KjClock(int rest_cat) : t(my_second()), rest(rest_cat) {}
    void lap(int cat) {double n = my_second(), dt = n - t; cpu_charge_child(cat, dt); t = n;}
    ~KjClock() {lap(rest);}
};
static double KetjuCollisionEnergy = 0; /* energy change by star collisions since the last KETJU STATS line (compute roots) */

/* Cache for gather_chain_centers result — shared between ketju_limit_timesteps
 * and ketju_find_regions to avoid a redundant MPI_Allgatherv on COMM_WORLD */
static std::vector<ketju_mpi_particle> CachedChainCenters;
static int CachedChainCentersValid = 0;
static std::vector<std::vector<ketju_mpi_particle>> CachedCenterGroups; /* the limiter's orbit grouping of the same centres */

/* ketju_limit_timesteps tallies since the last KETJU STATS reduction: limited particles moved to a
 * shorter / longer bin, timestep regions set, and those whose bin a limiting particle set (the
 * last two counted on task 0 only, since every task computes the same bins) */
static long long LimitTally[4] = {0, 0, 0, 0};

/* ============================================================
 *  Cost-based scheduling (Phase B)
 *  Based on public GADGET4-KETJU (Mannerkoski+ 2023).
 *  Estimates per-region cost and distributes compute tasks.
 * ============================================================ */
static double region_cost_estimate(int region_index)
{
    if(ActiveRegions[region_index].centers.empty()) return 1.0;
    MyIDType key = ActiveRegions[region_index].centers[0].ID;
    auto it = region_previous_cost.find(key);
    if(it != region_previous_cost.end()) return it->second;
    /* no previous cost — estimate from N² */
    double N = ActiveRegions[region_index].total_particle_count;
    return 20.0 * N * N;
}

/* Number of sequential rounds chosen by allocate_compute_tasks_for_regions.
 * Each region is assigned a round index (compute_sequence_position) and
 * ketju_run_integration iterates over rounds in outer loop, so regions in
 * the same round run in parallel while tasks move to the next round only
 * after all regions in the current round finish. Matches the multi-round
 * scheduler used by GADGET4-KETJU / gadget_tnt_new for better load balance. */
static int g_ketju_num_sequential_runs = 0;

/* Bucket regions into `num_runs` rounds using a longest-processing-time (LPT)
 * heuristic: repeatedly place the most-expensive unassigned region into the
 * least-loaded bucket. Each bucket is capped at NTask regions so every
 * region gets at least one task. */
static std::vector<std::vector<int>> lpt_bucket_regions(int num_runs)
{
    int n_regions = (int)ActiveRegions.size();
    typedef std::pair<double, int> cost_pair;   /* (cost, index) */

    /* max-heap over regions by cost */
    std::priority_queue<cost_pair> region_pq;
    for(int r = 0; r < n_regions; r++) region_pq.push({region_cost_estimate(r), r});

    /* min-heap over buckets by accumulated cost */
    std::priority_queue<cost_pair, std::vector<cost_pair>, std::greater<cost_pair>> bucket_pq;
    for(int b = 0; b < num_runs; b++) bucket_pq.push({0.0, b});

    std::vector<std::vector<int>> buckets(num_runs);
    while(!region_pq.empty() && !bucket_pq.empty()) {
        double rc = region_pq.top().first;
        int    r  = region_pq.top().second;
        region_pq.pop();
        double bc = bucket_pq.top().first;
        int    b  = bucket_pq.top().second;
        bucket_pq.pop();
        buckets[b].push_back(r);
        /* only push this bucket back if it can still fit more regions */
        if((int)buckets[b].size() < NTask)
            bucket_pq.push({bc + rc, b});
    }
    return buckets;
}

/* Estimate wall-time for a given bucketing: sum over buckets of the
 * max (region_cost / tasks_allocated_to_region). Scaling is cut off at
 * N_particles / particles_per_task_scaling_limit because the integrator
 * stops getting faster past that point. */
static double estimate_total_time_for(const std::vector<std::vector<int>>& buckets)
{
    const int ppts_limit = 7; /* see Mannerkoski+2022 scaling data */
    double total = 0;
    for(const auto& bucket : buckets) {
        if(bucket.empty()) continue;
        double bucket_cost = 0;
        for(int r : bucket) bucket_cost += region_cost_estimate(r);
        if(bucket_cost <= 0) bucket_cost = 1.0;

        double bucket_max = 0;
        for(int r : bucket) {
            double frac = region_cost_estimate(r) / bucket_cost;
            int n_tasks = DMAX(1, (int)(frac * NTask + 0.5));
            int npart = ActiveRegions[r].total_particle_count;
            int max_useful = DMAX(1, npart / ppts_limit);
            if(n_tasks > max_useful) n_tasks = max_useful;
            double t_reg = region_cost_estimate(r) / (double)n_tasks;
            if(t_reg > bucket_max) bucket_max = t_reg;
        }
        total += bucket_max;
    }
    return total;
}

/* Assign MPI task ranges to the regions in one bucket, proportional to cost. */
static void assign_tasks_for_bucket(const std::vector<int>& bucket_regions, int seq_pos)
{
    if(bucket_regions.empty()) return;
    double bucket_cost = 0;
    for(int r : bucket_regions) bucket_cost += region_cost_estimate(r);
    if(bucket_cost <= 0) bucket_cost = 1.0;

    int min_per_task = (int)DMAX(All.KetjuMinParticlesPerTask, 2);
    int tasks_used = 0;
    for(size_t i = 0; i < bucket_regions.size(); i++) {
        int r = bucket_regions[i];
        double frac = region_cost_estimate(r) / bucket_cost;
        int n_tasks = DMAX(1, (int)(frac * NTask + 0.5));

        /* respect minimum particles per task */
        if(ActiveRegions[r].total_particle_count > 0) {
            int max_tasks = ActiveRegions[r].total_particle_count / min_per_task;
            if(max_tasks < 1) max_tasks = 1;
            if(n_tasks > max_tasks) n_tasks = max_tasks;
        }

        /* don't exceed available tasks in this bucket. If the bucket is
         * already full (tasks_used >= NTask) we still need to assign this
         * region somewhere valid — co-locate it on the last task. Without
         * this clamp the indices go out of bounds (e.g. range=[4,4] on
         * NTask=4) and MPI_Group_incl/MPI_Comm_create_group return a
         * corrupt group → segfault in MPI_Group_intersection later. */
        if(tasks_used >= NTask) {
            ActiveRegions[r].compute_info.first_task_index          = NTask - 1;
            ActiveRegions[r].compute_info.final_task_index          = NTask - 1;
            ActiveRegions[r].compute_info.compute_sequence_position = seq_pos;
            continue;
        }
        if(tasks_used + n_tasks > NTask) n_tasks = NTask - tasks_used;
        if(n_tasks < 1) n_tasks = 1;

        ActiveRegions[r].compute_info.first_task_index          = tasks_used;
        ActiveRegions[r].compute_info.final_task_index          = tasks_used + n_tasks - 1;
        ActiveRegions[r].compute_info.compute_sequence_position = seq_pos;
        tasks_used += n_tasks;
    }
}

static void allocate_compute_tasks_for_regions(void)
{
    int n_regions = (int)ActiveRegions.size();
    if(n_regions == 0) { g_ketju_num_sequential_runs = 0; return; }

    /* Sweep num_runs from the minimum (limited by NTask per bucket) upward
     * and pick the one with the smallest estimated total wall-time. We
     * exit early when adding another round stops improving the estimate
     * (the cost surface is empirically monotone past the optimum). */
    int min_runs = (n_regions + NTask - 1) / NTask;
    if(min_runs < 1) min_runs = 1;

    double best_time = DBL_MAX;
    std::vector<std::vector<int>> best_buckets;
    for(int num_runs = min_runs; num_runs <= n_regions; num_runs++) {
        auto buckets = lpt_bucket_regions(num_runs);
        double t = estimate_total_time_for(buckets);
        if(t < best_time) {
            best_time   = t;
            best_buckets = std::move(buckets);
        } else {
            break;
        }
    }

    for(size_t s = 0; s < best_buckets.size(); s++) {
        assign_tasks_for_bucket(best_buckets[s], (int)s);
    }
    g_ketju_num_sequential_runs = (int)best_buckets.size();
}

static void update_region_costs(void)
{
    /* After integration, store actual cost for next step. The cost is only
     * known on each region's compute_root (it has the integrator), but the
     * map MUST be globally consistent — otherwise next step's
     * region_cost_estimate / allocate_compute_tasks_for_regions returns
     * different compute_info.first/final_task_index on different tasks,
     * which then produces different compute groups and deadlocks
     * MPI_Comm_create_group inside compute_tasks.init.
     *
     * Pack one cost per region into a dense array indexed by region, then
     * MPI_Allreduce(MAX). Non-roots contribute 0 → the compute_root's value
     * wins. */
    int n_regions = (int)ActiveRegions.size();
    if(n_regions == 0) return;

    std::vector<double> local_costs(n_regions, 0.0);
    for(int r = 0; r < n_regions; r++) {
        if(!ActiveRegions[r].compute_tasks.is_root()) continue;
        if(!ActiveRegions[r].integrator) continue;
        if(ActiveRegions[r].centers.empty()) continue;

        local_costs[r] = (double)(ActiveRegions[r].integrator->perf->successful_steps +
                                  ActiveRegions[r].integrator->perf->failed_steps) *
                         ActiveRegions[r].total_particle_count * ActiveRegions[r].total_particle_count;
    }

    std::vector<double> global_costs(n_regions, 0.0);
    MPI_Allreduce(local_costs.data(), global_costs.data(), n_regions, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    for(int r = 0; r < n_regions; r++) {
        if(ActiveRegions[r].centers.empty()) continue;
        if(global_costs[r] <= 0) continue;
        MyIDType key = ActiveRegions[r].centers[0].ID;
        region_previous_cost[key] = global_costs[r];
    }
}

/* ============================================================
 *  Helper: is this particle a valid chain center?
 * ============================================================ */
static int is_chain_center(int i)
{
    if(STELLA_OWNS_STAR(P[i]) && All.KetjuMinStarMass > 0 && P[i].Mass >= All.KetjuMinStarMass) return 1; /* stars */
    return 0;
}

/* Synchronised this step: its bin is active, so find_timesteps has just started its step at
 * All.Ti_Current. Only such particles may be retimed or enter a region (see ketju_limit_timesteps). */
static int is_synchronized(int i)
{
    return (P[i].Mass > 0 && TimeBinActive[P[i].TimeBin]) ? 1 : 0;
}

/* ============================================================
 *  Helper: is this particle eligible for chain membership?
 * ============================================================ */
static int is_chain_eligible(int i)
{
    /* STARFORGE: every Type 5 sink is a chain-member candidate (regardless of stage) */
    if(STELLA_OWNS_STAR(P[i])) return 1;
    return 0;
}



/* ============================================================
 *  Helper: distance between two positions (with box wrapping)
 * ============================================================ */
static double particle_distance(double *pos1, double *pos2)
{
    double dx, dy, dz;
    dx = pos1[0] - pos2[0];
    dy = pos1[1] - pos2[1];
    dz = pos1[2] - pos2[2];
#ifdef BOX_PERIODIC
    NEAREST_XYZ(dx, dy, dz, -1);
#endif
    return sqrt(dx*dx + dy*dy + dz*dz);
}



/* ============================================================
 *  PHASE 2: Find chain centers and build regions
 * ============================================================ */

static std::vector<ketju_mpi_particle> KjFoundCenters; /* this step's chain centers on every task */

/* carried by the reduction gravity_tree does anyway, so a step with no candidate anywhere adds no collective */
int ketju_active_candidates(void)
{
    if(All.KetjuRegionRadius <= 0) {return 0;}
    for(int i : ActiveParticleList) {if(is_chain_center(i) && is_synchronized(i)) {return 1;}}
    return 0;
}
void ketju_note_candidates(int any) {KetjuCandidateFlag = any; KetjuCandidateTi = All.Ti_Current;}

/* Gather all synchronised chain centers across MPI tasks */
static std::vector<ketju_mpi_particle> gather_chain_centers(void)
{
    if(KetjuCandidateTi == All.Ti_Current && !KetjuCandidateFlag) {return {};}
    /* count local centers */
    int n_local = 0;
    for(int i : ActiveParticleList) {
        if(is_chain_center(i) && is_synchronized(i)) n_local++;
    }

    /* fill local array */
    std::vector<ketju_mpi_particle> local_centers(n_local);
    int k = 0;
    for(int i : ActiveParticleList) {
        if(is_chain_center(i) && is_synchronized(i)) {
            local_centers[k].ID = P[i].ID; local_centers[k].Child = P[i].ID_child_number;
            local_centers[k].Type = P[i].Type;
            local_centers[k].Task = ThisTask;
            local_centers[k].Index = i;
            local_centers[k].Mass = P[i].Mass;
            for(int j = 0; j < 3; j++) {
                local_centers[k].Pos[j] = P[i].Pos[j];
                local_centers[k].Vel[j] = P[i].Vel[j];
            }
            k++;
        }
    }

    /* gather counts */
    std::vector<int> counts(NTask), displs(NTask);
    MPI_Allgather(&n_local, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int n_total = 0;
    for(int t = 0; t < NTask; t++) {
        displs[t] = n_total;
        n_total += counts[t];
    }
    if(n_total == 0) return {};

    /* gather data — use raw bytes since we don't have a registered MPI type */
    int local_bytes = n_local * sizeof(ketju_mpi_particle);
    std::vector<int> byte_counts(NTask), byte_displs(NTask);
    for(int t = 0; t < NTask; t++) {
        byte_counts[t] = counts[t] * sizeof(ketju_mpi_particle);
        byte_displs[t] = displs[t] * sizeof(ketju_mpi_particle);
    }

    std::vector<ketju_mpi_particle> all_centers(n_total);
    MPI_Allgatherv(local_centers.data(), local_bytes, MPI_BYTE,
                   all_centers.data(), byte_counts.data(), byte_displs.data(),
                   MPI_BYTE, MPI_COMM_WORLD);

    return all_centers;
}


/* ============================================================
 *  Orbit-based subsystem membership for Type-4 stars (BIFROST, Rantala+ 2023 Sec 3.4):
 *  chain centres are grouped by pair orbital elements (bound with a < r_region, or a predicted
 *  pericentre < r_region inside an anticipatory window); only those centres become members,
 *  so a dense cluster does not merge into one geometric super-region.
 * ============================================================ */
static const double KJ_CAPTURE_K = 16.0; /* capture window in units of r_ngb; no pair beyond it forms a subsystem */
static int pair_forms_subsystem(const ketju_mpi_particle &a, const ketju_mpi_particle &b)
{
    const double r_ngb = All.KetjuRegionRadius;
    double dr[3], dv[3]; double r2 = 0, v2 = 0;
    for(int k = 0; k < 3; k++) {
        dr[k] = a.Pos[k] - b.Pos[k]; dv[k] = a.Vel[k] - b.Vel[k];
        r2 += dr[k] * dr[k]; v2 += dv[k] * dv[k];
    }
    double r = sqrt(r2);
    if(r < r_ngb) return 1;                       /* already deep — must be co-integrated regardless */
    if(r >= 50.0 * r_ngb) return 0;               /* pair-scan cutoff (BIFROST r_ngb,max analogue) */

    double GM = All.G * (a.Mass + b.Mass);
    if(GM <= 0 || v2 <= 0) return 0;
    double E = 0.5 * v2 - GM / r;                 /* specific orbital energy of the pair */
    if(E < 0) {
        double sma = -GM / (2.0 * E);
        if(sma < r_ngb) return 1;                 /* bound + tight: permanent member */
    }
    /* predicted pericenter from the (Keplerian 2-body) angular momentum: r_peri = (h^2/GM)/(1+e).
     * Valid for bound and hyperbolic orbits alike. */
    double rv = dr[0]*dv[0] + dr[1]*dv[1] + dr[2]*dv[2];
    double h2 = r2 * v2 - rv * rv;                /* |r x v|^2 */
    double e2 = 1.0 + 2.0 * E * h2 / (GM * GM); if(e2 < 0) e2 = 0;
    double r_peri = (h2 / GM) / (1.0 + sqrt(e2));
    if(r_peri >= r_ngb) return 0;                 /* passage never gets inside the regularized scale */

    /* anticipatory window: capture while |r|/|v_rel| < tau AND r < K*r_ngb, with tau sized so that on
     * a near-parabolic infall (v ~ sqrt(2GM/r)) the time criterion fires at r_capture ~ K * r_ngb:
     * r/v = r^1.5/sqrt(2GM) = tau  =>  r_capture = (2 GM tau^2)^(1/3) = K * r_ngb. Weak-field
     * handoffs: at K=16 the mutual force at capture/release is ~256x weaker than at r_ngb.
     * The EXPLICIT distance bound is essential: for fast HYPERBOLIC encounters (v_rel set by the
     * cluster dispersion, not by the pair potential) r/v < tau alone fires at huge distances
     * (~0.1 pc for r_ngb=500 AU), chaining hundreds of stars into giant merged subsystems. */
    const double K = KJ_CAPTURE_K;
    if(r >= K * r_ngb) return 0;
    double tau = K * sqrt(K) * r_ngb * sqrt(r_ngb) / sqrt(2.0 * GM);
    if(r / sqrt(v2) < tau) return 1;
    return 0;
}

static std::vector<std::vector<ketju_mpi_particle>> build_orbit_regions(
    const std::vector<ketju_mpi_particle> &centers)
{
    int n = centers.size();
    std::vector<int> group(n);
    for(int i = 0; i < n; i++) group[i] = i;
    auto find = [&](int x) {
        while(group[x] != x) { group[x] = group[group[x]]; x = group[x]; }
        return x;
    };
    auto unite = [&](int a, int b) { a = find(a); b = find(b); if(a != b) group[a] = b; };

    /* only pairs closer than KJ_CAPTURE_K * r_ngb can form a subsystem (bound ones with a < r_ngb lie within 2 r_ngb):
     * test pairs from neighbouring cells of that size, O(n) instead of O(n^2); the partition does not depend on pair order */
    const double cell = KJ_CAPTURE_K * All.KetjuRegionRadius;
    std::unordered_map<long long, std::vector<int>> grid;
    auto key = [](long long x, long long y, long long z) {return (x & 0x1fffff) | ((y & 0x1fffff) << 21) | ((z & 0x1fffff) << 42);};
    std::vector<long long> ci(3 * n);
    for(int i = 0; i < n; i++) {
        for(int k = 0; k < 3; k++) ci[3*i+k] = (long long)floor(centers[i].Pos[k] / cell);
        grid[key(ci[3*i], ci[3*i+1], ci[3*i+2])].push_back(i);
    }
    for(int i = 0; i < n; i++)
        for(int dx = -1; dx <= 1; dx++) for(int dy = -1; dy <= 1; dy++) for(int dz = -1; dz <= 1; dz++) {
            auto it = grid.find(key(ci[3*i] + dx, ci[3*i+1] + dy, ci[3*i+2] + dz));
            if(it == grid.end()) continue;
            for(int j : it->second) if(j > i && pair_forms_subsystem(centers[i], centers[j])) unite(i, j);
        }

    std::vector<std::vector<ketju_mpi_particle>> regions;
    std::vector<int> root_to_region(n, -1);
    for(int i = 0; i < n; i++) {
        int r = find(i);
        if(root_to_region[r] < 0) { root_to_region[r] = regions.size(); regions.push_back({}); }
        regions[root_to_region[r]].push_back(centers[i]);
    }
    return regions;
}

/* members of an orbit group: the synchronised centres this rank owns */
static std::set<int> local_members_from_group(const std::vector<ketju_mpi_particle> &group)
{
    std::set<int> members;
    for(const ketju_mpi_particle &c : group) {
        if(c.Task != ThisTask) continue;
        if(!is_chain_eligible(c.Index) || !is_synchronized(c.Index)) continue;
        members.insert(c.Index);
    }
    return members;
}

/* Find synchronised local particles within `radius` of any center of a region, with no
 * chain-eligibility filter. Used to gather "limiting particles" — non-chain
 * perturbers whose individual timesteps constrain the chain's macro step.
 * Pass `exclude` to skip members that are already in the chain itself. */
__attribute__((unused)) static std::set<int> find_local_particles_in_radius(const std::vector<ketju_mpi_particle> &centers,
                                                    double radius,
                                                    const std::set<int> &exclude)
{
    std::set<int> hits;
    for(int i = 0; i < NumPart; i++) {
        if(exclude.count(i)) continue;
        if(!is_synchronized(i)) continue;
        if((P[i].Type != 4 && P[i].Type != 5) || is_chain_center(i)) continue; /* only stars outside chains limit a chain; gas and other chains do not */
        for(size_t c = 0; c < centers.size(); c++) {
            double dist = particle_distance(P[i].Pos.data_ptr(), const_cast<double*>(centers[c].Pos));
            if(dist < radius) { hits.insert(i); break; }
        }
    }
    return hits;
}
static void forward_com_to_affected_root(KetjuRegion &reg)
{
    if(reg.affected_tasks.root_sim == reg.compute_tasks.root_sim) return;
    double buf[6];
    if(ThisTask == reg.compute_tasks.root_sim) {
        for(int j = 0; j < 3; j++) { buf[j] = reg.com_pos[j]; buf[3 + j] = reg.com_vel[j]; }
        MPI_Send(buf, 6, MPI_DOUBLE, reg.affected_tasks.root_sim, 996, MPI_COMM_WORLD);
    }
    if(ThisTask == reg.affected_tasks.root_sim) {
        MPI_Recv(buf, 6, MPI_DOUBLE, reg.compute_tasks.root_sim, 996, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int j = 0; j < 3; j++) { reg.com_pos[j] = buf[j]; reg.com_vel[j] = buf[3 + j]; }
    }
}

#ifdef KJ_STAR_COLLISIONS
/* Contact radius of a live Type-4 star, physical code length; 0 for remnants and dead or unsampled stars.
 * Main-sequence R/Rsun = M^0.8 (M < 1 Msun), M^0.57 above (Demircan & Kahraman 1991). */
static double ketju_star_contact_radius(int i)
{
    if(P[i].Type != 4 || P[i].Mass <= 0) return 0;
    double R_cm;
#if defined(GALSF_SINGLE_STARS_FROM_IMF) && defined(GALSF_SINGLE_STARS_EVOLUTION_TABLES) && defined(GALSF_FB_MECHANICAL)
    if(P[i].StarIdx > 0 && SingleStars.event_bits[single_star_slot(P[i].StarIdx)] != 0) {return 0;} /* a remnant */
#endif
    double M = P[i].Mass * UNIT_MASS_IN_SOLAR;
    R_cm = 6.957e10 * ((M < 1) ? pow(M, 0.8) : pow(M, 0.57));
    return R_cm / UNIT_LENGTH_IN_CGS; /* physical */
}
#endif

/* counts[t] = members of this region on task t (gathered for all regions at once in ketju_find_regions) */
static void setup_integrator(KetjuRegion &reg, const int *counts)
{
    int n_local = reg.local_member_indices.size();

    reg.total_particle_count = 0;
    for(int t = 0; t < NTask; t++) reg.total_particle_count += counts[t];
    if(reg.total_particle_count < 2) return; /* need at least 2 particles */

    /* create affected_tasks communicator: only tasks with particles */
    {
        std::vector<int> affected_indices;
        for(int t = 0; t < NTask; t++) { if(counts[t] > 0) affected_indices.push_back(t); }
        if(affected_indices.empty()) return;
        reg.affected_tasks.init(affected_indices, 0);
        reg.affected_sim_indices = affected_indices;
#ifdef TEST_KETJU_REGION_LOG
        if(affected_indices.size() > 1 && ThisTask == 0) {static int said = 0; if(!said) {printf("KETJU_REGION_ACROSS_RANKS\n"); said = 1;}}
#endif
        reg.particle_counts_on_affected.resize(affected_indices.size());
        for(size_t i = 0; i < affected_indices.size(); i++)
            reg.particle_counts_on_affected[i] = counts[affected_indices[i]];
    }

    /* create compute_tasks communicator: contiguous range from cost-based allocator.
     * Scatter uses MPI_Scatterv on affected_tasks, so compute/affected groups can
     * safely differ — data is forwarded via Send/Recv between roots if needed. */
    {
        std::vector<int> compute_indices;
        for(int t = reg.compute_info.first_task_index; t <= reg.compute_info.final_task_index; t++)
            compute_indices.push_back(t);
        reg.compute_tasks.init(compute_indices, 1);
        reg.compute_tasks.set_common_root(reg.affected_tasks);
    }

    /* fill local particle data */
    std::vector<ketju_mpi_particle> local_parts(n_local);
    int k = 0;
    for(int idx : reg.local_member_indices) {
        ketju_mpi_particle &mp = local_parts[k];
        mp.ID = P[idx].ID; mp.Child = P[idx].ID_child_number;
        mp.Type = P[idx].Type;
        mp.Task = ThisTask;
        mp.Index = idx;
        mp.Mass = P[idx].Mass;
        for(int j = 0; j < 3; j++) {
            mp.Pos[j] = P[idx].Pos[j];
            mp.Vel[j] = P[idx].Vel[j];
        }
        tree_pair_softening(idx, mp.Soft, mp.Zeta);
#ifdef KJ_STAR_COLLISIONS
        mp.CollRadius = ketju_star_contact_radius(idx);
#ifdef GALSF
        mp.BirthTime = (P[idx].Type == 4) ? P[idx].StellarAge : 0;
#else
        mp.BirthTime = 0;
#endif
        mp.FormDens = 0;
#endif
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
#endif
        k++;
    }

    /* gather all particles on affected root */
    if(reg.affected_tasks.is_member()) {
        int n_aff = reg.affected_tasks.size;
        std::vector<int> aff_counts(n_aff), byte_counts(n_aff), byte_displs(n_aff);
        MPI_Allgather(&n_local, 1, MPI_INT, aff_counts.data(), 1, MPI_INT, reg.affected_tasks.comm);
        int total_bytes = 0;
        for(int t = 0; t < n_aff; t++) {
            byte_counts[t] = aff_counts[t] * sizeof(ketju_mpi_particle);
            byte_displs[t] = total_bytes;
            total_bytes += byte_counts[t];
        }
        if(reg.affected_tasks.is_root()) {
            reg.all_particles.resize(reg.total_particle_count);
        }
        MPI_Gatherv(local_parts.data(), n_local * sizeof(ketju_mpi_particle), MPI_BYTE,
                    reg.all_particles.data(), byte_counts.data(), byte_displs.data(),
                    MPI_BYTE, reg.affected_tasks.root, reg.affected_tasks.comm);
    }

    /* if compute root differs from affected root, forward data */
    if(reg.affected_tasks.root_sim != reg.compute_tasks.root_sim) {
        if(ThisTask == reg.affected_tasks.root_sim) {
            MPI_Send(reg.all_particles.data(), reg.total_particle_count * sizeof(ketju_mpi_particle),
                     MPI_BYTE, reg.compute_tasks.root_sim, 999, MPI_COMM_WORLD);
        }
        if(ThisTask == reg.compute_tasks.root_sim) {
            reg.all_particles.resize(reg.total_particle_count);
            MPI_Recv(reg.all_particles.data(), reg.total_particle_count * sizeof(ketju_mpi_particle),
                     MPI_BYTE, reg.affected_tasks.root_sim, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    /* broadcast particle data to all compute tasks */
    if(reg.compute_tasks.is_member()) {
        if(!reg.compute_tasks.is_root()) reg.all_particles.resize(reg.total_particle_count);
        MPI_Bcast(reg.all_particles.data(), reg.total_particle_count * sizeof(ketju_mpi_particle),
                  MPI_BYTE, reg.compute_tasks.root, reg.compute_tasks.comm);
    }

    /* set up integrator on compute tasks */
    if(reg.compute_tasks.is_member()) {
        /* compute CoM position and velocity */
        double total_mass = 0;
        for(int j = 0; j < 3; j++) { reg.com_pos[j] = 0; reg.com_vel[j] = 0; }
        for(int i = 0; i < reg.total_particle_count; i++) {
            for(int j = 0; j < 3; j++) {
                reg.com_pos[j] += reg.all_particles[i].Mass * reg.all_particles[i].Pos[j];
                reg.com_vel[j] += reg.all_particles[i].Mass * reg.all_particles[i].Vel[j];
            }
            total_mass += reg.all_particles[i].Mass;
        }
        for(int j = 0; j < 3; j++) { reg.com_pos[j] /= total_mass; reg.com_vel[j] /= total_mass; }

        /* allocate integrator — use compute_tasks.comm for parallel integration */
        reg.integrator = (struct ketju_system *)calloc(1, sizeof(struct ketju_system));
        int n_other = reg.total_particle_count - reg.num_pn_particles;
        kj_mstar_create_system(reg.integrator, reg.num_pn_particles, n_other, reg.compute_tasks.comm);

        /* set units */
        reg.integrator->constants->G = All.G * KJ_LEN_TO_NAT * KJ_VEL_TO_NAT * KJ_VEL_TO_NAT / KJ_MASS_TO_NAT;

        /* set options */
        reg.integrator->options->gbs_relative_tolerance = All.KetjuIntegrationTolerance;
        if(All.KetjuMaxStepCount > 0) reg.integrator->options->max_step_count = All.KetjuMaxStepCount;
        if(All.KetjuUseStarStarSoftening) {
            double h = All.ForceSoftening[4] * All.cf_atime;
            reg.integrator->options->star_star_softening = h * KJ_LEN_TO_NAT;
        }

        /* fill particle data (relative to CoM, converted to physical coordinates) */
        reg.extra_data.resize(reg.total_particle_count);
        struct ketju_system_physical_state *ps = reg.integrator->physical_state;
        ps->time = 0;
        double a = All.cf_atime; /* scale factor: 1 for non-cosmo */
        for(int i = 0; i < reg.total_particle_count; i++) {
            ps->mass[i] = reg.all_particles[i].Mass;
            for(int j = 0; j < 3; j++) {
                /* comoving -> physical: pos_phys = (pos_comov - com_comov) * a */
                ps->pos[i][j] = (reg.all_particles[i].Pos[j] - reg.com_pos[j]) * a;
                /* GIZMO Vel stores canonical momentum p = a^2 * dx_comov/dt_phys.
                 * Peculiar velocity = Vel / a. Physical velocity = v_pec + H*r_phys.
                 * MSTAR needs physical velocity (includes Hubble flow).
                 * In non-cosmo: a=1, Vel is already physical, H=0. */
                double v_pec_rel = (reg.all_particles[i].Vel[j] - reg.com_vel[j]) / a;
                ps->vel[i][j] = v_pec_rel;
                if(All.ComovingIntegrationOn)
                    ps->vel[i][j] += hubble_function(a) * ps->pos[i][j];
            }
            reg.extra_data[i].ID = reg.all_particles[i].ID; reg.extra_data[i].Child = reg.all_particles[i].Child;
            reg.extra_data[i].Task = reg.all_particles[i].Task;
            reg.extra_data[i].Index = reg.all_particles[i].Index;
            reg.extra_data[i].Type = reg.all_particles[i].Type;
#ifdef KJ_STAR_COLLISIONS
            reg.extra_data[i].BirthTime = reg.all_particles[i].BirthTime; reg.extra_data[i].FormDens = reg.all_particles[i].FormDens;
#endif
        }
#ifdef TEST_KETJU_REGION_LOG
        {
            double e[2] = {0, 0}, M = 0, c[3] = {0, 0, 0}, q[3] = {0, 0, 0}; int n = reg.total_particle_count;
            for(int i = 0; i < n; i++) {M += ps->mass[i]; for(int j = 0; j < 3; j++) {c[j] += ps->mass[i] * ps->pos[i][j]; q[j] += ps->mass[i] * ps->vel[i][j];}}
            for(int j = 0; j < 3; j++) {e[0] = fmax(e[0], fabs(c[j]) / M); e[1] = fmax(e[1], fabs(q[j]) / M);}
            if(All.Ti_Current == 0) {printf("KETJU_REGION n %d com_error %.3e p_error %.3e\n", n, e[0] / All.KetjuRegionRadius, e[1]);}
        }
#endif
        reg.integrator->particle_extra_data = reg.extra_data.data();
        reg.integrator->particle_extra_data_elem_size = sizeof(ketju_extra_data);
#ifdef KJ_STAR_COLLISIONS
        {
            int any = 0; reg.coll_radius.assign(reg.total_particle_count, 0.0);
            for(int i = reg.num_pn_particles; i < reg.total_particle_count; i++) {
                reg.coll_radius[i] = reg.all_particles[i].CollRadius * KJ_LEN_TO_NAT; if(reg.coll_radius[i] > 0) any = 1;
            }
            if(any) reg.integrator->collision_radius = reg.coll_radius.data();
        }
#endif

    }

    /* broadcast CoM data to all tasks that need it (affected group for scatter) */
    forward_com_to_affected_root(reg);
    if(reg.affected_tasks.is_member()) {
        MPI_Bcast(reg.com_pos, 3, MPI_DOUBLE, reg.affected_tasks.root, reg.affected_tasks.comm);
        MPI_Bcast(reg.com_vel, 3, MPI_DOUBLE, reg.affected_tasks.root, reg.affected_tasks.comm);
    }
}

/* ============================================================
 *  PHASE 4: Negative half-step kick and integration
 * ============================================================ */

/* Subtract softened pairwise gravity between chain members from their velocities.
 * This undoes the tree force contribution so MSTAR can replace it with exact forces. */
static void do_negative_halfstep_kick(KetjuRegion &reg, double kick_factor, int at_start, std::vector<double> *dv_out = NULL)
{
    if(!reg.compute_tasks.is_member() || !reg.integrator) return;

    int n = reg.integrator->num_particles;
    struct ketju_system_physical_state *ps = reg.integrator->physical_state;

    /* softening of a pair of chain members (all Type 5) as the tree computes it: the larger of the two
     * particles' softenings, i.e. ForceSoftening[5] (comoving -> physical) */
    double h = All.ForceSoftening[5] * All.cf_atime;

    /* distribute N² loop across compute tasks */
    int loop_start = loop_scheduling_block_edge(n, reg.compute_tasks.size, reg.compute_tasks.rank);
    int loop_end   = loop_scheduling_block_edge(n, reg.compute_tasks.size, reg.compute_tasks.rank + 1);

    std::vector<double> dv(3 * n, 0.0);
    /* absolute comoving positions as the tree uses them: at the start, the stored host positions; at the end, the exact
       targets scatter_results computes and the drift is snapped to */
    std::vector<double> abs_pos(3 * n);
    double a_kick = All.cf_atime;
    if(at_start) {
        for(int i = 0; i < n; i++) for(int k = 0; k < 3; k++) abs_pos[3*i + k] = reg.all_particles[i].Pos[k];
    } else {
        if(All.ComovingIntegrationOn) a_kick = All.cf_atime * exp(reg.ti_step * All.Timebase_interval);
        double dt_drift = get_drift_factor(All.Ti_Current, All.Ti_Current + reg.ti_step, 0, 0);
        for(int i = 0; i < n; i++) for(int k = 0; k < 3; k++) {
            double sp_pos = reg.com_pos[k] + ps->pos[i][k] / a_kick;
            abs_pos[3*i + k] = (dt_drift > 0) ? sp_pos + reg.com_vel[k] * dt_drift : sp_pos;
        }
    }

    for(int i = loop_start; i < loop_end; i++) {
        for(int j = i + 1; j < n; j++) {
            double dr[3], r2 = 0;
            for(int k = 0; k < 3; k++) {
                dr[k] = ps->pos[i][k] - ps->pos[j][k];
                r2 += dr[k] * dr[k];
            }
            double r = sqrt(r2);
            if(r == 0) continue;

            { /* the tree's own pair kernel and separation (absolute comoving positions, as the tree differences them), with the
                 softening/zeta of the gravity call whose kick is cancelled */
                const ketju_mpi_particle &pi = reg.all_particles[i], &pj = reg.all_particles[j];
                double drc[3]; for(int k = 0; k < 3; k++) {drc[k] = abs_pos[3*i + k] - abs_pos[3*j + k];}
#if defined(BOX_PERIODIC) && !defined(GRAVITY_NOT_PERIODIC)
                NEAREST_XYZ(drc[0], drc[1], drc[2], 1);
#endif
                const double *si = pi.Soft, *zi = pi.Zeta, *sj = pj.Soft, *zj = pj.Zeta;
                double r2c = drc[0]*drc[0] + drc[1]*drc[1] + drc[2]*drc[2], rc = sqrt(r2c), f_ij, f_ji, d0, d1, d2;
                if(rc <= 0) continue;
                tree_pair_gravity_factors(rc, r2c, ps->mass[j], si[0], sj[1], pi.Type, pj.Type, zi[0], zj[1], ps->mass[i], ags_gravity_kernel_shared_BITFLAG(pi.Type), f_ij, d0, d1, d2);
                tree_pair_gravity_factors(rc, r2c, ps->mass[i], sj[0], si[1], pj.Type, pi.Type, zj[0], zi[1], ps->mass[j], ags_gravity_kernel_shared_BITFLAG(pj.Type), f_ji, d0, d1, d2);
                double fac_c = kick_factor * All.G / (a_kick * a_kick); /* comoving pair force -> physical acceleration */
                for(int k = 0; k < 3; k++) {
                    dv[3*i + k] += fac_c * f_ij * drc[k];
                    dv[3*j + k] -= fac_c * f_ji * drc[k];
                }
                (void)h;
            }
        }
    }

    /* collect results across compute tasks and apply */
    MPI_Allreduce(MPI_IN_PLACE, dv.data(), 3 * n, MPI_DOUBLE, MPI_SUM, reg.compute_tasks.comm);
    if(dv_out) *dv_out = dv;

    for(int i = 0; i < n; i++) {
        for(int k = 0; k < 3; k++) {
            ps->vel[i][k] += dv[3*i + k];
        }
    }
}

/* ============================================================
 *  Expand tight binaries: if the tightest binary has a period
 *  much shorter than the timestep and is an outlier, double its
 *  semi-major axis to prevent integrator stalling.
 *  Based on the public GADGET4-KETJU algorithm (Mannerkoski+ 2023).
 * ============================================================ */
static void expand_tight_binaries(KetjuRegion &reg, double dt_physical)
{
    if(!reg.compute_tasks.is_root() || !reg.integrator) return;
    if(All.KetjuExpandBinariesFactor <= 0) return; /* disabled */

    int n = reg.integrator->num_particles;
    struct ketju_system_physical_state *ps = reg.integrator->physical_state;

    /* find the tightest bound binary */
    int best_i = -1, best_j = -1;
    double min_period = 1e30;
    int n_tight = 0; /* count of binaries with period < 2*min_period */

    for(int i = 0; i < n; i++) {
        for(int j = i + 1; j < n; j++) {
            double dr[3], dv[3], r2 = 0, v2 = 0;
            for(int k = 0; k < 3; k++) {
                dr[k] = ps->pos[i][k] - ps->pos[j][k];
                dv[k] = ps->vel[i][k] - ps->vel[j][k];
                r2 += dr[k] * dr[k]; v2 += dv[k] * dv[k];
            }
            double GM = All.G * (ps->mass[i] + ps->mass[j]);
            double E = 0.5 * v2 - GM / sqrt(r2);
            if(E >= 0) continue; /* unbound */
            double a = -GM / (2.0 * E);
            double period = 2.0 * M_PI * a * sqrt(a / GM);
            if(period < min_period) {
                min_period = period; best_i = i; best_j = j;
            }
        }
    }

    if(best_i < 0) return; /* no bound binaries */
    if(min_period > All.KetjuExpandBinariesFactor * dt_physical) return; /* not tight enough */

    /* count binaries with period < 2*min_period */
    for(int i = 0; i < n; i++) {
        for(int j = i + 1; j < n; j++) {
            double dr[3], dv[3], r2 = 0, v2 = 0;
            for(int k = 0; k < 3; k++) {
                dr[k] = ps->pos[i][k] - ps->pos[j][k];
                dv[k] = ps->vel[i][k] - ps->vel[j][k];
                r2 += dr[k] * dr[k]; v2 += dv[k] * dv[k];
            }
            double GM = All.G * (ps->mass[i] + ps->mass[j]);
            double E = 0.5 * v2 - GM / sqrt(r2);
            if(E >= 0) continue;
            double a = -GM / (2.0 * E);
            double period = 2.0 * M_PI * a * sqrt(a / GM);
            if(period < 2.0 * min_period) n_tight++;
        }
    }

    /* only expand if it's an outlier (< max(N/100, 5) tight binaries) */
    if(n_tight > DMAX(n / 100, 5)) return;

    /* double the semi-major axis: scale relative velocity by 1/sqrt(2)
     * so a_new = 2*a_old (via virial theorem: E_new = E_old/2) */
    double scale = 1.0 / sqrt(2.0);
    double com_vel[3];
    double mtot = ps->mass[best_i] + ps->mass[best_j];
    for(int k = 0; k < 3; k++) {
        com_vel[k] = (ps->mass[best_i] * ps->vel[best_i][k] + ps->mass[best_j] * ps->vel[best_j][k]) / mtot;
    }
    for(int k = 0; k < 3; k++) {
        ps->vel[best_i][k] = com_vel[k] + scale * (ps->vel[best_i][k] - com_vel[k]);
        ps->vel[best_j][k] = com_vel[k] + scale * (ps->vel[best_j][k] - com_vel[k]);
    }

    printf("KETJU: expanded tight binary (period=%g -> %g, dt=%g) in region with %d particles\n",
           min_period, min_period * 2.0 * sqrt(2.0), dt_physical, n);
}

/* Snapshot state of members part-way through their step: for each output time inside a region's step (and at its start,
 * where the first output after the IC falls: written after this integration), a copy of the region's MSTAR system is
 * integrated to it (collisions off) and its state is kept by particle ID until the output, so the region's own
 * integration is unchanged. The state is that of the host scheme at that time: the CoM drifting with its velocity after
 * the first half-kick, the internal motion from MSTAR. With an output list only the next output time is covered. */
struct ketju_output_state {MyIDType ID, Child; integertime ti; double pos[3], vel[3];};
static std::vector<ketju_output_state> KetjuOutputStates; /* compute roots, until gathered at the output */
static void record_midstep_output_state(KetjuRegion &reg)
{
    if(!reg.compute_tasks.is_member() || !reg.integrator) return;
    integertime t0 = All.Ti_Current, t1 = All.Ti_Current + reg.ti_step;
    std::vector<integertime> outs;
    for(integertime to = All.Ti_nextoutput; to >= t0 && to < t1; to = find_next_outputtime(to + 1)) {outs.push_back(to); if(All.OutputListOn) break;}
    if(outs.empty()) return;
    struct ketju_system *R = reg.integrator, C;
    int n = R->num_particles;
    memset(&C, 0, sizeof(C));
    kj_mstar_create_system(&C, R->num_pn_particles, n - R->num_pn_particles, reg.compute_tasks.comm);
    *C.options = *R->options; *C.constants = *R->constants;
    for(int i = 0; i < n; i++) {
        C.physical_state->mass[i] = R->physical_state->mass[i];
        for(int k = 0; k < 3; k++) {C.physical_state->pos[i][k] = R->physical_state->pos[i][k]; C.physical_state->vel[i][k] = R->physical_state->vel[i][k];}
    }
    for(int i = 0; i < R->num_pn_particles; i++) for(int k = 0; k < 3; k++) C.physical_state->spin[i][k] = R->physical_state->spin[i][k];
    C.physical_state->time = 0;
    double t_done = 0; /* physical time C has been integrated */
    for(integertime to : outs) {
        double t_out, a_out = All.cf_atime, hub = 0;
        if(All.ComovingIntegrationOn) {
            t_out = get_gravkick_factor(t0, to, -1, 0) * All.cf_atime * exp(0.5 * (to - t0) * All.Timebase_interval);
            a_out = All.cf_atime * exp((to - t0) * All.Timebase_interval); hub = hubble_function(a_out);
        } else {t_out = (to - t0) * All.Timebase_interval;}
        if(t_out > t_done) {
            kj_state_change_units(C.physical_state, n, 1);
            kj_mstar_run_integrator(&C, (t_out - t_done) * KJ_TIME_TO_NAT);
            kj_state_change_units(C.physical_state, n, 0);
            t_done = t_out;
        }
        if(!reg.compute_tasks.is_root()) continue;
        double drift = (to > t0) ? get_drift_factor(t0, to, 0, 0) : 0;
        for(int i = 0; i < n; i++) {
            ketju_output_state o; o.ID = reg.extra_data[i].ID; o.Child = reg.extra_data[i].Child; o.ti = to;
            for(int k = 0; k < 3; k++) {
                o.pos[k] = reg.com_pos[k] + reg.com_vel[k] * drift + C.physical_state->pos[i][k] / a_out;
                o.vel[k] = reg.com_vel[k] + (C.physical_state->vel[i][k] - hub * C.physical_state->pos[i][k]) * a_out;
            }
            KetjuOutputStates.push_back(o);
        }
    }
    kj_mstar_free_system(&C);
}


/* Exact two-body drift (KetjuKeplerPairs): universal variables with Stumpff functions, Laguerre-Conway iteration (Danby
 * 1992 sec. 6.9; Conway 1986). r, v are the relative separation and velocity, advanced in place over dt; returns 0 if the
 * iteration does not converge (the caller then uses MSTAR). */
static void stumpff(double z, double &C, double &S)
{
    if(fabs(z) < 1e-3) {C = 0.5 - z / 24 + z * z / 720 - z * z * z / 40320; S = 1.0 / 6 - z / 120 + z * z / 5040 - z * z * z / 362880; return;}
    if(z > 0) {double q = sqrt(z); C = (1 - cos(q)) / z; S = (q - sin(q)) / (z * q); return;}
    double q = sqrt(-z); C = (cosh(q) - 1) / (-z); S = (sinh(q) - q) / (-z * q);
}
static int kepler_drift(double r[3], double v[3], double mu, double dt)
{
    double r0 = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]), v2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2], rv = r[0]*v[0] + r[1]*v[1] + r[2]*v[2];
    double alpha = 2 / r0 - v2 / mu, smu = sqrt(mu), sig = rv / smu;
    if(alpha > 0) {double P = 2 * M_PI / (sqrt(alpha) * alpha * smu); dt = fmod(dt, P);} /* whole periods return the state */
    double x = (alpha > 0) ? smu * alpha * dt : smu * dt / r0; /* initial universal anomaly */
    double C, S, F = 0, Fp = 1;
    int it;
    for(it = 0; it < 100; it++) {
        double z = alpha * x * x; stumpff(z, C, S);
        F = sig * x * x * C + (1 - alpha * r0) * x * x * x * S + r0 * x - smu * dt;
        Fp = sig * x * (1 - z * S) + (1 - alpha * r0) * x * x * C + r0;
        double Fpp = sig * (1 - z * C) + (1 - alpha * r0) * x * (1 - z * S), n = 5;
        double disc = fabs((n - 1) * (n - 1) * Fp * Fp - n * (n - 1) * F * Fpp);
        double dx = n * F / (Fp + (Fp >= 0 ? 1 : -1) * sqrt(disc));
        x -= dx;
        if(fabs(dx) <= 1e-15 * fabs(x) + 1e-300) break;
    }
    if(it == 100) return 0;
    double z = alpha * x * x; stumpff(z, C, S);
    double f = 1 - x * x / r0 * C, g = dt - x * x * x / smu * S, rn[3];
    for(int k = 0; k < 3; k++) rn[k] = f * r[k] + g * v[k];
    double rr = sqrt(rn[0]*rn[0] + rn[1]*rn[1] + rn[2]*rn[2]);
    double fd = smu / (rr * r0) * (z * S - 1) * x, gd = 1 - x * x / rr * C;
    for(int k = 0; k < 3; k++) {v[k] = fd * r[k] + gd * v[k]; r[k] = rn[k];}
    return 1;
}
/* Two members without PN advanced as an exact Kepler orbit about their CoM; not when a contact collision is possible */
static int kepler_pair_step(KetjuRegion &reg, double dt)
{
    struct ketju_system *R = reg.integrator; struct ketju_system_physical_state *ps = R->physical_state;
    if(!All.KetjuKeplerPairs || R->num_particles != 2 || R->num_pn_particles != 0 || R->options->enable_external_tide) return 0;
    double m0 = ps->mass[0], m1 = ps->mass[1], M = m0 + m1, mu = All.G * M, r[3], v[3], rc[3], vc[3];
    for(int k = 0; k < 3; k++) {r[k] = ps->pos[1][k] - ps->pos[0][k]; v[k] = ps->vel[1][k] - ps->vel[0][k]; rc[k] = (m0 * ps->pos[0][k] + m1 * ps->pos[1][k]) / M; vc[k] = (m0 * ps->vel[0][k] + m1 * ps->vel[1][k]) / M;}
    if(R->collision_radius) {
        double r2 = 0, v2 = 0, rv = 0; for(int k = 0; k < 3; k++) {r2 += r[k]*r[k]; v2 += v[k]*v[k]; rv += r[k]*v[k];}
        double h2 = r2 * v2 - rv * rv, e2 = 1 + (v2 - 2 * mu / sqrt(r2)) * h2 / (mu * mu), rp = h2 / mu / (1 + sqrt(DMAX(e2, 0.0)));
        if(rp < (R->collision_radius[0] + R->collision_radius[1]) / KJ_LEN_TO_NAT) return 0;
    }
    if(!kepler_drift(r, v, mu, dt)) return 0;
    for(int k = 0; k < 3; k++) {
        rc[k] += vc[k] * dt;
        ps->pos[0][k] = rc[k] - m1 / M * r[k]; ps->pos[1][k] = rc[k] + m0 / M * r[k];
        ps->vel[0][k] = vc[k] - m1 / M * v[k]; ps->vel[1][k] = vc[k] + m0 / M * v[k];
    }
    ps->time += dt; R->perf->successful_steps = 1; R->perf->failed_steps = 0; R->perf->relative_energy_error = 0;
    return 1;
}

/* Run the full integration step for one region.
 * Uses proper cosmological kick factors for comoving integration
 * (based on public GADGET4-KETJU, Mannerkoski+ 2023). */
static void integrate_region(KetjuRegion &reg, double dt_physical)
{
    if(dt_physical <= 0) return;

    /* Compute kick factors: in comoving integration, the half-step kick
     * factor is the gravkick integral (dt/a²), not just 0.5*dt_physical.
     * The scale factor correction accounts for the tree force being computed
     * in comoving coordinates while KETJU works in physical coordinates. */
    double first_halfkick, second_halfkick;
    if(All.ComovingIntegrationOn) {
        integertime t0 = All.Ti_Current;
        integertime t1 = t0 + reg.ti_step;
        integertime tmid = t0 + reg.ti_step / 2;
        double a0 = All.TimeBegin * exp(t0 * All.Timebase_interval);
        double a1 = All.TimeBegin * exp(t1 * All.Timebase_interval);
        first_halfkick  = get_gravkick_factor(t0, tmid, -1, 0) * a0;
        second_halfkick = get_gravkick_factor(tmid, t1, -1, 0) * a1;
    } else {
        first_halfkick  = 0.5 * dt_physical;
        second_halfkick = 0.5 * dt_physical;
    }

    /* subtract first half of tree force */
    std::vector<double> pair_dv;
    do_negative_halfstep_kick(reg, first_halfkick, 1, &pair_dv);

    /* expand tight binaries before integration to prevent stalling.
     * Runs on compute root only — broadcast updated velocities to all compute tasks. */
    expand_tight_binaries(reg, dt_physical);
    if(reg.compute_tasks.is_member() && reg.integrator && reg.compute_tasks.size > 1) {
        int n = reg.integrator->num_particles;
        MPI_Bcast(reg.integrator->physical_state->vel, 3 * n, MPI_DOUBLE,
                  reg.compute_tasks.root, reg.compute_tasks.comm);
    }

    record_midstep_output_state(reg);
    /* run MSTAR integrator (all compute tasks participate via compute_tasks.comm) */
    if(reg.compute_tasks.is_member() && reg.integrator) {
        int kepler = kepler_pair_step(reg, dt_physical);
        if(!kepler) {
        kj_state_change_units(reg.integrator->physical_state, reg.integrator->num_particles, 1);
        kj_mstar_run_integrator(reg.integrator, dt_physical * KJ_TIME_TO_NAT);
        kj_state_change_units(reg.integrator->physical_state, reg.integrator->num_particles, 0);
        }
        if(reg.integrator->num_particles < reg.total_particle_count) {
            /* MSTAR removed merged particles by shifting the arrays: the tail slots hold stale copies in its natural units.
             * Zero them, and put all_particles in the order of extra_data, which MSTAR permuted the same way. */
            struct ketju_system_physical_state *ps = reg.integrator->physical_state;
            for(int i = reg.integrator->num_particles; i < reg.total_particle_count; i++) {
                ps->mass[i] = 0; for(int j = 0; j < 3; j++) {ps->pos[i][j] = 0; ps->vel[i][j] = 0;}
            }
            std::unordered_map<KetjuUID, ketju_mpi_particle, KetjuUIDHash> by_id;
            for(const ketju_mpi_particle &q : reg.all_particles) by_id[{q.ID, q.Child}] = q;
            for(int i = 0; i < reg.total_particle_count; i++) reg.all_particles[i] = by_id[{reg.extra_data[i].ID, reg.extra_data[i].Child}];
        }

        int n_steps = reg.integrator->perf->successful_steps + reg.integrator->perf->failed_steps;
        double dE = reg.integrator->perf->relative_energy_error;
        if(All.KetjuMaxStepCount > 0 && n_steps >= All.KetjuMaxStepCount) {
            printf("KETJU WARNING: integration hit max step count (%d) — results may be inaccurate!\n",
                   All.KetjuMaxStepCount);
        }
        if(fabs(dE) > 1e-4) {
            printf("KETJU WARNING: large energy error dE/E=%g in region with %d particles\n",
                   dE, reg.total_particle_count);
        }
    }

    if(reg.compute_tasks.is_member() && reg.integrator) {
        int n = reg.integrator->num_particles; reg.true_vel.assign(3 * reg.total_particle_count, 0.0);
        for(int i = 0; i < n; i++) for(int j = 0; j < 3; j++) reg.true_vel[3*i+j] = reg.integrator->physical_state->vel[i][j];
    }
    /* subtract second half of tree force */
    do_negative_halfstep_kick(reg, second_halfkick, 0);
}

/* ============================================================
 *  PHASE 5a: Merger handling
 *
 *  After MSTAR integration, merged particles have mass=0.
 *  Detect these, transfer metals to the survivor (nearest
 *  surviving particle), and mark merged particles for removal.
 *  Controlled by KETJU_MERGE_STARS (Type 4+4) and
 *  KETJU_MERGE_BH (Type 5+5, Type 5+4).
 * ============================================================ */

/* Merger action descriptor (ketju_star_mergers.h) -- determined on affected root, broadcast to all affected tasks */

#if defined(KETJU_MERGE_STARS)
typedef star_merger_action ketju_merger_action;
/* Survivor of each merged particle (final_mass 0): the particle MSTAR merged it into (merged_into, followed to a particle
 * still alive), else the nearest survivor (sink coalescences found after the integration). Each survivor receives the
 * metal mass of all its victims at once, so several mergers into one survivor conserve metals. */
static void handle_mergers(KetjuRegion &reg,
                           const std::vector<double> &final_abs_pos,
                           const std::vector<double> &final_mass,
                           const std::vector<double> &original_mass,
                           const std::vector<KetjuUID> &merged_into)
{
    if(!reg.affected_tasks.is_member()) return;

    /* ---- Phase 1: affected root determines merger pairs ---- */
    int n_mergers = 0;
    std::vector<ketju_merger_action> actions;

    if(reg.affected_tasks.is_root()) {
        int n = reg.total_particle_count;
        std::unordered_map<KetjuUID, int, KetjuUIDHash> slot;
        for(int i = 0; i < n; i++) slot[{reg.extra_data[i].ID, reg.extra_data[i].Child}] = i;
        for(int i = 0; i < n; i++) {
            if(final_mass[i] > 0) continue;

            int type_merged = reg.extra_data[i].Type;
            int survivor = -1;
            if(merged_into[i].id) {
                int j = i;
                for(int hop = 0; hop < n && merged_into[j].id && final_mass[j] <= 0; hop++) {
                    auto it = slot.find(merged_into[j]); if(it == slot.end()) break; j = it->second;
                }
                if(final_mass[j] > 0) survivor = j;
            } else {
                double min_dist2 = 1e60;
                for(int j = 0; j < n; j++) {
                    if(j == i || final_mass[j] <= 0) continue;
                    double d2 = 0;
                    for(int k = 0; k < 3; k++) {
                        double dx = final_abs_pos[3*i + k] - final_abs_pos[3*j + k];
                        d2 += dx * dx;
                    }
                    if(d2 < min_dist2) { min_dist2 = d2; survivor = j; }
                }
            }
            if(survivor < 0) continue;

            int type_survivor = reg.extra_data[survivor].Type;

            int merge_allowed = 0;
#ifdef KETJU_MERGE_STARS
            if(type_merged == 4 && type_survivor == 4) merge_allowed = 1;
#endif
            if(!merge_allowed) continue;

            double M_merged = original_mass[i];
            double M_survivor_old = original_mass[survivor];
            double M_survivor_new = final_mass[survivor];
            if(M_survivor_new <= 0 || M_merged <= 0) continue;

            ketju_merger_action act;
            act.id_merged      = reg.extra_data[i].ID;
            act.id_survivor    = reg.extra_data[survivor].ID;
            act.type_merged    = type_merged;
            act.type_survivor  = type_survivor;
            act.task_merged    = reg.extra_data[i].Task;
            act.task_survivor  = reg.extra_data[survivor].Task;
            act.idx_merged     = reg.extra_data[i].Index;
            act.idx_survivor   = reg.extra_data[survivor].Index;
            act.M_merged       = M_merged;
            act.M_survivor_old = M_survivor_old;
            act.M_survivor_new = M_survivor_new;
            actions.push_back(act);

            printf("KETJU MERGER: ID %llu (Type %d, M=%g) merged into ID %llu (Type %d, M=%g->%g)\n",
                   (unsigned long long)act.id_merged, type_merged, M_merged,
                   (unsigned long long)act.id_survivor, type_survivor,
                   M_survivor_old, M_survivor_new);
        }
        n_mergers = actions.size();
    }

    /* ---- Phase 2: broadcast merger list to all affected tasks ---- */
    MPI_Bcast(&n_mergers, 1, MPI_INT, reg.affected_tasks.root, reg.affected_tasks.comm);
    if(n_mergers == 0) return;

    if(!reg.affected_tasks.is_root()) actions.resize(n_mergers);
    MPI_Bcast(actions.data(), n_mergers * sizeof(ketju_merger_action), MPI_BYTE,
              reg.affected_tasks.root, reg.affected_tasks.comm);

    /* ---- Phases 3-4: victims' data to the survivors, merger products (ketju_star_mergers.cc) ---- */
    star_mergers_apply(actions, "KETJU");
}
#endif /* KETJU_MERGE_STARS || KETJU_MERGE_BH */

/* ============================================================
 *  PHASE 5b: Scatter results and apply velocity trick
 *
 *  After MSTAR integration, positions are in CoM-relative frame.
 *  Convert back to GIZMO frame and set up velocity trick so that
 *  GIZMO's standard drift lands particles at the correct positions.
 *
 *  Velocity trick:
 *    Vel = com_vel + (r_new - r_old) / dt_drift
 *  so that after drift: Pos += Vel * dt_drift
 *    = com_vel*dt_drift + r_new - r_old
 *    = (com_pos + com_vel*dt_drift) + r_new - (com_pos + r_old)
 *    = com_pos_drifted + r_new - Pos_current
 *  => new Pos = com_pos_drifted + r_new  (correct!)
 *
 *  True velocity (KetjuFinalVel) = com_vel + v_integrator
 *  is swapped in after the drift by ketju_set_final_velocities().
 * ============================================================ */

/* Per-particle data packed for MPI_Scatterv (one struct per particle) */
struct ketju_scatter_particle {
    MyIDType ID;
    MyIDType Child;
    int Task;
    int Index;
    int Type;
    double Mass;
    double Pos[3];       /* absolute comoving position */
    double Vel[3];       /* CoM-relative peculiar velocity */
#if defined(KETJU_MERGE_STARS)
    double OriginalMass;
    KetjuUID MergedInto; /* ID and child number of the particle MSTAR merged this one into (ID 0: none) */
#endif
    double Sphere[4];    /* chain bounding sphere at end of step (comoving center, radius) */
    double TrueVel[3];   /* CoM-relative velocity before the second negative half-kick */
};

/* comparison for sorting scatter particles by Task (for MPI_Scatterv) */
static int scatter_particle_task_cmp(const void *a, const void *b)
{
    const ketju_scatter_particle *pa = (const ketju_scatter_particle *)a;
    const ketju_scatter_particle *pb = (const ketju_scatter_particle *)b;
    return (pa->Task > pb->Task) - (pa->Task < pb->Task);
}

static void scatter_results(KetjuRegion &reg)
{
    int n = reg.total_particle_count;
    bool is_affected     = reg.affected_tasks.is_member();
    bool is_compute_root = (ThisTask == reg.compute_tasks.root_sim);
    bool needs_forward   = (reg.compute_tasks.root_sim != reg.affected_tasks.root_sim);

    /* compute_root must participate to pack and Send the result to affected_root,
     * even if it is not itself a member of the affected_tasks group. */
    if(!is_affected && !(is_compute_root && needs_forward)) return;

    if(is_affected) {
        MPI_Bcast(&n, 1, MPI_INT, reg.affected_tasks.root, reg.affected_tasks.comm);
    }
    if(n < 2) return;

    int n_aff = reg.affected_tasks.size;
    std::vector<ketju_scatter_particle> scatter_buf;

    /* ---- Phase 1: compute root packs results into scatter_buf ---- */
    /* compute end-of-step scale factor for position/velocity back-conversion */
    double a_end = All.cf_atime; /* 1 for non-cosmo */
    double hubble_end = 0; /* H(a_end); 0 for non-cosmo */
    if(All.ComovingIntegrationOn) {
        a_end = All.cf_atime * exp(reg.ti_step * All.Timebase_interval);
        hubble_end = hubble_function(a_end);
    }

    if(reg.compute_tasks.is_root() && reg.integrator) {
        struct ketju_system_physical_state *ps = reg.integrator->physical_state;
        scatter_buf.resize(n);

        for(int i = 0; i < n; i++) {
            ketju_scatter_particle &sp = scatter_buf[i];
            sp.ID    = reg.extra_data[i].ID;
            sp.Child = reg.extra_data[i].Child;
            sp.Task  = reg.extra_data[i].Task;
            sp.Index = reg.extra_data[i].Index;
            sp.Type  = reg.extra_data[i].Type;
            sp.Mass  = ps->mass[i];
#if defined(KETJU_MERGE_STARS)
            sp.OriginalMass = reg.all_particles[i].Mass;
            sp.MergedInto = {0, 0};
#endif
            for(int j = 0; j < 3; j++) {
                /* physical -> comoving: use a_end since positions are at end of step */
                sp.Pos[j] = reg.com_pos[j] + ps->pos[i][j] / a_end;
                /* physical velocity -> canonical momentum: remove Hubble flow, multiply by a_end.
                 * sp.Vel stores CoM-relative canonical momentum (com_vel added later in velocity trick).
                 * In non-cosmo: a_end=1, hubble_end=0, so sp.Vel = ps->vel (unchanged). */
                sp.Vel[j] = (ps->vel[i][j] - hubble_end * ps->pos[i][j]) * a_end;
                sp.TrueVel[j] = (reg.true_vel[3*i+j] - hubble_end * ps->pos[i][j]) * a_end;
            }
        }
        { /* chain sphere from the end positions the members drift to; the end-of-step tree opens nodes overlapping it */
            double c[3] = {0,0,0}, rmax2 = 0; int n_live = reg.integrator->num_particles; /* merged ones sit in the tail */
            for(int i = 0; i < n_live; i++) for(int j = 0; j < 3; j++) c[j] += ps->pos[i][j] / (a_end * n_live);
            for(int i = 0; i < n_live; i++) {double d2 = 0; for(int j = 0; j < 3; j++) {double d = ps->pos[i][j] / a_end - c[j]; d2 += d * d;} if(d2 > rmax2) rmax2 = d2;}
            for(int i = 0; i < n; i++) {for(int j = 0; j < 3; j++) {scatter_buf[i].Sphere[j] = reg.com_pos[j] + c[j];} scatter_buf[i].Sphere[3] = 1.01 * sqrt(rmax2) + MIN_REAL_NUMBER;}
        }

        /* log merger info from integrator */
        if(reg.integrator->num_mergers > 0) {
            for(int m = 0; m < reg.integrator->num_mergers; m++) {
                struct ketju_bh_merger_info *info = &reg.integrator->merger_infos[m];
#if defined(KETJU_MERGE_STARS)
                scatter_buf[info->extra_index2].MergedInto = {reg.extra_data[info->extra_index1].ID, reg.extra_data[info->extra_index1].Child};
#endif
                if(info->is_collision) {
                    double mu = KJ_MASS_TO_NAT, eu = mu * KJ_VEL_TO_NAT * KJ_VEL_TO_NAT; /* MSTAR natural -> host units */
                    KetjuCollisionEnergy += info->dE_collision / eu;
#ifdef KJ_STAR_COLLISIONS
                    const ketju_extra_data &x1 = reg.extra_data[info->extra_index1], &x2 = reg.extra_data[info->extra_index2];
                    double age1 = evaluate_time_since_t_initial_in_Gyr(x1.BirthTime) * 1e9, age2 = evaluate_time_since_t_initial_in_Gyr(x2.BirthTime) * 1e9;
                    double rho_y = (age1 < age2) ? x1.FormDens : x2.FormDens; /* younger star's formation density */
                    double tff_yr = (rho_y > 0) ? sqrt(3. * M_PI / (32. * All.G * rho_y)) * UNIT_TIME_IN_YR : 0;
                    printf("KETJU STAR COLLISION: ID %llu (M=%g) + ID %llu (M=%g) at t=%g, dE=%g sibling=%d dt_birth=%g yr (%g t_ff) age_younger=%g yr\n",
                           (unsigned long long)x1.ID, info->m1 / mu, (unsigned long long)x2.ID, info->m2 / mu, All.Time, info->dE_collision / eu,
                           (int)(x1.ID == x2.ID), fabs(age1 - age2), (tff_yr > 0) ? fabs(age1 - age2) / tff_yr : -1., DMIN(age1, age2));
#endif
                    continue;
                }
                printf("KETJU BH MERGER [region]: m1=%g m2=%g -> m_rem=%g, chi1=%g chi2=%g -> chi_rem=%g, v_kick=%g\n",
                       info->m1, info->m2, info->m_remnant,
                       info->chi1, info->chi2, info->chi_remnant, info->v_kick);
            }
        }

    }
    /* ---- Phase 2: forward from compute root to affected root if they differ ---- */
    if(reg.compute_tasks.root_sim != reg.affected_tasks.root_sim) {
        if(ThisTask == reg.compute_tasks.root_sim) {
            MPI_Send(scatter_buf.data(), n * sizeof(ketju_scatter_particle),
                     MPI_BYTE, reg.affected_tasks.root_sim, 998, MPI_COMM_WORLD);
        }
        if(ThisTask == reg.affected_tasks.root_sim) {
            scatter_buf.resize(n);
            MPI_Recv(scatter_buf.data(), n * sizeof(ketju_scatter_particle),
                     MPI_BYTE, reg.compute_tasks.root_sim, 998, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    /* compute_root that is not in affected_tasks has finished its job (Send done) */
    if(!is_affected) return;

    /* ---- Phase 3: affected root sorts by Task, Scattervs to affected tasks ---- */

    /* build map: world task -> affected rank */
    std::unordered_map<int, int> task_to_aff_rank;
    for(size_t i = 0; i < reg.affected_sim_indices.size(); i++)
        task_to_aff_rank[reg.affected_sim_indices[i]] = (int)i;

    std::vector<int> sendcounts(n_aff, 0);
    std::vector<int> byte_sendcounts(n_aff, 0), byte_displs(n_aff, 0);

    if(reg.affected_tasks.is_root()) {
        /* sort particles by Task */
        qsort(scatter_buf.data(), n, sizeof(ketju_scatter_particle), scatter_particle_task_cmp);

        /* compute sendcounts per affected rank */
        for(int i = 0; i < n; i++) {
            auto it = task_to_aff_rank.find(scatter_buf[i].Task);
            if(it != task_to_aff_rank.end())
                sendcounts[it->second]++;
        }
        for(int t = 0; t < n_aff; t++) byte_sendcounts[t] = sendcounts[t] * sizeof(ketju_scatter_particle);
        for(int t = 1; t < n_aff; t++) byte_displs[t] = byte_displs[t-1] + byte_sendcounts[t-1];
    }

    /* scatter sendcounts so each task knows how many particles it receives */
    int my_count = 0;
    MPI_Scatter(sendcounts.data(), 1, MPI_INT, &my_count, 1, MPI_INT,
                reg.affected_tasks.root, reg.affected_tasks.comm);

    /* scatter particle data */
    std::vector<ketju_scatter_particle> my_particles(my_count);
    MPI_Scatterv(scatter_buf.data(), byte_sendcounts.data(), byte_displs.data(), MPI_BYTE,
                 my_particles.data(), my_count * sizeof(ketju_scatter_particle), MPI_BYTE,
                 reg.affected_tasks.root, reg.affected_tasks.comm);

    /* ---- Phase 4: each task applies velocity trick to its particles ---- */
    /* also collect full particle data on affected root for stellar collision check + mergers */
    std::vector<double> final_abs_pos, final_rel_vel, final_mass;
#if defined(KETJU_MERGE_STARS)
    std::vector<double> original_mass;
    std::vector<KetjuUID> merged_into;
#endif

    /* affected root keeps the sorted scatter_buf for collision/merger checks */
    if(reg.affected_tasks.is_root()) {
        final_abs_pos.resize(3 * n);
        final_rel_vel.resize(3 * n);
        final_mass.resize(n);
#if defined(KETJU_MERGE_STARS)
        original_mass.resize(n); merged_into.resize(n);
#endif
        /* also rebuild extra_data and all_particles on root from scatter_buf for collision/merger code */
        reg.extra_data.resize(n);
        reg.all_particles.resize(n);
        for(int i = 0; i < n; i++) {
            final_abs_pos[3*i]   = scatter_buf[i].Pos[0];
            final_abs_pos[3*i+1] = scatter_buf[i].Pos[1];
            final_abs_pos[3*i+2] = scatter_buf[i].Pos[2];
            final_rel_vel[3*i]   = scatter_buf[i].Vel[0];
            final_rel_vel[3*i+1] = scatter_buf[i].Vel[1];
            final_rel_vel[3*i+2] = scatter_buf[i].Vel[2];
            final_mass[i] = scatter_buf[i].Mass;
#if defined(KETJU_MERGE_STARS)
            original_mass[i] = scatter_buf[i].OriginalMass; merged_into[i] = scatter_buf[i].MergedInto;
#endif
            reg.extra_data[i].ID    = scatter_buf[i].ID;
            reg.extra_data[i].Child = scatter_buf[i].Child;
            reg.extra_data[i].Task  = scatter_buf[i].Task;
            reg.extra_data[i].Index = scatter_buf[i].Index;
            reg.extra_data[i].Type  = scatter_buf[i].Type;
            reg.all_particles[i].Mass = scatter_buf[i].Mass;
        }
    }

    for(int i = 0; i < my_count; i++) {
        ketju_scatter_particle &sp = my_particles[i];
        int idx = sp.Index;

        /* sanity check */
        if(P[idx].ID != sp.ID || P[idx].ID_child_number != sp.Child) {
            printf("KETJU ERROR: particle index mismatch! ID %llu vs %llu on task %d\n",
                   (unsigned long long)P[idx].ID, (unsigned long long)sp.ID, ThisTask);
            endrun(667);
        }

        /* update mass */
        if(P[idx].Mass != sp.Mass) {TreeMomentsStaleFlag = 1;} /* node masses and centres follow a merger at the next gravity call */
        P[idx].Mass = sp.Mass;
        struct ketju_member_state &km = ketju_member_open(idx);
        for(int k = 0; k < 4; k++) {km.chain_sphere[k] = sp.Sphere[k];}
        P[idx].KetjuOpenRadius = (float)(((reg.affected_sim_indices.size() > 1) ? 2.0 : -2.0) * sp.Sphere[3]); /* any partner lies within the sphere's diameter of this member; < 0: on this rank */
        for(int k = 0; k < 3; k++) {km.true_vel[k] = sp.TrueVel[k] + reg.com_vel[k];}
        for(int k = 0; k < 3; k++) {km.chain_vel[k] = reg.com_vel[k];}

        /* skip velocity trick for merged particles (mass=0, will be removed) */
        if(sp.Mass <= 0) {
            P[idx].KetjuIntegrated = 1;
            continue;
        }

        /* compute per-particle drift factor: what drift_particle() will use */
        double dt_drift = get_drift_factor(P[idx].Ti_current, P[idx].Ti_current + reg.ti_step, idx, 0);

        if(dt_drift > 0) {
            for(int j = 0; j < 3; j++) {
                /* desired final position = final_abs_pos + CoM displacement during drift */
                double desired_pos = sp.Pos[j] + reg.com_vel[j] * dt_drift;
                km.target_pos[j] = desired_pos;
                double delta_pos = desired_pos - P[idx].Pos[j];
#ifdef BOX_PERIODIC
                double box = (j == 0) ? boxSize_X : (j == 1) ? boxSize_Y : boxSize_Z;
                if(box > 0) { while(delta_pos > 0.5 * box) delta_pos -= box; while(delta_pos < -0.5 * box) delta_pos += box; }
#endif
                P[idx].dp[j] += P[idx].Mass * (delta_pos / dt_drift - P[idx].Vel[j]); /* the nodes' bulk velocity follows this write as it does a kick */
                P[idx].Vel[j] = delta_pos / dt_drift;
                km.final_vel[j] = sp.Vel[j] + reg.com_vel[j];
            }
        } else {
            /* no drift — set position and velocity directly */
            for(int j = 0; j < 3; j++) {
                P[idx].Pos[j] = sp.Pos[j];
                km.target_pos[j] = sp.Pos[j];
                P[idx].dp[j] += P[idx].Mass * (sp.Vel[j] + reg.com_vel[j] - P[idx].Vel[j]);
                P[idx].Vel[j] = sp.Vel[j] + reg.com_vel[j];
                km.final_vel[j] = P[idx].Vel[j];
            }
        }
        gizmo_motion_bound_raise(&idx, 1); /* velocity set outside the kick: nodes must cover this drift */

        P[idx].KetjuIntegrated = 1;
    }


    /* handle mergers (integrator mergers + stellar collisions) — merger decisions
     * are made on affected root then broadcast to all affected tasks for execution */
#if defined(KETJU_MERGE_STARS)
    handle_mergers(reg, final_abs_pos, final_mass, original_mass, merged_into);
#endif
}

/* ============================================================
 *  PUBLIC INTERFACE FUNCTIONS
 * ============================================================ */

/* ============================================================
 *  Helper: move a particle between timebin linked lists
 * ============================================================ */
static void move_particle_timebin(int i, int old_bin, int new_bin)
{
    if(old_bin == new_bin) return;

    TimeBinCount[old_bin]--;
    if(PrevInTimeBin[i] >= 0)
        NextInTimeBin[PrevInTimeBin[i]] = NextInTimeBin[i];
    else
        FirstInTimeBin[old_bin] = NextInTimeBin[i];
    if(NextInTimeBin[i] >= 0)
        PrevInTimeBin[NextInTimeBin[i]] = PrevInTimeBin[i];
    else
        LastInTimeBin[old_bin] = PrevInTimeBin[i];

    if(FirstInTimeBin[new_bin] >= 0)
        PrevInTimeBin[FirstInTimeBin[new_bin]] = i;
    NextInTimeBin[i] = FirstInTimeBin[new_bin];
    PrevInTimeBin[i] = -1;
    FirstInTimeBin[new_bin] = i;
    if(LastInTimeBin[new_bin] < 0)
        LastInTimeBin[new_bin] = i;
    TimeBinCount[new_bin]++;

    P[i].TimeBin = new_bin;
}



/* PHASE 6: put every chain region on one timebin, changing only particles synchronised this step.
 *
 * The velocity trick needs every member of a region to start and end its step with the region: the
 * first half-kick, the MSTAR interval and the drift all have to cover [Ti_Current, Ti_Current +
 * ti_step]. A particle part-way through its step cannot be retimed -- it has had no first half-kick
 * for this interval, and its step marker would be left pointing at the old one. So, as in
 * GADGET4-KETJU (RegionManager::find_timestep_regions / set_region_timebins), only synchronised
 * particles are considered at all: regions are built around synchronised centres from synchronised
 * members (gather_chain_centers, find_local_members), and a star still in mid-step inside R joins
 * at its own next step boundary, integrated by the host until then.
 *
 * What keeps members co-binned is the "limited" set: every chain-eligible synchronised particle
 * within 2R of a centre is put on the region's bin, so anything that drifts inside R is already
 * on it. Centres whose 2R spheres overlap share one "timestep region", so the limited sets are
 * disjoint and each chain region (merge radius 2R) lies inside exactly one of them. The shared
 * bin is the MIN over
 *   (a) the limited particles' own steps from find_timesteps, and
 *   (b) the steps of the "limiting" particles: synchronised particles of any type within
 *       KetjuTimestepLimitingFactor * R, so the region resolves perturbers approaching it.
 * Every such step was assigned at this sync point and obeys the grow-only-when-synchronised rule,
 * so their minimum does too; moving a synchronised particle to it is always legal. A particle
 * that is not synchronised has a longer step than any bin chosen now, so leaving it out of (b)
 * cannot change the minimum. All bins are found before any particle is moved, so the result does
 * not depend on region order. */
/* host writes to P (bins, chain flags): the device copy is re-uploaded before its next use */
struct KjArenaInvalidate {~KjArenaInvalidate() {gpu_particles_arena_invalidate();}};

void ketju_limit_timesteps(void)
{
    KjArenaInvalidate arena_guard;
    KjClock clk(CPU_KETJULIMIT);
    CachedChainCentersValid = 0;

    if(All.KetjuRegionRadius <= 0) return;
    if(All.KetjuMinStarMass <= 0) return;

    /* gather synchronised chain centres (cached for ketju_find_regions) */
    std::vector<ketju_mpi_particle> centers = gather_chain_centers();
    CachedChainCenters = centers;
    CachedChainCentersValid = 1;
    if(centers.empty()) return;

    const double limited_radius = 2.0 * All.KetjuRegionRadius;
    const double limiting_radius = All.KetjuTimestepLimitingFactor * All.KetjuRegionRadius; /* GADGET-4 KETJU default 100 */
    std::vector<std::vector<ketju_mpi_particle>> tsr_centers = build_orbit_regions(centers); (void)limited_radius;
    tsr_centers.erase(std::remove_if(tsr_centers.begin(), tsr_centers.end(), [](const std::vector<ketju_mpi_particle> &g) {return g.size() < 2;}), tsr_centers.end()); /* a lone star forms no chain */
    CachedCenterGroups = tsr_centers;
    if(tsr_centers.empty()) {return;}
    const int n_tsr = (int)tsr_centers.size();

    std::vector<std::set<int>> limited(n_tsr);
    std::vector<long long> min_ti(2 * n_tsr, (long long)TIMEBASE); /* [2r] over limited+limiting, [2r+1] limited only */
    for(int r = 0; r < n_tsr; r++) {
        limited[r] = local_members_from_group(tsr_centers[r]);
        for(int idx : limited[r]) {
            long long ti = GET_INTEGERTIME_FROM_TIMEBIN(P[idx].TimeBin);
            if(ti > 0 && ti < min_ti[2*r]) min_ti[2*r] = ti;
            if(ti > 0 && ti < min_ti[2*r+1]) min_ti[2*r+1] = ti;
        }
#if !defined(BOX_PERIODIC)
    }
    { /* limiting particles: one pass over the local particles, each tested only against centres in neighbouring cells of
       * size limiting_radius (was: a full particle scan per timestep region) */
        std::unordered_map<long long, std::vector<std::pair<int, int>>> grid; /* cell -> (group, centre in group) */
        auto key = [](long long x, long long y, long long z) {return (x & 0x1fffff) | ((y & 0x1fffff) << 21) | ((z & 0x1fffff) << 42);};
        for(int r = 0; r < n_tsr; r++)
            for(int c = 0; c < (int)tsr_centers[r].size(); c++) {
                const double *x = tsr_centers[r][c].Pos;
                grid[key((long long)floor(x[0] / limiting_radius), (long long)floor(x[1] / limiting_radius), (long long)floor(x[2] / limiting_radius))].push_back({r, c});
            }
        std::vector<int> last_group(n_tsr, -1); /* each particle counts once per group */
        for(int i : ActiveParticleList) {
            if(!is_synchronized(i)) continue;
            if((P[i].Type != 4 && P[i].Type != 5) || is_chain_center(i)) continue; /* only stars outside chains limit a chain; gas and other chains do not */
            long long ti = GET_INTEGERTIME_FROM_TIMEBIN(P[i].TimeBin);
            long long cx = (long long)floor(P[i].Pos[0] / limiting_radius), cy = (long long)floor(P[i].Pos[1] / limiting_radius), cz = (long long)floor(P[i].Pos[2] / limiting_radius);
            for(int dx = -1; dx <= 1; dx++) for(int dy = -1; dy <= 1; dy++) for(int dz = -1; dz <= 1; dz++) {
                auto it = grid.find(key(cx + dx, cy + dy, cz + dz));
                if(it == grid.end()) continue;
                for(const std::pair<int, int> &gc : it->second) {
                    int r = gc.first;
                    if(last_group[r] == i || limited[r].count(i)) continue;
                    if(particle_distance(P[i].Pos.data_ptr(), const_cast<double*>(tsr_centers[r][gc.second].Pos)) >= limiting_radius) continue;
                    last_group[r] = i;
                    if(ti > 0 && ti < min_ti[2*r]) min_ti[2*r] = ti;
                }
            }
        }
    }
#else
        std::set<int> limiting = find_local_particles_in_radius(tsr_centers[r], limiting_radius, limited[r]);
        for(int idx : limiting) {
            long long ti = GET_INTEGERTIME_FROM_TIMEBIN(P[idx].TimeBin);
            if(ti > 0 && ti < min_ti[2*r]) min_ti[2*r] = ti;
        }
    }
#endif
    MPI_Allreduce(MPI_IN_PLACE, min_ti.data(), 2 * n_tsr, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);

    long long counts[4] = {0, 0, 0, 0}; /* limited particles moved (down, up), timestep regions set by limiting particles, unsynchronised target bins */
    for(int r = 0; r < n_tsr; r++) {
        if(min_ti[2*r] <= 0 || min_ti[2*r] >= TIMEBASE) continue;
        int bin = get_timestep_bin((integertime)min_ti[2*r]);
        if(bin <= 0) continue;
        if(!TimeBinActive[bin]) {counts[3]++; while(bin > 0 && !TimeBinActive[bin]) {bin--;}} /* cannot happen for a minimum of legal steps */
        if(min_ti[2*r] < min_ti[2*r+1]) counts[2]++;
        integertime ti_bin = GET_INTEGERTIME_FROM_TIMEBIN(bin);
        for(int idx : limited[r]) {
            if(P[idx].TimeBin == bin) continue;
            counts[(bin < P[idx].TimeBin) ? 0 : 1]++;
            move_particle_timebin(idx, P[idx].TimeBin, bin);
            P[idx].dt_step = ti_bin; /* find_timesteps set it from the bin just replaced; integertime_step() reads it */
        }
    }

    LimitTally[0] += counts[0]; LimitTally[1] += counts[1];
    if(ThisTask == 0) {LimitTally[2] += n_tsr; LimitTally[3] += counts[2];}
    if(counts[3]) {PRINT_WARNING("KETJU: %lld timestep-region bins were not synchronised and were lowered", counts[3]);}
}

void ketju_find_regions(void)
{
#ifdef TEST_KETJU_MSTAR_PAIR
    if(All.Ti_Current == 0) {ketju_test_mstar_pair();}
#endif
    KjArenaInvalidate arena_guard;
    KjClock clk(CPU_KETJUFIND);
    /* Clear the chain flags of particles whose step has ended, and only theirs: a member of a region
     * on a longer bin is still inside the step MSTAR integrated, and its flags keep it out of Hermite
     * and mark it for the velocity swap at the end of that step. KetjuIntegrated is cleared here, not
     * in finish_step, so it also survives to the guard checks of the step that follows. */
    for(int i : ActiveParticleList) {
        if(!is_synchronized(i)) continue;
        if(P[i].KetjuIntegrated) {ketju_member_close(i);}
        P[i].KetjuIntegrated = 0; P[i].KetjuChainID = 0; P[i].KetjuOpenRadius = 0;
    }
    ActiveRegions.clear();
    KetjuNumMultiRegions = 0;
    if(KetjuCommCache.size() > 1024) {ketju_comm_cache_flush();}

    /* validate parameters */
    if(All.KetjuRegionRadius <= 0) return;
    if(All.KetjuMinStarMass <= 0) return;

    /* reuse chain centers from ketju_limit_timesteps if available, else gather fresh */
    std::vector<ketju_mpi_particle> centers;
    const int reuse_groups = CachedChainCentersValid; /* same centres as the limiter, so the same orbit groups */
    if(CachedChainCentersValid) {
        centers = std::move(CachedChainCenters);
        CachedChainCentersValid = 0;
    } else {
        centers = gather_chain_centers();
    }
    KjFoundCenters = centers;
    if(centers.empty()) return;

    /* merge overlapping regions */
    double merge_radius = 2.0 * All.KetjuRegionRadius;
    std::vector<std::vector<ketju_mpi_particle>> region_centers = reuse_groups ? std::move(CachedCenterGroups) : build_orbit_regions(centers); (void)merge_radius;
    region_centers.erase(std::remove_if(region_centers.begin(), region_centers.end(), [](const std::vector<ketju_mpi_particle> &g) {return g.size() < 2;}), region_centers.end());
    if(region_centers.empty()) {return;}

    /* build regions (first pass: find members) */
    int n_regions = region_centers.size();

    for(int r = 0; r < n_regions; r++) {
        KetjuRegion reg;
        reg.centers = region_centers[r];
        reg.local_member_indices = local_members_from_group(reg.centers);

        ActiveRegions.push_back(std::move(reg));
    }

    /* member totals of all regions in one reduction, then per-task counts of the multi-member ones in one gather */
    std::vector<int> region_counts(n_regions);
    for(int r = 0; r < n_regions; r++) region_counts[r] = (int)region_centers[r].size(); /* the group is known on every task, and all its centres are members */
    std::vector<int> multi; /* regions with >= 2 members; the rest stay as placeholders with nothing to integrate */
    for(int r = 0; r < n_regions; r++) {
        ActiveRegions[r].total_particle_count = region_counts[r];
        if(region_counts[r] < 2) continue;
        multi.push_back(r);
    }
    const int n_multi = (int)multi.size();
    KetjuNumMultiRegions = n_multi;
    std::vector<int> task_counts((size_t)NTask * n_multi); /* [t * n_multi + k] */
    for(int k = 0; k < n_multi; k++) {for(const ketju_mpi_particle &c : region_centers[multi[k]]) {task_counts[(size_t)c.Task * n_multi + k]++;}}

    /* allocate compute tasks across regions based on cost estimates */
    allocate_compute_tasks_for_regions();
    /* A region given one compute task is integrated where its members live (its affected root, the lowest task holding
     * members): no forwarding to and from a remote task, which would make the affected root wait idle on the remote MSTAR run. */
    for(int k = 0; k < n_multi; k++) {
        region_compute_info &ci = ActiveRegions[multi[k]].compute_info;
        if(ci.first_task_index != ci.final_task_index) continue;
        for(int t = 0; t < NTask; t++) {if(task_counts[(size_t)t * n_multi + k] > 0) {ci.first_task_index = ci.final_task_index = t; break;}}
    }

    /* second pass: set up integrators */
    clk.lap(CPU_KETJUFIND);
    {
        std::vector<int> counts(NTask);
        for(int k = 0; k < n_multi; k++) {
            for(int t = 0; t < NTask; t++) counts[t] = task_counts[(size_t)t * n_multi + k];
            setup_integrator(ActiveRegions[multi[k]], counts.data());
        }
    }
    clk.lap(CPU_KETJUEXCH);

    /* KetjuChainID: hash of the group's smallest (ID, child); the group is known on every task */
    for(int k = 0; k < n_multi; k++) {
        KetjuRegion &reg = ActiveRegions[multi[k]];
        MyIDType chain_id = (MyIDType)-1, min_child = (MyIDType)-1;
        for(const ketju_mpi_particle &c : reg.centers) {if(c.ID < chain_id || (c.ID == chain_id && c.Child < min_child)) {chain_id = c.ID; min_child = c.Child;}}
        chain_id ^= min_child * 0x9E3779B97F4A7C15ULL; if(chain_id == 0) {chain_id = 1;} /* siblings share P.ID: the child number keeps chains distinct */
        for(int idx : reg.local_member_indices) {P[idx].KetjuChainID = chain_id;}
    }

    /* Region-set summary: only print when the set actually changed (otherwise
     * we emit ~1+N_region lines per step, which dominates IO). The per-region
     * listing is only printed when the topology changed; silence it otherwise. */
    if(ThisTask == 0 && !ActiveRegions.empty()) {
        static int last_n_regions = -1, last_n_centers = -1;
        int n_centers = (int)centers.size();
        if(n_regions != last_n_regions || n_centers != last_n_centers) {
            printf("KETJU: Found %d region(s) from %d center(s) at t=%g\n", n_regions, n_centers, All.Time);
            last_n_regions = n_regions;
            last_n_centers = n_centers;
        }
    }
}

/* Cumulative MSTAR statistics: counted per step, reduced and printed every 1024 region steps and at the end of the run
   (integer sums and maxima: deferring the reduction is exact). */
static long long StatCalls = 0, StatLoc[8] = {0, 0, 0, 0, 0, 0, 0, 0}; static double StatMax[2] = {0, 0};
static void ketju_collect_stats(void)
{
    StatCalls++;
    for(int k = 0; k < 4; k++) {StatLoc[4 + k] += LimitTally[k]; LimitTally[k] = 0;} /* regions integrated, accepted, failed, |dE/E| > 1e-4, then LimitTally */
    for(size_t r = 0; r < ActiveRegions.size(); r++) {
        KetjuRegion &reg = ActiveRegions[r];
        if(!reg.compute_tasks.is_root() || !reg.integrator) continue;
        long long n = (long long)reg.integrator->perf->successful_steps + reg.integrator->perf->failed_steps;
        if(n <= 0) continue;
        double dE = fabs(reg.integrator->perf->relative_energy_error);
        StatLoc[0]++; StatLoc[1] += reg.integrator->perf->successful_steps; StatLoc[2] += reg.integrator->perf->failed_steps;
        if(dE > 1e-4) StatLoc[3]++;
        if((double)n > StatMax[0]) StatMax[0] = (double)n;
        if(dE > StatMax[1]) StatMax[1] = dE;
    }
}
#ifdef TEST_KETJU_WALK_OPENINGS
long long ketju_walk_openings = 0;
#endif
void ketju_report_stats(int final)
{
#ifdef TEST_KETJU_WALK_OPENINGS
    if(final) {long long g = 0; MPI_Reduce(&ketju_walk_openings, &g, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD); if(ThisTask == 0) {printf("KETJU_WALK_OPENINGS %lld\n", g);}}
#endif
    if(final) {ActiveRegions.clear();} /* integrators go before Kokkos does */
    if(StatCalls == 0) {return;} /* rank-uniform: no region ever */
    if(!final && (StatCalls % 1024) != 0) {return;}
    static long long cum[8] = {0, 0, 0, 0, 0, 0, 0, 0}; static double cum_max[2] = {0, 0};
    long long glob[8]; double glob_max[2];
    MPI_Reduce(StatLoc, glob, 8, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(StatMax, glob_max, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    for(int k = 0; k < 8; k++) {StatLoc[k] = 0;}
    StatMax[0] = StatMax[1] = 0;
    if(ThisTask != 0) {return;}
    for(int k = 0; k < 8; k++) {cum[k] += glob[k];}
    cum_max[0] = DMAX(cum_max[0], glob_max[0]); cum_max[1] = DMAX(cum_max[1], glob_max[1]);
    printf("KETJU STATS: t=%g steps=%lld region-integrations=%lld MSTAR accepted=%lld failed=%lld "
           "max-per-region=%lld |dE/E|>1e-4: %lld max|dE/E|=%g limited-moves down=%lld up=%lld "
           "timestep-regions=%lld set-by-limiting=%lld\n", All.Time, StatCalls, cum[0], cum[1], cum[2], (long long)cum_max[0], cum[3], cum_max[1], cum[4], cum[5], cum[6], cum[7]);
}

/* chains alive this step: a reused import keeps the orbital phase of the last build. The tree is rebuilt at a sync point while a chain
   has members on several ranks, or an active star of another rank within KETJU_NEIGHBOUR_RANK_FACTOR chain sizes */
#define KETJU_NEIGHBOUR_RANK_FACTOR 10.
struct KjLiveChain {double c[3], size; long long end; int spans;};
static std::vector<KjLiveChain> KjLiveChains;
static integertime KjRebuildCheckTi = -1;

static void ketju_note_live_chains(void)
{
    for(const KetjuRegion &reg : ActiveRegions) {
        if(reg.total_particle_count < 2 || reg.local_member_indices.empty()) {continue;}
        KjLiveChain l = {{0, 0, 0}, 0, 0, (reg.affected_sim_indices.size() > 1) ? 1 : 0};
        double M = 0; for(const ketju_mpi_particle &c : reg.centers) {M += c.Mass; for(int k = 0; k < 3; k++) {l.c[k] += c.Mass * c.Pos[k];}}
        for(int k = 0; k < 3; k++) {l.c[k] /= M;}
        for(const ketju_mpi_particle &c : reg.centers) {double d2 = 0; for(int k = 0; k < 3; k++) {double d = c.Pos[k] - l.c[k]; d2 += d * d;} l.size = fmax(l.size, sqrt(d2));}
        for(int idx : reg.local_member_indices) {l.end = fmax(l.end, (long long)(P[idx].Ti_begstep + P[idx].dt_step));}
        KjLiveChains.push_back(l);
    }
}
/* collective, while chains are alive: called on steps with centres gathered, every task */
static void ketju_tree_rebuild_rule(void)
{
    int need = 0; size_t w = 0;
    for(const KjLiveChain &l : KjLiveChains) {
        if(l.end <= (long long)All.Ti_Current) {continue;}
        KjLiveChains[w++] = l;
        if(l.spans) {need = 1; continue;}
        const double rn = KETJU_NEIGHBOUR_RANK_FACTOR * l.size;
        for(const ketju_mpi_particle &c : KjFoundCenters) {
            if(c.Task == ThisTask) {continue;}
            double dx = c.Pos[0] - l.c[0], dy = c.Pos[1] - l.c[1], dz = c.Pos[2] - l.c[2];
            gravity_box_nearest_image(dx, dy, dz, -1);
            if(dx*dx + dy*dy + dz*dz < rn*rn) {need = 1; break;}
        }
    }
    KjLiveChains.resize(w);
    int glob = 0; MPI_Allreduce(&need, &glob, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if(glob) {TreeReconstructFlag = 1;}
}

void ketju_run_integration(void)
{
    KjClock clk(CPU_KETJUFIND);
    ketju_member_step_end_init(); /* before this step's regions raise it */
    if(KetjuNumMultiRegions == 0) {
        if(!ketju_no_live_chains()) {ketju_tree_rebuild_rule();}
        return; /* rank-uniform: nothing to integrate */
    }
    /* Outer loop over sequential rounds. Within a round, regions run in
     * parallel on disjoint task ranges; rounds run back-to-back so that
     * expensive regions can use all tasks first before smaller regions. */
    /* Members are synchronised and share one bin (ketju_limit_timesteps), and MSTAR substeps
     * internally, so the region step is that bin's. Check the sharing: a member on another bin
     * would be drifted over a different interval than the one the velocity trick was set for.
     * One reduction for all regions, so no world collective separates the regions' integrations. */
    const int n_act = (int)ActiveRegions.size();
    std::vector<long long> ti_range(2 * n_act); /* {-min, max} over members, per region */
    for(int r = 0; r < n_act; r++) {
        ti_range[2*r] = -(long long)TIMEBASE; ti_range[2*r+1] = 0;
        for(int idx : ActiveRegions[r].local_member_indices) {
            long long ti_i = P[idx].dt_step;
            if(-ti_i > ti_range[2*r]) ti_range[2*r] = -ti_i;
            if(ti_i > ti_range[2*r+1]) ti_range[2*r+1] = ti_i;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, ti_range.data(), 2 * n_act, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    for(int r = 0; r < n_act; r++) {if(ActiveRegions[r].total_particle_count >= 2 && ti_range[2*r+1] > 0) {KetjuMemberStepEnd = std::max(KetjuMemberStepEnd, (long long)All.Ti_Current + ti_range[2*r+1]);}}

    int n_rounds = g_ketju_num_sequential_runs > 0 ? g_ketju_num_sequential_runs : 1;
    for(int seq = 0; seq < n_rounds; seq++) {
    for(size_t r = 0; r < ActiveRegions.size(); r++) {
        KetjuRegion &reg = ActiveRegions[r];
        if(reg.compute_info.compute_sequence_position != seq) continue;
        if(reg.total_particle_count < 2) continue;

        integertime global_max_ti = ti_range[2*r+1];
        if(global_max_ti <= 0 || global_max_ti > TIMEBASE) continue;
        if(-ti_range[2*r] != ti_range[2*r+1] && ThisTask == 0) {
            PRINT_WARNING("KETJU: region of %d particles has members on steps %lld..%lld; integrating over the longest",
                          reg.total_particle_count, -ti_range[2*r], ti_range[2*r+1]);
        }

        reg.ti_step = global_max_ti;

        /* convert to physical time for MSTAR integrator */
        double dt_physical;
        if(All.ComovingIntegrationOn) {
            integertime t0 = All.Ti_Current;
            integertime t1 = t0 + reg.ti_step;
            double a_mid = All.cf_atime * exp(0.5 * reg.ti_step * All.Timebase_interval);
            dt_physical = get_gravkick_factor(t0, t1, -1, 0) * a_mid;
        } else {
            dt_physical = reg.ti_step * All.Timebase_interval;
        }

        if(dt_physical <= 0 || !isfinite(dt_physical)) continue;

        /* integrate and scatter results */
        clk.lap(CPU_KETJUFIND);
        integrate_region(reg, dt_physical);
        clk.lap(CPU_KETJUMSTAR);
        scatter_results(reg);
        gpu_particles_arena_invalidate(); /* host P[] changed */
        clk.lap(CPU_KETJUSCATTER);
    }
    } /* end for(seq) */

    /* update cost estimates for next step's load balancing */
    clk.rest = CPU_KETJUFINISH;
    update_region_costs();

    ketju_collect_stats();
    ketju_note_live_chains();
    ketju_tree_rebuild_rule();
    ketju_report_stats(0);

    if(ThisTask == 0) {
        int total_mergers = 0;
        for(size_t r = 0; r < ActiveRegions.size(); r++) {
            if(ActiveRegions[r].integrator && ActiveRegions[r].integrator->num_mergers > 0)
                total_mergers += ActiveRegions[r].integrator->num_mergers;
        }
        if(total_mergers > 0)
            printf("KETJU: %d merger(s) this step\n", total_mergers);
    }
}

/* Snapshots are written after the drift to the output time but before the step's end: a chain member whose step ends
 * there still carries the velocity-trick chord velocity, so MSTAR's end-of-step velocity is swapped in for the write, and
 * back. A member part-way through its step sits on the chord: its position and velocity are swapped with the state
 * record_midstep_output_state integrated to this time. Called before and after the write (each call swaps). */
void ketju_swap_output_velocities(void)
{
    static std::vector<ketju_output_state> here; /* this output's states, kept between the two calls */
    static int swapped = 0;
    ketju_member_step_end_init();
    if(!swapped && ketju_no_live_chains()) {return;} /* rank-uniform: no chain, no collective */
    if(!swapped) {
        int nl = 0; for(const ketju_output_state &o : KetjuOutputStates) if(o.ti == All.Ti_Current) nl++;
        std::vector<ketju_output_state> mine; for(const ketju_output_state &o : KetjuOutputStates) if(o.ti == All.Ti_Current) mine.push_back(o);
        std::vector<int> cnt(NTask), bc(NTask), bd(NTask); MPI_Allgather(&nl, 1, MPI_INT, cnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
        int nt = 0; for(int t = 0; t < NTask; t++) {bc[t] = cnt[t] * sizeof(ketju_output_state); bd[t] = nt * sizeof(ketju_output_state); nt += cnt[t];}
        here.resize(nt);
        MPI_Allgatherv(mine.data(), nl * sizeof(ketju_output_state), MPI_BYTE, here.data(), bc.data(), bd.data(), MPI_BYTE, MPI_COMM_WORLD);
        std::vector<ketju_output_state> keep; for(const ketju_output_state &o : KetjuOutputStates) if(o.ti > All.Ti_Current) keep.push_back(o);
        KetjuOutputStates.swap(keep);
    }
    std::unordered_map<KetjuUID, int, KetjuUIDHash> by_id; for(size_t q = 0; q < here.size(); q++) by_id[{here[q].ID, here[q].Child}] = (int)q;
    for(int i = 0; i < NumPart; i++) {
        if(!P[i].KetjuIntegrated || P[i].Mass <= 0) continue;
        if(P[i].Ti_begstep + P[i].dt_step == All.Ti_Current) {
            {double *tv = ketju_member(i).true_vel; for(int j = 0; j < 3; j++) {double v = P[i].Vel[j]; P[i].Vel[j] = tv[j]; tv[j] = v;}}
            continue;
        }
        auto it = by_id.find({P[i].ID, P[i].ID_child_number}); if(it == by_id.end()) continue;
        ketju_output_state &o = here[it->second];
        for(int j = 0; j < 3; j++) {double x = P[i].Pos[j], v = P[i].Vel[j]; P[i].Pos[j] = o.pos[j]; P[i].Vel[j] = o.vel[j]; o.pos[j] = x; o.vel[j] = v;}
    }
    swapped = !swapped;
    if(!swapped) here.clear();
}

void ketju_set_final_velocities(void)
{
    KjClock clk(CPU_KETJUFINISH);
    /* called after drift: swap in true physical velocities and correct dp[] momentum.
     * The velocity trick set Vel = delta_pos/dt_drift for the drift.
     * Now we replace it with the true MSTAR velocity and correct dp[].
     * Only for members whose step ends at this sync point (their bin is active and they were just
     * drifted to it): a region on a longer bin is still drifting along its chord. */
    ketju_member_step_end_init();
    if(ketju_no_live_chains()) {return;}
    std::vector<int> raised;
    for(int n = 0; n < TIMEBINS; n++) {if(!TimeBinActive[n]) {continue;}
    for(int i = FirstInTimeBin[n]; i >= 0; i = NextInTimeBin[i]) {
        if(P[i].KetjuIntegrated) {
            { /* land exactly on the MSTAR end position the negative half-kick used (the drift only reaches it to roundoff) */
                const struct ketju_member_state &km = ketju_member(i); double d2 = 0; for(int j = 0; j < 3; j++) {double d = P[i].Pos[j] - km.target_pos[j]; d2 += d * d;}
                if(d2 < 1e-12 * All.KetjuRegionRadius * All.KetjuRegionRadius) {for(int j = 0; j < 3; j++) {P[i].Pos[j] = km.target_pos[j];}}
            }
            for(int j = 0; j < 3; j++) {
                /* dp correction: the difference between true velocity and trick velocity,
                 * converted to momentum. This ensures the kick integrator is consistent. */
                double dv = ketju_member(i).final_vel[j] - P[i].Vel[j];
                P[i].dp[j] += P[i].Mass * dv;
                P[i].Vel[j] = ketju_member(i).final_vel[j];
            }
            raised.push_back(i);
        }
    }
    }
    if(!raised.empty()) {gizmo_motion_bound_raise(raised.data(), (int)raised.size()); gpu_particles_arena_invalidate();}
}

/* Chain-external gravity and tidal timescale for chain members whose step ends at this sync point.
 *
 * The host tree's GravAccel and tidal_tensorps include the chain members' pull on each other, so
 * the standard criteria would step a member at its internal orbital cadence -- the dynamics MSTAR
 * integrates -- rather than at the cadence of the field external to the chain. This subtracts the
 * other members' pair field as the tree computed it, softened with the Type-5 pair softening
 * (ForceSoftening[5]; an unsoftened term diverges inside the kernel, where the tree's does not),
 * leaving COM_GravAccel and COM_dt_tidal = |T_external|_F^(-1/2) for find_timesteps (timestep.cc).
 * Only members within KetjuRegionRadius are subtracted: those stay in the chain for the next step.
 * Runs in finish_step, after compute_grav_accelerations has refreshed GravAccel/tidal_tensorps (which
 * are already multiplied by G there, gravity/gravtree.cc) at the members' end-of-step positions.
 * Every member of a chain shares its bin, so all of them end together and are found here; the
 * chain is identified by KetjuChainID, which persists over the member's step, not by ActiveRegions,
 * which only holds the regions started this step. */

void ketju_compute_com_quantities(void)
{
    struct ChainMemberData {
        MyIDType chain;
        double mass;
        double pos[3], vel[3]; /* vel: peculiar, after the second half-kick */
        int type;
        double soft[2], zeta[2]; /* tree softening/zeta */
    };

    ketju_member_step_end_init();
    if(ketju_no_live_chains()) {return;} /* rank-uniform: no member can end its step here */
    std::vector<ChainMemberData> local_data;
    std::vector<int> local_index;
    for(int i : ActiveParticleList) {
        if(!P[i].KetjuIntegrated || P[i].KetjuChainID == 0 || P[i].Mass <= 0) continue;
        ChainMemberData d;
        d.chain = P[i].KetjuChainID; d.mass = P[i].Mass;
        for(int j = 0; j < 3; j++) {d.pos[j] = P[i].Pos[j]; d.vel[j] = P[i].Vel[j] / All.cf_atime;}
        d.type = P[i].Type; tree_pair_softening(i, d.soft, d.zeta);
        local_data.push_back(d); local_index.push_back(i);
    }
    int n_local = local_data.size();
    std::vector<int> counts(NTask), byte_counts(NTask), byte_displs(NTask);
    MPI_Allgather(&n_local, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int n_total = 0;
    for(int t = 0; t < NTask; t++) {
        byte_counts[t] = counts[t] * sizeof(ChainMemberData);
        byte_displs[t] = n_total * sizeof(ChainMemberData);
        n_total += counts[t];
    }
    if(n_total == 0) return;
    std::vector<ChainMemberData> all_data(n_total);
    MPI_Allgatherv(local_data.data(), n_local * sizeof(ChainMemberData), MPI_BYTE,
                   all_data.data(), byte_counts.data(), byte_displs.data(), MPI_BYTE, MPI_COMM_WORLD);
    std::sort(all_data.begin(), all_data.end(), [](const ChainMemberData &a, const ChainMemberData &b) {return a.chain < b.chain;});

    double h = All.ForceSoftening[5]; /* comoving, like P[].Pos and the tree's pair softening */
    for(int n = 0; n < n_local; n++) {
        int idx = local_index[n];
        auto lo = std::lower_bound(all_data.begin(), all_data.end(), local_data[n],
                                   [](const ChainMemberData &a, const ChainMemberData &b) {return a.chain < b.chain;});
        Vec3<double> chain_grav = {0, 0, 0}, chain_grav_all = {0, 0, 0}; /* members within the region radius; all members */
        SymmetricTensor2<MyFloat> chain_tidal = {0,0,0,0,0,0};
        for(auto it = lo; it != all_data.end() && it->chain == local_data[n].chain; ++it) {
            double dr[3];
            for(int j = 0; j < 3; j++) dr[j] = it->pos[j] - P[idx].Pos[j];
#ifdef BOX_PERIODIC
            NEAREST_XYZ(dr[0], dr[1], dr[2], -1);
#endif
            double r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
            if(r2 <= 0) continue; /* self */
            double r = sqrt(r2), r3 = r * r2, r5 = r3 * r2, fac, fac2;
            { /* the tree's own pair force/tidal factors with the softening it just used (tidal terms carry no zeta, as in the tree) */
                const ChainMemberData &me = local_data[n];
                double fa, fp, ft, ft2;
                tree_pair_gravity_factors(r, r2, it->mass, me.soft[0], it->soft[1], me.type, it->type, me.zeta[0], it->zeta[1], me.mass, ags_gravity_kernel_shared_BITFLAG(me.type), fa, fp, ft, ft2);
                for(int j = 0; j < 3; j++) chain_grav_all[j] += All.G * fa * dr[j];
                if(r2 >= All.KetjuRegionRadius * All.KetjuRegionRadius) continue; /* may leave the chain next step: keep it, as the tree walk does for dt_2body */
                double ftid = ft, ftid2 = ft2;
#ifndef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
                ftid = fa; ftid2 = 3.0 * it->mass / r5;
#endif
                for(int j = 0; j < 3; j++) chain_grav[j] += All.G * fa * dr[j];
                chain_tidal.data[0] += All.G * (ftid2 * dr[0]*dr[0] - ftid);
                chain_tidal.data[1] += All.G * (ftid2 * dr[1]*dr[1] - ftid);
                chain_tidal.data[2] += All.G * (ftid2 * dr[2]*dr[2] - ftid);
                chain_tidal.data[3] += All.G * ftid2 * dr[0]*dr[1];
                chain_tidal.data[4] += All.G * ftid2 * dr[1]*dr[2];
                chain_tidal.data[5] += All.G * ftid2 * dr[0]*dr[2];
                (void)fp; (void)fac; (void)fac2; (void)h;
                continue;
            }
            if(r >= h) {
                fac  = All.G * it->mass / r3;
                fac2 = 3.0 * All.G * it->mass / r5;
            } else { /* inside the kernel: the tree's kernel_gravity() force and tidal factors */
                double h_inv = 1.0 / h, h3_inv = h_inv * h_inv * h_inv, u = r * h_inv;
                fac  = All.G * it->mass * kernel_gravity(u, h_inv, h3_inv, 1);
                fac2 = All.G * it->mass * kernel_gravity(u, h_inv, h3_inv, 2);
            }
            for(int j = 0; j < 3; j++) chain_grav[j] += fac * dr[j];
            chain_tidal.data[0] += fac2 * dr[0]*dr[0] - fac;  /* xx */
            chain_tidal.data[1] += fac2 * dr[1]*dr[1] - fac;  /* yy */
            chain_tidal.data[2] += fac2 * dr[2]*dr[2] - fac;  /* zz */
            chain_tidal.data[3] += fac2 * dr[0]*dr[1];        /* xy */
            chain_tidal.data[4] += fac2 * dr[1]*dr[2];        /* yz */
            chain_tidal.data[5] += fac2 * dr[0]*dr[2];        /* xz */
        }
        for(int j = 0; j < 3; j++) ketju_member(idx).com_accel[j] = P[idx].GravAccel[j] - chain_grav[j];
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
        SymmetricTensor2<MyFloat> external_tidal = P[idx].tidal_tensorps - chain_tidal;
        double tidal_norm = external_tidal.frobenius_norm();
        ketju_member(idx).com_dt_tidal = (tidal_norm > 0) ? sqrt(1.0 / tidal_norm) : MAX_REAL_NUMBER;
#else
        ketju_member(idx).com_dt_tidal = MAX_REAL_NUMBER; (void)chain_tidal; /* no tree tidal tensor: no tidal limit, but marks the external values as set */
#endif
    }
}

void ketju_finish_step(void)
{
    KjClock clk(CPU_KETJUFINISH);
    /* chain-external field for members ending their step here, read by the next find_timesteps */
    ketju_compute_com_quantities();

    /* NOTE: KetjuIntegrated flags are NOT cleared here — they persist until
     * ketju_find_regions() at the start of the next step, so that guard checks
     * in kicks.cc/predict.cc/gravtree.cc can see which particles were KETJU-integrated */

    /* free region data (integrators, particle arrays, communicators) */
    ActiveRegions.clear();
}

/* ============================================================
 *  Phase C: HDF5 output for KETJU diagnostics
 *  Based on public GADGET4-KETJU (Mannerkoski+ 2023).
 *  Writes per-BH trajectories, region data, and merger events.
 *  Only Task 0 writes; data collected from compute roots.
 * ============================================================ */

struct ketju_bh_output {
    double pos[3], vel[3], spin[3];
    double mass;
    int timestep_index, n_particles, n_bh;
};

struct ketju_merger_output {
    MyIDType id1, id2;
    double m1, m2, m_remnant;
    double time, redshift;
};

static hid_t ketju_output_file = -1;
static int ketju_output_tstep_index = 0;

void ketju_open_output_file(void)
{
    if(ThisTask != 0) return;
    char buf[512];
    snprintf(buf, sizeof(buf), "%sketju_output.hdf5", All.OutputDir);
    if(RestartFlag == 0) {
        ketju_output_file = H5Fcreate(buf, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        /* create groups */
        hid_t grp = H5Gcreate2(ketju_output_file, "BHs", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Gclose(grp);
        /* create extendable timestep dataset */
        hsize_t dims[1] = {0}, maxdims[1] = {H5S_UNLIMITED}, chunk[1] = {64};
        hid_t space = H5Screate_simple(1, dims, maxdims);
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);
        H5Dcreate2(ketju_output_file, "timesteps", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Pclose(dcpl); H5Sclose(space);
        /* create extendable merger dataset */
        hid_t merger_type = H5Tcreate(H5T_COMPOUND, sizeof(ketju_merger_output));
        H5Tinsert(merger_type, "ID1", HOFFSET(ketju_merger_output, id1), H5T_NATIVE_LLONG);
        H5Tinsert(merger_type, "ID2", HOFFSET(ketju_merger_output, id2), H5T_NATIVE_LLONG);
        H5Tinsert(merger_type, "M1", HOFFSET(ketju_merger_output, m1), H5T_NATIVE_DOUBLE);
        H5Tinsert(merger_type, "M2", HOFFSET(ketju_merger_output, m2), H5T_NATIVE_DOUBLE);
        H5Tinsert(merger_type, "M_remnant", HOFFSET(ketju_merger_output, m_remnant), H5T_NATIVE_DOUBLE);
        H5Tinsert(merger_type, "time", HOFFSET(ketju_merger_output, time), H5T_NATIVE_DOUBLE);
        H5Tinsert(merger_type, "redshift", HOFFSET(ketju_merger_output, redshift), H5T_NATIVE_DOUBLE);
        dims[0] = 0;
        space = H5Screate_simple(1, dims, maxdims);
        dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);
        H5Dcreate2(ketju_output_file, "mergers", merger_type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Pclose(dcpl); H5Sclose(space); H5Tclose(merger_type);
    } else {
        ketju_output_file = H5Fopen(buf, H5F_ACC_RDWR, H5P_DEFAULT);
        /* read current timestep index */
        hid_t ds = H5Dopen2(ketju_output_file, "timesteps", H5P_DEFAULT);
        hid_t sp = H5Dget_space(ds);
        hsize_t dims[1]; H5Sget_simple_extent_dims(sp, dims, NULL);
        ketju_output_tstep_index = (int)dims[0];
        H5Sclose(sp); H5Dclose(ds);
    }
    if(ThisTask == 0 && ketju_output_file >= 0)
        printf("KETJU: Opened HDF5 output file (tstep_index=%d)\n", ketju_output_tstep_index);
}

void ketju_close_output_file(void)
{
    if(ThisTask == 0 && ketju_output_file >= 0) {
        H5Fclose(ketju_output_file);
        ketju_output_file = -1;
    }
}

static void ketju_hdf5_append_double(const char *dset_name, double value)
{
    if(ThisTask != 0 || ketju_output_file < 0) return;
    hid_t ds = H5Dopen2(ketju_output_file, dset_name, H5P_DEFAULT);
    hid_t sp = H5Dget_space(ds);
    hsize_t dims[1]; H5Sget_simple_extent_dims(sp, dims, NULL);
    hsize_t newdims[1] = {dims[0] + 1};
    H5Dset_extent(ds, newdims);
    H5Sclose(sp);
    sp = H5Dget_space(ds);
    hsize_t offset[1] = {dims[0]}, count[1] = {1};
    H5Sselect_hyperslab(sp, H5S_SELECT_SET, offset, NULL, count, NULL);
    hid_t memsp = H5Screate_simple(1, count, NULL);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, memsp, sp, H5P_DEFAULT, &value);
    H5Sclose(memsp); H5Sclose(sp); H5Dclose(ds);
}

static void ketju_hdf5_write_merger(ketju_merger_output *merger)
{
    if(ThisTask != 0 || ketju_output_file < 0) return;
    hid_t ds = H5Dopen2(ketju_output_file, "mergers", H5P_DEFAULT);
    hid_t sp = H5Dget_space(ds);
    hsize_t dims[1]; H5Sget_simple_extent_dims(sp, dims, NULL);
    hsize_t newdims[1] = {dims[0] + 1};
    H5Dset_extent(ds, newdims);
    H5Sclose(sp);
    sp = H5Dget_space(ds);
    hsize_t offset[1] = {dims[0]}, count[1] = {1};
    H5Sselect_hyperslab(sp, H5S_SELECT_SET, offset, NULL, count, NULL);
    hid_t memsp = H5Screate_simple(1, count, NULL);
    hid_t mtype = H5Dget_type(ds);
    H5Dwrite(ds, mtype, memsp, sp, H5P_DEFAULT, merger);
    H5Tclose(mtype); H5Sclose(memsp); H5Sclose(sp); H5Dclose(ds);
}

void ketju_write_output(void)
{
    if(ActiveRegions.empty()) return;
    if(ThisTask != 0 && ketju_output_file < 0) return; /* only task 0 has the file */

    /* write timestep */
    double phys_time = All.Time; /* scale factor in cosmo, time otherwise */
    ketju_hdf5_append_double("timesteps", phys_time);
    ketju_output_tstep_index++;
}

#endif /* KETJU_REGULARIZATION */
