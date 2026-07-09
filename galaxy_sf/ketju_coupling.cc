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
#include <algorithm>
#include <cctype>
#include <vector>
#include <set>
#include <unordered_map>
#include <queue>
#include <utility>
#include <cfloat>
#include <queue>
#include <functional>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"   /* kernel_gravity: match the tree's softened force law exactly */
#include "ketju_coupling.h"
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
#include "resolvedism_stellar_tables.h"
#endif

#ifdef KETJU_REGULARIZATION

/* Cost tracking for load balancing (keyed by first center ID per region) */
static std::unordered_map<MyIDType, double> region_previous_cost;

/* ============================================================
 *  MSTAR NATURAL UNITS (2026-07-06, after the Stella port)
 *  MSTAR's GBS error control is not scale-free: fed Stella host units
 *  (kpc / 1e10 Msun), a trivial 1000 AU circular binary has code mass 1e-10
 *  and orbital energy ~1e-11 and the integrator grinds through >1e5 steps per
 *  call at dE/E ~ 1e-6, while the identical system in pc / Msun / km/s runs
 *  in O(100) steps at dE/E ~ 1e-13.  We therefore integrate in NATURAL N-body
 *  units (pc, Msun, km/s) regardless of host units: the state is converted in
 *  place at integrate_region entry and back-converted at scatter_results entry
 *  (ps is freshly host-filled by setup/setup_reuse before every integration,
 *  so exactly one round trip per step).  Everything outside the MSTAR interface
 *  stays in host units.
 * ============================================================ */
#define KJ_LEN_TO_NAT  (All.UnitLength_in_cm / 3.085678e18)      /* host length unit in pc  */
#define KJ_MASS_TO_NAT (All.UnitMass_in_g   / 1.989e33)          /* host mass unit in Msun  */
#define KJ_VEL_TO_NAT  (All.UnitVelocity_in_cm_per_s / 1.0e5)    /* host vel unit in km/s   */
#define KJ_TIME_TO_NAT (KJ_LEN_TO_NAT / KJ_VEL_TO_NAT)           /* host time unit in pc/(km/s) */

static void kj_state_change_units(struct ketju_system_physical_state *ps, int n, int to_natural)
{
    double fL = KJ_LEN_TO_NAT, fM = KJ_MASS_TO_NAT, fV = KJ_VEL_TO_NAT;
    if(!to_natural) { fL = 1./fL; fM = 1./fM; fV = 1./fV; }
    for(int i = 0; i < n; i++) {
        ps->mass[i] *= fM;
        for(int j = 0; j < 3; j++) { ps->pos[i][j] *= fL; ps->vel[i][j] *= fV; }
#ifdef SINK_PARTICLES
        for(int j = 0; j < 3; j++) { ps->spin[i][j] *= fL * fM * fV; }
#endif
    }
    ps->time *= (to_natural ? (fL/fV) : (fL/fV)); /* fL,fV already inverted in host branch */
}

/* ============================================================
 *  Lightweight particle data for MPI communication
 * ============================================================ */
struct ketju_mpi_particle {
    MyIDType ID;
    int Type;
    int Task;
    int Index;
    double Mass;
    double Pos[3];
    double Vel[3];
    double GravAccel[3]; /* full tree acceleration (incl. member-member; G folded in) — used to build the external perturbation fed to MSTAR */
    short int is_dead_remnant; /* 1 = dead Type 4 star (NS/BH), 0 = living star or sink */
#ifdef SINK_PARTICLES
    short int SinkSubType;
    double Spin[3];
#endif
#ifdef KETJU_PN_REMNANT_TAG
    signed char RemnantType;   /* StellarRemnantType (-1 = alive) — for PN dispatch on Type 4 remnants */
#endif
};

/* ============================================================
 *  Per-particle extra data tracked alongside integrator arrays
 * ============================================================ */
struct ketju_extra_data {
    MyIDType ID;
    int Task;
    int Index;
    int Type;
#ifdef SINK_PARTICLES
    short int SinkSubType;
#endif
};

/* ============================================================
 *  MPI task group for parallel KETJU computation
 *  Based on public GADGET4-KETJU (Mannerkoski+ 2023)
 * ============================================================ */
struct KetjuTaskGroup {
    MPI_Group group;
    MPI_Comm comm;
    int rank;      /* rank within this group, MPI_UNDEFINED if not a member */
    int size;
    int root;      /* rank of root within the group */
    int root_sim;  /* rank of root in MPI_COMM_WORLD */

    KetjuTaskGroup() : group(MPI_GROUP_NULL), comm(MPI_COMM_NULL), rank(MPI_UNDEFINED), size(0), root(MPI_UNDEFINED), root_sim(MPI_UNDEFINED) {}

    void init(const std::vector<int> &task_sim_indices, int tag = 0) {
        MPI_Group world_group;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        MPI_Group_incl(world_group, task_sim_indices.size(), task_sim_indices.data(), &group);
        MPI_Group_size(group, &size);
        MPI_Group_rank(group, &rank);
        MPI_Comm_create_group(MPI_COMM_WORLD, group, tag, &comm);
        MPI_Group_free(&world_group);
        root = 0;
        root_sim = task_sim_indices[0];
    }

    void free_comms() {
        if(comm != MPI_COMM_NULL) { MPI_Comm_free(&comm); comm = MPI_COMM_NULL; }
        if(group != MPI_GROUP_NULL) { MPI_Group_free(&group); group = MPI_GROUP_NULL; }
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
    std::vector<double> mstar_end_vel;             /* compute-root: ps->vel captured right after MSTAR, before neg2 (true t_sync velocity), 3*n */
    std::vector<double> external_accel;            /* compute tasks: per-member EXTERNAL accel = GravAccel - member-member force, in sorted order, 3*n */
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

    /* CoM frame data (stored at setup, used in scatter) */
    double com_pos[3];           /* region CoM position; added back to MSTAR-relative positions at scatter */
    double com_vel[3];           /* region CoM velocity; added back to MSTAR-relative velocities at scatter */
    integertime ti_step;     /* integer timestep for this region */

    int state_in_natural_units = 0;  /* 1 while ps holds MSTAR natural units (pc/Msun/km-s); guards against double conversion when integrate_region runs more than once before scatter */
    KetjuRegion() : total_particle_count(0), num_pn_particles(0), integrator(NULL), ti_step(0) {
        for(int j = 0; j < 3; j++) { com_pos[j] = 0; com_vel[j] = 0; }
        compute_info.compute_sequence_position = 0;
        compute_info.first_task_index = 0;
        compute_info.final_task_index = 0;
    }
    ~KetjuRegion() {
        if(integrator) { ketju_free_system(integrator); free(integrator); }
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
static std::set<int> AllKetjuParticleIndices; /* local indices of all particles in any region */

/* ============================================================
 *  Region persistence: cached regions across steps
 *  Communicators are preserved when membership is unchanged.
 * ============================================================ */
struct CachedRegionKey {
    MyIDType ID;
    int Task;
    int Index; /* local P[] index — stable between domain decompositions */
    bool operator==(const CachedRegionKey &o) const { return ID == o.ID && Task == o.Task && Index == o.Index; }
    bool operator<(const CachedRegionKey &o) const {
        if(ID != o.ID) return ID < o.ID;
        if(Task != o.Task) return Task < o.Task;
        return Index < o.Index;
    }
};

struct CachedRegionInfo {
    std::vector<CachedRegionKey> sorted_particle_keys; /* sorted (ID,Task) for staleness check */
    KetjuTaskGroup affected_tasks;
    KetjuTaskGroup compute_tasks;
    std::vector<int> affected_sim_indices;
    std::vector<int> particle_counts_on_affected;
    region_compute_info compute_info;
    int total_particle_count;

    CachedRegionInfo() : total_particle_count(0) {}
    ~CachedRegionInfo() {
        affected_tasks.free_comms();
        compute_tasks.free_comms();
    }
    /* move-only to protect MPI handles */
    CachedRegionInfo(CachedRegionInfo &&o) noexcept
        : sorted_particle_keys(std::move(o.sorted_particle_keys)),
          affected_sim_indices(std::move(o.affected_sim_indices)),
          particle_counts_on_affected(std::move(o.particle_counts_on_affected)),
          compute_info(o.compute_info),
          total_particle_count(o.total_particle_count)
    {
        affected_tasks = o.affected_tasks; o.affected_tasks.comm = MPI_COMM_NULL; o.affected_tasks.group = MPI_GROUP_NULL;
        compute_tasks = o.compute_tasks; o.compute_tasks.comm = MPI_COMM_NULL; o.compute_tasks.group = MPI_GROUP_NULL;
    }
    CachedRegionInfo &operator=(CachedRegionInfo &&o) noexcept {
        if(this != &o) {
            affected_tasks.free_comms(); compute_tasks.free_comms();
            sorted_particle_keys = std::move(o.sorted_particle_keys);
            affected_sim_indices = std::move(o.affected_sim_indices);
            particle_counts_on_affected = std::move(o.particle_counts_on_affected);
            compute_info = o.compute_info; total_particle_count = o.total_particle_count;
            affected_tasks = o.affected_tasks; o.affected_tasks.comm = MPI_COMM_NULL; o.affected_tasks.group = MPI_GROUP_NULL;
            compute_tasks = o.compute_tasks; o.compute_tasks.comm = MPI_COMM_NULL; o.compute_tasks.group = MPI_GROUP_NULL;
        }
        return *this;
    }
    CachedRegionInfo(const CachedRegionInfo &) = delete;
    CachedRegionInfo &operator=(const CachedRegionInfo &) = delete;
};

static std::vector<CachedRegionInfo> CachedRegions;
static int KetjuRegionsStale = 1; /* 1 = force rebuild (after domain decomp or first call) */

/* Cache for gather_chain_centers result — shared between ketju_limit_timesteps
 * and ketju_find_regions to avoid a redundant MPI_Allgatherv on COMM_WORLD */
static std::vector<ketju_mpi_particle> CachedChainCenters;
static int CachedChainCentersValid = 0;

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
    /* Stella: Type 4 stars above mass threshold */
    if(P[i].Type == 4 && All.KetjuMinStarMass > 0) {
        if(P[i].Mass >= All.KetjuMinStarMass) return 1;
    }
#ifdef SINK_PARTICLES
    /* Type 5 sinks above mass threshold */
    if(P[i].Type == 5 && All.KetjuMinBHMass > 0) {
        if(P[i].Mass >= All.KetjuMinBHMass) return 1;
    }
#endif
    return 0;
}

/* ============================================================
 *  Helper: is this particle eligible for chain membership?
 * ============================================================ */
static int is_chain_eligible(int i)
{
    if(P[i].Type == 4) return 1;
#ifdef SINK_PARTICLES
    if(P[i].Type == 5) return 1;
#endif
    return 0;
}

/* Helper: is this a dead remnant (NS or BH) still stored as Type 4? */
static int is_dead_remnant_type4(int i)
{
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
    if(P[i].Type == 4 && P[i].MstarSampleIMF[0] <= 0 && P[i].Mass > 0) return 1;
#endif
    return 0;
}

/* ============================================================
 *  Helper: is this particle PN-enabled? (compact objects only)
 * ============================================================ */
static int is_pn_particle(ketju_mpi_particle *p)
{
#ifdef SINK_PARTICLES
    if(p->Type == 5 && p->SinkSubType == 0) return 1; /* SMBH: always PN */
#ifdef KETJU_PN_COMPACT_OBJECTS
    if(p->Type == 5 && p->SinkSubType == 1) return 1; /* stellar-mass BH (promoted): PN */
#endif
#endif
#ifdef KETJU_PN_REMNANT_TAG
    /* Type 4 compact remnants (NS or BH variants only — exclude WD/PISN), tagged at SN time
     * by GALSF_RESOLVEDISM_FB.  No need for sink/BH_PROMOTION machinery. */
    if(p->Type == 4 && (p->RemnantType == REM_ECSN ||  /* electron-capture SN -> NS */
                        p->RemnantType == REM_CCSN ||  /* core-collapse SN -> NS or BH */
                        p->RemnantType == REM_FSN  ||  /* failed SN -> BH */
                        p->RemnantType == REM_PPISN || /* pulsational PISN -> BH */
                        p->RemnantType == REM_DBH))    /* direct BH */
        return 1;
#else
#ifdef KETJU_PN_COMPACT_OBJECTS
    /* Legacy path: any dead Type 4 remnant (NS or BH, but also WDs unless filtered upstream) */
    if(p->Type == 4 && p->is_dead_remnant) return 1;
#endif
#endif
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
 *  Helper: GIZMO softening kernel (same as gravtree.cc)
 * ============================================================ */
static double softened_force_factor(double r, double h)
{
    /* Returns G*m/r^2 factor with softening: equivalent to 1/r^2 for r>h */
    if(h <= 0 || r >= h) return 1.0 / (r * r * r);
    double h_inv = 1.0 / h;
    double u = r * h_inv;
    if(u < 0.5) {
        return h_inv * h_inv * h_inv * (32.0/3.0 + u*u*(32.0*u - 38.4));
    } else {
        return h_inv * h_inv * h_inv * (-1.0/(30.0*u*u*u) + 64.0/3.0 + u*u*(-48.0 + u*(38.4 - 32.0/3.0*u)));
    }
}

/* Store the EXTERNAL acceleration on each region member. The gravity tree now EXCLUDES
 * member<->member interactions for chain particles (see KetjuRegionTag / forcetree.cc), so a
 * member's gathered GravAccel already IS the exact external (non-member) field — computed with
 * the tree's own force law (adaptive/few-body softening included), so there is no hand-rebuilt
 * internal force to subtract and thus no softening/kernel mismatch to leak energy at tight
 * high-eccentricity pericenters. For an isolated region GravAccel = 0 exactly, so external = 0
 * and the scheme reduces to pure-MSTAR. The host KDK applies this external field to members;
 * MSTAR integrates the internal forces. Result (sorted order, aligned with reg.all_particles)
 * is stored in reg.external_accel, redundantly on every compute task. */
static void ketju_compute_external_accel(KetjuRegion &reg)
{
    if(!reg.compute_tasks.is_member()) return;
    int n = reg.total_particle_count;
    reg.external_accel.assign(3 * n, 0.0);
    for(int i = 0; i < n; i++) {
        for(int k = 0; k < 3; k++)
            reg.external_accel[3*i + k] = reg.all_particles[i].GravAccel[k];
    }
#ifdef KETJU_ENERGY_TRACE
    if(reg.compute_tasks.is_root() && n == 2) {
        double ga=0, ai=0, ex=0;
        for(int k=0;k<3;k++){ ga+=reg.all_particles[0].GravAccel[k]*reg.all_particles[0].GravAccel[k];
            ex+=reg.external_accel[k]*reg.external_accel[k]; }
        double dr2=0; for(int k=0;k<3;k++){double d=reg.all_particles[0].Pos[k]-reg.all_particles[1].Pos[k]; dr2+=d*d;}
        printf("EXTTRACE: |GravAccel0|=%.6g |external0|=%.6g r=%.4g AU\n", sqrt(ga), sqrt(ex), sqrt(dr2)*206265.0);
        (void)ai;
    }
#endif
}

/* ============================================================
 *  Parse PN terms string (same logic as GADGET4-KETJU)
 * ============================================================ */
static int parse_pn_terms(void)
{
    char terms[20];
    strncpy(terms, All.KetjuPNTerms, sizeof(terms));
    terms[sizeof(terms)-1] = '\0';
    for(int i = 0; terms[i]; i++) terms[i] = tolower(terms[i]);

    if(strcmp(terms, "all") == 0) return KETJU_PN_ALL;
    if(strcmp(terms, "no_spin") == 0) return KETJU_PN_DYNAMIC_ALL;
    if(strcmp(terms, "none") == 0) return KETJU_PN_NONE;

    int flags = KETJU_PN_NONE;
    for(int i = 0; terms[i]; i++) {
        switch(terms[i]) {
            case '2': flags |= KETJU_PN_1_0_ACC; break;
            case '4': flags |= KETJU_PN_2_0_ACC; break;
            case '5': flags |= KETJU_PN_2_5_ACC; break;
            case '6': flags |= KETJU_PN_3_0_ACC; break;
            case '7': flags |= KETJU_PN_3_5_ACC; break;
            case 's': flags |= KETJU_PN_SPIN_ALL; break;
            case 'c': flags |= KETJU_PN_THREEBODY; break;
            default:
                if(ThisTask == 0) printf("KETJU: Warning: unknown PN term character '%c'\n", terms[i]);
                break;
        }
    }
    return flags;
}

/* ============================================================
 *  PHASE 2: Find chain centers and build regions
 * ============================================================ */

/* Gather all chain centers across MPI tasks */
static std::vector<ketju_mpi_particle> gather_chain_centers(void)
{
    /* count local centers */
    int n_local = 0;
    for(int i = 0; i < NumPart; i++) {
        if(is_chain_center(i)) n_local++;
    }

    /* fill local array */
    std::vector<ketju_mpi_particle> local_centers(n_local);
    int k = 0;
    for(int i = 0; i < NumPart; i++) {
        if(is_chain_center(i)) {
            local_centers[k].ID = P[i].ID;
            local_centers[k].Type = P[i].Type;
            local_centers[k].Task = ThisTask;
            local_centers[k].Index = i;
            local_centers[k].Mass = P[i].Mass;
            for(int j = 0; j < 3; j++) {
                local_centers[k].Pos[j] = P[i].Pos[j];
                local_centers[k].Vel[j] = P[i].Vel[j];
            }
            local_centers[k].is_dead_remnant = is_dead_remnant_type4(i);
#ifdef SINK_PARTICLES
            local_centers[k].SinkSubType = (P[i].Type == 5) ? P[i].SinkSubType : -1;
            for(int j = 0; j < 3; j++) local_centers[k].Spin[j] = (P[i].Type == 5) ? P[i].KetjuSpin[j] : 0;
#endif
#ifdef KETJU_PN_REMNANT_TAG
            local_centers[k].RemnantType = P[i].RemnantType;
#endif
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

/* Merge overlapping regions: two centers within 2*r_region get merged */
static std::vector<std::vector<ketju_mpi_particle>> merge_overlapping_regions(
    const std::vector<ketju_mpi_particle> &centers, double merge_radius)
{
    int n = centers.size();
    std::vector<int> group(n);
    for(int i = 0; i < n; i++) group[i] = i;

    /* union-find */
    auto find = [&](int x) {
        while(group[x] != x) { group[x] = group[group[x]]; x = group[x]; }
        return x;
    };
    auto unite = [&](int a, int b) {
        a = find(a); b = find(b);
        if(a != b) group[a] = b;
    };

    for(int i = 0; i < n; i++) {
        for(int j = i+1; j < n; j++) {
            double dist = particle_distance(
                const_cast<double*>(centers[i].Pos),
                const_cast<double*>(centers[j].Pos));
            if(dist < merge_radius) unite(i, j);
        }
    }

    /* group centers by their root */
    std::vector<std::vector<ketju_mpi_particle>> regions;
    std::vector<int> root_to_region(n, -1);
    for(int i = 0; i < n; i++) {
        int r = find(i);
        if(root_to_region[r] < 0) {
            root_to_region[r] = regions.size();
            regions.push_back({});
        }
        regions[root_to_region[r]].push_back(centers[i]);
    }

    return regions;
}

/* ============================================================
 *  BIFROST-style ORBIT-BASED subsystem membership
 *  (Rantala, Naab et al. 2023, MNRAS 522, 5180, Sec 3.4)
 *
 *  Membership from PAIR ORBITAL ELEMENTS instead of the instantaneous
 *  separation. The whole point: capture/release handoffs must happen in the
 *  WEAK-COUPLING regime. A pure distance trigger (separation < 2R) fires at
 *  the deepest, fastest point of an eccentric passage, where any handoff
 *  bookkeeping imperfection is amplified into a large energy error (measured:
 *  the e=0.99 cluster is clean at 1e-5 when binaries never cross the boundary
 *  and blows up to dE ~ 0.1-1 the moment per-orbit boundary crossings begin).
 *  Orbit-based membership instead gives:
 *    - bound pair with semi-major axis a < r_ngb: PERMANENT member. A tight
 *      binary never churns in/out at apocenter — membership is a property of
 *      the orbit, not the phase.
 *    - any pair with predicted pericenter r_peri < r_ngb inside an
 *      anticipatory time window r/|v_rel| < tau: captured EARLY on approach
 *      (typically at ~K*r_ngb where the mutual force is ~K^2 weaker), carried
 *      by MSTAR through the entire deep passage, released at the symmetric
 *      weak point outbound (r/|v_rel| grows back through tau).
 * ============================================================ */
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
    const double K = 16.0;
    if(r >= K * r_ngb) return 0;
    double tau = K * sqrt(K) * r_ngb * sqrt(r_ngb) / sqrt(2.0 * GM);
    if(r / sqrt(v2) < tau) return 1;
    return 0;
}

/* Group eligible stars into subsystems by the pair-orbit predicate (union-find, same output
 * shape as merge_overlapping_regions). Deterministic from the globally-gathered centers, so
 * all ranks derive identical groups without communication. */
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

    for(int i = 0; i < n; i++)
        for(int j = i + 1; j < n; j++)
            if(pair_forms_subsystem(centers[i], centers[j])) unite(i, j);

    std::vector<std::vector<ketju_mpi_particle>> regions;
    std::vector<int> root_to_region(n, -1);
    for(int i = 0; i < n; i++) {
        int r = find(i);
        if(root_to_region[r] < 0) { root_to_region[r] = regions.size(); regions.push_back({}); }
        regions[root_to_region[r]].push_back(centers[i]);
    }
    return regions;
}

/* Membership of an orbit-based group: the group IS the member list (each center is an eligible
 * star). Local members = the centers this rank owns, filtered by the capture admission gate. */
static std::set<int> local_members_from_group(const std::vector<ketju_mpi_particle> &group)
{
    std::set<int> members;
    for(const ketju_mpi_particle &c : group) {
        if(c.Task != ThisTask) continue;
        int i = c.Index;
        if(!is_chain_eligible(i)) continue;
        /* CAPTURE ADMISSION GATE (see find_local_members): only capture particles that are ACTIVE at
         * this sync or already members — never overwrite a mid-bin particle's state. */
        if(!TimeBinActive[P[i].TimeBin] && !P[i].KetjuIntegrated) continue;
        members.insert(i);
    }
    return members;
}

/* Find local particles within r_region of any center in a region */
static std::set<int> find_local_members(const std::vector<ketju_mpi_particle> &centers, double radius)
{
    std::set<int> members;
    for(int i = 0; i < NumPart; i++) {
        if(!is_chain_eligible(i)) continue;
        /* CAPTURE ADMISSION GATE: a particle may be captured into a region only at a sync point where it
         * is ACTIVE (or if it already is a member from the previous step — membership stays sticky so
         * live regions are never ripped apart mid-step). Capturing an INACTIVE particle lets MSTAR's
         * scatter overwrite it mid-way through its host bin, shredding its Hermite history and the
         * velocity-trick drift chord (episodic-capture blowups: dE +194..+203 at the first e=0.99
         * pericenter). A deferred star leaves its companion in a <2-member region, which is skipped
         * entirely, so the capture just waits (a step or two — the buffer pass keeps approaching pairs
         * on short synchronized bins) until both are active together. Consistent across
         * ketju_limit_timesteps and ketju_find_regions: PASS 1 sets KetjuIntegrated for admitted
         * members, so the later find_regions call reproduces the same admission decisions. */
        if(!TimeBinActive[P[i].TimeBin] && !P[i].KetjuIntegrated) continue;
        for(size_t c = 0; c < centers.size(); c++) {
            double dist = particle_distance(P[i].Pos, const_cast<double*>(centers[c].Pos));
            if(dist < radius) { members.insert(i); break; }
        }
    }
    return members;
}

/* ============================================================
 *  PHASE 3: MPI gather and integrator setup
 * ============================================================ */

static void setup_integrator(KetjuRegion &reg)
{
    /* gather particle counts per task */
    int n_local = reg.local_member_indices.size();
    std::vector<int> counts(NTask);
    MPI_Allgather(&n_local, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

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
        mp.ID = P[idx].ID;
        mp.Type = P[idx].Type;
        mp.Task = ThisTask;
        mp.Index = idx;
        mp.Mass = P[idx].Mass;
        for(int j = 0; j < 3; j++) {
            mp.Pos[j] = P[idx].Pos[j];
            mp.Vel[j] = P[idx].Vel[j];
            mp.GravAccel[j] = P[idx].GravAccel[j];
        }
        mp.is_dead_remnant = is_dead_remnant_type4(idx);
#ifdef SINK_PARTICLES
        mp.SinkSubType = (P[idx].Type == 5) ? P[idx].SinkSubType : -1;
        for(int j = 0; j < 3; j++) mp.Spin[j] = (P[idx].Type == 5) ? P[idx].KetjuSpin[j] : 0;
#endif
#ifdef KETJU_PN_REMNANT_TAG
        mp.RemnantType = P[idx].RemnantType;
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
        /* count PN particles */
        reg.num_pn_particles = 0;
        for(int i = 0; i < reg.total_particle_count; i++) {
            if(is_pn_particle(&reg.all_particles[i])) reg.num_pn_particles++;
        }

        /* sort: PN particles first */
        std::stable_sort(reg.all_particles.begin(), reg.all_particles.end(),
            [](const ketju_mpi_particle &a, const ketju_mpi_particle &b) {
                int a_pn = 0, b_pn = 0;
#ifdef SINK_PARTICLES
                if(a.Type == 5 && a.SinkSubType == 0) a_pn = 1;
                if(b.Type == 5 && b.SinkSubType == 0) b_pn = 1;
#ifdef KETJU_PN_COMPACT_OBJECTS
                if(a.Type == 5 && a.SinkSubType == 1) a_pn = 1;
                if(b.Type == 5 && b.SinkSubType == 1) b_pn = 1;
#endif
#endif
#ifdef KETJU_PN_REMNANT_TAG
                /* keep the sort consistent with is_pn_particle: NS/BH Type-4 remnants are PN */
                if(a.Type == 4 && (a.RemnantType == REM_ECSN || a.RemnantType == REM_CCSN || a.RemnantType == REM_FSN || a.RemnantType == REM_PPISN || a.RemnantType == REM_DBH)) a_pn = 1;
                if(b.Type == 4 && (b.RemnantType == REM_ECSN || b.RemnantType == REM_CCSN || b.RemnantType == REM_FSN || b.RemnantType == REM_PPISN || b.RemnantType == REM_DBH)) b_pn = 1;
#elif defined(KETJU_PN_COMPACT_OBJECTS)
                if(a.Type == 4 && a.is_dead_remnant) a_pn = 1;
                if(b.Type == 4 && b.is_dead_remnant) b_pn = 1;
#endif
                return a_pn > b_pn;
            });

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
        ketju_create_system(reg.integrator, reg.num_pn_particles, n_other, reg.compute_tasks.comm);

        /* set units */
        /* natural units (see KJ_* above): G and c in pc / Msun / km/s */
        reg.integrator->constants->G = All.G * KJ_LEN_TO_NAT * KJ_VEL_TO_NAT * KJ_VEL_TO_NAT / KJ_MASS_TO_NAT;
#if defined(C_LIGHT_CODE)
        reg.integrator->constants->c = C_LIGHT_CODE;
#else
        reg.integrator->constants->c = 2.9979e10 / 1.0e5; /* km/s: natural velocity unit */
#endif

        /* set options */
        reg.integrator->options->PN_flags = parse_pn_terms();
        reg.integrator->options->gbs_relative_tolerance = All.KetjuIntegrationTolerance;
        reg.integrator->options->enable_bh_mergers = (All.KetjuEnableBHMergerKicks >= 0);
        reg.integrator->options->enable_bh_merger_kicks = All.KetjuEnableBHMergerKicks;
        if(All.KetjuMaxStepCount > 0) reg.integrator->options->max_step_count = All.KetjuMaxStepCount;
        /* star_star_softening left at the integrator's library default (0): gravity
         * INSIDE a KETJU region is ALWAYS unsoftened — that is the entire point of
         * regularization. The former KetjuUseStarStarSoftening path (which injected
         * the host tree softening into the chain) was a Stella-side band-aid that
         * defeated regularization; removed 2026-07-08. */

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
            reg.extra_data[i].ID = reg.all_particles[i].ID;
            reg.extra_data[i].Task = reg.all_particles[i].Task;
            reg.extra_data[i].Index = reg.all_particles[i].Index;
            reg.extra_data[i].Type = reg.all_particles[i].Type;
#ifdef SINK_PARTICLES
            reg.extra_data[i].SinkSubType = reg.all_particles[i].SinkSubType;
            if(i < reg.num_pn_particles) {
                for(int j = 0; j < 3; j++) ps->spin[i][j] = reg.all_particles[i].Spin[j];
            }
#endif
        }
        reg.integrator->particle_extra_data = reg.extra_data.data();
        reg.integrator->particle_extra_data_elem_size = sizeof(ketju_extra_data);

        if(reg.compute_tasks.is_root() && ThisTask == 0) {
            printf("KETJU: Region with %d particles (%d PN), compute tasks %d-%d\n",
                   reg.total_particle_count, reg.num_pn_particles,
                   reg.compute_info.first_task_index, reg.compute_info.final_task_index);
        }
        ketju_compute_external_accel(reg);  /* external = GravAccel (member-member excluded in tree), for the host kick */
    }

    /* CoM was computed on compute members only (above). The broadcast below is rooted at
     * affected_tasks.root, which need NOT be a compute member — in that case it still holds
     * the constructor's zero CoM and would wipe every binary's bulk (streaming) velocity at
     * scatter (KetjuFinalVel = sp.Vel + 0). Forward the authoritative CoM from compute_root
     * (always a compute member) to affected_root first, mirroring the all_particles forward. */
    if(reg.compute_tasks.root_sim != reg.affected_tasks.root_sim) {
        double combuf[6];
        if(ThisTask == reg.compute_tasks.root_sim) {
            for(int j = 0; j < 3; j++) { combuf[j] = reg.com_pos[j]; combuf[j+3] = reg.com_vel[j]; }
            MPI_Send(combuf, 6, MPI_DOUBLE, reg.affected_tasks.root_sim, 997, MPI_COMM_WORLD);
        }
        if(ThisTask == reg.affected_tasks.root_sim) {
            MPI_Recv(combuf, 6, MPI_DOUBLE, reg.compute_tasks.root_sim, 997, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(int j = 0; j < 3; j++) { reg.com_pos[j] = combuf[j]; reg.com_vel[j] = combuf[j+3]; }
        }
    }

    /* broadcast CoM data to all tasks that need it (affected group for scatter) */
    if(reg.affected_tasks.is_member()) {
        MPI_Bcast(reg.com_pos, 3, MPI_DOUBLE, reg.affected_tasks.root, reg.affected_tasks.comm);
        MPI_Bcast(reg.com_vel, 3, MPI_DOUBLE, reg.affected_tasks.root, reg.affected_tasks.comm);
    }
}

/* Setup integrator reusing existing communicators (region persistence).
 * Skips MPI_Comm_create_group — the expensive part — but still gathers
 * particle data and builds the integrator state for this step. */
static void setup_integrator_reuse(KetjuRegion &reg)
{
    if(reg.total_particle_count < 2) return;

    int n_local = reg.local_member_indices.size();

    /* fill local particle data */
    std::vector<ketju_mpi_particle> local_parts(n_local);
    int k = 0;
    for(int idx : reg.local_member_indices) {
        ketju_mpi_particle &mp = local_parts[k];
        mp.ID = P[idx].ID;
        mp.Type = P[idx].Type;
        mp.Task = ThisTask;
        mp.Index = idx;
        mp.Mass = P[idx].Mass;
        for(int j = 0; j < 3; j++) {
            mp.Pos[j] = P[idx].Pos[j];
            mp.Vel[j] = P[idx].Vel[j];
            mp.GravAccel[j] = P[idx].GravAccel[j];
        }
        mp.is_dead_remnant = is_dead_remnant_type4(idx);
#ifdef SINK_PARTICLES
        mp.SinkSubType = (P[idx].Type == 5) ? P[idx].SinkSubType : -1;
        for(int j = 0; j < 3; j++) mp.Spin[j] = (P[idx].Type == 5) ? P[idx].KetjuSpin[j] : 0;
#endif
#ifdef KETJU_PN_REMNANT_TAG
        mp.RemnantType = P[idx].RemnantType;
#endif
        k++;
    }

    /* gather all particles on affected root (reusing existing communicator) */
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

    /* forward from affected root to compute root if they differ */
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

    /* broadcast to all compute tasks */
    if(reg.compute_tasks.is_member()) {
        if(!reg.compute_tasks.is_root()) reg.all_particles.resize(reg.total_particle_count);
        MPI_Bcast(reg.all_particles.data(), reg.total_particle_count * sizeof(ketju_mpi_particle),
                  MPI_BYTE, reg.compute_tasks.root, reg.compute_tasks.comm);
    }

    /* set up integrator on compute tasks (same as setup_integrator from here) */
    if(reg.compute_tasks.is_member()) {
        reg.num_pn_particles = 0;
        for(int i = 0; i < reg.total_particle_count; i++) {
            if(is_pn_particle(&reg.all_particles[i])) reg.num_pn_particles++;
        }

        std::stable_sort(reg.all_particles.begin(), reg.all_particles.end(),
            [](const ketju_mpi_particle &a, const ketju_mpi_particle &b) {
                int a_pn = 0, b_pn = 0;
#ifdef SINK_PARTICLES
                if(a.Type == 5 && a.SinkSubType == 0) a_pn = 1;
                if(b.Type == 5 && b.SinkSubType == 0) b_pn = 1;
#ifdef KETJU_PN_COMPACT_OBJECTS
                if(a.Type == 5 && a.SinkSubType == 1) a_pn = 1;
                if(b.Type == 5 && b.SinkSubType == 1) b_pn = 1;
#endif
#endif
#ifdef KETJU_PN_REMNANT_TAG
                /* keep the sort consistent with is_pn_particle: NS/BH Type-4 remnants are PN */
                if(a.Type == 4 && (a.RemnantType == REM_ECSN || a.RemnantType == REM_CCSN || a.RemnantType == REM_FSN || a.RemnantType == REM_PPISN || a.RemnantType == REM_DBH)) a_pn = 1;
                if(b.Type == 4 && (b.RemnantType == REM_ECSN || b.RemnantType == REM_CCSN || b.RemnantType == REM_FSN || b.RemnantType == REM_PPISN || b.RemnantType == REM_DBH)) b_pn = 1;
#elif defined(KETJU_PN_COMPACT_OBJECTS)
                if(a.Type == 4 && a.is_dead_remnant) a_pn = 1;
                if(b.Type == 4 && b.is_dead_remnant) b_pn = 1;
#endif
                return a_pn > b_pn;
            });

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

        reg.integrator = (struct ketju_system *)calloc(1, sizeof(struct ketju_system));
        int n_other = reg.total_particle_count - reg.num_pn_particles;
        ketju_create_system(reg.integrator, reg.num_pn_particles, n_other, reg.compute_tasks.comm);

        /* natural units (see KJ_* above): G and c in pc / Msun / km/s */
        reg.integrator->constants->G = All.G * KJ_LEN_TO_NAT * KJ_VEL_TO_NAT * KJ_VEL_TO_NAT / KJ_MASS_TO_NAT;
#if defined(C_LIGHT_CODE)
        reg.integrator->constants->c = C_LIGHT_CODE;
#else
        reg.integrator->constants->c = 2.9979e10 / 1.0e5; /* km/s: natural velocity unit */
#endif
        reg.integrator->options->PN_flags = parse_pn_terms();
        reg.integrator->options->gbs_relative_tolerance = All.KetjuIntegrationTolerance;
        reg.integrator->options->enable_bh_mergers = (All.KetjuEnableBHMergerKicks >= 0);
        reg.integrator->options->enable_bh_merger_kicks = All.KetjuEnableBHMergerKicks;
        if(All.KetjuMaxStepCount > 0) reg.integrator->options->max_step_count = All.KetjuMaxStepCount;
        /* star_star_softening left at library default (0): unsoftened inside the
         * region, always (see the matching note above; removed 2026-07-08). */

        reg.extra_data.resize(reg.total_particle_count);
        struct ketju_system_physical_state *ps = reg.integrator->physical_state;
        ps->time = 0;
        double a = All.cf_atime;
        for(int i = 0; i < reg.total_particle_count; i++) {
            ps->mass[i] = reg.all_particles[i].Mass;
            for(int j = 0; j < 3; j++) {
                ps->pos[i][j] = (reg.all_particles[i].Pos[j] - reg.com_pos[j]) * a;
                double v_pec_rel = (reg.all_particles[i].Vel[j] - reg.com_vel[j]) / a;
                ps->vel[i][j] = v_pec_rel;
                if(All.ComovingIntegrationOn)
                    ps->vel[i][j] += hubble_function(a) * ps->pos[i][j];
            }
            reg.extra_data[i].ID = reg.all_particles[i].ID;
            reg.extra_data[i].Task = reg.all_particles[i].Task;
            reg.extra_data[i].Index = reg.all_particles[i].Index;
            reg.extra_data[i].Type = reg.all_particles[i].Type;
#ifdef SINK_PARTICLES
            reg.extra_data[i].SinkSubType = reg.all_particles[i].SinkSubType;
            if(i < reg.num_pn_particles) {
                for(int j = 0; j < 3; j++) ps->spin[i][j] = reg.all_particles[i].Spin[j];
            }
#endif
        }
        reg.integrator->particle_extra_data = reg.extra_data.data();
        reg.integrator->particle_extra_data_elem_size = sizeof(ketju_extra_data);

        if(reg.compute_tasks.is_root() && ThisTask == 0) {
            printf("KETJU: Region with %d particles (%d PN), compute tasks %d-%d [cached comms]\n",
                   reg.total_particle_count, reg.num_pn_particles,
                   reg.compute_info.first_task_index, reg.compute_info.final_task_index);
        }
        ketju_compute_external_accel(reg);  /* external = GravAccel (member-member excluded in tree), for the host kick */
    }

    /* see setup_integrator: forward authoritative CoM compute_root -> affected_root before the
     * affected-group broadcast, else a non-compute affected_root broadcasts a zero CoM and the
     * binaries' bulk velocity is lost at scatter. */
    if(reg.compute_tasks.root_sim != reg.affected_tasks.root_sim) {
        double combuf[6];
        if(ThisTask == reg.compute_tasks.root_sim) {
            for(int j = 0; j < 3; j++) { combuf[j] = reg.com_pos[j]; combuf[j+3] = reg.com_vel[j]; }
            MPI_Send(combuf, 6, MPI_DOUBLE, reg.affected_tasks.root_sim, 997, MPI_COMM_WORLD);
        }
        if(ThisTask == reg.affected_tasks.root_sim) {
            MPI_Recv(combuf, 6, MPI_DOUBLE, reg.compute_tasks.root_sim, 997, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(int j = 0; j < 3; j++) { reg.com_pos[j] = combuf[j]; reg.com_vel[j] = combuf[j+3]; }
        }
    }

    if(reg.affected_tasks.is_member()) {
        MPI_Bcast(reg.com_pos, 3, MPI_DOUBLE, reg.affected_tasks.root, reg.affected_tasks.comm);
        MPI_Bcast(reg.com_vel, 3, MPI_DOUBLE, reg.affected_tasks.root, reg.affected_tasks.comm);
    }
}

/* ============================================================
 *  PHASE 4: External-force half-step kick and integration
 * ============================================================ */

/* TNT-pattern internal-force negative half-kick (mirrors gadget_tnt's
 * negative_half_step_kicks_region). Subtracts the member-member softened pairwise
 * force from the chain members' velocities, removing the *internal* part of the
 * host's full gravity kick — the host (kicks.cc) now kicks KETJU members with the
 * full tree force exactly like every other particle (so the external force is
 * carried by the same, energy-consistent KDK as the cluster), and MSTAR supplies
 * the exact internal force. Operates on the current ps->pos, so it must be called
 * with start-of-step positions before MSTAR (matching the first host kick) and with
 * the MSTAR-evolved positions after (matching the second host kick at the drifted
 * positions). */
static void do_negative_halfstep_kick(KetjuRegion &reg, double kick_factor)
{
    if(!reg.compute_tasks.is_member() || !reg.integrator) return;

    int n = reg.integrator->num_particles;
    struct ketju_system_physical_state *ps = reg.integrator->physical_state;

    /* softening: use the star softening (comoving -> physical), matching the tree; natural (pc) for MSTAR state */
    double h = All.ForceSoftening[4] * All.cf_atime * KJ_LEN_TO_NAT;
#ifdef SINK_PARTICLES
    if(All.ForceSoftening[5] > 0) h = DMIN(h, All.ForceSoftening[5] * All.cf_atime);
#endif

    /* distribute N² loop across compute tasks */
    int loop_start = loop_scheduling_block_edge(n, reg.compute_tasks.size, reg.compute_tasks.rank);
    int loop_end   = loop_scheduling_block_edge(n, reg.compute_tasks.size, reg.compute_tasks.rank + 1);

    std::vector<double> dv(3 * n, 0.0);
    for(int i = loop_start; i < loop_end; i++) {
        for(int j = i + 1; j < n; j++) {
            double dr[3], r2 = 0;
            for(int k = 0; k < 3; k++) { dr[k] = ps->pos[i][k] - ps->pos[j][k]; r2 += dr[k] * dr[k]; }
            double r = sqrt(r2);
            if(r == 0) continue;
            /* +G m_j softened(r) dr_ij = -internal_accel_i: undoes the internal part of the host kick */
            double fac = kick_factor * reg.integrator->constants->G * softened_force_factor(r, h); /* natural units */
            for(int k = 0; k < 3; k++) {
                dv[3*i + k] += ps->mass[j] * fac * dr[k];
                dv[3*j + k] -= ps->mass[i] * fac * dr[k];
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, dv.data(), 3 * n, MPI_DOUBLE, MPI_SUM, reg.compute_tasks.comm);
    for(int i = 0; i < n; i++)
        for(int k = 0; k < 3; k++) ps->vel[i][k] += dv[3*i + k];
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
            double GM = reg.integrator->constants->G * (ps->mass[i] + ps->mass[j]); /* natural units */
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
            double GM = reg.integrator->constants->G * (ps->mass[i] + ps->mass[j]); /* natural units */
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

#ifdef KETJU_ENERGY_TRACE
/* Trace the 2-body internal energy of an n==2 region at a given stage (single-rank debug). */
static void ketju_trace_energy(KetjuRegion &reg, const char *label)
{
    if(!reg.compute_tasks.is_root() || !reg.integrator) return;
    if(reg.integrator->num_particles != 2) return;
    struct ketju_system_physical_state *ps = reg.integrator->physical_state;
    double dr2 = 0, dv2 = 0, vcom2 = 0;
    double mtot = ps->mass[0] + ps->mass[1];
    for(int k = 0; k < 3; k++) {
        double dx = ps->pos[0][k] - ps->pos[1][k];
        double dv = ps->vel[0][k] - ps->vel[1][k];
        double vc = (ps->mass[0]*ps->vel[0][k] + ps->mass[1]*ps->vel[1][k]) / mtot;
        dr2 += dx*dx; dv2 += dv*dv; vcom2 += vc*vc;
    }
    double mu = ps->mass[0]*ps->mass[1]/mtot;
    double Eint = 0.5*mu*dv2 - reg.integrator->constants->G*ps->mass[0]*ps->mass[1]/sqrt(dr2); /* natural units */
    printf("KETRACE %-10s E_int=%.14g  r=%.8g  v_com=%.8g\n", label, Eint, sqrt(dr2), sqrt(vcom2));
    fflush(stdout);
}
#endif

/* Run the full integration step for one region.
 * Uses proper cosmological kick factors for comoving integration
 * (based on public GADGET4-KETJU, Mannerkoski+ 2023). */
static void integrate_region(KetjuRegion &reg, double dt_physical)
{
    if(dt_physical <= 0) return;

    /* enter MSTAR natural units: state in place, dt scaled (see KJ_* layer above).
     * scatter_results converts the state back to host units. */
    if(reg.compute_tasks.is_member() && reg.integrator && !reg.state_in_natural_units) {
        kj_state_change_units(reg.integrator->physical_state, reg.integrator->num_particles, 1);
        reg.state_in_natural_units = 1;
    }
    dt_physical *= KJ_TIME_TO_NAT;
#ifdef KETJU_UNITS_DEBUG
    if(reg.compute_tasks.is_root() && reg.integrator && reg.integrator->num_particles >= 2) {
        struct ketju_system_physical_state *psd = reg.integrator->physical_state;
        double dr0 = psd->pos[0][0]-psd->pos[1][0], dr1 = psd->pos[0][1]-psd->pos[1][1], dr2v = psd->pos[0][2]-psd->pos[1][2];
        printf("KJUNITS: G=%g c=%g m0=%g sep=%g dt=%g tol=%g maxstep=%d\n",
               reg.integrator->constants->G, reg.integrator->constants->c, psd->mass[0],
               sqrt(dr0*dr0+dr1*dr1+dr2v*dr2v), dt_physical,
               reg.integrator->options->gbs_relative_tolerance, reg.integrator->options->max_step_count);
        fflush(stdout);
    }
#endif

    /* Pure-MSTAR coupling: the host applies NO gravity kick to chain members (see kicks.cc),
     * so there is no internal tree force to remove here — MSTAR integrates the members in full
     * over the step. (The old negative-half-kick subtraction injected energy at pericenters.) */
#ifdef KETJU_ENERGY_TRACE
    ketju_trace_energy(reg, "mstar_in");
#endif

    /* expand tight binaries before integration to prevent stalling.
     * Runs on compute root only — broadcast updated velocities to all compute tasks. */
    expand_tight_binaries(reg, dt_physical);
    if(reg.compute_tasks.is_member() && reg.integrator && reg.compute_tasks.size > 1) {
        int n = reg.integrator->num_particles;
        MPI_Bcast(reg.integrator->physical_state->vel, 3 * n, MPI_DOUBLE,
                  reg.compute_tasks.root, reg.compute_tasks.comm);
    }

    /* run MSTAR integrator (all compute tasks participate via compute_tasks.comm) */
    if(reg.compute_tasks.is_member() && reg.integrator) {
        ketju_run_integrator(reg.integrator, dt_physical);

        int n_steps = reg.integrator->perf->successful_steps + reg.integrator->perf->failed_steps;
        double dE = reg.integrator->perf->relative_energy_error;

        /* Always-on per-subsystem dump of the MSTAR-reported relative energy error.
         * One greppable line per region per integration call (region root task only).
         * This is the cluster-valid per-subsystem metric — logged for EVERY subsystem,
         * not just the >1e-4 warning path below, so the full distribution can be plotted. */
        if(reg.compute_tasks.is_root()) {
            printf("KETJU_SUBSYS dE/E=%.6e npart=%d nsteps=%d\n", dE, reg.total_particle_count, n_steps);
            fflush(stdout);
        }

#ifdef KETJU_VERBOSE_STEPS
        if(reg.compute_tasks.is_root())
        printf("KETJU [task %d]: Integrated %d particles for dt=%g, %d steps (%d failed), dE/E=%g\n",
               ThisTask, reg.total_particle_count, dt_physical,
               reg.integrator->perf->successful_steps,
               reg.integrator->perf->failed_steps, dE);
#endif

        if(All.KetjuMaxStepCount > 0 && n_steps >= All.KetjuMaxStepCount) {
            printf("KETJU WARNING: integration hit max step count (%d) — results may be inaccurate!\n",
                   All.KetjuMaxStepCount);
        }
        if(fabs(dE) > 1e-4) {
            printf("KETJU WARNING: large energy error dE/E=%g in region with %d particles\n",
                   dE, reg.total_particle_count);
        }
    }

#ifdef KETJU_ENERGY_TRACE
    ketju_trace_energy(reg, "after_mstar"); /* MSTAR output: should equal after_neg1 (MSTAR conserves) */
#endif
    /* Capture MSTAR's end-of-step velocity (pure-MSTAR: this is already the true t_sync
     * velocity, nothing to undo) for the energy diagnostic via P.KetjuTrueVel. */
    if(reg.compute_tasks.is_root() && reg.integrator) {
        int n = reg.integrator->num_particles;
        struct ketju_system_physical_state *ps = reg.integrator->physical_state;
        reg.mstar_end_vel.assign(3 * n, 0.0);
        for(int i = 0; i < n; i++)
            for(int k = 0; k < 3; k++) reg.mstar_end_vel[3*i + k] = ps->vel[i][k];
    }
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

/* Merger action descriptor — determined on affected root, broadcast to all affected tasks */
struct ketju_merger_action {
    MyIDType id_merged, id_survivor;
    int type_merged, type_survivor;
    int task_merged, task_survivor;
    int idx_merged, idx_survivor;
    double M_merged, M_survivor_old, M_survivor_new;
};

#if defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH)
static void handle_mergers(KetjuRegion &reg,
                           const std::vector<double> &final_abs_pos,
                           const std::vector<double> &final_mass,
                           const std::vector<double> &original_mass)
{
    if(!reg.affected_tasks.is_member()) return;

    /* ---- Phase 1: affected root determines merger pairs ---- */
    int n_mergers = 0;
    std::vector<ketju_merger_action> actions;

    if(reg.affected_tasks.is_root()) {
        int n = reg.total_particle_count;
        for(int i = 0; i < n; i++) {
            if(final_mass[i] > 0) continue;

            int type_merged = reg.extra_data[i].Type;

            /* find nearest surviving particle */
            int survivor = -1;
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
            if(survivor < 0) continue;

            int type_survivor = reg.extra_data[survivor].Type;

            int merge_allowed = 0;
#ifdef KETJU_MERGE_BH
            if(type_merged == 5 || type_survivor == 5) merge_allowed = 1;
#endif
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

    /* ---- Phase 3: each task executes its local part ---- */
    int n_metals = 0;
#ifdef METALS
    n_metals = NUM_METAL_SPECIES;  /* under GALSF_RESOLVEDISM_METALS_INDIVIDUAL this covers all 27 elements */
#endif
    /* buffer layout: [0..n_metals) = Metallicity[]; last slot = victim's sampled IMF mass
     * (Stella star-star merger: the survivor's stellar mass must grow so lifetimes/FB/luminosity
     * stay consistent with the dynamical mass MSTAR merged) */
    int buf_size = n_metals + 1;

    for(int m = 0; m < n_mergers; m++) {
        ketju_merger_action &act = actions[m];

        if(buf_size > 0) {
            std::vector<double> merged_buf(buf_size, 0.0);

            if(act.task_merged == ThisTask) {
#ifdef METALS
                for(int k = 0; k < NUM_METAL_SPECIES; k++)
                    merged_buf[k] = P[act.idx_merged].Metallicity[k];
#endif
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
                merged_buf[n_metals] = P[act.idx_merged].MstarSampleIMF[0];
#endif
            }

            /* point-to-point transfer of metals + IMF mass */
            if(act.task_merged == act.task_survivor) {
                /* same task — no MPI needed */
            } else {
                if(act.task_merged == ThisTask)
                    MPI_Send(merged_buf.data(), buf_size, MPI_DOUBLE,
                             act.task_survivor, 997, MPI_COMM_WORLD);
                if(act.task_survivor == ThisTask)
                    MPI_Recv(merged_buf.data(), buf_size, MPI_DOUBLE,
                             act.task_merged, 997, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            if(act.task_survivor == ThisTask) {
#ifdef METALS
                /* mass-weighted Metallicity[] merge (FIRE-pattern layout: slot 0 = total Z) */
                for(int k = 0; k < NUM_METAL_SPECIES; k++)
                    P[act.idx_survivor].Metallicity[k] =
                        (act.M_survivor_old * P[act.idx_survivor].Metallicity[k] +
                         act.M_merged * merged_buf[k]) / act.M_survivor_new;
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
                /* force Sigma Met[1..27] = 1 bit-exact via H from conservation */
                {
                    double X_H_new = 1.0 - P[act.idx_survivor].Metallicity[0] - P[act.idx_survivor].Metallicity[MET_OF(ELEM_He)];
                    if(X_H_new < 0) X_H_new = 0;
                    P[act.idx_survivor].Metallicity[MET_OF(ELEM_H)] = (MyFloat)X_H_new;
                }
#endif
#endif
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
                /* stellar collision: the survivor IS the merged star — grow its sampled stellar
                 * mass so table lookups (lifetime, luminosity, Mej, remnant) see the merger product.
                 * (No rejuvenation model: survivor keeps its age & BirthMetallicity.) */
                if(P[act.idx_survivor].Type == 4 && merged_buf[n_metals] > 0)
                    P[act.idx_survivor].MstarSampleIMF[0] += merged_buf[n_metals];
#endif
            }
        }

        /* mark merged particle for removal — and make the ghost INERT for Stella:
         * a Mass=0 particle with live MstarSampleIMF/luminosities would still trigger
         * FB (SN/winds) and shine UV. Same retirement pattern as the FB death cleanup. */
        if(act.task_merged == ThisTask) {
            P[act.idx_merged].Mass = 0;
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
            P[act.idx_merged].MstarSampleIMF[0] = 0;
#endif
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
            P[act.idx_merged].UV_luminosity = 0;
            P[act.idx_merged].LW_luminosity = 0;
#ifdef GALSF_RESOLVEDISM_NUV_VARIABLE
            P[act.idx_merged].NUV_luminosity = 0;
#endif
#ifdef GALSF_RESOLVEDISM_OPT_VARIABLE
            P[act.idx_merged].OPT_luminosity = 0;
#endif
#endif
        }
    }
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
    int Task;
    int Index;
    int Type;
    double Mass;
    double Pos[3];       /* absolute comoving position */
    double Vel[3];       /* CoM-relative peculiar velocity (carries the -neg2 half-kick, for the host-KDK handoff) */
    double TrueVel[3];   /* CoM-relative true MSTAR end-of-step velocity (pre-neg2), for energy diagnostics */
    double ExtAccel[3];  /* external acceleration (GravAccel - member-member) for the host KDK */
#if defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH)
    double OriginalMass;
#endif
    short int is_dead_remnant;
#ifdef SINK_PARTICLES
    short int SinkSubType;
    double Spin[3];      /* only meaningful for PN particles */
    int is_pn;           /* 1 if this is a PN particle (index < n_pn) */
#endif
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
        /* leave MSTAR natural units: back-convert the state to host units before ANY
         * host-side use (positions/velocities/masses, incl. post-merger masses).
         * The captured true end-of-step velocities are natural too. */
        if(reg.state_in_natural_units) {
            kj_state_change_units(ps, reg.integrator->num_particles, 0);
            for(size_t kv = 0; kv < reg.mstar_end_vel.size(); kv++) { reg.mstar_end_vel[kv] /= KJ_VEL_TO_NAT; }
            reg.state_in_natural_units = 0;
        }
        scatter_buf.resize(n);

        for(int i = 0; i < n; i++) {
            ketju_scatter_particle &sp = scatter_buf[i];
            sp.ID    = reg.extra_data[i].ID;
            sp.Task  = reg.extra_data[i].Task;
            sp.Index = reg.extra_data[i].Index;
            sp.Type  = reg.extra_data[i].Type;
            sp.Mass  = ps->mass[i];
            sp.is_dead_remnant = reg.all_particles[i].is_dead_remnant;
#if defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH)
            sp.OriginalMass = reg.all_particles[i].Mass;
#endif
            for(int j = 0; j < 3; j++) {
                /* physical -> comoving: use a_end since positions are at end of step */
                sp.Pos[j] = reg.com_pos[j] + ps->pos[i][j] / a_end;
                /* physical velocity -> canonical momentum: remove Hubble flow, multiply by a_end.
                 * sp.Vel stores CoM-relative canonical momentum (com_vel added later in velocity trick).
                 * In non-cosmo: a_end=1, hubble_end=0, so sp.Vel = ps->vel (unchanged). */
                sp.Vel[j] = (ps->vel[i][j] - hubble_end * ps->pos[i][j]) * a_end;
                /* true MSTAR end-of-step velocity (captured pre-neg2), same back-conversion */
                double vtrue = (i*3+j < (int)reg.mstar_end_vel.size()) ? reg.mstar_end_vel[3*i + j] : ps->vel[i][j];
                sp.TrueVel[j] = (vtrue - hubble_end * ps->pos[i][j]) * a_end;
                /* external accel for the host kick (already in code accel units, no a_end conversion) */
                sp.ExtAccel[j] = (3*i+j < (int)reg.external_accel.size()) ? reg.external_accel[3*i + j] : 0.0;
            }
#ifdef SINK_PARTICLES
            sp.SinkSubType = reg.extra_data[i].SinkSubType;
            sp.is_pn = (i < reg.num_pn_particles) ? 1 : 0;
            for(int j = 0; j < 3; j++)
                sp.Spin[j] = (i < reg.num_pn_particles) ? ps->spin[i][j] : 0;
#endif
        }

        /* log merger info from integrator */
        if(reg.integrator->num_mergers > 0) {
            for(int m = 0; m < reg.integrator->num_mergers; m++) {
                struct ketju_bh_merger_info *info = &reg.integrator->merger_infos[m];
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
#if defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH)
    std::vector<double> original_mass;
#endif

    /* affected root keeps the sorted scatter_buf for collision/merger checks */
    if(reg.affected_tasks.is_root()) {
        final_abs_pos.resize(3 * n);
        final_rel_vel.resize(3 * n);
        final_mass.resize(n);
#if defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH)
        original_mass.resize(n);
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
#if defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH)
            original_mass[i] = scatter_buf[i].OriginalMass;
#endif
            reg.extra_data[i].ID    = scatter_buf[i].ID;
            reg.extra_data[i].Task  = scatter_buf[i].Task;
            reg.extra_data[i].Index = scatter_buf[i].Index;
            reg.extra_data[i].Type  = scatter_buf[i].Type;
            reg.all_particles[i].is_dead_remnant = scatter_buf[i].is_dead_remnant;
            reg.all_particles[i].Mass = scatter_buf[i].Mass;
        }
    }

    for(int i = 0; i < my_count; i++) {
        ketju_scatter_particle &sp = my_particles[i];
        int idx = sp.Index;

        /* sanity check */
        if(P[idx].ID != sp.ID) {
            printf("KETJU ERROR: particle index mismatch! ID %llu vs %llu on task %d\n",
                   (unsigned long long)P[idx].ID, (unsigned long long)sp.ID, ThisTask);
            endrun(667);
        }

        /* update mass */
        P[idx].Mass = sp.Mass;

        /* skip velocity trick for merged particles (mass=0, will be removed) */
        if(sp.Mass <= 0) {
            P[idx].KetjuIntegrated = 1;
            continue;
        }

        /* compute per-particle drift factor: what drift_particle() will use */
        double dt_drift = get_drift_factor(P[idx].Ti_current, P[idx].Ti_current + reg.ti_step, idx, 0);

        if(dt_drift > 0) {
            P[idx].KetjuTrickUntil = P[idx].Ti_current + reg.ti_step;  /* velocity-trick chord completes here; no final-velocity swap before */
            for(int j = 0; j < 3; j++) {
                /* desired final position = final_abs_pos + CoM displacement during drift */
                double desired_pos = sp.Pos[j] + reg.com_vel[j] * dt_drift;
                double delta_pos = desired_pos - P[idx].Pos[j];
#ifdef BOX_PERIODIC
                double box = (j == 0) ? boxSize_X : (j == 1) ? boxSize_Y : boxSize_Z;
                if(box > 0) { while(delta_pos > 0.5 * box) delta_pos -= box; while(delta_pos < -0.5 * box) delta_pos += box; }
#endif
                P[idx].Vel[j] = delta_pos / dt_drift;
                P[idx].KetjuFinalVel[j] = sp.Vel[j] + reg.com_vel[j];
                P[idx].KetjuTrueVel[j]  = sp.TrueVel[j] + reg.com_vel[j];  /* true t_sync velocity for diagnostics */
                P[idx].KetjuTruePos[j]  = desired_pos;                     /* true t_sync position (MSTAR end + CoM drift), synced with KetjuTrueVel */
                P[idx].KetjuExtAccel[j] = sp.ExtAccel[j];                  /* external field for next host KDK */
            }
        } else {
            /* no drift — set position and velocity directly */
            P[idx].KetjuTrickUntil = P[idx].Ti_current;  /* no chord — swap allowed immediately (no-op: Vel already final) */
            for(int j = 0; j < 3; j++) {
                P[idx].Pos[j] = sp.Pos[j];
                P[idx].Vel[j] = sp.Vel[j] + reg.com_vel[j];
                P[idx].KetjuFinalVel[j] = P[idx].Vel[j];
                P[idx].KetjuTrueVel[j]  = sp.TrueVel[j] + reg.com_vel[j];
                P[idx].KetjuTruePos[j]  = sp.Pos[j];
                P[idx].KetjuExtAccel[j] = sp.ExtAccel[j];
            }
        }

#ifdef SINK_PARTICLES
        /* update spin for PN particles */
        if(sp.is_pn && P[idx].Type == 5) {
            for(int j = 0; j < 3; j++) P[idx].KetjuSpin[j] = sp.Spin[j];
        }
#endif
        P[idx].KetjuIntegrated = 1;
        P[idx].KetjuFreshScatter = 1;  /* MSTAR results written THIS step -> velocity swap after drift is valid */
#ifdef KETJU_HANDOFF_TRACE
        printf("HTRACE-SC t=%.10f id=%llu trickvel=%.8g,%.8g,%.8g finalvel=%.8g,%.8g,%.8g (Ti=%lld TrickUntil=%lld regstep=%lld)\n",
               All.Time, (unsigned long long)P[idx].ID, P[idx].Vel[0], P[idx].Vel[1], P[idx].Vel[2],
               P[idx].KetjuFinalVel[0], P[idx].KetjuFinalVel[1], P[idx].KetjuFinalVel[2],
               (long long)P[idx].Ti_current, (long long)P[idx].KetjuTrickUntil, (long long)reg.ti_step); fflush(stdout);
#endif
    }

    /* ---- Phase 5: stellar collision check (Stella: Type 4, radii from the stellar tables) ---- */
#ifdef KETJU_MERGE_STARS
#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
    {
        /* Each task computes radii for its local Type 4 living stars, then Gatherv
         * to affected root where the N^2 collision check runs. Results (mass updates
         * from collisions) flow through handle_mergers which updates final_mass. */
        std::vector<double> my_radii(my_count, 0.0);
        for(int i = 0; i < my_count; i++) {
            if(my_particles[i].Type != 4 || my_particles[i].is_dead_remnant) continue;
            int idx = my_particles[i].Index;
            double logM = log10(DMAX(P[idx].MstarSampleIMF[0], 0.08));
            double logZ = log10(DMAX(P[idx].BirthMetallicity, 1e-10));
            double age_yr = evaluate_stellar_age_Gyr(idx) * 1e9;
            double table_age = age_yr - stellar_t_PMS(logM, logZ);
            double log_age = log10(DMAX(table_age, 100.0));
            my_radii[i] = pow(10., stellar_log_R_cm(logM, logZ, log_age));
        }

        /* Gatherv radii to affected root (same ordering as Scatterv) */
        std::vector<double> star_radius_cm;
        std::vector<int> recv_counts(n_aff, 0), recv_displs(n_aff, 0);
        MPI_Gather(&my_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                   reg.affected_tasks.root, reg.affected_tasks.comm);
        if(reg.affected_tasks.is_root()) {
            for(int t = 1; t < n_aff; t++) recv_displs[t] = recv_displs[t-1] + recv_counts[t-1];
            star_radius_cm.resize(n, 0.0);
        }
        MPI_Gatherv(my_radii.data(), my_count, MPI_DOUBLE,
                     star_radius_cm.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                     reg.affected_tasks.root, reg.affected_tasks.comm);

        /* collision check on affected root (has full position/mass/radius arrays) */
        if(reg.affected_tasks.is_root()) {
            for(int i = 0; i < n; i++) {
                if(final_mass[i] <= 0 || reg.extra_data[i].Type != 4) continue;
                if(star_radius_cm[i] <= 0) continue;
                for(int j = i + 1; j < n; j++) {
                    if(final_mass[j] <= 0 || reg.extra_data[j].Type != 4) continue;
                    if(star_radius_cm[j] <= 0) continue;
                    double d2 = 0;
                    for(int k = 0; k < 3; k++) {
                        double dx = final_abs_pos[3*i + k] - final_abs_pos[3*j + k];
                        d2 += dx * dx;
                    }
                    double sep_cm = sqrt(d2) * UNIT_LENGTH_IN_CGS * All.cf_atime;
                    if(sep_cm < star_radius_cm[i] + star_radius_cm[j]) {
                        int victim = (final_mass[i] < final_mass[j]) ? i : j;
                        int surv = (victim == i) ? j : i;
                        final_mass[surv] += final_mass[victim];
                        final_mass[victim] = 0;
                        double Mi = original_mass[i] * UNIT_MASS_IN_SOLAR;
                        double Mj = original_mass[j] * UNIT_MASS_IN_SOLAR;
                        printf("KETJU STELLAR COLLISION: ID %llu (%.1f Msun) + ID %llu (%.1f Msun) at sep=%.2e cm (R1+R2=%.2e cm)\n",
                            (unsigned long long)reg.extra_data[i].ID, Mi,
                            (unsigned long long)reg.extra_data[j].ID, Mj,
                            sep_cm, star_radius_cm[i] + star_radius_cm[j]);
                    }
                }
            }
        }
    }
#endif
#endif

    /* handle mergers (integrator mergers + stellar collisions) — merger decisions
     * are made on affected root then broadcast to all affected tasks for execution */
#if defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH)
    handle_mergers(reg, final_abs_pos, final_mass, original_mass);
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

/* PHASE 6: Limit & synchronize chain-particle timesteps (gadget_tnt recipe).
 *
 * This ports gadget_tnt's get_region_max_timestep + set_limited_timesteps. The
 * regularized interior is integrated exactly by MSTAR, but the host couples to
 * the EXTERNAL (cluster) field only once per host step, so the operator-split
 * coupling error grows with the host step AND with desynchronization between
 * regions. We therefore bound each region's step by the region CoM acceleration
 * and the CoM displacement through the region radius (NOT the tidal tensor, and
 * NOT "promote to the longest member bin", which let dense-core binaries take
 * oversized, desynced steps and leak energy), then put all of the region's
 * active members on that single shared bin.
 *
 *   acc_dt = sqrt(0.2 * ErrTolIntAccuracy * R_region / |com_acc|)
 *   dt_fast = 0.1 * R_region / |com_vel|              (kappa = 0.1)
 *   region_dt = min(acc_dt, dt_fast, MaxSizeTimestep)
 *
 * com_acc = |Sum_i m_i a_i| / M and com_vel = |Sum_i m_i v_i| / M over the
 * region members (the member-member internal force cancels in the sum, so the
 * full GravAccel gives the external CoM acceleration).
 *
 * All particles in a region must be on the same timebin for the velocity trick
 * to work (they all drift by the same dt after KETJU sets their velocities). */
/* Set KetjuRegionTag on chain members WITHOUT touching timebins or caches. Needed before the
 * pre-loop gravity (run.cc, before the main loop): that first gravity_tree runs before
 * ketju_limit_timesteps has tagged anyone, so without this the very first MSTAR integration would
 * gather an UN-excluded GravAccel (full member-member force) as its "external" field and inject the
 * whole internal binding energy once (a one-time constant energy offset). In HERMITE builds the
 * post-tag gravity_tree at run.cc:155 hides this; with HERMITE off we tag here instead. Uses the
 * same detection as ketju_limit_timesteps/ketju_find_regions, so the tags agree with them. */
void ketju_tag_regions(void)
{
    for(int i = 0; i < NumPart; i++) {P[i].KetjuRegionTag = 0;}
    if(All.KetjuRegionRadius <= 0) return;
    if(All.KetjuMinBHMass <= 0 && All.KetjuMinStarMass <= 0) return;
    std::vector<ketju_mpi_particle> centers = gather_chain_centers();
    if(centers.empty()) return;
    std::vector<std::vector<ketju_mpi_particle>> region_centers = build_orbit_regions(centers);
    for(size_t r = 0; r < region_centers.size(); r++) {
        if(region_centers[r].size() < 2) continue;  /* single stars are not chains */
        std::set<int> local_members = local_members_from_group(region_centers[r]);
        for(int idx : local_members) {P[idx].KetjuRegionTag = (int)(r + 1);}
    }
}

void ketju_limit_timesteps(void)
{
    CachedChainCentersValid = 0;

    if(All.KetjuRegionRadius <= 0) return;
    if(All.KetjuMinBHMass <= 0 && All.KetjuMinStarMass <= 0) return;

    /* clear region tags AND per-member region timesteps before (re)assigning — a non-member must have
     * tag 0 (tree excludes nothing) and KetjuRegionTiStep 0 (get_timestep uses the normal criteria).
     * Done before the empty-centers early-out so nothing stale lingers. */
    for(int i = 0; i < NumPart; i++) {P[i].KetjuRegionTag = 0; P[i].KetjuRegionTiStep = 0;}

    /* gather chain center positions (cache for ketju_find_regions) */
    std::vector<ketju_mpi_particle> centers = gather_chain_centers();
    CachedChainCenters = centers;
    CachedChainCentersValid = 1;
    if(centers.empty()) return;

    /* group stars into subsystems by the BIFROST pair-orbit predicate (weak-field capture/release);
     * merge_radius retained only as the length scale for the capture-buffer pass below */
    double merge_radius = 2.0 * All.KetjuRegionRadius;
    std::vector<std::vector<ketju_mpi_particle>> region_centers = build_orbit_regions(centers);

    /* TNT region-adaptive timestep (ports get_region_max_timestep + set_limited_timesteps): bound each
     * region's host step by its EXTERNAL field and CoM motion, then put ALL its members on that single
     * shared, synchronized bin. Because member<->member forces are excluded from the tree, each member's
     * GravAccel IS its external acceleration, so the mass-weighted sum gives the region CoM external
     * acceleration directly. This replaces the old "promote everyone to the longest member bin", which
     * let dense-cluster regions take oversized, desynced steps and leak the external coupling energy.
     *   acc_dt  = sqrt(0.2 * eta * R_region / |a_com|)   (limit CoM displacement relative to region size)
     *   dt_fast = 0.1 * R_region / |v_com|                (CoM may not cross the region in one step)
     *   region_dt = min(acc_dt, dt_fast, MaxSizeTimestep)
     * Moving members to a SHORTER bin at a sync point is always safe (short bins subdivide long ones),
     * unlike the old promotion to a longer bin. (Comoving factors assume a=1, i.e. STARFORGE's non-cosmo
     * case; revisit for cosmological runs.) */
    double eta = All.ErrTolIntAccuracy;
    double R_region = All.KetjuRegionRadius;
    int n_moved_local = 0;

    /* PASS 1: compute each region's own external-field-limited power-of-two step and TAG members.
     * We also track the GLOBAL MINIMUM step over all regions. */
    std::vector<std::set<int>> region_members(region_centers.size());
    integertime global_min_ti = TIMEBASE;
    /* accumulators for the bulk velocity-dispersion criterion (KETJU Eq. 6): mass-weighted mean and
     * mean-square of the region CoM velocities (NOT individual member velocities — those are dominated by
     * the internal orbit, which MSTAR handles; using them would wrongly force the host step down to the
     * internal-orbit scale). sigma from the region CoMs is the bulk/perturber dispersion KETJU intends. */
    double disp_M = 0, disp_Mv[3] = {0,0,0}, disp_Mv2 = 0; int disp_nreg = 0;

    for(size_t r = 0; r < region_centers.size(); r++) {
        std::set<int> local_members = local_members_from_group(region_centers[r]);

        /* mass-weighted local sums: [0]=M, [1..3]=Sum m*a_ext (a_ext=member GravAccel), [4..6]=Sum m*v,
         * [7]=member count */
        double loc[8] = {0,0,0,0,0,0,0,0};
        for(int idx : local_members) {
            double m = P[idx].Mass;
            loc[0] += m; loc[7] += 1.0;
            for(int k = 0; k < 3; k++) { loc[1+k] += m * P[idx].GravAccel[k]; loc[4+k] += m * P[idx].Vel[k]; }
        }
        double glob[8];
        MPI_Allreduce(loc, glob, 8, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double M = glob[0];
        if(M <= 0) continue;
        /* A region with fewer than 2 members is not a chain — MSTAR will skip it (ketju_find_regions
         * skips n<2 regions). It must NOT be tagged, NOT be flagged KetjuIntegrated, and above all NOT
         * vote in the global-min shared timestep: with small KetjuRegionRadius (e.g. episodic-encounter
         * production configs, R ~ few x softening) every star >= KetjuMinStarMass forms its own
         * single-member region, and their R-scaled dt criteria (0.1*R/v etc.) would force the ENTIRE
         * star set onto an absurdly small shared bin (~100x slowdown observed) while marking every star
         * KetjuIntegrated (wrongly bypassing Hermite in the kicks). Skip them entirely: such stars are
         * ordinary Hermite/KDK particles until a real >=2-member chain forms around them. */
        if(glob[7] < 1.5) continue;

        double a_com2 = 0, v_com2 = 0;
        for(int k = 0; k < 3; k++) { double a = glob[1+k]/M, v = glob[4+k]/M; a_com2 += a*a; v_com2 += v*v; }
        double a_com = sqrt(a_com2), v_com = sqrt(v_com2);

        /* accumulate this region's CoM velocity into the bulk-dispersion sums (mass-weighted) */
        disp_M += M; disp_nreg++;
        for(int k = 0; k < 3; k++) { double vc = glob[4+k]/M; disp_Mv[k] += M * vc; }
        disp_Mv2 += M * v_com2;

        double region_dt = All.MaxSizeTimestep;
        if(a_com > 0 && R_region > 0) { double acc_dt  = sqrt(0.2 * eta * R_region / a_com); if(acc_dt  < region_dt) region_dt = acc_dt;  }
        if(v_com > 0 && R_region > 0) { double dt_fast = 0.1 * R_region / v_com;              if(dt_fast < region_dt) region_dt = dt_fast; }

        /* physical dt -> largest power-of-two integer timestep. */
        integertime ti_target = (integertime)(region_dt / All.Timebase_interval);
        integertime ti_pow2 = TIMEBASE;
        while(ti_pow2 > ti_target) { ti_pow2 >>= 1; }
        if(ti_pow2 < 2) ti_pow2 = 2;
        if(ti_pow2 < global_min_ti) global_min_ti = ti_pow2;

        /* Tag members now (before do_first_halfstep_kick) so the gravity_tree excludes member<->member
         * forces. The TIMESTEP is assigned in pass 2 from the global minimum. */
        for(int idx : local_members) {
            P[idx].KetjuIntegrated = 1;
            P[idx].KetjuRegionTag = (int)(r + 1);
        }
        region_members[r] = std::move(local_members);
    }

    /* KETJU Eq. 6 bulk velocity-dispersion criterion: dt <= R_region / (6 sigma), sigma = dispersion of
     * the region CoM velocities (bulk/perturber motion). Applied as a global cap so the shared step also
     * resolves how fast the surrounding perturber configuration reshuffles. We use it IN ADDITION to our
     * own a_com/v_com criteria (which for the dense plummer test are tighter, so this is usually inert);
     * KETJU also bounds by the min GADGET step of nearby NON-member perturbers within a search radius —
     * vacuous here since every particle is a region member, but worth porting for sparse/realistic clusters. */
    if(disp_nreg >= 2 && disp_M > 0 && R_region > 0) {
        /* need >=2 regions for a meaningful CoM-velocity dispersion: with one region the variance is
         * zero up to floating-point cancellation, sigma ~ 1e-16 made dt_sigma overflow the integertime
         * cast to NEGATIVE and the pow2 loop below spun forever (0 > negative). Guard the cancellation
         * floor and cap dt_sigma at MaxSizeTimestep (beyond which it cannot bind) before any cast. */
        double sigma2 = disp_Mv2 / disp_M, v2mean = disp_Mv2 / disp_M;
        for(int k = 0; k < 3; k++) { double mv = disp_Mv[k] / disp_M; sigma2 -= mv * mv; }
        double sigma = (sigma2 > 1e-12 * v2mean) ? sqrt(sigma2) : 0.0;
        if(sigma > 0) {
            double dt_sigma = R_region / (6.0 * sigma);
            if(dt_sigma < All.MaxSizeTimestep) {
                integertime ti_target = (integertime)(dt_sigma / All.Timebase_interval);
                integertime ti_pow2 = TIMEBASE;
                while(ti_pow2 > ti_target && ti_pow2 > 1) { ti_pow2 >>= 1; }
                if(ti_pow2 < 2) ti_pow2 = 2;
                if(ti_pow2 < global_min_ti) global_min_ti = ti_pow2;
            }
        }
    }

    /* PASS 1.5 — CAPTURE BUFFER (ports tnt's timestep_limiting_radius; REQUIRED for episodic/small-R
     * configs). A region capture must only ever happen to particles that are ACTIVE and on
     * synchronized bins at the capture sync: MSTAR's scatter overwrites Pos/Vel of ALL members, so
     * capturing a star that is mid-way through a long Hermite bin shreds its Hermite history and the
     * velocity-trick drift bookkeeping — each such capture injects energy (observed: episodic-KETJU
     * cluster runs blew up to dE ~ 1e5 via per-pericenter captures). Remedy: any pair of chain-eligible
     * stars in DIFFERENT merged groups (same-group pairs are already co-members — MSTAR's job, their
     * host bin is the region step) that approaches within a buffer radius gets its host timestep capped
     * by an approach-time criterion, dt <= 0.1 * d / |v_rel|. Both stars compute the same symmetric cap
     * -> same power-of-two bin -> both active together on the global timeline well before they can
     * cross the capture radius (2R). KetjuRegionTiStep is min-combined in get_timestep, so normal
     * Hermite criteria still apply. Runs BEFORE PASS 2 because the buffer caps must also lower the
     * shared member bin (global_min_ti): the velocity-trick bookkeeping is only consistent when region
     * step == member bin == global sync spacing, i.e. NO KETJU-relevant particle may sit on a bin
     * smaller than the members' (otherwise member drift chords span multiple syncs and the end-of-drift
     * velocity swap fires mid-chord). Caps are computed from globally-gathered centers, so the same
     * minimum is derived identically on every rank. */
    {
        /* tnt uses timestep_search_radius = 100 * region_radius; the buffer must engage EARLY in the
         * infall — a particle's bin can only shrink when it is ACTIVE, so the cap needs to be in place
         * several bin-boundaries before the pair reaches the capture radius (a 4R buffer engaged only
         * ~1e-5 before an e=0.99 pericenter — far too late, first captures still hit mid-bin). */
        const double buffer_radius = 50.0 * merge_radius;  /* = 100 R_region; capture happens at 2R */
        const double buf2 = buffer_radius * buffer_radius;
        int n_buffered_local = 0;
        for(size_t g1 = 0; g1 < region_centers.size(); g1++) {
            for(size_t g2 = g1 + 1; g2 < region_centers.size(); g2++) {
                for(const ketju_mpi_particle &a : region_centers[g1]) {
                    for(const ketju_mpi_particle &b : region_centers[g2]) {
                        double d2 = 0, vr2 = 0;
                        for(int k = 0; k < 3; k++) {
                            double dx = a.Pos[k] - b.Pos[k]; d2 += dx * dx;
                            double dv = a.Vel[k] - b.Vel[k]; vr2 += dv * dv;
                        }
                        if(d2 >= buf2 || vr2 <= 0) continue;
                        double dt_cap = 0.1 * sqrt(d2 / vr2);
                        if(dt_cap >= All.MaxSizeTimestep) continue;  /* cannot bind; also guards the integertime cast */
                        integertime ti_target = (integertime)(dt_cap / All.Timebase_interval);
                        integertime ti_pow2 = TIMEBASE;
                        while(ti_pow2 > ti_target && ti_pow2 > 1) { ti_pow2 >>= 1; }
                        if(ti_pow2 < 2) ti_pow2 = 2;
                        if(ti_pow2 < global_min_ti) global_min_ti = ti_pow2;  /* members must share the smallest KETJU bin */
                        /* apply on the rank that owns each star (min-combine with any existing cap) */
                        if(a.Task == ThisTask && (P[a.Index].KetjuRegionTiStep == 0 || ti_pow2 < P[a.Index].KetjuRegionTiStep))
                            { P[a.Index].KetjuRegionTiStep = ti_pow2; n_buffered_local++; }
                        if(b.Task == ThisTask && (P[b.Index].KetjuRegionTiStep == 0 || ti_pow2 < P[b.Index].KetjuRegionTiStep))
                            { P[b.Index].KetjuRegionTiStep = ti_pow2; n_buffered_local++; }
                    }
                }
            }
        }
        n_moved_local += n_buffered_local;
    }

    /* PASS 2: put ALL regions on ONE shared, synchronized bin = the global minimum over region steps
     * AND buffer caps. This ports tnt's set_limited_timesteps grouping: tnt groups every region within
     * 100*R_region into one "timestep region" that shares a single minimum step, so neighbouring
     * regions never desync. Our previous per-region-independent step let the embedded ECCENTRIC
     * binaries fall onto different bins (198/256 active on the first sync); the inter-region external
     * coupling was then evaluated at inconsistent times and leaked energy catastrophically (dE/|E0| ~
     * 16). All members on one (smallest-KETJU) bin => the external KDK coupling between regions is
     * consistent AND member drift chords never span multiple syncs. */
    for(size_t r = 0; r < region_centers.size(); r++) {
        for(int idx : region_members[r]) {
            n_moved_local++;
            P[idx].KetjuRegionTiStep = global_min_ti;  /* consumed by get_timestep */
        }
    }

    int n_moved_global;
    MPI_Reduce(&n_moved_local, &n_moved_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask == 0 && n_moved_global > 0) {
        printf("KETJU TIMESTEP: set region-adaptive external step for %d chain particle(s)\n",
               n_moved_global);
    }

#ifdef KETJU_HANDOFF_TRACE
    /* exhaustive per-star handoff state at the END of limit_timesteps (tags/flags/caps now set for this
     * step). Only sane for tiny N — meant for the isolated-binary capture/release reproduction. */
    for(int i = 0; i < NumPart; i++) {
        if(P[i].Type != 5) continue;
        printf("HTRACE-LT t=%.10f id=%llu bin=%d act=%d tag=%d integ=%d fresh=%d cap=%lld pos=%.8g,%.8g,%.8g vel=%.8g,%.8g,%.8g\n",
               All.Time, (unsigned long long)P[i].ID, P[i].TimeBin, TimeBinActive[P[i].TimeBin] ? 1 : 0,
               P[i].KetjuRegionTag, P[i].KetjuIntegrated, P[i].KetjuFreshScatter,
               (long long)P[i].KetjuRegionTiStep,
               P[i].Pos[0], P[i].Pos[1], P[i].Pos[2], P[i].Vel[0], P[i].Vel[1], P[i].Vel[2]);
    }
    fflush(stdout);
#endif
}

/* Collect sorted (ID,Task) keys for a region across all tasks (for staleness check).
 * Uses (ID,Task) pairs instead of bare IDs to handle duplicate star IDs correctly. */
static std::vector<CachedRegionKey> gather_sorted_region_keys(const std::set<int> &local_members)
{
    int n_local = local_members.size();
    std::vector<CachedRegionKey> local_keys(n_local);
    int k = 0;
    for(int idx : local_members) { local_keys[k].ID = P[idx].ID; local_keys[k].Task = ThisTask; local_keys[k].Index = idx; k++; }

    std::vector<int> counts(NTask), displs(NTask);
    MPI_Allgather(&n_local, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int n_total = 0;
    for(int t = 0; t < NTask; t++) { displs[t] = n_total; n_total += counts[t]; }

    std::vector<CachedRegionKey> all_keys(n_total);
    std::vector<int> byte_counts(NTask), byte_displs(NTask);
    for(int t = 0; t < NTask; t++) {
        byte_counts[t] = counts[t] * sizeof(CachedRegionKey);
        byte_displs[t] = displs[t] * sizeof(CachedRegionKey);
    }
    MPI_Allgatherv(local_keys.data(), n_local * sizeof(CachedRegionKey), MPI_BYTE,
                   all_keys.data(), byte_counts.data(), byte_displs.data(),
                   MPI_BYTE, MPI_COMM_WORLD);
    std::sort(all_keys.begin(), all_keys.end());
    return all_keys;
}

void ketju_find_regions(void)
{
    /* clear previous state — KetjuIntegrated is cleared here (not in finish_step)
     * so the flag survives across the step boundary for guard checks in kicks/predict/gravtree */
#ifdef HERMITE_INTEGRATION
    /* Mark particles that were KETJU-integrated last step as "Hermite history stale".
     * If they remain in a chain this step they never reach the stale check (in-region
     * guard fires first); if they exit, the next eligible_for_hermite() call falls
     * back to KDK once before re-engaging Hermite with fresh OldPos/Acc/Jerk. */
    for(int i = 0; i < NumPart; i++) {if(P[i].KetjuIntegrated) {P[i].HermiteHistoryStale = 1;}}
#endif
    /* NOTE: KetjuIntegrated is NOT cleared here anymore — the capture admission gate inside
     * find_local_members ("active OR already-member") reads it, and membership decisions here MUST
     * reproduce those made by ketju_limit_timesteps earlier in this step (both use the previous step's
     * flags). It is cleared after the membership loop below; scatter_results then re-sets it for the
     * particles MSTAR actually integrated. */
    /* clear region tags; re-set per region below so the post-drift gravity_tree (run.cc:189/231)
     * also excludes member<->member forces at the MSTAR-evolved positions */
    for(int i = 0; i < NumPart; i++) {P[i].KetjuRegionTag = 0;}
    ActiveRegions.clear();
    AllKetjuParticleIndices.clear();

    /* validate parameters */
    if(All.KetjuRegionRadius <= 0) return;
    if(All.KetjuMinBHMass <= 0 && All.KetjuMinStarMass <= 0) return;

    /* reuse chain centers from ketju_limit_timesteps if available, else gather fresh */
    std::vector<ketju_mpi_particle> centers;
    if(CachedChainCentersValid) {
        centers = std::move(CachedChainCenters);
        CachedChainCentersValid = 0;
    } else {
        centers = gather_chain_centers();
    }
    if(centers.empty()) { for(int i = 0; i < NumPart; i++) {P[i].KetjuIntegrated = 0;} CachedRegions.clear(); return; }

    /* group stars into subsystems by the BIFROST pair-orbit predicate (same grouping as
     * ketju_limit_timesteps computed earlier this step — deterministic from the shared centers) */
    std::vector<std::vector<ketju_mpi_particle>> region_centers = build_orbit_regions(centers);

    /* build regions (first pass: find members, collect IDs, check staleness) */
    int n_regions = region_centers.size();
    int n_cached = CachedRegions.size();
    int any_changed = KetjuRegionsStale || (n_regions != n_cached);

    /* per-region: collect sorted (ID,Task) keys and compare to cache */
    std::vector<std::vector<CachedRegionKey>> new_keys(n_regions);
    std::vector<int> region_changed(n_regions, 1); /* 1 = needs rebuild */

    for(int r = 0; r < n_regions; r++) {
        KetjuRegion reg;
        reg.centers = region_centers[r];
        reg.local_member_indices = local_members_from_group(reg.centers);

        int n_local = reg.local_member_indices.size();
        MPI_Allreduce(&n_local, &reg.total_particle_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        /* tag members so the post-drift gravity_tree excludes member<->member forces (region index+1,
         * globally consistent — same ordering as ketju_limit_timesteps, which shares the merged centers) */
        for(int idx : reg.local_member_indices) {P[idx].KetjuRegionTag = r + 1;}

        /* skip regions with fewer than 2 particles — nothing to integrate */
        if(reg.total_particle_count < 2) {
            ActiveRegions.push_back(std::move(reg)); /* placeholder to keep indexing aligned with cache */
            new_keys[r] = {};
            continue;
        }

        for(int idx : reg.local_member_indices)
            AllKetjuParticleIndices.insert(idx);

        ActiveRegions.push_back(std::move(reg));

        /* collect sorted (ID,Task) keys for staleness comparison */
        new_keys[r] = gather_sorted_region_keys(ActiveRegions[r].local_member_indices);

        /* check if this region matches a cached one */
        if(!any_changed && r < n_cached && new_keys[r] == CachedRegions[r].sorted_particle_keys) {
            region_changed[r] = 0; /* unchanged — can reuse communicators */
        }
    }

    /* memberships are fixed now — clear the previous step's integrated flags (deferred earlier for the
     * admission gate); scatter_results re-sets them for the particles MSTAR actually integrates */
    for(int i = 0; i < NumPart; i++) {P[i].KetjuIntegrated = 0;}

    /* allocate compute tasks across regions based on cost estimates */
    allocate_compute_tasks_for_regions();

    /* second pass: set up integrators — reuse cached communicators where possible */
    int n_reused = 0;
    for(int r = 0; r < n_regions; r++) {
        /* single-member placeholders exist only to keep cache indexing aligned — no integrator, no
         * communicators (with small KetjuRegionRadius every eligible star forms one; per-step MPI comm
         * setup for hundreds of them was a dominant cost in Hermite-backbone configs). Decision is rank-
         * consistent: total_particle_count is Allreduced. */
        if(ActiveRegions[r].total_particle_count < 2) continue;
        if(!region_changed[r]) {
            /* reuse cached communicators — transfer ownership to ActiveRegion */
            ActiveRegions[r].affected_tasks = CachedRegions[r].affected_tasks;
            CachedRegions[r].affected_tasks.comm = MPI_COMM_NULL;
            CachedRegions[r].affected_tasks.group = MPI_GROUP_NULL;
            ActiveRegions[r].compute_tasks = CachedRegions[r].compute_tasks;
            CachedRegions[r].compute_tasks.comm = MPI_COMM_NULL;
            CachedRegions[r].compute_tasks.group = MPI_GROUP_NULL;
            ActiveRegions[r].affected_sim_indices = std::move(CachedRegions[r].affected_sim_indices);
            ActiveRegions[r].particle_counts_on_affected = std::move(CachedRegions[r].particle_counts_on_affected);
            ActiveRegions[r].compute_info = CachedRegions[r].compute_info;
            /* still need to gather particles and set up integrator state, but skip MPI_Comm_create */
            setup_integrator_reuse(ActiveRegions[r]);
            n_reused++;
        } else {
            /* full rebuild */
            setup_integrator(ActiveRegions[r]);
        }
    }

    /* clear old cache (frees any remaining communicators not transferred) */
    CachedRegions.clear();
    KetjuRegionsStale = 0;

#ifdef KETJU_VERBOSE_STEPS
    /* per-step region listing: useful for debugging, but with many regions and small host steps this
     * printing alone can I/O-throttle the run (observed 3x slowdown in the Hermite-backbone config) */
    if(ThisTask == 0 && !ActiveRegions.empty()) {
        printf("KETJU: Found %d region(s) from %d center(s) at t=%g (%d reused from cache)\n",
               n_regions, (int)centers.size(), All.Time, n_reused);
        for(int r = 0; r < n_regions; r++) {
            printf("KETJU:   region %d: %d particles, %d centers%s\n",
                   r, ActiveRegions[r].total_particle_count, (int)ActiveRegions[r].centers.size(),
                   region_changed[r] ? "" : " [cached]");
        }
    }
#endif
}

void ketju_run_integration(void)
{
    if(ActiveRegions.empty()) return;

    /* Outer loop over sequential rounds. Within a round, regions run in
     * parallel on disjoint task ranges; rounds run back-to-back so that
     * expensive regions can use all tasks first before smaller regions. */
    int n_rounds = g_ketju_num_sequential_runs > 0 ? g_ketju_num_sequential_runs : 1;
    for(int seq = 0; seq < n_rounds; seq++) {
    for(size_t r = 0; r < ActiveRegions.size(); r++) {
        KetjuRegion &reg = ActiveRegions[r];
        if(reg.compute_info.compute_sequence_position != seq) continue;
        if(reg.total_particle_count < 2) continue;

        /* CHORD CONSISTENCY GATE: never integrate a region while ANY member's velocity-trick drift
         * chord is still in flight (FreshScatter set, chord end not reached). Gathering such a member
         * hands MSTAR its TRICK velocity (~CoM velocity) as an initial condition — garbage in, garbage
         * out (observed as -44 energy events when an active intruder merged with a mid-chord pair).
         * Deferring one/two syncs is safe: the mid-chord members are inactive, the active ones wait. */
        int local_midchord = 0;
        for(int idx : reg.local_member_indices)
            if(P[idx].KetjuFreshScatter && P[idx].Ti_current < P[idx].KetjuTrickUntil) { local_midchord = 1; break; }
        int any_midchord = 0;
        MPI_Allreduce(&local_midchord, &any_midchord, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if(any_midchord) {
            /* STICKY MEMBERSHIP (2026-07-08, the 256-binary-cluster ionization fix):
             * ketju_find_regions cleared KetjuIntegrated for ALL particles this sync; if we
             * skip this region (chord in flight) WITHOUT restoring the flag, the admission
             * gate drops the still-mid-chord members next sync and they lose their tags —
             * the tree then re-applies the companion force MSTAR already integrated and the
             * Hermite corrector fires on stale Old* (double-kill, trace-proven: E -0.835 ->
             * -1.284 -> +0.187, binary ionized; all 256 in lockstep). LATENT IN STARFORGE
             * TOO (identical code) — never fired there because sink timesteps keep members
             * on the chord bin (no mid-chord syncs). Members must hold ownership until their
             * chord completes; release semantics for predicate-dissolved groups unchanged. */
            for(int idx : reg.local_member_indices) {P[idx].KetjuIntegrated = 1;}
            continue;
        }

        /* Subcycling: use the MAXIMUM active timebin among chain members.
         * This is the normal hydro/gravity step for these particles.
         * MSTAR internally substeps to whatever accuracy it needs (GBS adaptive).
         * Previously we used the MINIMUM, which dragged the whole simulation down. */
        integertime local_max_ti = 0;
        for(int idx : reg.local_member_indices) {
            if(!TimeBinActive[P[idx].TimeBin]) continue;
            integertime ti_i = GET_PARTICLE_INTEGERTIME(idx);
            if(ti_i > local_max_ti) local_max_ti = ti_i;
        }

        /* global maximum across all tasks (all tasks participate since region spans tasks) */
        integertime global_max_ti;
        MPI_Allreduce(&local_max_ti, &global_max_ti, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
#ifdef KETJU_HANDOFF_TRACE
        printf("HTRACE-RI t=%.10f region=%d n=%d ti_step=%lld (members:", All.Time, (int)r, reg.total_particle_count, (long long)global_max_ti);
        for(int idx : reg.local_member_indices) printf(" id=%llu bin=%d act=%d", (unsigned long long)P[idx].ID, P[idx].TimeBin, TimeBinActive[P[idx].TimeBin] ? 1 : 0);
        printf(")%s\n", (global_max_ti <= 0 || global_max_ti > TIMEBASE) ? " SKIPPED-no-active" : ""); fflush(stdout);
#endif
        if(global_max_ti <= 0 || global_max_ti > TIMEBASE) {
            for(int idx : reg.local_member_indices) {P[idx].KetjuIntegrated = 1;} /* sticky membership, see above */
            continue;
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

        if(dt_physical <= 0 || !isfinite(dt_physical)) {
            for(int idx : reg.local_member_indices) {P[idx].KetjuIntegrated = 1;} /* sticky membership, see above */
            continue;
        }

        /* integrate and scatter results */
        integrate_region(reg, dt_physical);
        scatter_results(reg);
    }
    } /* end for(seq) */

    /* update cost estimates for next step's load balancing */
    update_region_costs();

    if(ThisTask == 0 && !ActiveRegions.empty()) {
        int total_mergers = 0;
        for(size_t r = 0; r < ActiveRegions.size(); r++) {
            if(ActiveRegions[r].integrator && ActiveRegions[r].integrator->num_mergers > 0)
                total_mergers += ActiveRegions[r].integrator->num_mergers;
        }
        if(total_mergers > 0)
            printf("KETJU: %d merger(s) this step\n", total_mergers);
    }
}

void ketju_set_final_velocities(void)
{
    /* called after drift: swap in true physical velocities and correct dp[] momentum.
     * The velocity trick set Vel = delta_pos/dt_drift for the drift.
     * Now we replace it with the true MSTAR velocity and correct dp[]. */
    for(int i = 0; i < NumPart; i++) {
        if(P[i].KetjuFreshScatter
           && P[i].Ti_current >= P[i].KetjuTrickUntil) {  /* swap PENDING MSTAR results exactly when the
            * velocity-trick drift CHORD completes — never earlier and never never. Not earlier: this routine
            * runs after every sync, and an intermediate sync (other particles on smaller bins) must not swap
            * mid-chord or the rest of the drift proceeds with the physical velocity and misses MSTAR's end
            * position (a 15 AU hard binary got teleported to 230 AU: +0.92 in one step). Never never:
            * deliberately NOT gated on KetjuIntegrated — if the member is RELEASED while its chord is still
            * in flight, the flag gets cleared but the pending swap must still fire at chord end, else the
            * star keeps the trick velocity (~CoM velocity) as its physical velocity forever, deleting its
            * internal orbital motion (observed: dE -44 events). FreshScatter itself guards staleness: it is
            * set only by scatter_results and consumed exactly once here. */
            for(int j = 0; j < 3; j++) {
                /* dp correction: the difference between true velocity and trick velocity,
                 * converted to momentum. This ensures the kick integrator is consistent. */
                double dv = P[i].KetjuFinalVel[j] - P[i].Vel[j];
                P[i].dp[j] += P[i].Mass * dv;
                P[i].Vel[j] = P[i].KetjuFinalVel[j];
            }
            P[i].KetjuFreshScatter = 0;
#ifdef KETJU_HANDOFF_TRACE
            printf("HTRACE-SW t=%.10f id=%llu swapped-> %.8g,%.8g,%.8g (Ti=%lld TrickUntil=%lld)\n",
                   All.Time, (unsigned long long)P[i].ID, P[i].Vel[0], P[i].Vel[1], P[i].Vel[2],
                   (long long)P[i].Ti_current, (long long)P[i].KetjuTrickUntil); fflush(stdout);
#endif
        }
    }
}

void ketju_finish_step(void)
{
    /* NOTE: KetjuIntegrated flags are NOT cleared here — they persist until
     * ketju_find_regions() at the start of the next step, so that guard checks
     * in kicks.cc/predict.cc/gravtree.cc can see which particles were KETJU-integrated */

    /* cache region communicators for reuse next step */
    CachedRegions.clear();
    for(size_t r = 0; r < ActiveRegions.size(); r++) {
        CachedRegionInfo ci;
        /* Collect sorted (ID,Task,Index) keys for staleness check next step.
         * MUST be globally consistent on every task — using ActiveRegions[r].all_particles
         * is wrong because it is only populated on the affected_root after Gatherv,
         * leaving the cached keys empty on every other task and causing split decisions
         * (some tasks setup_integrator_reuse, others setup_integrator) → MPI deadlock. */
        ci.sorted_particle_keys = gather_sorted_region_keys(ActiveRegions[r].local_member_indices);

        /* transfer communicator ownership — avoid free in KetjuRegion destructor */
        ci.affected_tasks = ActiveRegions[r].affected_tasks;
        ActiveRegions[r].affected_tasks.comm = MPI_COMM_NULL;
        ActiveRegions[r].affected_tasks.group = MPI_GROUP_NULL;
        ci.compute_tasks = ActiveRegions[r].compute_tasks;
        ActiveRegions[r].compute_tasks.comm = MPI_COMM_NULL;
        ActiveRegions[r].compute_tasks.group = MPI_GROUP_NULL;

        ci.affected_sim_indices = std::move(ActiveRegions[r].affected_sim_indices);
        ci.particle_counts_on_affected = std::move(ActiveRegions[r].particle_counts_on_affected);
        ci.compute_info = ActiveRegions[r].compute_info;
        ci.total_particle_count = ActiveRegions[r].total_particle_count;

        CachedRegions.push_back(std::move(ci));
    }

    /* free region data (integrators, particle arrays — but comms are now in cache) */
    ActiveRegions.clear();
    AllKetjuParticleIndices.clear();
}

int ketju_is_particle_in_region(int i)
{
    return (AllKetjuParticleIndices.find(i) != AllKetjuParticleIndices.end()) ? 1 : 0;
}

void ketju_mark_regions_stale(void)
{
    KetjuRegionsStale = 1;
}

/* Free all cached/active KETJU MPI communicators and integrator state. MUST be called
 * before MPI_Finalize: CachedRegions/ActiveRegions are static, so otherwise their
 * destructors run at program exit AFTER MPI_Finalize and segfault inside MPI_Comm_free
 * (this is the rc=139 shutdown crash). Clearing here frees the comms while MPI is alive;
 * the empty static vectors then destruct harmlessly at exit. */
void ketju_finalize(void)
{
    CachedRegions.clear();
    ActiveRegions.clear();
    CachedChainCenters.clear();
    CachedChainCentersValid = 0;
    KetjuRegionsStale = 1;
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

/* One appended row per (active region x output step): the MSTAR-reported
 * per-subsystem relative energy error and step counts. This is the
 * cluster-valid per-subsystem metric, recorded through KETJU's own HDF5
 * apparatus (dataset /regions/data). See ketju_write_output(). */
struct ketju_region_output {
    double time;         /* All.Time at this output step */
    int    npart;        /* reg.total_particle_count */
    double dE_rel;       /* reg.integrator->perf->relative_energy_error (last call) */
    int    nsteps_ok;    /* reg.integrator->perf->successful_steps */
    int    nsteps_fail;  /* reg.integrator->perf->failed_steps */
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
        /* per-subsystem diagnostics group: /regions/data is a flat, extensible
         * compound dataset with one row per (active region x output step),
         * /regions/step_offsets records how many region rows each step appended. */
        grp = H5Gcreate2(ketju_output_file, "regions", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Gclose(grp);
        hid_t region_type = H5Tcreate(H5T_COMPOUND, sizeof(ketju_region_output));
        H5Tinsert(region_type, "time",        HOFFSET(ketju_region_output, time),        H5T_NATIVE_DOUBLE);
        H5Tinsert(region_type, "npart",       HOFFSET(ketju_region_output, npart),       H5T_NATIVE_INT);
        H5Tinsert(region_type, "dE_rel",      HOFFSET(ketju_region_output, dE_rel),      H5T_NATIVE_DOUBLE);
        H5Tinsert(region_type, "nsteps_ok",   HOFFSET(ketju_region_output, nsteps_ok),   H5T_NATIVE_INT);
        H5Tinsert(region_type, "nsteps_fail", HOFFSET(ketju_region_output, nsteps_fail), H5T_NATIVE_INT);
        dims[0] = 0;
        space = H5Screate_simple(1, dims, maxdims);
        dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);
        H5Dcreate2(ketju_output_file, "regions/data", region_type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Pclose(dcpl); H5Sclose(space); H5Tclose(region_type);
        dims[0] = 0;
        space = H5Screate_simple(1, dims, maxdims);
        dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);
        H5Dcreate2(ketju_output_file, "regions/step_offsets", H5T_NATIVE_INT, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Pclose(dcpl); H5Sclose(space);
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

static void ketju_hdf5_append_int(const char *dset_name, int value)
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
    H5Dwrite(ds, H5T_NATIVE_INT, memsp, sp, H5P_DEFAULT, &value);
    H5Sclose(memsp); H5Sclose(sp); H5Dclose(ds);
}

static void ketju_hdf5_append_region(ketju_region_output *rec)
{
    if(ThisTask != 0 || ketju_output_file < 0) return;
    hid_t ds = H5Dopen2(ketju_output_file, "regions/data", H5P_DEFAULT);
    hid_t sp = H5Dget_space(ds);
    hsize_t dims[1]; H5Sget_simple_extent_dims(sp, dims, NULL);
    hsize_t newdims[1] = {dims[0] + 1};
    H5Dset_extent(ds, newdims);
    H5Sclose(sp);
    sp = H5Dget_space(ds);
    hsize_t offset[1] = {dims[0]}, count[1] = {1};
    H5Sselect_hyperslab(sp, H5S_SELECT_SET, offset, NULL, count, NULL);
    hid_t memsp = H5Screate_simple(1, count, NULL);
    hid_t rtype = H5Dget_type(ds);
    H5Dwrite(ds, rtype, memsp, sp, H5P_DEFAULT, rec);
    H5Tclose(rtype); H5Sclose(memsp); H5Sclose(sp); H5Dclose(ds);
}

void ketju_write_output(void)
{
    if(ActiveRegions.empty()) return;

    /* --- Per-subsystem diagnostics (/regions) ---
     * Gather the MSTAR-reported relative energy error (+ step counts) for every
     * active region to task 0. Same dense-array + Allreduce pattern used for the
     * load-balance costs (line ~478): each region's compute root fills its own
     * slot, every other task contributes 0, so MPI_SUM delivers the root's value
     * unchanged (correct even for negative/tiny dE) to task 0 for writing. */
    int n_regions = (int)ActiveRegions.size();
    std::vector<double> loc(4 * n_regions, 0.0), glob(4 * n_regions, 0.0);
    for(int r = 0; r < n_regions; r++) {
        KetjuRegion &reg = ActiveRegions[r];
        if(!reg.compute_tasks.is_root() || !reg.integrator || reg.centers.empty()) continue;
        loc[4*r + 0] = (double)reg.total_particle_count;
        loc[4*r + 1] = reg.integrator->perf->relative_energy_error;
        loc[4*r + 2] = (double)reg.integrator->perf->successful_steps;
        loc[4*r + 3] = (double)reg.integrator->perf->failed_steps;
    }
    MPI_Allreduce(loc.data(), glob.data(), 4 * n_regions, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if(ThisTask == 0 && ketju_output_file >= 0) {
        int n_written = 0;
        for(int r = 0; r < n_regions; r++) {
            int npart = (int)(glob[4*r + 0] + 0.5);
            if(npart < 2) continue; /* placeholder / inactive region */
            ketju_region_output rec;
            rec.time        = All.Time;
            rec.npart       = npart;
            rec.dE_rel      = glob[4*r + 1];
            rec.nsteps_ok   = (int)(glob[4*r + 2] + 0.5);
            rec.nsteps_fail = (int)(glob[4*r + 3] + 0.5);
            ketju_hdf5_append_region(&rec);
            n_written++;
        }
        ketju_hdf5_append_int("regions/step_offsets", n_written);
    }

    /* only task 0 has the file for the remaining scalar writes */
    if(ThisTask != 0 && ketju_output_file < 0) return;

    /* write timestep */
    double phys_time = All.Time; /* scale factor in cosmo, time otherwise */
    ketju_hdf5_append_double("timesteps", phys_time);
    ketju_output_tstep_index++;
}

#endif /* KETJU_REGULARIZATION */
