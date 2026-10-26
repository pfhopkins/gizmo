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
#include <utility>
#include <cfloat>
#include <queue>
#include <functional>
#include <array>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "ketju_coupling.h"
#include "../gravity/tree_pair_kernel.h"
#ifndef KETJU_LEGACY_C_LIBRARY
#include "ketju_mstar_kokkos.h" /* the three library calls on the C++/Kokkos MSTAR library (default) */
#define ketju_create_system kj_mstar_create_system
#define ketju_run_integrator kj_mstar_run_integrator
#define ketju_free_system kj_mstar_free_system
#endif
#if defined(GALSF_RESOLVEDISM_METALS_INDIVIDUAL) || defined(KETJU_PN_REMNANT_TAG)
#include "resolvedism_stellar_tables.h" /* ELEM_*, MET_OF, REM_* */
#endif

#ifdef KETJU_REGULARIZATION

#if defined(KETJU_MERGE_STARS) && defined(KETJU_TYPE4_STARS)
#define KJ_STAR_COLLISIONS /* Type-4 stars merge on contact (R1 + R2) inside MSTAR */
#if defined(GALSF_RESOLVEDISM_SAMPLE_IMF) && defined(GALSF_RESOLVEDISM_STELLAR_TABLES)
#include "resolvedism_star_age.h"
#endif
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
#ifdef KETJU_SINKS
        for(int j = 0; j < 3; j++) {ps->spin[i][j] *= fL * fM * fV;}
#endif
    }
    ps->time *= fL / fV;
}

/* PN classification for KETJU integrator (0 = PN/BH, 1 = non-PN/star, -1 = not Type 5).
 * Derived from ProtoStellarStage: stage 7 (relic/BH) → 0, stage <7 (star) → 1.
 * When SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION is off we lack stage info, so
 * fall back to non-PN/star (relativistic terms only matter when KetjuPNTerms != "none"). */
#ifdef SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION
#define KETJU_PN_CLASS(i) ((P[(i)].ProtoStellarStage == 7) ? 0 : 1)
#elif defined(KETJU_TYPE4_STARS) && defined(SINK_PARTICLES)
#define KETJU_PN_CLASS(i) (P[(i)].SinkSubType) /* Stella sinks: 0 = BH, 1 = promoted stellar remnant */
#else
#define KETJU_PN_CLASS(i) (1)
#endif

/* Cost tracking for load balancing (keyed by first center ID per region) */
static std::unordered_map<MyIDType, double> region_previous_cost;

/* ============================================================
 *  Lightweight particle data for MPI communication
 * ============================================================ */
/* Stella IMF siblings share P.ID and differ in ID_child_number: KETJU keys particles by both */
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
#ifdef KETJU_SINKS
    short int KetjuPNClass;
    double Spin[3];
#endif
#ifdef KETJU_PN_REMNANT_TAG
    signed char RemnantType; /* Type-4 compact remnant code (-1 = live star) */
#endif
#ifdef KETJU_TYPE4_STARS
    double SoftGrav[2], ZetaGrav[2]; /* tree softening/zeta of the last gravity call ([0] target, [1] secondary) */
    double SoftNow[2], ZetaNow[2];   /* values the next gravity call will use */
#endif
#ifdef KJ_STAR_COLLISIONS
    double CollRadius; /* contact radius, physical code length (0: never collides) */
    double BirthTime, FormDens; /* StellarAge and physical formation density, for the collision log */
#endif
    double GravAcc[3]; /* comoving GravAccel of the gravity call whose kick started this step */
    double TidalNorm;  /* physical |T| of the field external to its chain at its last step end (0: not known) */
    double Tidal[6];   /* tree tidal tensor of the gravity call that started this step (xx yy zz xy yz xz, comoving, with G) */
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
#ifdef KETJU_SINKS
    short int KetjuPNClass;
#endif
#ifdef KJ_STAR_COLLISIONS
    double BirthTime, FormDens;
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
        /* the cached regions are file-scope statics, destroyed at exit after MPI_Finalize, and
           freeing a handle then aborts the job with a nonzero status */
        int finalized = 0; MPI_Finalized(&finalized);
        if(finalized) { comm = MPI_COMM_NULL; group = MPI_GROUP_NULL; rank = MPI_UNDEFINED; size = 0; root = MPI_UNDEFINED; root_sim = MPI_UNDEFINED; return; }
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
    int tide_on = 0; double tide_T[6] = {0, 0, 0, 0, 0, 0}; /* KetjuSmoothTide: the region's external tidal tensor (physical, xx yy zz xy yz xz) */

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

/* Wall-clock laps into the KETJU cpu.txt categories; WallclockTime is advanced by the same amount so the
 * enclosing GIZMO timer (gravity, drift) does not count it again. The destructor laps into `rest`. */
struct KjClock {
    double t; int rest;
    explicit KjClock(int rest_cat) : t(my_second()), rest(rest_cat) {}
    void lap(int cat) {double n = my_second(), dt = n - t; CPU_Step[cat] += dt; WallclockTime += dt; t = n;}
    ~KjClock() {lap(rest);}
};
static std::set<int> AllKetjuParticleIndices; /* local indices of all particles in any region */
static double KetjuCollisionEnergy = 0; /* energy change by star collisions since the last KETJU STATS line (compute roots) */
/* Coupling energy ledger since the last KETJU STATS line, host units (rank-local sums):
 * [0] internal energy the host's external kicks put into region members (start: compute roots; end: member owners),
 * [1] its absolute value per region and kick, [2] MSTAR's own energy change (integration error, collisions excluded) */
static double KetjuEnergyLedger[3] = {0, 0, 0};
static double KetjuEtot = 0, KetjuEtot0 = 0; static int KetjuEtotSet = 0; /* direct-sum total energy (rank 0) */
void ketju_note_total_energy(double E, double E0) {KetjuEtot = E; KetjuEtot0 = E0; KetjuEtotSet = 1;}

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
#ifdef KETJU_TYPE4_STARS
    if(P[i].Type == 4 && All.KetjuMinStarMass > 0 && P[i].Mass >= All.KetjuMinStarMass) return 1; /* Stella stars */
#ifndef SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION
    if(P[i].Type == 5) {return (All.KetjuMinBHMass > 0 && P[i].Mass >= All.KetjuMinBHMass) ? 1 : 0;} /* Stella sinks are BHs/remnants */
#endif
#endif
    /* STARFORGE: all sinks are Type 5; ProtoStellarStage selects role.
     * Stages 0-6 (proto / MS / pre-collapse) use KetjuMinStarMass.
     * Stage 7 (relic/BH) uses KetjuMinBHMass.
     * Without SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION there's no stage
     * field — treat every Type 5 as a star (use KetjuMinStarMass only). */
#ifdef SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION
    if(P[i].Type == 5 && P[i].ProtoStellarStage <  7 && All.KetjuMinStarMass > 0) {
        if(P[i].Mass >= All.KetjuMinStarMass) return 1;
    }
    if(P[i].Type == 5 && P[i].ProtoStellarStage == 7 && All.KetjuMinBHMass > 0) {
        if(P[i].Mass >= All.KetjuMinBHMass) return 1;
    }
#else
    if(P[i].Type == 5 && All.KetjuMinStarMass > 0) {
        if(P[i].Mass >= All.KetjuMinStarMass) return 1;
    }
#endif
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
    if(P[i].Type == 5) return 1;
#ifdef KETJU_TYPE4_STARS
    if(P[i].Type == 4) return 1;
#endif
    return 0;
}

/* ============================================================
 *  Helper: is this particle PN-enabled? (BH only)
 * ============================================================ */
static int is_pn_remnant_type4(const ketju_mpi_particle *p)
{
#ifdef KETJU_PN_REMNANT_TAG
    if(p->Type == 4 && (p->RemnantType == REM_ECSN || p->RemnantType == REM_CCSN || p->RemnantType == REM_FSN ||
                        p->RemnantType == REM_PPISN || p->RemnantType == REM_DBH)) return 1; /* NS/BH remnants */
#endif
    return 0;
}

static int is_pn_particle(ketju_mpi_particle *p)
{
    if(is_pn_remnant_type4(p)) return 1;
#ifdef KETJU_SINKS
    if(p->Type == 5 && p->KetjuPNClass == 0) return 1; /* relic BH: PN */
#ifdef KETJU_PN_COMPACT_OBJECTS
    if(p->Type == 5 && p->KetjuPNClass == 1) return 1; /* opt-in: stellar-mass non-relic sinks treated as PN */
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
 *  Helper: the tree's softened pair force, F = G m dr * softened_force_factor(r, h)
 *  The negative half-kicks must remove exactly what the tree kick added, so this is the tree's own
 *  kernel_gravity() (mesh/kernel.h), not a copy of it.
 * ============================================================ */
static double softened_force_factor(double r, double h)
{
    if(h <= 0 || r >= h) return 1.0 / (r * r * r);
    double h_inv = 1.0 / h;
    return kernel_gravity(r * h_inv, h_inv, h_inv * h_inv * h_inv, 1);
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

/* Gather all synchronised chain centers across MPI tasks */
static std::vector<ketju_mpi_particle> gather_chain_centers(void)
{
    /* count local centers */
    int n_local = 0;
    for(int i = 0; i < NumPart; i++) {
        if(is_chain_center(i) && is_synchronized(i)) n_local++;
    }

    /* fill local array */
    std::vector<ketju_mpi_particle> local_centers(n_local);
    int k = 0;
    for(int i = 0; i < NumPart; i++) {
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
#ifdef KETJU_SINKS
            local_centers[k].KetjuPNClass = (P[i].Type == 5) ? KETJU_PN_CLASS(i) : -1;
            for(int j = 0; j < 3; j++) local_centers[k].Spin[j] = (P[i].Type == 5) ? P[i].KetjuSpin[j] : 0;
#endif
#ifdef KETJU_PN_REMNANT_TAG
            local_centers[k].RemnantType = (P[i].Type == 4) ? P[i].RemnantType : -1;
#endif
            local_centers[k].TidalNorm = 0;
#ifndef SINGLE_STAR_TIMESTEPPING
            if(P[i].COM_dt_tidal > 0 && P[i].COM_dt_tidal < 0.5 * MAX_REAL_NUMBER) local_centers[k].TidalNorm = All.cf_a3inv / (P[i].COM_dt_tidal * P[i].COM_dt_tidal);
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

#ifdef KETJU_TYPE4_STARS
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
#endif

/* Find synchronised local chain-eligible particles within `radius` of any center in a region */
static std::set<int> find_local_members(const std::vector<ketju_mpi_particle> &centers, double radius)
{
    std::set<int> members;
    for(int i = 0; i < NumPart; i++) {
        if(!is_chain_eligible(i) || !is_synchronized(i)) continue;
        for(size_t c = 0; c < centers.size(); c++) {
            double dist = particle_distance(P[i].Pos.data_ptr(), const_cast<double*>(centers[c].Pos));
            if(dist < radius) { members.insert(i); break; }
        }
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
#ifdef KETJU_TYPE4_STARS
        if((P[i].Type != 4 && P[i].Type != 5) || is_chain_center(i)) continue; /* only stars outside chains limit a chain; gas and other chains do not */
#endif
        for(size_t c = 0; c < centers.size(); c++) {
            double dist = particle_distance(P[i].Pos.data_ptr(), const_cast<double*>(centers[c].Pos));
            if(dist < radius) { hits.insert(i); break; }
        }
    }
    return hits;
}

/* ============================================================
 *  PHASE 3: MPI gather and integrator setup
 * ============================================================ */

/* com_pos/com_vel are computed on the compute tasks, but the scatter reads them on the affected
 * tasks. The two groups can be disjoint (the cost allocator does not look at where a region's
 * particles live), and then the affected root never computed them: forward them first. */
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
 * Stella: the table radius at the star's ZAMS mass, birth Z and table age (PMS stars: the ZAMS radius, a lower bound).
 * Otherwise: main-sequence R/Rsun = M^0.8 (M < 1 Msun), M^0.57 above (Demircan & Kahraman 1991). */
static double ketju_star_contact_radius(int i)
{
    if(P[i].Type != 4 || P[i].Mass <= 0) return 0;
#ifdef KETJU_PN_REMNANT_TAG
    if(P[i].RemnantType >= 0) return 0;
#endif
    double R_cm;
#if defined(GALSF_RESOLVEDISM_SAMPLE_IMF) && defined(GALSF_RESOLVEDISM_STELLAR_TABLES)
    if(!P[i].sampled || P[i].MstarSampleIMF[0] <= 0) return 0;
    double logM = log10(P[i].MstarSampleIMF[0]), logZ = log10(DMAX(P[i].BirthMetallicity, 1e-10));
    double age_tab = resolvedism_star_age_yr(i) - stellar_t_PMS(logM, logZ);
    if(age_tab >= stellar_lifetime(logM, logZ)) return 0;
    R_cm = pow(10.0, stellar_log_R_cm(logM, logZ, log10(DMAX(age_tab, 100.0))));
#else
    double M = P[i].Mass * UNIT_MASS_IN_SOLAR;
    R_cm = 6.957e10 * ((M < 1) ? pow(M, 0.8) : pow(M, 0.57));
#endif
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
#ifdef KETJU_SINKS
        mp.KetjuPNClass = (P[idx].Type == 5) ? KETJU_PN_CLASS(idx) : -1;
        for(int j = 0; j < 3; j++) mp.Spin[j] = (P[idx].Type == 5) ? P[idx].KetjuSpin[j] : 0;
#endif
#ifdef KETJU_PN_REMNANT_TAG
            mp.RemnantType = (P[idx].Type == 4) ? P[idx].RemnantType : -1;
#endif
#ifdef KETJU_TYPE4_STARS
        for(int k = 0; k < 2; k++) {mp.SoftGrav[k] = P[idx].KetjuTreeSoft[k]; mp.ZetaGrav[k] = P[idx].KetjuTreeZeta[k];}
        tree_pair_softening(idx, mp.SoftNow, mp.ZetaNow);
#endif
#ifdef KJ_STAR_COLLISIONS
        mp.CollRadius = ketju_star_contact_radius(idx);
#ifdef GALSF
        mp.BirthTime = (P[idx].Type == 4) ? P[idx].StellarAge : 0;
#else
        mp.BirthTime = 0;
#endif
#if defined(GALSF_RESOLVEDISM_SAMPLE_IMF) || defined(GALSF_RESOLVEDISM_STOCHASTIC_IMF)
        mp.FormDens = (P[idx].Type == 4) ? P[idx].FormationDensity : 0;
#else
        mp.FormDens = 0;
#endif
#endif
        for(int j = 0; j < 3; j++) mp.GravAcc[j] = P[idx].GravAccel[j];
        for(int j = 0; j < 6; j++) mp.Tidal[j] = 0;
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
        for(int j = 0; j < 6; j++) mp.Tidal[j] = P[idx].tidal_tensorps.data[j];
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
#ifdef KETJU_SINKS
                if(a.Type == 5 && a.KetjuPNClass == 0) a_pn = 1;
                if(b.Type == 5 && b.KetjuPNClass == 0) b_pn = 1;
#endif
#ifdef KETJU_PN_COMPACT_OBJECTS
                if(a.Type == 5 && a.KetjuPNClass == 1) a_pn = 1;
                if(b.Type == 5 && b.KetjuPNClass == 1) b_pn = 1;
#endif
                if(is_pn_remnant_type4(&a)) a_pn = 1;
                if(is_pn_remnant_type4(&b)) b_pn = 1;
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
        reg.integrator->constants->G = All.G * KJ_LEN_TO_NAT * KJ_VEL_TO_NAT * KJ_VEL_TO_NAT / KJ_MASS_TO_NAT;
#if defined(C_LIGHT_CODE)
        reg.integrator->constants->c = C_LIGHT_CODE * KJ_VEL_TO_NAT;
#else
        reg.integrator->constants->c = 2.9979e10 / All.UnitVelocity_in_cm_per_s * KJ_VEL_TO_NAT;
#endif

        /* set options */
        reg.integrator->options->PN_flags = parse_pn_terms();
        reg.integrator->options->gbs_relative_tolerance = All.KetjuIntegrationTolerance;
        reg.integrator->options->enable_bh_mergers = (All.KetjuEnableBHMergerKicks >= 0);
        reg.integrator->options->enable_bh_merger_kicks = All.KetjuEnableBHMergerKicks;
        if(All.KetjuMaxStepCount > 0) reg.integrator->options->max_step_count = All.KetjuMaxStepCount;
        if(All.KetjuUseStarStarSoftening) {
            double h = All.ForceSoftening[4] * All.cf_atime;
#ifdef KETJU_SINKS
            if(All.ForceSoftening[5] > 0) h = DMIN(h, All.ForceSoftening[5] * All.cf_atime);
#endif
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
#ifdef KETJU_SINKS
            reg.extra_data[i].KetjuPNClass = reg.all_particles[i].KetjuPNClass;
            if(i < reg.num_pn_particles) {
                for(int j = 0; j < 3; j++) ps->spin[i][j] = reg.all_particles[i].Spin[j];
            }
#endif
        }
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

#ifdef KETJU_VERBOSE_INTEGRATION
        if(reg.compute_tasks.is_root() && ThisTask == 0) {
            printf("KETJU: Region with %d particles (%d PN), compute tasks %d-%d\n",
                   reg.total_particle_count, reg.num_pn_particles,
                   reg.compute_info.first_task_index, reg.compute_info.final_task_index);
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
#ifdef KETJU_TYPE4_STARS
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
#endif

    for(int i = loop_start; i < loop_end; i++) {
        for(int j = i + 1; j < n; j++) {
            double dr[3], r2 = 0;
            for(int k = 0; k < 3; k++) {
                dr[k] = ps->pos[i][k] - ps->pos[j][k];
                r2 += dr[k] * dr[k];
            }
            double r = sqrt(r2);
            if(r == 0) continue;

#ifdef KETJU_TYPE4_STARS
            { /* the tree's own pair kernel and separation (absolute comoving positions, as the tree differences them), with the
                 softening/zeta of the gravity call whose kick is cancelled */
                const ketju_mpi_particle &pi = reg.all_particles[i], &pj = reg.all_particles[j];
                double drc[3]; for(int k = 0; k < 3; k++) {drc[k] = abs_pos[3*i + k] - abs_pos[3*j + k];}
#if defined(BOX_PERIODIC) && !defined(GRAVITY_NOT_PERIODIC)
                NEAREST_XYZ(drc[0], drc[1], drc[2], 1);
#endif
                const double *si = at_start ? pi.SoftGrav : pi.SoftNow, *zi = at_start ? pi.ZetaGrav : pi.ZetaNow;
                const double *sj = at_start ? pj.SoftGrav : pj.SoftNow, *zj = at_start ? pj.ZetaGrav : pj.ZetaNow;
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
#else
            double fac = kick_factor * All.G * softened_force_factor(r, h);
            for(int k = 0; k < 3; k++) {
                dv[3*i + k] += ps->mass[j] * fac * dr[k];
                dv[3*j + k] -= ps->mass[i] * fac * dr[k];
            }
            (void)at_start;
#endif
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

#ifdef KETJU_TYPE4_STARS
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
    ketju_create_system(&C, R->num_pn_particles, n - R->num_pn_particles, reg.compute_tasks.comm);
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
            ketju_run_integrator(&C, (t_out - t_done) * KJ_TIME_TO_NAT);
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
    ketju_free_system(&C);
}
#endif

/* kick the members by the region's linear tide T x (x relative to the CoM, physical) over dt */
static void ketju_region_tide_kick(KetjuRegion &reg, double dt)
{
    struct ketju_system_physical_state *ps = reg.integrator->physical_state; const double *t = reg.tide_T;
    for(int i = 0; i < reg.integrator->num_particles; i++) {
        const double *x = ps->pos[i];
        ps->vel[i][0] += dt * (t[0]*x[0] + t[3]*x[1] + t[5]*x[2]);
        ps->vel[i][1] += dt * (t[3]*x[0] + t[1]*x[1] + t[4]*x[2]);
        ps->vel[i][2] += dt * (t[5]*x[0] + t[4]*x[1] + t[2]*x[2]);
    }
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
#if defined(COMPUTE_TIDAL_TENSOR_IN_GRAVTREE) && defined(KETJU_TYPE4_STARS)
    if(All.KetjuSmoothTide && reg.compute_tasks.is_member() && reg.integrator && reg.integrator->num_particles >= 2) {
        /* Smooth tide: T = mass-weighted mean over members of (tree tidal tensor - the other members' pair tides, with the
         * tree's kernel and softening of that gravity call). MSTAR applies T (x - x_CoM) during the step; the host's kicks
         * by it are removed here and, at the end positions, with the second negative half-kick, so the host kicks the
         * members by the CoM field and the nonlinear rest only. */
        int n = reg.integrator->num_particles; struct ketju_system_physical_state *ps = reg.integrator->physical_state;
        double M = 0, T[6] = {0, 0, 0, 0, 0, 0};
        for(int i = 0; i < n; i++) {
            const ketju_mpi_particle &me = reg.all_particles[i]; double Ti[6]; for(int k = 0; k < 6; k++) Ti[k] = me.Tidal[k];
            for(int j = 0; j < n; j++) {
                if(j == i) continue; const ketju_mpi_particle &it = reg.all_particles[j];
                double dr[3]; for(int k = 0; k < 3; k++) dr[k] = it.Pos[k] - me.Pos[k];
#if defined(BOX_PERIODIC) && !defined(GRAVITY_NOT_PERIODIC)
                NEAREST_XYZ(dr[0], dr[1], dr[2], 1);
#endif
                double r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2], r = sqrt(r2), fa, fp, ft, ft2; if(r <= 0) continue;
                tree_pair_gravity_factors(r, r2, it.Mass, me.SoftGrav[0], it.SoftGrav[1], me.Type, it.Type, me.ZetaGrav[0], it.ZetaGrav[1], me.Mass, ags_gravity_kernel_shared_BITFLAG(me.Type), fa, fp, ft, ft2);
                Ti[0] -= All.G * (ft2 * dr[0]*dr[0] - ft); Ti[1] -= All.G * (ft2 * dr[1]*dr[1] - ft); Ti[2] -= All.G * (ft2 * dr[2]*dr[2] - ft);
                Ti[3] -= All.G * ft2 * dr[0]*dr[1]; Ti[4] -= All.G * ft2 * dr[1]*dr[2]; Ti[5] -= All.G * ft2 * dr[0]*dr[2];
                (void)fa; (void)fp;
            }
            for(int k = 0; k < 6; k++) T[k] += ps->mass[i] * Ti[k];
            M += ps->mass[i];
        }
        for(int k = 0; k < 6; k++) T[k] *= All.cf_a3inv / M;
        /* a linear tide is a weak perturbation only: gamma = |T| r_max^3 / (G M) above 0.05 keeps the host's kicks */
        double r2max = 0; for(int i = 0; i < n; i++) r2max = DMAX(r2max, ps->pos[i][0]*ps->pos[i][0] + ps->pos[i][1]*ps->pos[i][1] + ps->pos[i][2]*ps->pos[i][2]);
        double Tn = sqrt(T[0]*T[0] + T[1]*T[1] + T[2]*T[2] + 2 * (T[3]*T[3] + T[4]*T[4] + T[5]*T[5]));
        if(Tn * r2max * sqrt(r2max) < 0.05 * All.G * M) {
        reg.tide_on = 1;
        for(int k = 0; k < 6; k++) reg.tide_T[k] = T[k];
        double tn = 1.0 / (KJ_TIME_TO_NAT * KJ_TIME_TO_NAT);
        reg.integrator->options->enable_external_tide = true;
        for(int k = 0; k < 6; k++) reg.integrator->options->external_tidal_tensor[k] = reg.tide_T[k] * tn;
        ketju_region_tide_kick(reg, -first_halfkick);
        }
    }
#endif
    if(reg.compute_tasks.is_root() && reg.integrator) {
        /* energy the host's first half-kick put into the internal motion: its external part (tree kick minus the pair kick
         * just removed) relative to its mass-weighted mean, against the CoM-relative velocities it produced */
        int n = reg.integrator->num_particles; struct ketju_system_physical_state *ps = reg.integrator->physical_state;
        double a = All.cf_atime, mtot = 0, mean[3] = {0, 0, 0}; std::vector<double> ext(3 * n);
        for(int i = 0; i < n; i++) {
            double tr[3] = {0, 0, 0};
            if(reg.tide_on) {const double *t = reg.tide_T; const double *x = ps->pos[i]; tr[0] = t[0]*x[0] + t[3]*x[1] + t[5]*x[2]; tr[1] = t[3]*x[0] + t[1]*x[1] + t[4]*x[2]; tr[2] = t[5]*x[0] + t[4]*x[1] + t[2]*x[2];}
            for(int k = 0; k < 3; k++) {ext[3*i+k] = reg.all_particles[i].GravAcc[k] * first_halfkick / (a * a) + pair_dv[3*i+k] - tr[k] * first_halfkick; mean[k] += ps->mass[i] * ext[3*i+k];}
            mtot += ps->mass[i];
        }
        double W = 0;
        for(int i = 0; i < n; i++) for(int k = 0; k < 3; k++) {double d = ext[3*i+k] - mean[k] / mtot, v = ps->vel[i][k]; W += 0.5 * ps->mass[i] * (v * v - (v - d) * (v - d));}
        KetjuEnergyLedger[0] += W; KetjuEnergyLedger[1] += fabs(W);
    }

    /* expand tight binaries before integration to prevent stalling.
     * Runs on compute root only — broadcast updated velocities to all compute tasks. */
    expand_tight_binaries(reg, dt_physical);
    if(reg.compute_tasks.is_member() && reg.integrator && reg.compute_tasks.size > 1) {
        int n = reg.integrator->num_particles;
        MPI_Bcast(reg.integrator->physical_state->vel, 3 * n, MPI_DOUBLE,
                  reg.compute_tasks.root, reg.compute_tasks.comm);
    }

#ifdef KETJU_TYPE4_STARS
    record_midstep_output_state(reg);
#endif
    /* run MSTAR integrator (all compute tasks participate via compute_tasks.comm) */
    auto E_internal = [&reg]() { /* unsoftened internal energy of the live members, host units */
        struct ketju_system_physical_state *ps = reg.integrator->physical_state; int n = reg.integrator->num_particles; double E = 0;
        for(int i = 0; i < n; i++) {E += 0.5 * ps->mass[i] * (ps->vel[i][0]*ps->vel[i][0] + ps->vel[i][1]*ps->vel[i][1] + ps->vel[i][2]*ps->vel[i][2]);
            for(int j = i + 1; j < n; j++) {double d2 = 0; for(int k = 0; k < 3; k++) {double d = ps->pos[i][k] - ps->pos[j][k]; d2 += d * d;} E -= All.G * ps->mass[i] * ps->mass[j] / sqrt(d2);}}
        return E;
    };
    double E_before = (reg.compute_tasks.is_root() && reg.integrator) ? E_internal() : 0;
    if(reg.compute_tasks.is_member() && reg.integrator) {
        int kepler = kepler_pair_step(reg, dt_physical);
        if(!kepler) {
        kj_state_change_units(reg.integrator->physical_state, reg.integrator->num_particles, 1);
        ketju_run_integrator(reg.integrator, dt_physical * KJ_TIME_TO_NAT);
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
        if(reg.compute_tasks.is_root()) {
            double dE_coll = 0, eu = KJ_MASS_TO_NAT * KJ_VEL_TO_NAT * KJ_VEL_TO_NAT;
            for(int m = 0; m < reg.integrator->num_mergers; m++) if(reg.integrator->merger_infos[m].is_collision) dE_coll += reg.integrator->merger_infos[m].dE_collision / eu;
            /* MSTAR: its error of T - U - U_tide net of collisions (U softened as integrated); exact Kepler pair: direct */
            KetjuEnergyLedger[2] += kepler ? E_internal() - E_before - dE_coll : dE * fabs(E_before);
        }

        /* Per-region integration logs are silenced (they generate ~1 line per
         * region per step, which produces multi-GB stdout files and dominates
         * wall time via syscall/format overhead). Mergers and the warnings
         * below remain. Re-enable by defining KETJU_VERBOSE_INTEGRATION. */
#ifdef KETJU_VERBOSE_INTEGRATION
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

#ifdef KETJU_TYPE4_STARS
    if(reg.compute_tasks.is_member() && reg.integrator) {
        int n = reg.integrator->num_particles; reg.true_vel.assign(3 * reg.total_particle_count, 0.0);
        for(int i = 0; i < n; i++) for(int j = 0; j < 3; j++) reg.true_vel[3*i+j] = reg.integrator->physical_state->vel[i][j];
    }
#endif
    /* subtract second half of tree force */
    do_negative_halfstep_kick(reg, second_halfkick, 0);
    if(reg.tide_on) ketju_region_tide_kick(reg, -second_halfkick); /* the host's end kick by the linear tide, pre-subtracted */
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
#if defined(KJ_STAR_COLLISIONS) && defined(GALSF_RESOLVEDISM_SAMPLE_IMF) && defined(GALSF_RESOLVEDISM_STELLAR_TABLES)
#define KJ_STELLA_PRODUCT /* collision products get Stella table state (Hurley+2002 rejuvenation) */
/* formation time (StellarAge convention) of a star that is age_yr old now */
static double ketju_formation_time_for_age(double age_yr)
{
    double age_gyr = age_yr * 1e-9;
    if(!All.ComovingIntegrationOn) return All.Time - age_gyr / UNIT_TIME_IN_GYR;
    double lo = 1e-6, hi = All.Time; /* time since a decreases with a */
    if(evaluate_time_since_t_initial_in_Gyr(lo) <= age_gyr) return lo;
    for(int it = 0; it < 80; it++) {double mid = 0.5 * (lo + hi); if(evaluate_time_since_t_initial_in_Gyr(mid) > age_gyr) lo = mid; else hi = mid;}
    return 0.5 * (lo + hi);
}
/* main-sequence fraction used for the rejuvenation */
static double ketju_ms_fraction(double M0, double Zb, double age_yr)
{
    double logM = log10(M0), logZ = log10(DMAX(Zb, 1e-10));
    double tab = age_yr - stellar_t_PMS(logM, logZ), t0 = stellar_t_MS_start(logM, logZ), t1 = stellar_t_MS_end(logM, logZ);
    if(!(t1 > t0)) return 1;
    return DMIN(DMAX((tab - t0) / (t1 - t0), 0.0), 1.0);
}
#endif

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

    /* ---- Phase 3: victims send their data to the survivors' tasks ---- */
#ifdef METALS
    const int n_met = NUM_METAL_SPECIES;
#else
    const int n_met = 0;
#endif
    /* [metals][MstarSampleIMF 0][MstarSampleIMF 4][birth Z][age yr] */
    const int buf_size = n_met + 4;
    struct survivor_acc {double M_old; std::vector<double> buf_sum; std::vector<std::vector<double>> victims; std::vector<double> victim_mass;};
    std::unordered_map<int, survivor_acc> acc; /* by local survivor index */

    for(int m = 0; m < n_mergers; m++) {
        ketju_merger_action &act = actions[m];
        std::vector<double> buf(buf_size, 0.0);
        if(act.task_merged == ThisTask) {
            int v = act.idx_merged; (void)v;
#ifdef METALS
            for(int k = 0; k < NUM_METAL_SPECIES; k++) buf[k] = P[v].Metallicity[k];
#endif
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
            if(P[v].Type == 4) {
                buf[n_met] = P[v].MstarSampleIMF[0];
#if (N_STELLAR_MASS >= 5)
                buf[n_met + 1] = P[v].MstarSampleIMF[4];
#endif
            }
#endif
#ifdef KJ_STELLA_PRODUCT
            if(P[v].Type == 4) {buf[n_met + 2] = P[v].BirthMetallicity; buf[n_met + 3] = resolvedism_star_age_yr(v);}
#endif
        }
        if(act.task_merged != act.task_survivor) {
            if(act.task_merged == ThisTask) MPI_Send(buf.data(), buf_size, MPI_DOUBLE, act.task_survivor, 997, MPI_COMM_WORLD);
            if(act.task_survivor == ThisTask) MPI_Recv(buf.data(), buf_size, MPI_DOUBLE, act.task_merged, 997, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if(act.task_survivor == ThisTask) {
            survivor_acc &a = acc[act.idx_survivor];
            if(a.buf_sum.empty()) {a.M_old = act.M_survivor_old; a.buf_sum.assign(n_met, 0.0);}
            for(int k = 0; k < n_met; k++) a.buf_sum[k] += act.M_merged * buf[k];
            a.victims.push_back(buf); a.victim_mass.push_back(act.M_merged);
        }

        /* mark merged particle for removal */
        if(act.task_merged == ThisTask) {
            P[act.idx_merged].Mass = 0;
            TreeMomentsStaleFlag = 1;
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
            P[act.idx_merged].MstarSampleIMF[0] = 0; /* inert ghost: no FB from the merged star */
#endif
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
            P[act.idx_merged].UV_luminosity = 0; P[act.idx_merged].PE_luminosity = 0; P[act.idx_merged].LW_luminosity = 0;
#ifdef GALSF_RESOLVEDISM_NUV_VARIABLE
            P[act.idx_merged].NUV_luminosity = 0;
#endif
#ifdef GALSF_RESOLVEDISM_OPT_VARIABLE
            P[act.idx_merged].OPT_luminosity = 0;
#endif
#endif
        }
    }

    /* ---- Phase 4: each survivor takes its victims' metal mass and (Stella) the merger-product state ---- */
    for(auto &it : acc) {
        int s_idx = it.first; survivor_acc &a = it.second; (void)s_idx;
        double M_sum = a.M_old; for(double mv : a.victim_mass) M_sum += mv;
#ifdef METALS
        for(int k = 0; k < NUM_METAL_SPECIES; k++) P[s_idx].Metallicity[k] = (a.M_old * P[s_idx].Metallicity[k] + a.buf_sum[k]) / M_sum;
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
        {double X_H_new = 1.0 - P[s_idx].Metallicity[0] - P[s_idx].Metallicity[MET_OF(ELEM_He)]; /* close sum(Met[1..]) = 1 via H */
         P[s_idx].Metallicity[MET_OF(ELEM_H)] = (MyFloat)DMAX(X_H_new, 0);}
#endif
#endif
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
        if(P[s_idx].Type != 4) continue;
#ifdef KJ_STELLA_PRODUCT
        if(P[s_idx].sampled && P[s_idx].MstarSampleIMF[0] > 0) {
            /* Product: a main-sequence star of the merged mass at the CoM, birth Z mass-weighted, rejuvenated after Hurley,
             * Pols & Tout 2002 eq. 80: its MS fraction is 0.1 sum_i M_i f_i / M (f_i the components' MS fractions). The wind
             * closure reference (MstarSampleIMF[4]) is reset so the product's cumulative wind table starts at its new age. */
            double Ms = a.M_old * UNIT_MASS_IN_SOLAR, Mnew = P[s_idx].Mass * UNIT_MASS_IN_SOLAR;
            double Zsum = Ms * P[s_idx].BirthMetallicity, fsum = Ms * ketju_ms_fraction(P[s_idx].MstarSampleIMF[0], P[s_idx].BirthMetallicity, resolvedism_star_age_yr(s_idx));
            for(size_t v = 0; v < a.victims.size(); v++) {
                const std::vector<double> &b = a.victims[v]; double Mv = a.victim_mass[v] * UNIT_MASS_IN_SOLAR;
                Zsum += Mv * b[n_met + 2];
                if(b[n_met] > 0) fsum += Mv * ketju_ms_fraction(b[n_met], b[n_met + 2], b[n_met + 3]); else fsum += Mv;
            }
            double Mtot = Ms; for(double mv : a.victim_mass) Mtot += mv * UNIT_MASS_IN_SOLAR;
            double Zb = Zsum / Mtot, logM = log10(Mnew), logZ = log10(DMAX(Zb, 1e-10));
            double t0 = stellar_t_MS_start(logM, logZ), t1 = stellar_t_MS_end(logM, logZ);
            double tab = t0 + 0.1 * (fsum / Mtot) * DMAX(t1 - t0, 0.0);
            P[s_idx].BirthMetallicity = Zb;
            P[s_idx].MstarSampleIMF[0] = Mnew;
            P[s_idx].StellarAge = ketju_formation_time_for_age(stellar_t_PMS(logM, logZ) + tab);
#ifdef GALSF_RESOLVEDISM_WINDS
            {
                double lage = log10(DMAX(tab, 100.0)), cum = 0;
                for(int kk = 0; kk < STBL_NELEM; kk++) cum += stellar_elem_ej_wind_cumulative(logM, logZ, lage, kk);
                P[s_idx].MstarSampleIMF[4] = (Mnew + cum) / UNIT_MASS_IN_SOLAR; /* closure: pending wind = cum(age) - (birth - M) = 0 now */
                P[s_idx].last_wind_log_age = lage; P[s_idx].M_current_old = Mnew; P[s_idx].WindMassAccum = 0; P[s_idx].WindMomentumAccum = 0;
            }
#elif (N_STELLAR_MASS >= 5)
            for(const std::vector<double> &b : a.victims) P[s_idx].MstarSampleIMF[4] += b[n_met + 1];
#endif
            printf("KETJU STAR MERGER PRODUCT: ID %llu M=%g Msun Zb=%g MS fraction %g -> age %g yr\n", (unsigned long long)P[s_idx].ID,
                   Mnew, Zb, 0.1 * fsum / Mtot, stellar_t_PMS(logM, logZ) + tab);
            continue;
        }
#endif
        for(const std::vector<double> &b : a.victims) {
            if(b[n_met] > 0) P[s_idx].MstarSampleIMF[0] += b[n_met]; /* survivor carries the merged stellar mass; keeps its age */
#if (N_STELLAR_MASS >= 5)
            if(b[n_met + 1] > 0) P[s_idx].MstarSampleIMF[4] += b[n_met + 1]; /* birth mass too: wind closure reference */
#endif
        }
#endif
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
    MyIDType Child;
    int Task;
    int Index;
    int Type;
    double Mass;
    double Pos[3];       /* absolute comoving position */
    double Vel[3];       /* CoM-relative peculiar velocity */
#if defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH)
    double OriginalMass;
    KetjuUID MergedInto; /* ID and child number of the particle MSTAR merged this one into (ID 0: none) */
#endif
#ifdef KETJU_SINKS
    short int KetjuPNClass;
    double Spin[3];      /* only meaningful for PN particles */
    int is_pn;           /* 1 if this is a PN particle (index < n_pn) */
#endif
#ifdef KETJU_TYPE4_STARS
    double Sphere[4];    /* chain bounding sphere at end of step (comoving center, radius) */
    double TrueVel[3];   /* CoM-relative velocity before the second negative half-kick */
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
        scatter_buf.resize(n);

        for(int i = 0; i < n; i++) {
            ketju_scatter_particle &sp = scatter_buf[i];
            sp.ID    = reg.extra_data[i].ID;
            sp.Child = reg.extra_data[i].Child;
            sp.Task  = reg.extra_data[i].Task;
            sp.Index = reg.extra_data[i].Index;
            sp.Type  = reg.extra_data[i].Type;
            sp.Mass  = ps->mass[i];
#if defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH)
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
#ifdef KETJU_TYPE4_STARS
                sp.TrueVel[j] = (reg.true_vel[3*i+j] - hubble_end * ps->pos[i][j]) * a_end;
#endif
            }
#ifdef KETJU_SINKS
            sp.KetjuPNClass = reg.extra_data[i].KetjuPNClass;
            sp.is_pn = (i < reg.num_pn_particles) ? 1 : 0;
            for(int j = 0; j < 3; j++)
                sp.Spin[j] = (i < reg.num_pn_particles) ? ps->spin[i][j] : 0;
#endif
        }
#ifdef KETJU_TYPE4_STARS
        { /* chain sphere from the end positions the members drift to; the end-of-step tree opens nodes overlapping it */
            double c[3] = {0,0,0}, rmax2 = 0; int n_live = reg.integrator->num_particles; /* merged ones sit in the tail */
            for(int i = 0; i < n_live; i++) for(int j = 0; j < 3; j++) c[j] += ps->pos[i][j] / (a_end * n_live);
            for(int i = 0; i < n_live; i++) {double d2 = 0; for(int j = 0; j < 3; j++) {double d = ps->pos[i][j] / a_end - c[j]; d2 += d * d;} if(d2 > rmax2) rmax2 = d2;}
            for(int i = 0; i < n; i++) {for(int j = 0; j < 3; j++) {scatter_buf[i].Sphere[j] = reg.com_pos[j] + c[j];} scatter_buf[i].Sphere[3] = 1.01 * sqrt(rmax2) + MIN_REAL_NUMBER;}
        }
#endif

        /* log merger info from integrator */
        if(reg.integrator->num_mergers > 0) {
            for(int m = 0; m < reg.integrator->num_mergers; m++) {
                struct ketju_bh_merger_info *info = &reg.integrator->merger_infos[m];
#if defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH)
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
#if defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH)
    std::vector<double> original_mass;
    std::vector<KetjuUID> merged_into;
#endif

    /* affected root keeps the sorted scatter_buf for collision/merger checks */
    if(reg.affected_tasks.is_root()) {
        final_abs_pos.resize(3 * n);
        final_rel_vel.resize(3 * n);
        final_mass.resize(n);
#if defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH)
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
#if defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH)
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
        P[idx].Mass = sp.Mass;
#ifdef KETJU_TYPE4_STARS
        for(int k = 0; k < 4; k++) {P[idx].KetjuChainSphere[k] = sp.Sphere[k];}
        for(int k = 0; k < 3; k++) {P[idx].KetjuTrueVel[k] = sp.TrueVel[k] + reg.com_vel[k];}
#endif

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
#ifdef KETJU_TYPE4_STARS
                P[idx].KetjuTargetPos[j] = desired_pos;
#endif
                double delta_pos = desired_pos - P[idx].Pos[j];
#ifdef BOX_PERIODIC
                double box = (j == 0) ? boxSize_X : (j == 1) ? boxSize_Y : boxSize_Z;
                if(box > 0) { while(delta_pos > 0.5 * box) delta_pos -= box; while(delta_pos < -0.5 * box) delta_pos += box; }
#endif
                P[idx].Vel[j] = delta_pos / dt_drift;
                P[idx].KetjuFinalVel[j] = sp.Vel[j] + reg.com_vel[j];
            }
        } else {
            /* no drift — set position and velocity directly */
            for(int j = 0; j < 3; j++) {
                P[idx].Pos[j] = sp.Pos[j];
#ifdef KETJU_TYPE4_STARS
                P[idx].KetjuTargetPos[j] = sp.Pos[j];
#endif
                P[idx].Vel[j] = sp.Vel[j] + reg.com_vel[j];
                P[idx].KetjuFinalVel[j] = P[idx].Vel[j];
            }
        }

#ifdef KETJU_SINKS
        /* update spin for PN particles */
        if(sp.is_pn && P[idx].Type == 5) {
            for(int j = 0; j < 3; j++) P[idx].KetjuSpin[j] = sp.Spin[j];
        }
#endif
        P[idx].KetjuIntegrated = 1;
    }

    /* ---- Phase 5: geometric coalescence check for Type 5 sinks ----
     *  All sinks have ProtoStellarRadius_inSolar populated by the protostellar
     *  evolution: protostars (stage 1-5) get their physical radius, MS stars
     *  (stage 6) get the MS radius, relics (stage 7) get 6 * 2GM/c^2 (an ISCO-
     *  like criterion). Coalescence is therefore detected when (R_i + R_j) >
     *  separation, regardless of whether the pair is star-star, star-BH, or
     *  BH-BH. handle_mergers() (gated on KETJU_MERGE_BH) accepts any Type 5
     *  pair and transfers metals to the survivor. */
#if defined(KETJU_MERGE_BH) && defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
    {
        const double R_SUN_IN_CM = 6.957e10;
        std::vector<double> my_radii(my_count, 0.0);
        for(int i = 0; i < my_count; i++) {
            if(my_particles[i].Type != 5) continue;
            int idx = my_particles[i].Index;
            if(P[idx].ProtoStellarRadius_inSolar > 0)
                my_radii[i] = P[idx].ProtoStellarRadius_inSolar * R_SUN_IN_CM;
        }

        std::vector<double> sink_radius_cm;
        std::vector<int> recv_counts(n_aff, 0), recv_displs(n_aff, 0);
        MPI_Gather(&my_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                   reg.affected_tasks.root, reg.affected_tasks.comm);
        if(reg.affected_tasks.is_root()) {
            for(int t = 1; t < n_aff; t++) recv_displs[t] = recv_displs[t-1] + recv_counts[t-1];
            sink_radius_cm.resize(n, 0.0);
        }
        MPI_Gatherv(my_radii.data(), my_count, MPI_DOUBLE,
                     sink_radius_cm.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                     reg.affected_tasks.root, reg.affected_tasks.comm);

        if(reg.affected_tasks.is_root()) {
            for(int i = 0; i < n; i++) {
                if(final_mass[i] <= 0 || reg.extra_data[i].Type != 5) continue;
                if(sink_radius_cm[i] <= 0) continue;
                for(int j = i + 1; j < n; j++) {
                    if(final_mass[j] <= 0 || reg.extra_data[j].Type != 5) continue;
                    if(sink_radius_cm[j] <= 0) continue;
                    /* If both are PN-enabled BHs AND KETJU is integrating PN terms,
                     * MSTAR's BH-merger handler (ketju_calculate_bh_merger_remnant_properties)
                     * will handle the coalescence with full GR remnant properties
                     * (mass + spin + recoil kick) at its 12·2GM/c² threshold.
                     * Skip here to avoid double-merging the same BH-BH pair.
                     * When PN is off (KetjuPNTerms="none"), MSTAR is dormant and
                     * we keep the geometric check active for BH-BH too. */
                    if(reg.extra_data[i].KetjuPNClass == 0 && reg.extra_data[j].KetjuPNClass == 0
                       && strcmp(All.KetjuPNTerms, "none") != 0) continue;
                    double d2 = 0;
                    for(int k = 0; k < 3; k++) {
                        double dx = final_abs_pos[3*i + k] - final_abs_pos[3*j + k];
                        d2 += dx * dx;
                    }
                    double sep_cm = sqrt(d2) * UNIT_LENGTH_IN_CGS * All.cf_atime;
                    if(sep_cm < sink_radius_cm[i] + sink_radius_cm[j]) {
                        int victim = (final_mass[i] < final_mass[j]) ? i : j;
                        int surv = (victim == i) ? j : i;
                        final_mass[surv] += final_mass[victim];
                        final_mass[victim] = 0;
                        double Mi = original_mass[i] * UNIT_MASS_IN_SOLAR;
                        double Mj = original_mass[j] * UNIT_MASS_IN_SOLAR;
                        printf("KETJU SINK COALESCENCE: ID %llu (%.2f Msun) + ID %llu (%.2f Msun) at sep=%.2e cm (R1+R2=%.2e cm)\n",
                            (unsigned long long)reg.extra_data[i].ID, Mi,
                            (unsigned long long)reg.extra_data[j].ID, Mj,
                            sep_cm, sink_radius_cm[i] + sink_radius_cm[j]);
                    }
                }
            }
        }
    }
#endif

    /* handle mergers (integrator mergers + stellar collisions) — merger decisions
     * are made on affected root then broadcast to all affected tasks for execution */
#if defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH)
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

/* Caps on each timestep group's step, integer time (TIMEBASE: none); only groups of >= 2 centres are integrated.
 * Motion (Mannerkoski+ 2023 sec. 2.4.2, eqs. 4 and 6): dt <= 0.1 r_region / v_CoM and dt <= r_region / (6 sigma), so stars
 * cannot cross a region in one step. v_CoM and sigma (1D) are taken relative to the stars around the group (all Type 4/5
 * within the timestep-limiting radius, other groups' centres included) rather than the simulation frame: a cluster moving
 * through a galaxy at 100 km/s is no reason to shorten its regions' steps.
 * Tide (KetjuTidalStepRes > 0): the external field reaches members only as two kicks per step. For a bound pair the energy
 * one step's kicks put into its orbit is about mu |T| a v_p dt; above KetjuTidalErrorBudget |E_b| the step is reduced to
 * that budget, but not below KetjuTidalStepRes P (1 - e)^1.5, where the kicks start to resolve the pericentre passage. */
static long long CapTally[2] = {0, 0}; /* groups whose bin a motion / tide cap set (task 0) */
static long long FlybyTally = 0;        /* fly-by steps that shortened a star's or group's step (rank-local) */
static std::vector<long long> region_step_caps(const std::vector<std::vector<ketju_mpi_particle>> &groups)
{
    const int n = (int)groups.size();
    std::vector<long long> caps(2 * n, (long long)TIMEBASE); /* [2r] motion, [2r+1] tide */
    const double R = All.KetjuRegionRadius, Rlim = DMAX(All.KetjuTimestepLimitingFactor, 2.0) * R; /* physical */
    const double a = All.cf_atime, Rlim_c = Rlim / a;
    auto to_ti = [](double dt) {double t = dt * All.cf_hubble_a / All.Timebase_interval; return (t >= (double)TIMEBASE) ? (long long)TIMEBASE : (long long)DMAX(t, 1.0);};

    /* field statistics: N, sum v, sum v^2 of the stars within Rlim of a group's centres, not counting its own centres */
    std::vector<double> st(5 * n, 0.0);
    std::unordered_map<long long, std::vector<std::pair<int, std::array<double, 3>>>> grid; /* cell -> (group, centre position or image) */
    auto key = [](long long x, long long y, long long z) {return (x & 0x1fffff) | ((y & 0x1fffff) << 21) | ((z & 0x1fffff) << 42);};
    auto cell = [Rlim_c](double x) {return (long long)floor(x / Rlim_c);};
    std::unordered_map<int, int> own; /* local index of a centre -> its group */
    for(int r = 0; r < n; r++) {
        if(groups[r].size() < 2) continue;
        for(const ketju_mpi_particle &c : groups[r]) {
            if(c.Task == ThisTask) own[c.Index] = r;
            int nim[3] = {1, 1, 1}; double sh[3][3];
            for(int k = 0; k < 3; k++) {sh[k][0] = 0; (void)nim;}
#ifdef BOX_PERIODIC
            double box[3] = {boxSize_X, boxSize_Y, boxSize_Z};
            for(int k = 0; k < 3; k++) {nim[k] = 1; if(c.Pos[k] < Rlim_c) sh[k][nim[k]++] = box[k]; if(c.Pos[k] > box[k] - Rlim_c) sh[k][nim[k]++] = -box[k];}
#endif
            for(int ix = 0; ix < nim[0]; ix++) for(int iy = 0; iy < nim[1]; iy++) for(int iz = 0; iz < nim[2]; iz++) {
                std::array<double, 3> x = {c.Pos[0] + sh[0][ix], c.Pos[1] + sh[1][iy], c.Pos[2] + sh[2][iz]};
                grid[key(cell(x[0]), cell(x[1]), cell(x[2]))].push_back({r, x});
            }
        }
    }
    std::vector<int> last(n, -1);
    for(int i = 0; i < NumPart; i++) {
        if((P[i].Type != 4 && P[i].Type != 5) || P[i].Mass <= 0) continue;
        auto ow = own.find(i); int mine = (ow == own.end()) ? -1 : ow->second;
        long long cx = cell(P[i].Pos[0]), cy = cell(P[i].Pos[1]), cz = cell(P[i].Pos[2]);
        for(int dx = -1; dx <= 1; dx++) for(int dy = -1; dy <= 1; dy++) for(int dz = -1; dz <= 1; dz++) {
            auto it = grid.find(key(cx + dx, cy + dy, cz + dz)); if(it == grid.end()) continue;
            for(const auto &gc : it->second) {
                int r = gc.first; if(r == mine || last[r] == i) continue;
                double d2 = 0; for(int k = 0; k < 3; k++) {double d = P[i].Pos[k] - gc.second[k]; d2 += d * d;}
                if(d2 >= Rlim_c * Rlim_c) continue;
                last[r] = i; st[5*r] += 1;
                for(int k = 0; k < 3; k++) {double v = P[i].Vel[k] / a; st[5*r+1+k] += v; st[5*r+4] += v * v;}
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, st.data(), 5 * n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for(int r = 0; r < n; r++) {
        const std::vector<ketju_mpi_particle> &g = groups[r];
        if(g.size() < 2) continue;
        double N = st[5*r], fc = All.KetjuRegionMotionCapFactor;
        if(N >= 1 && fc > 0) {
            double M = 0, vc[3] = {0, 0, 0}, vf[3], vrel2 = 0;
            for(const ketju_mpi_particle &c : g) {M += c.Mass; for(int k = 0; k < 3; k++) vc[k] += c.Mass * c.Vel[k] / a;}
            for(int k = 0; k < 3; k++) {vf[k] = st[5*r+1+k] / N; double d = vc[k] / M - vf[k]; vrel2 += d * d;}
            double dt = MAX_REAL_NUMBER;
            if(vrel2 > 0) dt = fc * 0.1 * R / sqrt(vrel2);
            double sig2 = (st[5*r+4] / N - (vf[0]*vf[0] + vf[1]*vf[1] + vf[2]*vf[2])) / 3.0;
            if(N >= 2 && sig2 > 0) dt = DMIN(dt, fc * R / (6.0 * sqrt(sig2)));
            if(dt < MAX_REAL_NUMBER) caps[2*r] = to_ti(dt);
        }
        if(All.KetjuTidalStepRes > 0 && g.size() <= 64) {
            for(size_t i = 0; i < g.size(); i++) { /* each member's tightest bound partner */
                int best = -1; double best_a = MAX_REAL_NUMBER, best_e = 0;
                for(size_t j = 0; j < g.size(); j++) {
                    if(j == i) continue;
                    double dr[3], dv[3], r2 = 0, v2 = 0, rv = 0;
                    for(int k = 0; k < 3; k++) {dr[k] = (g[i].Pos[k] - g[j].Pos[k]) * a; dv[k] = (g[i].Vel[k] - g[j].Vel[k]) / a; r2 += dr[k]*dr[k]; v2 += dv[k]*dv[k]; rv += dr[k]*dv[k];}
                    double mu = All.G * (g[i].Mass + g[j].Mass), E = 0.5 * v2 - mu / sqrt(r2);
                    if(E >= 0) continue;
                    double sma = -mu / (2 * E); if(sma >= best_a) continue;
                    double h2 = r2 * v2 - rv * rv, e2 = 1 + 2 * E * h2 / (mu * mu);
                    best = (int)j; best_a = sma; best_e = sqrt(DMAX(e2, 0.0));
                }
                if(best < 0) continue; /* a pair found from both sides gives the same cap twice */
                double T = DMAX(g[i].TidalNorm, g[best].TidalNorm); if(T <= 0) continue;
                double m1 = g[i].Mass, m2 = g[best].Mass, M = m1 + m2, mu = All.G * M, e = DMIN(best_e, 0.999999);
                double Eb = All.G * m1 * m2 / (2 * best_a), red = m1 * m2 / M, vp = sqrt(mu * (1 + e) / (best_a * (1 - e)));
                double Pb = 2 * M_PI * sqrt(best_a * best_a * best_a / mu);
                double dt_budget = All.KetjuTidalErrorBudget * Eb / (red * T * best_a * vp), floor_dt = All.KetjuTidalStepRes * Pb * pow(1 - e, 1.5);
                long long ti = to_ti(DMAX(dt_budget, floor_dt));
                if(ti < caps[2*r+1]) caps[2*r+1] = ti;
            }
        }
    }
    return caps;
}

/* Fly-by step (prototype, KetjuFlybyStepEta > 0): for every synchronised local star the symmetrised pair step
 * eta min(t_ff, t_fb) (Pelupessy+ 2012; FROST eq. 23 with its approach correction) over all stars of other timestep groups,
 * positions predicted to now. No distance cut: a star on a long step must already see the stars that can reach it within that
 * step. O(N_sync N) after gathering every star each step: test-size runs only (production: the gravity tree walk).
 * Returns the step per local index (integer time, TIMEBASE where none). */
static std::unordered_map<int, long long> flyby_steps(const std::vector<std::vector<ketju_mpi_particle>> &groups)
{
    std::unordered_map<int, long long> out;
    struct S {MyIDType id, child; int group; double m, x[3], v[3];};
    std::unordered_map<KetjuUID, int, KetjuUIDHash> gid; for(size_t r = 0; r < groups.size(); r++) for(const ketju_mpi_particle &c : groups[r]) gid[{c.ID, c.Child}] = (int)r;
    std::vector<S> loc;
    const double a = All.cf_atime;
    for(int i = 0; i < NumPart; i++) {
        if((P[i].Type != 4 && P[i].Type != 5) || P[i].Mass <= 0) continue;
        S q; q.id = P[i].ID; q.child = P[i].ID_child_number; auto it = gid.find({P[i].ID, P[i].ID_child_number}); q.group = (it == gid.end()) ? -1 : it->second; q.m = P[i].Mass;
        double dtd = (P[i].Ti_current < All.Ti_Current) ? get_drift_factor(P[i].Ti_current, All.Ti_Current, i, 0) : 0;
        for(int k = 0; k < 3; k++) {q.x[k] = (P[i].Pos[k] + P[i].Vel[k] * dtd) * a; q.v[k] = P[i].Vel[k] / a;}
        loc.push_back(q);
    }
    int nl = loc.size(); std::vector<int> cnt(NTask), bc(NTask), bd(NTask);
    MPI_Allgather(&nl, 1, MPI_INT, cnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int nt = 0; for(int t = 0; t < NTask; t++) {bc[t] = cnt[t] * sizeof(S); bd[t] = nt * sizeof(S); nt += cnt[t];}
    std::vector<S> all(nt);
    MPI_Allgatherv(loc.data(), nl * sizeof(S), MPI_BYTE, all.data(), bc.data(), bd.data(), MPI_BYTE, MPI_COMM_WORLD);
    const double eta = All.KetjuFlybyStepEta;
    std::unordered_map<KetjuUID, int, KetjuUIDHash> local_index; for(int i = 0; i < NumPart; i++) if(is_synchronized(i) && (P[i].Type == 4 || P[i].Type == 5)) local_index[{P[i].ID, P[i].ID_child_number}] = i;
    for(int q = bd[ThisTask] / (int)sizeof(S), qe = q + nl; q < qe; q++) {
        auto li = local_index.find({all[q].id, all[q].child}); if(li == local_index.end()) continue;
        const S &A = all[q]; double tmin = MAX_REAL_NUMBER;
        {
            for(int j = 0; j < nt; j++) {
                const S &B = all[j]; if(j == q || (A.group >= 0 && A.group == B.group)) continue;
                double d[3], u[3], r2 = 0, v2 = 0, rv = 0;
                for(int k = 0; k < 3; k++) {d[k] = B.x[k] - A.x[k]; u[k] = B.v[k] - A.v[k]; r2 += d[k]*d[k]; v2 += u[k]*u[k]; rv += d[k]*u[k];}
                if(r2 <= 0) continue;
                double r = sqrt(r2), GM = All.G * (A.m + B.m), tff = sqrt(r2 * r / GM), tfb = r / sqrt(v2 + MIN_REAL_NUMBER);
                double cff = 1.5 * rv / r2, cfb = rv / r2 * (1 + GM / (r * (v2 + MIN_REAL_NUMBER)));
                double ctf = cff * tff * tff * eta, ctb = cfb * tfb * tfb * eta;
                double sf = sqrt(1 + cff * ctf * eta), sb = sqrt(1 + cfb * ctb * eta); /* rv < 0 (approach) shortens both */
                double t = eta * DMIN(tff + ctf / (sf + 1), tfb + ctb / (sb + 1));
                if(t < tmin) tmin = t;
            }
        }
        if(tmin < MAX_REAL_NUMBER) {double t = DMAX(tmin * All.cf_hubble_a / All.Timebase_interval, 1.0); out[li->second] = (t >= (double)TIMEBASE) ? (long long)TIMEBASE : (long long)t;}
    }
    return out;
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
void ketju_limit_timesteps(void)
{
    KjClock clk(CPU_KETJULIMIT);
    CachedChainCentersValid = 0;

    if(All.KetjuRegionRadius <= 0) return;
    if(All.KetjuMinBHMass <= 0 && All.KetjuMinStarMass <= 0) return;

    /* gather synchronised chain centres (cached for ketju_find_regions) */
    std::vector<ketju_mpi_particle> centers = gather_chain_centers();
    CachedChainCenters = centers;
    CachedChainCentersValid = 1;
    if(centers.empty()) return;

    const double limited_radius = 2.0 * All.KetjuRegionRadius;
    const double limiting_radius = All.KetjuTimestepLimitingFactor * All.KetjuRegionRadius; /* GADGET-4 KETJU default 100 */
#ifdef KETJU_TYPE4_STARS
    std::vector<std::vector<ketju_mpi_particle>> tsr_centers = build_orbit_regions(centers); (void)limited_radius;
    CachedCenterGroups = tsr_centers;
#else
    std::vector<std::vector<ketju_mpi_particle>> tsr_centers = merge_overlapping_regions(centers, 2.0 * limited_radius);
#endif
    const int n_tsr = (int)tsr_centers.size();

    std::vector<std::set<int>> limited(n_tsr);
    std::vector<long long> min_ti(2 * n_tsr, (long long)TIMEBASE); /* [2r] over limited+limiting, [2r+1] limited only */
    for(int r = 0; r < n_tsr; r++) {
#ifdef KETJU_TYPE4_STARS
        limited[r] = local_members_from_group(tsr_centers[r]);
#else
        limited[r] = find_local_members(tsr_centers[r], limited_radius);
#endif
        for(int idx : limited[r]) {
            long long ti = GET_INTEGERTIME_FROM_TIMEBIN(P[idx].TimeBin);
            if(ti > 0 && ti < min_ti[2*r]) min_ti[2*r] = ti;
            if(ti > 0 && ti < min_ti[2*r+1]) min_ti[2*r+1] = ti;
        }
#if defined(KETJU_TYPE4_STARS) && !defined(BOX_PERIODIC)
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
        for(int i = 0; i < NumPart; i++) {
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
    if(All.KetjuFlybyStepEta > 0) { /* fly-by steps: a group's members set its shared bin; stars in no group move directly */
        std::unordered_map<int, long long> fb = flyby_steps(tsr_centers);
        std::unordered_map<int, int> member_group; for(int r = 0; r < n_tsr; r++) for(int idx : limited[r]) member_group[idx] = r;
        for(const auto &f : fb) {
            auto mg = member_group.find(f.first);
            if(mg != member_group.end()) {if(f.second < min_ti[2*mg->second]) {min_ti[2*mg->second] = f.second; FlybyTally++;}}
            else if(f.second < (long long)GET_INTEGERTIME_FROM_TIMEBIN(P[f.first].TimeBin)) {
                int bin = get_timestep_bin((integertime)f.second); while(bin > 0 && !TimeBinActive[bin]) {bin--;}
                if(bin > 0 && bin < P[f.first].TimeBin) {move_particle_timebin(f.first, P[f.first].TimeBin, bin); P[f.first].dt_step = GET_INTEGERTIME_FROM_TIMEBIN(bin); FlybyTally++;}
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, min_ti.data(), 2 * n_tsr, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    { /* motion and tide caps (identical on all tasks) */
        std::vector<long long> caps = region_step_caps(tsr_centers);
        for(int r = 0; r < n_tsr; r++) for(int c = 0; c < 2; c++) {
            if(caps[2*r+c] < min_ti[2*r] && caps[2*r+c] < (long long)TIMEBASE) {min_ti[2*r] = caps[2*r+c]; if(ThisTask == 0) CapTally[c]++;}
        }
    }

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
    KjClock clk(CPU_KETJUFIND);
    /* Clear the chain flags of particles whose step has ended, and only theirs: a member of a region
     * on a longer bin is still inside the step MSTAR integrated, and its flags keep it out of Hermite
     * and mark it for the velocity swap at the end of that step. KetjuIntegrated is cleared here, not
     * in finish_step, so it also survives to the guard checks of the step that follows. */
    for(int i = 0; i < NumPart; i++) {
        if(!is_synchronized(i)) continue;
#ifdef HERMITE_INTEGRATION
        /* A member whose step just ended: if it stays in a chain it never reaches the stale check
         * (in-region guard fires first); if it exits, the next eligible_for_hermite() call falls
         * back to KDK once before re-engaging Hermite with fresh OldPos/Acc/Jerk. */
        if(P[i].KetjuIntegrated) {P[i].HermiteHistoryStale = 1;}
#endif
        P[i].KetjuIntegrated = 0; P[i].KetjuChainID = 0; P[i].COM_dt_tidal = 0; /* 0 = no chain-external value (see timestep.cc) */
#ifdef KETJU_TYPE4_STARS
        P[i].KetjuChainSphere[3] = 0;
#endif
    }
    ActiveRegions.clear();
    AllKetjuParticleIndices.clear();

    /* validate parameters */
    if(All.KetjuRegionRadius <= 0) return;
    if(All.KetjuMinBHMass <= 0 && All.KetjuMinStarMass <= 0) return;

    /* reuse chain centers from ketju_limit_timesteps if available, else gather fresh */
    std::vector<ketju_mpi_particle> centers;
    const int reuse_groups = CachedChainCentersValid; /* same centres as the limiter, so the same orbit groups */
    if(CachedChainCentersValid) {
        centers = std::move(CachedChainCenters);
        CachedChainCentersValid = 0;
    } else {
        centers = gather_chain_centers();
    }
    if(centers.empty()) return;

    /* merge overlapping regions */
    double merge_radius = 2.0 * All.KetjuRegionRadius;
#ifdef KETJU_TYPE4_STARS
    std::vector<std::vector<ketju_mpi_particle>> region_centers = reuse_groups ? std::move(CachedCenterGroups) : build_orbit_regions(centers); (void)merge_radius;
#else
    std::vector<std::vector<ketju_mpi_particle>> region_centers = merge_overlapping_regions(centers, merge_radius); (void)reuse_groups;
#endif

    /* build regions (first pass: find members) */
    int n_regions = region_centers.size();

    for(int r = 0; r < n_regions; r++) {
        KetjuRegion reg;
        reg.centers = region_centers[r];
#ifdef KETJU_TYPE4_STARS
        reg.local_member_indices = local_members_from_group(reg.centers);
#else
        reg.local_member_indices = find_local_members(reg.centers, All.KetjuRegionRadius);
#endif

        ActiveRegions.push_back(std::move(reg));
    }

    /* member totals of all regions in one reduction, then per-task counts of the multi-member ones in one gather */
    std::vector<int> region_counts(n_regions);
    for(int r = 0; r < n_regions; r++) region_counts[r] = (int)ActiveRegions[r].local_member_indices.size();
    MPI_Allreduce(MPI_IN_PLACE, region_counts.data(), n_regions, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    std::vector<int> multi; /* regions with >= 2 members; the rest stay as placeholders with nothing to integrate */
    for(int r = 0; r < n_regions; r++) {
        ActiveRegions[r].total_particle_count = region_counts[r];
        if(region_counts[r] < 2) continue;
        multi.push_back(r);
        for(int idx : ActiveRegions[r].local_member_indices) AllKetjuParticleIndices.insert(idx);
    }
    const int n_multi = (int)multi.size();
    std::vector<int> task_counts((size_t)NTask * n_multi); /* [t * n_multi + k] */
    if(n_multi > 0) {
        std::vector<int> mine(n_multi);
        for(int k = 0; k < n_multi; k++) mine[k] = (int)ActiveRegions[multi[k]].local_member_indices.size();
        MPI_Allgather(mine.data(), n_multi, MPI_INT, task_counts.data(), n_multi, MPI_INT, MPI_COMM_WORLD);
    }

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

    /* Assign each chain member a shared KetjuChainID = global-min member ID
     * across the region. forcetree.cc uses this to skip Type-5 neighbors that
     * belong to the same chain (i.e., MSTAR is integrating that pair internally)
     * when accumulating Min_Sink_Approach_Time / Min_Sink_Freefall_time, so
     * dt_2body reflects only non-chain encounters. */
    for(size_t r = 0; r < ActiveRegions.size(); r++) {
        KetjuRegion &reg = ActiveRegions[r];
        if(!reg.affected_tasks.is_member()) continue;
        if(reg.total_particle_count < 2) continue; /* single-particle "placeholder" regions */
        MyIDType local_min = (MyIDType)-1; /* unsigned-max */
        for(int idx : reg.local_member_indices) {
            if(P[idx].ID < local_min) local_min = P[idx].ID;
        }
        MyIDType chain_id, min_child, local_minc = (MyIDType)-1;
        MPI_Allreduce(&local_min, &chain_id, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, reg.affected_tasks.comm);
        for(int idx : reg.local_member_indices) {if(P[idx].ID == chain_id && P[idx].ID_child_number < local_minc) local_minc = P[idx].ID_child_number;}
        MPI_Allreduce(&local_minc, &min_child, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, reg.affected_tasks.comm);
        chain_id ^= min_child * 0x9E3779B97F4A7C15ULL; if(chain_id == 0) chain_id = 1; /* siblings share P.ID: the child number keeps chains distinct; = min ID when it is 0 */
        for(int idx : reg.local_member_indices) P[idx].KetjuChainID = chain_id;
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
#ifdef KETJU_VERBOSE_INTEGRATION
            for(int r = 0; r < n_regions; r++) {
                printf("KETJU:   region %d: %d particles, %d centers\n",
                       r, ActiveRegions[r].total_particle_count, (int)ActiveRegions[r].centers.size());
            }
#endif
        }
    }
}

void ketju_run_integration(void)
{
    KjClock clk(CPU_KETJUFIND);
    if(ActiveRegions.empty()) return;

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
        clk.lap(CPU_KETJUSCATTER);
    }
    } /* end for(seq) */

    /* update cost estimates for next step's load balancing */
    clk.rest = CPU_KETJUFINISH;
    update_region_costs();

    /* Cumulative MSTAR statistics, printed every 1024 steps and on the step that reaches the end of
     * the timeline: the per-region log is compiled out, so otherwise nothing shows how hard the
     * integrator works. A step-count bail-out cannot appear here: MSTAR aborts the run past
     * KetjuMaxStepCount. */
    {
        static long long cum_calls = 0, cum_regions = 0, cum_ok = 0, cum_failed = 0, cum_big_dE = 0, cum_limit[4] = {0, 0, 0, 0};
        static long long cum_max_steps = 0; static double cum_max_dE = 0;
        long long loc[8] = {0, 0, 0, 0, LimitTally[0], LimitTally[1], LimitTally[2], LimitTally[3]}, glob[8]; /* regions integrated, accepted, failed, |dE/E| > 1e-4, then LimitTally */
        for(int k = 0; k < 4; k++) LimitTally[k] = 0;
        double loc_max[2] = {0, 0}, glob_max[2]; /* max steps in one region, max |dE/E| */
        integertime max_ti_step = 0;
        for(size_t r = 0; r < ActiveRegions.size(); r++) {
            KetjuRegion &reg = ActiveRegions[r];
            if(reg.ti_step > max_ti_step) max_ti_step = reg.ti_step;
            if(!reg.compute_tasks.is_root() || !reg.integrator) continue;
            long long n = (long long)reg.integrator->perf->successful_steps + reg.integrator->perf->failed_steps;
            if(n <= 0) continue;
            double dE = fabs(reg.integrator->perf->relative_energy_error);
            loc[0]++; loc[1] += reg.integrator->perf->successful_steps; loc[2] += reg.integrator->perf->failed_steps;
            if(dE > 1e-4) loc[3]++;
            if((double)n > loc_max[0]) loc_max[0] = (double)n;
            if(dE > loc_max[1]) loc_max[1] = dE;
        }
        if(GET_INTEGERTIME_FROM_TIMEBIN(All.HighestActiveTimeBin) > max_ti_step) max_ti_step = GET_INTEGERTIME_FROM_TIMEBIN(All.HighestActiveTimeBin); /* end of run without a region reaching it */
        MPI_Reduce(loc, glob, 8, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(loc_max, glob_max, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        static double cum_energy[4] = {0, 0, 0, 0}; /* external kick work on internal motion (net, |.|), MSTAR error, collisions */
        double eloc[4] = {KetjuEnergyLedger[0], KetjuEnergyLedger[1], KetjuEnergyLedger[2], KetjuCollisionEnergy}, eglob[4];
        for(int k = 0; k < 3; k++) KetjuEnergyLedger[k] = 0;
        KetjuCollisionEnergy = 0;
        MPI_Reduce(eloc, eglob, 4, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if(ThisTask == 0) {
            cum_calls++; cum_regions += glob[0]; cum_ok += glob[1]; cum_failed += glob[2]; cum_big_dE += glob[3];
            for(int k = 0; k < 4; k++) cum_energy[k] += eglob[k];
            for(int k = 0; k < 4; k++) cum_limit[k] += glob[4 + k];
            if((long long)glob_max[0] > cum_max_steps) cum_max_steps = (long long)glob_max[0];
            if(glob_max[1] > cum_max_dE) cum_max_dE = glob_max[1];
            if((cum_calls % 1024) == 0 || All.Ti_Current + max_ti_step >= TIMEBASE)
                printf("KETJU STATS: t=%g steps=%lld region-integrations=%lld MSTAR accepted=%lld failed=%lld "
                       "max-per-region=%lld |dE/E|>1e-4: %lld max|dE/E|=%g limited-moves down=%lld up=%lld "
                       "timestep-regions=%lld set-by-limiting=%lld\n", All.Time, cum_calls, cum_regions,
                       cum_ok, cum_failed, cum_max_steps, cum_big_dE, cum_max_dE, cum_limit[0], cum_limit[1], cum_limit[2], cum_limit[3]);
            if((cum_calls % 1024) == 0 || All.Ti_Current + max_ti_step >= TIMEBASE) {
                /* coupling energy check: if the direct-sum drift is the kick work, the coupling (impulsive external field) dominates;
                 * if it is the MSTAR term, the integrator does. Normalised by |E0| of the direct sum when that is computed. */
                double norm = (KetjuEtotSet && KetjuEtot0 != 0) ? fabs(KetjuEtot0) : 1;
                printf("KETJU CAPS: t=%g timestep-regions set by motion cap=%lld tide cap=%lld fly-by (task 0)=%lld\n", All.Time, CapTally[0], CapTally[1], FlybyTally);
                printf("KETJU ENERGY: t=%g units=%s kick-work-internal net=%.4e abs=%.4e MSTAR-dE=%.4e collisions-dE=%.4e ETOT dE=%.4e\n",
                       All.Time, (norm != 1) ? "|E0|" : "code", cum_energy[0] / norm, cum_energy[1] / norm, cum_energy[2] / norm, cum_energy[3] / norm,
                       KetjuEtotSet ? (KetjuEtot - KetjuEtot0) / norm : 0.0);
            }
        }
    }

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

/* Snapshots are written after the drift to the output time but before the step's end: a chain member whose step ends
 * there still carries the velocity-trick chord velocity, so MSTAR's end-of-step velocity is swapped in for the write, and
 * back. A member part-way through its step sits on the chord: its position and velocity are swapped with the state
 * record_midstep_output_state integrated to this time. Called before and after the write (each call swaps). */
void ketju_swap_output_velocities(void)
{
#ifdef KETJU_TYPE4_STARS
    static std::vector<ketju_output_state> here; /* this output's states, kept between the two calls */
    static int swapped = 0;
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
            for(int j = 0; j < 3; j++) {double v = P[i].Vel[j]; P[i].Vel[j] = P[i].KetjuTrueVel[j]; P[i].KetjuTrueVel[j] = v;}
            continue;
        }
        auto it = by_id.find({P[i].ID, P[i].ID_child_number}); if(it == by_id.end()) continue;
        ketju_output_state &o = here[it->second];
        for(int j = 0; j < 3; j++) {double x = P[i].Pos[j], v = P[i].Vel[j]; P[i].Pos[j] = o.pos[j]; P[i].Vel[j] = o.vel[j]; o.pos[j] = x; o.vel[j] = v;}
    }
    swapped = !swapped;
    if(!swapped) here.clear();
#endif
}

void ketju_set_final_velocities(void)
{
    KjClock clk(CPU_KETJUFINISH);
    /* called after drift: swap in true physical velocities and correct dp[] momentum.
     * The velocity trick set Vel = delta_pos/dt_drift for the drift.
     * Now we replace it with the true MSTAR velocity and correct dp[].
     * Only for members whose step ends at this sync point (their bin is active and they were just
     * drifted to it): a region on a longer bin is still drifting along its chord. */
    for(int i = 0; i < NumPart; i++) {
        if(P[i].KetjuIntegrated && TimeBinActive[P[i].TimeBin]) {
#ifdef KETJU_TYPE4_STARS
            { /* land exactly on the MSTAR end position the negative half-kick used (the drift only reaches it to roundoff) */
                double d2 = 0; for(int j = 0; j < 3; j++) {double d = P[i].Pos[j] - P[i].KetjuTargetPos[j]; d2 += d * d;}
                if(d2 < 1e-12 * All.KetjuRegionRadius * All.KetjuRegionRadius) {for(int j = 0; j < 3; j++) {P[i].Pos[j] = P[i].KetjuTargetPos[j];}}
            }
#endif
            for(int j = 0; j < 3; j++) {
                /* dp correction: the difference between true velocity and trick velocity,
                 * converted to momentum. This ensures the kick integrator is consistent. */
                double dv = P[i].KetjuFinalVel[j] - P[i].Vel[j];
                P[i].dp[j] += P[i].Mass * dv;
                P[i].Vel[j] = P[i].KetjuFinalVel[j];
            }
        }
    }
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
#ifdef KETJU_TYPE4_STARS
/* record the softening/zeta the gravity call just used (AGS updates them before the next one) */
void ketju_snapshot_tree_softening(void)
{
    for(int i : ActiveParticleList) {
        if(P[i].Type != 4 && P[i].Type != 5) continue;
        double s2[2], z2[2]; tree_pair_softening(i, s2, z2);
        for(int k = 0; k < 2; k++) {P[i].KetjuTreeSoft[k] = s2[k]; P[i].KetjuTreeZeta[k] = z2[k];}
    }
}
#endif

void ketju_compute_com_quantities(void)
{
    struct ChainMemberData {
        MyIDType chain;
        double mass;
        double pos[3], vel[3]; /* vel: peculiar, after the second half-kick */
#ifdef KETJU_TYPE4_STARS
        int type;
        double soft[2], zeta[2]; /* tree softening/zeta of the gravity call that just ran */
#endif
    };

    std::vector<ChainMemberData> local_data;
    std::vector<int> local_index;
    for(int i : ActiveParticleList) {
        if(!P[i].KetjuIntegrated || P[i].KetjuChainID == 0 || P[i].Mass <= 0) continue;
        ChainMemberData d;
        d.chain = P[i].KetjuChainID; d.mass = P[i].Mass;
        for(int j = 0; j < 3; j++) {d.pos[j] = P[i].Pos[j]; d.vel[j] = P[i].Vel[j] / All.cf_atime;}
#ifdef KETJU_TYPE4_STARS
        d.type = P[i].Type; for(int k = 0; k < 2; k++) {d.soft[k] = P[i].KetjuTreeSoft[k]; d.zeta[k] = P[i].KetjuTreeZeta[k];}
#endif
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
    std::vector<double> ext_kick(3 * n_local, 0.0);

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
#ifdef KETJU_TYPE4_STARS
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
#else
            if(r2 >= All.KetjuRegionRadius * All.KetjuRegionRadius) continue; /* may leave the chain next step: keep it, as the tree walk does for dt_2body */
#endif
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
        for(int j = 0; j < 3; j++) P[idx].COM_GravAccel[j] = P[idx].GravAccel[j] - chain_grav[j];
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
        SymmetricTensor2<MyFloat> external_tidal = P[idx].tidal_tensorps - chain_tidal;
        double tidal_norm = external_tidal.frobenius_norm();
        P[idx].COM_dt_tidal = (tidal_norm > 0) ? sqrt(1.0 / tidal_norm) : MAX_REAL_NUMBER;
#else
        P[idx].COM_dt_tidal = MAX_REAL_NUMBER; (void)chain_tidal; /* no tree tidal tensor: no tidal limit, but marks the external values as set */
#endif
        { /* the host's second half-kick by the field external to the whole chain (peculiar velocity) */
            integertime ts = P[idx].integertime_step();
            double gk = get_gravkick_factor(P[idx].Ti_begstep + ts / 2, P[idx].Ti_begstep + ts, idx, 0);
            for(int j = 0; j < 3; j++) ext_kick[3*n + j] = (P[idx].GravAccel[j] - chain_grav_all[j]) * gk / All.cf_atime;
        }
    }
    { /* energy the host's end kick put into each chain's internal motion: external kicks relative to their mass-weighted mean,
       * against the CoM-relative velocities after the kick; counted once per chain, by the task of its first gathered member */
        struct KickData {MyIDType chain; int task; double mass, vel[3], dv[3];};
        std::vector<KickData> kl(n_local);
        for(int n = 0; n < n_local; n++) {
            kl[n].chain = local_data[n].chain; kl[n].task = ThisTask; kl[n].mass = local_data[n].mass;
            for(int j = 0; j < 3; j++) {kl[n].vel[j] = local_data[n].vel[j]; kl[n].dv[j] = ext_kick[3*n + j];}
        }
        int nt = 0; for(int t = 0; t < NTask; t++) {byte_counts[t] = counts[t] * sizeof(KickData); byte_displs[t] = nt * sizeof(KickData); nt += counts[t];}
        std::vector<KickData> ka(nt);
        MPI_Allgatherv(kl.data(), n_local * sizeof(KickData), MPI_BYTE, ka.data(), byte_counts.data(), byte_displs.data(), MPI_BYTE, MPI_COMM_WORLD);
        std::stable_sort(ka.begin(), ka.end(), [](const KickData &a, const KickData &b) {return a.chain < b.chain;});
        for(size_t lo = 0; lo < ka.size();) {
            size_t hi = lo; while(hi < ka.size() && ka[hi].chain == ka[lo].chain) hi++;
            if(ka[lo].task == ThisTask) {
                double M = 0, vb[3] = {0, 0, 0}, db[3] = {0, 0, 0}, W = 0;
                for(size_t q = lo; q < hi; q++) {M += ka[q].mass; for(int j = 0; j < 3; j++) {vb[j] += ka[q].mass * ka[q].vel[j]; db[j] += ka[q].mass * ka[q].dv[j];}}
                for(size_t q = lo; q < hi; q++) for(int j = 0; j < 3; j++) {
                    double v = ka[q].vel[j] - vb[j] / M, d = ka[q].dv[j] - db[j] / M; W += 0.5 * ka[q].mass * (v * v - (v - d) * (v - d));
                }
                KetjuEnergyLedger[0] += W; KetjuEnergyLedger[1] += fabs(W);
            }
            lo = hi;
        }
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
    AllKetjuParticleIndices.clear();
}

int ketju_is_particle_in_region(int i)
{
    return (AllKetjuParticleIndices.find(i) != AllKetjuParticleIndices.end()) ? 1 : 0;
}

void ketju_mark_regions_stale(void) {} /* regions are rebuilt every step; kept for the domain/merge-split call sites */

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
