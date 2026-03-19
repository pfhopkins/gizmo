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
#include <functional>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "ketju_coupling.h"

#ifdef KETJU_REGULARIZATION

/* Cost tracking for load balancing (keyed by first center ID per region) */
static std::unordered_map<MyIDType, double> region_previous_cost;

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
    short int is_dead_remnant; /* 1 = dead Type 4 star (NS/BH), 0 = living star or sink */
#ifdef SINK_PARTICLES
    short int SinkSubType;
    double Spin[3];
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
    std::set<int> local_member_indices;            /* local P[] indices in this region */
    int total_particle_count;
    int num_pn_particles;                          /* count of PN-enabled (SMBH) particles */

    /* MPI task groups for parallel computation */
    KetjuTaskGroup affected_tasks;  /* tasks that hold particles in this region */
    KetjuTaskGroup compute_tasks;   /* tasks that run the integrator */
    region_compute_info compute_info;

    /* integrator state (only allocated on compute root) */
    struct ketju_system *integrator;
    std::vector<ketju_extra_data> extra_data;

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
static std::set<int> AllKetjuParticleIndices; /* local indices of all particles in any region */

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

static void allocate_compute_tasks_for_regions(void)
{
    int n_regions = ActiveRegions.size();
    if(n_regions == 0) return;

    /* collect costs and sort regions by cost (most expensive first) */
    typedef std::pair<double, int> cost_idx;
    std::priority_queue<cost_idx> cost_pq;
    for(int r = 0; r < n_regions; r++) {
        cost_pq.push({region_cost_estimate(r), r});
    }

    /* greedy allocation: assign tasks proportional to cost */
    double total_cost = 0;
    for(int r = 0; r < n_regions; r++) total_cost += region_cost_estimate(r);
    if(total_cost <= 0) total_cost = 1.0;

    int tasks_used = 0;
    std::vector<cost_idx> sorted_regions;
    while(!cost_pq.empty()) { sorted_regions.push_back(cost_pq.top()); cost_pq.pop(); }

    for(size_t i = 0; i < sorted_regions.size(); i++) {
        int r = sorted_regions[i].second;
        double frac = sorted_regions[i].first / total_cost;
        int n_tasks = DMAX(1, (int)(frac * NTask + 0.5));

        /* ensure minimum particles per task */
        int min_per_task = (int)DMAX(All.KetjuMinParticlesPerTask, 2);
        if(ActiveRegions[r].total_particle_count > 0) {
            int max_tasks = ActiveRegions[r].total_particle_count / min_per_task;
            if(max_tasks < 1) max_tasks = 1;
            if(n_tasks > max_tasks) n_tasks = max_tasks;
        }

        /* don't exceed available tasks */
        if(tasks_used + n_tasks > NTask) n_tasks = NTask - tasks_used;
        if(n_tasks < 1) n_tasks = 1;

        ActiveRegions[r].compute_info.first_task_index = tasks_used;
        ActiveRegions[r].compute_info.final_task_index = tasks_used + n_tasks - 1;
        ActiveRegions[r].compute_info.compute_sequence_position = 0; /* all parallel for now */
        tasks_used += n_tasks;
    }
}

static void update_region_costs(void)
{
    /* after integration, store actual cost for next step */
    for(size_t r = 0; r < ActiveRegions.size(); r++) {
        if(!ActiveRegions[r].compute_tasks.is_root()) continue;
        if(!ActiveRegions[r].integrator) continue;
        if(ActiveRegions[r].centers.empty()) continue;

        MyIDType key = ActiveRegions[r].centers[0].ID;
        double cost = (double)(ActiveRegions[r].integrator->perf->successful_steps +
                               ActiveRegions[r].integrator->perf->failed_steps) *
                      ActiveRegions[r].total_particle_count * ActiveRegions[r].total_particle_count;
        region_previous_cost[key] = cost;
    }
}

/* ============================================================
 *  Helper: is this particle a valid chain center?
 * ============================================================ */
static int is_chain_center(int i)
{
    /* Type 4 stars above mass threshold */
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

/* ============================================================
 *  Helper: is this particle PN-enabled? (SMBH only)
 * ============================================================ */
/* Helper: is this a dead remnant (NS or BH) still stored as Type 4? */
static int is_dead_remnant_type4(int i)
{
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
    if(P[i].Type == 4 && P[i].MstarSampleIMF[0] <= 0 && P[i].Mass > 0) return 1;
#endif
    return 0;
}

static int is_pn_particle(ketju_mpi_particle *p)
{
#ifdef SINK_PARTICLES
    if(p->Type == 5 && p->SinkSubType == 0) return 1; /* SMBH: always PN */
#ifdef KETJU_PN_COMPACT_OBJECTS
    if(p->Type == 5 && p->SinkSubType == 1) return 1; /* stellar-mass BH (promoted): PN */
#endif
#endif
#ifdef KETJU_PN_COMPACT_OBJECTS
    /* Dead Type 4 compact object remnants (NS or BH, no BH_PROMOTION) */
    if(p->Type == 4 && p->is_dead_remnant) return 1;
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
 *  Helper: move a particle between timebin linked lists
 *  (same logic as find_timesteps in timestep.cc)
 * ============================================================ */
static void move_particle_timebin(int i, int old_bin, int new_bin)
{
    if(old_bin == new_bin) return;

    /* remove from old bin */
    TimeBinCount[old_bin]--;
    if(PrevInTimeBin[i] >= 0)
        NextInTimeBin[PrevInTimeBin[i]] = NextInTimeBin[i];
    else
        FirstInTimeBin[old_bin] = NextInTimeBin[i];
    if(NextInTimeBin[i] >= 0)
        PrevInTimeBin[NextInTimeBin[i]] = PrevInTimeBin[i];
    else
        LastInTimeBin[old_bin] = PrevInTimeBin[i];

    /* add to new bin */
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
            local_centers[k].is_dead_remnant = is_dead_remnant_type4(i);
            for(int j = 0; j < 3; j++) {
                local_centers[k].Pos[j] = P[i].Pos[j];
                local_centers[k].Vel[j] = P[i].Vel[j];
            }
#ifdef SINK_PARTICLES
            local_centers[k].SinkSubType = (P[i].Type == 5) ? P[i].SinkSubType : -1;
            for(int j = 0; j < 3; j++) local_centers[k].Spin[j] = (P[i].Type == 5) ? P[i].KetjuSpin[j] : 0;
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

/* Find local particles within r_region of any center in a region */
static std::set<int> find_local_members(const std::vector<ketju_mpi_particle> &centers, double radius)
{
    std::set<int> members;
    for(int i = 0; i < NumPart; i++) {
        if(!is_chain_eligible(i)) continue;
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
    }

    /* create compute_tasks communicator from allocated task range.
     * For now, use affected tasks as compute tasks (single-group mode).
     * Phase B (cost-based scheduling) will set compute_info from the allocator. */
    {
        std::vector<int> compute_indices;
        int first = reg.compute_info.first_task_index;
        int last  = reg.compute_info.final_task_index;
        if(first == 0 && last == 0) {
            /* no allocation yet — use affected tasks as compute group */
            for(int t = 0; t < NTask; t++) { if(counts[t] > 0) compute_indices.push_back(t); }
        } else {
            for(int t = first; t <= last; t++) compute_indices.push_back(t);
        }
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
        mp.is_dead_remnant = is_dead_remnant_type4(idx);
        for(int j = 0; j < 3; j++) {
            mp.Pos[j] = P[idx].Pos[j];
            mp.Vel[j] = P[idx].Vel[j];
        }
#ifdef SINK_PARTICLES
        mp.SinkSubType = (P[idx].Type == 5) ? P[idx].SinkSubType : -1;
        for(int j = 0; j < 3; j++) mp.Spin[j] = (P[idx].Type == 5) ? P[idx].KetjuSpin[j] : 0;
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
#endif
#ifdef KETJU_PN_COMPACT_OBJECTS
                if(a.Type == 4 && a.is_dead_remnant) a_pn = 1;
                if(b.Type == 4 && b.is_dead_remnant) b_pn = 1;
                if(a.Type == 5 && a.SinkSubType == 1) a_pn = 1;
                if(b.Type == 5 && b.SinkSubType == 1) b_pn = 1;
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
        reg.integrator->constants->G = All.G;
#if defined(C_LIGHT_CODE)
        reg.integrator->constants->c = C_LIGHT_CODE;
#else
        reg.integrator->constants->c = 2.9979e10 / All.UnitVelocity_in_cm_per_s;
#endif

        /* set options */
        reg.integrator->options->PN_flags = parse_pn_terms();
        reg.integrator->options->gbs_relative_tolerance = All.KetjuIntegrationTolerance;
        reg.integrator->options->enable_bh_mergers = (All.KetjuEnableBHMergerKicks >= 0);
        reg.integrator->options->enable_bh_merger_kicks = All.KetjuEnableBHMergerKicks;
        if(All.KetjuMaxStepCount > 0) reg.integrator->options->max_step_count = All.KetjuMaxStepCount;
        if(All.KetjuUseStarStarSoftening) {
            double h = All.ForceSoftening[4] * All.cf_atime;
#ifdef SINK_PARTICLES
            if(All.ForceSoftening[5] > 0) h = DMIN(h, All.ForceSoftening[5] * All.cf_atime);
#endif
            reg.integrator->options->star_star_softening = h;
        }

        /* fill particle data (relative to CoM) */
        reg.extra_data.resize(reg.total_particle_count);
        struct ketju_system_physical_state *ps = reg.integrator->physical_state;
        ps->time = 0;
        for(int i = 0; i < reg.total_particle_count; i++) {
            ps->mass[i] = reg.all_particles[i].Mass;
            for(int j = 0; j < 3; j++) {
                ps->pos[i][j] = reg.all_particles[i].Pos[j] - reg.com_pos[j];
                ps->vel[i][j] = reg.all_particles[i].Vel[j] - reg.com_vel[j];
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
    }

    /* broadcast CoM data to all tasks that need it (affected group for scatter) */
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
static void do_negative_halfstep_kick(KetjuRegion &reg, double kick_factor)
{
    if(!reg.compute_tasks.is_member() || !reg.integrator) return;

    int n = reg.integrator->num_particles;
    struct ketju_system_physical_state *ps = reg.integrator->physical_state;

    /* softening: use the star softening (comoving -> physical) */
    double h = All.ForceSoftening[4] * All.cf_atime;
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
            for(int k = 0; k < 3; k++) {
                dr[k] = ps->pos[i][k] - ps->pos[j][k];
                r2 += dr[k] * dr[k];
            }
            double r = sqrt(r2);
            if(r == 0) continue;

            double fac = kick_factor * All.G * softened_force_factor(r, h);
            for(int k = 0; k < 3; k++) {
                dv[3*i + k] += ps->mass[j] * fac * dr[k];
                dv[3*j + k] -= ps->mass[i] * fac * dr[k];
            }
        }
    }

    /* collect results across compute tasks and apply */
    MPI_Allreduce(MPI_IN_PLACE, dv.data(), 3 * n, MPI_DOUBLE, MPI_SUM, reg.compute_tasks.comm);

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
    do_negative_halfstep_kick(reg, first_halfkick);

    /* expand tight binaries before integration to prevent stalling */
    expand_tight_binaries(reg, dt_physical);

    /* run MSTAR integrator (all compute tasks participate via compute_tasks.comm) */
    if(reg.compute_tasks.is_member() && reg.integrator) {
        ketju_run_integrator(reg.integrator, dt_physical);

        int n_steps = reg.integrator->perf->successful_steps + reg.integrator->perf->failed_steps;
        double dE = reg.integrator->perf->relative_energy_error;

        if(reg.compute_tasks.is_root())
        printf("KETJU [task %d]: Integrated %d particles for dt=%g, %d steps (%d failed), dE/E=%g\n",
               ThisTask, reg.total_particle_count, dt_physical,
               reg.integrator->perf->successful_steps,
               reg.integrator->perf->failed_steps, dE);

        if(All.KetjuMaxStepCount > 0 && n_steps >= All.KetjuMaxStepCount) {
            printf("KETJU WARNING: integration hit max step count (%d) — results may be inaccurate!\n",
                   All.KetjuMaxStepCount);
        }
        if(fabs(dE) > 1e-4) {
            printf("KETJU WARNING: large energy error dE/E=%g in region with %d particles\n",
                   dE, reg.total_particle_count);
        }
    }

    /* subtract second half of tree force */
    do_negative_halfstep_kick(reg, second_halfkick);
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

#if defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH)
static void handle_mergers(KetjuRegion &reg,
                           const std::vector<double> &final_abs_pos,
                           const std::vector<double> &final_mass,
                           const std::vector<double> &original_mass)
{
    int n = reg.total_particle_count;
    if(n < 2) return;

    /* scan for merged particles (mass <= 0 after integration) */
    for(int i = 0; i < n; i++) {
        if(final_mass[i] > 0) continue;

        int type_merged = reg.extra_data[i].Type;

        /* find nearest surviving particle — the actual merger partner */
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

        /* check type-specific merger flags */
        int merge_allowed = 0;
#ifdef KETJU_MERGE_BH
        if(type_merged == 5 || type_survivor == 5) merge_allowed = 1; /* any BH involvement */
#endif
#ifdef KETJU_MERGE_STARS
        if(type_merged == 4 && type_survivor == 4) merge_allowed = 1; /* both stars */
#endif
        if(!merge_allowed) continue;

        /* mass bookkeeping */
        double M_merged = original_mass[i];
        double M_survivor_old = original_mass[survivor];
        double M_survivor_new = final_mass[survivor];
        if(M_survivor_new <= 0 || M_merged <= 0) continue;

        int task_merged = reg.extra_data[i].Task;
        int task_survivor = reg.extra_data[survivor].Task;
        int idx_merged = reg.extra_data[i].Index;
        int idx_survivor = reg.extra_data[survivor].Index;

        /* transfer metals: gather from merged particle, mix into survivor */
        int n_metals = 0, n_elements = 0;
#ifdef METALS
        n_metals = NUM_METAL_SPECIES;
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
        n_elements = NUM_RESOLVEDISM_ELEMENTS;
#endif
        int buf_size = n_metals + n_elements;

        if(buf_size > 0) {
            std::vector<double> merged_metals(buf_size, 0.0);

            /* merged particle's task fills in its metals */
            if(task_merged == ThisTask) {
#ifdef METALS
                for(int k = 0; k < NUM_METAL_SPECIES; k++)
                    merged_metals[k] = P[idx_merged].Metallicity[k];
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
                for(int k = 0; k < NUM_RESOLVEDISM_ELEMENTS; k++)
                    merged_metals[n_metals + k] = P[idx_merged].ElementAbundance[k];
#endif
            }

            /* broadcast metals from merged particle's task (collective — all tasks participate) */
            MPI_Bcast(merged_metals.data(), buf_size, MPI_DOUBLE, task_merged, MPI_COMM_WORLD);

            /* survivor's task applies mass-weighted mixing */
            if(task_survivor == ThisTask) {
#ifdef METALS
                for(int k = 0; k < NUM_METAL_SPECIES; k++)
                    P[idx_survivor].Metallicity[k] =
                        (M_survivor_old * P[idx_survivor].Metallicity[k] +
                         M_merged * merged_metals[k]) / M_survivor_new;
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
                for(int k = 0; k < NUM_RESOLVEDISM_ELEMENTS; k++)
                    P[idx_survivor].ElementAbundance[k] =
                        (M_survivor_old * P[idx_survivor].ElementAbundance[k] +
                         M_merged * merged_metals[n_metals + k]) / M_survivor_new;
#endif
            }
        }

        /* mark merged particle for removal (cleaned up by rearrange_particle_sequence) */
        if(task_merged == ThisTask) {
            P[idx_merged].Mass = 0;
        }

        /* log the merger event (task 0 only, since all tasks have the same data) */
        if(ThisTask == 0) {
            printf("KETJU MERGER: ID %llu (Type %d, M=%g) merged into ID %llu (Type %d, M=%g->%g)\n",
                   (unsigned long long)reg.extra_data[i].ID, type_merged, M_merged,
                   (unsigned long long)reg.extra_data[survivor].ID, type_survivor,
                   M_survivor_old, M_survivor_new);
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

static void scatter_results(KetjuRegion &reg)
{
    int n = reg.total_particle_count;
    if(!reg.affected_tasks.is_member()) return; /* only affected tasks participate */
    MPI_Bcast(&n, 1, MPI_INT, reg.affected_tasks.root, reg.affected_tasks.comm);
    if(n < 2) return;

    /* prepare final positions (absolute) and velocities (CoM-relative) on compute root */
    std::vector<double> final_abs_pos(3 * n), final_rel_vel(3 * n);
    std::vector<double> final_mass(n);
#if defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH)
    std::vector<double> original_mass(n);
#endif
#ifdef SINK_PARTICLES
    int n_pn = 0;
    std::vector<double> final_spin; /* only for PN particles */
#endif
    if(reg.compute_tasks.is_root() && reg.integrator) {
        struct ketju_system_physical_state *ps = reg.integrator->physical_state;
#ifdef SINK_PARTICLES
        n_pn = reg.num_pn_particles;
        final_spin.resize(3 * n_pn, 0.0);
        for(int i = 0; i < n_pn; i++) {
            for(int j = 0; j < 3; j++) final_spin[3*i + j] = ps->spin[i][j];
        }
#endif
        for(int i = 0; i < n; i++) {
            final_mass[i] = ps->mass[i];
#if defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH)
            original_mass[i] = reg.all_particles[i].Mass;
#endif
            for(int j = 0; j < 3; j++) {
                final_abs_pos[3*i + j] = reg.com_pos[j] + ps->pos[i][j];
                final_rel_vel[3*i + j] = ps->vel[i][j];
            }
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

    /* broadcast results to all affected tasks */
    if(!reg.affected_tasks.is_root()) {
        reg.extra_data.resize(n);
    }
    MPI_Bcast(reg.extra_data.data(), n * sizeof(ketju_extra_data), MPI_BYTE,
              reg.affected_tasks.root, reg.affected_tasks.comm);
    MPI_Bcast(final_abs_pos.data(), 3 * n, MPI_DOUBLE, reg.affected_tasks.root, reg.affected_tasks.comm);
    MPI_Bcast(final_rel_vel.data(), 3 * n, MPI_DOUBLE, reg.affected_tasks.root, reg.affected_tasks.comm);
    MPI_Bcast(final_mass.data(), n, MPI_DOUBLE, reg.affected_tasks.root, reg.affected_tasks.comm);
#if defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH)
    MPI_Bcast(original_mass.data(), n, MPI_DOUBLE, reg.affected_tasks.root, reg.affected_tasks.comm);
#endif
#ifdef SINK_PARTICLES
    MPI_Bcast(&n_pn, 1, MPI_INT, reg.affected_tasks.root, reg.affected_tasks.comm);
    if(!reg.affected_tasks.is_root()) final_spin.resize(3 * n_pn, 0.0);
    if(n_pn > 0) MPI_Bcast(final_spin.data(), 3 * n_pn, MPI_DOUBLE, reg.affected_tasks.root, reg.affected_tasks.comm);
#endif

    /* apply velocity trick to local particles */
    for(int i = 0; i < n; i++) {
        if(reg.extra_data[i].Task != ThisTask) continue;
        int idx = reg.extra_data[i].Index;

        /* sanity check */
        if(P[idx].ID != reg.extra_data[i].ID) {
            printf("KETJU ERROR: particle index mismatch! ID %llu vs %llu on task %d\n",
                   (unsigned long long)P[idx].ID, (unsigned long long)reg.extra_data[i].ID, ThisTask);
            endrun(667);
        }

        /* update mass */
        P[idx].Mass = final_mass[i];

        /* skip velocity trick for merged particles (mass=0, will be removed) */
        if(final_mass[i] <= 0) {
            P[idx].KetjuIntegrated = 1;
            continue;
        }

        /* compute per-particle drift factor: what drift_particle() will use */
        double dt_drift = get_drift_factor(P[idx].Ti_current, P[idx].Ti_current + reg.ti_step, idx, 0);

        if(dt_drift > 0) {
            for(int j = 0; j < 3; j++) {
                /* desired final position = final_abs_pos + CoM displacement during drift */
                double desired_pos = final_abs_pos[3*i + j] + reg.com_vel[j] * dt_drift;
                double delta_pos = desired_pos - P[idx].Pos[j];
#ifdef BOX_PERIODIC
                double box = (j == 0) ? boxSize_X : (j == 1) ? boxSize_Y : boxSize_Z;
                if(box > 0) { while(delta_pos > 0.5 * box) delta_pos -= box; while(delta_pos < -0.5 * box) delta_pos += box; }
#endif
                P[idx].Vel[j] = delta_pos / dt_drift;
                P[idx].KetjuFinalVel[j] = final_rel_vel[3*i + j] + reg.com_vel[j];
            }
        } else {
            /* no drift — set position and velocity directly */
            for(int j = 0; j < 3; j++) {
                P[idx].Pos[j] = final_abs_pos[3*i + j];
                P[idx].Vel[j] = final_rel_vel[3*i + j] + reg.com_vel[j];
                P[idx].KetjuFinalVel[j] = P[idx].Vel[j];
            }
        }

#ifdef SINK_PARTICLES
        /* update spin for PN particles (first n_pn in integrator ordering) */
        if(i < n_pn && P[idx].Type == 5) {
            for(int j = 0; j < 3; j++) P[idx].KetjuSpin[j] = final_spin[3*i + j];
        }
#endif
        P[idx].KetjuIntegrated = 1;
    }

#ifdef KETJU_MERGE_STARS
    /* Stellar collision check: merge living Type 4 pairs closer than R1+R2 (physical radii from tables).
     * Dead remnants (NS/BH) are point masses — skip collision check for those. */
#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
    for(int i = 0; i < n; i++) {
        if(final_mass[i] <= 0 || reg.extra_data[i].Type != 4) continue;
        if(reg.all_particles[i].is_dead_remnant) continue; /* NS/BH — no physical radius */
        for(int j = i + 1; j < n; j++) {
            if(final_mass[j] <= 0 || reg.extra_data[j].Type != 4) continue;
            if(reg.all_particles[j].is_dead_remnant) continue;
            double d2 = 0;
            for(int k = 0; k < 3; k++) {
                double dx = final_abs_pos[3*i + k] - final_abs_pos[3*j + k];
                d2 += dx * dx;
            }
            double sep_cm = sqrt(d2) * UNIT_LENGTH_IN_CGS * All.cf_atime;
            /* get stellar radii from tables at current age */
            int idx_i = reg.extra_data[i].Index, idx_j = reg.extra_data[j].Index;
            int task_i = reg.extra_data[i].Task, task_j = reg.extra_data[j].Task;
            double logMi = 0, logMj = 0, logZi = -2.7, logZj = -2.7, log_age_i = 2.0, log_age_j = 2.0;
            if(task_i == ThisTask) {
                logMi = log10(DMAX(P[idx_i].MstarSampleIMF[0], 0.08));
                logZi = log10(DMAX(P[idx_i].BirthMetallicity, 1e-10));
                double age_yr = evaluate_stellar_age_Gyr(idx_i) * 1e9;
                log_age_i = log10(DMAX(age_yr, 100.0));
            }
            if(task_j == ThisTask) {
                logMj = log10(DMAX(P[idx_j].MstarSampleIMF[0], 0.08));
                logZj = log10(DMAX(P[idx_j].BirthMetallicity, 1e-10));
                double age_yr = evaluate_stellar_age_Gyr(idx_j) * 1e9;
                log_age_j = log10(DMAX(age_yr, 100.0));
            }
            MPI_Bcast(&logMi, 1, MPI_DOUBLE, task_i, MPI_COMM_WORLD);
            MPI_Bcast(&logZi, 1, MPI_DOUBLE, task_i, MPI_COMM_WORLD);
            MPI_Bcast(&log_age_i, 1, MPI_DOUBLE, task_i, MPI_COMM_WORLD);
            MPI_Bcast(&logMj, 1, MPI_DOUBLE, task_j, MPI_COMM_WORLD);
            MPI_Bcast(&logZj, 1, MPI_DOUBLE, task_j, MPI_COMM_WORLD);
            MPI_Bcast(&log_age_j, 1, MPI_DOUBLE, task_j, MPI_COMM_WORLD);
            double Ri_cm = pow(10., stellar_log_R_cm(logMi, logZi, log_age_i));
            double Rj_cm = pow(10., stellar_log_R_cm(logMj, logZj, log_age_j));
            if(sep_cm < Ri_cm + Rj_cm) {
                /* collision: absorb less massive into more massive */
                int victim = (final_mass[i] < final_mass[j]) ? i : j;
                int surv = (victim == i) ? j : i;
                final_mass[surv] += final_mass[victim];
                final_mass[victim] = 0;
                if(ThisTask == 0)
                    printf("KETJU STELLAR COLLISION: ID %llu (%.1f Msun) + ID %llu (%.1f Msun) at sep=%.2e cm (R1+R2=%.2e cm)\n",
                        (unsigned long long)reg.extra_data[i].ID, Mi,
                        (unsigned long long)reg.extra_data[j].ID, Mj,
                        sep_cm, Ri_cm + Rj_cm);
            }
        }
    }
#endif
#endif

#if defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH)
    handle_mergers(reg, final_abs_pos, final_mass, original_mass);
#endif
}

/* ============================================================
 *  PUBLIC INTERFACE FUNCTIONS
 * ============================================================ */

/* PHASE 6: Force all chain particles to the same (minimum) timebin.
 * Called after find_timesteps(), before the first half-kick. */
void ketju_limit_timesteps(void)
{
    if(All.KetjuRegionRadius <= 0) return;
    if(All.KetjuMinBHMass <= 0 && All.KetjuMinStarMass <= 0) return;

    /* gather chain center positions */
    std::vector<ketju_mpi_particle> centers = gather_chain_centers();
    if(centers.empty()) return;

    /* find local particles near any center */
    double radius = All.KetjuRegionRadius;
    std::vector<int> local_chain;
    for(int i = 0; i < NumPart; i++) {
        if(!is_chain_eligible(i)) continue;
        for(size_t c = 0; c < centers.size(); c++) {
            if(particle_distance(P[i].Pos, const_cast<double*>(centers[c].Pos)) < radius) {
                local_chain.push_back(i);
                break;
            }
        }
    }

    /* find local minimum active timebin among chain particles */
    int local_min_bin = TIMEBINS;
    for(size_t k = 0; k < local_chain.size(); k++) {
        int idx = local_chain[k];
        if(P[idx].TimeBin > 0 && TimeBinActive[P[idx].TimeBin] && P[idx].TimeBin < local_min_bin)
            local_min_bin = P[idx].TimeBin;
    }

    /* global minimum (all tasks participate) */
    int global_min_bin;
    MPI_Allreduce(&local_min_bin, &global_min_bin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(global_min_bin >= TIMEBINS || global_min_bin <= 0) return;

    /* force all local chain particles to global_min_bin if they have a larger timebin */
    int n_moved_local = 0;
    for(size_t k = 0; k < local_chain.size(); k++) {
        int idx = local_chain[k];
        if(P[idx].TimeBin > global_min_bin) {
            move_particle_timebin(idx, P[idx].TimeBin, global_min_bin);
            n_moved_local++;
        }
    }

    int n_moved_global;
    MPI_Reduce(&n_moved_local, &n_moved_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask == 0 && n_moved_global > 0) {
        printf("KETJU: Forced %d particles to timebin %d (dt=%g)\n",
               n_moved_global, global_min_bin,
               GET_INTEGERTIME_FROM_TIMEBIN(global_min_bin) * All.Timebase_interval / All.cf_hubble_a);
    }
}

void ketju_find_regions(void)
{
    /* clear previous state — KetjuIntegrated is cleared here (not in finish_step)
     * so the flag survives across the step boundary for guard checks in kicks/predict/gravtree */
    for(int i = 0; i < NumPart; i++) {P[i].KetjuIntegrated = 0;}
    ActiveRegions.clear();
    AllKetjuParticleIndices.clear();

    /* validate parameters */
    if(All.KetjuRegionRadius <= 0) return;
    if(All.KetjuMinBHMass <= 0 && All.KetjuMinStarMass <= 0) return;

    /* gather all chain centers across MPI */
    std::vector<ketju_mpi_particle> centers = gather_chain_centers();
    if(centers.empty()) return;

    /* merge overlapping regions */
    double merge_radius = 2.0 * All.KetjuRegionRadius;
    std::vector<std::vector<ketju_mpi_particle>> region_centers = merge_overlapping_regions(centers, merge_radius);

    /* build regions (first pass: find members and count particles) */
    for(size_t r = 0; r < region_centers.size(); r++) {
        KetjuRegion reg;
        reg.centers = region_centers[r];
        reg.local_member_indices = find_local_members(reg.centers, All.KetjuRegionRadius);

        /* count total particles for cost estimation */
        int n_local = reg.local_member_indices.size();
        MPI_Allreduce(&n_local, &reg.total_particle_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        /* track all local KETJU particles for guard checks */
        for(int idx : reg.local_member_indices) {
            AllKetjuParticleIndices.insert(idx);
        }

        ActiveRegions.push_back(std::move(reg));
    }

    /* allocate compute tasks across regions based on cost estimates */
    allocate_compute_tasks_for_regions();

    /* second pass: set up integrators with allocated task groups */
    for(size_t r = 0; r < ActiveRegions.size(); r++) {
        setup_integrator(ActiveRegions[r]);
    }

    if(ThisTask == 0 && !ActiveRegions.empty()) {
        printf("KETJU: Found %d region(s) from %d center(s)\n",
               (int)ActiveRegions.size(), (int)centers.size());
    }
}

void ketju_run_integration(void)
{
    if(ActiveRegions.empty()) return;

    for(size_t r = 0; r < ActiveRegions.size(); r++) {
        KetjuRegion &reg = ActiveRegions[r];
        if(reg.total_particle_count < 2) continue;

        /* find the minimum integer timestep among local chain members */
        integertime local_min_ti = TIMEBASE;
        for(int idx : reg.local_member_indices) {
            integertime ti_i = GET_PARTICLE_INTEGERTIME(idx);
            if(ti_i > 0 && ti_i < local_min_ti) local_min_ti = ti_i;
        }

        /* global minimum across all tasks */
        integertime global_min_ti;
        MPI_Allreduce(&local_min_ti, &global_min_ti, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
        if(global_min_ti <= 0 || global_min_ti > TIMEBASE) continue;

        reg.ti_step = global_min_ti;

        /* convert to physical time for MSTAR integrator */
        double dt_physical;
        if(All.ComovingIntegrationOn) {
            dt_physical = reg.ti_step * All.Timebase_interval / All.cf_hubble_a;
        } else {
            dt_physical = reg.ti_step * All.Timebase_interval;
        }

        if(dt_physical <= 0 || !isfinite(dt_physical)) continue;

        /* integrate and scatter results */
        integrate_region(reg, dt_physical);
        scatter_results(reg);
    }

    /* update cost estimates for next step's load balancing */
    update_region_costs();
}

void ketju_set_final_velocities(void)
{
    /* called after drift: swap in true physical velocities */
    for(int i = 0; i < NumPart; i++) {
        if(P[i].KetjuIntegrated) {
            for(int j = 0; j < 3; j++) {
                P[i].Vel[j] = P[i].KetjuFinalVel[j];
            }
        }
    }
}

void ketju_finish_step(void)
{
    /* NOTE: KetjuIntegrated flags are NOT cleared here — they persist until
     * ketju_find_regions() at the start of the next step, so that guard checks
     * in kicks.cc/predict.cc/gravtree.cc can see which particles were KETJU-integrated */

    /* free region data (particle flags survive) */
    ActiveRegions.clear();
    AllKetjuParticleIndices.clear();
}

int ketju_is_particle_in_region(int i)
{
    return (AllKetjuParticleIndices.find(i) != AllKetjuParticleIndices.end()) ? 1 : 0;
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
        hid_t grp = H5Gcreate(ketju_output_file, "BHs", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Gclose(grp);
        /* create extendable timestep dataset */
        hsize_t dims[1] = {0}, maxdims[1] = {H5S_UNLIMITED}, chunk[1] = {64};
        hid_t space = H5Screate_simple(1, dims, maxdims);
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);
        H5Dcreate(ketju_output_file, "timesteps", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
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
        H5Dcreate(ketju_output_file, "mergers", merger_type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Pclose(dcpl); H5Sclose(space); H5Tclose(merger_type);
    } else {
        ketju_output_file = H5Fopen(buf, H5F_ACC_RDWR, H5P_DEFAULT);
        /* read current timestep index */
        hid_t ds = H5Dopen(ketju_output_file, "timesteps", H5P_DEFAULT);
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
    hid_t ds = H5Dopen(ketju_output_file, dset_name, H5P_DEFAULT);
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
    hid_t ds = H5Dopen(ketju_output_file, "mergers", H5P_DEFAULT);
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
