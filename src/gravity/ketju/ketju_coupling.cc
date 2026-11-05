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

/* Wall-clock laps into the KETJU cpu.txt categories; WallclockTime is advanced by the same amount so the
 * enclosing GIZMO timer (gravity, drift) does not count it again. The destructor laps into `rest`. */
struct KjClock {
    double t; int rest;
    explicit KjClock(int rest_cat) : t(my_second()), rest(rest_cat) {}
    void lap(int cat) {double n = my_second(), dt = n - t; cpu_charge_child(cat, dt); t = n;}
    ~KjClock() {lap(rest);}
};



/* ============================================================

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



/* Gather all synchronised chain centers across MPI tasks */
static std::vector<ketju_mpi_particle> gather_chain_centers(void)
{
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
        tree_pair_softening(idx, mp.Soft, mp.Zeta);
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
#endif
        k++;
    }

    if(reg.total_particle_count != n_local) {endrun(667001);} /* region across ranks */
    reg.all_particles = local_parts;
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


    /* run MSTAR integrator (all compute tasks participate via compute_tasks.comm) */
    if(reg.compute_tasks.is_member() && reg.integrator) {
        kj_state_change_units(reg.integrator->physical_state, reg.integrator->num_particles, 1);
        kj_mstar_run_integrator(reg.integrator, dt_physical * KJ_TIME_TO_NAT);
        kj_state_change_units(reg.integrator->physical_state, reg.integrator->num_particles, 0);
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
            for(int j = 0; j < 3; j++) {
                /* physical -> comoving: use a_end since positions are at end of step */
                sp.Pos[j] = reg.com_pos[j] + ps->pos[i][j] / a_end;
                /* physical velocity -> canonical momentum: remove Hubble flow, multiply by a_end.
                 * sp.Vel stores CoM-relative canonical momentum (com_vel added later in velocity trick).
                 * In non-cosmo: a_end=1, hubble_end=0, so sp.Vel = ps->vel (unchanged). */
                sp.Vel[j] = (ps->vel[i][j] - hubble_end * ps->pos[i][j]) * a_end;
            }
        }

    }
    int my_count = n; std::vector<ketju_scatter_particle> &my_particles = scatter_buf;
    /* ---- Phase 4: each task applies velocity trick to its particles ---- */
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
}

/* ============================================================
 *  PUBLIC INTERFACE FUNCTIONS
 * ============================================================ */




/* host writes to P (bins, chain flags): the device copy is re-uploaded before its next use */
struct KjArenaInvalidate {~KjArenaInvalidate() {gpu_particles_arena_invalidate();}};


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

    std::vector<ketju_mpi_particle> centers = gather_chain_centers();
    if(centers.empty()) return;

    /* merge overlapping regions */
    double merge_radius = 2.0 * All.KetjuRegionRadius;
    std::vector<std::vector<ketju_mpi_particle>> region_centers = build_orbit_regions(centers); (void)merge_radius;
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
void ketju_report_stats(int final)
{
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


void ketju_run_integration(void)
{
    KjClock clk(CPU_KETJUFIND);
    if(KetjuNumMultiRegions == 0) {
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

    int n_rounds = 1;
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


    ketju_collect_stats();
    ketju_report_stats(0);

}


void ketju_set_final_velocities(void)
{
    KjClock clk(CPU_KETJUFINISH);
    /* called after drift: swap in true physical velocities and correct dp[] momentum.
     * The velocity trick set Vel = delta_pos/dt_drift for the drift.
     * Now we replace it with the true MSTAR velocity and correct dp[].
     * Only for members whose step ends at this sync point (their bin is active and they were just
     * drifted to it): a region on a longer bin is still drifting along its chord. */
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



#endif /* KETJU_REGULARIZATION */
