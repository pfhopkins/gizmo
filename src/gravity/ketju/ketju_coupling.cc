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
};

/* ============================================================

/* ============================================================
 *  MPI task group for parallel KETJU computation
 *  Based on public GADGET4-KETJU (Mannerkoski+ 2023)
 * ============================================================ */

/* ============================================================
 *  A single KETJU region
 * ============================================================ */
struct KetjuRegion {
    std::vector<ketju_mpi_particle> centers;      /* BH/massive-star region centers */
    std::vector<ketju_mpi_particle> all_particles; /* all chain members (gathered on root) */
    std::set<int> local_member_indices;            /* local P[] indices in this region */
    int total_particle_count;



};


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




/* ============================================================
 *  PHASE 4: Negative half-step kick and integration
 * ============================================================ */







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
        P[i].KetjuIntegrated = 0; P[i].KetjuChainID = 0; P[i].KetjuOpenRadius = 0;
    }
    ActiveRegions.clear();
    KetjuNumMultiRegions = 0;

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








#endif /* KETJU_REGULARIZATION */
