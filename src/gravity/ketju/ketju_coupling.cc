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

    if(ThisTask == 0 && All.Ti_Current == 0) {printf("KETJU: Found 0 region(s) from %d center(s)\n", (int)centers.size());}
}








#endif /* KETJU_REGULARIZATION */
