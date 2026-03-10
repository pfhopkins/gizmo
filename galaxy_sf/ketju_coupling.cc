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

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "ketju_coupling.h"

#ifdef KETJU_REGULARIZATION

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
 *  A single KETJU region
 * ============================================================ */
struct KetjuRegion {
    std::vector<ketju_mpi_particle> centers;      /* BH/massive-star region centers */
    std::vector<ketju_mpi_particle> all_particles; /* all chain members (gathered on root) */
    std::set<int> local_member_indices;            /* local P[] indices in this region */
    int total_particle_count;
    int num_pn_particles;                          /* count of PN-enabled (SMBH) particles */
    int root_task;                                 /* MPI task that runs the integrator */

    /* integrator state (only allocated on root_task) */
    struct ketju_system *integrator;
    std::vector<ketju_extra_data> extra_data;

    /* CoM frame data (stored at setup, used in scatter) */
    double com_pos[3];
    double com_vel[3];
    integertime ti_step;     /* integer timestep for this region */

    KetjuRegion() : total_particle_count(0), num_pn_particles(0), root_task(0), integrator(NULL), ti_step(0) {
        for(int j = 0; j < 3; j++) { com_pos[j] = 0; com_vel[j] = 0; }
    }
    ~KetjuRegion() { if(integrator) { ketju_free_system(integrator); free(integrator); } }
};

/* ============================================================
 *  Module-level state (persists within a timestep)
 * ============================================================ */
static std::vector<KetjuRegion> ActiveRegions;
static std::set<int> AllKetjuParticleIndices; /* local indices of all particles in any region */

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
static int is_pn_particle(ketju_mpi_particle *p)
{
#ifdef SINK_PARTICLES
    if(p->Type == 5 && p->SinkSubType == 0) return 1;
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

    /* assign root task = task with most particles (simple heuristic) */
    reg.root_task = 0;
    for(int t = 1; t < NTask; t++) {
        if(counts[t] > counts[reg.root_task]) reg.root_task = t;
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
        }
#ifdef SINK_PARTICLES
        mp.SinkSubType = (P[idx].Type == 5) ? P[idx].SinkSubType : -1;
        for(int j = 0; j < 3; j++) mp.Spin[j] = (P[idx].Type == 5) ? P[idx].KetjuSpin[j] : 0;
#endif
        k++;
    }

    /* gather all particles on root */
    std::vector<int> byte_counts(NTask), byte_displs(NTask);
    int total_bytes = 0;
    for(int t = 0; t < NTask; t++) {
        byte_counts[t] = counts[t] * sizeof(ketju_mpi_particle);
        byte_displs[t] = total_bytes;
        total_bytes += byte_counts[t];
    }

    if(ThisTask == reg.root_task) {
        reg.all_particles.resize(reg.total_particle_count);
    }

    MPI_Gatherv(local_parts.data(), n_local * sizeof(ketju_mpi_particle), MPI_BYTE,
                reg.all_particles.data(), byte_counts.data(), byte_displs.data(),
                MPI_BYTE, reg.root_task, MPI_COMM_WORLD);

    /* set up integrator on root task */
    if(ThisTask == reg.root_task) {
        /* count PN particles (SMBH = SinkSubType 0) — they go first */
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
                return a_pn > b_pn; /* PN first */
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

        /* allocate integrator */
        reg.integrator = (struct ketju_system *)calloc(1, sizeof(struct ketju_system));
        int n_other = reg.total_particle_count - reg.num_pn_particles;

        /* use MPI_COMM_SELF since only root runs the integrator */
        ketju_create_system(reg.integrator, reg.num_pn_particles, n_other, MPI_COMM_SELF);

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
            double h = All.ForceSoftening[4]; /* Type 4 softening */
#ifdef SINK_PARTICLES
            if(All.ForceSoftening[5] > 0) h = DMIN(h, All.ForceSoftening[5]);
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
            /* copy spin angular momentum for PN particles (sorted first) */
            if(i < reg.num_pn_particles) {
                for(int j = 0; j < 3; j++) ps->spin[i][j] = reg.all_particles[i].Spin[j];
            }
#endif
        }
        reg.integrator->particle_extra_data = reg.extra_data.data();
        reg.integrator->particle_extra_data_elem_size = sizeof(ketju_extra_data);

        if(ThisTask == 0) {
            printf("KETJU: Region with %d particles (%d PN), root task %d\n",
                   reg.total_particle_count, reg.num_pn_particles, reg.root_task);
        }
    }

    /* broadcast CoM data to all tasks (needed in scatter_results) */
    MPI_Bcast(reg.com_pos, 3, MPI_DOUBLE, reg.root_task, MPI_COMM_WORLD);
    MPI_Bcast(reg.com_vel, 3, MPI_DOUBLE, reg.root_task, MPI_COMM_WORLD);
}

/* ============================================================
 *  PHASE 4: Negative half-step kick and integration
 * ============================================================ */

/* Subtract softened pairwise gravity between chain members from their velocities.
 * This undoes the tree force contribution so MSTAR can replace it with exact forces. */
static void do_negative_halfstep_kick(KetjuRegion &reg, double kick_factor)
{
    if(ThisTask != reg.root_task || !reg.integrator) return;

    int n = reg.integrator->num_particles;
    struct ketju_system_physical_state *ps = reg.integrator->physical_state;

    /* softening: use the star softening */
    double h = All.ForceSoftening[4];
#ifdef SINK_PARTICLES
    if(All.ForceSoftening[5] > 0) h = DMIN(h, All.ForceSoftening[5]);
#endif

    std::vector<double> dv(3 * n, 0.0);

    for(int i = 0; i < n; i++) {
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

    /* apply negative kick (subtract tree force) */
    for(int i = 0; i < n; i++) {
        for(int k = 0; k < 3; k++) {
            ps->vel[i][k] += dv[3*i + k];
        }
    }
}

/* Run the full integration step for one region */
static void integrate_region(KetjuRegion &reg, double dt_physical)
{
    if(dt_physical <= 0) return;

    double half_kick = 0.5 * dt_physical;

    /* subtract first half of tree force */
    do_negative_halfstep_kick(reg, half_kick);

    /* run MSTAR integrator */
    if(ThisTask == reg.root_task && reg.integrator) {
        ketju_run_integrator(reg.integrator, dt_physical);

        int n_steps = reg.integrator->perf->successful_steps + reg.integrator->perf->failed_steps;
        double dE = reg.integrator->perf->relative_energy_error;

        printf("KETJU [task %d]: Integrated %d particles for dt=%g, %d steps (%d failed), dE/E=%g\n",
               ThisTask, reg.total_particle_count, dt_physical,
               reg.integrator->perf->successful_steps,
               reg.integrator->perf->failed_steps, dE);

        /* warn if step count hit the safety limit */
        if(All.KetjuMaxStepCount > 0 && n_steps >= All.KetjuMaxStepCount) {
            printf("KETJU WARNING: integration hit max step count (%d) — results may be inaccurate!\n",
                   All.KetjuMaxStepCount);
        }
        /* warn if energy error is large */
        if(fabs(dE) > 1e-4) {
            printf("KETJU WARNING: large energy error dE/E=%g in region with %d particles\n",
                   dE, reg.total_particle_count);
        }
    }

    /* subtract second half of tree force */
    do_negative_halfstep_kick(reg, half_kick);
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
    MPI_Bcast(&n, 1, MPI_INT, reg.root_task, MPI_COMM_WORLD);
    if(n < 2) return;

    /* prepare final positions (absolute) and velocities (CoM-relative) on root */
    std::vector<double> final_abs_pos(3 * n), final_rel_vel(3 * n);
    std::vector<double> final_mass(n);
#if defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH)
    std::vector<double> original_mass(n);
#endif
#ifdef SINK_PARTICLES
    int n_pn = 0;
    std::vector<double> final_spin; /* only for PN particles */
#endif
    if(ThisTask == reg.root_task && reg.integrator) {
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

    /* broadcast to all tasks */
    if(ThisTask != reg.root_task) {
        reg.extra_data.resize(n);
    }
    MPI_Bcast(reg.extra_data.data(), n * sizeof(ketju_extra_data), MPI_BYTE,
              reg.root_task, MPI_COMM_WORLD);
    MPI_Bcast(final_abs_pos.data(), 3 * n, MPI_DOUBLE, reg.root_task, MPI_COMM_WORLD);
    MPI_Bcast(final_rel_vel.data(), 3 * n, MPI_DOUBLE, reg.root_task, MPI_COMM_WORLD);
    MPI_Bcast(final_mass.data(), n, MPI_DOUBLE, reg.root_task, MPI_COMM_WORLD);
#if defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH)
    MPI_Bcast(original_mass.data(), n, MPI_DOUBLE, reg.root_task, MPI_COMM_WORLD);
#endif
#ifdef SINK_PARTICLES
    MPI_Bcast(&n_pn, 1, MPI_INT, reg.root_task, MPI_COMM_WORLD);
    if(ThisTask != reg.root_task) final_spin.resize(3 * n_pn, 0.0);
    if(n_pn > 0) MPI_Bcast(final_spin.data(), 3 * n_pn, MPI_DOUBLE, reg.root_task, MPI_COMM_WORLD);
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

    /* build regions */
    for(size_t r = 0; r < region_centers.size(); r++) {
        KetjuRegion reg;
        reg.centers = region_centers[r];
        reg.local_member_indices = find_local_members(reg.centers, All.KetjuRegionRadius);

        /* track all local KETJU particles for guard checks */
        for(int idx : reg.local_member_indices) {
            AllKetjuParticleIndices.insert(idx);
        }

        /* set up integrator and gather particles */
        setup_integrator(reg);

        ActiveRegions.push_back(std::move(reg));
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

#endif /* KETJU_REGULARIZATION */
