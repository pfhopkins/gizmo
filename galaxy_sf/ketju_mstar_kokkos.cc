/* KETJU_KOKKOS: the KETJU C interface used by ketju_coupling.cc, mapped onto the C++/Kokkos MSTAR library mstar::mstar
 * (ketju-integrator/kokkos, Steinwandel 2026; physics and options of the C library, Rantala+ 2020, Mannerkoski+ 2023). */
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <set>
#include <vector>

#include "../declarations/allvars.h"
#include "ketju_mstar_kokkos.h"

#if defined(KETJU_REGULARIZATION) && defined(KETJU_KOKKOS)
#include <mstar/mstar.hpp>

#ifndef KETJU_KOKKOS_DEVICE_MIN_N
#define KETJU_KOKKOS_DEVICE_MIN_N 32 /* GPU builds: smaller regions run on the host execution space (launch latency) */
#endif

namespace {

struct KjMstarSide {           /* kept in ketju_system::internal_state */
    MPI_Comm comm;
    mstar::System *sys;        /* rank 0 of comm, reused across calls */
};
std::set<KjMstarSide *> &kj_live = *new std::set<KjMstarSide *>;   /* systems to destroy before Kokkos::finalize; never destroyed */
bool kj_kokkos_owned = false;

void kj_abort(const char *msg)
{
    fprintf(stderr, "KETJU_KOKKOS: %s\n", msg); fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

void kj_kokkos_finalize()
{
    for(KjMstarSide *s : kj_live) {delete s->sys; s->sys = nullptr;}
    if(kj_kokkos_owned && mstar::is_initialized()) mstar::finalize();
    kj_kokkos_owned = false;
}

/* once per run, on first use; GPU builds map devices by MPI rank; Kokkos warnings on world rank 0 only */
void kj_kokkos_initialize()
{
    if(mstar::is_initialized()) return;
    int wrank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    static char a0[] = "gizmo", a1[] = "--kokkos-map-device-id-by=mpi_rank", a2[] = "--kokkos-disable-warnings";
    char *argv[4]; int argc = 0; argv[argc++] = a0;
    if(strcmp(mstar::execution_space_name(), mstar::host_execution_space_name()) != 0) argv[argc++] = a1;
    if(wrank != 0) argv[argc++] = a2;
    argv[argc] = nullptr;
    mstar::initialize(argc, argv);
    kj_kokkos_owned = true;
    std::atexit(kj_kokkos_finalize);
    if(wrank == 0) printf("KETJU_KOKKOS: MSTAR on Kokkos %s (host %s), regions below %d bodies on the host engine\n",
                          mstar::execution_space_name(), mstar::host_execution_space_name(), KETJU_KOKKOS_DEVICE_MIN_N);
}

KjMstarSide *kj_side(struct ketju_system *R) {return reinterpret_cast<KjMstarSide *>(R->internal_state);}

/* the C state after a call, from rank 0 of comm to the others */
void kj_bcast_state(struct ketju_system *R, int n0, MPI_Comm comm)
{
    int hdr[3] = {R->num_particles, R->num_pn_particles, R->num_mergers}, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Bcast(hdr, 3, MPI_INT, 0, comm);
    if(rank != 0) {
        R->num_particles = hdr[0]; R->num_pn_particles = hdr[1];
        if(hdr[2] > R->num_mergers) R->merger_infos = (struct ketju_bh_merger_info *)realloc(R->merger_infos, sizeof(struct ketju_bh_merger_info) * hdr[2]);
        R->num_mergers = hdr[2];
    }
    struct ketju_system_physical_state *ps = R->physical_state;
    MPI_Bcast(&ps->time, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(ps->mass, n0, MPI_DOUBLE, 0, comm);
    MPI_Bcast(ps->pos, 3 * n0, MPI_DOUBLE, 0, comm);
    MPI_Bcast(ps->vel, 3 * n0, MPI_DOUBLE, 0, comm);
    MPI_Bcast(ps->spin, 3 * n0, MPI_DOUBLE, 0, comm);
    MPI_Bcast(R->perf, sizeof(struct ketju_performance_counters), MPI_BYTE, 0, comm);
    if(R->num_mergers > 0) MPI_Bcast(R->merger_infos, (int)(sizeof(struct ketju_bh_merger_info) * R->num_mergers), MPI_BYTE, 0, comm);
    if(R->particle_extra_data) MPI_Bcast(R->particle_extra_data, (int)(R->particle_extra_data_elem_size * n0), MPI_BYTE, 0, comm);
    if(R->collision_radius) MPI_Bcast(R->collision_radius, n0, MPI_DOUBLE, 0, comm);
}

} // namespace

void kj_mstar_create_system(struct ketju_system *R, int num_pn_particles, int num_other_particles, MPI_Comm comm)
{
    kj_kokkos_initialize();
    const int n = num_pn_particles + num_other_particles, na = n > 0 ? n : 1;
    memset(R, 0, sizeof(*R));
    R->num_particles = n; R->num_pn_particles = num_pn_particles;
    R->physical_state = (struct ketju_system_physical_state *)calloc(1, sizeof(struct ketju_system_physical_state));
    R->physical_state->mass = (double *)calloc(na, sizeof(double));
    R->physical_state->pos = (double (*)[3])calloc(na, sizeof(double[3]));
    R->physical_state->vel = (double (*)[3])calloc(na, sizeof(double[3]));
    R->physical_state->spin = (double (*)[3])calloc(na, sizeof(double[3]));
    R->perf = (struct ketju_performance_counters *)calloc(1, sizeof(struct ketju_performance_counters));
    R->options = (struct ketju_integration_options *)malloc(sizeof(struct ketju_integration_options));
    *R->options = ketju_default_integration_options;
    R->constants = (struct ketju_natural_constants *)malloc(sizeof(struct ketju_natural_constants));
    *R->constants = ketju_default_natural_constants;
    KjMstarSide *s = new KjMstarSide{comm, nullptr};
    kj_live.insert(s);
    R->internal_state = reinterpret_cast<struct ketju_system_internal_state *>(s);
}

void kj_mstar_free_system(struct ketju_system *R)
{
    KjMstarSide *s = kj_side(R);
    if(s) {kj_live.erase(s); delete s->sys; delete s;}
    if(R->physical_state) {
        free(R->physical_state->mass); free(R->physical_state->pos); free(R->physical_state->vel); free(R->physical_state->spin);
        free(R->physical_state);
    }
    free(R->perf); free(R->options); free(R->constants); free(R->merger_infos);
    memset(R, 0, sizeof(*R));
}

void kj_mstar_run_integrator(struct ketju_system *R, double time_interval)
{
    KjMstarSide *side = kj_side(R);
    int rank = 0, size = 1;
    if(side->comm != MPI_COMM_NULL) {MPI_Comm_rank(side->comm, &rank); MPI_Comm_size(side->comm, &size);}
    const int n0 = R->num_particles, npn0 = R->num_pn_particles;
    if(rank == 0) {
        if(R->options->mst_algorithm != KETJU_MST_PRIM) kj_abort("only mst_algorithm = KETJU_MST_PRIM is ported");
        if(!side->sys) side->sys = new mstar::System(npn0, n0 - npn0); else side->sys->reset(npn0, n0 - npn0);
        mstar::System &S = *side->sys;
        S.constants.G = R->constants->G; S.constants.c = R->constants->c;
        const struct ketju_integration_options *o = R->options; mstar::Options &q = S.options;
        q.gbs_kmax = o->gbs_kmax;
        q.gbs_relative_tolerance = o->gbs_relative_tolerance;
        q.gbs_absolute_tolerance = o->gbs_absolute_tolerance;
        q.output_time_relative_tolerance = o->output_time_relative_tolerance;
        q.output_time_absolute_tolerance = o->output_time_absolute_tolerance;
        q.PN_flags = o->PN_flags; /* same bit values as enum ketju_PN_flags */
        q.steps_between_mst_reconstruction = o->steps_between_mst_reconstruction;
        q.max_tree_distance = o->max_tree_distance;
        q.max_step_count = o->max_step_count;
        q.star_star_softening = o->star_star_softening;
        q.no_star_star_gravity = o->no_star_star_gravity;
        q.enable_bh_mergers = o->enable_bh_mergers;
        q.bh_merger_distance_in_Rs = o->bh_merger_distance_in_Rs;
        q.enable_bh_merger_kicks = o->enable_bh_merger_kicks;
        q.enable_external_tide = o->enable_external_tide;
        for(int k = 0; k < 6; k++) q.external_tidal_tensor[k] = o->external_tidal_tensor[k];
        q.host_engine = (n0 < KETJU_KOKKOS_DEVICE_MIN_N);
        struct ketju_system_physical_state *ps = R->physical_state;
        const double t_start = ps->time;
        S.time = t_start;
        for(int i = 0; i < n0; i++) {
            S.mass[i] = ps->mass[i];
            for(int k = 0; k < 3; k++) {S.pos[3*i+k] = ps->pos[i][k]; S.vel[3*i+k] = ps->vel[i][k];}
        }
        for(int i = 0; i < npn0; i++) for(int k = 0; k < 3; k++) S.spin[3*i+k] = ps->spin[i][k];
        if(R->collision_radius) S.collision_radius.assign(R->collision_radius, R->collision_radius + n0);
        bool ok = false;
        try {ok = S.run(time_interval);} catch(const std::exception &e) {kj_abort(e.what());}
        if(!ok) kj_abort("Exceeded options->max_step_count steps!");
        /* state back; the tail slots of merged-away particles keep stale values as in the C library */
        const int n = S.num_particles, npn = S.num_pn_particles;
        R->num_particles = n; R->num_pn_particles = npn; ps->time = S.time;
        for(int i = 0; i < n; i++) {
            ps->mass[i] = S.mass[i];
            for(int k = 0; k < 3; k++) {ps->pos[i][k] = S.pos[3*i+k]; ps->vel[i][k] = S.vel[3*i+k];}
        }
        for(int i = 0; i < npn; i++) for(int k = 0; k < 3; k++) ps->spin[i][k] = S.spin[3*i+k];
        if(R->collision_radius) for(int i = 0; i < n; i++) R->collision_radius[i] = S.collision_radius[i];
        /* extra data permuted as S.extra_index (merged-away at the end); merger slots are positions in that array */
        std::vector<int> pos_of(n0);
        for(int s = 0; s < n0; s++) pos_of[S.extra_index[s]] = s;
        if(R->particle_extra_data && n < n0) {
            const size_t es = R->particle_extra_data_elem_size; char *ed = (char *)R->particle_extra_data;
            std::vector<char> tmp(ed, ed + es * n0);
            for(int s = 0; s < n0; s++) memcpy(ed + es * s, tmp.data() + es * S.extra_index[s], es);
        }
        for(int m = 0; m < R->num_mergers; m++) {
            int &e1 = R->merger_infos[m].extra_index1;
            if(e1 >= 0 && e1 < n0) e1 = pos_of[e1];
        }
        const int n_new = (int)S.mergers.size();
        if(n_new > 0) {
            R->merger_infos = (struct ketju_bh_merger_info *)realloc(R->merger_infos, sizeof(struct ketju_bh_merger_info) * (R->num_mergers + n_new));
            for(int m = 0; m < n_new; m++) {
                const mstar::MergerInfo &mi = S.mergers[m]; struct ketju_bh_merger_info &c = R->merger_infos[R->num_mergers + m];
                memset(&c, 0, sizeof(c));
                c.t_merger = mi.t_merger - t_start; /* C: time since the start of the call */
                c.m1 = mi.m1; c.m2 = mi.m2; c.m_remnant = mi.m_remnant;
                c.chi1 = mi.chi1; c.chi2 = mi.chi2; c.chi_remnant = mi.chi_remnant; c.v_kick = mi.v_kick;
                c.extra_index1 = pos_of[mi.extra_index1]; c.extra_index2 = pos_of[mi.extra_index2];
                c.is_collision = mi.is_collision; c.dE_collision = mi.dE_collision;
            }
            R->num_mergers += n_new;
        }
        /* counters accumulate over calls as in the C library */
        R->perf->successful_steps += (int)S.perf.successful_steps;
        R->perf->failed_steps += (int)S.perf.failed_steps;
        R->perf->total_work += S.perf.total_work;
        R->perf->relative_energy_error = S.perf.relative_energy_error;
        R->perf->time_mst += S.perf.t_mst;
        R->perf->time_total += S.perf.t_total;
    }
    if(size > 1) kj_bcast_state(R, n0, side->comm);
}

#endif
