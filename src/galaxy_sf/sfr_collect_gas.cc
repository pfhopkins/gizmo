/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* galaxy_sf/sfr_collect_gas.cc -- host side of the single-star gas collection (Specs in sfr_collect_gas_loop.h).
 * New stars are spawned with zero mass (spawn_star_from_gas, single_star_attach_to_new_star) and then collect their drawn IMF mass from the gas
 * around them, in passes with a growing radius: pass 0 uses
 * the parent cell's kernel radius, widened so it holds twice the mass still missing at the formation density; each
 * later pass widens the last radius by 1.5. A star is done when it holds its drawn mass to within 1%. A star still
 * short after 10 passes is a failed draw: everything granted to it goes back to its donors (exact integer quanta) and
 * the star is removed. No mass is created. The drawn mass is kept in the star record (single_star_record.h). */
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <map>
#include <tuple>
#include <array>
#include <Kokkos_Core.hpp>
#include "../declarations/gpu_all_mirror.h"  /* before allvars.h */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../gravity/forcetree.h"
#include "../core/wakeup_sidecar.h"
#include "../system/gpu_particles_arena.h"
#include "../mesh/gpu_neighbor_list.h"
#include "single_star_imf.h"
#ifdef GALSF_SINGLE_STARS_POPIII
#include "single_star_popIII.h"
#endif
#include "single_star_record.h"
#include "sfr_collect_gas_loop.h"

#ifdef GALSF_SINGLE_STARS_COLLECT_GAS_FROM_NEIGHBORS

#define SSC_MAX_PASSES 10
#define SSC_TOLERANCE 0.01   /* accepted |M_star - M_drawn| / M_drawn */
#define SSC_LOG_COLS 12      /* columns of single_star_births.txt */

/* stars spawned on this rank since the last collection, waiting for their mass */
struct ssc_birth {int i, parent; double rho_form;};
static std::vector<ssc_birth> ssc_births;

/* per pending star during one collection */
struct ssc_star {
    int i, parent; double M_drawn, rho_form, R; long long N_drawn, N_got, n_split, n_remote;
    ssc_i128 Z[SSC_NZ], p[3];
    int active, passes;
};

static double ssc_mu(void) /* the mass quantum: a power of two, so that ulp(m) <= mu for every star and donor mass */
{
    double M_ref = DMAX(SINGLE_STAR_IMF_MASS_MAX / UNIT_MASS_IN_SOLAR, 2. * All.MaxMassForParticleSplit);
    return ldexp(1.0, (int)ceil(log2(M_ref)) - 52);
}

static inline double ssc_pow2_at_least(double x) {return ldexp(1.0, (int)ceil(log2(DMAX(x, 1e-300))));}

/* the donor sums R_j, G'_j, G_j by local gas index: grown at the first collection that needs more, all-zero between
   collections (clear and apply drain every entry they touch; failed draws are subtracted before apply) */
static SscDonorSums *ssc_donor = nullptr;
static int ssc_donor_cap = 0;
static SscDonorSums *ssc_donor_sums(int n_gas)
{
    const int n = (n_gas > 0) ? n_gas : 1;
    if(n > ssc_donor_cap) {
        SscDonorSums *q = (SscDonorSums *) gizmo_gpu_alloc_shared((size_t) n * sizeof(SscDonorSums), "ssc_donor_sums");
        if(!q) {gizmo_request_controlled_stop(7813, "single-star gas collection: could not grow the donor buffer", __FILE__, __LINE__, __FUNCTION__); return nullptr;}
        Kokkos::fence();
        std::memset(q, 0, (size_t) n * sizeof(SscDonorSums));
        if(ssc_donor) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(ssc_donor);}
        gizmo_mem_account_add(GIZMO_MEM_SINGLE_STARS, (long long)(n - ssc_donor_cap) * (long long) sizeof(SscDonorSums));
        ssc_donor = q; ssc_donor_cap = n;
    }
    return ssc_donor;
}

/* ---- Spec hooks ---- */

static int ssc_is_pending_star(int i) {return (P[i].Type == 4 && P[i].Mass == 0);}
#if defined(OUTPUT_MASS_METAL_ENERGY_LEDGER) && defined(COSMIC_RAY_FLUID)
static double ssc_local_cr_energy(void) {double e = 0; for(int i = 0; i < N_gas; i++) {if(P[i].Type == 0) {for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {e += CellP[i].CosmicRayEnergy[k];}}} return e;}
#endif
bool SscSpecBase::is_active(int i) {return ssc_is_pending_star(i);}

double SscSpecBase::search_radius(const neighbor_loop_args& args, int active_slot, int)
{
    return nlr_aux<SscCountSpec>(args)->host_stars[active_slot].R;
}

SscCallScalars SscSpecBase::populate_call_scalars(const neighbor_loop_args& args)
{
    SscCallScalars s;
    s.common = nlr_common_scalars_from_all();
    s.mu = ssc_mu();
    s.m_floor = 0.01 * nlr_host_all_ptr()->MinMassForParticleMerger; /* donors stay far from empty */
    s.m_cap = ldexp(s.mu, 53);
    Aux *aux = nlr_aux<SscCountSpec>(args);
    s.n_pairs = aux ? aux->n_pairs : nullptr; s.records = aux ? aux->records : nullptr;
    s.n_records = aux ? aux->n_records : nullptr; s.capacity = aux ? aux->capacity : 0;
    s.donor = ssc_donor; s.n_donor = DMIN(N_gas, ssc_donor_cap); s.task = ThisTask;
#ifdef GALSF_FB_PHOTOIONIZATION_PER_DIRECTION
    s.ti_current = All.Ti_Current;
#endif
    return s;
}

void SscSpecBase::populate_device_context(const neighbor_loop_args& args, DeviceContext& ctx)
{
    Aux *aux = nlr_aux<SscCountSpec>(args);
    const int N = args.num_active;
    ctx.per_active_local = nullptr;
    if(N <= 0 || aux == nullptr || aux->host_stars == nullptr) {return;}
    SscStarIn *uvm = (SscStarIn *) gizmo_gpu_alloc_shared((size_t) N * sizeof(SscStarIn), NULL);
    if(!uvm) {
        ctx.populate_failed = 1;
        gizmo_request_controlled_stop(7811, "single-star gas collection: could not stage the new stars", __FILE__, __LINE__, __FUNCTION__);
        return;
    }
    std::memcpy(uvm, aux->host_stars, (size_t) N * sizeof(SscStarIn));
    ctx.per_active_local = uvm;
}

void SscSpecBase::cleanup_device_context(const neighbor_loop_args&, DeviceContext& ctx)
{
    if(ctx.per_active_local) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(const_cast<SscStarIn *>(ctx.per_active_local)); ctx.per_active_local = nullptr;}
}

void SscCountSpec::apply_active_writeback(const neighbor_loop_args& args, int slot, int, const AccumData& a)
{
    ((long long *) nlr_aux<SscCountSpec>(args)->out)[slot] = a.E;
}
void SscRequestSpec::apply_active_writeback(const neighbor_loop_args&, int, int, const AccumData&) {}
void SscGrantSpec::apply_active_writeback(const neighbor_loop_args& args, int slot, int, const AccumData& a)
{
    ((SscGrantOut *) nlr_aux<SscGrantSpec>(args)->out)[slot] = a;
}

/* one runner call over the given stars */
template <typename Spec>
static void ssc_run(std::vector<int>& list, std::vector<SscStarIn>& in, SscSpecBase::Aux& aux)
{
    aux.host_stars = in.data();
    neighbor_loop_args args = nlr_default_args();
    args.active_list = list.empty() ? nullptr : list.data(); args.num_active = (int) list.size(); args.aux = &aux;
    run_neighbor_loop<Spec>(args);
}

static FILE *FdSingleStarBirths = NULL;

/* collective: every rank calls it after the star-formation step; does nothing unless a star was spawned anywhere.
   Returns the mass this rank's new stars collected (code units). Collectives, only in steps that spawned a star: per
   pass one integer reduction plus the exchanges of four runner calls; one runner call to apply; a gather of the birth-log
   rows; and, only if a draw failed, a gather of the failed stars' identities. */
double single_stars_collect_gas(long long tot_spawned)
{
    if(tot_spawned <= 0) {ssc_births.clear(); return 0;}
    const double mu = ssc_mu();
    int no_buffer = (ssc_donor_sums(N_gas) == nullptr), any_no_buffer = 0;
    MPI_Allreduce(&no_buffer, &any_no_buffer, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if(any_no_buffer) {ssc_births.clear(); return 0;}
    std::vector<ssc_star> S(ssc_births.size());
    for(size_t s = 0; s < S.size(); s++) {
        ssc_star& t = S[s]; std::memset(&t, 0, sizeof(t));
        t.i = ssc_births[s].i; t.parent = ssc_births[s].parent; t.rho_form = ssc_births[s].rho_form;
        t.M_drawn = SingleStars.M_drawn[single_star_slot(P[t.i].StarIdx)];
        t.N_drawn = llround(t.M_drawn / UNIT_MASS_IN_SOLAR / mu);
        t.R = (P[t.i].KernelRadius > 0) ? P[t.i].KernelRadius : 5.0 * All.ForceSoftening[P[t.i].Type];
        t.active = 1;
    }
    ssc_births.clear();

    long long *counters = (long long *) gizmo_gpu_alloc_shared(2 * sizeof(long long), "ssc_counters"); /* pairs requested, grant records */
    std::vector<SscGrantRecord> records; /* this rank's grants to its own donors, all passes */
    int any_active = 1;
    for(int pass = 0; pass < SSC_MAX_PASSES && any_active; pass++)
    {
        std::vector<int> list; std::vector<SscStarIn> in; std::vector<int> which;
        for(size_t s = 0; s < S.size(); s++) {
            ssc_star& t = S[s];
            if(!t.active) {continue;}
            double need = (double)(t.N_drawn - t.N_got) * mu, h = (pass > 0) ? 1.5 * t.R : t.R;
            double M_kernel = (4.0 / 3.0) * M_PI * h * h * h * DMAX(t.rho_form, 1e-30);
            t.R = (2.0 * need > M_kernel && M_kernel > 0) ? h * pow(2.0 * need / M_kernel, 1.0 / 3.0) : h;
            t.passes = pass + 1;
            SscStarIn x; std::memset(&x, 0, sizeof(x));
            for(int k = 0; k < 3; k++) {x.Pos[k] = P[t.i].Pos[k];}
            x.R = t.R; x.id = P[t.i].ID; x.child = P[t.i].ID_child_number; x.gen = P[t.i].ID_generation; x.task = ThisTask;
            x.q_Z = ssc_pow2_at_least(t.N_drawn * mu) * ldexp(1.0, -100);   /* metal mass <= drawn mass */
            x.q_p = ssc_pow2_at_least(t.N_drawn * mu) * ldexp(1.0, -60);    /* momentum: |v| up to ~2^66 code units */
            list.push_back(t.i); in.push_back(x); which.push_back((int)s);
        }
        /* count: available quanta within R; f = remaining need / available, at most 1/2 */
        std::vector<long long> E(in.size() + 1, 0);
        SscSpecBase::Aux aux = {nullptr, E.data(), counters, nullptr, counters + 1, 0};
        ssc_run<SscCountSpec>(list, in, aux);
        for(size_t a = 0; a < in.size(); a++) {
            const ssc_star& t = S[which[a]];
            in[a].f = (E[a] > 0) ? DMIN((double)(t.N_drawn - t.N_got) / (double)E[a], 0.5) : 0;
#ifdef GALSF_SINGLE_STARS_POPIII
            if(SingleStars.popIII[single_star_slot(P[t.i].StarIdx)] && 2 * (t.N_drawn - t.N_got) > E[a]) {in[a].f = 0;} /* Pop III: take mass only once the whole remainder is in reach (the radius grows meanwhile); a draw still short after the last pass returns its gas */
#endif
        }
        /* request, then grant (records sized by the pairs requested on this rank), then clear */
        counters[0] = counters[1] = 0;
        ssc_run<SscRequestSpec>(list, in, aux);
        long long cap = counters[0];
        SscGrantRecord *rec = (SscGrantRecord *) gizmo_gpu_alloc_shared((size_t) DMAX(cap, 1) * sizeof(SscGrantRecord), "ssc_records");
        std::vector<SscGrantOut> G(in.size() + 1);
        aux.out = G.data(); aux.records = rec; aux.capacity = cap; counters[1] = 0;
        ssc_run<SscGrantSpec>(list, in, aux);
        if(counters[1] > cap) {gizmo_request_controlled_stop(7812, "single-star gas collection: more grants than requests", __FILE__, __LINE__, __FUNCTION__);}
        records.insert(records.end(), rec, rec + DMIN(counters[1], cap));
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(rec);
        aux.records = nullptr; aux.capacity = 0;
        ssc_run<SscClearSpec>(list, in, aux);
        /* stars inside the tolerance are done */
        int n_open = 0;
        for(size_t a = 0; a < in.size(); a++) {
            ssc_star& t = S[which[a]];
            t.N_got += G[a].N; t.n_split += G[a].n_split; t.n_remote += G[a].n_remote;
            for(int k = 0; k < 3; k++) {t.p[k] += G[a].p[k];}
            for(int k = 0; k < (SSC_NZ); k++) {t.Z[k] += G[a].Z[k];}
            if((double)(t.N_drawn - t.N_got) <= SSC_TOLERANCE * (double)t.N_drawn) {t.active = 0;} else {n_open++;}
        }
        MPI_Allreduce(&n_open, &any_active, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }

    /* failed draws: hand their grants back, exactly, on the donors' ranks */
    struct ssc_key {MyIDType id, child; int gen;};
    std::vector<ssc_key> failed_mine, failed_all;
    for(ssc_star& t : S) {if(t.active) {failed_mine.push_back({P[t.i].ID, P[t.i].ID_child_number, (int)P[t.i].ID_generation});}}
    int n_failed_mine = (int) failed_mine.size();
    if(any_active) { /* some star is still short after the last pass, on some rank: return the failed grants */
        int nb = n_failed_mine * (int) sizeof(ssc_key); std::vector<int> counts(NTask), displs(NTask); int tot = 0;
        MPI_Allgather(&nb, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        for(int t = 0; t < NTask; t++) {displs[t] = tot; tot += counts[t];}
        failed_all.resize(tot / sizeof(ssc_key));
        MPI_Allgatherv(failed_mine.data(), nb, MPI_BYTE, failed_all.data(), counts.data(), displs.data(), MPI_BYTE, MPI_COMM_WORLD);
        for(const SscGrantRecord& r : records) {
            for(const ssc_key& f : failed_all) {
                if(r.id == f.id && r.child == f.child && r.gen == f.gen) {ssc_donor[r.j].grant -= r.g; break;}
            }
        }
        gpu_particles_arena_invalidate();
    }

#ifdef GALSF_SINGLE_STARS_EVENT_LOGS
    { /* IMFinfo.txt: one row per (star, donor cell) with the mass taken over all passes (failed draws left out), written by rank 0 sorted by star and donor */
        std::map<std::tuple<unsigned long long, unsigned long long, int, unsigned long long>, long long> taken;
        for(const SscGrantRecord& r : records) {
            bool failed = false;
            for(const ssc_key& f : failed_all) {if(r.id == f.id && r.child == f.child && r.gen == f.gen) {failed = true; break;}}
            if(!failed) {taken[std::make_tuple((unsigned long long)r.id, (unsigned long long)r.child, r.gen, (unsigned long long)P[r.j].ID)] += r.g;}
        }
        std::vector<double> mine; for(const auto& kv : taken) {mine.push_back((double)std::get<0>(kv.first)); mine.push_back((double)std::get<3>(kv.first)); mine.push_back((double)kv.second * mu * UNIT_MASS_IN_SOLAR);}
        int nm = (int) mine.size(), nall = 0; std::vector<int> cnt(NTask), dsp(NTask);
        MPI_Gather(&nm, 1, MPI_INT, cnt.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        if(ThisTask == 0) {for(int t = 0; t < NTask; t++) {dsp[t] = nall; nall += cnt[t];}}
        std::vector<double> got(DMAX(nall, 1));
        MPI_Gatherv(mine.data(), nm, MPI_DOUBLE, got.data(), cnt.data(), dsp.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if(ThisTask == 0 && nall > 0) {
            static FILE *fd = NULL;
            if(!fd) {
                char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE]; snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s%s", All.OutputDir, "IMFinfo.txt");
                fd = fopen(buf, (RestartFlag == 0) ? "w" : "a");
                if(fd && RestartFlag == 0) {fprintf(fd, "# time star_ID donor_gas_ID dM_taken[Msun]\n");}
            }
            std::vector<std::array<double,3>> v(nall / 3); for(size_t k = 0; k < v.size(); k++) {v[k] = {got[3*k], got[3*k+1], got[3*k+2]};}
            std::sort(v.begin(), v.end());
            for(const auto& x : v) {if(fd) {fprintf(fd, "%.10g %.0f %.0f %.12e\n", All.Time, x[0], x[1], x[2]);}}
            if(fd) {fflush(fd);}
        }
    }
#endif

    /* apply: each donor loses its granted quanta once */
    {
        std::vector<int> list; std::vector<SscStarIn> in;
        for(ssc_star& t : S) {
            SscStarIn x; std::memset(&x, 0, sizeof(x));
            for(int k = 0; k < 3; k++) {x.Pos[k] = P[t.i].Pos[k];}
            x.R = t.R; list.push_back(t.i); in.push_back(x);
        }
        SscSpecBase::Aux aux = {nullptr, nullptr, counters, nullptr, counters + 1, 0};
#if defined(OUTPUT_MASS_METAL_ENERGY_LEDGER) && defined(COSMIC_RAY_FLUID)
        double e_cr_0 = ssc_local_cr_energy();
#endif
        ssc_run<SscApplySpec>(list, in, aux);
#if defined(OUTPUT_MASS_METAL_ENERGY_LEDGER) && defined(COSMIC_RAY_FLUID)
        ledger_add(LEDGER_E_CR_COLLECTED, e_cr_0 - ssc_local_cr_energy());
#endif
    }
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(counters);

    /* the stars: collected mass, its momentum and metals; failed draws keep zero mass and are removed with the next
       rearrangement of the particle list */
    std::vector<double> rows; /* time ID child gen M_drawn M_collected R passes status split_grants remote_grants rho_parent */
    double M_collected_here = 0;
    std::vector<int> moved;
    for(ssc_star& t : S) {
        int ok = !t.active;
        if(ok) {
            double M = (double)t.N_got * mu;
            P[t.i].Mass = M;
            for(int k = 0; k < 3; k++) {P[t.i].Vel[k] = (double)t.p[k] * ssc_pow2_at_least(t.N_drawn * mu) * ldexp(1.0, -60) / M;}
#ifdef METALS
            for(int k = 0; k < NUM_METAL_SPECIES; k++) {P[t.i].Metallicity[k] = (double)t.Z[k] * ssc_pow2_at_least(t.N_drawn * mu) * ldexp(1.0, -100) / M;}
            SingleStars.Z_birth[single_star_slot(P[t.i].StarIdx)] = P[t.i].Metallicity[0];
#ifdef GALSF_SINGLE_STARS_POPIII
            if(!popIII_gas_is_pristine(P[t.i].Metallicity[0], All.PopIII_Zcrit_solar * All.SolarAbundances[0])) {SingleStars.popIII[single_star_slot(P[t.i].StarIdx)] = 0;} /* collected above Z_crit: keeps its mass, labelled Pop II */
#endif
#endif
            moved.push_back(t.i); M_collected_here += M;
#ifdef OUTPUT_MASS_METAL_ENERGY_LEDGER
            ledger_add(LEDGER_M_STARS_FORMED, M); ledger_add(LEDGER_MZ_STARS_FORMED, M * P[t.i].Metallicity[0]);
#endif
#ifdef GALSF_SINGLE_STARS_SF_CRITERION /* the parent forms no star until its density is recomputed, then for a fraction of its free-fall time */
            CellP[t.parent].SingleStarSpawnBlock = 1;
            if(All.SFCooldownFreeFallFraction > 0) {CellP[t.parent].SingleStarSFCooldown = All.SFCooldownFreeFallFraction * sqrt(3. * M_PI / (32. * All.G * t.rho_form * All.cf_a3inv));}
#endif
        } else {
            P[t.i].Mass = 0; /* removed with the next rearrangement */
            single_star_record_free(P[t.i].StarIdx); P[t.i].StarIdx = 0;
        }
        double row[SSC_LOG_COLS] = {All.Time, (double)P[t.i].ID, (double)P[t.i].ID_child_number, (double)P[t.i].ID_generation,
                         t.M_drawn, (double)t.N_got * mu * UNIT_MASS_IN_SOLAR, t.R, (double)t.passes, (double)ok, (double)t.n_split, (double)t.n_remote, t.rho_form};
        rows.insert(rows.end(), row, row + SSC_LOG_COLS);
    }
    TreeMomentsStaleFlag = 1;
    if(!moved.empty()) {gizmo_motion_bound_raise(moved.data(), (int) moved.size());}
    gpu_particles_arena_invalidate();

    /* birth log on rank 0: one row per star */
    int n = (int) rows.size(), tot = 0; std::vector<int> counts(NTask), displs(NTask);
    MPI_Gather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(ThisTask == 0) {for(int t = 0; t < NTask; t++) {displs[t] = tot; tot += counts[t];}}
    std::vector<double> all(DMAX(tot, 1));
    MPI_Gatherv(rows.data(), n, MPI_DOUBLE, all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(ThisTask == 0 && tot > 0) {
        if(!FdSingleStarBirths) {
            char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE];
            snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s%s", All.OutputDir, "single_star_births.txt");
            FdSingleStarBirths = fopen(buf, (RestartFlag == 0) ? "w" : "a");
            if(FdSingleStarBirths && RestartFlag == 0) {fprintf(FdSingleStarBirths, "# time ID child generation M_drawn[Msun] M_collected[Msun] R_last[code] passes ok(1)/failed_draw(0) split_grants remote_grants rho_parent[code]\n");}
        }
        int n_fail = 0; long long n_split_tot = 0;
        for(int r = 0; r < tot / SSC_LOG_COLS; r++) {
            const double *x = &all[SSC_LOG_COLS * r];
            if(FdSingleStarBirths) {fprintf(FdSingleStarBirths, "%.16g %llu %llu %d %.17g %.17g %.10g %d %d %lld %lld %.10g\n", x[0], (unsigned long long)x[1], (unsigned long long)x[2], (int)x[3], x[4], x[5], x[6], (int)x[7], (int)x[8], (long long)x[9], (long long)x[10], x[11]);}
            if(x[8] == 0) {n_fail++;}
            n_split_tot += (long long)x[9];
        }
        if(FdSingleStarBirths) {fflush(FdSingleStarBirths);}
        printf("Single stars: %d new, %d failed draws (drawn mass not reached within %d passes; their gas returned), %lld grants from shared oversubscribed donors\n", tot / SSC_LOG_COLS, n_fail, SSC_MAX_PASSES, n_split_tot);
    }
    return M_collected_here;
}

/* make the zero-mass star i_star just spawned from gas cell i (spawn_star_from_gas) a single star of drawn mass M_drawn
   [Msun]: its record, and a place in the next collection */
void single_star_attach_to_new_star(int i_star, int i, double M_drawn)
{
    P[i_star].Type = 4;
    P[i_star].StellarAge = All.Time;
#ifdef TEST_SINGLE_STARS_BORN_WITH_GIVEN_AGE
    P[i_star].StellarAge = All.Time - All.TestStarInitAgeYr / UNIT_TIME_IN_YR; /* test only */
#endif
#ifdef DO_DENSITY_AROUND_NONGAS_PARTICLES
    P[i_star].DensityAroundParticle = CellP[i].Density;
#endif
    force_tree_note_type_presence(i_star);
    struct single_star_record r; memset(&r, 0, sizeof(r)); r.M_drawn = M_drawn;
#ifdef GALSF_SINGLE_STARS_EVOLUTION_TABLES
    r.rho_birth = (float)(CellP[i].Density * All.cf_a3inv); /* physical */
#endif
#ifdef GALSF_SINGLE_STARS_POPIII
    r.popIII = popIII_gas_is_pristine(P[i].Metallicity[0], All.PopIII_Zcrit_solar * All.SolarAbundances[0]) ? 1 : 0; /* the Pop III channel drew it */
#endif
    P[i_star].StarIdx = single_star_record_new(&r);
    ssc_births.push_back({i_star, i, CellP[i].Density});
}

#ifdef TEST_SINGLE_STARS_BIRTH_AT_GIVEN_CELLS
/* test only: at the first star-formation step, a star of the given drawn mass at each listed gas cell
   (parameter TestSingleStarBirths = "ID:M_drawn[Msun],ID:M_drawn,..."); stars_spawned = spawns so far this step;
   returns the number spawned here */
int single_star_test_births(int stars_spawned)
{
    static int done = 0;
    if(done) {return 0;}
    done = 1;
    int n = 0;
    const char *s = All.TestSingleStarBirths;
    while(*s) {
        unsigned long long id; double m; int used = 0;
        if(sscanf(s, "%llu:%lf%n", &id, &m, &used) != 2) {break;}
        for(int i = 0; i < N_gas; i++) {
            if(P[i].Type == 0 && P[i].ID == (MyIDType)id && P[i].ID_child_number == 0) {
                int i_star = spawn_star_from_gas(i, 0, stars_spawned + n);
                if(i_star >= 0) {single_star_attach_to_new_star(i_star, i, m); n++;}
            }
        }
        s += used; if(*s == ',') {s++;}
    }
    return n;
}
#endif

#endif /* GALSF_SINGLE_STARS_COLLECT_GAS_FROM_NEIGHBORS */
