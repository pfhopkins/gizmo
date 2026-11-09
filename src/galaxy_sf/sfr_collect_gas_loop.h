/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* galaxy_sf/sfr_collect_gas_loop.h -- a new single star collects its drawn IMF mass from the gas around it (LYRA-style,
 * Gutcke et al. 2021), as five neighbour-loop Specs. Design note: donors shared by several new stars are split in
 * proportion to the stars' requests; all transfers are whole multiples of one mass quantum mu, so every sum is an
 * integer sum (independent of order, thread and rank count) and the donor update is exact.
 *   count   (read-only)  E_s = sum of the available quanta a_j of the donors within R_s
 *   request (donor adds) R_j += r_sj = floor(f_s a_j)
 *   grant   (donor adds) g_sj = r_sj, or floor(r_sj A_j / R_j) if the donor is oversubscribed; G'_j += g_sj
 *   clear   (donor sets) R_j = 0, G_j += G'_j, G'_j = 0 (exactly once per donor: atomic exchange)
 *   apply   (donor sets) once per donor after the last pass: m_j -= G_j mu, G_j = 0
 * A donor's mass changes only in apply, and G_j only between passes, so every pass reads the same donor state
 * everywhere; its available mass is m_j - G_j mu. All five run where the donor lives (needs_live_neighbours).
 * R_j, G'_j, G_j live in a side buffer indexed by the local gas index (SscDonorSums), allocated at the first collection
 * and kept all-zero between collections (clear and apply drain what they touch), not in the cell struct. */
#ifndef SFR_COLLECT_GAS_LOOP_H
#define SFR_COLLECT_GAS_LOOP_H
#include <Kokkos_Core.hpp>
#include "../declarations/allvars.h"
#ifdef GALSF_SINGLE_STARS_COLLECT_GAS_FROM_NEIGHBORS
#if defined(SINK_PARTICLES) || (defined(RADTRANSFER) && !defined(GALSF_SINGLE_STARS_RT_TABLE_BANDS))
#error "GALSF_SINGLE_STARS_COLLECT_GAS_FROM_NEIGHBORS: removing gas does not yet scale radiation (outside the table bins) or sink-feedback quantities of the donor cells"
#endif
#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/mode_b_local_walker.h"

typedef __int128 ssc_i128; /* fixed-point metal and momentum sums */
#ifdef METALS
#define SSC_NZ NUM_METAL_SPECIES
#else
#define SSC_NZ 1
#endif

/* per new star, staged for the kernels */
struct SscStarIn {
    Vec3<double> Pos;
    double R;            /* collection radius of this pass (comoving) */
    double f;            /* requested fraction of each donor's available quanta */
    double q_Z, q_p;     /* fixed-point quanta of this star's metal and momentum sums */
    MyIDType id, child;  /* identity, for the grant records */
    int gen;
    int task;            /* rank that holds the star */
};

/* one grant, kept on the donor's rank so that a failed draw can be returned exactly */
struct SscGrantRecord {MyIDType id, child; int gen; int j; long long g;};

/* per local gas cell: R_j (requested this pass), G'_j (granted this pass), G_j (granted in earlier passes) */
struct SscDonorSums {long long request, grant_this_pass, grant;};

struct SscCallScalars {
    NlrCommonScalars common;
    double mu;       /* mass quantum (code units) */
    double m_floor;  /* donors keep more than this */
    double m_cap;    /* 2^53 mu: heavier cells are not donors (their ulp would exceed mu) */
    SscDonorSums *donor;         /* side buffer, by local gas index */
    int n_donor;                 /* its length (local gas cells) */
    int task;                    /* this rank, for the count of grants to stars of other ranks */
    long long *n_pairs;          /* this rank: pairs requested (sizes the grant records) */
    SscGrantRecord *records;     /* this rank: grants to its donors */
    long long *n_records, capacity;
#ifdef GALSF_FB_PHOTOIONIZATION_PER_DIRECTION
    integertime ti_current;      /* photoionised cells do not donate */
#endif
};

struct SscActiveState {
    Vec3<double> pos;
    double h_search;
    SscStarIn star;
};

struct SscDeviceContext : NeighborLoopDeviceContextBase {
    const SscStarIn *per_active_local;  /* UVM [num_active] */
};

struct SscNeighborData {
    struct particle_data *neighbor_particle;
    struct gas_cell_data *neighbor_cell;
    int j;
};

struct SscCountOut {long long E;};
struct SscGrantOut {long long N; ssc_i128 Z[SSC_NZ]; ssc_i128 p[3]; long long n_pairs, n_split, n_remote;};
struct SscNoOut {long long n;};

/* the donor's available quanta (-1 if it does not donate to a star at pos within R) */
KOKKOS_INLINE_FUNCTION
static long long ssc_available_quanta(const SscStarIn& s, const struct particle_data& Pj, const struct gas_cell_data *Cj, int j, const SscCallScalars& cs, double *m_cur_out)
{
    if(Pj.Type != 0 || Cj == nullptr || j >= cs.n_donor) {return -1;}
#ifdef GALSF_FB_FIRE_RT_HIIHEATING
    if(Cj->DelayTimeHII > 0) {return -1;} /* ionised cells do not donate */
#endif
#ifdef GALSF_FB_PHOTOIONIZATION_PER_DIRECTION
    if(cs.ti_current < Cj->PhotoionizedUntil) {return -1;}
#endif
    Vec3<double> dp;
    for(int k = 0; k < 3; k++) {dp[k] = s.Pos[k] - (double)Pj.Pos[k];}
    nearest_xyz(dp);
    if(dp.norm_sq() >= s.R * s.R) {return -1;} /* r = 0 allowed: the parent cell */
    double m_cur = (double)Pj.Mass - (double)Kokkos::atomic_load(&cs.donor[j].grant) * cs.mu; /* exact */
    if(m_cur <= cs.m_floor || (double)Pj.Mass >= cs.m_cap) {return -1;}
    *m_cur_out = m_cur;
    return (long long)floor(m_cur / cs.mu) - (long long)ceil(cs.m_floor / cs.mu);
}

KOKKOS_INLINE_FUNCTION
static long long ssc_request(const SscStarIn& s, long long a) {return (long long)floor(s.f * (double)a);}

struct SscSpecBase {
    static constexpr int                    search_mode        = MODE_B_SEARCH_ONEWAY;
    static constexpr unsigned int           neighbor_type_mask = (1u << 0);
    static constexpr mode_b_radius_policy_t radius_policy      = MODE_B_RADIUS_DEFAULT;
    static constexpr WritePattern  write_pattern   = WritePattern::ActiveReduceOnly;
    static constexpr SidxCacheKind sidx_cache_kind = SidxCacheKind::GasOnly;
    static constexpr bool needs_live_neighbours     = true;  /* every pass reads donor sums the previous pass wrote */
    static constexpr bool uses_ghost_writeback      = false; /* evaluated where the donor lives: its writes are local */
    static constexpr bool uses_ghost_write_detector = false;
    using CallScalars    = SscCallScalars;
    using ActiveData     = SscActiveState;
    using NeighborData   = SscNeighborData;
    using DeviceContext  = SscDeviceContext;
    using ScatterData    = NoScatter;
    using IdentityFields = NoIdentity;
    using IterControl    = NotIterative;
    struct Aux {const SscStarIn *host_stars; void *out; long long *n_pairs; SscGrantRecord *records; long long *n_records; long long capacity;};

    static bool        is_active(int i);
    static double      search_radius(const neighbor_loop_args& args, int active_slot, int i);
    static CallScalars populate_call_scalars(const neighbor_loop_args& args);
    static void populate_device_context(const neighbor_loop_args& args, DeviceContext& ctx);
    static void cleanup_device_context(const neighbor_loop_args& args, DeviceContext& ctx);

    KOKKOS_INLINE_FUNCTION
    static ActiveData load_active(const DeviceContext& dctx, int active_slot, int i, double h_search, const CallScalars&) {
        ActiveData a{};
        if(dctx.per_active_local != nullptr) {a.star = dctx.per_active_local[active_slot];}
        else {for(int k = 0; k < 3; k++) {a.star.Pos[k] = (double)dctx.P[i].Pos[k];} a.star.R = h_search;}
        a.pos = a.star.Pos; a.h_search = a.star.R;
        return a;
    }
    KOKKOS_INLINE_FUNCTION
    static NeighborData load_neighbor(const DeviceContext& dctx, int j, const IdentitySidecar&, const ActiveData&) {
        NeighborData n{};
        n.neighbor_particle = &dctx.P[j];
        n.neighbor_cell = (dctx.CellP != nullptr && dctx.P[j].Type == 0) ? &dctx.CellP[j] : nullptr;
        n.j = j;
        return n;
    }
};

struct SscCountSpec : SscSpecBase {
    static constexpr const char *loop_name = "single_star_collect_count";
    static constexpr ModeBEvalOMP modeb_eval_omp = ModeBEvalOMP::BitwiseReadonly;
    using AccumData = SscCountOut;
    static void apply_active_writeback(const neighbor_loop_args& args, int active_slot, int i, const AccumData& accum);
    KOKKOS_INLINE_FUNCTION static void zero_accum(AccumData& a) {a.E = 0;}
    KOKKOS_INLINE_FUNCTION static void merge_accum(AccumData& l, const AccumData& p) {l.E += p.E;}
    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& a, const NeighborData& nb, AccumData& out, NoScatter&, const CallScalars& cs) {
        double m; long long q = ssc_available_quanta(a.star, *nb.neighbor_particle, nb.neighbor_cell, nb.j, cs, &m);
        if(q > 0) {out.E += q;}
    }
};

struct SscRequestSpec : SscSpecBase {
    static constexpr const char *loop_name = "single_star_collect_request";
    static constexpr ModeBEvalOMP modeb_eval_omp = ModeBEvalOMP::EpsilonAtomic; /* integer atomic adds: bitwise in any order */
    static constexpr bool keeps_neighbour_mass_positive = true; /* writes only the integer request sum */
    using AccumData = SscNoOut;
    static void apply_active_writeback(const neighbor_loop_args& args, int active_slot, int i, const AccumData& accum);
    KOKKOS_INLINE_FUNCTION static void zero_accum(AccumData& a) {a.n = 0;}
    KOKKOS_INLINE_FUNCTION static void merge_accum(AccumData& l, const AccumData& p) {l.n += p.n;}
    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& a, const NeighborData& nb, AccumData& out, NoScatter&, const CallScalars& cs) {
        double m; long long q = ssc_available_quanta(a.star, *nb.neighbor_particle, nb.neighbor_cell, nb.j, cs, &m);
        if(q <= 0) {return;}
        long long r = ssc_request(a.star, q);
        if(r <= 0) {return;}
        Kokkos::atomic_add(&cs.donor[nb.j].request, r);
        Kokkos::atomic_add(cs.n_pairs, 1LL);
        out.n++;
    }
};

struct SscGrantSpec : SscSpecBase {
    static constexpr const char *loop_name = "single_star_collect_grant";
    static constexpr ModeBEvalOMP modeb_eval_omp = ModeBEvalOMP::EpsilonAtomic; /* integer atomic adds: bitwise in any order */
    static constexpr bool keeps_neighbour_mass_positive = true; /* writes only the integer grant sum */
    using AccumData = SscGrantOut;
    static void apply_active_writeback(const neighbor_loop_args& args, int active_slot, int i, const AccumData& accum);
    KOKKOS_INLINE_FUNCTION static void zero_accum(AccumData& a) {
        a.N = 0; a.n_pairs = 0; a.n_split = 0; a.n_remote = 0; for(int k = 0; k < 3; k++) {a.p[k] = 0;}
        for(int k = 0; k < (SSC_NZ); k++) {a.Z[k] = 0;}
    }
    KOKKOS_INLINE_FUNCTION static void merge_accum(AccumData& l, const AccumData& p) {
        l.N += p.N; l.n_pairs += p.n_pairs; l.n_split += p.n_split; l.n_remote += p.n_remote; for(int k = 0; k < 3; k++) {l.p[k] += p.p[k];}
        for(int k = 0; k < (SSC_NZ); k++) {l.Z[k] += p.Z[k];}
    }
    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& a, const NeighborData& nb, AccumData& out, NoScatter&, const CallScalars& cs) {
        const struct particle_data& Pj = *nb.neighbor_particle;
        double m_cur; long long q = ssc_available_quanta(a.star, Pj, nb.neighbor_cell, nb.j, cs, &m_cur);
        if(q <= 0) {return;}
        long long r = ssc_request(a.star, q);
        if(r <= 0) {return;}
        long long R = Kokkos::atomic_load(&cs.donor[nb.j].request);
        long long A = (long long)floor(0.5 * m_cur / cs.mu); if(A > q) {A = q;}  /* half the current mass, above the floor */
        long long g = (R <= A) ? r : (long long)(((unsigned __int128)r * (unsigned __int128)A) / (unsigned __int128)R);
        if(g <= 0) {return;}
        Kokkos::atomic_add(&cs.donor[nb.j].grant_this_pass, g);
        long long w = Kokkos::atomic_fetch_add(cs.n_records, 1LL);
        if(w < cs.capacity) {SscGrantRecord& rec = cs.records[w]; rec.id = a.star.id; rec.child = a.star.child; rec.gen = a.star.gen; rec.j = nb.j; rec.g = g;}
        out.N += g; out.n_pairs++; if(R > A) {out.n_split++;} if(a.star.task != cs.task) {out.n_remote++;}
        double dm = (double)g * cs.mu; /* exact */
        for(int k = 0; k < 3; k++) {out.p[k] += (ssc_i128)nearbyint(dm * (double)Pj.Vel[k] / a.star.q_p);}
#ifdef METALS
        for(int k = 0; k < NUM_METAL_SPECIES; k++) {out.Z[k] += (ssc_i128)nearbyint(dm * (double)Pj.Metallicity[k] / a.star.q_Z);}
#endif
    }
};

struct SscClearSpec : SscSpecBase {
    static constexpr const char *loop_name = "single_star_collect_clear";
    static constexpr ModeBEvalOMP modeb_eval_omp = ModeBEvalOMP::EpsilonAtomic; /* every writer stores the same zero; the pass's grants move once (exchange) */
    static constexpr bool keeps_neighbour_mass_positive = true;
    using AccumData = SscNoOut;
    static void apply_active_writeback(const neighbor_loop_args&, int, int, const AccumData&) {}
    KOKKOS_INLINE_FUNCTION static void zero_accum(AccumData& a) {a.n = 0;}
    KOKKOS_INLINE_FUNCTION static void merge_accum(AccumData& l, const AccumData& p) {l.n += p.n;}
    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& a, const NeighborData& nb, AccumData&, NoScatter&, const CallScalars& cs) {
        if(nb.neighbor_cell == nullptr || nb.neighbor_particle->Type != 0 || nb.j >= cs.n_donor) {return;}
        Vec3<double> dp; for(int k = 0; k < 3; k++) {dp[k] = a.star.Pos[k] - (double)nb.neighbor_particle->Pos[k];}
        nearest_xyz(dp);
        if(dp.norm_sq() >= a.star.R * a.star.R) {return;}
        Kokkos::atomic_store(&cs.donor[nb.j].request, 0LL);
        long long g = Kokkos::atomic_exchange(&cs.donor[nb.j].grant_this_pass, 0LL);
        if(g > 0) {Kokkos::atomic_add(&cs.donor[nb.j].grant, g);}
    }
};

struct SscApplySpec : SscSpecBase {
    static constexpr const char *loop_name = "single_star_collect_apply";
    static constexpr ModeBEvalOMP modeb_eval_omp = ModeBEvalOMP::EpsilonAtomic; /* each donor is taken by the one pair that exchanges its grant sum */
    using AccumData = SscNoOut;
    static void apply_active_writeback(const neighbor_loop_args&, int, int, const AccumData&) {}
    KOKKOS_INLINE_FUNCTION static void zero_accum(AccumData& a) {a.n = 0;}
    KOKKOS_INLINE_FUNCTION static void merge_accum(AccumData& l, const AccumData& p) {l.n += p.n;}
    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& a, const NeighborData& nb, AccumData& out, NoScatter&, const CallScalars& cs) {
        if(nb.neighbor_cell == nullptr || nb.neighbor_particle->Type != 0 || nb.j >= cs.n_donor) {return;}
        Vec3<double> dp; for(int k = 0; k < 3; k++) {dp[k] = a.star.Pos[k] - (double)nb.neighbor_particle->Pos[k];}
        nearest_xyz(dp);
        if(dp.norm_sq() >= a.star.R * a.star.R) {return;}
        long long G = Kokkos::atomic_exchange(&cs.donor[nb.j].grant, 0LL);
        if(G <= 0) {return;}
        struct particle_data& Pj = *nb.neighbor_particle;
        struct gas_cell_data& Cj = *nb.neighbor_cell;
        double m_old = (double)Pj.Mass, dm = (double)G * cs.mu, m_new = m_old - dm; /* exact */
        double frac = m_new / m_old; /* conserved MHD quantities leave with the mass (merge_split parent side) */
#ifdef MAGNETIC
        Cj.B *= frac; Cj.BPred *= frac; Cj.DtB *= frac; Cj.divB *= frac;
#ifdef DIVBCLEANING_DEDNER
        Cj.Phi *= frac; Cj.PhiPred *= frac; Cj.DtPhi *= frac; Cj.DtB_PhiCorr *= frac;
#endif
#endif
#ifdef COSMIC_RAY_FLUID
        for(int kb = 0; kb < N_CR_PARTICLE_BINS; kb++) { /* CR energy and flux are extensive: the donor keeps its specific values */
            Cj.CosmicRayEnergy[kb] *= frac; Cj.CosmicRayEnergyPred[kb] *= frac; Cj.DtCosmicRayEnergy[kb] *= frac;
            Cj.CosmicRayFlux[kb] *= frac; Cj.CosmicRayFluxPred[kb] *= frac;
#ifdef CRFLUID_EVOLVE_SPECTRUM
            Cj.CosmicRay_Number_in_Bin[kb] *= frac; Cj.DtCosmicRay_Number_in_Bin[kb] *= frac;
#endif
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
            for(int w = 0; w < 2; w++) {Cj.CosmicRayAlfvenEnergy[kb][w] *= frac; Cj.CosmicRayAlfvenEnergyPred[kb][w] *= frac; Cj.DtCosmicRayAlfvenEnergy[kb][w] *= frac;}
#endif
        }
#endif
#ifdef RADTRANSFER
        for(int k = 0; k < N_RT_FREQ_BINS; k++) {Cj.Rad_E_gamma[k] *= frac; Cj.Rad_E_gamma_Pred[k] *= frac;} /* energy per cell: the cell's volume shrinks with its mass */
#ifdef RT_EVOLVE_FLUX
        for(int k = 0; k < N_RT_FREQ_BINS; k++) {Cj.Rad_Flux[k] *= frac; Cj.Rad_Flux_Pred[k] *= frac;}
#endif
#endif
        (void)frac;
        Pj.Mass = m_new; Cj.Mass = m_new;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        Cj.MassTrue -= dm;
#endif
        out.n++;
    }
};

#endif /* GALSF_SINGLE_STARS_COLLECT_GAS_FROM_NEIGHBORS */
#endif /* SFR_COLLECT_GAS_LOOP_H */
