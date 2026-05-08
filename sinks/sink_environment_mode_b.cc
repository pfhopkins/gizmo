/* sinks/sink_environment_mode_b.cc — Mode B SPIKE for sink_environment_loop.
 *
 * SPIKE DUPLICATE of the GPU lambda body in sink_environment_gpu.cc:152-275.
 * TODO(runner-extraction): consolidate to a shared KOKKOS_INLINE_FUNCTION.
 * REQUIRED before extending Mode B to a second caller. See contract memory.
 *
 * Architecture:
 *   - sink_env1_pair_kernel_host(): one pair iteration. Pure: reads (query, P[j],
 *     CellP[j]?, scalars), writes only into out. No MPI, no GPU state.
 *   - sink_env1_evaluate_one_query_local(): for one query, walk + iterate +
 *     accumulate into a single sink_env_gpu_out. The walker is Mode B local
 *     (tree or brute under oracle).
 *   - sink_env1_mode_b_evaluate(): collective. Pack queries, exchange via P2P
 *     transport (excluding self), self-pairs locally, merge replies, fill
 *     nl_outs[]. No ghost prep, no GPU NGL, no SIDX.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "../mesh/mode_b_local_walker.h"
#include "../mesh/mode_b_p2p_transport.h"
#include "../mesh/ghost_writeback.h"           /* ghost_get_num_local */
#include "sink_environment_mode_b.h"

/* Forward decl of a tiny shim defined in mesh/gpu_neighbor_list.cc (compiled
 * with nvcc, so it can include <Kokkos_Core.hpp>). Calls Kokkos::fence() —
 * waits for pending GPU work and makes pending writes coherent for the host.
 * Used as an A/B diagnostic here for the XVAL freshness hypothesis. */
extern "C" void gpu_host_fence(void);
extern "C" void gpu_xval_readback_pv(const int *indices, int n,
                                     double *out_pv, long long *out_ti);

#ifdef SINK_PARTICLES

/* -----------------------------------------------------------------------
 * Env flag (cached static).
 * --------------------------------------------------------------------- */
int sink_env1_mode_b_env_enabled(void)
{
    static int cached = -1;
    if(cached < 0) {
        const char *e = getenv("GIZMO_MODE_B_SINK_ENV1");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached;
}

static int xval_dump_enabled(void)
{
    static int cached = -1;
    if(cached < 0) {
        const char *e = getenv("GIZMO_MODE_B_XVAL_DUMP");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached;
}

static int phase0_diag_enabled(void)
{
    static int cached = -1;
    if(cached < 0) {
        const char *e = getenv("GIZMO_PHASE0_DIAG");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached;
}

/* File-scope call counter for the MODEB_XVAL_NB diagnostic so the per-query
 * dump in sink_env1_evaluate_one_query_local knows which call it's in. */
static long long g_mode_b_eval_call_id = 0;

/* MODEB_XVAL_NB diagnostic helper. Emits the per-query header + per-j detail
 * lines AFTER lazy-drift so Pos/Vel reflect the same Ti_Current snapshot the
 * pair kernel uses. Replay-grade format: includes i_vel, Pos[j], Vel[j] so
 * J = m * (Pos[j]-i_pos) x (Vel[j]-i_vel) can be reconstructed offline.
 * First-call only; gated by GIZMO_MODE_B_XVAL_NB_DUMP=1. */
static void mode_b_xval_nb_dump(const struct particle_data *P,
                                const sink_env1_query_t& q,
                                const int *cand, int n_cand)
{
    static const char *xv_nb_env = getenv("GIZMO_MODE_B_XVAL_NB_DUMP");
    static const int xv_nb_on = (xv_nb_env && xv_nb_env[0] == '1') ? 1 : 0;
    if(!xv_nb_on || g_mode_b_eval_call_id != 1 || n_cand <= 0) return;

    /* In-process CPU-vs-GPU readback diagnostic. Same memory (UVM-aliased
     * arena), but read once via a host pointer and once via a Kokkos kernel
     * that touches the same UVM bytes. If outputs differ, freshness/coherence
     * issue. Includes the active i (index q.origin_local_idx) at the head of
     * the index list so we can also see active i's CPU vs GPU view. */
    std::vector<int> idx_list;
    idx_list.reserve(n_cand + 1);
    idx_list.push_back(q.origin_local_idx);
    for(int i = 0; i < n_cand; i++) idx_list.push_back(cand[i]);
    std::vector<double> out_pv(9 * idx_list.size());
    std::vector<long long> out_ti(idx_list.size());
    gpu_xval_readback_pv(idx_list.data(), (int)idx_list.size(),
                         out_pv.data(), out_ti.data());
    printf("MODEB_XVAL_NB rank=%d call=1 active=%d path=mode_b "
           "h_search=%.17g i_pos=%.17g,%.17g,%.17g "
           "i_vel=%.17g,%.17g,%.17g i_id=%llu n_j=%d\n",
           ThisTask, q.origin_local_idx, q.h_search,
           q.pos[0], q.pos[1], q.pos[2],
           q.vel[0], q.vel[1], q.vel[2],
           (unsigned long long)q.id, n_cand);
    /* Active i CPU-vs-GPU comparison. idx_list[0] is the active. */
    {
        int ii = q.origin_local_idx;
        double *pv = out_pv.data() + 0;
        printf("MODEB_XVAL_I_CMP rank=%d call=1 active=%d ii=%d ID=%llu Type=%d "
               "Ti_current_host=%lld Ti_current_gpu=%lld "
               "PosHost=%.17g,%.17g,%.17g PosGpu=%.17g,%.17g,%.17g "
               "VelHost=%.17g,%.17g,%.17g VelGpu=%.17g,%.17g,%.17g "
               "VelPredGpu=%.17g,%.17g,%.17g\n",
               ThisTask, ii, ii, (unsigned long long)P[ii].ID, (int)P[ii].Type,
               (long long)P[ii].Ti_current, out_ti[0],
               (double)P[ii].Pos[0], (double)P[ii].Pos[1], (double)P[ii].Pos[2],
               pv[0], pv[1], pv[2],
               (double)P[ii].Vel[0], (double)P[ii].Vel[1], (double)P[ii].Vel[2],
               pv[3], pv[4], pv[5],
               pv[6], pv[7], pv[8]);
    }
    for(int i = 0; i < n_cand; i++) {
        int j = cand[i];
        double dx = (double)P[j].Pos[0] - q.pos[0];
        double dy = (double)P[j].Pos[1] - q.pos[1];
        double dz = (double)P[j].Pos[2] - q.pos[2];
        NEAREST_XYZ(dx, dy, dz, 1);
        double r2 = dx*dx + dy*dy + dz*dz;
        printf("MODEB_XVAL_NB_J rank=%d call=1 active=%d path=mode_b "
               "j=%d Type=%d Mass=%.17g KernelRadius=%.17g r2=%.17g "
               "Pos=%.17g,%.17g,%.17g Vel=%.17g,%.17g,%.17g ID=%llu\n",
               ThisTask, q.origin_local_idx, j, (int)P[j].Type,
               (double)P[j].Mass, (double)P[j].KernelRadius, r2,
               (double)P[j].Pos[0], (double)P[j].Pos[1], (double)P[j].Pos[2],
               (double)P[j].Vel[0], (double)P[j].Vel[1], (double)P[j].Vel[2],
               (unsigned long long)P[j].ID);
        /* Same j read via GPU. idx_list[1+i] = cand[i]. Only print MISMATCHES
         * (host bytes != gpu bytes) to keep dump tractable; suppress the
         * 99% byte-equal case. */
        const int slot = 1 + i;
        const double *pv = out_pv.data() + 9*slot;
        bool pos_diff = (pv[0] != (double)P[j].Pos[0]) || (pv[1] != (double)P[j].Pos[1])
                        || (pv[2] != (double)P[j].Pos[2]);
        bool vel_diff = (pv[3] != (double)P[j].Vel[0]) || (pv[4] != (double)P[j].Vel[1])
                        || (pv[5] != (double)P[j].Vel[2]);
        bool ti_diff  = (out_ti[slot] != (long long)P[j].Ti_current);
        if(pos_diff || vel_diff || ti_diff) {
            printf("MODEB_XVAL_NB_J_GPUDIFF rank=%d call=1 active=%d j=%d ID=%llu "
                   "Type=%d Ti_h=%lld Ti_g=%lld "
                   "PosHost=%.17g,%.17g,%.17g PosGpu=%.17g,%.17g,%.17g "
                   "VelHost=%.17g,%.17g,%.17g VelGpu=%.17g,%.17g,%.17g\n",
                   ThisTask, q.origin_local_idx, j, (unsigned long long)P[j].ID,
                   (int)P[j].Type, (long long)P[j].Ti_current, out_ti[slot],
                   (double)P[j].Pos[0], (double)P[j].Pos[1], (double)P[j].Pos[2],
                   pv[0], pv[1], pv[2],
                   (double)P[j].Vel[0], (double)P[j].Vel[1], (double)P[j].Vel[2],
                   pv[3], pv[4], pv[5]);
        }
    }
    /* Also emit a summary count so we know whether ANY mismatches were found. */
    int n_pos_mm = 0, n_vel_mm = 0, n_ti_mm = 0;
    for(int i = 0; i < n_cand; i++) {
        int j = cand[i];
        const double *pv = out_pv.data() + 9*(1+i);
        if(pv[0] != (double)P[j].Pos[0] || pv[1] != (double)P[j].Pos[1] || pv[2] != (double)P[j].Pos[2]) n_pos_mm++;
        if(pv[3] != (double)P[j].Vel[0] || pv[4] != (double)P[j].Vel[1] || pv[5] != (double)P[j].Vel[2]) n_vel_mm++;
        if(out_ti[1+i] != (long long)P[j].Ti_current) n_ti_mm++;
    }
    printf("MODEB_XVAL_GPUDIFF_SUMMARY rank=%d call=1 active=%d n_cand=%d "
           "n_pos_diff=%d n_vel_diff=%d n_ti_diff=%d\n",
           ThisTask, q.origin_local_idx, n_cand, n_pos_mm, n_vel_mm, n_ti_mm);
    fflush(stdout);
}

/* -----------------------------------------------------------------------
 * Cosmology + physical scalars the pair body needs. Captured once per
 * evaluator call from globals; passed by value into the kernel. No
 * implicit globals reach inside the kernel.
 * --------------------------------------------------------------------- */
struct sink_env1_scalars_t {
    double cf_atime;
    double cf_a2inv;
    double cf_a3inv;
    double G_val;
    double sink_radius_grav;
};

static inline sink_env1_scalars_t snapshot_scalars(void)
{
    return {All.cf_atime, All.cf_a2inv, All.cf_a3inv, All.G,
            SinkParticle_GravityKernelRadius};
}

/* -----------------------------------------------------------------------
 * Query packing: pull the active sink's pair-body inputs into a typed
 * payload. h_search comes from nl_radii[a] (NOT P[ii].KernelRadius).
 * --------------------------------------------------------------------- */
static sink_env1_query_t pack_query(struct particle_data *P,
                                    int ii, double h_search,
                                    int local_idx,
                                    const sink_env1_scalars_t& sc)
{
    sink_env1_query_t q;
    q.pos          = P[ii].Pos;
    q.vel          = P[ii].Vel;
    q.id           = P[ii].ID;
    q.h_search     = h_search;
#if (ADAPTIVE_GRAVSOFT_FORALL & 32)
    q.ags_h        = (double)P[ii].AGS_KernelRadius;
#else
    q.ags_h        = sc.sink_radius_grav;
#endif
#if defined(SINK_GRAVCAPTURE_GAS) || (SINK_GRAVACCRETION == 8)
    q.mass         = (double)P[ii].Mass;
#endif
#if defined(SINK_GRAVCAPTURE_FIXEDSINKRADIUS)
    q.sink_radius  = (double)P[ii].SinkRadius;
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
    for(int kv = 0; kv < 3; kv++) q.sink_angmom[kv] = P[ii].Sink_Specific_AngMom[kv];
#endif
    q.origin_local_idx = local_idx;
    q.origin_rank      = ThisTask;
    return q;
}

/* -----------------------------------------------------------------------
 * Pair kernel — SPIKE DUPLICATE of sink_environment_gpu.cc:155-275.
 *
 * Pure: reads only the args, writes only into `out`. Caller is
 * responsible for the per-active early-return (Mass<=0 || h<=0); this
 * function assumes valid inputs.
 *
 * Identical math to the GPU lambda. If you change the GPU lambda, you
 * MUST mirror the change here (or, ideally, do the runner extraction
 * that makes both call into a single KOKKOS_INLINE_FUNCTION).
 * --------------------------------------------------------------------- */
static inline void sink_env1_pair_kernel_host(const sink_env1_query_t& q,
                                              const struct particle_data& kp_j,
                                              const struct gas_cell_data* kc_j,
                                              const sink_env1_scalars_t& sc,
                                              struct sink_env_gpu_out& out)
{
    /* Self-skip / mass / type-5 filter — matches GPU line 155. */
    if(kp_j.Mass <= 0 || kp_j.Type == 5 || kp_j.ID == q.id) return;

    const double h_i  = q.h_search;
    const double hinv  = 1.0 / h_i;
    const double hinv3 = hinv * hinv * hinv;
    const double ags_h_i = q.ags_h;

    /* dP, dv with periodic wrap. Sign convention matches GPU (j - i). */
    Vec3<double> dP;
    dP[0] = (double)kp_j.Pos[0] - q.pos[0];
    dP[1] = (double)kp_j.Pos[1] - q.pos[1];
    dP[2] = (double)kp_j.Pos[2] - q.pos[2];
    Vec3<double> dv;
    dv[0] = (double)kp_j.Vel[0] - q.vel[0];
    dv[1] = (double)kp_j.Vel[1] - q.vel[1];
    dv[2] = (double)kp_j.Vel[2] - q.vel[2];
    nearest_xyz(dP, -1);
    /* NGB_SHEARBOX_BOUNDARY_VELCORR — matches GPU body. */
    NGB_SHEARBOX_BOUNDARY_VELCORR_(q.pos, kp_j.Pos, dv, -1);

    const double wt = (double)kp_j.Mass;

#ifdef SINK_REPOSITION_ON_POTMIN
    if(kp_j.Type != 0 && kp_j.Type != 5) {
        double rfac  = dP.norm_sq() * (10.0 / (h_i * h_i)
                                     + 0.1 / (sc.sink_radius_grav * sc.sink_radius_grav));
        double wtfac = wt / (1.0 + rfac);
        if((MyFloat)kp_j.Mass > out.DF_mmax_particles) out.DF_mmax_particles = (MyFloat)kp_j.Mass;
        for(int kv = 0; kv < 3; kv++) out.DF_mean_vel[kv] += wtfac * dv[kv];
        out.DF_rms_vel += wtfac;
        out.DF_rms_vel += wtfac;
        out.DF_rms_vel += wtfac;
    }
#endif

    if(kp_j.Type == 0) {
        out.Mgas_in_Kernel += wt;
        if(kc_j) { out.Sink_SurroudingGasInternalEnergy += wt * kc_j->InternalEnergy; }
        Vec3<double> J_gas = cross(dP, dv);
        for(int kv = 0; kv < 3; kv++) out.Jgas_in_Kernel[kv] += wt * J_gas[kv];
#if defined(SINK_OUTPUT_MOREINFO)
        if(kc_j) { out.Sfr_in_Kernel += kc_j->Sfr; }
#endif
#if (SINK_GRAVACCRETION >= 5) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINGLE_STAR_TIMESTEPPING)
        for(int kv = 0; kv < 3; kv++) out.Sink_SurroundingGasVel[kv] += wt * dv[kv];
#endif
#ifdef JET_DIRECTION_FROM_KERNEL_AND_SINK
        for(int kv = 0; kv < 3; kv++) out.Sink_SurroundingGasCOM[kv] += wt * dP[kv];
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS) || defined(SINK_RETURN_BFLUX)
        {
            double u_wb = dP.norm() / DMAX(h_i, (double)kp_j.KernelRadius);
            double wk_wb = 0, dwk_wb = 0;
            if(u_wb < 1) { kernel_main(u_wb, 1.0, 1.0, &wk_wb, &dwk_wb, -1); }
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
            double r2j = dP.norm_sq();
            double Lrj = dot(q.sink_angmom, dP);
            Vec3<double> Ang_pass = q.sink_angmom * r2j - dP * Lrj;
            for(int kv = 0; kv < 3; kv++)
                out.angmom_prepass_sum_for_passback[kv] += wk_wb * wt * Ang_pass[kv];
#endif
#if defined(SINK_RETURN_BFLUX)
            out.kernel_norm_topass_in_swallowloop += wk_wb;
#endif
        }
#endif
#if (SINK_GRAVACCRETION == 8)
        if(kc_j) {
            double u_h = dP.norm() / h_i;
            double wk_h = 0, dwk_h = 0;
            if(u_h < 1) { kernel_main(u_h, hinv3, hinv3 * hinv, &wk_h, &dwk_h, -1); }
            double rj = u_h * h_i * sc.cf_atime;
            double csj = kc_j->effective_soundspeed();
            double vdotrj = -dot(dP, dv);
            double vr_mdot = 4 * M_PI * wt * (wk_h * sc.cf_a3inv) * rj * vdotrj;
            if(rj < sc.sink_radius_grav * sc.cf_atime) {
                double bondi_mdot = 4 * M_PI * sc.G_val * sc.G_val * q.mass * q.mass
                    / pow(csj * csj + dv.norm_sq() * sc.cf_a2inv, 1.5)
                    * wt * (wk_h * sc.cf_a3inv);
                vr_mdot = DMAX(vr_mdot, bondi_mdot);
                out.hubber_mdot_bondi_limiter += bondi_mdot;
            }
            out.hubber_mdot_vr_estimator    += vr_mdot;
            out.hubber_mdot_disk_estimator  += wt * wk_h * sqrt(rj) / (kc_j->Density * csj * csj);
        }
#endif
    } else if(kp_j.Type == 4 ||
              ((kp_j.Type == 2 || kp_j.Type == 3) && !All.ComovingIntegrationOn)) {
        out.Mstar_in_Kernel += wt;
        Vec3<double> J_star = cross(dP, dv);
        for(int kv = 0; kv < 3; kv++) out.Jstar_in_Kernel[kv] += wt * J_star[kv];
    } else {
        out.Malt_in_Kernel += wt;
        Vec3<double> J_alt = cross(dP, dv);
        for(int kv = 0; kv < 3; kv++) out.Jalt_in_Kernel[kv] += wt * J_alt[kv];
    }

    /* SINK_GRAVCAPTURE_GAS branch — needs SwallowID j-side read; kp_j has it.
     * Note: the GPU lambda also does atomic_fetch_min on kp[j].SwallowTime
     * under SINGLE_STAR_SINK_DYNAMICS. fire_m11i has neither flag, so this
     * block is INACTIVE in the spike's target build. Leaving the host
     * version conservatively unimplemented under those flags would risk
     * silent divergence if Mode B is later enabled in a build that has
     * them — so the host helper REFUSES to run if SINGLE_STAR_SINK_DYNAMICS
     * is defined. The dispatch in sink_environment.cc must verify. */
#if defined(SINGLE_STAR_SINK_DYNAMICS)
#error "Mode B sink_env1 spike does not implement SINGLE_STAR_SINK_DYNAMICS j-side scatter. \
Disable GIZMO_MODE_B_SINK_ENV1 for builds with SINGLE_STAR_SINK_DYNAMICS, \
or implement the SwallowTime writeback channel before enabling."
#endif

#ifdef SINK_GRAVCAPTURE_GAS
#ifdef GRAIN_FLUID
    if(kp_j.Mass > 0 && (kp_j.Type == 0 || ((1 << kp_j.Type) & GRAIN_PTYPES)))
#else
    if(kp_j.Mass > 0 && kp_j.Type == 0)
#endif
    {
        double dr_code = dP.norm();
        double vrel = dv.norm() / sc.cf_atime;
#if defined(MAGNETIC) && defined(GRAIN_LORENTZFORCE)
        if((1 << kp_j.Type) & GRAIN_PTYPES) {
            Vec3<double> B_vec;
            for(int kv = 0; kv < 3; kv++) B_vec[kv] = (double)kp_j.Gas_B[kv];
            double vrel_dot = dot(dv, B_vec), bmag2 = B_vec.norm_sq();
            vrel = (fabs(vrel_dot) / sqrt(bmag2)) / sc.cf_atime;
        }
#endif
        struct gas_cell_data kc_j_local;
        if(kp_j.Type == 0 && kc_j) { kc_j_local = *kc_j; }
        else { memset(&kc_j_local, 0, sizeof(kc_j_local)); }
        double vbound = sink_vesc_gpu(kp_j, kc_j_local, q.mass, dr_code, ags_h_i);
        if(vrel < vbound) {
            double local_sink_radius = sc.sink_radius_grav;
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
            local_sink_radius = q.sink_radius;
            double spec_mom = dot(dv, dP);
            double r2 = dP.norm_sq();
            spec_mom = r2 * vrel * vrel - spec_mom * spec_mom * sc.cf_a2inv;
            if(spec_mom >= sc.G_val * (q.mass + kp_j.Mass) * local_sink_radius) return;
#endif
            if(sink_check_boundedness_gpu(kp_j, kc_j_local, vrel, vbound, dr_code, local_sink_radius) == 1) {
                if(kp_j.SwallowID < q.id) { out.mass_to_swallow_edd += kp_j.Mass; }
            }
        }
    }
#endif /* SINK_GRAVCAPTURE_GAS */
}

/* -----------------------------------------------------------------------
 * Run one query against the local pool (called on the rank that owns
 * the pool, whether that's the active rank for self-pairs or a peer for
 * remote queries). Walker is Mode B local; oracle adds brute walk.
 *
 * Caller must have zeroed out before calling. Returns counters for the
 * PHASE0_MODEB_NGL diagnostic.
 * --------------------------------------------------------------------- */
struct query_eval_diag_t {
    int n_candidates;       /* candidates returned by walker */
    int n_accepted;         /* candidates passing pair-kernel filter (i.e. j contributed) */
};

static query_eval_diag_t
sink_env1_evaluate_one_query_local(struct particle_data *P,
                                   struct gas_cell_data *CellP,
                                   const sink_env1_query_t& q,
                                   const sink_env1_scalars_t& sc,
                                   struct sink_env_gpu_out& out)
{
    query_eval_diag_t d{0, 0};

    /* Per-active validity check (matches GPU lambda's early-return at line 129). */
    if(q.h_search <= 0) return d;
    /* Mass>0 check: handled at pack_query time conceptually; an explicitly
     * invalid active wouldn't appear in nl_active[]. But guard here too. */

    /* Walker buffer — sized to worst case (num_local). For SYMMETRIC search
     * the always-open walker has no h-bound pre-pruning, so legitimate
     * neighbor count can be a substantial fraction of num_local for sinks
     * embedded in dense regions. Allocating num_local ints (~50MB on
     * fire_m11i) per query is fine for tiny-N spike — worst case total
     * memory = Nactive × num_local × sizeof(int) = ~200MB on 96GB GH200. */
    const int num_local = ghost_get_num_local();
    if(num_local <= 0) return d;   /* no local pool */
    std::vector<int> candidates(num_local);
    double pos_arr[3] = {q.pos[0], q.pos[1], q.pos[2]};
    /* sink_env1 uses MODE_B_RADIUS_DEFAULT: SYMMETRIC on gas (Type=0) via
     * KernelRadius; non-gas types fall back to ONEWAY (their P[].KernelRadius
     * is not physically meaningful as a symmetric-search radius). This matches
     * old GIZMO's gas-only Hmax convention. (Phil + codex 2026-05-07.) */
    int n_cand = mode_b_local_neighbor_walk(pos_arr,
                                            q.h_search,
                                            (unsigned int)SINK_NEIGHBOR_BITFLAG,
                                            MODE_B_SEARCH_SYMMETRIC,
                                            MODE_B_RADIUS_DEFAULT,
                                            candidates.data(), (int)candidates.size());
    if(n_cand < 0) {
        /* Even num_local was insufficient — shouldn't happen with this cap
         * unless the walker bug itself returns duplicates. Abort loudly. */
        fprintf(stderr,
                "[mode_b sink_env1 ABORT rank=%d] walker overflowed even at "
                "num_local=%d capacity. Walker bug?\n",
                ThisTask, num_local);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    d.n_candidates = n_cand;

    /* MODEB_XVAL_NB diagnostic now fires AFTER lazy-drift via helper calls
     * below (oracle and non-oracle paths each emit their own post-drift
     * snapshot). Emitting pre-drift gave the wrong snapshot for the J
     * reconstruction the dump exists to support. */

    /* CRITICAL: Mode B must honor the GPU NGL lazy-drift contract:
     *   1. walk candidates on whatever P[j] state exists (may be slightly stale)
     *   2. drift each candidate j to All.Ti_Current via drift_particle()
     *   3. THEN run pair kernel reading drifted P[j] / CellP[j]
     *
     * Skipping step 2 was the cause of the XVAL Jalt 13% divergence (env-on
     * vs env-off): GPU NGL drifts j's between walk and kernel; old Mode B
     * code did not, so the pair body read stale positions. (gpu_ngb_list_build:
     * 1542-1580 has the original contract.) */

    if(mode_b_oracle_enabled()) {
        /* Oracle path: collect BOTH candidate sets BEFORE any P[] mutation.
         * Otherwise the brute walk would test against drifted state while
         * the tree walk tested against pre-drift state, and the oracle
         * itself would pollute what it's measuring. */
        std::vector<int> brute_cand(num_local);
        int n_brute = mode_b_local_brute_walk(pos_arr,
                                              q.h_search,
                                              (unsigned int)SINK_NEIGHBOR_BITFLAG,
                                              MODE_B_SEARCH_SYMMETRIC,
                                              MODE_B_RADIUS_DEFAULT,
                                              brute_cand.data(), (int)brute_cand.size());
        if(n_brute < 0) {
            fprintf(stderr,
                    "[mode_b ORACLE rank=%d] brute walker overflowed (>num_local=%d). "
                    "Skipping oracle for this query.\n", ThisTask, num_local);
        } else {
            /* Drift the UNION of (tree, brute). drift_particle dedupes via
             * its time1==time0 early-return, so passing duplicates is fine. */
            mode_b_lazy_drift_candidates(candidates.data(), n_cand);
            mode_b_lazy_drift_candidates(brute_cand.data(), n_brute);

            /* Post-drift NB dump: tree-walker candidates only (the path under
             * test). State is now at Ti_Current matching what the pair kernel
             * will read. */
            mode_b_xval_nb_dump(P, q, candidates.data(), n_cand);

            /* Now P[] is at Ti_Current for the union. Compute both accums. */
            struct sink_env_gpu_out out_tree;
            memset(&out_tree, 0, sizeof(out_tree));
            for(int i = 0; i < n_cand; i++) {
                int j = candidates[i];
                const struct gas_cell_data *kc_j_ptr =
                    (P[j].Type == 0 && CellP) ? &CellP[j] : nullptr;
                sink_env1_pair_kernel_host(q, P[j], kc_j_ptr, sc, out_tree);
            }
            struct sink_env_gpu_out out_brute;
            memset(&out_brute, 0, sizeof(out_brute));
            for(int i = 0; i < n_brute; i++) {
                int j = brute_cand[i];
                const struct gas_cell_data *kc_j_ptr =
                    (P[j].Type == 0 && CellP) ? &CellP[j] : nullptr;
                sink_env1_pair_kernel_host(q, P[j], kc_j_ptr, sc, out_brute);
            }

            /* Field-by-field compare with FP tolerance. */
            const double tol = 1e-9;
            bool mismatch = false;
            #define CHECK(field) do { \
                double a = (double)out_tree.field, b = (double)out_brute.field; \
                double denom = fmax(fabs(a), fabs(b)) + 1e-30; \
                if(fabs(a - b) / denom > tol) mismatch = true; \
            } while(0)
            CHECK(Sink_SurroudingGasInternalEnergy);
            CHECK(Mgas_in_Kernel); CHECK(Mstar_in_Kernel); CHECK(Malt_in_Kernel);
            for(int kv = 0; kv < 3; kv++) {
                CHECK(Jgas_in_Kernel[kv]); CHECK(Jstar_in_Kernel[kv]); CHECK(Jalt_in_Kernel[kv]);
            }
            #undef CHECK
            if(mismatch) {
                fprintf(stderr,
                        "[mode_b ORACLE MISMATCH rank=%d caller=sink_env1 active_id=%llu] "
                        "tree: Mgas=%g Mstar=%g Jalt=%g,%g,%g, brute: Mgas=%g Mstar=%g Jalt=%g,%g,%g "
                        "(n_cand_tree=%d n_brute=%d)\n",
                        ThisTask, (unsigned long long)q.id,
                        (double)out_tree.Mgas_in_Kernel, (double)out_tree.Mstar_in_Kernel,
                        (double)out_tree.Jalt_in_Kernel[0], (double)out_tree.Jalt_in_Kernel[1], (double)out_tree.Jalt_in_Kernel[2],
                        (double)out_brute.Mgas_in_Kernel, (double)out_brute.Mstar_in_Kernel,
                        (double)out_brute.Jalt_in_Kernel[0], (double)out_brute.Jalt_in_Kernel[1], (double)out_brute.Jalt_in_Kernel[2],
                        n_cand, n_brute);
                fflush(stderr);
            }
            /* Use tree result (the path under test). */
            out = out_tree;
            d.n_accepted = n_cand;
            return d;
        }
        /* Fall through to non-oracle path on brute overflow. */
    }

    /* Non-oracle path: drift candidates, then accumulate. */
    mode_b_lazy_drift_candidates(candidates.data(), n_cand);
    mode_b_xval_nb_dump(P, q, candidates.data(), n_cand);
    for(int i = 0; i < n_cand; i++) {
        int j = candidates[i];
        const struct gas_cell_data *kc_j_ptr =
            (P[j].Type == 0 && CellP) ? &CellP[j] : nullptr;
        sink_env1_pair_kernel_host(q, P[j], kc_j_ptr, sc, out);
    }
    d.n_accepted = n_cand;
    return d;
}

/* -----------------------------------------------------------------------
 * Merge: out_dst <- out_dst (op) out_src, where op is per-field
 * (ASSIGN_ADD for additive fields, MAX for DF_mmax_particles).
 * --------------------------------------------------------------------- */
static inline void merge_into(struct sink_env_gpu_out& dst,
                              const struct sink_env_gpu_out& src)
{
    dst.Sink_SurroudingGasInternalEnergy += src.Sink_SurroudingGasInternalEnergy;
    dst.Mgas_in_Kernel  += src.Mgas_in_Kernel;
    dst.Mstar_in_Kernel += src.Mstar_in_Kernel;
    dst.Malt_in_Kernel  += src.Malt_in_Kernel;
    for(int kv = 0; kv < 3; kv++) {
        dst.Jgas_in_Kernel[kv]  += src.Jgas_in_Kernel[kv];
        dst.Jstar_in_Kernel[kv] += src.Jstar_in_Kernel[kv];
        dst.Jalt_in_Kernel[kv]  += src.Jalt_in_Kernel[kv];
    }
#ifdef SINK_REPOSITION_ON_POTMIN
    dst.DF_rms_vel += src.DF_rms_vel;
    for(int kv = 0; kv < 3; kv++) dst.DF_mean_vel[kv] += src.DF_mean_vel[kv];
    if(src.DF_mmax_particles > dst.DF_mmax_particles) dst.DF_mmax_particles = src.DF_mmax_particles;
#endif
#if defined(SINK_OUTPUT_MOREINFO)
    dst.Sfr_in_Kernel += src.Sfr_in_Kernel;
#endif
#if (SINK_GRAVACCRETION >= 5) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINGLE_STAR_TIMESTEPPING)
    for(int kv = 0; kv < 3; kv++) dst.Sink_SurroundingGasVel[kv] += src.Sink_SurroundingGasVel[kv];
#endif
#ifdef JET_DIRECTION_FROM_KERNEL_AND_SINK
    for(int kv = 0; kv < 3; kv++) dst.Sink_SurroundingGasCOM[kv] += src.Sink_SurroundingGasCOM[kv];
#endif
#if (SINK_GRAVACCRETION == 8)
    dst.hubber_mdot_bondi_limiter   += src.hubber_mdot_bondi_limiter;
    dst.hubber_mdot_vr_estimator    += src.hubber_mdot_vr_estimator;
    dst.hubber_mdot_disk_estimator  += src.hubber_mdot_disk_estimator;
#endif
#if defined(SINK_GRAVCAPTURE_GAS)
    dst.mass_to_swallow_edd += src.mass_to_swallow_edd;
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
    for(int kv = 0; kv < 3; kv++) dst.angmom_prepass_sum_for_passback[kv] += src.angmom_prepass_sum_for_passback[kv];
#endif
#if defined(SINK_RETURN_BFLUX)
    dst.kernel_norm_topass_in_swallowloop += src.kernel_norm_topass_in_swallowloop;
#endif
}

/* -----------------------------------------------------------------------
 * Evaluator — collective. Every rank enters.
 * --------------------------------------------------------------------- */
void sink_env1_mode_b_evaluate(struct particle_data *P,
                               struct gas_cell_data *CellP,
                               int *nl_active,
                               int  num_active,
                               const double *nl_radii,
                               struct sink_env_gpu_out *nl_outs)
{
    double t_entry = my_second();
    g_mode_b_eval_call_id++;     /* used by MODEB_XVAL_NB first-call gating */

    /* DIAG: fence to make pending GPU writes (e.g. earlier gravity/hydro/kick
     * kernels) visible to host before we read P[].Vel / Pos. P[] lives in
     * SharedSpace (UVM), but UVM coherence is fence-bracketed on Cuda — the
     * env-off path implicitly fences inside the GPU NGL pipeline; Mode B does
     * no GPU work and so reads whatever cached host view exists.
     * Gated by GIZMO_MODE_B_FENCE_AT_ENTRY=1 so we can A/B test cleanly. */
    {
        static const char *fe = getenv("GIZMO_MODE_B_FENCE_AT_ENTRY");
        static const int fence_on = (fe && fe[0] == '1') ? 1 : 0;
        if(fence_on) gpu_host_fence();
    }

    sink_env1_scalars_t sc = snapshot_scalars();

    /* Zero local outs (caller may have left them at whatever; be safe). */
    for(int a = 0; a < num_active; a++) memset(&nl_outs[a], 0, sizeof(nl_outs[0]));

    /* Reset transport diagnostics for this call. */
    mode_b_p2p_diag_reset();

    /* 1. Pack my queries. (One per local active sink. Even if num_active==0
     * this is empty; we still proceed because peers may have queries.) */
    std::vector<sink_env1_query_t> my_queries;
    my_queries.reserve(num_active);
    for(int a = 0; a < num_active; a++) {
        my_queries.push_back(pack_query(P, nl_active[a], nl_radii[a], a, sc));
    }
    double t_packed = my_second();

    /* 2. Self-pairs first (no MPI). My queries against MY local pool. */
    int n_cand_self_total = 0;
    for(size_t qi = 0; qi < my_queries.size(); qi++) {
        struct sink_env_gpu_out out_self;
        memset(&out_self, 0, sizeof(out_self));
        query_eval_diag_t d = sink_env1_evaluate_one_query_local(P, CellP,
                                                                 my_queries[qi],
                                                                 sc, out_self);
        n_cand_self_total += d.n_candidates;
        merge_into(nl_outs[my_queries[qi].origin_local_idx], out_self);
    }
    double t_self_walked = my_second();

    /* 3. P2P transport — only if NTask > 1. Otherwise the spike is just
     * the self-pair path above (still meaningful: skips ghost prep + GPU NGL). */
    int n_cand_remote_total = 0;
    int n_query_recv_total  = 0;
    int n_query_sent_total  = (int)my_queries.size() * (NTask > 1 ? (NTask - 1) : 0);
    double t_query_done = t_self_walked, t_walked = t_self_walked, t_replied = t_self_walked;
    if(NTask > 1) {
        /* 3a. Build per-peer query arrays. Same query list goes to every peer
         * in the spike (broadcast pattern, P2P transport — the locked
         * "all peers initially" rule). */
        std::vector<std::vector<sink_env1_query_t>> queries_per_peer(NTask);
        for(int p = 0; p < NTask; p++) {
            if(p == ThisTask) continue;
            queries_per_peer[p] = my_queries;   /* copy */
        }
        auto state = mode_b_exchange_queries(queries_per_peer);
        t_query_done = my_second();

        /* 3b. For each query I received, evaluate against my local pool,
         * package replies. */
        std::vector<std::vector<struct sink_env_gpu_out>> replies_per_peer(NTask);
        for(int p = 0; p < NTask; p++) {
            if(p == ThisTask) continue;
            replies_per_peer[p].resize(state.recv_counts[p]);
            n_query_recv_total += state.recv_counts[p];
            for(int qi = 0; qi < state.recv_counts[p]; qi++) {
                memset(&replies_per_peer[p][qi], 0, sizeof(struct sink_env_gpu_out));
                query_eval_diag_t d = sink_env1_evaluate_one_query_local(
                    P, CellP, state.recv_queries[p][qi], sc, replies_per_peer[p][qi]);
                n_cand_remote_total += d.n_candidates;
            }
        }
        t_walked = my_second();

        /* 3c. Reply exchange. */
        auto recv_replies = mode_b_exchange_replies<sink_env1_query_t,
                                                    struct sink_env_gpu_out>(
            replies_per_peer, state);
        t_replied = my_second();

        /* 3d. Merge per-peer replies into nl_outs (indexed by my_queries
         * order, which is the active's nl_active index). */
        for(int p = 0; p < NTask; p++) {
            if(p == ThisTask) continue;
            for(size_t qi = 0; qi < recv_replies[p].size(); qi++) {
                int local_idx = my_queries[qi].origin_local_idx;
                merge_into(nl_outs[local_idx], recv_replies[p][qi]);
            }
        }
    }
    double t_reduced = my_second();

    /* 4. Diagnostics. */
    if(phase0_diag_enabled()) {
        mode_b_p2p_diag_t pd = mode_b_p2p_diag_snapshot();
        printf("PHASE0_MODEB_NGL rank=%d caller=sink_env1 "
               "n_active_local=%d n_query_sent=%d n_query_recv=%d "
               "n_cand_self=%d n_cand_remote=%d "
               "bytes_q_sent=%zu bytes_q_recv=%zu bytes_r_sent=%zu bytes_r_recv=%zu "
               "peers_sent_to=%d peers_recv_from=%d "
               "dt_pack=%.6f dt_self_walk=%.6f dt_query_xchg=%.6f "
               "dt_remote_walk=%.6f dt_reply_xchg=%.6f dt_total=%.6f "
               "ghost_imports=0 sidx_dec=0 gpu_ngl=0\n",
               ThisTask, num_active, n_query_sent_total, n_query_recv_total,
               n_cand_self_total, n_cand_remote_total,
               pd.bytes_query_sent, pd.bytes_query_recv,
               pd.bytes_reply_sent, pd.bytes_reply_recv,
               pd.peers_sent_to, pd.peers_recv_from,
               timediff(t_entry, t_packed),
               timediff(t_packed, t_self_walked),
               timediff(t_self_walked, t_query_done),
               timediff(t_query_done, t_walked),
               timediff(t_walked, t_replied),
               timediff(t_entry, t_reduced));
        fflush(stdout);
    }

    /* 5. Cross-validation dump (offline diff against env-off run). */
    if(xval_dump_enabled()) {
        for(int a = 0; a < num_active; a++) {
            printf("MODEB_XVAL rank=%d caller=sink_env1 active_local=%d "
                   "Mgas=%g Mstar=%g Malt=%g IE=%g "
                   "Jgas=%g,%g,%g Jstar=%g,%g,%g Jalt=%g,%g,%g\n",
                   ThisTask, a,
                   (double)nl_outs[a].Mgas_in_Kernel,
                   (double)nl_outs[a].Mstar_in_Kernel,
                   (double)nl_outs[a].Malt_in_Kernel,
                   (double)nl_outs[a].Sink_SurroudingGasInternalEnergy,
                   (double)nl_outs[a].Jgas_in_Kernel[0], (double)nl_outs[a].Jgas_in_Kernel[1], (double)nl_outs[a].Jgas_in_Kernel[2],
                   (double)nl_outs[a].Jstar_in_Kernel[0], (double)nl_outs[a].Jstar_in_Kernel[1], (double)nl_outs[a].Jstar_in_Kernel[2],
                   (double)nl_outs[a].Jalt_in_Kernel[0], (double)nl_outs[a].Jalt_in_Kernel[1], (double)nl_outs[a].Jalt_in_Kernel[2]);
        }
        fflush(stdout);
    }
}

#endif /* SINK_PARTICLES */
