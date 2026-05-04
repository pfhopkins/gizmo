/* radfb_local_gpu.cc — GPU neighbor-loop port for radiation_pressure_winds_consolidated.
 *
 * Active Type-4 (stellar) particles scatter momentum kicks into surrounding gas
 * via UV single-scattering and IR multiple-scattering opacity coupling.
 * Two-pass kernel per source: pass 1 computes weight sum (wt_sum = Σ h_j²),
 * pass 2 applies momentum kicks.  Stochastic gate evaluated on CPU; sources
 * that don't fire are excluded from i_active_host before GPU dispatch.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"
#include "../system/gpu_particles_arena.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"

#include "../declarations/gpu_numeric_macros.h"
#include "../declarations/gpu_error_check.h"
#include "../declarations/gpu_dispatch_templates.h"
#include "../declarations/macros.h"
#include "../mesh/gpu_neighbor_list.h"
#include "../mesh/ghost_writeback.h"
#include "../mesh/ghost_symlist_lifecycle.h"

#if defined(GALSF_FB_FIRE_RT_LOCALRP)

#include "radfb_local_functions.h"

/* Age threshold above which a star contributes nothing — must match CPU function. */
static double radfb_rp_age_threshold(void)
{
#if defined(SINGLE_STAR_SINK_DYNAMICS) || (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
    return 1.0e10;
#else
    return 0.15; /* Gyr */
#endif
}


/* Fill RadFBRPLocalIn for source particle i.
 * Returns 1 if the source passes the stochastic gate and should fire, 0 to skip. */
static int radfb_rp_local_fill(int i,
                                struct particle_data *P_host,
                                struct gas_cell_data *CellP_host,
                                struct RadFBRPLocalIn *loc,
                                double *src_radius_out)
{
    /* type / mass / age guards */
    if(P_host[i].Type != 4) {
        /* non-comoving sims also allow Type 2/3 */
        if(!(!All.ComovingIntegrationOn &&
             (P_host[i].Type == 2 || P_host[i].Type == 3))) return 0;
    }
    if(P_host[i].Mass <= 0 || !isfinite(P_host[i].Mass)) return 0;
    if(P_host[i].DensityAroundParticle <= 0) return 0;

    double star_age = evaluate_stellar_age_Gyr(i);
    if(star_age >= radfb_rp_age_threshold()) return 0;

    double dt = get_particle_feedback_timestep_in_physical(i);
    if(dt <= 0 || !isfinite(dt)) return 0;

    /* luminosity */
    double lm_ssp   = evaluate_light_to_mass_ratio(star_age, i);
    double lum_cgs  = lm_ssp * SOLAR_LUM_CGS * (P_host[i].Mass * UNIT_MASS_IN_SOLAR);
    double f_lum_ion = particle_ionizing_luminosity_in_cgs(i) / lum_cgs;
    f_lum_ion = DMAX(0., DMIN(1., f_lum_ion));

    double dE_over_c = All.RP_Local_Momentum_Renormalization * lum_cgs
                     * (dt * UNIT_TIME_IN_CGS) / C_LIGHT_CGS;
    dE_over_c /= (UNIT_MASS_IN_CGS * UNIT_VEL_IN_CGS);

    /* search radius */
    double rho_phys = P_host[i].DensityAroundParticle * All.cf_a3inv;
    double h_phys   = P_host[i].KernelRadius * All.cf_atime;
    double RtauMax  = P_host[i].KernelRadius
                    * (5.0 + 2.0 * rt_kappa(i, RT_FREQ_BIN_FIRE_UV, P_host, CellP_host)
                       * P_host[i].KernelRadius * P_host[i].DensityAroundParticle * All.cf_a2inv);
    RtauMax = DMAX(1.0/(UNIT_LENGTH_IN_KPC*All.cf_atime),
                   DMIN(10.0/(UNIT_LENGTH_IN_KPC*All.cf_atime), RtauMax));

    /* stochastic gate */
    double v_wind_threshold = 15.0 / UNIT_VEL_IN_KMS;
#if defined(SINGLE_STAR_SINK_DYNAMICS) && !defined(GALSF_FB_FIRE_STELLAREVOLUTION)
    v_wind_threshold = 0.2 / UNIT_VEL_IN_KMS;
#endif
    double delta_v = v_wind_threshold;
    double tau_IR  = rt_kappa(i, RT_FREQ_BIN_FIRE_IR, P_host, CellP_host)
                   * rho_phys * h_phys;
    double dv_guess = (dE_over_c / P_host[i].Mass) * (1.0 + tau_IR);
    double prob = dv_guess / delta_v * 2000.0;

#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
    double v_grav = DMIN(1.82*(65.748/UNIT_VEL_IN_KMS)*pow(1.+rho_phys*UNIT_DENSITY_IN_NHCGS,-0.25),
                         sqrt(All.G*(P_host[i].Mass + 4.189*rho_phys*h_phys*h_phys*h_phys)/h_phys));
    delta_v = 2.0 * DMAX(v_grav, v_wind_threshold);
    prob    = dv_guess / delta_v;

    double jet_mom = 0;
    if(P_host[i].Type == 4 && P_host[i].NewStar_Momentum_For_JetFeedback > 0) {
        prob    = 1.0;
        jet_mom = P_host[i].NewStar_Momentum_For_JetFeedback;
    }
    if(prob < 1.0 && prob > 0) dE_over_c /= prob;
#endif

    double p_random = get_random_number(P_host[i].ID + ThisTask + i + 2);
    if(p_random > prob) return 0; /* doesn't fire this step */

    /* fill struct */
    loc->Pos          = P_host[i].Pos;
    loc->KernelRadius = (MyFloat)RtauMax;   /* use RtauMax so we capture all potential neighbors */
    loc->dE_over_c    = (MyFloat)dE_over_c;
    loc->f_lum_ion    = (MyFloat)f_lum_ion;
    loc->ID           = P_host[i].ID;
#if (GALSF_FB_FIRE_STELLAREVOLUTION <= 2)
    loc->delta_v_imparted_rp = (MyFloat)delta_v;
#endif
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
    loc->jet_momentum_tocouple = (MyFloat)jet_mom;
#endif

    *src_radius_out = RtauMax;
    return 1;
}


/* ================================================================
   GPU local radiation-pressure-winds evaluator
   ----------------------------------------------------------------
   Kernel writes (host-visible, by index):
     i-side: out_arr[aa].jet_momentum_used (atomic_add inside pair_kick)
     j-side (gas neighbors, indexed by j = neighbors[nn]; only Type==0):
       P_gpu[j].Vel[k]      — Kokkos::atomic_add  (k = 0..2)
       CellP_gpu[j].VelPred[k] — Kokkos::atomic_add  (k = 0..2)
       P_gpu[j].dp[k]       — Kokkos::atomic_add  (k = 0..2)

   Sparse-scatter target: walk neighbors[] and scatter the 3 fields above
   to host for touched j. The full memcpy(P_host, P_gpu, num_all*...) and
   memcpy(CellP_host, CellP_gpu, num_all*...) at lines ~282-283 cover ONLY
   these three j-side updates plus the per-source NewStar_Momentum loop
   that runs explicitly on host (active sources only). Replace with sparse
   scatter.
   ================================================================ */
void radiation_pressure_winds_gpu(struct particle_data *P_host,
                                   struct gas_cell_data *CellP_host,
                                   int num_total,
                                   const int *i_active_host, int num_active)
{
    GIZMO_GPU_ENSURE_ALL_FRESH(radfbrp);

    /* Build active-source list: evaluate gate on CPU, keep only sources that fire. */
    std::vector<int>    active_src;
    std::vector<struct RadFBRPLocalIn> src_local;
    std::vector<double> src_radii;

    if(All.RP_Local_Momentum_Renormalization > 0)
    {
        active_src.reserve(num_active);
        src_local.reserve(num_active);
        src_radii.reserve(num_active);

        for(int a = 0; a < num_active; a++) {
            int i = i_active_host[a];
            struct RadFBRPLocalIn loc;
            double radius;
            if(radfb_rp_local_fill(i, P_host, CellP_host, &loc, &radius)) {
                active_src.push_back(i);
                src_local.push_back(loc);
                src_radii.push_back(radius);
            }
        }
    }

    int num_src = (int)active_src.size();

    /* Multi-rank correctness: ghost_prep is collective so all ranks must agree.
     * MPI_Allreduce num_src; if no rank has work, every rank skips ghost_prep. */
    int global_num_src = num_src;
    MPI_Allreduce(&num_src, &global_num_src, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(global_num_src == 0) { return; }

    /* Ghost prep — must happen before neighbor list build.
     * Collective decision: active-aware exchange leaves ghost counts asymmetric on
     * tiny-N steps (density redo may give some ranks 0 ghosts).  Allreduce so all
     * ranks agree: if any rank needs an import, all re-import. */
    int imported_ghosts = 0;
    {
        int need_import_local = (ghost_get_num_ghosts() <= 0) ? 1 : 0;
        int need_import = 0;
        MPI_Allreduce(&need_import_local, &need_import, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if(need_import) {
            if(ghost_get_num_ghosts() > 0) ghost_exchange_cleanup();
            gizmo_density_prep_ghosts(gizmo_ghost_safety_factor());
            imported_ghosts = 1;
        }
    }

    int num_all = ghost_get_num_local() + ghost_get_num_ghosts();
    if(num_all <= 0) num_all = num_total;

    /* Wrapper fast-path: with no source particles, kernel does no host writes,
     * skip arena_acquire / d_local / d_out / gpu_ngb_list_build / kernel /
     * scatter / apply loops.  ghost_writeback_*_radfbrp self-guard on
     * NTask<=1 || num_ghosts<=0 (no-op single-rank); MPI participation kept
     * for multi-rank correctness. */
    if(num_src == 0) {
        ghost_write_detector_begin("radfb_local_rp");
        ghost_writeback_zero_radfbrp();
        ghost_writeback_radfbrp();
        ghost_write_detector_end();
        if(imported_ghosts) ghost_exchange_cleanup();
        return;
    }

    /* Step 13 Phase 1 arena. */
    gpu_particles_arena_set_site("radiation_pressure_winds_gpu");
    gpu_particles_arena_acquire(num_all, P_host, CellP_host);
    struct particle_data *P_gpu = gpu_particles_arena_P();
    struct gas_cell_data *CellP_gpu = gpu_particles_arena_CellP();

    /* Snapshot ghost Vel/VelPred/dp for writeback */
    ghost_write_detector_begin("radfb_local_rp");
    ghost_writeback_zero_radfbrp();

    int alloc_n = (num_src > 0) ? num_src : 1;

    /* Copy per-source inputs to SharedSpace */
    struct RadFBRPLocalIn *d_local = (struct RadFBRPLocalIn *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(alloc_n * sizeof(struct RadFBRPLocalIn));
    if(num_src > 0) memcpy(d_local, src_local.data(), num_src * sizeof(struct RadFBRPLocalIn));

    /* Per-source output */
    struct RadFBRPOut *d_out = (struct RadFBRPOut *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(alloc_n * sizeof(struct RadFBRPOut));
    memset(d_out, 0, alloc_n * sizeof(struct RadFBRPOut));

    /* Build cross-type neighbor list: Type-4 sources → gas (j_type_bitmask=1) */
    gpu_neighbor_list_t gnl;
    gpu_ngb_list_build(P_gpu, num_all,
                       active_src.data(), num_src,
                       NGB_SEARCH_ONEWAY, 1 /* gas only */,
                       &gnl, gpu_step_sidx_ptr(), 1.0, src_radii.data(), NULL, "radfb_g");

    PRINT_STATUS("  GPU radiation_pressure: %d sources, %d pairs", num_src, gnl.total_pairs);

    /* Launch kernel: two passes per source (wt_sum, then kicks) */
    {
        int  *offsets   = gnl.offsets;
        int  *neighbors = gnl.neighbors;
        struct RadFBRPLocalIn *local_arr = d_local;
        struct RadFBRPOut     *out_arr   = d_out;
        struct particle_data  *kp = P_gpu;
        struct gas_cell_data  *kc = CellP_gpu;

        Kokkos::parallel_for("radfb_rp", num_src, KOKKOS_LAMBDA(int aa) {
            const struct RadFBRPLocalIn& loc = local_arr[aa];
            double h2 = (double)loc.KernelRadius * (double)loc.KernelRadius;
            int start = offsets[aa], end = offsets[aa+1];

            /* Pass 1: compute wt_sum = Σ h_j² over valid neighbors */
            double wt_sum = 0.0;
            for(int nn = start; nn < end; nn++) {
                int j = neighbors[nn];
                if(kp[j].Type != 0 || kp[j].Mass <= 0) continue;
#ifdef SINK_WIND_SPAWN
                if(kp[j].ID == All.SpawnedWindCellID) continue;
#endif
                Vec3<double> dp = loc.Pos - kp[j].Pos;
                nearest_xyz(dp);
                if(dp.norm_sq() >= h2 || dp.norm_sq() <= 0) continue;
                double h_j = kp[j].Get_Particle_Size();
                wt_sum += h_j * h_j;
            }
            if(wt_sum <= 0) return;

            struct RadFBRPOut myout; myout.jet_momentum_used = 0;

            /* Pass 2: apply kicks */
            for(int nn = start; nn < end; nn++) {
                int j = neighbors[nn];
                if(kp[j].Type != 0 || kp[j].Mass <= 0) continue;
#ifdef SINK_WIND_SPAWN
                if(kp[j].ID == All.SpawnedWindCellID) continue;
#endif
                Vec3<double> dp = loc.Pos - kp[j].Pos;
                nearest_xyz(dp);
                double r2 = dp.norm_sq();
                if(r2 >= h2 || r2 <= 0) continue;
                radfb_rp_pair_kick(loc, j, kp, kc, r2, dp, wt_sum, nn - start, myout);
            }
            out_arr[aa].jet_momentum_used = myout.jet_momentum_used;
        });
    }
    Kokkos::fence();
    gizmo_gpu_check_last_error("radfb_rp", num_src);

    /* Row 3 of arena-scope sweep: sparse scatter over CSR neighbors[] —
     * replaces former full memcpy(P_host, P_gpu, num_all*...) and
     * memcpy(CellP_host, CellP_gpu, num_all*...). Per the kernel-writes
     * audit (commit a06e30ca), radfb_rp_pair_kick writes only
     *   P[j].Vel[k]      (atomic_add)
     *   CellP[j].VelPred[k] (atomic_add)
     *   P[j].dp[k]       (atomic_add)
     * and only for gas neighbors (kernel's Type==0 guard returns early
     * for non-gas j's, leaving them unchanged in arena vs host). Walk
     * the CSR; idempotent assignment makes duplicates correct. The
     * per-source NewStar_Momentum loop below mutates P_host[i] directly
     * for active sources only — those i's are not in this scatter set
     * (they're sources, not neighbors), so no conflict.
     *
     * gnl.neighbors lives in DEVICE_SPACE (CudaSpace) — not host-readable
     * on GH200. Deep-copy once, then iterate the host buffer. */
    if(gnl.total_pairs > 0) {
        std::vector<int> gnl_neighbors_host(gnl.total_pairs);
        gpu_ngb_copy_neighbors_to_host(&gnl, gnl_neighbors_host.data());
        for(int idx = 0; idx < gnl.total_pairs; idx++) {
            int j = gnl_neighbors_host[idx];
            P_host[j].Vel = P_gpu[j].Vel;
            P_host[j].dp  = P_gpu[j].dp;
            CellP_host[j].VelPred = CellP_gpu[j].VelPred;
        }
    }

    /* Ghost writeback: propagate Vel/VelPred/dp deltas to home ranks */
    ghost_writeback_radfbrp();
    ghost_write_detector_end();

    /* Apply jet momentum consumed back to source particles */
    std::vector<struct RadFBRPOut> h_out(num_src);
    if(num_src > 0) memcpy(h_out.data(), d_out, num_src * sizeof(struct RadFBRPOut));

#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
    for(int a = 0; a < num_src; a++) {
        int i = active_src[a];
        if(P_host[i].NewStar_Momentum_For_JetFeedback > 0) {
            P_host[i].NewStar_Momentum_For_JetFeedback -=
                (MyDouble)h_out[a].jet_momentum_used;
        }
    }
#endif

    /* MPI diagnostics (rank 0 only, mirrors CPU function) */
    double total_mom_wind = 0;
    for(int a = 0; a < num_src; a++) {
        int i = active_src[a];
        total_mom_wind += (double)src_local[a].dE_over_c * P_host[i].Mass;
    }
    double totMPI_n_send = num_src, totMPI_mom_send = total_mom_wind;
    double totMPI_n = 0, totMPI_mom = 0;
    MPI_Reduce(&totMPI_n_send,   &totMPI_n,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&totMPI_mom_send, &totMPI_mom, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask == 0 && totMPI_n > 0) {
        PRINT_STATUS(" ..GPU RadFB: Nsources=%g dP_coupled=%g", totMPI_n, totMPI_mom);
    }

    /* Full P+CellP scatter syncs arena→host; per-source NewStar_Momentum loop
     * above mutates P_host directly, so invalidate to cover that. */
    gpu_ngb_list_free(&gnl, gpu_step_sidx_ptr());
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_out);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_local);
    gpu_particles_arena_invalidate();

    if(imported_ghosts) { ghost_exchange_cleanup(); }
}

GPU_ALL_SYNC_FUNC(radfbrp)

#else /* stub */

void radiation_pressure_winds_gpu(struct particle_data *p,
                                   struct gas_cell_data *cp,
                                   int num_total,
                                   const int *i_active_host, int num_active)
{
    (void)p; (void)cp; (void)num_total; (void)i_active_host; (void)num_active;
}
GPU_ALL_SYNC_FUNC_STUB(radfbrp)

#endif /* GALSF_FB_FIRE_RT_LOCALRP */
