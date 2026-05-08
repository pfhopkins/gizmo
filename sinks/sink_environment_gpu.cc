/* sink_environment_gpu.cc — GPU-accelerated sink environment loop (B4).
 *
 * Ports sink_environment_evaluate (sinks/sink_environment.cc:163-293) to a
 * Kokkos parallel_for kernel.  i-particles are active sinks (Type=5);
 * j-particles are all types (SINK_NEIGHBOR_BITFLAG).  The neighbor list is
 * built with gpu_build_cross_type_neighbor_list (Infra-2).
 *
 * i-side accumulators are written to per-active-sink sink_env_gpu_out structs
 * and scattered into SinkTempInfo on the host after the kernel.
 *
 * Under SINGLE_STAR_SINK_DYNAMICS + SINK_GRAVCAPTURE_GAS, the kernel does
 * Kokkos::atomic_fetch_min on P[j].SwallowTime for bound particles.  The host
 * driver wraps this call with ghost_writeback_{zero_,}swallowtime() so
 * ghost-side minima are reverse-communicated to their home ranks.
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
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../system/gpu_particles_arena.h"
#include "../mesh/kernel.h"
#include "../mesh/gpu_neighbor_list.h"
#include "../mesh/neighbor_list.h"

#include "../declarations/gpu_numeric_macros.h"
#include "../declarations/gpu_error_check.h"
#include "../declarations/gpu_dispatch_templates.h"

#include "sinks_gpu_decls.h"

#if defined(SINK_PARTICLES)

#include "sink_functions.h"
#include "sink_environment_mode_b.h"  /* sink_env1_query_t */
#include "sink_env1_pair_kernel.h"    /* SSOT pair body — shared with Mode B */

/* ================================================================
   GPU sink-environment evaluator (Stage E1 — kernel-sum pass)
   ----------------------------------------------------------------
   Kernel writes (host-visible, by index):
     i-side: kout[aa] (per-source SinkEnvOut: Mgas_in_Kernel, Jgas_in_Kernel,
                       SurroundingGasVel, mass_to_swallow_edd, ...)
     j-side (neighbors, indexed by j = neighbors[nn]):
       #if defined(SINGLE_STAR_SINK_DYNAMICS)
       P_gpu[j].SwallowTime                   — Kokkos::atomic_fetch_min
       #endif
     Outside SINGLE_STAR_SINK_DYNAMICS: NO J-SIDE WRITES at all.

   Sparse-scatter target:
     - WITH SINGLE_STAR_SINK_DYNAMICS: walk neighbors[] CSR, scatter
       P_gpu[j].SwallowTime to host for touched j.
     - WITHOUT: delete the full memcpy(P_host, P_gpu, num_total*...)
       at line ~261 entirely (Case 1).
   For fire_m11i (SINGLE_STAR not defined): pure delete.
   The Stage-E2 (sink_environment_second) kernel is pure i-side, no
   j-writes; its scatter is also Case-1 delete-able.
   ================================================================ */
void sink_environment_evaluate_gpu(struct particle_data *P_host,
                                   struct gas_cell_data *CellP_host,
                                   int num_total,
                                   int *i_active_host, int num_active,
                                   const double *i_radii_host,
                                   int j_type_bitmask,
                                   struct sink_env_gpu_out *out_host)
{
    GIZMO_GPU_ENSURE_ALL_FRESH(sinkenv);

    /* Wrapper fast-path: no active sources → nothing to do.  Skip arena_acquire
     * (1.4s slow-path memcpy when invalidated), the 17 GB host scatter at the
     * end, allocations, kernel.  No MPI collectives in this function so we can
     * fully early-out. */
    if(num_active == 0) { return; }

    /* Step 13 Phase 1 arena: pass CellP_host=NULL when no gas; arena handles. */
    gpu_particles_arena_set_site("sink_environment_evaluate_gpu");
    gpu_particles_arena_acquire(num_total, P_host, (All.TotN_gas > 0) ? CellP_host : NULL);
    struct particle_data *P_gpu = gpu_particles_arena_P();
    struct gas_cell_data *CellP_gpu = (All.TotN_gas > 0) ? gpu_particles_arena_CellP() : NULL;

    gpu_neighbor_list_t gnl;
    /* Use the all-types step-persistent SIDX cache (built once per step;
     * sink_env1/feed/swk all share). j_type_bitmask MUST be 0x3f for cache
     * consistency. */
    gpu_ngb_list_build(P_gpu, num_total, i_active_host, num_active,
                       NGB_SEARCH_SYMMETRIC, j_type_bitmask, &gnl,
                       gpu_step_sidx_alltypes_ptr(),
                       1.0, i_radii_host, NULL, "sink_env1");

    struct sink_env_gpu_out *d_out = (struct sink_env_gpu_out *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
            ((num_active > 0) ? num_active : 1) * sizeof(struct sink_env_gpu_out));

    double *d_radii = (double *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
            ((num_active > 0) ? num_active : 1) * sizeof(double));
    memcpy(d_radii, i_radii_host, num_active * sizeof(double));

    /* Capture cosmological and physical scalars */
    double cf_atime    = All.cf_atime;
    double cf_a2inv    = All.cf_a2inv;
    double cf_a3inv    = All.cf_a3inv;
    double G_val       = All.G;
    double sink_radius_grav = SinkParticle_GravityKernelRadius;

    PRINT_STATUS("  GPU sink_environment: %d active, j_bitmask=%d, %d pairs",
                 num_active, j_type_bitmask, gnl.total_pairs);

    {
        int *offsets   = gnl.offsets;
        int *neighbors = gnl.neighbors;
        int *active    = gnl.d_active;
        struct particle_data   *kp  = P_gpu;
        struct gas_cell_data   *kc  = CellP_gpu;
        double *radii  = d_radii;
        struct sink_env_gpu_out *kout = d_out;

        /* Per-call scalars passed to the SSOT pair kernel (shared with Mode B). */
        sink_env1_scalars_t sc;
        sc.cf_atime          = cf_atime;
        sc.cf_a2inv          = cf_a2inv;
        sc.cf_a3inv          = cf_a3inv;
        sc.G_val             = G_val;
        sc.sink_radius_grav  = sink_radius_grav;

        gizmo_gpu_kernel_launch("sink_env_kernel", num_active, KOKKOS_LAMBDA(int aa) {
            int ii = active[aa];
            double h_i = radii[aa];
            memset(&kout[aa], 0, sizeof(struct sink_env_gpu_out));
            if(!(kp[ii].Mass > 0) || !(h_i > 0)) { return; }

            /* Pack the active sink's pair-body inputs into the SSOT query
             * struct (sink_env1_query_t in sink_environment_mode_b.h),
             * mirroring pack_query() on the Mode B host path. The shared
             * pair kernel reads only these fields plus j-side data. */
            sink_env1_query_t q;
            q.pos      = kp[ii].Pos;
            q.vel      = kp[ii].Vel;
            q.id       = kp[ii].ID;
            q.h_search = h_i;
#if (ADAPTIVE_GRAVSOFT_FORALL & 32)
            q.ags_h    = kp[ii].AGS_KernelRadius;
#else
            q.ags_h    = sink_radius_grav;
#endif
#if defined(SINK_GRAVCAPTURE_GAS) || (SINK_GRAVACCRETION == 8)
            q.mass     = kp[ii].Mass;
#endif
#if defined(SINK_GRAVCAPTURE_FIXEDSINKRADIUS)
            q.sink_radius = kp[ii].SinkRadius;
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
            for(int kv = 0; kv < 3; kv++) q.sink_angmom[kv] = kp[ii].Sink_Specific_AngMom[kv];
#endif
            q.origin_local_idx = aa;
            q.origin_rank      = -1;  /* device — rank N/A in lambda */

            int start = offsets[aa], end = offsets[aa + 1];
            for(int nn = start; nn < end; nn++) {
                int j = neighbors[nn];
                /* Self-skip / mass / type-5 — the shared pair kernel does
                 * the same check, but skipping early avoids the j-side
                 * write block below (SSD only). Behaviorally identical. */
                if(kp[j].Mass <= 0 || kp[j].Type == 5 || kp[j].ID == q.id) { continue; }

                /* SSOT call: same pair body as Mode B host walker.
                 * (Defined in sinks/sink_env1_pair_kernel.h.) */
                const struct gas_cell_data* kc_j_ptr = (kp[j].Type == 0 && kc) ? &kc[j] : nullptr;
                sink_env1_pair_kernel(q, kp[j], kc_j_ptr, sc, kout[aa]);
#if defined(SINK_GRAVCAPTURE_GAS) && defined(SINGLE_STAR_SINK_DYNAMICS)
#error "Phase A consolidation: GPU sink_env1 SSOT pair body does not yet handle the SINGLE_STAR_SINK_DYNAMICS j-side atomic write to kp[j].SwallowTime. Phase B (NeighborLoopSpec runner with SymmetricPairScatter writeback) will restore it. SSD builds: revert this commit pending Phase B, or wire the atomic write through the writeback channel."
#endif
            } /* end neighbor loop */
        });
    }


    /* Row 4c of arena-scope sweep: per kernel-writes audit (a06e30ca), the
     * sink_env kernel writes ZERO j-side fields outside SINGLE_STAR_SINK_DYNAMICS
     * (and only P[j].SwallowTime when that's defined). Everything else is
     * per-active-source kout[aa] or pure read.
     *
     * Outside SINGLE_STAR_SINK_DYNAMICS (e.g. fire_m11i): no j-side scatter
     * needed at all. The former full memcpy(P_host, P_gpu, num_total*...) was
     * pure cargo-cult and is DELETED.
     *
     * Inside SINGLE_STAR_SINK_DYNAMICS: scatter SwallowTime sparsely over the
     * CSR-touched j set (gnl.neighbors[], deep-copied to host). */
    memcpy(out_host, d_out, num_active * sizeof(struct sink_env_gpu_out));

    /* MODEB_XVAL_NB diagnostic: first-call-only dump of the GPU NGL neighbor
     * set per active. Lets us diff Mode B's host walker output against the
     * GPU SIDX path on identical state. Env-gated; instrumentation only. */
    {
        static const char *xv_nb_env = getenv("GIZMO_MODE_B_XVAL_NB_DUMP");
        static const int xv_nb_on = (xv_nb_env && xv_nb_env[0] == '1') ? 1 : 0;
        static int xv_nb_fired = 0;
        if(xv_nb_on && !xv_nb_fired && gnl.total_pairs > 0) {
            std::vector<int> nbrs_host(gnl.total_pairs);
            gpu_ngb_copy_neighbors_to_host(&gnl, nbrs_host.data());
            int *offsets = gnl.offsets;
            for(int aa = 0; aa < num_active; aa++) {
                int ii = i_active_host[aa];
                double h_i = i_radii_host[aa];
                int start = offsets[aa], end = offsets[aa + 1];
                int nj = end - start;
                printf("MODEB_XVAL_NB rank=%d call=1 active=%d path=gpu_ngl "
                       "h_search=%.17g i_pos=%.17g,%.17g,%.17g "
                       "i_vel=%.17g,%.17g,%.17g i_id=%llu n_j=%d\n",
                       ThisTask, aa, h_i,
                       (double)P_host[ii].Pos[0], (double)P_host[ii].Pos[1], (double)P_host[ii].Pos[2],
                       (double)P_host[ii].Vel[0], (double)P_host[ii].Vel[1], (double)P_host[ii].Vel[2],
                       (unsigned long long)P_host[ii].ID, nj);
                for(int idx = start; idx < end; idx++) {
                    int j = nbrs_host[idx];
                    double dx = (double)P_host[j].Pos[0] - (double)P_host[ii].Pos[0];
                    double dy = (double)P_host[j].Pos[1] - (double)P_host[ii].Pos[1];
                    double dz = (double)P_host[j].Pos[2] - (double)P_host[ii].Pos[2];
                    NEAREST_XYZ(dx, dy, dz, 1);
                    double r2 = dx*dx + dy*dy + dz*dz;
                    printf("MODEB_XVAL_NB_J rank=%d call=1 active=%d path=gpu_ngl "
                           "j=%d Type=%d Mass=%.17g KernelRadius=%.17g r2=%.17g "
                           "Pos=%.17g,%.17g,%.17g Vel=%.17g,%.17g,%.17g ID=%llu\n",
                           ThisTask, aa, j, (int)P_host[j].Type,
                           (double)P_host[j].Mass, (double)P_host[j].KernelRadius, r2,
                           (double)P_host[j].Pos[0], (double)P_host[j].Pos[1], (double)P_host[j].Pos[2],
                           (double)P_host[j].Vel[0], (double)P_host[j].Vel[1], (double)P_host[j].Vel[2],
                           (unsigned long long)P_host[j].ID);
                }
            }
            fflush(stdout);
            xv_nb_fired = 1;
        }
    }

#if defined(SINGLE_STAR_SINK_DYNAMICS)
    if(gnl.total_pairs > 0) {
        std::vector<int> gnl_neighbors_host(gnl.total_pairs);
        gpu_ngb_copy_neighbors_to_host(&gnl, gnl_neighbors_host.data());
        for(int idx = 0; idx < gnl.total_pairs; idx++) {
            int j = gnl_neighbors_host[idx];
            P_host[j].SwallowTime = P_gpu[j].SwallowTime;
        }
    }
#endif

    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_radii);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_out);
    /* Pass the shared all-types cache pointer so the free leaves the SIDX
     * storage intact for sink_feed/sink_swk reuse. With NULL the free
     * destroys d_tiles/d_bvh/d_pool/d_compact_xyzh out from under the cache. */
    gpu_ngb_list_free(&gnl, gpu_step_sidx_alltypes_ptr());

    /* Phase 8a Round 3a: arena coherence. With SINGLE_STAR_SINK_DYNAMICS off,
     * the kernel writes ZERO j-side fields (audit a06e30ca, row 4c
     * 5a01b004) — arena P/CellP are byte-identical to host throughout
     * this evaluator. The caller (sinks/sink_environment.cc:95-141) only
     * mutates SinkTempInfo (separate struct, not P/CellP). Safe to
     * mark_clean instead of invalidate.
     * With SINGLE_STAR_SINK_DYNAMICS on, the SwallowTime sparse scatter
     * above DOES mirror the j-side writes back to host, so arena is also
     * coherent. (If a debug-guard run flags this branch, sweep it then.) */
#if defined(SINGLE_STAR_SINK_DYNAMICS)
    gpu_particles_arena_mark_clean_after_scatter("sink_environment_evaluate_gpu(single_star)");
#else
    gpu_particles_arena_mark_clean_after_scatter("sink_environment_evaluate_gpu");
#endif
}

/* ========================================================================
 * Stage E2 — sink_environment_second: Bulge-Disk aggregator (pure, no j-writes)
 * ======================================================================== */
#if defined(SINK_GRAVACCRETION) && (SINK_GRAVACCRETION == 0)

void sink_environment_second_evaluate_gpu(struct particle_data *P_host,
                                           struct gas_cell_data *CellP_host,
                                           int num_total,
                                           int *i_active_host, int num_active,
                                           const double *i_radii_host,
                                           const MyFloat (*i_Jgas_host)[3],
                                           const MyFloat (*i_Jstar_host)[3],
                                           int j_type_bitmask,
                                           struct sink_env_second_gpu_out *out_host)
{
    GIZMO_GPU_ENSURE_ALL_FRESH(sinkenv);

    /* Wrapper fast-path: no active sources → fully skip arena+kernel+scatter. */
    if(num_active == 0) { (void)CellP_host; return; }

    /* Step 13 Phase 1 arena. CellP unused by this kernel; pass NULL. */
    gpu_particles_arena_set_site("sink_environment_second_evaluate_gpu");
    gpu_particles_arena_acquire(num_total, P_host, NULL);
    struct particle_data *P_gpu = gpu_particles_arena_P();
    (void)CellP_host;

    gpu_neighbor_list_t gnl;
    /* Same all-types cache as sink_env1 (see gpu_neighbor_list.h). */
    gpu_ngb_list_build(P_gpu, num_total, i_active_host, num_active,
                       NGB_SEARCH_SYMMETRIC, j_type_bitmask, &gnl,
                       gpu_step_sidx_alltypes_ptr(),
                       1.0, i_radii_host, NULL, "sink_env2");

    int alloc_n = (num_active > 0) ? num_active : 1;
    struct sink_env_second_gpu_out *d_out = (struct sink_env_second_gpu_out *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(alloc_n * sizeof(struct sink_env_second_gpu_out));

    /* Stage per-source J vectors into SharedSpace so the kernel can read them. */
    double (*d_Jgas)[3]  = (double (*)[3]) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(alloc_n * sizeof(double[3]));
    double (*d_Jstar)[3] = (double (*)[3]) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(alloc_n * sizeof(double[3]));
    for(int a = 0; a < num_active; a++) {
        for(int k = 0; k < 3; k++) {
            d_Jgas[a][k]  = (double)i_Jgas_host[a][k];
            d_Jstar[a][k] = (double)i_Jstar_host[a][k];
        }
    }

    PRINT_STATUS("  GPU sink_environment_second: %d active, %d pairs", num_active, gnl.total_pairs);

    int comoving_on = All.ComovingIntegrationOn;

    {
        int  *offsets   = gnl.offsets;
        int  *neighbors = gnl.neighbors;
        int  *active    = gnl.d_active;
        struct particle_data *kp   = P_gpu;
        double (*dJg)[3]  = d_Jgas;
        double (*dJs)[3]  = d_Jstar;
        struct sink_env_second_gpu_out *kout = d_out;

        gizmo_gpu_kernel_launch("sink_env_second_kernel", num_active, KOKKOS_LAMBDA(int aa) {
            int ii = active[aa];
            memset(&kout[aa], 0, sizeof(struct sink_env_second_gpu_out));
            if(!(kp[ii].Mass > 0)) return;

            Vec3<double> pos_i = kp[ii].Pos;
            Vec3<double> vel_i = kp[ii].Vel;
            Vec3<double> Jgas_i{dJg[aa][0], dJg[aa][1], dJg[aa][2]};
            Vec3<double> Jstar_i{dJs[aa][0], dJs[aa][1], dJs[aa][2]};

            int start = offsets[aa], end = offsets[aa + 1];
            for(int nn = start; nn < end; nn++) {
                int j = neighbors[nn];
                if(kp[j].Mass <= 0 || kp[j].KernelRadius <= 0 || kp[j].Type == 5) continue;
                Vec3<double> dP = kp[j].Pos - pos_i;
                Vec3<double> dv = kp[j].Vel - vel_i;
                nearest_xyz(dP, -1);
                NGB_SHEARBOX_BOUNDARY_VELCORR_(pos_i, kp[j].Pos, dv, -1);
                Vec3<double> J_tmp = cross(dP, dv);
                if(kp[j].Type == 0) {
                    if(dot(J_tmp, Jgas_i) < 0) { kout[aa].MgasBulge_in_Kernel += 2 * kp[j].Mass; }
                }
                if(kp[j].Type == 4 ||
                   ((kp[j].Type == 2 || kp[j].Type == 3) && !comoving_on)) {
                    if(dot(J_tmp, Jstar_i) < 0) { kout[aa].MstarBulge_in_Kernel += 2 * kp[j].Mass; }
                }
            }
        });
    }

    /* Read-only kernel on P; out_host is post-scattered by host sink_environment.cc
     * into P/CellP, so invalidate so the next acquire re-syncs from host. */
    memcpy(out_host, d_out, num_active * sizeof(struct sink_env_second_gpu_out));

    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_Jstar);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_Jgas);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_out);
    /* Preserve shared all-types SIDX for downstream sink phases. */
    gpu_ngb_list_free(&gnl, gpu_step_sidx_alltypes_ptr());

    /* Phase 8a Round 3a: kernel is read-only on P/CellP (audit comment at
     * function header confirms). Caller scatters out_host into SinkTempInfo
     * only (not P/CellP). Arena byte-identical to host — mark_clean. */
    gpu_particles_arena_mark_clean_after_scatter("sink_environment_second_evaluate_gpu");
}

#endif /* SINK_GRAVACCRETION == 0 */


GPU_ALL_SYNC_FUNC(sinkenv)

#else /* stubs when disabled */

void sink_environment_evaluate_gpu(struct particle_data *, struct gas_cell_data *,
                                   int, int *, int, const double *, int,
                                   struct sink_env_gpu_out *) {}
#if defined(SINK_GRAVACCRETION) && (SINK_GRAVACCRETION == 0)
void sink_environment_second_evaluate_gpu(struct particle_data *, struct gas_cell_data *,
                                           int, int *, int, const double *,
                                           const MyFloat (*)[3], const MyFloat (*)[3],
                                           int, struct sink_env_second_gpu_out *) {}
#endif
GPU_ALL_SYNC_FUNC_STUB(sinkenv)

#endif /* SINK_PARTICLES */
