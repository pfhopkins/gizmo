/* sink_environment_gpu.cc — GPU-accelerated sink environment, Stage E2 only.
 *
 * Stage E1 (sink_environment, "first pass") flows through
 * mesh/neighbor_loop_runner.cc::run_neighbor_loop<SinkEnv1Spec>, which uses
 * the inline pair body in sinks/sink_env1_loop.h with the Mode A (GPU
 * neighbor list), Mode B local, and Mode B remote (peer-to-peer) paths
 * selected by threshold dispatch.
 *
 * The only GPU evaluator that lives in this file is the Stage E2 aggregator
 * sink_environment_second_evaluate_gpu (Bulge-Disk; SINK_GRAVACCRETION==0).
 * It is pure i-side (no j-writes), takes per-active Jgas/Jstar inputs from
 * Stage E1 host scatter, and writes per-active sink_env_second_gpu_out.
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

#if defined(SINK_GRAVACCRETION) && (SINK_GRAVACCRETION == 0)
void sink_environment_second_evaluate_gpu(struct particle_data *, struct gas_cell_data *,
                                           int, int *, int, const double *,
                                           const MyFloat (*)[3], const MyFloat (*)[3],
                                           int, struct sink_env_second_gpu_out *) {}
#endif
GPU_ALL_SYNC_FUNC_STUB(sinkenv)

#endif /* SINK_PARTICLES */
