/* dm_dispersion_gpu.cc — GPU-accelerated DM velocity dispersion around gas.
 *
 * GPU translation unit: compiled by nvcc_wrapper when OPENMP_GPU_OFFLOAD is
 * enabled. Provides disp_density_evaluate_gpu(), called once per h-iteration
 * from disp_density() in dm_dispersion_rkern.cc. Builds a cross-type CSR list
 * (gas i → DM j, search radius = per-i CellP[i].KernelRadiusDM), runs the
 * accumulation kernel via Kokkos::parallel_for. Read-only on j, so no atomics
 * or ghost writeback are needed.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

/* Standard and Kokkos headers BEFORE global_data_all_struct.h */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Kokkos_Core.hpp>

/* GPU All mirror: per-TU managed pointer to shared UVM allocation. */
#include "../declarations/gpu_all_mirror.h"
#include "../system/gpu_particles_arena.h"

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/gpu_neighbor_list.h"
#include "../mesh/neighbor_list.h"

#include "../declarations/gpu_numeric_macros.h"
#include "../declarations/gpu_error_check.h"
#include "../declarations/gpu_dispatch_templates.h"

#include "dm_dispersion_gpu.h"


#if defined(GALSF_SUBGRID_WINDS) && (GALSF_SUBGRID_WIND_SCALING==2) \
    && defined(OPENMP_GPU_OFFLOAD) && defined(GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY)

/* DM type filter: the CPU tree-walk code calls ngb_treefind_variable_threads_targeted
   with type_bitmask=2 (i.e. 2^1 = high-res DM). We mirror that exactly. */
#define DM_J_TYPE_BITMASK 2

void disp_density_evaluate_gpu(struct particle_data *P_host, int num_total,
                               int *i_active_host, int num_active,
                               const double *i_radii_host,
                               void *out_host_void)
{
    GIZMO_GPU_ENSURE_ALL_FRESH(dispdensity);
    struct dispdens_gpu_out *out_host = (struct dispdens_gpu_out *)out_host_void;

    /* Step 13 Phase 1 arena. P-only kernel (CellP not used). */
    gpu_particles_arena_acquire(num_total, P_host, NULL);
    struct particle_data *P_gpu = gpu_particles_arena_P();

    /* Build cross-type CSR list: i = gas active indices, j filtered to DM.
       Per-i search radius is the caller's KernelRadiusDM (one-way, no symm). */
    gpu_neighbor_list_t gnl;
    gpu_ngb_list_build(P_gpu, num_total, i_active_host, num_active,
                       NGB_SEARCH_ONEWAY, DM_J_TYPE_BITMASK, &gnl, NULL,
                       1.0 /* search_radius_factor */, i_radii_host);

    /* Per-active output in SharedSpace */
    struct dispdens_gpu_out *d_out = (struct dispdens_gpu_out *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
        ((num_active > 0) ? num_active : 1) * sizeof(struct dispdens_gpu_out));

    /* Per-active search radii in SharedSpace — kernel needs them to re-check r<h_i
       (ONEWAY traversal filters by h_i already, but the cross-type list hasn't
       been exhaustively audited for edge-case neighbors and the re-check is cheap). */
    double *d_radii = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
        ((num_active > 0) ? num_active : 1) * sizeof(double));
    memcpy(d_radii, i_radii_host, num_active * sizeof(double));

    PRINT_STATUS("  GPU disp_density: %d gas active, %d gas→DM pairs", num_active, gnl.total_pairs);

    /* GPU accumulation kernel: read-only on neighbors, write-only to d_out[aa] */
    {
        int *offsets = gnl.offsets;
        int *neighbors = gnl.neighbors;
        int *active = gnl.d_active;
        struct particle_data *kp = P_gpu;
        double *radii = d_radii;
        struct dispdens_gpu_out *kout = d_out;

        gizmo_gpu_kernel_launch("disp_density_kernel", num_active, KOKKOS_LAMBDA(int aa) {
            int ii = active[aa];
            double h_i = radii[aa];
            double h2 = h_i * h_i;

            double Ngb = 0, DM_Vx = 0, DM_Vy = 0, DM_Vz = 0, DM_VelDisp = 0;

            int start = offsets[aa], end = offsets[aa + 1];
            for(int nn = start; nn < end; nn++) {
                int j = neighbors[nn];
                if(kp[j].Mass <= 0) continue;
                Vec3<double> dp = kp[ii].Pos - kp[j].Pos;
                nearest_xyz(dp);
                double r2 = dp.norm_sq();
                if(r2 >= h2) continue;

                double vxj = kp[j].Vel[0], vyj = kp[j].Vel[1], vzj = kp[j].Vel[2];
                DM_Vx += vxj; DM_Vy += vyj; DM_Vz += vzj;
                DM_VelDisp += vxj*vxj + vyj*vyj + vzj*vzj;
                Ngb += 1.0;
            }

            kout[aa].Ngb = Ngb;
            kout[aa].DM_Vx = DM_Vx;
            kout[aa].DM_Vy = DM_Vy;
            kout[aa].DM_Vz = DM_Vz;
            kout[aa].DM_Vel_Disp = DM_VelDisp;
        });
    }

    /* Copy results back to host */
    memcpy(out_host, d_out, num_active * sizeof(struct dispdens_gpu_out));

    /* Read-only on arena; out post-scattered by caller — invalidate. */
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_radii);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_out);
    gpu_ngb_list_free(&gnl, NULL);
    gpu_particles_arena_invalidate();
}

/* Per-TU init function: sets this TU's All_ptr to the shared UVM allocation */
GPU_ALL_SYNC_FUNC(dispdensity)

#else /* stubs when disabled */

void disp_density_evaluate_gpu(struct particle_data *, int, int *, int, const double *, void *) {}
void gizmo_gpu_sync_all_dispdensity(struct global_data_all_processes *p) { (void)p; }

#endif /* GALSF_SUBGRID_WINDS && GALSF_SUBGRID_WIND_SCALING==2 && OPENMP_GPU_OFFLOAD && GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY */
