/* ags_density_gpu.cc — GPU-accelerated AGS kernel-radius / density loop.
 *
 * GPU translation unit: (Kokkos/nvcc_wrapper). Called once per bitmask group from ags_density() in ags_rkern.cc
 * — see project_tier_b_infra_scope.md for the partition-by-bitmask design.
 *
 * Ports the per-pair body at ags_rkern.cc:174-229 verbatim except for:
 *   - wakeup writes use Kokkos::atomic_store (not #pragma omp atomic)
 *   - NeedToWakeupParticles_local gets atomic_or'd from the kernel
 *   - TimeBinActive[] is captured by value (TIMEBINS=60, negligible)
 *
 * Reads from neighbors (P/CellP) are read-only; the only j-side writes are
 * the atomic wakeup flag. No scatter/refresh needed after the kernel.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

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
#include "../mesh/kernel.h"
#include "../mesh/gpu_neighbor_list.h"
#include "../mesh/neighbor_list.h"

#include "../declarations/gpu_numeric_macros.h"
#include "../declarations/gpu_error_check.h"
#include "../declarations/gpu_dispatch_templates.h"

#include "ags_density_gpu.h"


#if defined(AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE)

/* Named struct (file scope) for TimeBinActive device-capture. CUDA nvcc
   rejects unnamed-type captures in __device__ lambdas ("type local to a
   function"), so we lift this out of the function body. */
struct ags_density_tba_cap_t { int v[TIMEBINS]; };

void ags_density_evaluate_gpu(struct particle_data *P_host,
                              struct gas_cell_data *CellP_host,
                              int num_total,
                              int *i_active_host, int num_active,
                              const double *i_radii_host,
                              int j_type_bitmask,
                              void *out_host_void)
{
    GIZMO_GPU_ENSURE_ALL_FRESH(agsdensity);
    struct ags_density_gpu_out *out_host = (struct ags_density_gpu_out *)out_host_void;

    /* Step 13 Phase 1 arena. CellP only needed when there's gas; pass NULL otherwise.
     * The kernel gates kc[] access on kp[j].Type == 0 so a NULL kc is safe. */
    gpu_particles_arena_set_site("ags_density_gpu");
    gpu_particles_arena_acquire(num_total, P_host, (All.TotN_gas > 0) ? CellP_host : NULL);
    struct particle_data *P_gpu = gpu_particles_arena_P();
    struct gas_cell_data *CellP_gpu = (All.TotN_gas > 0) ? gpu_particles_arena_CellP() : NULL;

    /* Build cross-type CSR: i = caller's active indices (all in one bitmask group),
       j = whatever j_type_bitmask selects. One-way, explicit per-i radii. */
    gpu_neighbor_list_t gnl;
    gpu_ngb_list_build(P_gpu, num_total, i_active_host, num_active,
                       NGB_SEARCH_ONEWAY, j_type_bitmask, &gnl, NULL,
                       1.0 /* search_radius_factor */, i_radii_host);

    /* Output + radii arrays in SharedSpace */
    struct ags_density_gpu_out *d_out = (struct ags_density_gpu_out *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
        ((num_active > 0) ? num_active : 1) * sizeof(struct ags_density_gpu_out));
    double *d_radii = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
        ((num_active > 0) ? num_active : 1) * sizeof(double));
    memcpy(d_radii, i_radii_host, num_active * sizeof(double));

    /* Copy TimeBinActive[] — small (TIMEBINS=60) and needed for wakeup test */
    int tba_host[TIMEBINS];
    for(int k = 0; k < TIMEBINS; k++) tba_host[k] = TimeBinActive[k];

    /* Single-int atomic flag for NeedToWakeupParticles_local (sticky across kernel) */
    int *d_need_wakeup = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sizeof(int));
    *d_need_wakeup = 0;

    /* Cache cf_atime to avoid going through All_ptr inside the lambda */
    double fac_mu = -3.0 / All.cf_atime;
    double time_val = All.Time;
    double time_begin = All.TimeBegin;
    int comoving = All.ComovingIntegrationOn;

    PRINT_STATUS("  GPU AGS-density: %d active, j_bitmask=%d, %d pairs",
                 num_active, j_type_bitmask, gnl.total_pairs);

    {
        int *offsets = gnl.offsets;
        int *neighbors = gnl.neighbors;
        int *active = gnl.d_active;
        struct particle_data *kp = P_gpu;
        struct gas_cell_data *kc = CellP_gpu;
        double *radii = d_radii;
        struct ags_density_gpu_out *kout = d_out;
        int *need_wakeup = d_need_wakeup;

        /* Lambda captures tba by value (int[TIMEBINS]) via a wrapper struct to
           avoid the "variable-length array capture" issue with plain C arrays.
           The struct is declared at file scope (not locally here) because CUDA
           nvcc refuses local types in device-lambda captures. */
        ags_density_tba_cap_t tba_cap;
        for(int k = 0; k < TIMEBINS; k++) tba_cap.v[k] = tba_host[k];

        gizmo_gpu_kernel_launch("ags_density_kernel", num_active, KOKKOS_LAMBDA(int aa) {
            int ii = active[aa];
            double h_i = radii[aa];
            if(!(kp[ii].Mass > 0) || !(h_i > 0)) {
                memset(&kout[aa], 0, sizeof(struct ags_density_gpu_out));
                return;
            }
            double h2 = h_i * h_i;
            double hinv, hinv3, hinv4;
            kernel_hinv(h_i, &hinv, &hinv3, &hinv4);

            Vec3<double> pos_i = kp[ii].Pos;
            Vec3<double> vel_i = kp[ii].Vel;   /* AGS uses P[i].Vel (non-gas side) */

            double Ngb = 0, DrkernNgb = 0, AGS_zeta = 0, AGS_vsig = 0, Particle_DivVel = 0;
#if defined(AGS_FACE_CALCULATION_IS_ACTIVE)
            double NV_T[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
#endif

            int start = offsets[aa], end = offsets[aa + 1];
            for(int nn = start; nn < end; nn++) {
                int j = neighbors[nn];
                if(kp[j].Mass <= 0) continue;
                Vec3<double> dp = pos_i - kp[j].Pos;
                nearest_xyz(dp);
                double r2 = dp.norm_sq();
                if(!(r2 < h2)) continue;
                double r = sqrt(r2);
                double u = r * hinv;
                double wk, dwk;
                kernel_main(u, hinv3, hinv4, &wk, &dwk, 0);

                Ngb += wk;
                DrkernNgb += -(NUMDIMS * hinv * wk + u * dwk);
                AGS_zeta += kp[j].Mass * kernel_gravity(u, hinv, hinv3, 0);

                if(r > 0) {
                    Vec3<double> dv;
                    if(kp[j].Type == 0) { dv = vel_i - kc[j].VelPred; }
                    else                { dv = vel_i - kp[j].Vel; }
                    NGB_SHEARBOX_BOUNDARY_VELCORR_(pos_i, kp[j].Pos, dv, 1);
                    double v_dot_r = dot(dp, dv);
                    if(v_dot_r > 0) v_dot_r *= 0.333333;
                    double vsig = 0.5 * fabs(fac_mu * v_dot_r / r);
                    short int TimeBin_j = kp[j].TimeBin;
                    if(TimeBin_j < 0) TimeBin_j = -TimeBin_j - 1;
                    if(vsig > AGS_vsig) AGS_vsig = vsig;
#if defined(ADAPTIVE_GRAVSOFT_FORALL) || defined(DM_FUZZY) || defined(CBE_INTEGRATOR)
                    volatile int wakeup_condition = 0; /* volatile: nvc++ folds to initial value otherwise */
                    if(!(tba_cap.v[TimeBin_j]) && (time_val > time_begin) && (vsig > WAKEUP * kp[j].AGS_vsig)) {
                        wakeup_condition = 1;
                    }
#if defined(GALSF)
                    if((kp[j].Type == 4) || ((comoving == 0) && ((kp[j].Type == 2) || (kp[j].Type == 3)))) {
                        wakeup_condition = 0;
                    }
#endif
                    if(wakeup_condition) {
                        Kokkos::atomic_store(&kp[j].wakeup, (short int)-1);
                        Kokkos::atomic_store(need_wakeup, 1);
                    }
#endif
                    Particle_DivVel -= dwk * dot(dp, dv) / r;
#if defined(AGS_FACE_CALCULATION_IS_ACTIVE)
                    NV_T[0][0] += wk * dp[0] * dp[0];
                    NV_T[0][1] += wk * dp[0] * dp[1];
                    NV_T[0][2] += wk * dp[0] * dp[2];
                    NV_T[1][1] += wk * dp[1] * dp[1];
                    NV_T[1][2] += wk * dp[1] * dp[2];
                    NV_T[2][2] += wk * dp[2] * dp[2];
#endif
                }
            } /* end neighbor loop */

            kout[aa].Ngb = Ngb;
            kout[aa].DrkernNgb = DrkernNgb;
            kout[aa].AGS_zeta = AGS_zeta;
            kout[aa].AGS_vsig = AGS_vsig;
            kout[aa].Particle_DivVel = Particle_DivVel;
#if defined(AGS_FACE_CALCULATION_IS_ACTIVE)
            kout[aa].NV_T[0][0] = NV_T[0][0]; kout[aa].NV_T[0][1] = NV_T[0][1]; kout[aa].NV_T[0][2] = NV_T[0][2];
            kout[aa].NV_T[1][1] = NV_T[1][1]; kout[aa].NV_T[1][2] = NV_T[1][2]; kout[aa].NV_T[2][2] = NV_T[2][2];
            /* Symmetric — CPU version only fills upper triangle and the
               ASSIGN_ADD driver lets the lower triangle stay zero from memset.
               Mirror that here so numerical results match CPU path. */
            kout[aa].NV_T[1][0] = 0; kout[aa].NV_T[2][0] = 0; kout[aa].NV_T[2][1] = 0;
#endif
        });
    }

    /* Copy results back to host */
    memcpy(out_host, d_out, num_active * sizeof(struct ags_density_gpu_out));
    if(*d_need_wakeup) NeedToWakeupParticles_local = 1;

    /* Read-only on arena; out post-scattered by caller — invalidate. */
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_need_wakeup);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_radii);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_out);
    gpu_ngb_list_free(&gnl, NULL);
    gpu_particles_arena_invalidate();
}

/* Per-TU init function */
GPU_ALL_SYNC_FUNC(agsdensity)

#else /* stubs when disabled */

void ags_density_evaluate_gpu(struct particle_data *, struct gas_cell_data *, int,
                              int *, int, const double *, int, void *) {}
void gizmo_gpu_sync_all_agsdensity(struct global_data_all_processes *p) { (void)p; }

#endif /* AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE */
