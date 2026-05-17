/* difffilter_gpu.cc — GPU-accelerated DiffFilter velocity smoothing for TURB_DIFF_DYNAMIC.
 *
 * This is a GPU translation unit: GPU translation unit (Kokkos/nvcc_wrapper).
 * Provides difffilter_evaluate_gpu() which builds a wider CSR neighbor list (search radius =
 * TurbDynamicDiffFac * h_i) and runs the DiffFilter per-pair kernel via Kokkos::parallel_for.
 *
 * Called from dynamic_diff_vel_calc() in dynamic_diffusion_velocities.cc.
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
#include "../mesh/kernel.h"
#include "../mesh/gpu_neighbor_list.h"
#include "../mesh/sfc_tiles.h"
#include "../mesh/sfc_tiles_functions.h"

#include "../declarations/gpu_numeric_macros.h"
#include "../declarations/gpu_error_check.h"
#include "../declarations/gpu_dispatch_templates.h"


#if defined(TURB_DIFF_DYNAMIC)

/* Output struct: matches the fields accumulated by DiffFilter_evaluate */
struct DiffFilter_out {
    double Norm_hat;
    double Velocity_bar[3];
    double FilterWidth_bar;
    double MaxDistance_for_grad;
};


void difffilter_evaluate_gpu(struct particle_data *P_host, struct gas_cell_data *CellP_host,
                             int num_total, int *active_indices_host, int num_active,
                             void *out_host_void)
{
    GIZMO_GPU_ENSURE_ALL_FRESH();
    struct DiffFilter_out *out_host = (struct DiffFilter_out *)out_host_void;

    /* Wrapper fast-path: no active gas → skip arena/allocs/kernel/scatter. */
    if(num_active == 0) { (void)out_host; return; }

    /* Step 13 Phase 1 arena. */
    gpu_particles_arena_acquire(num_total, P_host, CellP_host);
    struct particle_data *P_gpu = gpu_particles_arena_P();
    struct gas_cell_data *CellP_gpu = gpu_particles_arena_CellP();

    /* Codex 2026-05-12: capture All.* into host local from canonical accessor;
     * see feedback_all_dev_trap_host_side.md. Bare All.* in this GPU TU
     * would resolve to potentially-stale All_dev. */
    const double TurbDynDiffFac_host = gizmo_host_all_ptr()->TurbDynamicDiffFac;

    /* Build wider CSR neighbor list: search radius = TurbDynamicDiffFac * h_i */
    gpu_neighbor_list_t gnl;
    gpu_ngb_list_build(P_gpu, num_total, active_indices_host, num_active,
                       NGB_SEARCH_SYMMETRIC, 1 /* gas only */, &gnl, NULL,
                       TurbDynDiffFac_host, NULL, NULL, "diff");

    /* Allocate output array in SharedSpace */
    struct DiffFilter_out *d_out = (struct DiffFilter_out *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
        ((num_active > 0) ? num_active : 1) * sizeof(struct DiffFilter_out));

    PRINT_STATUS("  GPU DiffFilter: %d active, %d pairs (search_fac=%.1f)",
                 num_active, gnl.total_pairs, TurbDynDiffFac_host);

    /* GPU DiffFilter accumulation kernel */
    {
        int *offsets = gnl.offsets;
        int *neighbors = gnl.neighbors;
        int *active = gnl.d_active;
        struct particle_data *kp = P_gpu;
        struct gas_cell_data *kc = CellP_gpu;
        struct DiffFilter_out *kout = d_out;
        double TurbDynDiffFac = TurbDynDiffFac_host;

        gizmo_gpu_kernel_launch("DiffFilter", num_active, KOKKOS_LAMBDA(int aa) {
            int ii = active[aa];

            /* Load searching particle data (particle2in equivalent) */
            double h_i = kp[ii].KernelRadius;
            double h2_i = h_i * h_i;
            double density_i = kc[ii].Density;
            Vec3<double> velpred_i = kc[ii].VelPred;
#ifdef GALSF_SUBGRID_WINDS
            double delaytime_i = kc[ii].DelayTime;
#endif
            if(kp[ii].Mass <= 0 || density_i <= 0 || h_i <= 0) {
                memset(&kout[aa], 0, sizeof(struct DiffFilter_out));
                return;
            }

            /* Initialize output */
            double out_Norm_hat = 0;
            double out_Velocity_bar[3] = {0, 0, 0};
            double out_FilterWidth_bar = 0;
            double out_MaxDistance_for_grad = 0;

            double hinv_i, hinv3_i, hinv4_i;
            kernel_hinv(h_i, &hinv_i, &hinv3_i, &hinv4_i);

            /* Loop over neighbors */
            int start = offsets[aa], end = offsets[aa + 1];
            for(int nn = start; nn < end; nn++) {
                int j = neighbors[nn];
#ifdef GALSF_SUBGRID_WINDS
                if(delaytime_i == 0 && kc[j].DelayTime > 0) continue;
                if(delaytime_i > 0 && kc[j].DelayTime == 0) continue;
#endif
                if(kp[j].Mass <= 0) continue;
                if(kc[j].Density <= 0) continue;

                Vec3<double> dp = kp[ii].Pos - kp[j].Pos;
                nearest_xyz(dp);
                double r2 = dp.norm_sq();
                if(r2 <= 0) continue;

                double mean_weight = 0.5 * (density_i + kc[j].Density) / (density_i * kc[j].Density);
                double h_j = kp[j].KernelRadius;
                double V_j = kp[j].Mass * mean_weight;

                double hhat = TurbDynDiffFac * h_i;
                double hhat_j = TurbDynDiffFac * h_j;
                double hhat2 = hhat * hhat;
                double hhatj2 = hhat_j * hhat_j;
                if(r2 >= hhat2 && r2 >= hhatj2) continue;

                /* Norm_hat: wide-kernel contribution */
                double r = sqrt(r2);
                if(r < hhat) {
                    double hhatinv, hhatinv3, hhatinv4, uhat, wkhat, dwkhat;
                    kernel_hinv(hhat, &hhatinv, &hhatinv3, &hhatinv4);
                    uhat = DMIN(r * hhatinv, 1.0);
                    kernel_main(uhat, hhatinv3, hhatinv4, &wkhat, &dwkhat, 0);
                    out_Norm_hat += kp[j].Mass * wkhat;
                }

                /* Skip pair if beyond normal kernel range */
                if(r2 >= h2_i && r2 >= (h_j * h_j)) continue;

                if(r > out_FilterWidth_bar) out_FilterWidth_bar = r;
                if(r > out_MaxDistance_for_grad) out_MaxDistance_for_grad = r;

                /* Velocity smoothing: use average h */
                double h_avg = 0.5 * (h_i + h_j);
                double hinv, hinv3, hinv4;
                kernel_hinv(h_avg, &hinv, &hinv3, &hinv4);
                double u = DMIN(r * hinv, 1.0);
                double wk_i, dwk_i;
                if(u < 1) { kernel_main(u, hinv3, hinv4, &wk_i, &dwk_i, 0); } else { wk_i = dwk_i = 0; }
                double weight_i = wk_i * V_j;

                if(r < h_avg) {
                    for(int k=0; k<3; k++) {
                        out_Velocity_bar[k] += (kc[j].VelPred[k] - velpred_i[k]) * weight_i;
                    }
                }
            } /* end neighbor loop */

            kout[aa].Norm_hat = out_Norm_hat;
            for(int k=0; k<3; k++) kout[aa].Velocity_bar[k] = out_Velocity_bar[k];
            kout[aa].FilterWidth_bar = out_FilterWidth_bar;
            kout[aa].MaxDistance_for_grad = out_MaxDistance_for_grad;
        });
    }

    /* Copy results back to host */
    memcpy(out_host, d_out, num_active * sizeof(struct DiffFilter_out));

    /* Read-only on arena; out_host post-scattered by caller into CellP — invalidate. */
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_out);
    gpu_ngb_list_free(&gnl, NULL);
    gpu_particles_arena_invalidate();
}


/* ================================================================
   DynamicDiff GPU kernel — neighbor loop for dynamic Smagorinsky coefficient.
   Called once per dynamic_iteration from dynamic_diff_calc().
   Same wider CSR list as DiffFilter (search_radius = TurbDynamicDiffFac * h_i).
   ================================================================ */

/* Per-particle input gathered from CellP + DynamicDiffDataPasser on host */
struct DynDiff_gpu_in {
    double VelShear_bar[3][3];
    double TD_DynDiffCoeff;
    double MagShear_bar;
    double Velocity_bar[3];
    double Velocity_hat[3];
    double Norm_hat;
    double Dynamic_numerator;
    double Dynamic_denominator;
    double FilterWidth_bar;
    double KernelRadius;
    int sph_gradients_flag;
#ifdef GALSF_SUBGRID_WINDS
    double DelayTime;
#endif
};

/* Output for dynamic_iteration == 0: gradients + hat-filtered quantities */
struct DynDiff_gpu_out0 {
    double GradVelocity_hat[3][3];  /* velocity gradient tensor */
    double Maxima_Velocity_hat[3];
    double Minima_Velocity_hat[3];
    double FilterWidth_hat;
    double Dynamic_numerator_hat;
    double Dynamic_denominator_hat;
    double ProductVelocity_hat[3][3];
};

/* Output for all iterations: iterated dynamic_fac tensor */
struct DynDiff_gpu_out_iter {
    double dynamic_fac[3][3];
#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
    double dynamic_fac_const[3][3];
#endif
};


void dynamicdiff_evaluate_gpu(struct particle_data *P_host, struct gas_cell_data *CellP_host,
                              int num_total, int *active_indices_host, int num_active,
                              int *csr_offsets, int *csr_neighbors, int csr_total_pairs,
                              void *in_host_void, void *out0_host_void, void *out_iter_host_void,
                              int dynamic_iteration)
{
    GIZMO_GPU_ENSURE_ALL_FRESH();
    struct DynDiff_gpu_in *in_host = (struct DynDiff_gpu_in *)in_host_void;
    struct DynDiff_gpu_out0 *out0_host = (struct DynDiff_gpu_out0 *)out0_host_void;
    struct DynDiff_gpu_out_iter *out_iter_host = (struct DynDiff_gpu_out_iter *)out_iter_host_void;

    /* Wrapper fast-path: no active sources → skip arena/CSR copy/kernel/scatter. */
    if(num_active == 0) {
        (void)in_host; (void)out0_host; (void)out_iter_host;
        (void)csr_offsets; (void)csr_neighbors; (void)csr_total_pairs;
        (void)dynamic_iteration;
        return;
    }

    /* Step 13 Phase 1 arena. */
    gpu_particles_arena_acquire(num_total, P_host, CellP_host);
    struct particle_data *P_gpu = gpu_particles_arena_P();
    struct gas_cell_data *CellP_gpu = gpu_particles_arena_CellP();

    /* Copy CSR to SharedSpace */
    int *d_offsets = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>((num_active + 1) * sizeof(int));
    int *d_neighbors = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((csr_total_pairs > 0) ? csr_total_pairs : 1) * sizeof(int));
    int *d_active = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((num_active > 0) ? num_active : 1) * sizeof(int));
    memcpy(d_offsets, csr_offsets, (num_active + 1) * sizeof(int));
    memcpy(d_neighbors, csr_neighbors, csr_total_pairs * sizeof(int));
    memcpy(d_active, active_indices_host, num_active * sizeof(int));

    /* Copy per-particle input to SharedSpace */
    struct DynDiff_gpu_in *d_in = (struct DynDiff_gpu_in *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
        ((num_active > 0) ? num_active : 1) * sizeof(struct DynDiff_gpu_in));
    memcpy(d_in, in_host, num_active * sizeof(struct DynDiff_gpu_in));

    /* Allocate outputs in SharedSpace */
    struct DynDiff_gpu_out0 *d_out0 = (struct DynDiff_gpu_out0 *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
        ((num_active > 0) ? num_active : 1) * sizeof(struct DynDiff_gpu_out0));
    struct DynDiff_gpu_out_iter *d_out_iter = (struct DynDiff_gpu_out_iter *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
        ((num_active > 0) ? num_active : 1) * sizeof(struct DynDiff_gpu_out_iter));

    PRINT_STATUS("  GPU DynamicDiff: %d active, %d pairs, iter=%d", num_active, csr_total_pairs, dynamic_iteration);

    /* GPU kernel */
    {
        int *offsets = d_offsets;
        int *neighbors = d_neighbors;
        int *active = d_active;
        struct particle_data *kp = P_gpu;
        struct gas_cell_data *kc = CellP_gpu;
        struct DynDiff_gpu_in *kin = d_in;
        struct DynDiff_gpu_out0 *kout0 = d_out0;
        struct DynDiff_gpu_out_iter *kout_iter = d_out_iter;
        int dyn_iter = dynamic_iteration;
        /* Codex 2026-05-12: canonical host accessor; see feedback_all_dev_trap_host_side.md */
        const struct global_data_all_processes *host_all = gizmo_host_all_ptr();
        double TurbDynDiffFac = host_all->TurbDynamicDiffFac;
        double TurbDynDiffSmoothing = host_all->TurbDynamicDiffSmoothing;

        gizmo_gpu_kernel_launch("DynamicDiff", num_active, KOKKOS_LAMBDA(int aa) {
            int ii = active[aa];
            struct DynDiff_gpu_in local = kin[aa];

            memset(&kout0[aa], 0, sizeof(struct DynDiff_gpu_out0));
            /* Initialize minima to large values */
            for(int k=0; k<3; k++) { kout0[aa].Minima_Velocity_hat[k] = 1e30; kout0[aa].Maxima_Velocity_hat[k] = -1e30; }
            memset(&kout_iter[aa], 0, sizeof(struct DynDiff_gpu_out_iter));

            double h_i = TurbDynDiffFac * local.KernelRadius;
            double h2_i = h_i * h_i;
            double mass_i = (local.sph_gradients_flag) ? -kp[ii].Mass : kp[ii].Mass;
            if(kp[ii].Mass <= 0 || kc[ii].Density <= 0 || local.KernelRadius <= 0) return;

            double prefactor_i = local.FilterWidth_bar * local.FilterWidth_bar * local.TD_DynDiffCoeff * local.MagShear_bar;
#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
            double prefactor_const_i = local.FilterWidth_bar * local.FilterWidth_bar * local.MagShear_bar;
#endif

            int start = offsets[aa], end = offsets[aa + 1];
            for(int nn = start; nn < end; nn++) {
                int j = neighbors[nn];
#ifdef GALSF_SUBGRID_WINDS
                if(local.DelayTime == 0 && kc[j].DelayTime > 0) continue;
                if(local.DelayTime > 0 && kc[j].DelayTime == 0) continue;
#endif
                if(kp[j].Mass <= 0 || kc[j].Density <= 0) continue;

                Vec3<double> dp = kp[ii].Pos - kp[j].Pos;
                nearest_xyz(dp);
                double r2 = dp.norm_sq();
                double h_j = TurbDynDiffFac * kp[j].KernelRadius;
                if(r2 <= 0) continue;
                if(r2 >= h2_i && r2 >= (h_j * h_j)) continue;

                double h_avg = 0.5 * (h_i + h_j);
                double mean_weight = 0.5 * (kc[j].Norm_hat + local.Norm_hat) / (local.Norm_hat * kc[j].Norm_hat);
                double V_j = kp[j].Mass * mean_weight;
                double r = sqrt(r2);

                if(r > kout0[aa].FilterWidth_hat) kout0[aa].FilterWidth_hat = r;

                double hinv, hinv3, hinv4;
                kernel_hinv(h_avg, &hinv, &hinv3, &hinv4);
                double u = DMIN(r * hinv, 1.0);
                double wk_i, dwk_i;
                if(u < 1) { kernel_main(u, hinv3, hinv4, &wk_i, &dwk_i, 0); } else { wk_i = dwk_i = 0; }
                double weight_i = wk_i * V_j;

                if(dyn_iter == 0) {
                    /* Velocity_hat differences for gradient + min/max */
                    double dv_hat[3];
                    for(int k=0; k<3; k++) {
                        dv_hat[k] = kc[j].Velocity_hat[k] - local.Velocity_hat[k];
                        if(dv_hat[k] < kout0[aa].Minima_Velocity_hat[k]) kout0[aa].Minima_Velocity_hat[k] = dv_hat[k];
                        if(dv_hat[k] > kout0[aa].Maxima_Velocity_hat[k]) kout0[aa].Maxima_Velocity_hat[k] = dv_hat[k];
                    }

                    /* Gradient of Velocity_hat */
                    if(r < local.KernelRadius) {
                        double hinv_fg, hinv3_fg, hinv4_fg, u_fg, wk_fg, dwk_fg;
                        kernel_hinv(local.KernelRadius, &hinv_fg, &hinv3_fg, &hinv4_fg);
                        u_fg = DMIN(r * hinv_fg, 1.0);
                        kernel_main(u_fg, hinv3_fg, hinv4_fg, &wk_fg, &dwk_fg, 0);
                        if(local.sph_gradients_flag) { wk_fg = -dwk_fg * kp[j].Mass / r; }
                        for(int k=0; k<3; k++) {
                            double grad_pf = -wk_fg * dp[k];
                            for(int v=0; v<3; v++) {
                                kout0[aa].GradVelocity_hat[v][k] += grad_pf * dv_hat[v];
                            }
                        }
                    }
                } /* dyn_iter == 0 */

                if(r < h_avg) {
                    if(dyn_iter == 0) {
                        double Dnum_diff = kc[j].Dynamic_numerator - local.Dynamic_numerator;
                        double Dden_diff = kc[j].Dynamic_denominator - local.Dynamic_denominator;
                        kout0[aa].Dynamic_numerator_hat += Dnum_diff * weight_i;
                        kout0[aa].Dynamic_denominator_hat += Dden_diff * weight_i;
                    }

                    double prefactor_j = kc[j].FilterWidth_bar * kc[j].FilterWidth_bar * kc[j].TD_DynDiffCoeff * kc[j].MagShear_bar;
#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
                    double prefactor_const_j = kc[j].FilterWidth_bar * kc[j].FilterWidth_bar * kc[j].MagShear_bar;
#endif
                    for(int k=0; k<3; k++) {
                        for(int v=0; v<3; v++) {
                            double dfac_diff = prefactor_j * kc[j].VelShear_bar[k][v] - prefactor_i * local.VelShear_bar[k][v];
                            kout_iter[aa].dynamic_fac[k][v] += dfac_diff * weight_i;
#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
                            double dfac_const_diff = prefactor_const_j * kc[j].VelShear_bar[k][v] - prefactor_const_i * local.VelShear_bar[k][v];
                            kout_iter[aa].dynamic_fac_const[k][v] += dfac_const_diff * weight_i;
#endif
                            if(dyn_iter == 0) {
                                double pv_diff = kc[j].Velocity_bar[k] * kc[j].Velocity_bar[v] - local.Velocity_bar[k] * local.Velocity_bar[v];
                                kout0[aa].ProductVelocity_hat[k][v] += pv_diff * weight_i;
                            }
                        }
                    }
                }
            } /* end neighbor loop */
        });
    }

    /* Copy results back to host */
    memcpy(out0_host, d_out0, num_active * sizeof(struct DynDiff_gpu_out0));
    memcpy(out_iter_host, d_out_iter, num_active * sizeof(struct DynDiff_gpu_out_iter));

    /* Free GPU allocations */
    /* Read-only on arena; out post-scattered by caller — invalidate. */
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_out_iter);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_out0);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_in);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_active);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_neighbors);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_offsets);
    gpu_particles_arena_invalidate();
}


/* Per-TU init function: sets this TU's All_ptr to the shared UVM allocation */

#else /* stubs */

void difffilter_evaluate_gpu(struct particle_data *, struct gas_cell_data *,
                             int, int *, int, void *) {}
void dynamicdiff_evaluate_gpu(struct particle_data *, struct gas_cell_data *,
                              int, int *, int, int *, int *, int,
                              void *, void *, void *, int) {}

#endif /* TURB_DIFF_DYNAMIC */
