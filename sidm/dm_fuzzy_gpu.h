/* dm_fuzzy_gpu.h — shared types + prototype for GPU DMGrad port.
 *
 * DMGrad_evaluate is a 2-iteration gradient-estimator loop (iter 0 computes
 * gradient of AGS_Density + Psi_Re + Psi_Im; iter 1 computes the tensor of
 * gradients-of-gradients). No j-writes — pure accumulation on the searcher.
 * Host driver DMGrad_gradient_calc() in dm_fuzzy.cc loops over iterations and
 * post-processes with the NV_T matrix.
 *
 * Partition-by-bitmask is trivial here: DMGrad only activates for Type==1
 * particles (see ags_density_isactive + AGSForce_isactive), so effectively
 * one group with the DM bitmask.
 */
#ifndef DM_FUZZY_GPU_H
#define DM_FUZZY_GPU_H

#ifdef DM_FUZZY

/* Per-active output covering both iterations. Unused fields are zeroed per
   iteration by the kernel. Host scatters only the fields matching the
   current iteration into P[i] / DMGradDataPasser[i]. */
struct dmgrad_gpu_out {
    /* Iter 0: first-order gradient + minmax of density-like scalars */
    double grad_rho[3];
    double max_rho, min_rho;
#if (DM_FUZZY > 0)
    double grad_psi_re[3];
    double grad_psi_im[3];
#endif
    /* Iter 1: tensor (second-order) gradient of the gradients, plus minmax */
    double grad2_rho[3][3];     /* grad2_rho[k2][k] = grad in direction k2 of component k of AGS_Gradients_Density */
    double max_grho[3], min_grho[3];
#if (DM_FUZZY > 0)
    double grad2_psi_re[3][3];
    double grad2_psi_im[3][3];
#endif
};

/* Input per-searcher gathered on host, uploaded to SharedSpace. Mirrors the
   CPU INPUT_STRUCT fields needed inside the kernel. */
struct dmgrad_gpu_in {
    double AGS_Density;
    double AGS_Gradients_Density[3];
#if (DM_FUZZY > 0)
    double AGS_Psi_Re;
    double AGS_Gradients_Psi_Re[3];
    double AGS_Psi_Im;
    double AGS_Gradients_Psi_Im[3];
#endif
};

/* Dispatch entry. Builds cross-type CSR with the caller's shared bitmask
   and explicit per-i AGS_KernelRadius, then runs the appropriate iteration's
   accumulation kernel. loop_iteration=0 → density gradient + minmax;
   loop_iteration=1 → gradient-of-gradient tensor. */
void dmgrad_evaluate_gpu(struct particle_data *P_host, int num_total,
                         int *i_active_host, int num_active,
                         const double *i_radii_host,
                         const struct dmgrad_gpu_in *i_inputs_host,
                         int j_type_bitmask,
                         int loop_iteration,
                         void *out_host_void);

#endif /* DM_FUZZY */
#endif /* DM_FUZZY_GPU_H */
