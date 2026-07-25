/* hydro/gradients.cc — RETIRED as part of the hydro corridor cleanup.
 *
 * The legacy toplevel hydro_gradient_calc(), `GasGrad_isactive`,
 * `local_slopelimiter`, `construct_gradient`, the file-static
 * `temporary_data_topass` / `GasGradDataPasser`, and the
 * `out2particle_GasGrad{,_iter}` writeback functions all migrated to
 * `hydro/gradients_loop.{h,cc}` as part of the GradientsSpec port.
 *
 * File kept as an empty TU to avoid Makefile churn — `hydro/gradients.o`
 * is still listed in $(OBJS). A follow-up cleanup drops the
 * legacy GPU walker `gradient_evaluate_gpu` from `hydro/density_gpu.cc`
 * (function-bounded delete after pre-grep confirms no remaining callers).
 *
 * Written by Philip F. Hopkins (phopkins@caltech.edu) for GIZMO. */
