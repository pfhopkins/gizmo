/* rt_source_injection_gpu.h — declarations for the B5 GPU port of
 * rt_source_injection.  The host driver in rt_source_injection.cc calls
 * rt_source_injection_evaluate_gpu() as a drop-in replacement for the
 * CPU tree-walk.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef RT_SOURCE_INJECTION_GPU_H
#define RT_SOURCE_INJECTION_GPU_H

void rt_source_injection_evaluate_gpu(struct particle_data *P_host,
                                       struct gas_cell_data *CellP_host,
                                       int num_total,
                                       int *i_active_host, int num_active,
                                       const double *src_radii_host);

#endif /* RT_SOURCE_INJECTION_GPU_H */
