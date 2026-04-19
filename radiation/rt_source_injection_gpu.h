/* rt_source_injection_gpu.h — declarations for the B5 GPU port of
 * rt_source_injection.  The host driver in rt_source_injection.cc calls
 * rt_source_injection_evaluate_gpu() as a drop-in replacement for the
 * CPU tree-walk when GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY is active.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef RT_SOURCE_INJECTION_GPU_H
#define RT_SOURCE_INJECTION_GPU_H

void rt_source_injection_evaluate_gpu(struct particle_data *P_host,
                                       struct gas_cell_data *CellP_host,
                                       int num_total);
void gizmo_gpu_sync_all_rtsrcinjection(struct global_data_all_processes *p);

#endif /* RT_SOURCE_INJECTION_GPU_H */
