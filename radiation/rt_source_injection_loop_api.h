/* rt_source_injection_loop_api.h — host-clean API surface between
 * radiation/rt_source_injection.cc (host orchestration) and
 * radiation/rt_source_injection_loop.cc (runner-template wrapper +
 * Spec hooks).
 *
 * Kokkos- and runner-template-free: the host TU includes this header (plus
 * rt_source_injection_local_in.h) and never sees the full Spec definition.
 * The wrapper `rt_source_injection_run_loop` instantiates
 * `run_neighbor_loop<RtSrcInjectionSpec>` inside the GPU TU.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef RT_SOURCE_INJECTION_LOOP_API_H
#define RT_SOURCE_INJECTION_LOOP_API_H

#include "rt_source_injection_local_in.h"

#ifdef RT_SOURCE_INJECTION

struct particle_data;
struct gas_cell_data;

/* Per-source host pre-fill (relocated from the legacy rt_source_injection_gpu.cc).
 * Reads P_host[i] and CellP_host[i]; under RT_REINJECT_ACCRETED_PHOTONS+Type==5
 * also clears P_host[i].Sink_accreted_photon_energy after capture (deterministic
 * host-serial state mutation; safe by construction since this is called once per
 * active source from the toplevel pre-runner build). */
void rt_source_injection_fill_local(int i,
                                     struct particle_data *P_host,
                                     struct gas_cell_data *CellP_host,
                                     struct RtSrcLocalIn *loc);

/* Non-template runner wrapper. Host TU calls this with the pre-built
 * active-source list + LocalIn pack; the wrapper inside rt_source_injection_loop.cc
 * sets up Aux + NlrSubgroup + neighbor_loop_args and invokes
 * run_neighbor_loop<RtSrcInjectionSpec>. */
void rt_source_injection_run_loop(const int *active_src,
                                   const struct RtSrcLocalIn *host_locals,
                                   int num_active);

#endif /* RT_SOURCE_INJECTION */

#endif /* RT_SOURCE_INJECTION_LOOP_API_H */
