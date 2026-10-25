/* KETJU_KOKKOS: ketju_create_system / ketju_run_integrator / ketju_free_system on the C++/Kokkos MSTAR library
 * (mstar::System, ketju-integrator/kokkos). The C structs of ketju_integrator.h stay GIZMO's data model; the state is
 * integrated on rank 0 of the region's communicator and broadcast, with the C library's semantics (merger records,
 * extra-data and collision_radius reordering, accumulating step counters, abort on max_step_count). */
#ifndef KETJU_MSTAR_KOKKOS_H
#define KETJU_MSTAR_KOKKOS_H
#if defined(KETJU_REGULARIZATION) && defined(KETJU_KOKKOS)
#include <ketju_integrator/ketju_integrator.h>

void kj_mstar_create_system(struct ketju_system *R, int num_pn_particles, int num_other_particles, MPI_Comm comm);
void kj_mstar_run_integrator(struct ketju_system *R, double time_interval);
void kj_mstar_free_system(struct ketju_system *R);

#endif
#endif
