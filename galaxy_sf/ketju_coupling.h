/* KETJU regularized integrator coupling for GIZMO
 * Provides accurate N-body dynamics near massive stars and BHs
 * using the MSTAR integrator (Rantala+ 2020) with chain regularization.
 *
 * Key design: particles stay in the tree (TREE_RAD / G0 / PI remain correct).
 * Internal chain gravity is subtracted via negative half-step kicks,
 * then replaced by MSTAR's accurate unsoftened integration.
 */
#ifndef KETJU_COUPLING_H
#define KETJU_COUPLING_H

#ifdef KETJU_REGULARIZATION

#include <ketju_integrator/ketju_integrator.h>

/* called from proto.h — these are the 6 interface functions */
void ketju_limit_timesteps(void);
void ketju_find_regions(void);
void ketju_run_integration(void);
void ketju_set_final_velocities(void);
void ketju_finish_step(void);
int ketju_is_particle_in_region(int i);

#endif /* KETJU_REGULARIZATION */
#endif /* KETJU_COUPLING_H */
