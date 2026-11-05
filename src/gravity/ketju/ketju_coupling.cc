/* KETJU regularized integrator coupling for GIZMO
 *
 * Implements the coupling between GIZMO's gravity tree and the MSTAR
 * regularized N-body integrator. Based on the GADGET4-KETJU architecture
 * (Mannerkoski+ 2022) but adapted for GIZMO's data structures and extended
 * to keep particles individually visible in the tree for radiation physics.
 *
 * Architecture:
 *   1. limit_timesteps(): force chain particles to shared timebin
 *   2. find_regions(): detect chain regions around massive particles
 *   3. run_integration(): collect particles, subtract tree force, run MSTAR
 *   4. set_final_velocities(): swap in true velocities after drift
 *   5. finish_step(): clear region data at end of step
 */

#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <climits>
#include <algorithm>
#include <cctype>
#include <vector>
#include <set>
#include <unordered_map>
#include <queue>
#include <map>
#include <utility>
#include <cfloat>
#include <queue>
#include <functional>
#include <array>

#include "../../declarations/allvars.h"
#include "../../core/proto.h"
#include "../../mesh/kernel.h"
#include "ketju_coupling.h"
#include "../../galaxy_sf/single_star_record.h"
#include "../../system/gpu_particles_arena.h"

#ifdef KETJU_REGULARIZATION

/* KETJU phases all charge the misc bucket (no KETJU slots in the CPU table) */
#define CPU_KETJULIMIT   CPU_MISC
#define CPU_KETJUFIND    CPU_MISC
#define CPU_KETJUEXCH    CPU_MISC
#define CPU_KETJUMSTAR   CPU_MISC
#define CPU_KETJUSCATTER CPU_MISC
#define CPU_KETJUFINISH  CPU_MISC
#ifndef STELLA_OWNS_STAR
#define STELLA_OWNS_STAR(p) ((p).Type == 4)
#endif



struct ketju_member_state &ketju_member(int i) {return SingleStars.ketju[single_star_slot(P[i].StarIdx)];}

/* start-up state of the KETJU fields */
void ketju_init_particle(int i)
{
    P[i].KetjuIntegrated = 0;
    P[i].KetjuChainID = 0;
#ifndef GALSF_SINGLE_STARS_FROM_IMF
    P[i].StarIdx = 0;
#endif
    P[i].KetjuOpenRadius = 0;
}




#endif /* KETJU_REGULARIZATION */
