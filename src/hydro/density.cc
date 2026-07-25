#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "../mesh/mesh_motion.h"
#include "../mesh/neighbor_list.h"
#include "../mesh/sfc_tiles.h"
#include "../mesh/ghost_symlist_lifecycle.h"
#include "../core/step_phases.h"
#include "../system/gpu_particles_arena.h"

/*! \file density.c
 *  \brief hydro kernel size and neighbor determination, volumetric quantities calculated
 *
 *  This file contains the "first hydro loop", where the gas densities and some
 *  auxiliary quantities are computed.  There is also functionality that corrects the kernel length if needed.
 */
/*!
 * This file was originally part of the GADGET3 code developed by Volker Springel.
 * The code has been modified substantially (condensed, different criteria for kernel lengths, optimizatins,
 * rewritten parallelism, new physics included, new variable/memory conventions added, fundamentally different
 * criteria and conditioning and calcuilations actually being done for the modular hydro solvers)
 * by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */


/* density() and density_isactive() live in hydro/density_loop.cc. The
 * cellcorrections routines remain here as peer step-phase entries. */


/* Routines for a loop after the iterative density loop needed to find neighbors, etc, once all have converged, to apply additional correction terms to the cell volumes and faces (for those needed -before- the gradients loop because they alter primitive quantities needed for gradients, such as particle densities, pressures, etc.)
    This was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO. */
#ifdef HYDRO_VOLUME_CORRECTIONS

/* The old CPU exchange scaffolding for cell corrections has been retired.
 * The modern path walks the prebuilt symmetric CSR neighbor list directly. */

/* final operations for after the updates are computed */
void cellcorrections_final_operations_and_cleanup(void)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int _apl = 0; _apl < (int)ActiveParticleList.size(); _apl++) { int i = ActiveParticleList[_apl]; /* check all active elements */
        if(GasGrad_isactive(i, P, CellP)) /* only cells eligible for gradients and hydro */
        {
            if(CellP[i].Volume_1 > 0) {CellP[i].Density = P[i].Mass / CellP[i].Volume_1;} else {CellP[i].Volume_1 = CellP[i].Volume_0;} // set the updated density. other variables that need volumes will all scale off this, so we can rely on it to inform everything else [if bad value here, revert to the 0th-order volume quadrature]
            set_eos_pressure(i, P, CellP);
        }}
}

/* cellcorrections_calc() moved to hydro/cellcorrections_loop.cc. The final
 * per-active closure (Density = Mass/Volume_1 + set_eos_pressure) remains
 * here as cellcorrections_final_operations_and_cleanup above; the toplevel
 * calls it after the runner dispatch. */

#endif // parent if statement for all code in the HYDRO_VOLUME_CORRECTIONS block
