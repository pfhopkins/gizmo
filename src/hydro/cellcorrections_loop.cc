/* hydro/cellcorrections_loop.cc — host hooks + toplevel for CellcorrectionsSpec.
 *
 * See hydro/cellcorrections_loop.h for the Spec contract. This file owns the
 * host writeback (apply_active_writeback), the oracle compare, and the
 * toplevel cellcorrections_calc() that replaces the legacy walker that lived
 * in hydro/density.cc:62-162. The final per-active closure (Density,
 * Pressure update) remains in hydro/density.cc as
 * cellcorrections_final_operations_and_cleanup(), called from this toplevel.
 *
 * Written by Philip F. Hopkins (phopkins@caltech.edu) for GIZMO. */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"  /* MUST precede allvars.h: installs device-pass `#define All AllDeviceMirror` redirect before cell_data.h is parsed */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"                   /* MUST precede cellcorrections_loop.h
                                                * — kernel.h has no include guards */
#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/neighbor_list.h"     /* gizmo_sym_* globals (corridor CSR view) */
#include "hydro_corridor.h"             /* mode + external_csr accessors */
#include "cellcorrections_loop.h"

#ifdef HYDRO_VOLUME_CORRECTIONS

/* Per-active writeback: accumulator into CellP[i].Volume_1.
 *
 * Legacy semantic preservation: the old walker zeroed CellP[i].Volume_1
 * upstream (set_volume_corrections_to_zero — called once before the loop)
 * and then accumulated. The runner-Spec equivalent: the active list is
 * gated by is_active, so we only touch eligible cells; the accumulator
 * starts at zero (zero_accum) per call. We use += (not =) here in case
 * any future caller pre-loads Volume_1 with a baseline. */
void CellcorrectionsSpec::apply_active_writeback(const neighbor_loop_args& /*args*/,
                                                  int /*active_slot*/, int i,
                                                  const AccumData& accum)
{
    CellP[i].Volume_1 += (MyDouble)accum.volume_1;
}

/* Oracle comparison: single-field L2 relative residual with floor. */

/* Toplevel — replaces the legacy walker in hydro/density.cc:62-162. The
 * runner handles ghost import + arena + CSR build + kernel launch
 * internally; the toplevel is just active-list filtering + dispatch +
 * the final closure (Density / Pressure) call. */
void cellcorrections_calc(void)
{
    CPU_Step[CPU_DENSMISC] += measure_time();
    double t00 = my_second();  const double child0_span = CPU_ChildCharged;
    PRINT_STATUS(" ..calculating first-order corrections to cell sizes/faces");

    /* Corridor consumption: when the corridor has published a Mode-A
     * external CSR (any rank count — see hydro_corridor.cc), consume it
     * directly using the corridor's BROAD row list (Type==0 && Mass>0).
     * The narrow GasGrad_isactive predicate is applied per-row inside the
     * Spec via ActiveData::enabled — rows that don't pass contribute zero,
     * so apply_active_writeback's += leaves CellP[i].Volume_1 untouched.
     * Otherwise (Mode B, UNSET, or an unpublished corridor), build the
     * narrow active list ourselves. */
    const nlr_external_csr *corridor_csr = gizmo_hydro_corridor_external_csr();
    const GizmoHydroCorridorMode corridor_mode = gizmo_hydro_corridor_get_mode();

    int *active_list_local = nullptr;       /* allocated by nlr_build_active_list in fallback */
    int  num_active = 0, num_global_active = 0;
    neighbor_loop_args args = nlr_default_args();

    if(corridor_csr != nullptr) {
        /* Mode A external-CSR path. Row list = corridor's broad list
         * (gizmo_sym_active_indices); ownership stays with the corridor
         * (no nlr_free_active_list call from here). */
        args.active_list = corridor_csr->active_indices;
        args.num_active  = corridor_csr->num_active;
        args.external_csr     = corridor_csr;
        args.dispatch_override = NlrForceMode::A;
    } else {
        /* Mode B (request-driven, no corridor CSR): narrow active list built
         * here. */
        if(!nlr_build_active_list(CellcorrectionsSpec::is_active,
                                   &active_list_local, &num_active, &num_global_active,
                                   "cellcorrections_active")) {
            /* No active gas anywhere globally — nothing to do. */
            CPU_Step[CPU_DENSMISC] += measure_time();
            return;
        }
        /* Mode A always publishes a corridor view when there is active gas;
         * reaching here in Mode A is a corridor sequencing bug — fail loudly,
         * never quietly rebuild (that dual path is retired). */
        if(corridor_mode == GizmoHydroCorridorMode::MODE_A) {
            printf("FATAL: cellcorrections_calc in Mode A with active gas but no published corridor CSR on task %d.\n", ThisTask);
            fflush(stdout);
            endrun(7316);
        }
        args.active_list = active_list_local;
        args.num_active  = num_active;
        if(corridor_mode == GizmoHydroCorridorMode::MODE_B) {
            args.dispatch_override = NlrForceMode::B;
        }
    }

    run_neighbor_loop<CellcorrectionsSpec>(args);

    if(active_list_local) nlr_free_active_list(active_list_local);

    /* Final per-active closure: Density = Mass / Volume_1, set_eos_pressure.
     * This still lives in hydro/density.cc as a host helper (no need to pull
     * it into the GPU TU). */
    cellcorrections_final_operations_and_cleanup();

    double t1 = my_second(); cpu_chain_sync(t1);
    CPU_Step[CPU_DENSMISC] += cpu_minus_children(timediff(t00, t1), child0_span);
}

#endif /* HYDRO_VOLUME_CORRECTIONS */
