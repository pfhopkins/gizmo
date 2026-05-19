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

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../core/step_phases.h"
#include "../mesh/kernel.h"                   /* MUST precede cellcorrections_loop.h
                                                * — kernel.h has no include guards */
#include "../mesh/neighbor_loop_runner.h"
#include "cellcorrections_loop.h"

/* Note: do NOT include declarations/gpu_all_mirror.h here. That header
 * transitively pulls in declarations/macros.h (via global_data_all_struct.h)
 * which defines a `terminate(x)` macro; if it precedes allvars.h's
 * `<vector>` → `<exception>` chain, libc++ `void terminate() _NOEXCEPT`
 * gets macro-mangled. Existing Spec _loop.cc files (thermal_fb, sink_env1,
 * density) follow this pattern — All-mirror registration is auto-via global
 * constructor in cooling/cooling.cc, no explicit per-TU include needed. */

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
double CellcorrectionsSpec::compare_accum(const AccumData& local, const AccumData& oracle)
{
    double a = local.volume_1;
    double b = oracle.volume_1;
    double denom = fmax(fmax(fabs(a), fabs(b)), 1e-30);
    return fabs(a - b) / denom;
}

/* Toplevel — replaces the legacy walker in hydro/density.cc:62-162. The
 * runner handles ghost import + arena + CSR build + kernel launch
 * internally; the toplevel is just active-list filtering + dispatch +
 * the final closure (Density / Pressure) call. */
void cellcorrections_calc(void)
{
    CPU_Step[CPU_DENSMISC] += measure_time();
    double t00 = my_second();
    PRINT_STATUS(" ..calculating first-order corrections to cell sizes/faces");

    /* Active list: gas + Mass>0 + GasGrad_isactive + KernelRadius>0
     * (matches legacy hydro/density.cc:79-86 filter exactly via
     * CellcorrectionsSpec::is_active). */
    int *active_list = nullptr;
    int  num_active = 0, num_global_active = 0;
    if(!nlr_build_active_list(CellcorrectionsSpec::is_active,
                               &active_list, &num_active, &num_global_active,
                               "cellcorrections_active")) {
        /* No active gas anywhere globally — nothing to do (and no
         * cleanup pass needed; nlr_build_active_list freed the unused
         * active_list buffer for us before returning false). */
        CPU_Step[CPU_DENSMISC] += measure_time();
        return;
    }

    neighbor_loop_args args = nlr_default_args();
    args.active_list = active_list;
    args.num_active  = num_active;
    /* aux unused; external_csr unused (commit 5b will add corridor injection) */
    run_neighbor_loop<CellcorrectionsSpec>(args);

    nlr_free_active_list(active_list);

    /* Final per-active closure: Density = Mass / Volume_1, set_eos_pressure.
     * This still lives in hydro/density.cc as a host helper (no need to pull
     * it into the GPU TU). */
    cellcorrections_final_operations_and_cleanup();

    double t1 = WallclockTime = my_second();
    CPU_Step[CPU_DENSMISC] += timediff(t00, t1);
}

#endif /* HYDRO_VOLUME_CORRECTIONS */
