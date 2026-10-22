#ifndef GIZMO_SYSTEM_TIMER_MPI_H
#define GIZMO_SYSTEM_TIMER_MPI_H

#include <cstdio>
#include <mpi.h>
#include "timer.h"

namespace gizmo {
namespace timing {
enum class TimerTreeSort { InclusiveMax = 0, InclusiveMean = 1 };

struct TimerReportContext {
    const char *mode = "cumulative";
    double simulation_time = 0, simulation_dt = 0;
    long long particles_total = -1;
    long long active_particles_local = -1, active_gas_local = -1;
    int highest_active_bin = -1;
};

/* Collective on comm, from its main thread only. Streams are used only on rank
 * zero and may be null. Names are reconciled every snapshot; ranks may visit
 * different regions in different orders. Default values are cumulative since startup.
 * Worker rows sum thread seconds, separately from main-thread elapsed seconds.
 * Tree siblings are sorted by descending inclusive max (or mean); TSV rows keep
 * canonical path order. The sort argument is used only on rank zero.
 * An optional rank-local snapshot can contain interval deltas. Context counts
 * are reduced across ranks; context/snapshot presence must agree on all ranks. */
void TimerWriteTSVHeader(FILE *stream);
bool TimerReportMPI(MPI_Comm comm, FILE *tree, FILE *tsv,
                    const char *label, long long step,
                    TimerTreeSort sort = TimerTreeSort::InclusiveMax,
                    const std::vector<TimerRecord> *snapshot = nullptr,
                    const TimerReportContext *context = nullptr);
} // namespace timing
} // namespace gizmo

#endif
