#ifndef GIZMO_SYSTEM_TIMER_MPI_H
#define GIZMO_SYSTEM_TIMER_MPI_H

#include <cstdio>
#include <mpi.h>

namespace gizmo {
namespace timing {
/* Collective on comm, from its main thread only. Streams are used only on rank
 * zero and may be null. Names are reconciled every snapshot; ranks may visit
 * different regions in different orders. Values are cumulative since startup.
 * Worker rows sum thread seconds, separately from main-thread elapsed seconds. */
void TimerWriteTSVHeader(FILE *stream);
bool TimerReportMPI(MPI_Comm comm, FILE *tree, FILE *tsv,
                    const char *label, long long step);
} // namespace timing
} // namespace gizmo

#endif
