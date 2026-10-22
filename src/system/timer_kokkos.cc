/* Build with the Kokkos compiler. Keep Kokkos headers out of timer.h. */
#include <Kokkos_Core.hpp>
#include "timer.h"

namespace {
void push_region(const char *path) {Kokkos::Profiling::pushRegion(path);}
void pop_region() {Kokkos::Profiling::popRegion();}
}

namespace gizmo {
namespace timing {
RegionHooks TimerKokkosHooks()
{
    RegionHooks hooks;
    hooks.push = push_region;
    hooks.pop = pop_region;
    // Some tools maintain one process-wide region stack. Do not assume that
    // external tools understand interleaved regions from arbitrary host threads.
    hooks.main_thread_only = true;
    return hooks;
}
} // namespace timing
} // namespace gizmo
