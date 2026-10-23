#ifndef GIZMO_SYSTEM_TIMER_H
#define GIZMO_SYSTEM_TIMER_H

/* Host instrumentation only. This header deliberately includes neither Kokkos,
 * MPI, GIZMO_config.h nor allvars.h, so both compiler paths can include it. */
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gizmo {
namespace timing {

struct RegionHooks {
    void (*push)(const char *) = nullptr;
    void (*pop)() = nullptr;
    bool main_thread_only = false;
};

struct TimerRecord {
    std::size_t thread = 0;       // 0 is the thread that called TimerInitialize
    std::string path;
    std::uint64_t calls = 0;      // started calls, including active calls
    std::uint64_t completed = 0;
    std::uint64_t active = 0;
    double inclusive = 0;        // elapsed seconds, including children
    double exclusive = 0;        // elapsed seconds, excluding direct children
};

/* Initialize once, on the main thread, before starting instrumented workers.
 * Hooks must be a pair. They run on the thread entering/leaving the region.
 * Disabled timers perform no clock reads, registration or hook calls. */
void TimerInitialize(bool enabled, RegionHooks hooks = RegionHooks());
bool TimerEnabled() noexcept;
std::uint64_t TimerStart(const char *name);
bool TimerStop(const char *name = nullptr) noexcept;
std::vector<TimerRecord> TimerSnapshot();
std::uint64_t TimerErrorCount() noexcept;

/* After workers have quiesced, close this thread's active regions and disable
 * instrumentation. Call before destroying the runtime used by the hooks.
 * Snapshots remain available. No MPI calls or explicit fences; hooks may fence.
 * This function must run on the initializing thread. */
void TimerShutdown() noexcept;

class TimerScope {
public:
    explicit TimerScope(const char *name);
    ~TimerScope() noexcept;
    TimerScope(const TimerScope &) = delete;
    TimerScope &operator=(const TimerScope &) = delete;
    TimerScope(TimerScope &&) = delete;
    TimerScope &operator=(TimerScope &&) = delete;
private:
    std::uint64_t token_;
    const void *owner_;
};

} // namespace timing
} // namespace gizmo

using gizmo::timing::TimerStart;
using gizmo::timing::TimerStop;

#define GIZMO_TIMER_JOIN_INNER(a, b) a##b
#define GIZMO_TIMER_JOIN(a, b) GIZMO_TIMER_JOIN_INNER(a, b)
#define GIZMO_TIMER_SCOPE(name) \
    const ::gizmo::timing::TimerScope GIZMO_TIMER_JOIN(gizmo_timer_scope_, __LINE__)(name)

/* Place at function entry. __func__ keeps the report aligned with source names;
 * use GIZMO_TIMER_SCOPE for phases or an explicitly documented short alias. */
#define GIZMO_TIMER_FUNCTION() GIZMO_TIMER_SCOPE(__func__)

#endif
