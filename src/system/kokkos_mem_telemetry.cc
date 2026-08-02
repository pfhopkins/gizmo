/** \file
    Kokkos allocation telemetry: a process-wide observation of ALL Kokkos-managed
    allocations (SharedSpace/UVM, device, host) via the Kokkos Tools allocate/free
    callbacks. This is a SUPERSET stream, not a per-family counter -- it includes the
    explicitly-accounted particle and tree arrays, plus device scratch and any Kokkos
    internal allocations. The memory ledger reports it separately (never summed with
    the family totals). Kokkos-touching code lives here (a device TU) so the host-only
    ledger TU never includes Kokkos headers.

    Compile-gated: if a backend lacks the Kokkos Tools callback API, define
    GIZMO_KOKKOS_MEM_TELEMETRY=0 for that build and the accessors become inert (the
    ledger then prints "unavailable"), without blocking the rest of the memory report. */

#ifndef GIZMO_KOKKOS_MEM_TELEMETRY
#define GIZMO_KOKKOS_MEM_TELEMETRY 1
#endif

#if GIZMO_KOKKOS_MEM_TELEMETRY

#include <Kokkos_Core.hpp>
#include <atomic>
#include <cstdint>

/* Updated from the allocation callbacks. Atomic because the callback fires on
   whichever host thread performs the allocation; current can be raced by concurrent
   allocations, and the high-water needs an atomic compare-max. */
static std::atomic<long long> g_kok_current{0};
static std::atomic<long long> g_kok_highwater{0};

static void gizmo_kok_on_alloc(Kokkos::Tools::SpaceHandle, const char *, const void *, uint64_t size)
{
    long long cur = g_kok_current.fetch_add((long long) size, std::memory_order_relaxed) + (long long) size;
    long long hw = g_kok_highwater.load(std::memory_order_relaxed);
    while(cur > hw && !g_kok_highwater.compare_exchange_weak(hw, cur, std::memory_order_relaxed)) {}
}

static void gizmo_kok_on_free(Kokkos::Tools::SpaceHandle, const char *, const void *, uint64_t size)
{
    g_kok_current.fetch_sub((long long) size, std::memory_order_relaxed);
}

/* Register after Kokkos is initialized and before GIZMO's own Kokkos allocations.
   Allocations made inside Kokkos initialization itself are before this point and are
   therefore not observed (the ledger labels the number "observed after registration"). */
extern "C" void gizmo_kokkos_mem_register(void)
{
    Kokkos::Tools::Experimental::set_allocate_data_callback(gizmo_kok_on_alloc);
    Kokkos::Tools::Experimental::set_deallocate_data_callback(gizmo_kok_on_free);
}

extern "C" int       gizmo_kokkos_mem_available(void)       {return 1;}
extern "C" long long gizmo_kokkos_mem_current_bytes(void)   {return g_kok_current.load(std::memory_order_relaxed);}
extern "C" long long gizmo_kokkos_mem_highwater_bytes(void) {return g_kok_highwater.load(std::memory_order_relaxed);}

#else  /* callback API unavailable on this backend */

extern "C" void      gizmo_kokkos_mem_register(void)        {}
extern "C" int       gizmo_kokkos_mem_available(void)       {return 0;}
extern "C" long long gizmo_kokkos_mem_current_bytes(void)   {return 0;}
extern "C" long long gizmo_kokkos_mem_highwater_bytes(void) {return 0;}

#endif
