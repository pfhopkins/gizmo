/** \file
    Kokkos allocation telemetry: a process-wide observation of ALL Kokkos-managed
    allocations (SharedSpace/UVM, device, host) via the Kokkos Tools allocate/free
    callbacks. This is a SUPERSET stream, not a per-family counter -- it includes the
    explicitly-accounted particle and tree arrays, plus device scratch and any Kokkos
    internal allocations. The memory ledger reports it separately (never summed with
    the family totals). Kokkos-touching code lives here (a device TU) so the host-only
    ledger TU never includes Kokkos headers.

    The same stream is also CLASSIFIED into buckets by allocation-label prefix and
    memory-space class (see the GIZMO_KOKBUCKET_* / GIZMO_KOKSPACE_* enums in proto.h),
    so the ledger can decompose "Kokkos observed" into particle SoA / tree arrays /
    tree-build scratch / moment-refresh scratch / neighbor-list, with an explicit
    UNCLASSIFIED remainder for everything unlabeled or unrecognized. Space
    classification is conservative substring matching: an unrecognized space name
    buckets as UNKNOWN (and is reported), never guessed.

    Compile-gated: if a backend lacks the Kokkos Tools callback API, define
    GIZMO_KOKKOS_MEM_TELEMETRY=0 for that build and the accessors become inert (the
    ledger then prints "unavailable"), without blocking the rest of the memory report. */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef GIZMO_KOKKOS_MEM_TELEMETRY
#define GIZMO_KOKKOS_MEM_TELEMETRY 1
#endif

/* Kokkos headers come before allvars.h: allvars.h pulls in macros.h, which supplies
   host-only fallback definitions of KOKKOS_FUNCTION / KOKKOS_INLINE_FUNCTION for TUs
   built without Kokkos. Including it first would leave anything in between compiled
   against the fallbacks and make Kokkos redefine them. */
#if GIZMO_KOKKOS_MEM_TELEMETRY
#include <Kokkos_Core.hpp>
#include <atomic>
#include <cstdint>
#include <cstring>
#endif

#include "../declarations/allvars.h"
#include "../core/proto.h"   /* GIZMO_KOKBUCKET_* / GIZMO_KOKSPACE_* enums */

#if GIZMO_KOKKOS_MEM_TELEMETRY

/* Updated from the allocation callbacks. Atomic because the callback fires on
   whichever host thread performs the allocation; current can be raced by concurrent
   allocations, and the high-water needs an atomic compare-max. */
static std::atomic<long long> g_kok_current{0};
static std::atomic<long long> g_kok_highwater{0};

/* Bucket x space counters (row-major bucket x space, same layout the accessor emits). */
static std::atomic<long long> g_bucket_cur[GIZMO_KOKCELL_COUNT];
static std::atomic<long long> g_bucket_hw [GIZMO_KOKCELL_COUNT];
/* First unrecognized space name seen. The callback fires on any host thread, so a
   single atomic "claimed" flag lets exactly one thread write the buffer race-free;
   after that it is only read (at ledger print, well after all writers). */
static std::atomic<bool> g_unknown_space_claimed{false};
static char g_unknown_space_name[64] = {0};

/* Deterministic first-match label-prefix table. treescratch_* precede tree_ for
   clarity even though "treescratch" does not match the "tree_" prefix. Unlabeled
   allocations arrive as "" or "no-label" and fall through to UNCLASSIFIED. */
static int classify_bucket(const char *label)
{
    if(!label) {return GIZMO_KOKBUCKET_UNCLASSIFIED;}
    if(!strncmp(label, "particle_soa_",      13)) {return GIZMO_KOKBUCKET_PARTICLE_SOA;}
    if(!strncmp(label, "gravity_tree_soa",   16)) {return GIZMO_KOKBUCKET_GRAVITY_TREE_SOA;}
    if(!strncmp(label, "gravity_walk",       12)) {return GIZMO_KOKBUCKET_GRAVITY_WALK;}
    if(!strncmp(label, "modea_",              6))  {return GIZMO_KOKBUCKET_MODEA_RUNNER;}
    if(!strncmp(label, "fine_sidecar",       12)) {return GIZMO_KOKBUCKET_FINE_SIDECAR;}
    if(!strncmp(label, "treescratch_build_", 18)) {return GIZMO_KOKBUCKET_TREESCRATCH_BUILD;}
    if(!strncmp(label, "treescratch_moment_",19)) {return GIZMO_KOKBUCKET_TREESCRATCH_MOMENT;}
    if(!strncmp(label, "tree_",               5)) {return GIZMO_KOKBUCKET_TREE_ARRAYS;}
    if(!strncmp(label, "ngl_",                4)) {return GIZMO_KOKBUCKET_NGL;}
    return GIZMO_KOKBUCKET_UNCLASSIFIED;
}

/* Conservative space classification. Order matters: UVM/Shared before the device
   names ("CudaUVM" contains "Cuda"); "Host" before device names ("CudaHostPinned"
   is host-resident). Anything else is UNKNOWN -- better than wrong. */
static int classify_space(const char *space)
{
    if(!space || !space[0]) {return GIZMO_KOKSPACE_UNKNOWN;}
    if(strstr(space, "UVM") || strstr(space, "Shared") || strstr(space, "SHARED")) {return GIZMO_KOKSPACE_SHARED;}
    if(strstr(space, "Host") || strstr(space, "HOST")) {return GIZMO_KOKSPACE_HOST;}
    if(strstr(space, "Cuda") || strstr(space, "CUDA") || strstr(space, "HIP") ||
       strstr(space, "ROCm") || strstr(space, "SYCL")) {return GIZMO_KOKSPACE_DEVICE;}
    if(!g_unknown_space_claimed.exchange(true, std::memory_order_relaxed)) {
        strncpy(g_unknown_space_name, space, sizeof(g_unknown_space_name) - 1);
    }
    return GIZMO_KOKSPACE_UNKNOWN;
}

static void gizmo_kok_on_alloc(Kokkos::Tools::SpaceHandle space, const char *label, const void *, uint64_t size)
{
    long long cur = g_kok_current.fetch_add((long long) size, std::memory_order_relaxed) + (long long) size;
    long long hw = g_kok_highwater.load(std::memory_order_relaxed);
    while(cur > hw && !g_kok_highwater.compare_exchange_weak(hw, cur, std::memory_order_relaxed)) {}

    int cell = classify_bucket(label) * GIZMO_KOKSPACE_COUNT + classify_space(space.name);
    long long bcur = g_bucket_cur[cell].fetch_add((long long) size, std::memory_order_relaxed) + (long long) size;
    long long bhw = g_bucket_hw[cell].load(std::memory_order_relaxed);
    while(bcur > bhw && !g_bucket_hw[cell].compare_exchange_weak(bhw, bcur, std::memory_order_relaxed)) {}
}

static void gizmo_kok_on_free(Kokkos::Tools::SpaceHandle space, const char *label, const void *, uint64_t size)
{
    g_kok_current.fetch_sub((long long) size, std::memory_order_relaxed);
    int cell = classify_bucket(label) * GIZMO_KOKSPACE_COUNT + classify_space(space.name);
    g_bucket_cur[cell].fetch_sub((long long) size, std::memory_order_relaxed);
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

extern "C" void gizmo_kokkos_mem_buckets(long long *cur, long long *hw)
{
    for(int i = 0; i < GIZMO_KOKCELL_COUNT; i++) {
        cur[i] = g_bucket_cur[i].load(std::memory_order_relaxed);
        hw[i]  = g_bucket_hw[i].load(std::memory_order_relaxed);
    }
}

extern "C" const char *gizmo_kokkos_mem_unknown_space_name(void)
{
    return g_unknown_space_name[0] ? g_unknown_space_name : NULL;
}

#else  /* callback API unavailable on this backend */

extern "C" void      gizmo_kokkos_mem_register(void)        {}
extern "C" int       gizmo_kokkos_mem_available(void)       {return 0;}
extern "C" long long gizmo_kokkos_mem_current_bytes(void)   {return 0;}
extern "C" long long gizmo_kokkos_mem_highwater_bytes(void) {return 0;}
extern "C" void gizmo_kokkos_mem_buckets(long long *cur, long long *hw)
{
    memset(cur, 0, sizeof(long long) * GIZMO_KOKCELL_COUNT);
    memset(hw,  0, sizeof(long long) * GIZMO_KOKCELL_COUNT);
}
extern "C" const char *gizmo_kokkos_mem_unknown_space_name(void) {return 0;}

#endif
