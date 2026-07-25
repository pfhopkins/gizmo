/* rt_source_injection_local_in.h — host-clean per-source input pack for the
 * runner-template port of rt_source_injection.
 *
 * This struct lives in its own tiny header so the host orchestration TU
 * (radiation/rt_source_injection.cc) can build `std::vector<RtSrcLocalIn>`
 * WITHOUT pulling Kokkos / runner-template headers. Full Spec definition
 * (and the inline KOKKOS_INLINE_FUNCTION pair kernel) lives in
 * rt_source_injection_loop.h, which is included only from the GPU TU
 * (rt_source_injection_loop.cc).
 *
 * Field set matches the legacy `RtSrcLocalIn` (formerly in
 * rt_source_injection_functions.h, retired in cleanup commit). No physics
 * fields added or removed.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef RT_SOURCE_INJECTION_LOCAL_IN_H
#define RT_SOURCE_INJECTION_LOCAL_IN_H

#include "../declarations/allvars.h"   /* Vec3<>, MyFloat, MyDouble, N_RT_FREQ_BINS */

#ifdef RT_SOURCE_INJECTION

struct RtSrcLocalIn {
    Vec3<MyDouble> Pos;
    MyFloat        KernelRadius;
    MyFloat        KernelSum_Around_RT_Source;
    MyFloat        Luminosity[N_RT_FREQ_BINS];
#if defined(RT_EVOLVE_FLUX)
    Vec3<MyFloat>  Vel;
#endif
#if defined(RT_REPROCESS_INJECTED_PHOTONS) && defined(RT_CHEM_PHOTOION)
    MyDouble       Dt;
    MyDouble       Density;
#endif
};

#endif /* RT_SOURCE_INJECTION */

#endif /* RT_SOURCE_INJECTION_LOCAL_IN_H */
