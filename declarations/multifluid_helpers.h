/* multifluid_helpers.h — strict Lagrangian-partition multi-fluid framework.
 *
 * Gated entirely by HYDRO_MULTIFLUID. When the gate is off, this header is
 * still includable safely (no symbols defined), and any call site that
 * references the symbols below without gating produces a hard compile error
 * ("symbol not declared") — which is the intended catch for missing gates.
 *
 * SSOT for: (a) canonical FluidID enum, (b) packed-id pair predicates.
 *
 * Design contract:
 *   - Canonical FluidIDs are STABLE across Configs. Sub-flags
 *     (HYDRO_MULTIFLUID_DUSTGAS, _IONNEUTRAL, ...) gate which handled
 *     cases inside the existing master compute functions (per design §6 —
 *     EOS / cooling / opacity / conductivity / viscosity / SF eligibility /
 *     dust evolution, each branched in-place via switch(P[i].FluidType)
 *     inside the EXISTING function that already populates that quantity)
 *     and which cross-fluid coupling cases inside the existing evaluator /
 *     backreaction-pair prototypes (per design §7, §8) are enabled.
 *     Sub-flags do NOT gate which IDs exist. A snapshot written with one
 *     Config can be read by another with the same IDs.
 *   - NEVER renumber a canonical ID. NEVER reuse a retired ID.
 *   - The _id predicates operate on packed bytes from ActiveData /
 *     NeighborData (the only safe place to read FluidType in a runner
 *     pair body). NO `same_lagrangian_fluid_P(int i, int j)` host-sugar
 *     helper exists — reading global P[] from a KOKKOS_INLINE_FUNCTION
 *     is a device-correctness footgun.
 *   - There is NO per-property "fluid_*" dispatcher layer (no
 *     fluid_eos_pressure, fluid_cooling_update, fluid_opacity_band, etc.).
 *     FluidType variation lives where the quantity is computed, not at
 *     readers. See design §6 + §16 non-goal.
 *
 * Init-time validation (see init.cc): every FluidType value actually
 * present in the loaded particle set must have a handled case in every
 * ENABLED physics master compute function and every enabled coupling
 * pair/evaluator prototype (per design §3, §6). Unknown-but-present
 * FluidType is a HARD FAIL. The `default:` case inside each in-place
 * switch(P[i].FluidType) block is a hard-abort guard, NOT a silent
 * fallback to baseline physics. (In the current Phase 0–2 framework
 * state, no non-default cases exist yet; the init.cc check therefore
 * hard-fails on any nonzero FluidType outright — see init.cc.)
 *
 * Padding note (see declarations/particle_data.h): FluidType is placed
 * after `short int TimeBin` and before `MyIDType ID`, consuming one byte
 * of the existing 4-byte alignment padding before 8-byte-aligned ID.
 * Expected `sizeof(struct particle_data)` is UNCHANGED relative to a
 * baseline build of the same Config. Any size shift is a warning/error
 * — verify with offsetof() per Config at init time.
 *
 * Full design: gpu_bench_new/OPEN_hydro_multifluid_design.md.
 */

#ifndef MULTIFLUID_HELPERS_H
#define MULTIFLUID_HELPERS_H

#include "macros.h"   /* KOKKOS_INLINE_FUNCTION fallback for non-Kokkos builds */

#ifdef HYDRO_MULTIFLUID

/* Stable canonical FluidIDs. Defined unconditionally whenever
 * HYDRO_MULTIFLUID is on; do NOT gate the IDs themselves on sub-flags.
 * Storage type in particle_data.h is `unsigned char` → 256 values total. */
enum FluidID
{
    FLUID_DEFAULT     = 0,
    FLUID_DUST_GRAIN  = 1,
    FLUID_ION         = 2,
    FLUID_NEUTRAL     = 3,
    /* Additional canonical IDs reserved as multifluid physics grows.
     * NEVER renumber. NEVER reuse a retired ID. */
    FLUID_TYPE_MAX_ID = 255   /* uint8 ceiling */
};

/* Pair-skip predicates. Operate on packed FluidType bytes from
 * ActiveData / NeighborData. The ONLY pair-body API; do not add a P[]-based
 * variant marked KOKKOS_INLINE_FUNCTION (device-correctness footgun). */
KOKKOS_INLINE_FUNCTION
bool same_lagrangian_fluid_id(unsigned char ft_i, unsigned char ft_j)
{
    return ft_i == ft_j;
}

KOKKOS_INLINE_FUNCTION
bool cross_lagrangian_fluid_pair_id(unsigned char ft_i, unsigned char ft_j)
{
    return ft_i != ft_j;
}

#endif /* HYDRO_MULTIFLUID */
#endif /* MULTIFLUID_HELPERS_H */
