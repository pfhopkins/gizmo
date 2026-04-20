/* mechanical_fb_types.h — plain (Kokkos-free) shared type for mechanical_fb.
 *
 * Only MechFBGasDelta lives here so non-GPU translation units (CPU
 * mechanical_fb.cc, mesh/ghost_writeback.cc) can use it without needing to
 * drag in Kokkos / kernel.h. The GPU-facing kernel bodies and the
 * Kokkos-using per-source setup live in mechanical_fb_functions.h, which
 * also includes this header.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#pragma once

#ifdef GALSF_FB_MECHANICAL

/* Per-gas accumulation struct. Layout is shared across CPU + GPU paths,
 * and across the ghost-writeback MPI reduction. Do not reorder without
 * updating ghost_writeback_mechfb in mesh/ghost_writeback.cc. */
struct MechFBGasDelta
{
    int N_injected;
    double m_injected, p_injected[3], KE_injected, TE_injected, Z_injected[NUM_METAL_SPECIES];
#if defined(GALSF_ISMDUSTCHEM_MODEL)
    double Mass_Where_Dust_Shocked;
#endif
};

#endif /* GALSF_FB_MECHANICAL */
