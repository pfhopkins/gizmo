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
#if defined(COSMIC_RAY_FLUID)
    /* Phase 2 CR fields. CR_dir_weighted accumulates Σ (Σ_bin dEcr) × dir̂ across
     * all sources; host scatter normalizes to a unit vector and projects onto B
     * if MAGNETIC, then applies per-bin flux = CR_energy_injected[k] × c_red(k). */
    double CR_energy_injected[N_CR_PARTICLE_BINS];
#if defined(CRFLUID_EVOLVE_SPECTRUM)
    double CR_number_injected[N_CR_PARTICLE_BINS];
#endif
    double CR_dir_weighted[3];
#endif
};

#endif /* GALSF_FB_MECHANICAL */
