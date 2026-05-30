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
 * and across the ghost-writeback MPI reduction. Field additions must update
 * mechfb_gas_delta_{nonzero,zero,add} and the mechfb_writeback_detail
 * callback path in galaxy_sf/mechfb_loop.cc. */
struct MechFBGasDelta
{
    int N_injected;
    int max_source_wakeup;  /* MAX over source events into this receiver of
                             * (source.TimeBin + 1) -- hydro-convention wakeup
                             * value. Applied by mechanical_fb.cc's direct-dU
                             * branch as P[j].wakeup = max(P[j].wakeup, ...).
                             * Identity = 0 (no source events yet). Rides
                             * along with N_injected -- mechfb_gas_delta_nonzero
                             * remains gated on N_injected > 0 (every wakeup
                             * accumulation is coupled to a real feedback delta
                             * write, so this field does not create standalone
                             * ghost packets). */
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
