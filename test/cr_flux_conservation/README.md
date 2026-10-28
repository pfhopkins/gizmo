# cr_flux_conservation

A Gaussian cosmic-ray energy bump diffuses along a weak uniform magnetic field (x) through static, cold gas in a 2D
periodic box, for ~33 equal steps (`FORCE_EQUAL_TIMESTEPS`), with two-moment transport (`CRFLUID_DIFFUSION_MODEL=0`).
The IC comes from `make_cr_flux_conservation_ics.py`, and the run starts with restart flag 2 so the IC's CR energies
are kept.

Checks, between the IC and the final snapshot:
- E_cr + E_thermal + E_kinetic changes by less than 1e-5 of the initial CR energy. Magnetic energy is left out: the
  field is uniform and the gas static, and its snapshot value differs from the IC's only through the kernel volume
  estimate.
- The bump spreads along B: `<dx^2>` grows by more than 50%, and by more than 3x the growth of `<dy^2>`.

It takes about 10 s including the build.

## Known sensitivity

- Fails (energy changes by -7.3e-4) when `cosmic_ray_diffusion.h` pairs our predicted flux with the neighbour's
  conserved `CosmicRayFlux` instead of `CosmicRayFluxPred`: the two sides of a face then exchange different amounts.
- Fails the spreading check without `MAGNETIC`: there the M1 flux relaxes toward +grad(P_cr), so the bump sharpens
  (`<dx^2>` 0.0100 -> 0.0098) while energy is still conserved. That is why the test runs with a field.
