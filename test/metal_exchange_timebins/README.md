# metal_exchange_timebins

Temporary check for `METALS_EXACT_PAIRWISE_EXCHANGE`, not proposed for the permanent suite: it tests one bookkeeping
property (metal mass closes to round-off with individual timesteps), not broader physics.

A periodic box of 20^3 cells, cold (1e3 K) for x < L/2 and hot (1e7 K) for x > L/2 at equal density, so neighbours
sit in five or more timebins and the pressure jump drives flow and shear. Five metal species (`GALSF_FB_FIRE_RPROCESS=4`
adds four slots): a step 0.1 | 60 Zsun, the reverse step, a step into exactly zero, a smooth profile along y, and a
random per-cell field over four decades with exact zeros. The IC (`make_ics.py`) is read as a snapshot (RestartFlag 2),
so its metallicities are kept.

| test | checks |
|---|---|
| `mfm_closes_at_every_snapshot[1,2,4]` | MFM (diffusion only): each species' metal mass, exact sum over the double-precision snapshot, equal to the IC's to 1e-12 at every snapshot; >= 5 occupied timebins; no metallicity below 0 |
| `mfv_closes_at_the_synchronised_end[second order, LOWORDER]` | MFV (diffusion and metal advection), 4 ranks: the same at the end-of-run snapshot, written after the last synchronised kick (mid-run snapshots carry the drifted mass, not the conserved one) |
| `default_exchange_drifts` | without the flag, same IC, MFM: the metal mass drifts by more than 1e-9, so the check separates the two paths |
