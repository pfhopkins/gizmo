# single_stars_collect_gas_cr

The `single_stars_collect_gas` lattice (1 Msun cells, stars of 8 and 40 Msun drawn at the first step, merging of light cells off) built
with `COSMIC_RAY_FLUID` and `MAGNETIC`, 2 CR energy units in every cell, a 2 microG field, 2 ranks, start flag 2, one step.

Bar, set before the first run: total CR energy after the step = initial x gas mass ratio, to 1e-6 (without the scaling the mismatch is
the collected mass fraction, 3.5e-3); the ledger row `E_cr_collected` holds what left, to 1e-6. Without a field the box misses the bar by
5.5e-6: the streaming loss then takes the sound speed as Alfven speed. Before the change the build stops with the `#error` in `sfr_collect_gas_loop.h`.
