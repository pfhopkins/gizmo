# hot_gas_timebin_clamp

`RESOLVEDISM_HOTSYNC_CLAMP`: while gas above 1e6 K exists, every particle inside a sphere around it (hot-cell centroid, largest hot-cell
distance plus 3 largest kernel radii) takes a step no longer than a quarter of the shortest hot-cell step. Hydro applies each side of a
pair flux on that side's own step, so pairs on different steps across a blast front break energy conservation; the clamp keeps
them in step while the gas is hot. Coordinates are not wrapped, and several remnants at once share one sphere, as stella_dev.

Problem: adiabatic gas, 20^3 cells of 1 Msun, n about 10 cm^-3, one 10 Msun star dying 200 yr in, 2 ranks, 7.4 kyr, snapshots just after each coarsest sync.

Check, on each snapshot: at least 98 % of the gas inside the sphere has a step at most twice the shortest step of the hot gas.
The clamp acts when a particle next becomes active and wakes nobody, so gas entering the growing sphere lags by up to one of its own steps:
between coarse syncs the share is lower (75 % at 2.9 kyr, 6 % without the flag).
