# ketju_cloud_restart

KETJU in a gas cloud across a restart. The gmc_cooling cloud (gas as in `test/gmc_cooling`, gravity on, cooling)
holds 16 Type-4 stars near its centre: 4 binaries (a = 100-400 AU, 2-30 Msun) and 8 single stars
(`make_ketju_cloud_restart_ics.py`). Every star can be a chain centre (KetjuRegionRadius 0.01 pc).

Run: straight to t = 0.002, and split: to t = 0.001 (restart file at the final time), then continued with
RestartFlag 1 and TimeMax 0.002. 2 ranks x 2 threads.

Checks: the split run ends where the straight run ends (stars to 1e-6 pc and 1e-3 km/s, gas to 1e-4 pc),
masses equal and total mass conserved to 1e-12 (double-precision output), KETJU regions were integrated and
MSTAR's per-region |dE/E| stays below 1e-6.

The four binaries keep their semi-major axes to 0.01 AU.
