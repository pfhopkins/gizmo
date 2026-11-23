# ketju_task_schedule

KETJU scheduling flags against the unchanged code. A Plummer sphere of 256 equal-mass circular binaries (plummer_binaries
setup: 1 Msun stars, 1000 AU binaries, Type 5) runs to t = 0.648 on 4 MPI ranks (8 ranks with 64 binaries stop in the domain decomposition: too few particles per rank). KetjuRegionRadius = 0.1 pc merges
neighbouring binaries into regions of 4 and more members, and KetjuMinParticlesPerTask = 2 lets the cost allocator give
such regions several compute tasks, so the multi-task paths (split negative half-kick, compute/affected forwards,
kj_bcast_state) run. The test asserts from the KETJU_VERBOSE_INTEGRATION lines that at least one region got several tasks.

- Control: the unchanged code twice, bitwise equal.
- `KETJU_SCATTER_AFTER_ALL_REGIONS` (all regions integrated before any write-back, same order): bitwise equal to the unchanged code.
- `KETJU_ONE_TASK_PER_REGION` (each region integrated by its affected root, one round): with and without the scatter flag
  bitwise equal to each other. Against the unchanged code it changes the summation order of the negative half-kick of
  multi-task regions (roundoff, amplified by the chaotic dynamics), so only the energy error is compared:
  |dE/E|(one task) <= 2 |dE/E|(unchanged) + 1e-6, both from a direct sum on the synchronised (r, v) of the first and last
  snapshot. The numbers and the bitwise result are written to `ketju_task_schedule_compare.txt`.
