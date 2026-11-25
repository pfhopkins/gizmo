# ketju_mstar_mpi

MSTAR's MPI sharing (the default with KETJU) against `KETJU_MSTAR_SINGLE_TASK`. By default regions of
`KETJU_MSTAR_MPI_MIN_N` (128) or more bodies are integrated by all the compute tasks GIZMO assigns to them (the O(N^2)
loops are divided over the tasks, each body's force sum is computed by one task in the serial order); with
`KETJU_MSTAR_SINGLE_TASK` one of the tasks integrates and the others wait. The result is the same bit for bit.

Setup: a Plummer sphere of 256 equal-mass circular binaries (plummer_binaries setup: 1 Msun stars, 1000 AU binaries,
Type 5) runs to t = 0.648 on 4 MPI ranks. KetjuRegionRadius = 0.1 pc merges neighbouring binaries into regions of 4 and
more members, and KetjuMinParticlesPerTask = 2 lets the cost allocator give such regions several compute tasks. The
shared arm sets `KETJU_MSTAR_MPI_MIN_N=4` so that these small regions are shared. `TEST_KETJU_MSTAR_TIMING=4` logs every
MSTAR call of a region with 4 or more bodies (bodies, compute tasks, tasks that shared it, steps, wall time) to stderr.

- Control: single task twice, bitwise equal.
- Shared: every snapshot bitwise equal to single task; the log must show that every region with several compute tasks
  was shared by all of them, and that the single-task build shared none.

The counts are written to `ketju_mstar_mpi_compare.txt`. Needs the KETJU install of scripts/get_stella_deps.sh (built
with MSTAR's MPI sharing).

    pytest test/ketju_mstar_mpi/test_ketju_mstar_mpi.py
