# ketju_mstar_mpi

`KETJU_MSTAR_MPI` against the unchanged code. Without the flag the Kokkos MSTAR integrates a region on one of the compute
tasks GIZMO assigns to it and the others wait; with it, regions of `KETJU_MSTAR_MPI_MIN_N` (default 128) or more bodies
are integrated by all their compute tasks together in CPU builds (MSTAR's MPI sharing: the O(N^2) loops are divided over the tasks,
each body's force sum is computed by one task in the serial order). The result is the same bit for bit.

Setup: a Plummer sphere of 256 equal-mass circular binaries (`make_ketju_mstar_mpi_ics.py`: 1 Msun stars, 1000 AU binaries,
Type 5) runs to t = 0.648 on 4 MPI ranks. KetjuRegionRadius = 0.1 pc merges neighbouring binaries into regions of 4 and
more members, and KetjuMinParticlesPerTask = 2 lets the cost allocator give such regions several compute tasks. The
variant sets `KETJU_MSTAR_MPI_MIN_N=4` so that these small regions are shared. `TEST_KETJU_MSTAR_TIMING=4` logs every
MSTAR call of a region with 4 or more bodies (bodies, compute tasks, tasks that shared it, steps, wall time).

- Control: the unchanged code twice, bitwise equal.
- `KETJU_MSTAR_MPI`: every snapshot bitwise equal to the unchanged code; the log must show that every region with
  several compute tasks was shared by all of them, and that the unchanged code shared none.

The counts are written to `ketju_mstar_mpi_compare.txt`.

Needs a KETJU install built with `MSTAR_ENABLE_MPI` at `KETJU_PATH` (its `ketju.stamp` says so); skipped otherwise.

    KETJU_PATH=<MSTAR install with MPI> pytest test/ketju_mstar_mpi/test_ketju_mstar_mpi.py
