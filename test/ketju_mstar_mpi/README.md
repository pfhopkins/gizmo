# ketju_mstar_mpi

MSTAR's MPI sharing (the default with KETJU) against `KETJU_MSTAR_SINGLE_TASK`. By default regions of
`KETJU_MSTAR_MPI_MIN_N` (128) or more bodies are integrated by all the compute tasks GIZMO assigns to them in CPU builds (the O(N^2)
loops are divided over the tasks, each body's force sum is computed by one task in the serial order); with
`KETJU_MSTAR_SINGLE_TASK` one of the tasks integrates and the others wait. The result is the same bit for bit.

Setup: a Plummer sphere of 256 equal-mass circular binaries (`make_ketju_mstar_mpi_ics.py`: 1 Msun stars, 1000 AU binaries,
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

## Speed (measured 2026-11-25/26 with the CPU branch stella_dev, the same MSTAR commit and switch)

- ketju2 zoom restart, 224 ranks, one library commit (MSTAR main 684438c): the region of 764-776 stars takes 611-627 s
  per MSTAR call with `KETJU_MSTAR_SINGLE_TASK` and 18.1-18.8 s shared by its 76-77 tasks (33.7x).
- MSTAR scan on one node (2 x 56 cores), time per GBS step against one task, hard binary, 2 / 4 / 8 / 16 / 32 / 64 / 112
  tasks: N = 128: 1.8 / 1.8 / 2.8 / 4.2 / 5.9 / 7.2 / 5.5x; N = 770: 2.0 / 2.1 / 4.1 / 7.6 / 14.1 / 26.5 / 31.3x;
  N = 1500: 2.0 / 4.0 / 6.0 / 7.9 / 15.3 / 30.9 / 42.2x. Below the threshold (N = 8-64) the default is the single-task
  path: 0.99-1.01 of its time (one point 1.045, re-measured 1.000). Sharing forced at N = 64 would still be faster than
  one task; at N = 16 and 8 it is slower, hence a threshold.
- Small regions inside GIZMO on 4 ranks, KETJU part, default / single task: this test's 256 binaries 10.58 / 10.79 s;
  a longer run of the same setup 55.66 / 55.86 s. Neither setup runs on 112 ranks (too few particles per rank).
- One Kokkos thread per MPI task is the setting to use when the ranks fill the cores.
