# ketju_task_schedule

KETJU scheduling flags must not change results. A Plummer sphere of 64 equal-mass circular binaries (the plummer_binaries
setup, 1000 AU binaries, Type 5 stars) runs to t = 0.648 on 8 MPI ranks, so the 64 chain regions are spread over ranks and
some have members on two ranks. Variants:

- unchanged code (KETJU_REGULARIZATION only);
- `KETJU_SCATTER_AFTER_ALL_REGIONS`: every region is integrated before any is written back, in the same order;
- `KETJU_ONE_TASK_PER_REGION`: each region integrated by one task, its affected root, in one round;
- both flags.

Every snapshot of every variant must be bitwise equal to the unchanged code's (all datasets, sorted by ID). Bitwise holds
here because each region has 2 members, so the unchanged allocator also gives it one compute task (n / KetjuMinParticlesPerTask
< 1); a region with n >= 20 members can differ at roundoff under `KETJU_ONE_TASK_PER_REGION` (its negative half-kick is then
summed on one task instead of several). Runs take a few minutes.
