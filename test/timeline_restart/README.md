# timeline_restart

A restart that raises TimeMax past the timeline's end halves the integer timeline (`readjust_timebase`). 64 stars with close pairs (long and short steps). `TEST_RESTART_AT_STEP` with `GIZMO_TEST_RESTART_AT_STEP` stops a run part-way through the long steps and writes restart files. A run stopped there with TimeMax = T/2 and restarted with T (timeline halved) must end where the same run stopped with TimeMax = T and restarted ends: same physical steps.

Bar: largest velocity difference below 1e-12 km/s. A straight run is not the reference: any restart differs from it by about 1.5e-4 km/s here (close pairs, forces rebuilt), with or without a timeline change.
