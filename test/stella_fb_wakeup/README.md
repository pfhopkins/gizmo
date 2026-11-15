# stella_fb_wakeup

Stella feedback (winds, supernovae, radiation-pressure kicks) flags the gas cells it heats or kicks for wake-up, so a cell
on a long step moves to a short one at the next sync point. `process_wake_ups` scans only the cells marked in the dirty
sidecar `WakeupDirty`, and only when some rank raised `NeedToWakeupParticles_local`, so every flag must also mark the sidecar
and raise the flag. (stella_dev keeps a pending index list instead and fixed five Stella sites that did not record into it,
fix/stella-feedback-wakeup-list 29f84ab81; here every writer of `P.wakeup` marks the sidecar: `mechanical_fb.cc`, the
thermal feedback kernel, the ghost write-back, the supernova causal wake, merge/split, sinks.)

`TEST_WAKEUP_REQUEST_AUDIT` (test only): before each scan, counts flagged gas cells off their sync point that the scan
would skip: not in the sidecar (`missing`), or a request new since the last scan (flag value changed) while no rank raised
the flag (`gate_shut`). A flag stays until its cell is next active, so flags seen by an earlier scan are not counted as
shut out; steps after an index remap (domain decomposition) skip the gate count. Prints the running totals as
`WAKEUP_REQUEST_AUDIT pending <n> missing <n> gate_shut <n>`. (The first version counted every flag behind a shut gate:
750 of 798, all of them flags already scanned.)

Problem: the Stella cloud flags (`scripts/stella/Config_stella_cloud.sh`) plus `TEST_SINGLE_STARS_SN_AT_FORCED_TIMES`;
16^3 lattice of 1 Msun cells at n_H = 10 cm^-3, 100 K, solar composition (the `single_stars_metals` lattice), one 20 Msun
star 1 Myr old at the centre (ID 1, winds) forced to explode at 1e-3 time units; gas steps at most 2.5e-4, the star 10 yr;
2e-3 time units, 2 ranks.

Checks (bars written before the run): the supernova of star 1 is delivered; cells are woken and flagged off their sync
point; none of them is missing from the sidecar and none is left behind a shut gate.
