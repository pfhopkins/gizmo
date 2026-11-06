# wakeup_falling_sedov

A point blast (E = 1 in one cell) in cold uniform gas on a 32^3 lattice, in a periodic box that falls in a uniform field (g = -0.5 in y).
The blast wakes cold cells sitting on long timesteps. Adiabatic. From stella_dev test/wakeup_falling_sedov, on the snapshots instead of its
energy-budget log.

| test | checks |
|---|---|
| `test_default_is_highest_active_bin` | `WakeupTargetBin` defaults to 2 (`WAKEUP_TARGET_BIN_PARAMETER`) |
| `test_centre_of_mass` | v_cm,y = -0.5 t to 1e-3 in modes 2 and 0: a woken cell keeps its field kick |
