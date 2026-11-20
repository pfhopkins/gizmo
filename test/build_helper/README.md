# build_helper

The test build helper (`build_gizmo_for_test` in test/harness/gizmo/test.py) and the tracked test configurations; no GIZMO build.

- `test_missing_test_config_stops_the_build`: with no `test/<name>/Config.sh` the helper stops before `make` (it used to build
  whatever `src/Config.sh` held).
- `test_every_building_test_has_a_tracked_config`: every test directory that builds under its own name has its Config.sh in git
  (`.gitignore` ignores Config.sh, so a test's file is tracked only with `git add -f`).
- `test_no_adaptive_star_softening_in_single_star_tests`: no single-star test or Stella script config gives stars (bit 16 of
  `ADAPTIVE_GRAVSOFT_FORALL`) adaptive softening.
