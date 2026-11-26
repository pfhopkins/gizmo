# Stella audit fix round 1 (2026-11-26)

Tests of the round-1 fixes from the Stella module audit (reports `stella_module_audit_2026-11-26/06_` and `09_`). Every
solver change is behind a flag, default off; each test runs with the flag off (strict expected failure: the defect) and on
(the bar must hold). Bars and expected values are in each file's docstring, fixed before the first run.

| file | row | flag |
|---|---|---|
| test_ejecta_birth_composition.py | STL-403 | GALSF_RESOLVEDISM_EJECTA_CARRY_BIRTH_COMPOSITION |
| test_tables_phase_matched.py (+ phase_driver.cc) | STL-391, 392, 411 | GALSF_RESOLVEDISM_TABLES_PHASE_MATCHED_LOOKUP |
| test_photoion_bill.py (+ bill_driver.cc) | STL-409 | GALSF_RESOLVEDISM_PHOTOION_BILL_HE_ELECTRONS (network 17 only) |
| test_merge_mhd_ledger.py (case in ../merge_split_conservation/merge_split_driver.cc) | STL-428 | none (energy-trace ledger) |
| test_snapshot_last_wind_age.py | STL-36, 431a | GALSF_RESOLVEDISM_SNAPSHOT_KEEPS_LAST_WIND_AGE |
| test_guards.py | STL-417, 423, 427, 432, 446 | none (guards) |
| test_bookkeeping.py | STL-393 | none (element ledger booking) |
| test_cr_full_share.py | STL-356 | GALSF_RESOLVEDISM_CR_INJECT_FULL_SHARE |

STL-431b (KETJU_PN_REMNANT_TAG: a snapshot restart with a Stella remnant stops, since snapshots do not restore its tag) is
compiled in the KETJU builds only and has no run test here. The KETJU handle_mergers endrun (STL-427) has no run test either.

Binaries: `STELLA_AUDIT_BIN_<KIND>` (e.g. `STELLA_AUDIT_BIN_ROUND1_ON`) names a prebuilt GIZMO with its drivers next to it;
otherwise each test builds its configuration through `build_gizmo_for_test`. Tables: the in-tree default paths
(`scripts/get_stella_tables.sh --into-tree`).
