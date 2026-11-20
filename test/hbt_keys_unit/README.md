# hbt_keys_unit

Unit test for the SUBFIND-HBT particle key (`structure/hbt/hbt_key.h`, `hbt_key_mpi.cc`), test T1 of
`reports/subfind_hbt_study.md`. It emulates GIZMO's split and spawn rules and checks: keys unique; the top-bit rule
recovers every parent; lineage lookup passes labels through a synthetic 10% split; GADGET-4 64-bit packing; the
distributed duplicate count on 1 and 4 ranks. Two negative controls (generation rollover after 31 splits, and a
spawn from a generation-0 star) must produce duplicates that the check detects.

Run in a batch job: `module load gcc openmpi; python3 -m pytest -s test/hbt_keys_unit`.
