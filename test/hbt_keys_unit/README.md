# hbt_keys_unit

Unit test for the SUBFIND-HBT particle key (`structure/hbt/hbt_key.h`, `hbt_key_mpi.cc`), test T1 of
`reports/subfind_hbt_study.md`. It emulates GIZMO's split and spawn rules and checks: keys unique; the top-bit rule
recovers every parent; lineage lookup passes labels through a synthetic 10% split; GADGET-4 64-bit packing; the
distributed duplicate count on 1 and 4 ranks. Two negative controls (generation rollover after 31 splits, and a
spawn from a star born with generation 0, the rule before STL-148) must produce duplicates that the check detects.
With the STL-148 rule (a star takes the parent's incremented generation) spawns from stars keep keys unique, and
`test_star_spawn_generation_rule` checks that on gku the star spawn (galaxy_sf/sfr_eff.cc) and the splits (mesh/merge_split.cc) all go through galaxy_sf/single_star_identity.h, which gives the child the parent's incremented generation (GKS-145). Sources are under src/.

Run in a batch job: `module load gcc openmpi; python3 -m pytest -s test/hbt_keys_unit`.
