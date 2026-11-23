# hbt_catalogue_unit

Unit test for the group-catalogue offsets of the SUBFIND-HBT catalogue writer (`structure/hbt/hbt_catalogue_core.h`,
`hbt_catalogue_mpi.cc`), test T-C1 of `reports/subfind_hbt_study.md` section 12.1. Random group tables distributed
over 1 and 4 ranks (some ranks empty): the per-type offsets into the per-type member lists must equal exclusive
prefix sums of GroupLenType in group order, and the per-type totals the list lengths.

Run in a batch job: `module load gcc openmpi; python3 -m pytest -s test/hbt_catalogue_unit`.
