# single_stars_metal_origin

`GALSF_SINGLE_STARS_METAL_ORIGIN_TRACERS`: four extra passive metal slots, after the 16 extra elements of
`GALSF_SINGLE_STARS_27_ELEMENT_YIELDS`, carry the metal mass fraction each single-star source brought into a cell: SN, AGB
deaths, winds, Type Ia (Metallicity columns 27-30 in this build, labelled -4, -5, -6, -7 in the snapshot header attribute
`Metals_Atomic_Number_Or_Key`). The feedback loop deposits them like every other slot, so they are carried, mixed,
merged/split, diffused and written by the existing metal machinery.

Build: the `single_stars_metals` build (SGChem network 5, single stars from the tables, Type Ia with the test-only rate
multiplier) with `GALSF_SINGLE_STARS_27_ELEMENT_YIELDS` and the switch (31 metal slots). Problem: the 16^3 lattice of
1 Msun cells at n_H = 10 cm^-3, 100 K, the tables' solar composition, and the four stars of `single_stars_metals`, each
in its own octant: a 20 Msun core collapse, a 3 Msun AGB death (its white dwarf then explodes as a Type Ia), a 1 Gyr
white dwarf (Type Ia at once), a 35 Msun star 1 Myr old (winds). 5e-3 time units (4.9 kyr), 2 ranks.

Checks (bars written before the first run):
1. `test_metal_origin_by_source`: 31 slots, origin labels -4..-7; the gas metal mass in each logged source slot equals the
   metals of that source's events in `single_star_events.txt` (1e-10; measured 2e-16 to 7e-16), the wind slot is positive
   (5.0e-3 Msun); total metals minus the four sources stays the initial gas metal mass (1e-12; measured 0); each star's
   octant carries only its own sources (SN; AGB and Type Ia; Type Ia; winds), every other origin slot is exactly 0.
2. `test_metal_origin_snapshot_restart`: a snapshot restart (start flag 2) from the last snapshot, 1e-5 time units on: every
   cell outside the wind star's octant keeps the origin slots it was written with (1e-14), so the slots are read back.

Without the switch the build has 27 slots and check 1 fails at its first line.

Reference implementation: stella_dev `CellP.MetalMassFrom[4]` (snapshot block `MetalOrigin`), same channels and the same
mass-fraction blend at each deposit.
