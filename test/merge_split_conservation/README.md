# merge_split_conservation

A unit test of `mesh/merge_split.cc`, not a simulation. `merge_split_driver.cc` is compiled with the test's build flags and
linked in place of `core/main.o` against the objects of that build, then run on hand-built cells:

- `split_particle_i`: mass, momentum, every metal species, B and (where present) the sink-feedback reservoir are
  conserved to 1e-13; the daughters sit symmetrically about the parent, each within 0.4 of the pre-split nearest-neighbour
  distance; no `CellP` write past `MaxPartGas` (a canary region follows the gas block).
- `merge_particles_ij`: the same totals; the merged hydrogen fraction is mass-weighted where the layout tracks it.
- star-star merges and `rearrange_particle_sequence` on zero-mass stars past `MaxPartGas`: `CellP` untouched.

The driver also carries checks for the SN ejecta spawn and for `TURB_DIFF_METALS_EXACT_CONSERVATION`'s pending
exchange; neither exists on this branch and both compile out.

Variants (each its own test directory, sharing this driver): `merge_split_conservation_starforge`
(SINGLE_STAR_STARFORGE_DEFAULTS with SINK_THERMALFEEDBACK; needs the cooling tables) and
`merge_split_conservation_nometals_sinkfb` (SINK_THERMALFEEDBACK without METALS: the reservoir must still be summed on
merge when the metal block is compiled out; on starforge_dev that sum lived inside `#ifdef METALS`).

The parameterfile is read for units and allocation sizes only; no IC is loaded.
