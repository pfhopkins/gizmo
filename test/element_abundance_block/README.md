# element_abundance_block

`GALSF_SINGLE_STARS_ELEMENT_ABUNDANCE_BLOCK` (with `GALSF_SINGLE_STARS_27_ELEMENT_YIELDS`): the `ElementAbundance` block holds
the 27 element mass fractions H..Zn of a gas cell or star (H = 1 minus the rest). A snapshot restart (start flag 2) reads it
into the metal slots (FIRE slots 1-10 for He, C, N, O, Ne, Mg, Si, S, Ca, Fe and the 16 extra slots for the other elements; total
Z = the sum over C..Zn); the block follows `Metallicity` in the read order and wins over it. Snapshots write it back from the
slots. The ICs of stella_dev carry this block instead of `Metallicity`.

Problem: the gas lattice and one star of `single_stars_metals` with zero `Metallicity` slots and a 27-element block, start flag 2,
2 ranks, one short step.

Check: the metal slots of gas and star equal the block through the slot map, and the block written back equals the one read.
Inputs: `SUNDIALS_DIR`, the chemistry and stellar tables.
