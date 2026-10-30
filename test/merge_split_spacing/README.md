# merge_split_spacing

Split placement in a live run. A static 10^3 lattice of unit-mass cells with 8 close pairs of 3x over-massive cells,
each pair inside a lattice void (`make_merge_split_spacing_ics.py`); the header's `Maximum_Mass_For_Cell_Split`
makes only the heavy cells split, `Minimum_Mass_For_Cell_Merge` lets nothing merge. One rank: a neighbour owned by another
rank is not seen by the split spacing.

From the first snapshot after every heavy cell has split exactly once: each daughter is offset at most 0.4 of its
parent's distance to its nearest neighbour (its partner), so no two daughters of neighbouring parents come closer than 0.2
of that distance. Daughters are symmetric about the parent, so the parent position and offset are recovered from the
daughters' midpoint and half-separation whatever the cells did before or after.
