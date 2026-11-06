# photoion_per_direction

Per-direction photoionisation around single stars (`GALSF_FB_PHOTOIONIZATION_PER_DIRECTION`). Design note:
design_photoionisation_pixelised.md; reference: stella_dev galaxy_sf/resolvedism_photoion.cc.

Problem: a 32^3 lattice of 1 Msun cells at n_H = 10 cm^-3, no gravity, no cooling (the gas stays static), with single
stars from the stellar-evolution tables.

Checks:
- `test_ionising_stars_in_the_photoionisation_bin`: all ionising stars share one cadence. With MaxSizeTimestep 20 kyr and
  PhotoionMaxTimestep 1 kyr, a 35 Msun star 1 Myr old steps at most 1 kyr, while the gas keeps its coarse step. A 35 Msun
  star 1000 yr old steps onto its ionising onset (15 kyr) and then keeps to the same bin. Without the change both stars
  step with the gas.
- `test_stromgren_sphere`: a 35 Msun star 1 Myr old near the centre (Q = 7.7e48 /s, Stromgren radius 12.7 pc, 9 cells).
  Per pixel the front is where the cells' recombination rates (case B, 2.6e-13 cm^3/s, at each cell's density), summed
  outward, use up Q/12. The smallest and largest of the 12 fronts (log columns 10, 11) lie within one cell of the
  Stromgren radius. Without the change there is no photoionisation log.
  With the stamp (cells inside the fronts ionised until the next solve, `PhotoionizedUntil`): the recombination rate of
  the ionised cells is Q to 2% (measured 0.9994-0.99999; log column 7), their volume radius lies within one cell of the
  Stromgren radius, and static gas keeps the same cells at every solve. Without the stamp column 7 is 0.
- `test_ranks_and_order`: 1, 2 and 4 ranks and a permuted particle order give the same fronts, search passes and
  ionised cells (the histogram sums are integers).
- `test_overlapping_sources_share`: two identical stars 0.5 Stromgren radii apart. Every source first claims its own
  sphere; then, in rounds, a cell claimed by several sources bills each its share q_sc / A_c (integer weights from the
  photons each delivers to the cell), so fronts advance until the bills are paid once. The pair ionises 2.0 x the cells of
  one star (measured 6068 / 3035 = 1.999; 1.385 without sharing, where both stars bill the shared cells in full) and the
  union recombines at the summed photon rate (0.999); 1, 2 and 4 ranks give the same cells and rounds.
- `test_bright_and_faint_source`: a 35 and a 20 Msun star (Q ratio 7) 0.9 pc apart: the union recombines at the summed
  rate and its cells scale with it (1.146 vs 1.144; without sharing 1.002).
- `test_sgchem_reads_the_ionised_cells` (second build, SGChem network 5 as the cooling): inside the fronts the cells have
  ionised abundances (H+ 0.9998, molecules destroyed) and at least the energy of 1e4 K ionised gas, with no network solve
  (as stella_dev). Outside, the photons leave the cells untouched. The heating to 1e4 K is booked in the ledger row
  E_photoionisation_heating_erg, within 5% of the thermal energy the ionised cells gained; the chemistry rows hold the
  network only. Without the change the ionised cells stay neutral and cold.
- `test_quiet_steps`: a 35 Msun star 10 kyr old, onset at 15.3 kyr. Log column 12 counts the module's reductions and
  neighbour-loop calls since the previous row. The first solve comes within one photoionisation step of the onset and
  costs 7 + 5 per search pass plus the one source count at start-up. Before the change it cost 16: one reduction on
  each of the 5 photoionisation steps without a source. Ionising onsets ride the existing star-formation reduction.
- `test_compact_exchange`: the histograms (12 pixels x 64 bins, 12 kB per source) do not travel in the neighbour-loop
  reply records. The rank holding the cells fills a pool by global source index; the non-empty bins go to the
  source's rank in one exchange per pass. Four separate stars on 4 ranks with a 0.02 MB communication chunk trigger no
  'cap ineffective' round (before the change: 2 records of 12.6 kB from one source already exceeded a round), and
  give the cells and fronts of 1 rank. All other photoion logs are bitwise unchanged by the change.

Performance (`run_performance.sh`, not part of pytest; Orion, 8 ranks x 8 threads, one source, 10 steps with every cell
active; log column 13 is the wall time of each solve on the slowest rank):

| cells | solve [ms] | of which runner calls and exchanges | of which source scan | whole run (cpu.txt total), flag on / off |
|---|---|---|---|---|
| 1e6 | 3.8 | 1.9 | 1.9 (with the first reduction) | 20.7 / 19.6 s |
| 8e6 | 18.8 | 1.5 (+2.4 first reduction) | 14.4 | 105.0 / 102.5 s |

The work of the solve itself does not grow with the cell count. What grows is the scan of the active particles for
sources, one particle read per active particle (here all 1e6 per rank), about 0.3% of the run at 8e6. Memory: +8 B per gas
cell in the cell struct (368 vs 360 B), +8 B per local gas cell in the claim buffer from the first solve, about 200 kB
per rank of histogram pool; nothing per particle; 4 B per single star (birth density).
