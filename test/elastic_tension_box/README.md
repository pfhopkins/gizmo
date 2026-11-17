# elastic_tension_box

The classic tensile-instability problem: a periodic 2D box of elastic material (the ring material) under uniform isotropic
tension (areal density 5% below the reference, about 2.6% linear stretch in each direction), at rest, for about 400 cell sound-crossing times. The exact solution is that nothing moves; particle
methods clump into pairs unless the negative pressure is corrected. Two problems share the directory:

- `elastic_tension_box_fresh.params`: nothing yields, so the tensile correction acts everywhere;
- `elastic_tension_box_preyielded.params`: every element starts with `RecentYieldMemory` 0.88 (as if it had just flowed
  plastically), so the correction starts switched off and must come back as that memory decays.

Checks: the smallest nearest-neighbour distance stays above 0.9 of the lattice spacing (measured 0.98; with the
correction off it falls below 0.2); the mean pressure holds; in the pre-yielded box the memory decays by more than
1000 (measured 1e6).

    pytest test/elastic_tension_box           # ~2 minutes for both; make_ics.py writes both ICs
