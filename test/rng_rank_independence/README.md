# rng_rank_independence

Keyed random draws (`declarations/gpu_rng.h`): a stochastic outcome depends on the particle and the event, not on which
rank or thread evaluates it.

One small IC (4096 gas cells, 2000 pre-existing stars, 64 sinks in a 1 kpc ball; written by the test) is run with
GALSF + sink accretion (no self-gravity) on 1, 2 and 4 ranks and on 1 or 2 OpenMP threads, to two snapshots:

- initial draws (stellar ages from `InitStellarAge`, sink spins) agree bit for bit across all runs;
- runtime draws agree exactly: the sets of gas cells converted to stars are identical at both snapshots. Self-gravity
  is off because the tree force depends on the decomposition: with it, states drift apart and a cell's single-precision
  position (part of its key) flips, which made the overlap a measure of trajectory divergence, not of the draws.
  With one random stream per rank the overlap is chance, ~2%;
- draws from different call sites are uncorrelated (sink age vs spin, one sink's spin vs the next's), and the initial
  stellar ages are uniform on [-2 InitStellarAge, 0];
- duplicate IDs (a snapshot reused as an IC) and missing IDs (`ASSIGN_NEW_IDS`) do not collapse particles onto one draw.

Written on starforge_dev when its per-rank random stream was replaced by this branch's keyed draws; there the old stream
failed the first two checks at every rank count other than 1.
