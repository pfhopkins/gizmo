# single_stars_forced_events

Test hooks of the single-star supernova timing. `TEST_SINGLE_STARS_SN_AT_FORCED_TIMES`: star ID k of the first `TestNumStars` (first generation)
explodes at `TestSNTime0 + (k-1) TestSNSpacing` [code time], whatever its table lifetime. `TEST_SINGLE_STARS_NO_GAS_IN_REACH_UNTIL_TIME`: before
`TestFBNoGasUntilTime` the feedback of an event finds no gas, so the event stays pending and fires at a later step (the retry of an
event without neighbour weight, `no_deposit_count`).

Problem: 24^3 lattice of 1 Msun cells, n_H = 10, 100 K, two 20 Msun stars 1 Myr old (table death ~10 Myr), IDs 1 and 2; 8 kyr, 2 ranks.

Checks: each star one supernova row, at its forced time to within one step (1.5e-4); with no gas until 0.004 the first star (forced 0.002)
fires at 0.004, after at least 10 steps without deposit, the second is not delayed; mass conserved (1e-12), mass residual 0.

Reference: the definitions of the parameters (stella_dev `resolvedism_fb.cc`).
