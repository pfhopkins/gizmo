# hernquist_convergence

Calibration and convergence gate for `TIDAL_TIMESTEP_CRITERION`, on an equilibrium Hernquist
sphere.

## Why Hernquist, and why calibrate against it

The tidal criterion's published calibration is against an **optimally-softened Plummer sphere**,
where it is documented as reaching the same energy error as the Power 2003 acceleration criterion
over ~100 crossing times. A Hernquist sphere is the opposite end of that range: its cusp gives a
steep central density gradient, so the tidal tensor varies sharply where the acceleration
criterion is already well behaved, and the paper treats it as the **worst case** for the
criterion.

Calibrating here is deliberate. Tuning the coefficient so the tidal criterion matches the
acceleration criterion on its worst case guarantees it does not degrade accuracy on easier,
more general setups — at the cost of taking shorter steps, and therefore more of them, than a
Plummer-calibrated coefficient would. That trade is the intended one: the criterion exists to
make timestepping robust in configurations the acceleration criterion handles poorly, and it is
not worth having if it silently costs accuracy elsewhere. The substep count is reported alongside
the error so the price stays visible.

## What is measured

**Parity.** The tidal criterion must land within a factor of 2 of the acceleration criterion on the
accumulated `|dE/E0|`, at the **default** tolerance `ErrTolIntAccuracy = 0.01`. At the historical
coefficient of 0.5 it was 142x worse in drift rate there (1.81e-03 vs 1.27e-05) and 215x in
accumulated error (2.78e-02 vs 1.29e-04).

Parity is required only at the default, and cannot be required across the range: the two criteria
respond to tolerance differently, so any single coefficient matches at one point and diverges
either side of it. With the calibrated 0.10 the measured ratios are 3.03x at `eta=0.04`, 0.59x at
the default, and 0.13x at `eta=0.0025` — the curves cross at the default, and the tight end, where
the tidal criterion is the more accurate of the two, is the safe direction to err in.

**Convergence.** The tidal drift must fall when `ErrTolIntAccuracy` is tightened. This separates an
under-resolved criterion from a broken one, and it is the check that survives someone re-tuning
the coefficient: a criterion selecting wrong timesteps can be made to pass parity at one tolerance
while still not converging.

## What the calibration rests on

Two scans, both on this problem:

| scan | range | drift response |
|---|---|---|
| `ErrTolIntAccuracy` | 16x | falls 42x, effective order ~1.36 steepening toward 2 |
| `ErrTolForceAcc` | 250x | flat to 0.7%, at 4.3x the force work |

Insensitive to force accuracy, convergent in timestep. So the error is a timestep-**selection**
defect — not a force error, and not noise in the tidal tensor, which was the first hypothesis and
was wrong. Shortening `dt_tidal` via its coefficient is therefore the correct lever.

Both criteria step identically — `dt ~ sqrt(eta)` for each, verified to 1% from substep counts —
so the difference is entirely in how error responds, not in the stepping. Baseline goes as
`eta^1.03`, textbook for a 2nd-order integrator. The tidal criterion does not follow a power law:
its secular drift is 21.7 sigma at `eta=0.04` and 0.0 sigma at `eta=0.01`, so it switches off
rather than decaying, and below the default there is no drift left to converge. The calibration
therefore works by pushing `dt` past that threshold, not by scaling down a smooth error term.

The mechanism behind the threshold is **not established**. In particular it is not the
companion-mass pair splitting described under `SINK_TIMESTEP_SAFETY_FACTOR` in `core/timestep.cc`:
that belongs to the sink path and is not compiled here. These runs are Type-1 only, with no sinks,
no binaries, and `SINGLE_STAR_TIMESTEPPING` off, so `dt_tidal` reduces to the bare tidal-tensor
expression.

## The coefficient, and where it must not apply

`TIDAL_TIMESTEP_PREFAC` in `core/timestep.cc`. It is **not** applied when
`SINGLE_STAR_TIMESTEPPING` is active.

There, `dt_tidal` must stay long enough that the symmetric 2-body criterion binds for both members
of a bound pair. `dt_tidal` keys on the **companion** mass, so it differs between unequal members
by `sqrt(m1/m2)`; shortening it would undercut the 2-body criterion for the lighter member and
split the pair across timebins. That asymmetry is what a production momentum-conservation
violation was traced to, and what the `SINK_TIMESTEP_SAFETY_FACTOR` handling above the criterion
exists to prevent. Binary timestepping is calibrated separately and must not be overridden here.

## Run length

Not arbitrary. The tidal drift clears 5 sigma over the sampling noise floor by `t~2`, so
`TimeMax=15` is ~7x more than needed to resolve it while being 8x cheaper than `test/hernquist`'s
118. `TimeBetSnapshot=0.25` gives 61 energy samples: the fitted rate is limited by the ~3e-4 noise
floor rather than by run length, and that floor falls as `sqrt(N)`.

Distinguishing the *conserving* variants from each other would need `t~800-1000` — they are all at
the floor and plausibly all zero. This test does not attempt that; it only requires baseline to
sit at the floor, which is what makes it a valid reference.

## Relationship to `test/hernquist`

`test/hernquist` exercises eight gravity variants for structure and momentum, and gates on the
half-mass radius. It has **no energy assertion at all**, which is why a ~70x energy deficit in its
`ags` and `tidal` variants went unnoticed while it reported `7 passed`. This test covers the
energy axis that one leaves open.
