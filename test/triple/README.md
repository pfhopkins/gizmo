# Hierarchical Triple

Three Type 5 sinks: an equal-mass 4 AU inner binary orbited by a 1 M<sub>⊙</sub> tertiary at
100 AU. Both orbits circular. Measures energy and momentum conservation across a **deep timebin
hierarchy**.

## Why this exists alongside `test/binary`

A bound *pair* splits by at most ~1 timebin — the symmetric 2-body criterion binds both members
together — so `test/binary` cannot produce a deep hierarchy no matter how eccentric it is. Here
the period ratio is 88, which puts the tertiary **6 bins coarser** than the inner stars at 99.9%
of sync points. The fine stars therefore evaluate the tertiary mid-step, up to 2⁶ of their own
steps from its last sync.

That is the regime the Hermite source prediction in `gravity/forcetree.cc` exists for, and it is
the configuration of the production momentum violation this work started from (a hardening
triple). The equal-mass inner pair shares a bin by symmetry and cannot leak against itself, so
any secular COM drift is unambiguously the **inner↔outer** channel.

## Setup

| | |
|---|---|
| inner binary | 0.5 + 0.5 M<sub>⊙</sub>, a = 4 AU, circular, P = 8.18e-6 |
| tertiary | 1.0 M<sub>⊙</sub>, a = 100 AU, circular, P = 7.23e-4 |
| period ratio | 88 → ~6.5 bins |
| run length | TimeMax 3.616e-2 = 50 outer orbits (~4400 inner) |
| eta | 1.25e-3 |
| softening | 1e-7 pc, 194× below the inner separation — Newtonian throughout |

Both orbits circular is deliberate: no pericentre refinement, so **the timestep is static for the
whole run** (measured: one value for 100% of 4101 sync points). A change in the diagnostics
cannot be blamed on the bin structure shifting under it.

## What is asserted

1. **Inner orbit intact** — a/a₀ within 5%. If the configuration is lost, nothing below describes
   what the test was calibrated for.
2. **Energy** |E/E₀ − 1| below `MAX_DE_OVER_E`.
3. **COM drift** |v_com| below `MAX_COM_DRIFT`. The ICs are built in the exact COM frame, so any
   drift is entirely integration error.
4. **Drift growth exponent** below `SECULAR_EXPONENT` = 0.85 — a **hard failure**. This is the
   assertion that actually guards the fix: with the source prediction the drift is a bounded
   band, without it it is secular. Magnitudes alone do not separate the two at this length.

Energy drifting secularly is recorded as an expected failure, not a regression — that is the
known 4th-order block-step residual.

Measured from the `IO_HERMITE_SYNC` datasets. The plain `Coordinates`/`Velocities` are a mixed
state (positions drifted to the output time, velocities at the last kick), so any quantity
combining them is evaluated on a configuration the system never occupied.

Per-orbit statistics use the **median**, not the minimum. The minimum was inherited from
`test/binary` and is wrong here: with synced datasets the within-orbit spread is the inner
binary's phase rather than an artifact, so taking the minimum tracks oscillation dips instead of
the error. On one run it reported t⁻⁰·⁷⁷ where the median gave t⁺⁰·⁸⁶ — the error was growing
linearly and the statistic said it was shrinking.

## The convergence sweep — read before trusting its exponent

The test also sweeps `ErrTolIntAccuracy` and fits the energy error's order, to catch a change
that keeps magnitudes in tolerance while degrading the scheme's order. **This sweep has not yet
produced a trustworthy measurement.** Three attempts gave η^1.54, η^2.11 and η^3.40, all of which
passed the assertion and none of which meant anything. The reasons are worth knowing, because
each is a trap the next person will hit:

- **GIZMO clamps eta at 0.01** under `GRAVITY_ACCURATE_FEWBODY_INTEGRATION`
  (`core/begrun.cc:2747`), silently, and `params-usedvalues` shows the pre-clamp value. Two sweep
  points above it produced *bit-identical* runs and a confident-looking fit.
- **Steps must be factors of 4, never 2.** dt ∝ √eta, so ×4 is exactly one timebin and the
  hierarchy translates rigidly. ×2 is half a bin.
- **The error floor.** At 20 orbits the finer points measured *higher* than coarser ones —
  sampling round-off and phase rather than truncation. The sweep now runs the full 50 orbits so
  every point clears it.
- **The coarse end may not be asymptotic.** At eta=0.01 the error is 2e-3, and the fitted order
  across the usable window is ~η^3.9 — far steeper than the η² a 4th-order scheme gives. With
  the clamp above and the floor below there is no third whole-bin point to disambiguate.

Two guards now assert on the sweep *definition* (no point above the clamp; steps of exactly 4),
because both failures were silent. Read the left panel of the convergence plot — the per-eta
error evolution — rather than the fitted exponent.

## Files

- `make_triple_ics.py` — IC generation; run standalone to inspect the configuration
- `triple.params` — run parameters
- `Config.sh` — `SINGLE_STAR_STARFORGE_DEFAULTS` + `IO_HERMITE_SYNC`
- `test_triple.py` — diagnostics and assertions
