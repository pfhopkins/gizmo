# Hierarchical Triple

Three Type 5 sinks: an **unequal-mass** (0.8 + 0.2 M<sub>⊙</sub>) eccentric 4 AU inner binary
orbited by a 1 M<sub>⊙</sub> tertiary at 100 AU, mutually inclined and randomly oriented.
Reports energy and momentum conservation across a **deep timebin hierarchy**.

## Why this exists alongside `test/binary`

A bound *pair* splits by at most ~1 timebin — the symmetric 2-body criterion binds both members
together — so `test/binary` cannot produce a deep hierarchy no matter how eccentric it is. Here
the period ratio is 88, putting the tertiary **~6 bins coarser** than the inner stars, so the fine
stars evaluate the tertiary mid-step, up to 2⁶ of their own steps from its last sync. That is the
regime the Hermite source prediction in `gravity/forcetree.cc` exists for, and the configuration
of the production momentum violation this work started from.

## Setup

| | |
|---|---|
| inner binary | **0.8 + 0.2** M<sub>⊙</sub>, a = 4 AU, **e = 0.5**, P = 8.18e-6 |
| tertiary | 1.0 M<sub>⊙</sub>, a = 100 AU, **e = 0.3**, P = 7.23e-4 |
| mutual inclination | 25° |
| orientation | one uniformly random rigid rotation, seed 20250825 |
| period ratio | 88 → ~6.5 bins (P depends only on *a*, so eccentricity does not change it) |
| run length | TimeMax 3.616e-2 = 50 outer orbits (~4400 inner) |
| eta | 1.25e-3 |
| softening | 1e-7 pc, 97× below the inner pericentre — Newtonian throughout |

### Two splits, deliberately

The **6-bin inner↔outer gap** is the deep one, and the reason this test exists.

The **unequal inner masses** add a second, 1-bin split *inside* the pair: `dt_tidal` is set by the
companion mass, so the components differ by √(m₁/m₂) = 2 — the asymmetry the original leak was
traced to. This matters because the inner binary carries **86% of the total energy** (−27.7 of
−32.2); at equal masses it shares a bin by symmetry and is structurally blind to the source
prediction. The cost is that COM drift is no longer unambiguously the inner↔outer channel.

### Eccentric, inclined, randomly oriented

So that no part of the configuration is a special case. Circular coplanar orbits are measure-zero:
the separation never varies, so the source prediction is probed at a single staleness, and with all
motion in one plane the *z* components of the momentum error are structurally unlike *x* and *y*.
Eccentricity also sweeps dt through the orbit (5.2×, 2.4 bins at e_in = 0.5), exercising bin
*transitions* that a static hierarchy never reaches.

### Staying non-chaotic

- **Hierarchy** — a_out/a_in = 25 against a Mardling & Aarseth (2001) critical ratio of 4.22, a
  **5.9× margin**. Asserted in the IC generator, not assumed.
- **Mutual inclination below the Kozai–Lidov critical angle**, cos²*i* = 3/5 → 39.2°. Inside that
  window e_in and *i* librate on a t ~ P_out²/P_in timescale, moving the inner pericentre — and
  with it the bin structure — underneath the diagnostics. At 25° there is only regular precession.

Verified by round-tripping the elements out of the IC: a, e and *i* recover to 6 digits,
Σmv = 1.6e-16, and the inner orbit normal has all three components nonzero.

## What is asserted

**One thing:** |E/E₀ − 1| below `MAX_DE_OVER_E` = 1.7e-4, as a per-outer-orbit median envelope.
That is 3× the 5.66e-5 measured on this configuration, and it bounds gross breakage only.

COM drift, growth exponents and the inner semi-major axis are **printed and saved** to
`summary_<variant>.npz`, not judged.

> **This test does not currently guard the source-prediction fix — and nothing in the committed
> suite does.** What discriminated the defect here was the secular growth exponent on the COM
> drift — t^+0.95 defective vs t^+0.55 fixed, across a 0.85 threshold — which the defect *passed*
> on magnitude while failing on trend. That check was calibrated on a configuration this IC no
> longer produces and has been removed pending recalibration. It cannot be delegated:
> `test/binary`'s pair shares a timebin so the prediction never fires there, and `test/fewbody`'s
> 10% ceiling passes fixed and unfixed code alike. The evidence for the fix lives off-suite — the
> fewbody per-problem medians (2.09×/2.44×), the production seed4 A/B (22× leak suppression), and
> the M2e3 survey. Restoring an in-suite guard means recalibrating the drift-exponent check on
> this configuration.

An integration-order sweep was also removed. Its KDK control measured leapfrog converging at
dt^3.4 — above leapfrog's 2nd-order ceiling, therefore impossible — so the metric was not
measuring integration order. Two causes were identified: |Δx| was differenced in the box frame, so
COM drift entered as a bulk translation comparable to the signal; and over 5 outer orbits the inner
binary turns 442 times, so its phase error saturates |Δx| at ~a_in and flattens the slope.
Re-adding one means fixing both the quantity (difference in the COM frame; track the tertiary and
the inner pair separately, each with ~5 periods of accumulation) and the estimator
(self-convergence over consecutive pairs needs no converged reference).

Measured from the `IO_HERMITE_SYNC` datasets. The plain `Coordinates`/`Velocities` are a mixed
state — positions drifted to the output time, velocities at the last kick — so any quantity
combining them is evaluated on a configuration the system never occupied.

Per-orbit statistics use the **median**, not the minimum: with synced datasets the within-orbit
spread is the inner binary's phase rather than an artifact, so a minimum tracks oscillation dips
and can report a falling trend where the error is growing.

## Files

- `make_triple_ics.py` — IC generation; run standalone to inspect the configuration
- `triple.params` — run parameters
- `Config.sh` — `SINGLE_STAR_STARFORGE_DEFAULTS` + `IO_HERMITE_SYNC`
- `test_triple.py` — diagnostics and the assertion
