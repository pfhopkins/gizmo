# Plummer Cluster with a Realistic Binary Population

A Plummer sphere of Type 5 sinks whose binary population is drawn from the observed
distributions, rather than being identical equal-mass circular pairs as in
[`test/plummer_binaries`](../plummer_binaries). Measures energy and momentum conservation
under a heterogeneous population.

## Why this exists alongside `plummer_binaries`

`plummer_binaries` puts every star in a 1000 AU equal-mass circular binary: one point in
parameter space, cleanly controlled, and every star on effectively the same timebin. That is
the right design for isolating tree-vs-direct gravity, which is what it tests.

It cannot exercise the thing the Hermite sink integrator is most sensitive to. The source
prediction in `gravity/forcetree.cc` and the shared timestep normalization in `core/timestep.cc`
both act on **timebin structure**, and timebin structure is what a heterogeneous population
produces: stars of different mass on different bins, binaries spanning ~8 bins in orbital
period, and — impossible in any equal-mass test — **single stars sharing the cluster with
binaries**, so mixed-eligibility neighbours are routine rather than absent.

## Distributions drawn

| Quantity | Distribution | Source |
|---|---|---|
| Primary mass | Broken power law, dN/dm ∝ m<sup>−1.3</sup> (0.08–0.5 M<sub>⊙</sub>), m<sup>−2.3</sup> (0.5–30 M<sub>⊙</sub>) | Kroupa (2001) |
| Binary fraction | f(M₁) rising with primary mass, log-interpolated: 0.22 at 0.1 M<sub>⊙</sub> → 0.80 above 16 M<sub>⊙</sub> | Duchêne & Kraus (2013), Table 1 |
| Mass ratio | q = m₂/m₁ uniform on [0.1, 1] | Duchêne & Kraus (2013), γ ≈ 0 for solar-type |
| Orbital period | log₁₀(P/days) ~ N(5.03, 2.28) | Raghavan et al. (2010) |
| Eccentricity | Thermal, f(e) = 2e, for P > 10 d; circular below | Jeans (1919); Duquennoy & Mayor (1991) for the circularisation cutoff |
| Cluster structure | Plummer sphere, isotropic velocities by inverse-CDF | Plummer (1911) |

The IMF is sampled by **exact inverse-CDF**, not rejection, so the star count is deterministic
for a given seed. Positions are stratified (`(i + U)/N` rather than N independent uniforms),
which is inherited from `test/plummer` and reduces shot noise in the radial profile.

### References

- Kroupa, P. 2001, *MNRAS* **322**, 231 — "On the variation of the initial mass function"
- Duchêne, G. & Kraus, A. 2013, *ARA&A* **51**, 269 — "Stellar Multiplicity"
- Raghavan, D. et al. 2010, *ApJS* **190**, 1 — "A Survey of Stellar Families: Multiplicity of Solar-type Stars"
- Duquennoy, A. & Mayor, M. 1991, *A&A* **248**, 485 — "Multiplicity among solar-type stars in the solar neighbourhood. II"
- Jeans, J. H. 1919, *MNRAS* **79**, 408 — thermal eccentricity distribution
- Plummer, H. C. 1911, *MNRAS* **71**, 460 — "On the problem of distribution in globular star clusters"

## Truncation — read this before interpreting results

The period distribution is **truncated, and the truncation is severe**. Sampled raw, the
log-normal's short-period tail puts systems at a ≪ 1 AU whose orbits would set the global sync
cadence, so the run cost would be dominated by a handful of binaries that say nothing the wider
ones do not.

Two limits are applied. Semi-major axis is clipped to [`A_MIN_AU`, `A_MAX_AU`]; and the
**pericentre** a(1−e) carries its own floor, because eccentricity — not a — is what sets the
smallest lengthscale the integrator sees. Systems violating the pericentre floor have their
**(period, eccentricity) pair redrawn**, not their eccentricity clipped: clipping piles every
offender onto a single eccentricity, replacing the sampled f(e) with an artificial spike, while
redrawing gives the correctly truncated joint distribution.

Consequences at the default 100 AU floor, which are stated rather than buried:

- 53 of 256 systems clipped in a, and 186 (period, e) redraws — the accept region is a small
  slice of what is sampled.
- The surviving population is **biased wide**: median a ≈ 1717 AU, *wider* than the 1000 AU of
  the equal-mass test.
- **This is therefore not a harder test than `plummer_binaries` in the tight-binary sense.** Its
  value is the diversity and the ~8 timebins of orbital spread, not hardness.

The IC generator prints both clip counts on every run, so the truncation is visible.

## Cost

Runtime is set by the **tightest pericentre**, not the star count: that pair's orbit fixes the
smallest timestep and therefore the global sync cadence. Measured scaling at M = 1 M<sub>⊙</sub>,
dt ≈ t<sub>dyn</sub>(peri)/30, per unit TimeMax = 32.4:

| pericentre | steps |
|---|---|
| 1000 AU | 3.0e4 |
| 300 AU | 1.8e5 |
| **100 AU** (default) | 9.5e5 |
| 20 AU | 1.1e7 |

`plummer_binaries` sits at the top of that table (1000 AU, circular) and costs 15–40 min. This
test keeps a 100 AU floor and pays for it by running 10× shorter — `TimeMax 3.24` rather than
32.4 — landing within a few × of the equal-mass test.

### Duration

The cluster is **133.85 M<sub>☉</sub>** — the Kroupa IMF gives 256 systems far less mass than
512 equal 1 M<sub>☉</sub> stars — at the same 1 pc scale radius, so its timescales are ~2×
longer than `plummer_binaries`':

| | |
|---|---|
| t<sub>dyn</sub>(r<sub>half</sub>) | 1.96 code = 1.92 Myr |
| t<sub>cross</sub> | 2.92 code = 2.86 Myr |
| σ<sub>1D</sub> | 0.516 km/s |

`TimeMax 29.2` is **10 crossings**, matching what `plummer_binaries` covers, so the two are
comparable. An earlier value of 3.24 was set as "a tenth of the sibling's 32.4" without
accounting for the mass difference and delivered 1.1 crossings — roughly a twentieth of the
evolution, which is why the cluster appeared not to evolve.

Measured cost at TimeMax 3.24: ~4 min for 102 snapshots. The timestep structure is static (both
orbits circular, no pericentre refinement), so this scales linearly to **~36 min** at 29.2 —
against ~47 min for the equal-mass test.

## What is measured

Both diagnostics come from the `IO_HERMITE_SYNC` datasets, and the potential energy is
recomputed by direct O(N²) pairwise sum from those same positions rather than read from the
snapshot's `Potential` field. Pairing a last-kick velocity with a drifted-position potential
gives a kinetic and a potential term belonging to *different times*; at N ≈ 335 the exact sum
costs milliseconds. (The same fix was applied to `plummer_binaries`.)

- **Energy** |E/E₀ − 1| over the run.
- **Momentum** |v_com| in units of the cluster velocity dispersion. The ICs are built in the
  exact COM frame — Σmv = 0 to ~1e-15 — so any drift is entirely integration error.
- **Per-binary elements**, a and e, from a `BinaryCatalog` group written into the IC so the test
  does not have to re-derive the pairing. Elements round-trip through the snapshot to 1e-10 in a
  and 2e-8 in e.

Constants come from `gizmo.units`, which mirrors GIZMO's `declarations/constants.h` — **not**
astropy. The code integrates with `GRAVITY_G_CGS = 6.672e-8`, so reconstructing energies with
the CODATA value injects a spurious term that sweeps with each orbit.

**Tolerances are placeholders** pending a compute-node run. Until then the test's value is the
numbers it reports, not its pass/fail.

## Files

- `make_plummer_binaries_realistic_ics.py` — population sampling and IC generation; run
  standalone to inspect a draw without running GIZMO
- `plummer_binaries_realistic.params` — run parameters, with the cost reasoning
- `Config.sh` — identical physics flags to `plummer_binaries`, so the two differ **only** in
  initial conditions and run length
- `test_plummer_binaries_realistic.py` — diagnostics and assertions
