# Single-Star Wind Bubble Test

A 100 Msun ZAMS star drives a wind-blown bubble into uniform ambient gas. Tests the Weaver+ 1977 similarity solution and, in the adiabatic runs, exact energy conservation. Gravity is off, so the
only dynamics are wind injection and hydro.

## Setup

- 64^3 gas cells, 50000 Msun box at n_H = 100 cm^-3, periodic
- `SINGLE_STAR_FB_WINDS=2` with Mdot = 1e-4 Msun/yr, v_w = 3000 km/s
- Wind injection mode 1 (spawn discrete cells) or 2 (local mechanical injection)
- Adiabatic, `COOLING`, or `COOLING` + `SINGLE_STAR_FB_RAD` (the star's own radiative feedback)

## What is tested

- Shell radius against Weaver+ 1977 R2 = alpha (L_w t^3 / rho_0)^(1/5), the shock front. Weaver+
  solve both regimes and only the coefficient differs: alpha = 0.88 adiabatic, 0.76 radiative. Both
  to 10%.
- Growth index R ~ t^(3/5), and constancy of the similarity coefficient in time.
- Adiabatic only: E_kin + E_th = L_w t exactly, to 10% for spawning.

## Why this is a test of spawned-wind merging criteria

The exact solution has a **contact discontinuity** separating shocked wind from shocked ambient gas:
the two are in pressure equilibrium but jump in density and entropy, and they **never mix**. Any
mixing of the two is therefore purely numerical, and merging a spawned wind cell into an ambient one
is the most direct way to produce it. That makes this test unusually clean for evaluating merge
criteria — the correct answer is known, and the failure mode is unambiguous.

Merging spawned cells costs ~11 percentage points of retained energy under cooling (0.63 -> 0.52 of
L_w t) and pushes the growth index to 0.52, i.e. momentum-driven rather than energy-driven, for a
saving of only ~2% in cell count. `SINK_SPAWN_NO_MERGE` recovers alpha to 0.5% adiabatic and 1.6%
radiative, but is not viable in production, where winds that genuinely mix into the ISM must
eventually be retired or the cell count grows without bound. `SINK_SPAWN_MERGE_WHEN_AMBIENT` matches
it here while keeping that path open: it retires a spawned cell only once the cell's own state has
equilibrated with its kernel, kinematically and thermally. In this problem nothing ever qualifies,
which is the correct answer -- so the test bounds the harm a criterion can do, but cannot by itself
demonstrate that a criterion retires cells when it should. The wind cells carry a passive scalar (`Metallicity[:,-2]`) which labels the material, but
note it is subject to the metal diffusion model and is not a conserved tracer, so use it to locate
the interface rather than to budget mass.

## Shell radius definition

`measure_shell_radius` returns the peak of the spherically-averaged density, which needs no
threshold. Thresholded alternatives are all worse and are documented in that function: a density
contrast tracks the outer edge and depends on bin width; "innermost undisturbed particle" is biased
low however `undisturbed` is defined, because initial density perturbations relax acoustically
(defeating a velocity criterion) and the ambient itself cools or heats (defeating an entropy one).

## Known issues

Injection mode 2 delivers ~30% *more* energy than L_w t, unexplained and held at a loose tolerance so
the spawning path can be tested at 10%. Unrelated to the merge treatment: with local injection there
are no spawned cells to merge.
