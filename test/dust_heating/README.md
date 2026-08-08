# Protostellar dust heating

Tests for `SINK_DUST_HEATING_PLANCKMEAN`: the per-source Planck-mean dust heating rate summed in
the gravity tree, with an emergent photospheric colour temperature at the emitter and a
direct/thermalised opacity blend at the absorber.

One radiating sink, `SELFGRAVITY_OFF`, `InterstellarRadiationFieldStrength = 0`. Nothing else is
active, so a discrepancy is the radiation scheme's.

## What is being tested, and why in this order

**1. The tree sum — `glassbox`, uniform, optically thin.** The only fully quantitative check.
`DustHeatingRate` should equal κ_P(T_eff)·L/(4πr²) exactly. Measured **0.971, flat to 0.5% over
942–7743 AU**. This validates the per-source weighting, node aggregation across all four tree
paths, the MPI partial sums, and the 1/4πr² geometry in one number.

**2. The equilibrium profile — `glassbox`, converged.** d ln T_dust / d ln r = **−0.350** against
−0.363 for radiative equilibrium with this opacity law. The solved dust temperature also equals the
root of 4σT⁴κ_P(T) = Γ to four digits at every radius, which checks `rt_eqm_dust_temp` itself.

**3. The optically thick regime — `core`, after Chakrabarti & McKee (2005).** A rho ~ r^-p core
spans thin to thick, so it exercises the emergent-photosphere and blend terms that `glassbox`
cannot reach (there τ(r₀) = 0.12, and the emitter correctly falls back to T_star).

## Why the sink must not sit at the centre of the core

**This is the most important thing to know before using the `core` tests.**

The emitter estimates the envelope slope as `n_env = |grad rho| * KernelRadius / rho`. For
rho ~ r^-n that expression equals n·(h/r) — it is the slope only when the kernel radius happens to
equal the distance from the density peak. A sink at the centre of its own envelope has r = 0, the
gradient vanishes by symmetry, and what survives is noise.

Measured on a rho ~ r^-1.5 core, moving the sink by two cell lengths (39 AU in a 10⁴ AU core):

| | centred | displaced 39 AU |
|---|---|---|
| n_env | **1.20 (hits the floor)** | 1.94 (true 1.5) |
| R_raw | 3202 AU | 24.5 AU |
| R_max | 78 AU | 121 AU |
| clamp | **R_max cap binds** | unclamped |
| T_phot | 40.7 K | 72.6 K |
| slope | −0.659 | −0.366 |
| T_dust at 100 AU | 147 K | 38.5 K |

A 39 AU displacement changes the dust temperature by ~4×. The centred configuration is
pathological, not merely noisy.

It is also **not radiative at all**. Solving pure radiative balance against the code's own applied
heating rate, T_dust/T_radeq reaches 1.76 in the centred inner region while staying at a flat
1.15–1.20 across a factor of 12 in radius when displaced. So the centred inner temperature is set
by the gas, and profile slopes measured there do not test the radiation scheme. Two displaced runs
(2 and 4 cells) agree with each other to a few percent at every radius, which is the positive
control.

**Known issue:** the `GradRho` estimator is therefore ill-posed for the only geometry this code
path serves. It should be replaced by a fixed prior (n = 1.5 or 2) or a two-scale density ratio
that stays defined at r = 0. Not yet decided, because n between 1.2 and 3.1 moves T_phot by 2×.

## IC construction

Both families are glass-based, verified empirically rather than assumed — nearest-neighbour spacing
scatter is 0.084 for the stretched core and 0.047 for the uniform box, against 0.367 for Poisson.
MakeCloud's Poisson fallback only triggers on the `--makebox` path when N is not a cube of a power
of two, which `make_glassbox` asserts against.

Two traps the generator works around:

- **MakeCloud's `--nH` disagrees with GIZMO's.** Its makebox path uses ρ = n_H·m_p/0.71, while
  `UNIT_DENSITY_IN_NHCGS` is ρ/m_p with no hydrogen mass fraction. Passing `--nH` would hand GIZMO
  1.41× the requested density. The generator sets the mass explicitly instead.
- **MakeCloud writes `SinkRadius` = the softening value**, but GIZMO's own sink-formation path
  assigns `ForceSoftening_KernelRadius` = softening/0.357. An IC-placed sink and a self-formed sink
  would get radii differing by 2.8× from identical parameters. `_fix_sink_radius` corrects it.

The `relaxed` variant tests whether the radial anisotropy from glass-stretching matters. MakeCloud
maps a uniform glass through r_new ~ r^a with a = 3/(3+p), elongating cells radially by exactly a
(a = 2 for p = 1.5; measured ⟨|cos|⟩ = 0.434 against 0.500 isotropic). Relaxing to ⟨|cos|⟩ = 0.475
at fixed rho(r) changes the measured slope by **−0.000**. The anisotropy is innocent; the variant
is kept because that null result is what licenses ignoring it.

## Files

| file | what |
|---|---|
| `Config.sh` | compile flags, with the reasoning for `SELFGRAVITY_OFF` |
| `make_dust_heating_ics.py` | all IC variants: glass box, power-law cores, relaxed, displaced |
| `dust_heating.params` | fiducial parameter file |
| `test_dust_heating.py` | pytest: the quantitative tree-sum and profile checks |

ICs are generated, not committed (`test/*/*_ics.hdf5` is gitignored). Requires MakeCloud on PATH.
