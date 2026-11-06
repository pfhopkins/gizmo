# conduction_step: isotropic and field-aligned thermal conduction

A sinusoidal internal-energy mode in a frozen 2D periodic box, conducted isotropically (no `MAGNETIC`) and along a uniform magnetic field at 0, 45 and 90 degrees to the temperature gradient. The decay rate and the final profile are compared to the analytic diffusion solution.

## Why

Production STARFORGE builds (`SINGLE_STAR_STARFORGE_DEFAULTS` with `COOLING` and `MAGNETIC`, `declarations/precompiler_logic.h` ~483) and `FIRE_MHD` (~198) switch on `CONDUCTION` + `CONDUCTION_SPITZER` + `VISCOSITY` + `VISCOSITY_BRAGINSKII`. Before this test, the conduction operator was covered only by compile checks (`test/compile_suite`). OpenGadget3 (Dolag et al. 2026, arXiv:2609.17658, sec. 2.5/9) runs a temperature-step test with B at 45 degrees. This test is a quantitative version of it for GIZMO's MFM operator.

## Physics

GIZMO's conduction coefficient is a conductivity (`core/timestep.cc`: `dt_cond ~ L^2 rho / kappa`), and the flux it adds to the energy equation is `-kappa A.grad u` per face (`hydro/conduction.h`), so

    rho du/dt = div(kappa_eff grad u),   D = kappa / rho.

With `MAGNETIC`, the flux is projected along the face-averaged field direction `bhat`: `(bhat.A)(bhat.grad u)`. For a gradient at angle theta to B this gives `kappa_eff = kappa cos^2(theta)`. Without `CONDUCTION_SPITZER`, `kappa = ConductionCoeff`, a constant (`eos/eos.cc` ~657, tag at `core/begrun.cc` ~1881).

The initial condition is `u(x) = u0 (1 + A sin(kx))` with `k = 2 pi / L`, which is an eigenmode. The solution is

    u(x, t) = u0 + u0 A sin(kx) exp(-D_eff k^2 t),   D_eff = (kappa / rho) cos^2(theta).

A step or erf profile, as in OpenGadget3, has no closed form in a periodic box without image sums, and its error is concentrated at the step. The single mode gives one clean decay rate to fit, and a profile to compare everywhere.

## Setup

| | |
|---|---|
| box | 2D periodic, L = 1, square 64 x 64 cell-centred lattice, rho0 = 1, at rest |
| mode | u0 = 1, A = 0.2 |
| kappa | `ConductionCoeff = 1/(4 pi^2)`, so `D k^2 = 1`: theta = 0 or isotropic decays by e over `TimeMax = 1`; 45 degrees by e^-0.5; 90 degrees not at all |
| B | uniform, in-plane, magnitude 0.1 (only its direction matters), from the IC `MagneticField` dataset |
| steps | `MaxSizeTimestep = 0.001`, so dt = 1/1024 for every cell (1024 steps, all on one timebin) |
| snapshots | every 0.125, 9 in total |

**Timestep.** GIZMO's conduction limit is `0.5 L_cond^2 rho / kappa`. The gradient-based `L_cond` in `core/timestep.cc` is `DMAX(L_particle, 1/(|grad u|/u + 1/L_particle))`, which is always `L_particle`. The limit is therefore `0.5 dx^2 rho / kappa = 4.8e-3`, about 207 steps per run. The test caps the step at about 0.2 of that, for two reasons:
- Every cell stays on one bin, which energy conservation to round-off needs.
- The per-pair flux limiter in `conduction.h` (`|dt F| <= 0.25 |m_i u_i - m_j u_j|`) stays inactive, except on faces where it is physically irrelevant (below).

The run measures the spatial operator, not time-integration error, which at `lambda dt = 1e-3` is negligible.

**Lattice size.** 64^2 cells give `k dx = 0.098` and `k h = 0.25` (h = 2.52 dx at `DesNumNgb 20`). Resolution error in the rate is then well under 1%, at about 4e6 cell-steps per case.

**FREEZE_HYDRO** isolates conduction:
- The Riemann fluxes are zeroed before `conduction.h` is included (`hydro/hydro_evaluate.h` ~279).
- Conduction then adds to `Fluxes.p`, which reaches `DtInternalEnergy` (~388) and the kick (`core/kicks.cc`).
- Positions are not drifted (`core/predict.cc` ~148).
- Every `DtB`/`divB`/Dedner contribution is skipped (`hydro_evaluate.h` ~393, ~435), and those accumulators are zeroed each step (`hydro/hydro_toplevel.cc` ~1031), so B stays constant and the angle is fixed.
- Dedner cleaning (on by default with `MAGNETIC`) acts on `divB`, which stays zero.
- The kick's entropy switch needs gravity, and gravity is off.

**One binary, several runs.** The three field angles differ only in their IC. They share one `MAGNETIC` build and run with `InitCondFile`/`OutputDir` overridden per case, into `output_iso`, `output_theta00`, `output_theta45` and `output_theta90`. The isotropic case uses the plain `Config.sh`. A whole session is two builds and four short runs.

## Assertions

All tolerances are PROVISIONAL (untuned, not yet run).

| check | tolerance | catches |
|---|---|---|
| premise: cells, velocities and field direction unchanged; t=0 amplitude = u0 A | round-off | `FREEZE_HYDRO` not in effect, B evolving, IC misread |
| (c) total internal energy `sum m u` constant across snapshots | 1e-10 | a non-antisymmetric pairwise exchange, cells on different timesteps, any other energy source |
| (d) u stays inside its initial range | 1e-3 u0 A | overshoot and instability (timestep, broken flux limiter, a sign error in the HLL/limiter terms) |
| (a) fitted decay rate vs `D_eff k^2` for iso, 0 and 45 degrees | 2% | wrong coefficient or units (kappa vs D), wrong anisotropy projection (e.g. `cos(theta)` instead of `cos^2`, giving 41% at 45 degrees), slope-limiter or HLL over- or under-diffusion |
| (a) 90 degrees: decay rate below 1e-3 of `kappa k^2 / rho` | 1e-3 | cross-field leakage (isotropic terms surviving the projection) |
| (b) final-profile L1 vs analytic, relative to u0 A | 5e-3 | distortions of the mode shape that a single rate does not show |

`D` uses the mean kernel density from the snapshot (0.9995 on this lattice), the rho in the code's discrete operator.

## Expected results, before the first run

A numpy re-implementation of the t=0 flux on this lattice was integrated over the run (not shipped). It included the kernel, `NV_T`, MFM faces, the slope-limited gradients from `hydro/gradients.cc`, and every branch of `conduction.h` with the flux limiter at this dt. Fitted rate / analytic rate:

| case | predicted |
|---|---|
| theta = 0 | 0.998 |
| theta = 45 | 1.0007 (of the halved rate) |
| theta = 90 | 0 to round-off |
| iso | **0.79** |

**The isotropic case is predicted to fail by about 21%.** The cause is in the isotropic branch of `conduction.h`: `grad_ij[k] = MINMOD(q_grad[k], q_direct[k])` limits each component of the face gradient against `d_scalar * dp[k] / r^2`, which is the gradient projected onto the pair direction. For a face at angle phi to grad u, the flux falls by `cos^2(phi)`. The deficit is the same at 32^2 and 64^2, so it does not converge away. With the MINMOD removed, the emulated rate is 0.998, the same as theta = 0. The `MAGNETIC` branch uses `MINMOD_G`, which returns its first argument (`core/proto.h` ~163), so it is unaffected. Production runs always have `MAGNETIC`, so this affects users of isotropic `CONDUCTION`. Treat it as a candidate code defect to confirm on the first run, not as a reason to loosen the tolerance.

At 45 degrees, faces whose neighbours share u (pairs along y) carry a nonzero anisotropic flux that the pair flux limiter zeroes, because `|m_i u_i - m_j u_j| = 0`. Here that changes nothing: those fluxes have zero divergence when u depends on x alone. A glass IC or a gradient not aligned with the lattice would expose it.

On the exact lattice, the 90-degree case conducts nothing at all, because the y-component of every least-squares gradient cancels by symmetry. The leakage bound therefore tests the projection code path, not noise-driven leakage. A glass IC would make that check sharper.

## Running

    pytest test/conduction_step

The test writes its ICs (`make_conduction_step_ics.py`, regenerated on each run), the four `output_<case>/` directories, and `conduction_step_<case>.png`: final u(x) against analytic, and mode amplitude against time.
