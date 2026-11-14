# sn_cluster_adiabatic: energy budget of FIRE-3 supernovae + MFM hydro

A 1e3 Msun FIRE-3 stellar population (age 3.5 Myr, 12 O stars, split into star particles of the gas
cell mass within 2 pc) sits at the centre of a static, uniform, periodic box of gas: 1e5 Msun,
n_H = 1 cm^-3, T = 1e4 K, 141.5 pc on a side. Units are pc, Msun, km/s. The run lasts 10 Myr
(t = 10.24), in which 3-7 core-collapse supernovae go off: the first into quiescent gas on long
timesteps, the later ones into the superbubble, which wraps around the box.

The build has no cooling, no gravity (`SELFGRAVITY_OFF`), no radiation and no star formation, and
stellar mass loss is off (`GasReturnFraction 0`), so the supernovae are the gas's only energy source
and the budget is trivial: gas thermal + kinetic energy = initial value + supernova energy. The FIRE
flags are set explicitly rather than through `FIRE_PHYSICS_DEFAULTS=3`, which forces `COOLING` on;
`COOL_METAL_LINES_BY_SPECIES` and the age tracers are left out because they need cooling tables, and
without them the FIRE-3 supernova energy does not read a metallicity slot that would not exist.

The ICs are written by `make_sn_cluster_adiabatic_ics.py` and read with restart flag 2 (a snapshot
restart): a flag-0 start zeroes the star particles' formation times and O-star counts, and a FIRE-3
population with no O stars has no core-collapse supernovae.

## What is checked

Energies come from `ENERGY_BUDGET_DIAGNOSTIC`, at full synchronizations only (the `Energy (synced)`
lines; see Template_Config.sh for why snapshots are not usable for this). Besides the gas energy E,
that line carries E_fb, the cumulative energy mechanical feedback has written into gas velocities and
internal energies. N_SN is the sum of column 4 of `MechFeedbackEvents.txt`. Two independent tests per
case, at the last synchronization at or before t = 10.24:

* `test_hydro_conserves_energy`: |E - E0 - E_fb| / E_fb < 5%. The hydro and time integration alone. With
  the time integration resolved (the `m100_resolved` control, timestep capped at 0.005 = 5 kyr) the
  budget closes to 1e-4 at 100 Msun (3e-3 at 10 Msun), so 5% sits more than 10x above that floor.
* `test_sne_broadcast_timestep_conserves_energy`: the four cases that fail without `SNE_BROADCAST_TIMESTEP`,
  in the default build (which has it), at |E - E0 - E_fb| / E_fb < 20%. The flag removes the error at the explosion; what
  remains (+5% to +16%) builds up as the blasts expand through cells on steps set by the unchanged Courant
  factor. A 4x smaller Courant factor, or a non-local Courant condition through the gravity tree, takes it
  below 1% at 1.3-2x the cost; neither is in the flag.
* `test_feedback_delivers_sn_energy`: |E_fb / (N_SN * 1e51 erg) - 1| < 5%. The coupling alone. The FIRE
  scheme is built to couple exactly E_SN per event, except that ejecta mass joins a cell at the cell's
  specific internal energy, adding m_ej * u_cell; inside a hot bubble that is percent-level (0.4% at
  100 Msun and 1.9% at 10 Msun).

The product of the two is the full budget, (E - E0 - N_SN E51)/(N_SN E51), which each case prints.

Cases: gas cell mass 100 or 10 Msun, and TimeMax = 10.24 or 1024 at the same 10 Myr duration (TimeMax
fixes the integer timestep ladder, (TimeMax - TimeBegin) 2^-k; the long-TimeMax runs are stopped on a
`stop` file once a synchronization reaches t = 10.24). Runtime about 2 minutes at 2 ranks x 2 threads.

## Current status

`SNE_BROADCAST_TIMESTEP` is on by default for this configuration. The hydro check fails when it is turned off
with `SNE_BROADCAST_TIMESTEP_DISABLE` (marked xfail(strict=True)) and passes in the default build; the
coupling check passes.

| case | (dE - E_fb)/E_fb, disabled | with `SNE_BROADCAST_TIMESTEP` | E_fb/(N_SN E51) |
|---|---|---|---|
| m100_resolved | -0.0003 | | 1.004 |
| m100_TimeMax1x | +0.37 (+0.45 before the coupling fix) | +0.057 | 1.001 |
| m100_TimeMax100x | +3.40 | +0.052 | 1.005 |
| m10_TimeMax1x | +2.13 | +0.161 | 1.018 |
| m10_TimeMax100x | +1.56 | +0.143 | 1.016 |

1. Hydro. Nearly all of the error without the flag appears at the first supernova into quiescent gas (+8 to
   +22 E51 from one 1e51 erg event); later supernovae into the hot bubble add +0.1-0.6 E51 each. Mechanical
   feedback is deposited after the density pass and before the hydro force pass (`compute_stellar_feedback` in
   `core/accel.cc`), so a receiving cell's closing half-kick of its old step applies the post-explosion
   pressure forces over half a step sized for the pre-explosion gas. The error is a steep function of that
   step: capping MaxSizeTimestep at 10 Msun (uncapped cold gas sits on 0.16) gives +2.49 (0.16), +0.32
   (0.08), +0.09 (0.04), -0.02 (0.02), -0.02 (0.01), -0.003 (0.005). That steepness is why a change of
   TimeMax, which shifts the ladder by a non-power-of-2 factor, changes the result so much.
   `SNE_BROADCAST_TIMESTEP` removes the error at the explosion; the residual builds up as the blasts expand.
2. Coupling. The weighting pass (loop -1 in `galaxy_sf/mechanical_fb.cc`) used the star particle's mass in
   place of the ejecta mass in the mass-loading factor of the kinetic-energy coefficients, so the momentum
   solution mis-sized the kinetic energy: 1.06-1.08 at 100 Msun, 1.5 for one 1e3 Msun particle in 100 Msun
   gas, 0.87 at 1 Msun. It now uses the ejecta mass of the channel being coupled (SNe, else mass loss).

