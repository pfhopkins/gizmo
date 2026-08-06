# Rotating Shu 1977 collapse with protostellar jets

A variant of the [shu1977](../shu1977) test: the same singular isothermal sphere at 2x the
critical density (10 Msun of gas in 128000 cells, R = 0.269 pc), but spun up into solid-body
rotation and run with kinematic protostellar jets enabled.

The rotation gives the collapse a well-defined angular momentum axis, so the sink that forms
launches a bipolar jet along z instead of an unconstrained direction. The test checks that
exactly one sink forms, that it spawns jet cells at the requested mass resolution, and that
those cells form a collimated outflow rather than an isotropic wind.

## Initial conditions

`make_shu_jets_ics.py` takes the shu1977 ICs (which start at rest) and adds solid-body rotation
about the z axis, normalized to

```
beta = E_rot / |E_grav| = 0.1
```

E_grav is evaluated with the spherical-shell formula, exact for the spherically symmetric SIS.
This gives Omega = 1.411 (km/s)/pc = 1.443 / Myr, an edge velocity of 380 m/s, and a Peebles spin parameter
lambda = J|E|^(1/2)/(G M^(5/2)) = 0.21.

## Jets

`SINGLE_STAR_FB_JETS` implies `SINK_WIND_SPAWN`, so jet material is injected as newly spawned
gas cells. Their mass is set by `Sink_outflow_particlemass`, which is an absolute mass in code
units here (`SINK_SCALE_SPAWNINGMASS_WITH_INITIALMASS` is off), and is set to

```
Sink_outflow_particlemass = 7.8125e-06 = 0.1 * m_gas
```

i.e. jets are resolved 10x more finely than the ambient gas. Of the mass the sink accretes,
`1 - Sink_accreted_fraction = 0.3` is relaunched. The launch velocity is not a free parameter:
without `SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION` the code uses
`Sink_outflow_jetlaunchvelscaling` (default 0.3) times the Kepler velocity at a fiducial 10 Rsun
protostellar radius, following Federrath et al. (2014).

Compile-time flags. `Config.sh` carries only `SINGLE_STAR_STARFORGE_DEFAULTS`; the test appends
`SINGLE_STAR_FB_JETS`, `JET_DIRECTION_FIXED_Z`, `COOLING`, `OUTPUT_COOLRATE_DETAIL`, plus
`SINK_SPAWN_NO_MERGE` for the second run. `SINK_SPAWN_MERGE_WHEN_AMBIENT` and
`MERGE_SPLIT_LIMIT_KINETIC_DISSIPATION` are auto-enabled by `SINGLE_STAR_FB_JETS`.

## What is tested

Two runs, both with cooling: the **default merge criteria** against **no spawned-cell merging**.
Cooling rather than isothermal because `EOS_ENFORCE_ADIABAT` pins the specific internal energy,
which makes the thermal half of the retirement criterion inert (measured: 0 of 474 cells blocked,
all in the lowest `|dln u|` bin); with cooling it discriminates (207 of 1190).

Per run:

- exactly one sink forms, and it spawns cells at `Sink_outflow_particlemass`
- the spawned cells move outward (median radial velocity > 0)
- **positional** collimation: `<cos^2(theta)>` of the cells' angles about the sink > 0.9. Healthy
  runs give 0.96-0.99. With the axis pinned this is a launch-geometry check with a known answer.
- **kinematic** collimation: principal-axis ratio `L1/L2` of the kinetic-energy tensor
  `T_ij = sum m dv_i dv_j` > 50, and its principal axis within 0.99 of z. Weighting by energy
  matters -- a cell-averaged `<(dvz/|dv|)^2>` rewards aggressive retirement, since discarding slow
  deflected cells leaves a more axial sample, and ranks the regimes backwards. The floor is loose
  by design: runs differing only in whether *ambient* gas merges came out 213 vs 155.

Across the two runs, `test_shu_jets_merging_energetics` requires total gas energy, outward momentum
and sink mass to agree within 10%. That is the point of the retirement criterion -- retire cells
that have joined the ISM, protect those that have not -- so it should match the no-merging case
while still retiring (it ends with ~0.3% fewer cells). Note the purely kinetic measures sit at
10.5-11% between these runs, so they are deliberately not the ones bounded; they are also the most
sensitive to domain decomposition (the retirement block fraction moved 7.9% -> 17.4% between 8 and
48 ranks on identical physics).
