# Hernquist halo: dark-matter physics examples

Three optional setups that run the equilibrium Hernquist halo of this directory with
non-standard dark-matter physics. They are examples to start from, not part of the
pytest sweep: `pytest test/hernquist` runs only the pure-gravity variants.

| Example | Config | Params | Module |
| --- | --- | --- | --- |
| self-interacting DM | `Config_sidm.sh` | `hernquist_sidm.params` | `DM_SIDM` (cross-section 1 cm^2/g) |
| fuzzy DM | `Config_dmfuzzy.sh` | `hernquist_dmfuzzy.params` | `DM_FUZZY` (1e-22 eV boson) |
| continuum Vlasov | `Config_cbe.sh` | `hernquist_cbe.params` | `CBE_INTEGRATOR` (8 bases, second moments, gradients) |

All three use the same IC, the pure-gravity test's G=1 halo at lower resolution:

```bash
cd test/hernquist
python make_hernquist_ics.py --N 4096 --boxsize 500 --out hernquist_dm_ics.hdf5
```

The params choose units of kpc, 1e10 Msun and 207.41 km/s, for which G = 1 exactly, so
the halo has M = 1e10 Msun and a = 1 kpc, and one time unit is 4.71 Myr (half-mass
crossing time ~11.8). Cross-sections and boson masses are therefore in physical units.

## Running an example

```bash
cp test/hernquist/Config_sidm.sh src/Config.sh      # pick an example
# ... build GIZMO ...
cd test/hernquist && mpirun -np 2 ../../src/GIZMO hernquist_sidm.params
python validate_snapshot_vs_ic.py --snapshot output_sidm/snapshot_010.hdf5
```

Each example writes its own output directory (`output_sidm/`, `output_dmfuzzy/`,
`output_cbe/`). `validate_snapshot_vs_ic.py` compares any snapshot with the IC: centre-of-mass
drift, kinetic energy, the cumulative mass and speed distributions, and the radial
velocity-dispersion profile, with a summary plot. Collisionless runs should stay
statistically indistinguishable from the IC; SIDM should flatten the inner density cusp
into a core over a few mean free times, with a hotter, more isotropic centre.
