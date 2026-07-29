# hse_earth_smoke — layered self-gravitating body in hydrostatic equilibrium

A ~Earth-mass Tillotson body (olivine mantle + iron core) built in HSE and run
for a few dynamical times. The body should just sit there: if the IC and the
solid EOS agree, `rms|v|/v_dyn` stays small and the density profile does not
drift. It is the end-to-end check on the `initial_conditions/eos_tools` HSE
builder, and it is what catches EOS/composition regressions that the unit
tests miss.

Code units are CGS (`UnitLength = UnitMass = UnitVelocity = 1`).
`t_dyn(R) ~ sqrt(R^3/GM) ~ 810 s` for the default body.

## Default run

```bash
cd test/hse_earth_smoke
python make_ics.py                       # writes hse_earth_smoke_ics.hdf5 (2000 particles)
cp Config.sh ../../src/Config.sh         # or build via the usual test harness
# ... build GIZMO ...
mpirun -np 2 ../../src/GIZMO hse_earth_smoke.params
python analyze_hse.py output             # HSE diagnostics per snapshot
```

`analyze_hse.py` reports, per snapshot: mean and rms `|v|/v_dyn`, the radial
density profile against the IC, centre-of-mass drift, and the largest
excursion of any particle from its IC radius in units of `R_body`.

## Variants (optional — none of these are the default)

Each variant is one `make_ics_*.py` + one `.params`, and reuses `Config.sh`
unless a different Config is named. They exist because shell-based ICs expose
a specific failure mode in the smoothing-length iteration; keep them.

| Variant | Build with | IC script | Params | What it is for |
|---|---|---|---|---|
| default | `Config.sh` | `make_ics.py` | `hse_earth_smoke.params` | olivine mantle + iron core, glass-relaxed |
| basalt | `Config.sh` | `make_ics_basalt.py` | `hse_basalt.params` | single-material body — isolates EOS behaviour from the layer interface |
| fibonacci | `Config.sh` | `make_ics_fibo.py` | `hse_earth_fibo.params` | Fibonacci-shell placement instead of glass relaxation |
| fibonacci + h guess | `Config_hguess.sh` | `make_ics_fibo_hguess.py` | `hse_earth_hguess.params` | as above, but the IC carries a precomputed kernel radius (`INPUT_READ_KERNELRADIUS`), testing whether a better initial guess alone escapes the iteration trap on shell ICs |
| no gravity | `Config_nograv.sh` | any of the above | matching `.params` | `SELFGRAVITY_OFF`; separates EOS/pressure errors from gravity errors |

The IC builders import from `initial_conditions/eos_tools` (`hse_solver`,
`shell_placement`, `sph_density`, `build_layered_body`), the same modules used
by `test/aneos_giant_impact`.

## Files
- `make_ics.py`, `make_ics_basalt.py`, `make_ics_fibo.py`, `make_ics_fibo_hguess.py` — IC builders (see table)
- `analyze_hse.py` — HSE diagnostics over an output directory
- `Config.sh` — default build (`EOS_TILLOTSON`, `ADAPTIVE_GRAVSOFT_FORGAS`, double precision in/out)
- `Config_hguess.sh` — adds `INPUT_READ_KERNELRADIUS`
- `Config_nograv.sh` — adds `SELFGRAVITY_OFF`
- generated ICs (`*_ics.hdf5`) and `output*/` are gitignored
