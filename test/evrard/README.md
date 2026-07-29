# evrard — adiabatic collapse of a cold gas sphere

The Evrard (1988) collapse, as presented in Hopkins (2015). A cold `gamma=5/3`
sphere with `rho ~ 1/r` collapses until an outward-propagating accretion shock
forms. The test compares the shocked radial density profile against
`evrard_exact.txt`. It is the standard check that self-gravity, adaptive
softening and the hydro solver stay consistent with each other under strong
compression.

## Default run

`pytest test/evrard` runs two variants automatically, via the harness's
`extra_config_flags` mechanism (each gets its own `output*` directory):

| id | extra flags |
| --- | --- |
| `baseline` | none |
| `tidal_adaptive` | `TIDAL_TIMESTEP_CRITERION`, `ADAPTIVE_TREEFORCE_UPDATE=0.06` |

## Variants (optional — run by hand)

`extra_config_flags` can only *append* to `Config.sh`, so variants that need to
*replace* a flag get their own Config. Build by copying the Config over
`src/Config.sh`, then run the matching params from this directory.

| Variant | Config | Params | What it changes |
| --- | --- | --- | --- |
| default | `Config.sh` | `evrard.params` | `ADAPTIVE_GRAVSOFT_FORGAS`; full collapse to `TimeMax=0.8` |
| all-type adaptive softening | `Config_forall.sh` | `evrard.params` | `ADAPTIVE_GRAVSOFT_FORALL=1` in place of `FORGAS` — adaptive softening for every particle type, not just gas |
| gravity-walk validation | `Config.sh` or `Config_forall.sh` | `evrard_agswalk.params` | short run (`TimeMax=0.05`), larger `SofteningGas`, aggressive `TreeRebuild_ActiveFraction` — for comparing gravity-walk output between two builds rather than against the exact solution; writes to `output_agswalk/` |

`Config_forall.sh` was previously a separate `evrard_forall/` test directory,
but its test file was a byte-identical copy that hardcoded `test_name =
"evrard"` — so it rebuilt from `test/evrard/Config.sh` and never exercised
`FORALL` at all. It is a Config variant, and is now stored as one.

## Files

- `Config.sh`, `Config_forall.sh` — build configs (see table)
- `evrard.params` — the default run
- `evrard_agswalk.params` — short gravity-walk validation run
- `evrard_exact.txt` — reference solution
- `test_evrard.py` — pytest implementation
