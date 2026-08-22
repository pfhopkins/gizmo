# isodisk_mechfb — mechanical feedback in an isolated disk

An isolated disk galaxy with star formation and mechanical (SNe/wind) feedback.
The default run is the activation test for `GALSF_FB_MECHANICAL`: the coupling
must inject the expected momentum and energy without the disk tearing itself
apart.

All variants share the same `isodisk_mechfb_ics` and differ only in Config, so
each writes to its own output directory and they can coexist. `pytest
test/isodisk_mechfb` runs the default.

## Variants

| Variant | Config | Params | What it changes |
| --- | --- | --- | --- |
| default | `Config.sh` | `isodisk_mechfb.params` | `GALSF_FB_MECHANICAL` on a plain `COOLING`+`GALSF`+`METALS` disk |
| cosmic rays | `Config_cr.sh` | `isodisk_mechfb_cr.params` | the FIRE physics set with `FIRE_CRS=(-1)`, `FIRE_MHD` and a reduced CR speed of light — mechanical feedback with the CR fluid live. Writes `output_cr/` |
| cosmic rays, multi-bin | `Config_cr0.sh` | `isodisk_mechfb_cr0.params` | as above but `FIRE_CRS=0` (multi-bin CR spectrum). Writes `output_cr0/` |

The two cosmic-ray variants were formerly a separate `isodisk_mechfb_cr/`
directory that pointed at these same ICs; they are Config variants of this
test, and are stored as such.

`isodisk_mechfb_sinks` remains its own test: it builds on a *different* IC
(`isodisk_mechfb_sinks_ics`) and has its own pass/fail implementation.

## Running a variant

```bash
cp test/isodisk_mechfb/Config_cr.sh src/Config.sh
# ... build GIZMO ...
cd test/isodisk_mechfb && mpirun -np 2 ../../src/GIZMO isodisk_mechfb_cr.params
```
