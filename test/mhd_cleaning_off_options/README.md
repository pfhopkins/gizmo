# mhd_cleaning_off_options

Two debugging switches of the MHD scheme, both test only and off by default:

- `TEST_MHD_NO_DEDNER_CLEANING`: `DIVBCLEANING_DEDNER` is not defined for a `MAGNETIC` build, so there is no hyperbolic or
  parabolic cleaning field.
- `TEST_MHD_NO_POWELL_SOURCE_TERMS`: the source terms proportional to div B (in the induction equation, the acceleration and
  the internal energy) are left out of the hydro step.

Problem: Orszag-Tang vortex on a 32^2 lattice (2D, periodic), 0.2 time units, three builds (default and one with each flag),
2 ranks.

Checks: the default run has a nonzero cleaning field and no cleaning field is written without the cleaning; without the Powell terms the
evolved magnetic field differs from the default run (above 1e-6 of the maximum) while the cleaning field is still nonzero.
