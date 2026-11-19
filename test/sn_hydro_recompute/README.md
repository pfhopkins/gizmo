# sn_hydro_recompute

Feedback after the closing kick with the hydro pass redone (stella_dev `RESOLVED_ISM_FB_AFTER_CLOSING_KICK`, formerly
`FB_G17_BOUNDARY_INJECTION`; `RESOLVED_ISM_RECOMPUTE_HYDRO_POST_FB`). Adiabatic gas, one 10 Msun star blowing its wind and dying
200 yr in, 20^3 cells of 1 or 0.1 Msun, 2 ranks, 5 kyr, turbulent metal diffusion across a metallicity step far from the star.

- no repeat without the flags; with `RESOLVED_ISM_FB_AFTER_CLOSING_KICK` the SN step repeats, and with
  `RESOLVED_ISM_RECOMPUTE_HYDRO_POST_FB` also the wind steps (more repeats);
- the repeated pass applies no second metal exchange: cells beyond the remnant (0.4 L) end with the metallicity of the run without the flag (1e-6 of the exchange); one metal slot in the ICs, as the build; a shear layer at that interface drives the exchange;
- old names, the second flag alone and the flag with MFV stop the build.

The energy excess of the runs (the single-SN ladder) is printed.
