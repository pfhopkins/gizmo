# turbulent_box

Driven isothermal turbulence in a periodic box at a fixed driving amplitude. Carried over from
`starforge_dev`, where the amplitude is instead held at a target Mach number by the
`TURB_DRIVING_TARGET_SIGMA` control loop; this tree does not have that loop, so the problem is a
stirring and timing vehicle here, with no pass/fail test. It is the box of GitHub issue #78
(hydro-dominated cost on PCIe GPUs).

| | |
|-|-|
| units | pc - km/s - Msun |
| box | 10 pc, periodic, 32^3 cells on a lattice by default, initially at rest |
| EOS | `EOS_GAMMA=1.001`, `EOS_ENFORCE_ADIABAT=0.04`: isothermal, c_s = 0.2 km/s |
| driving | solenoidal, parabolic spectrum on 1 <= kL/2pi <= 3 peaked at k = 2, coherence time L/(2 v_rms), amplitude from `TurbDrive_ApproxRMSVturb` = 2 km/s (Mach 10) |
| mass | 3875 Msun: alpha_vir = 5 sigma_3D^2 L / (6 G M) = 2 at Mach 10 |
| gravity | off (`SELFGRAVITY_OFF`) |
| variants | MHD: add `MAGNETIC` and `MHD_CONSTRAINED_GRADIENT=1` (uniform B_z = 1.6 uG in the ICs, plasma beta 1; builds without `MAGNETIC` ignore it) |

ICs: `python3 make_turbulent_box_ics.py [--n N] [--out FILE]`. The #78 probes use `--n 128`
(2.1M cells) with `TimeMax 0.4`, `MaxMemSize 4000`, `PartAllocFactor 2.0`; the long runs `--n 256`.
With `TreeDomainUpdateFrequency 0.05` the box develops a timestep hierarchy: about half the steps of
the 128^3 probe are all-active, far fewer at 256^3.

**Gravity.** This is the stirring phase. The mass is already set for alpha_vir = 2, so removing
`SELFGRAVITY_OFF` turns on a self-gravitating box, but with no sinks nothing stops isothermal
collapse once it starts.
