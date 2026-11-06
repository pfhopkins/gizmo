# turbulent_box

Driven isothermal turbulence in a periodic box. `Config.sh` drives at a fixed amplitude: that build is
the stirring and timing vehicle of GitHub issue #78 (hydro-dominated cost on PCIe GPUs), with no
pass/fail test of its own. `test_turbulent_box.py` builds the same problem with
`TURB_DRIVING_TARGET_SIGMA=0` added, so the control loop holds the Mach number at its target (see
"Control-loop test" below).

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

**Control-loop test.** `test_turbulent_box.py` adds `TURB_DRIVING_TARGET_SIGMA=0` to the build and
runs four variants: hydro / MHD (the MHD flags above), each with the driving force recomputed every
step (`full`) or only on driving updates every 0.25 = t_eddy/20 (`lazy`,
`TURB_DRIVING_UPDATE_FORCE_ON_TURBUPDATE`). The target is Mach 10, i.e. 3D mass-weighted
v_rms = `TurbDrive_ApproxRMSVturb` = 2 km/s. The Mach number is measured from the snapshots,
independently of the loop's own measurement: mass-weighted rms velocity about the mean over the
isothermal sound speed. After 2 t_eddy (t_eddy = `TurbDrive_MaxWavelength / TurbDrive_ApproxRMSVturb`
= 5) the mean must be within 3% of the target and every snapshot within 10%, and the peak over the
whole run, spin-up included, must stay under 1.1x the target. The analysis and checks live in
`turb_driving_common.py`, shared with `test/turbsphere`. `mach_vs_time<variant>.png` shows the
snapshot Mach number, the loop's measurement, the driving gain, and for MHD the plasma beta.
