# Athena++ reference solutions for briowu and toth

`test/briowu/briowu_exact.txt` and `test/toth/toth_exact.txt` are 1D ideal-MHD solutions from
Athena++ (Stone et al. 2020, ApJS 249, 4), commit `053bc738d5c85dac717326b4338b34c7db4e715d`:
HLLD fluxes, PLM reconstruction (`xorder = 2`), VL2 integrator, CFL 0.4, 8192 cells on the
periodic domain [0, 4] with the jump at x = 2, matching the GIZMO boxes (including the inverted
image of the jump at x = 0 = 4). Athena++ B is in the same units as GIZMO's internal field
(magnetic pressure B^2/2), so no conversion is applied. The harness also tries to download
`<test>_exact.txt` from the test mirrors; the copies there are identical.

`athinput.briowu` and `athinput.toth` are the inputs; `athena_tab_to_exact.py` converts the final
tab dump into the `_exact.txt` format (columns `x rho vx vy vz By Bz P`, Bx and provenance in the
header).

## Regenerating

```sh
git clone https://github.com/PrincetonUniversity/athena && cd athena
git checkout 053bc738d5c85dac717326b4338b34c7db4e715d
python configure.py -b --prob=shock_tube --flux=hlld --coord=cartesian
make -j4
T=<gizmo>/test/mhd_reference_athena
for p in briowu toth; do
  bin/athena -i $T/athinput.$p -d run_$p mesh/nx1=8192
  (cd <gizmo>/test/$p && python3 $T/athena_tab_to_exact.py $p <athena>/run_$p/$p.block0.out1.00001.tab \
     053bc738d5c85dac717326b4338b34c7db4e715d "-b --prob=shock_tube --flux=hlld --coord=cartesian" \
     "integrator vl2, xorder 2 (PLM)")
done
```

Each run takes under a minute on one core. The PPM cross-check needs `--nghost=4` at configure
time and `time/xorder=3` (optionally `time/integrator=rk3`) on the command line.

## Checks (2026-11-05)

- toth plateaus match Ryu & Jones (1995) table 1a to the digits tabulated: rho 2.6797 / 3.7481,
  P 150.98 / 143.57, By 3.8388 / 5.4271 behind the left / right fast shocks.
- briowu plateaus (t = 0.2 on [0, 4]): rho 0.676, 0.697, 0.235, 0.117.
- 4096 vs 8192 cells, L1 over the tests' comparison window relative to each field's range:
  briowu <= 0.14% (vx), toth <= 0.17% (vy). PLM vs PPM (VL2 or RK3) at 8192 cells: <= 0.27%;
  plateau means agree to <0.1%.
