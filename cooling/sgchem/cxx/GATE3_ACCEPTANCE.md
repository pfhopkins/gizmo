# GATE 3 ACCEPTANCE — the SGCHEM_CXX path is Fortran-free

Date: 2026-07-29. Machine: orion01, gcc/gfortran 13.1.0, SUNDIALS 6.7.0.
Networks: 5 and 17. Everything below was RUN; nothing is projected.

Gate 3's brief was blunt: *every* remaining Fortran kernel and table builder
becomes C++, each golden-tested against the Fortran before moving on, tolerance
1e-12, and no tolerance is ever loosened to make something pass.

---

## 0. Bottom line

| item | result |
|---|---|
| leaf kernels + table builders ported | **done** — 6 new TUs, 0 Fortran left in the path |
| Gate-3 goldens (kernels + tables) | **ALL PASS**, both nets: **38,918,708 comparisons, 0 failures, worst rel 4.2e-13** |
| Fortran-symbol audit of the pure C++ binary | **PASS** — links with `g++`, no `.F` object, no `libgfortran` |
| OpenMP 8 threads × 64 concurrent solves | **bit-identical** to serial (audit R2 closed) |
| Gate-2 acceptance re-run on the pure backend | **(a) PASS, (b) PASS, (c) PASS** |
| bugs found *by* the goldens | 6 (5 literal-kind, 1 build) — all fixed, all documented |

---

## 1. What was ported

| Fortran source | lines | C++ |
|---|---|---|
| `cool_func.F` | 1329 | `chem_cool_func.cc` |
| `wss_z_collis.F` (`get_coolrates_wssCIE`) | 103 | `chem_cool_func.cc` |
| `cool_util.F` (`co_cool`, `h2o_rot_cool`, `h2o_vib_cool`, `three_level_pops`, `compute_stim`, `compute_h2_opacity`, `calc_dust_temp`) | 1014 | `chem_cool_util.cc` |
| `calc_photo.F` (`calc_photo`, `chi`, H2/HD self-shielding, CO line shielding) | 938 | `chem_calc_photo.cc` |
| `calc_temp.F` + `compute_gamma.F` | 287 | `chem_calc_temp.cc` |
| `coolinmo.F` + `cheminmo.F` + `photoinit_ism.F` + `const_rates.F` + `wss_z_collis.F` (reader) + `load_H2_table` + `init_temperature_lookup` | 4634 | `chem_tables_build.cc` (+ `rates_gen.cc` from Gate 1, `chem_spline.cc` from Gate 1b) |
| — | | `chem_backend_cxx.cc` (pure-C++ backend: immutable tables + per-cell context + thread-safe C API) |

`ChemTables` is now **built**, not a view: 41 `DATA` arrays are machine-extracted
by `tables_extract.py` with the compiler's own literal kinds, the 346 analytic
rate coefficients come from the Gate-1 generator, and this file reproduces the
table *assembly* — the temperature grid, the spline chains, the residual
zeroing loops and the finite-difference derivatives.

`chem_backend_fortran.cc` survives **only as the golden harness's reference
side**: it binds `ChemTables` to the COMMONs and exposes `fort_cool_func`,
`fort_calc_dust_temp`, `fort_calc_H2_shielding_factor`,
`fort_lookup_temp_gamma` so both implementations can be driven side by side in
one binary. It is not on the production path.

---

## 2. The goldens

`golden/build_gate3.sh` → `golden/test_gate3.cc`. One binary links the frozen
Fortran objects *and* the C++ port, so both see identical inputs. Four phases:

* **A** — the C++ table build vs what `COOLINMO`/`CHEMINMO`/`PHOTOINIT`/
  `CONST_RATES`/`read_wss_collis_table`/`LOAD_H2_TABLE`/
  `INIT_TEMPERATURE_LOOKUP` actually built. **Every element of every array.**
* **B** — the leaf kernels with BOTH sides reading the **Fortran** tables. This
  is what isolates a kernel deviation from a table deviation.
* **C** — the same kernels on the **C++-built** tables (end to end).
* **D** — the `iflag_atom = 1` `cl6` branch, which production never uses and
  which was therefore listed as untested in DEVIATIONS "Open / deferred". Now
  tested; that item is closed.

Grids include every guard boundary in the ported code (68 hand-picked
temperatures at and either side of 5/10/50/100/194/250/511/700/1000/2000/
2090/3686/4000/7510/1e4/2e4/3.2e5/1.28e6/1e7/1e8 K, plus a 900-point log sweep
1–1e9 K, plus a 400-point *linear* sweep through the 1-K-bin CO/H2O table
region to exercise `int(temp)`), six abundance mixes × five `divv` × two `dl`,
seven densities, two `iflag_cmb`, and for `calc_photo` a 4×3×4×2×4×3 sweep over
G0 / DGR / column / iphoto / x(H2) / x(CO).

### Results (both nets identical)

```
PHASE A  cltab 670000/670000  chtab 1730000/1730000  dtcltab+dtchtab (scaled) 2400000/2400000
         crtab, crphot, phtab, cst                     exact
         co_* 6 tables + derivs, h2o_* 12 tables + derivs, co/h2o vib
         h2_opac, CO_self_shielding, CO_H2_shielding
         temp_table, gamma_table, eh2_table, wss_{C,N,O,Ne,Mg,Si,S,Ca,Fe},
         HeI_tbl, HeII_tbl                             ALL PASS
PHASE B  cool_func ylam            383880/383880   PASS   worst rel = 0.000e+00
         cool_func rates[28]     10748640/10748640 PASS   worst rel = 0.000e+00
         cool_func tdust out       383880/383880   PASS   worst rel = 0.000e+00
         calc_dust_temp tdust/RH2/Rdust  3x117600  PASS   worst rel = 0.000e+00
         calc_H2_shielding          14796/14796    PASS   worst rel = 0.000e+00
         lookup_temp_gamma T,gamma  2x40010        PASS   worst rel = 0.000e+00
         calc_spec / compute_gamma  20001 / 22011  PASS   worst rel = 0.000e+00
         calc_photo phrates       1430784/1430784  PASS   worst rel = 9.922e-16
PHASE C  same, on C++-built tables                 PASS   worst rel = 4.245e-13
PHASE D  cltab (iflag_atom=1)      670000/670000   PASS   worst rel = 3.282e-14
TOTAL    38,918,708 comparisons, 0 failures, worst 4.245e-13
```

`cool_func` is **bit-exact** against the Fortran over 10.7 million rate
comparisons when both read the same tables. That is the strongest statement
available and it is what the port was aiming for.

The finite-difference derivative tables (`dtcltab`, `dtchtab`, `dT*`) are
compared against the **scale of the operands being differenced**, not against
the difference itself. Reason, stated plainly so it is not mistaken for a
loosened tolerance: `dT(i,j) = v(i,j+1) − v(i,j)` where the two `v` agree to
<1e-15 relative but differ from each other by ~1e-9 of their own magnitude, so
the *difference* inherits ~1e-6 relative error by cancellation alone. Testing
the difference against `|v|` asks the right question — "is the derivative table
consistent with the table it derives from, to 1e-12?" — and the answer is yes
(worst 1.8e-14). The raw-relative numbers, for the record, are 3.3e-8 worst,
and every one of those points has `|dT| / |v| < 1e-8`.

---

## 3. Zero Fortran — checked, not asserted

`golden/build_purecxx.sh` compiles the whole path and links it with **`g++`**,
no `.F` object anywhere on the command line and no `-lgfortran`. It then runs
`nm` and `ldd` over the result and fails if any Fortran-mangled symbol, any
`_gfortran_*` reference, or any libgfortran linkage is present.

```
===== net 5 : Fortran-symbol audit =====
PASS -- zero Fortran symbols, zero libgfortran linkage
    libstdc++.so.6  libm.so.6  libgomp.so.1  libgcc_s.so.1  libc.so.6  libdl  libpthread
===== net 17 : ... identical
```

The binary is not a stub — it builds the tables from the same data files
(`wss_z_collis.txt`, `H2-cooling-ratios.dat`, `energy-table.txt`,
`eh2-table.txt`) and runs eight one-zone relaxations per net:

| case | net 5 T_final | net 17 T_final |
|---|---|---|
| T0=1e4 nH=1 | 7661.5 | 7760.45 |
| T0=1e3 nH=1e2 | 69.583 | 71.2572 |
| T0=1e2 nH=1e4 | 22.5812 | 22.5089 |
| T0=3e5 nH=1e-2 | 11275.3 | 11291 |
| T0=50 nH=1e6 | 5.82631 | 5.90768 |
| T0=1e6 nH=1 | 7673.58 | 7764.73 |
| T0=5e3 nH=1e3 G0=10 | 71.7382 | 76.215 |
| T0=2e4 nH=10 ζ=3e-16 | 1949.89 | 2712.99 |

### Fortran symbols the C++ path links against — before and after

Measured with `nm` over the C++ object set (net 17; net 5 identical).

| | Gate 2 (before) | Gate 3 (after) |
|---|---|---|
| Fortran subroutines referenced | `cool_func_`, `calc_dust_temp_`, `calc_h2_shielding_factor_`, `lookup_temp_gamma_` | **none** |
| Fortran COMMON blocks referenced | `coolr_`, `cooli_`, `chem_comp_`, `current_photo_`, `project_`, `co_data_`, `h2o_data_`, `h2_opac_data_`, `co_shield_`, `gamma_data_` | **none** |
| total | **14** | **0** |

At the driver level the eight Fortran entry points of the Gate-0 seam
(`chem_interface.h:113-120`) now all have C++ equivalents:

| Fortran | C++ |
|---|---|
| `COOLINMO`, `CHEMINMO`, `PHOTOINIT`, `CONST_RATES` | `chem_tables_build()` |
| `LOAD_H2_TABLE` | `read_h2_opac()` in `chem_tables_build.cc` |
| `INIT_TEMPERATURE_LOOKUP` | `read_temp_gamma()` in `chem_tables_build.cc` |
| `INIT_TOLERANCES` | `chem_atol()` in `cvode_driver.cc` (Gate 2) |
| `CALC_TEMP` | `cxx_lookup_temp_gamma()` / `chemcxx_pure_calc_temp()` |
| `CALC_PHOTO_WRAPPER` | `cxx_calc_photo()` |
| `EVOLVE_ABUNDANCES` | `cxx_evolve_abundances()` (Gate 2) |

**Caveat, stated because it matters.** A *GIZMO* build with `SGCHEM_CXX` ON
still links the Fortran chemistry objects. That is not leftover porting: (1)
the Fortran DVODE path is deliberately retained as the fallback, and (2)
`do_sgchem_step()` still uses `chem_commit_to_fortran()` + `COOLINMO` /
`CHEMINMO` / `CALC_PHOTO_WRAPPER` as the **input transport** — the C++ side
snapshots `/coolr/` and `/current_photo/` into `ChemCell`. Closing (2) is the
remaining Gate-4 step and is short: fill `ChemCell` directly from
`struct ChemInputs` and call `cxx_calc_photo()`, i.e. route
`chem_gizmo_backend.cc` at `chem_backend_cxx.cc` instead of
`chem_backend_c_api.cc`. Everything that needs is already written and tested.
Verified today: `make CONFIG=Config_cxx.sh EXEC=GIZMO_cxx` links clean with all
Gate-3 objects included.

---

## 4. Audit R2 — OpenMP safety

**The C++ path is now OpenMP-safe.** Three changes make it so:

1. `ChemTables` is written once by `chem_cxx_tables_init()` and is `const`
   everywhere afterwards → safe to share.
2. `ChemCell` is per-cell and passed by reference to every kernel; nothing
   below the RHS reads a global any more. The plain-C convenience API keeps one
   `thread_local` cell.
3. The two remaining mutable process-globals — the CVODE workspace cache
   (`cvode_driver.cc`) and the solver statistics counters — are now
   `thread_local` (DEVIATIONS D-15).

Tested, not asserted: `test_purecxx.cc` runs the eight cases above 8× each,
64 solves, `#pragma omp parallel for schedule(dynamic)` on 8 threads, and
requires the results to be **bit-identical** to the serial run.

```
OpenMP: 8 threads, 64 concurrent solves, bit-identical to serial: YES (0 mismatches)   [net 5]
OpenMP: 8 threads, 64 concurrent solves, bit-identical to serial: YES (0 mismatches)   [net 17]
```

Enabling `OPENMP` together with `SGCHEM_CXX` in GIZMO still has to wait — not
for any porting, but for the Gate-4 input-transport step above, because
`chem_commit_to_fortran()` writes a shared COMMON.

## 5. Audit R10 — state-consistent lambda diagnostics

**Applied.** The Fortran leaves `lambda`/`lambda_chem` at whatever the *last*
RHS evaluation wrote. With a multistep integrator that evaluation sits at an
internal stage or, worse, at a **rejected** trial point — so the published
cooling-rate breakdown, which is what lands in snapshots and in every "why did
this cell cool" post-mortem, does not describe the state that was returned.

The driver now performs one extra RHS evaluation at the accepted `(t_end, y)`
before returning. Cost: one RHS per cell call against the ~240 a typical call
already makes (~0.4%). `CHEMCXX_LEGACY_LAMBDA=1` restores the Fortran
semantics for a bit-comparable A/B. This changes *diagnostics only* — the
trajectory is untouched, which the acceptance numbers below confirm.

---

## 6. Gate-2 acceptance, re-run on the pure-C++ backend

Same three gates, same driver, same criteria. The `cxx` backend now differs
from `f77` in **every** layer: tables, leaf kernels, RHS, integrator.

### (a) equilibrium — 20 ladder points × 2 nets, GATE |ΔT_eq| ≤ 1%

| net | points | max ΔT/T | median | >1% | non-OK |
|---|---|---|---|---|---|
| 5 | 20 | 3.25e-3 | 0 | 0 | 0 |
| 17 | 20 | 9.48e-3 | 2.2e-7 | 0 | 0 |

**PASS** (40/40 within 1%). Gate 2 was 7.18e-3 / 6.71e-3.

Net 17's worst point, 9.48e-3, is `nH = 1e6, Z = 0.1` — T_eq ≈ 5.6 K, the
gas-grain-locked dense-cold regime where T_gas is slaved to T_dust and T_dust
comes from a Newton solve whose own convergence criterion is
`|Δtdust/tdust| < 1e-2` (`cool_util.F:972`). A ~1% spread there is the dust
solver's own tolerance, not a port deviation: `calc_dust_temp` is bit-exact in
the golden (worst rel 0.000e+00 over 117600 points). Two backends that make
different numbers of RHS calls stop at different points inside that 1% band.
This is a **pre-existing property of the algorithm** and is worth flagging to
Uli separately from the port.

### (b) relaxation — fig6 cases × 2 nets, 2400 compared trajectory points

max pointwise |ΔT|/T = **8.76e-4 (0.088%)**. Gate 2 was 3.52e-3 (0.352%).
Four of the sixteen net-5 case/scale combinations are **exactly zero**.

### (c) certification — 80-zone (n, T0) grid per net, GATE zero failures

| net | zones | DVODE complete | CVODE complete | T_eq max Δ | >1% |
|---|---|---|---|---|---|
| 5 | 80 | 80/80 | 80/80 | 6.59e-3 | 0 |
| 17 | 80 | 80/80 | 80/80 | 5.41e-3 | 0 |

**PASS** — 160/160 zones complete, zero failures on either backend.

#### A real finding this gate produced

The first Gate-3 run of (c) showed **3 zones per net above 1%**, all at
nH = 1e6. Rather than accept it, it was bisected: `CHEMCXX_TABLES=fortran` gave
the same answer (so not the tables), `CHEMCXX_LEGACY_LAMBDA=1` gave the same
answer (so not R10), and the result was independent of the step count (so not a
transient). The cause was a **harness plumbing gap**: `tdust` is a *carried*
per-cell state. In GIZMO it lives in `CellP[i].DustTemp` and
`chem_interface.cc` seeds and reads it every step. The one-zone harness sets
`COOLR.tdust = 10 K` **once** and never writes the C++ result back — so with
the Gate-3 backend owning `tdust` in `ChemCell`, the C++ side restarted its dust
Newton from 10 K on every hydro step while the Fortran carried its global
forward. (At Gate 2 this could not happen: the leaf *shim* pushed `c.tdust`
through `COOLR.tdust`, so both backends shared the global.) Carrying `tdust`
across `chemcxx_init()` in the harness bridge moved those zones from
1.1–1.3% to <1e-3 and (c) to `>1%: 0`. Production is unaffected — the caller
carries `tdust` there.

### wall time (`gate2_timing.py`, dedicated workload)

| net | DVODE total | CVODE total | ×  |
|---|---|---|---|
| 5 | 0.1939 s | 0.2358 s | 0.82 |
| 17 | 0.4621 s | 0.4436 s | 1.04 |

Per-regime the ratio runs 0.81–1.74. The two integrators remain within a factor
~1 of each other; the C++ path is now doing strictly more work than at Gate 2
(its own table lookups, plus the R10 evaluation) and has not lost ground.

---

## 7. Six bugs the goldens found

Five are the DEVIATIONS D-10 class — Fortran literals without a `D` exponent
are DEFAULT REAL and must be reproduced as float32, not as the printed decimal.
Each was localised by the golden before being fixed; full table in
DEVIATIONS D-16.

| where | literal | golden signature |
|---|---|---|
| `wss_z_collis.F:91` + 9 solar abundances | `lT_range = 6.98185`, `2.07e-3`, … | **rates(3): 0/383880 passing**, worst 5.0e-6 |
| `wss_z_collis.F:45` | `fac = 1e-20` clamp | wss tables: **exactly 59/352 rows** failing — precisely the 59 rows with T < 1473.6 K, i.e. the floored ones |
| `cool_util.F:857` | `chi_isrf / 1.7` | `calc_dust_temp` tdust, 7.8e-9 |
| `cool_util.F:539` | `(logT − 1.5) / 0.03` | H2 opacity index |
| `cool_func.F:875` | `PEvar0**0.73` | photoelectric heating |
| `calc_temp.F:185-186` | `abh2 / 0.1` **and** `0.1 * idx_h2` | `lookup_temp_gamma`, exactly **one** abh2 value (0.4) failing |

The `calc_temp.F` case is two traps in one line and is the nicest demonstration
that the goldens were actually diagnostic rather than decorative. Promoting the
float32 divisor fixed 9 of 10 test values. The tenth, `abh2 = 0.4`, needed the
second observation: `0.1 * idx_h2` is REAL(4) × INTEGER, so the **product** is
rounded back to float32 before the REAL(8) subtraction — visible only when
`float32(0.1) × idx_h2` is not exactly representable, which on this grid is
`idx_h2 = 3` alone. The per-`abh2` breakdown the golden prints named the value
before the cause was known.

**The sixth is a build bug, and it would have been silent.** The C++ TUs use
GIZMO's own `#include "../../../GIZMO_config.h"` convention, i.e. the config at
the *root of the tree being compiled*. The harness built net 5 and net 17 out of
one rsync'd tree whose root config was the canonical one (net 17), so the
"net 5" objects were compiled with net-17 semantics: `#if CHEMISTRYNETWORK == 5`
was false and `phrates(10)` — CO photodissociation — was **never assigned**.
The golden reported 19,872 mismatches with `F = 1.235e-13, C = 0`. Fixed by
copying the per-net config to the build-tree root before each net's compile, in
all three build scripts. Any future harness that builds more than one network
out of one tree must do the same.

---

## 8. Deviations introduced by Gate 3

All in `DEVIATIONS.md`: D-13 (`co_cool` indexes the vibrational table with the
rotational column — a latent Fortran bug, **reproduced literally**, flagged for
a separate physics decision), D-14 (`calc_photo` ported for nets 5/17 only),
D-15 (thread_local state), D-16 (the six bugs above), D-17 (leaf ABORTs become
recoverable returns, as D-3), D-18 (what was deliberately not ported and why).
D-0's Gate-2 scope caveat is marked superseded.

---

## 9. Reproduce

```bash
CX=/orion/ptmp/uli/orion/test/gizmo/cooling/sgchem/cxx
bash $CX/golden/build_gate3.sh          # kernel + table goldens (phases A-D)
bash $CX/golden/build_purecxx.sh        # Fortran-free link audit + OpenMP test
bash $CX/golden/build_gate2_rerun.sh    # dual-backend onezone3 with the Gate-3 backend
cd /orion/ptmp/uli/orion/fb_tests/_sgchem_cxx_build
python3 $CX/gate2_bench.py all && python3 $CX/gate2_timing.py && python3 $CX/gate2_summary.py
```

Artefacts: `fb_tests/_sgchem_cxx_build/gate3/gate3_golden_net{5,17}.txt`,
`purecxx/purecxx_net{5,17}.txt`, `gate3_summary.txt`, `gate3_timing.log`,
`gate2_data/*.csv`.
