# CHEMCOOL C++/CVODE port — conscious deviations register

Every place the port does something the canonical Fortran does not, with the
reason. Anything NOT listed here is intended to be behaviour-identical.
Reference: `stella/CHEMCOOL_CONTROL_FLOW_AUDIT.md` §7 (R1–R10) and
`stella/CHEMICAL_MODEL_NETS_5_17.md`.

Canonical source of truth: `/orion/ptmp/uli/orion/test/gizmo` (branch
`uli-singlestar-galaxy`). **Zero edits** were made there at any point.

---

## D-0. SCOPE (SUPERSEDED BY GATE 3 — kept for the record)

**What.** The C++ side implements: `evolve_abundances.F` (driver loop, TNOCHEM
bypass, `check_abundances`, `init_tolerances`) → `cvode_driver.cc`;
`rate_eq_simple.F` / `rate_eq_ulli.F` (RHS assembly + `comp_equilib`) →
`chem_rhs_net5.cc` / `chem_rhs_net17.cc`; `validate_rates.F` →
`chem_rhs_common.h`; `spline.F` + the cl1/cl2/cl3/cl6 table construction →
`chem_spline.cc`; the 346 analytic rate coefficients → `rates_gen.cc` (Gate 1).

Everything strictly BELOW the RHS is still the canonical Fortran, reached
through thin shims in `chem_backend_fortran.cc`:
`cool_func`, `calc_dust_temp`, `calc_H2_shielding_factor`,
`lookup_temp_gamma`, and the table builders `coolinmo`/`cheminmo`.

**Why.** This is what makes the Gate-2 benchmark meaningful. Both backends run
inside the *same binary*, link the *same* `coolinmo`-built `/coolr/` tables and
call the *same* cooling function; the only things that differ are the
integrator, the RHS assembly and the driver policy. Any deviation observed in
GATE2_ACCEPTANCE.md is therefore attributable to the ported layer and nothing
else. Porting `cool_func` (1329 lines) in the same step would have confounded
solver deviations with cooling-function deviations.

**STATUS 2026-07-29: Gate 3 closed this.** `cool_func`, `calc_dust_temp`,
`calc_H2_shielding_factor`, `lookup_temp_gamma`, `calc_photo`, `calc_temp`,
`compute_gamma` and the table builders `coolinmo`/`cheminmo`/`photoinit`/
`const_rates`/`read_wss_collis_table`/`load_H2_table`/`init_temperature_lookup`
are all C++ now (`chem_cool_func.cc`, `chem_cool_util.cc`, `chem_calc_photo.cc`,
`chem_calc_temp.cc`, `chem_tables_build.cc`). `chem_backend_fortran.cc` is
retained ONLY as the golden harness's reference side (`fort_*` wrappers).
`golden/build_purecxx.sh` links the path with g++, no Fortran object and no
`-lgfortran`, and audits the binary with `nm`. The paragraph below describes
the Gate-2 state.

**Consequence / what Gate 3 had to do.** The shims write `COOLR.tdust` before each
leaf call and read it back after (`tdust` is a genuine in/out algebraic state,
audit §3.1 item 2). The leaves still read per-cell inputs from COMMON, so the
Gate-2 backend inherits the Fortran's thread-unsafety (audit R2). One-zone has
exactly one cell, so nothing can leak; production use requires Gate 3.

---

## D-1. (R3) `validate_input` no longer mutates the integrator's state vector

**Fortran** (`validate_rates.F:35-72`): writes into `y` in place — negatives →
`0d0`, non-H2 species > 1 → 1, H2 > 0.5 → 0.5 — while `y` is DVODE's own state
echo. Legal-ish for DVODE, explicitly forbidden by CVODE.

**Port** (`chem_rhs_common.h:chem_validate_input`): `y` is `const` in the RHS
signature; the identical floors are applied to a local copy and the RHS is
evaluated from the copy. **Values and ordering of the floors are unchanged**
(neg→0, cap 1 / cap 0.5, `EPS_MAX = 1e-4` reject threshold) — they are
load-bearing stiffness control, not cosmetics.

**Signalling change.** A gross violation (NaN, or excess > `EPS_MAX`) returns
`CHEM_RHS_RECOVERABLE` (positive) instead of setting `ipar(1)=1` and returning
`ydot = 0`. CVODE reacts by cutting its internal step and retrying; if it cannot
recover it returns a negative flag and the driver's restore/halve/re-init path
runs — i.e. the Fortran's outcome is still reachable, with an extra
recovery layer in front of it. This makes the port *more* robust, not less;
it is the one place where the port can succeed where DVODE would have halved.

## D-2. `temp <= 0` poisoned-input path returns immediately

**Fortran** (`rate_eq_simple.F:150-168`, `rate_eq_ulli.F:230-248`): sets
`ipar(1)=1` AND continues to build a full `ydot` evaluated at `tmin`; the
substep is rejected only after `dvode` returns.

**Port:** returns `CHEM_RHS_RECOVERABLE` at that point. The `ydot` computed
at `tmin` in the Fortran is discarded by the rejection anyway, so the accepted
trajectory is unaffected; only the wasted work differs.

## D-3. net-5 `validate_output` NaN-ydot ABORT(1) → recoverable return

**Fortran:** net 5 aborts the whole run on a NaN `ydot`
(`validate_rates.F:100-115`); net 17 prints, sets `ipar(1)=1`, zeroes `ydot`
and returns (`rate_eq_ulli.F:1014-1032`). The audit flags this as an
inconsistency (R7) and recommends unifying on the net-17 behaviour.

**Port:** unified on net-17 — both nets print loudly and return recoverable.
`DECIDE-WITH-ULI` item: this converts a hard abort into a step rejection on
net 5. It cannot mask a real problem silently (the print is unconditional and
a persistent NaN still collapses dt to 1e-8 s → return code 100).

## D-4. (R7 cleanup) the two bare Fortran `stop`s are not ported

`rate_eq_simple.F:505-511` (`temp < 0` debug) and `calc_photo.F:706` are bare
`stop`s that would kill one MPI rank and deadlock the job — the exact failure
mode the `ABORT` macro exists to prevent. Both are unreachable in practice
(step 5 clamps `temp<=0` to `tmin` first). Not ported.

## D-5. (R9) dead modes and dead flags dropped

Not ported: `do_chemcool_step` modes 1/2/3 (no callers; mode 3 returns a stale
energy), `ipar(2)` (write-only), `CHEM_EQ_H2` (contains an out-of-bounds `y(0)`
store), `column_est` / the `iphoto=3/4/5` potential-column paths (always 0),
`NO_CO_FORM`, `BS19_XH2`, `ADIABATIC_DENSITY_THRESHOLD`,
`CONST_NCRIT_UVPUMP`, `H2_FORM_TEST`. `pdv_term` is hard-zeroed per cell in
production (`chemcool.cc:721`) so net-5's `+ pdv_term` is not carried.

## D-6. `no_dchem` made explicit

Audit §3.1 item 4: for net 17 `no_dchem` is read at six sites and written by
nothing — the deuterium network is enabled purely by Fortran BSS zero-init.
`chem_bind_cell()` sets `c.no_dchem = 0` explicitly.

## D-7. Solver-workspace lifetime (fair timing, identical semantics)

DVODE reuses static `rwork/iwork` across cell calls and re-initialises with
`istate=1`. Allocating a fresh `SUNContext` + CVode memory per cell call would
charge the port a malloc/free cost DVODE never pays and would corrupt the
wall-time comparison. The port caches one workspace per `nspec`
(`cvode_driver.cc:CvWork`) and calls `CVodeReInit` — the exact `istate=1`
equivalent. Like the Fortran side, this is **not thread-safe**.

## D-8. `ChemTables` index layout corrected

The Gate-2 scaffold declared `chtab[NCHTAB][NMD]` and indexed
`chtab[k-1][itemp-1]`. `cool.h:244,255` declares `chtab(nchtab, nmd)` —
species index FAST — so the correct element is at `(itemp-1)*NCHTAB + (k-1)`;
the scaffold was transposed and would have read garbage rate coefficients.
`ChemTables` is now a view over the Fortran `/coolr/` arrays with the accessor
`chem_CH/chem_CL` indexing `[itemp-1][k-1]`. Fixed before any benchmark ran.

## D-9. The "8 TABLE splines" are 4 splines + 4 classification artifacts

`RATE_AUDIT.md` classified 8 `cltab` entries as `kind=TABLE` (cl1-3, cl6,
cl52, cl59, cl60, cl61) because the extractor saw each assigned both in
coolinmo's analytic temperature loop and in a later residual loop. Reading the
residual loop (`coolinmo.F:1520-1590`) resolves it:

| entry | verdict |
|---|---|
| cl1, cl2, cl3 | genuinely TABLE — analytic outside [1e2, 1e4] K, natural-cubic spline over the 41 `mol_data.h` points inside. **Ported** (`chem_spline.cc`). |
| cl6 | genuinely TABLE — `iflag_atom=2` (the production setting) splines the 76-point `ca2` DATA array, zero below 10^4.3 K. **Ported**. |
| cl52, cl59, cl60, cl61 | **not table-driven in the production config.** The residual loop zeroes them only under `iflag_atom==1`; under `iflag_atom==2` the zeroing is additionally `#ifndef WSS_CIE_COOL`, and `WSS_CIE_COOL` is ON in every Stella build. They keep their analytic main-loop values, which are already in `RATE_AUDIT.md` and covered by the Gate-1 rate goldens. Nothing to port. |

The claim in the last row is *tested*, not asserted: `golden/test_splines.cc`
reports how many grid points of each are nonzero after the real `COOLINMO` runs
(all four are nonzero over 5000-7500 of 10000 points ⇒ not zeroed ⇒ analytic).

**Spline semantics note.** `spline.F` evaluates the cubic in the *normalized*
local coordinate `dpt = (x-x_i)/(x_{i+1}-x_i)` while `spline_derivatives` solves
the *uniform-knot* Bartels tridiagonal system — the knot spacing never enters
the derivative solve. That is not a standard non-uniform natural cubic spline.
Both cl1-3 and cl6 have non-uniform knots, so a "corrected" spline would give
different numbers. The port reproduces the Fortran literally
(`chem_spline.cc:chem_spline_derivatives`), including this quirk.

## D-10. Physical constants are the float32-rounded values (**not** a cosmetic)

`chemcool_consts.h` spells its constants as `1.38066e-16`, `1.60219e-12`,
`1.6726e-24`, `6.672e-8` — Fortran **default REAL** (single-precision)
literals. `cool.h:204` does `REAL kboltz; parameter (kboltz = BOLTZMANN)`, and
the build carries no `-fdefault-real-8` (`tests/onezone/Makefile` FFLAGS:
`-O2 -cpp -ffixed-line-length-none -fno-second-underscore`). The literal is
therefore rounded to float32 and only then widened to `real*8`. What the
running Fortran actually holds is up to **5e-8 relative** away from the decimal:

| macro | decimal | value in the running Fortran | rel |
|---|---|---|---|
| `BOLTZMANN` | 1.38066e-16 | 1.3806599982553653e-16 | 1.26e-09 |
| `ELECTRON_VOLT` | 1.60219e-12 | 1.6021899518908755e-12 | 3.00e-08 |
| `PROTONMASS` | 1.6726e-24 | 1.6726000836652178e-24 | 5.00e-08 |
| `GRAVITY` | 6.672e-8 | 6.6719998415010195e-08 | 2.38e-08 |

The same applies to every bare-decimal `DATA` array (`h2_lte`, `h2_h_rate`,
`h2_h2_rate`, `coolatom`, `ca2`, …): each element is float32-rounded.

**Attribution.** The Gate-1 rate generator already got this right —
`rates_gen.cc:40-41` emits `1.38066e-16f` / `1.60219e-12f` with the comment
"default-real literal", which is why the 346 rate goldens passed at 1e-12. What
was *missed* was (a) the hand-written constants in `chem_types.h`, which used
the exact decimals, and (b) `tables_extract.py`, which parsed every `DATA`
literal as a Python double. Both are now fixed: `chem_types.h` uses the
float32-widened values, and `tables_extract.py` reproduces the compiler's
literal-kind rule (a `D` exponent ⇒ double, everything else ⇒ float32).
**This is a deviation from the printed decimals in order to be faithful to the
compiled code.**

Caught by golden-testing the cl1-3 splines: the worst relerr was exactly
`float32(kboltz)`, 1.264e-09. Progression of the spline goldens:

| state | result | worst relerr |
|---|---|---|
| `DATA` as doubles, `kboltz` exact | 23102/40000 | 2.55e-06 |
| `DATA` float32-rounded, `kboltz` exact | 35227/40000 | 1.26e-09 |
| both faithful | **40000/40000** | **1.65e-14** (roundoff) |

Physically the effect is negligible. Operationally it is not: reading the
decimals would have "silently improved" the tables by ~2e-6 and left an
unexplained floor under every future A/B comparison.

## D-11. GIZMO integration: what was changed in the canonical tree

The port is compiled INTO GIZMO behind the Gate-0 seam, selected by the
`CHEMCOOL_CXX` compile flag (**default OFF**). Changes to canonical files:

| file | change |
|---|---|
| `cooling/chemcool/chemcool.cc` | the single `EVOLVE_ABUNDANCES(...)` call is wrapped in `#ifdef CHEMCOOL_CXX` → `chem_evolve_cxx(...)`, `#else` → the Fortran call. Nothing else in the function changes. |
| `cooling/chemcool/chem_interface.h` | declares `chem_evolve_cxx` under the same guard. |
| `cooling/chemcool/cxx/` | new directory: the whole port. |
| `Template_Config.sh` | new documented `#CHEMCOOL_CXX` entry (commented out). |
| `Makefile` | new `ifeq (CHEMCOOL_CXX,...)` block inside the existing `CHEMCOOL` block: `SUNDIALS_DIR`/`SUNDIALS_INCL`/`SUNDIALS_LIBS`, `CHEMCXX_OBJS`, `INCL`. `$(SUNDIALS_INCL)` appended to `CFLAGS`, `$(SUNDIALS_LIBS)` to `LIBS`, `$(CHEMCXX_OBJS)` to both link rules and to `clean`. |

`CHEMCXX_OBJS` is a **wildcard** over `cooling/chemcool/cxx/*.cc`, not a
hand-maintained list. That is deliberate: this directory is under active
development (Gate 3 is porting the leaf kernels), and a stale explicit list does
not fail at parse time — it fails at LINK time with undefined `cxx_*`
references, which is a much worse error to debug. `INCL` is wildcarded over
`cxx/*.h` for the same reason. `clean` also removes `cooling/chemcool/cxx/*.o`
unconditionally, because `CHEMCXX_OBJS` expands to empty when the flag is off
and would otherwise leave objects from a flag-on build behind.

**With the flag OFF all of this is inert**: `SUNDIALS_INCL`/`SUNDIALS_LIBS`/
`CHEMCXX_OBJS` expand to empty, no cxx object is built, no SUNDIALS symbol is
linked, and `chemcool.cc` compiles the Fortran branch. Verified: the OFF build
links exactly the same object list as before the change.

**Why `CHEMCXX_OBJS` is a separate list with its own recipe.** SUNDIALS'
`sundials_context.hpp` uses `std::make_unique`, which needs C++14, while most
SYSTYPE blocks pin GIZMO to `-std=c++11`. Rather than silently promoting the
whole build to C++14 (which would happen if the flag were folded into `CFLAGS`,
since the last `-std` on the command line wins), the six port sources get their
own pattern rule that appends `-std=c++14 -I.` The rest of GIZMO keeps its
SYSTYPE standard. The extra `-I.` is needed because GIZMO's `CFLAGS` carries no
`-I.` and the port sources include `"GIZMO_config.h"` / `"cooling/chemcool/*.h"`
by tree-root-relative path (the same convention the one-zone build uses, so one
set of sources serves both).

**Failure funnel.** `chem_evolve_cxx` maps a nonzero driver return code onto
`chemcool_fatal_(code)` — the same `endrun(78100+code) → MPI_Abort` funnel and
the same code numbers as the Fortran `ABORT(...)` macro (audit R7), so existing
run-forensics knowledge transfers unchanged.

**Network guard.** `#error` if `CHEMISTRYNETWORK` is not 5 or 17 — the other
networks have no ported RHS and must build without the flag.

## D-12. Source-of-truth layout (single copy, no divergence)

The port lives at `cooling/chemcool/cxx/` inside the canonical tree, and the
one-zone benchmark compiles **those exact files** — `src/build3.sh` sets
`CXX_DIR` to `cooling/chemcool/cxx` inside the rsync'd build copy. There is no
second copy of the port and no benchmark-only variant of any source file. That
is what makes the Gate-2 numbers meaningful: what passes the benchmark is
literally what GIZMO compiles.

Per the standing rule, nothing is ever compiled inside `test/gizmo`: the tree is
rsync'd to `fb_tests/_chemcool_cxx_build/gizmo/` and all objects, binaries,
goldens, logs and CSVs live under `fb_tests/_chemcool_cxx_build/`.

## D-13. (Gate 3) `co_cool` indexes the vibrational table with the ROTATIONAL column

`cool_util.F:131` computes the vibrational column index as
`iN_co = int((10 * N_co_eff) - 129)` — using `N_co_eff`, the *rotational*
effective column, although the surrounding branch tests and interpolates with
`N_co_eff_vib`. In production the two are equal (`cool_func.F:471-473` sets
`N_co_eff_vib = N_co_eff` and the clamps are applied to each separately), so the
bug is latent, but it is not always so: the two clamps differ
(`maximum_CO_column_rot` = 20.0 vs `maximum_CO_column_vib` = 19.0), so a cell
whose rotational column is clamped and whose vibrational column is not will pick
a mismatched `iN_co`. **Reproduced literally** (`chem_cool_util.cc`), because
Gate 3's contract is bit-identity, not correction. Flag to Uli as a separate
physics decision.

## D-14. (Gate 3) `calc_photo` is ported for networks 5 and 17 only

`calc_photo.F` fills `phrates(11..54)` for networks 3/7/8/11/15 using dust
factors (`df09`, `df12`, `df17`, `df18`, `df21`, `df23`, `df24`, `df28`,
`df31`, `df37`, `df38`, `df39`, `dfsp1`) that are themselves only *computed*
under those networks' own `#if CHEMISTRYNETWORK` guards. Porting them would
mean carrying dead branches for networks Stella does not build. The C++ port
covers nets 1/2/4/5/6/17 exactly and prints a loud
`calc_photo: network N not ported` for the rest. Gate 4 must keep the Fortran
path available for those networks (which it does — `CHEMCOOL_CXX` is opt-in).

## D-15. (Gate 3, audit R2) shared mutable state is now thread_local

Two objects in the C++ path are mutable and were process-global at Gate 2:
the CVODE workspace cache (`cvode_driver.cc:g_work`, DEVIATIONS D-7) and the
solver statistics counters. Both are now `thread_local`. Consequences:

* an OpenMP loop over cells is race-free — proven, not asserted:
  `golden/test_purecxx.cc` runs 64 concurrent solves on 8 threads and requires
  the results to be **bit-identical** to the serial run;
* the cost is one CVODE workspace per (thread, network) — a few kB per thread;
* `chemcxx_stats()` now reports the **calling thread's** counters. Single-
  threaded runs (every harness here) are unchanged; a future OpenMP production
  path wanting global totals must reduce across threads.

The remaining process-global object, `ChemTables`, is written once in
`chem_cxx_tables_init()` and is `const` everywhere afterwards, so it is safe to
share. `ChemCell` is per-cell and passed by reference to every kernel; the
plain-C convenience API in `chem_backend_cxx.cc` keeps one per thread.

## D-16. (Gate 3) five more default-REAL literals — and one build bug — found by the goldens

D-10 established that Fortran literals *without* a `D` exponent are DEFAULT REAL
(float32) and must be reproduced as such. Gate 3's kernel goldens found five
more instances, each of which the golden localised precisely before it was
fixed. They are listed here because each one is a place where "reading the
printed decimal" silently changes an answer:

| where | literal | effect if read as an exact double | golden signature |
|---|---|---|---|
| `cool_func.F:875` | `PEvar0**0.73` | 0.73 → 0.7300000190734863 | photoelectric heating |
| `wss_z_collis.F:91` + the nine solar reference abundances | `lT_range = 6.98185`, `2.07e-3`, … | interpolation index shifts | **rates(3): 0/383880 passing**, worst 5.0e-6 |
| `wss_z_collis.F:45-47` | `fac = 1e-20` clamp | log10 shifted by 1.378e-8 on every floored row | wss_* tables: exactly 59/352 rows failing (the 59 rows with T < 1473.6 K, i.e. the floored ones) |
| `cool_util.F:857` | `chi_isrf / 1.7` | ISRF dust heating off by 2.8e-8 | `calc_dust_temp` tdust, 7.8e-9 |
| `cool_util.F:539` | `(logT - 1.5) / 0.03` | H2 opacity index | (folded into the above) |
| `calc_temp.F:185-186` | `abh2 / 0.1` **and** `0.1 * idx_h2` | see below | `lookup_temp_gamma`, 1.45e-8 |

The `calc_temp.F` case is worth spelling out because it is two distinct traps in
one line. `abh2 / 0.1` promotes the float32 literal to double — that fixed 9 of
the 10 test values. `0.1 * idx_h2` is **REAL(4) × INTEGER**, so the *product* is
rounded back to float32 before the REAL(8) subtraction; that extra rounding is
visible only when `float32(0.1) * idx_h2` is not exactly representable, which on
the test grid is `idx_h2 = 3` alone — and `abh2 = 0.4` was precisely the one
value still failing. Diagnosis and fix confirmed by the per-`abh2` breakdown the
golden prints.

**The build bug.** The C++ TUs use GIZMO's own convention,
`#include "../../../GIZMO_config.h"`, i.e. the config at the *root of the tree
being compiled*. The Gate-3 harness compiled net 5 and net 17 out of one rsync'd
tree whose root config was the canonical one (net 17), so the "net 5" objects
were built with net-17 semantics: `#if CHEMISTRYNETWORK == 5` was false and
`phrates(10)` (CO photodissociation) was never assigned. The golden caught it as
19872 mismatches with `F = 1.235e-13, C = 0`. Fixed by copying the per-net
`GIZMO_config.h` to the build-tree root before each net's compile
(`golden/build_gate3.sh`, `golden/build_gate2_rerun.sh`,
`golden/build_purecxx.sh`). **Any future harness that builds more than one
network out of one tree must do the same.**

## D-17. (Gate 3) leaf-kernel ABORTs become recoverable returns

`cool_func`'s two `ABORT` cases (bad temperature index; `abh2 > 0.5 + eps`) and
`calc_dust_temp`'s `ABORT(10)` (Newton budget exhausted) are, in the port,
loud prints plus a flag on `ChemCell` (`cool_func_failed`, `dust_iter_failed`)
and a nonzero return, which the RHS turns into `CHEM_RHS_RECOVERABLE`. This is
the same treatment as D-3 and for the same reason: a hard abort inside an RHS
evaluation kills the job on what may be a *rejected trial step*. A persistent
failure still collapses `dt` to 1e-8 s and returns code 100, so nothing is
masked. `cool_func`'s `temp < 0` path keeps the Fortran behaviour exactly —
print and return **without touching** `ylam` or `rates`.

## D-18. (Gate 3) not ported from `cool_util.F` / `calc_photo.F`

* `compute_h2_heating_fraction` (and the `h2heat.h` table): originally no live
  caller in nets 5 or 17. **Now ported** (D-22) and live in every network:
  H2 formation heating is (h2_form_kin + f_ex * h2_form_ex) * 4.48 eV.
* `dma_calc`: guarded by `#if defined DMA`, which no Stella build sets, and its
  body returns hard zeros for both outputs anyway.
* `chi`'s `index == 200` branch is a bare Fortran `stop` (D-4 class). The port
  keeps the print and returns the AV>200 asymptote instead of killing the rank.
  Unreachable: `AV > 200` already returned above it.
* `column_est` / `iphoto = 3/4/5`: the code paths are ported (they are three
  lines) but `column_est` is hard-zero in production, as D-5 records.

---

## D-22. Network 15 (NL99, `rate_eq_nl99.F`) — `chem_rhs_net15.cc`

Ported 2026-10-19 with the same contract as nets 5/17 (calc_photo stays the
Fortran `CALC_PHOTO_WRAPPER` in GIZMO, D-14). Driver: Glover's step control
(safety/used for H2, C+, CO, energy only), net-15 pool floors, atol, max
abundances and the hot-gas state (H+=1, C+=abundc, He+=abhe, M+=abundM,
molecules and CHx/OHx/HCO+ zero) from `evolve_abundances.F`.
`compute_h2_heating_fraction` + `h2_rv_heat` (machine-extracted by
`tables_extract.py`, float32 literals) ported into `chem_cool_util.cc`.

Differences from nets 5/17 that the port reproduces, not "fixes":
* `validate_input` does not short-circuit `rate_eq_nl99`: ydot is built from
  the clipped copy even when the input is flagged (and for `temp <= 0`, at
  `tmin`). The port builds the full ydot too and returns
  `CHEM_RHS_RECOVERABLE` at the end, so the driver's `dt_est` sees the same
  ydot as the Fortran. Clipping continues past an excess flag, as in
  `validate_rates.F` (`chem_validate_input_full`).
* `calc_H2_shielding_factor` is called for every `iphoto` (no `iphoto==0 →
  fshield=1` shortcut), `NH = 0.5*yn*dl`, grain recombination uses `phi < 1`
  and no `exp(-T/3e4)` factor, no `T < 100 K` zeroing of rates_chem(6).
* Default-REAL literals: `PI` in L_jeans (3.1415927410125732) and the `1.4`
  in the UV-pumping `cdex` are float32, as compiled (D-10 class); nets 5/17
  now use the same values (`CHEM_PI`) and `(yn*yn)` for `yn**2`.

Fixed in Fortran and C++ together: `rate_eq_nl99.F` now publishes
`lambda_chem(1..6)`; neutral carbon no longer subtracts OHx; the bilinear
branch of `compute_h2_heating_fraction` weighted density by dT and
temperature by dn (table stride: +1 density, +29 temperature).

Deviations (same classes as D-1/D-3): a NaN input species returns at once
with ydot = 0 (Fortran continues with NaN); a NaN ydot returns recoverable
(net 15 has no `validate_output`); `validate_input` acts on a copy, so the
driver's own `y` is not clipped in place.

One-zone A/B through GIZMO's cooling path (2026-10-19, 15 cells, 400 steps):
nH <= 100 at T0 <= 8000 K agree to <= 4.4e-9 in T; nH = 1e3-1e4 differ by up
to 7.8e-5, except the fast-cooling starts (1e4/8000 K: 3.2e-3; T0 = 1e6: up to
3.5e-2 mid-trajectory, final T <= 1.6e-4; net 5 shows 1.9e-2 / 1.1e-4 there).
That is integrator
truncation: only H2/C+/CO/energy limit the outer substep in net 15, and each
backend moves 1e-3..2e-2 in T between RTOL 1e-4 and 1e-8, while at RTOL 1e-8
the F-vs-C gap falls to <= 5.6e-6 in final T (1000/100: 7.1e-5 -> 6.8e-9).
Removing the Fortran in-place clip (D-1) changed only three hot cells.

## D-23. (Gate 4) direct path: entry temperature uses this cell's abhe
With `GALSF_CHEMCOOL_VARIABLE_XH_AND_ABHE`, the Fortran entry `CALC_TEMP` (chem_fill_inputs) runs before
the commit and reads the previous cell's `CHEM_COMP.abhe`; `CHEMCOOL_CXX` passes `in->abhe`.
Identical without variable composition. Skipped (PHOTOION-ionized) cells still report the last solved
cell's lambda, as the Fortran path does.

## D-24. (Gate 4) calc_photo, net 15 and RT overrides
Net 15 fills only phrates 48-51 beyond the shared set (rate_eq_nl99 reads 1, 2, 10, 48-51); the
Fortran also fills 11-37 and 52-54 from df factors never assigned for net 15. df17 = exp(-1.5 AV)
for net 15 as in calc_photo.F. Fixed: the TreeCol pixel loop never accumulated df05/df19 (net-17
phrates 4, 5, 8, 9 were zero in TreeCol mode), and the RADTRANSFER/TREE_RAY overrides did not reset
df05/df19.

## D-25. Network 16 (Gong+17, `rate_eq_gong17.F`) — `chem_rhs_net16.cc`
Ported 2026-10-21 with the net-15 contract (same slots, M = Si; driver policy, pools, atol, max
abundances and hot-gas state shared via `chem_net_nl99()`, as `evolve_abundances.F` shares them).
calc_photo: df187/df418/df388/df376/df261/df212/df266 in both AV branches and the RADTRANSFER /
TREE_RAY overrides; phrates 1, 2, 3, 10, 48, 49 only (the Fortran jumps to 212, so the M1 `rt_*`
additions are not applied for net 16 either); G_dust and the CR_ATTEN_K23 tau use df187.
cool_func: net 16 always takes the Gong+17/Kim+23 CR-heating branch. Tables: chtab 174-181,
crphot 13, cst 92-94 and the net-16 variants of ch40, ch147, crphot 1/11, cst 85 come from
`extract_rates.py` (`rate16_*`, `rates_registry16.inc`); phtab 2/48/49/50 in `build_phtab`.
`CHEM_NCHTAB/NCRPHOT/NCONST` follow CHEMISTRYNETWORK (chem_types.h now includes GIZMO_config.h).

Same deviation classes as nets 5/17: `validate_input` on a copy, a flagged input returns ydot = 0
and recoverable (the Fortran returns zero ydot); temp <= 0 builds ydot at tmin and returns
recoverable; NaN ydot returns recoverable (no `validate_output` in the Fortran). Not ported:
THERMAL_INFO_DUMP diagnostics.

Goldens (`golden/build_onezone_ab.sh`, nets 16 and 15): tables and leaf kernels incl. calc_photo
at 1e-12; RHS ydot vs `rate_eq` at <= 4e-15 over 3960 states; one-zone A/B (87 points, nH 0.1-1e5,
G0 0.1-17, Z 0.1-1, atomic and molecular starts, 300 Myr) final T within 2.3e-3, trajectory 3.6e-3.

Also fixed: `chem_backend_fortran.cc` declared /gamma_data/ with the pre-netab_x size (250000),
so the Fortran-bound gamma/eh2 view (goldens, `CHEMCXX_TABLES=fortran`) was misaligned.

## D-26. (2026-10-21) element/charge conservation fixes, both backends (Uli ruling: fix all)
- nets 15/16: free H excludes HCO+ and 3 H3+ (floored at 0, H3+ NOT capped: capping zeroed H3+ in fully
  molecular gas and cut H3+ + O -> CO; residual H non-conservation <= 3 x_H3+ ~ 1e-8); net 15 free O excludes HCO+.
- nets 15/16: He+ + H2 H2 loss = 2 f_H3+ cst75 (H2+ either returns to H2 or takes a second H2 into H3+).
- net 16: O+ capped at the free O pool; HCO+ + e 8% branch -> OH + C (Geppert+05), was CHx.
- net 17: free H excludes HD; H2 photoionization feeds H2+; HeH+ formation debits H+.
- net 3 (Fortran only): O + H- -> OH + e added to the H- destruction rate.
- drivers (evolve_abundances.F clamp_pool_group, cvode_driver.cc chem_clamp_pool_group) and the solver-entry
  clamp (chemcool_network.h): over-subscribed element groups scaled down proportionally, H last; the excess
  returns to the derived neutral, so element totals are exact. Ionized-cell override uses the cell's abhe.

## D-27. (2026-10-21) networks 5 and 17 aligned, both backends (Uli-approved set)
- net 17: H2 self-shielding local column `NH = dl*yn` (was `0.5*dl*yn`); the ionizing-photon test keeps `0.5*dl*yn`.
- net 17: cool_func gets Si+ = abundsi (slot 9) and HD (slot 13).
- O I for cool_func: `(abundo - CO) * (1 - x_H+)` in both nets; option `CHEMCOOL_NET5_OXYGEN_IONIZED` removed.
- net 17: H recombination cooling zeroed below 100 K, as net 5.
- net 17: ABHE 0.1 and HYDROGEN_MASSFRAC 0.7065 (net 1 keeps 0.079 / 0.76); `chem_cxx_cell_defaults` abhe 0.1.
- cool_func: WSS_CIE_COOL He-state override only for nets 4, 5, 6, 9 (no He species) and 15 (He+ tracked but
  not passed to cool_func); nets passing He states (1, 2, 3, 7, 8, 11, 16, 17) keep their own.
- net 5: `cs2 < 0 -> 0` guard in the Jeans length.
- net 5 (shared code, also nets 4/6): H2 collisional dissociation as net 17 (MSM96 CID + DT by H, He term
  `cdh2_he` with He neutral, `ncrinv = ch5 + abhe*ch139`), incl. the 4.48 eV cooling; UV-pumping factor 6.94 (was 8.5).
- Not changed: net 5 CR destruction of H2 (3 channels, 1.257 zeta); net 17's H2+ route has no exact net-5 equivalent.

## D-28. (2026-10-21) Network 3 (Glover+10, `rate_eq_mol.F`) — `chem_rhs_net3.cc`
Ported with the net-15 contract; driver policy from `evolve_abundances.F` (step control on H2, H+ and energy
only, net-3 pool floors, atol, max abundances, pool projection, hot-gas state). As in the Fortran, the RHS calls
`calc_photo` at every evaluation (live abh2/abco, abhd = 0); calc_photo now fills df09-df39/dfsp1 (both AV
branches, RADTRANSFER/TREE_RAY overrides) and phrates 11-37, 52-54 for nets 3/7/8/11, plus 31/32 for net 3.
Reproduced, not fixed: `ykdis = phrates(1)` without H2 self-shielding (dtemp above tmax: D-38); the
14-species fixed point needs every species non-zero to converge early (otherwise 10 passes).
Deviations (same classes as D-25): flagged input returns ydot = 0 and recoverable; temp <= 0 builds ydot at
tmin and returns recoverable; NaN ydot returns recoverable. comp_equilib ABORT(9751/9752) and the bad-index
ABORT(1) stay fatal: the RHS returns CHEM_RHS_FATAL with `c.fatal_code`, the driver returns that code to the
chemcool_fatal path (a recoverable return made CVODE retry into dt collapse or minutes-long stalls). `CHEM_NSPEC_MAX` 20, `CHEM_NRATES_CHEM_MAX` 23.
Not gated: G0 = 0, where calc_photo returns before setting chi_isrf and rate_eq_mol passes it on uninitialized
to calc_dust_temp (measured: 484/700 G0 = 0 states differ in tdust-coupled terms, 0 at G0 > 0); the net-3
phase-R grids use G0 = 1e-4 instead.
Goldens (`golden/build_conservation.sh SRC OUT 3`): RHS vs `rate_eq` <= 6.7e-15 over 2071 states (359 states
ABORT identically in both); Gate-3 tables/kernels/calc_photo ALL PASS at 1e-12; one-zone A/B (87 points)
final T within 1.4e-3, trajectory 4.1e-3; conservation: identical failure sets (360/4200), no element
overdraft above 3.4e-16 after any call, trajectories within 5.1e-3. GIZMO r17 dense box, Fortran vs C++: InternalEnergy
median/p99 7.5e-7/1.9e-5, ChemcoolDustTemp 9.2e-5/9.9e-4, ChemcoolTracAbund 3.9e-6/6.3e-4.
## D-29. (2026-10-21) Network 20: TIGRESS-NCR thermochemistry (ApJS 264, 10; arXiv:2210.08024), C++ only
Files: `chem_cool_net20.cc` (physics), `chem_rhs_net20.cc`, `chem_net20.h`, `chem_net20_cie.h` (generated by
`make_cie_table.py`); hooks in calc_photo, cvode_driver, backend, gizmo backend, chemcool_network.h, chemcool.cc,
chem_interface; `#error` for net 20 without CHEMCOOL_CXX; Fortran gets only a `rate_eq` link stub that aborts.
- ODE: x_H2 (eq. 15-19), x_H+ (eqs. 20-21), e (eq. 56). Algebraic, solved inside every RHS: x_C+ (eq. 24 with
  W01c grain recombination; C+ + e from the reference implementation's fit), CO (eqs. 25-26), C/O by
  conservation, x_O+ = x_O,tot x_H+, x_e (eq. 27; MH+ per D11 eq. 16.25 x 2x_H2) or eq. 28 above 2e4 K.
  T and x_e are a joint fixed point: T = e / [1.5 k n_H (1 + abhe + x_e - x_H2)] (gamma = 5/3). H, C, O and charge
  close by construction. GIZMO TracAbund = {H2, H+, CO}; CO is the eq.-25 value written after each solve.
- Heating: PE eqs. 30-31 (psi + 50, G0 = 1.7 chi_FUV), CR eqs. 32-33, PI eqs. 34-35, H2 eqs. 36-39.
  Cooling eqs. 40-48: Lya eq. 42, ci 43, cd 44, H2 ro-vib (M21 fit), ff/rr (D11 eqs. 10.12/27.14/27.23),
  CII/CI/OI (G3 rates), CO (LVG tables cited via G3, eff. column n_CO/|dv/dr|), grain recombination
  (W01b Table 3, R_V=3.1, b_C=4, ISRF), gas-dust eq. 45 with T_d from eq. 46, nebular eq. 47, CIE eq. 48 via S(T).
  CIE He/metal cooling and electrons: GF12 ion-by-ion rates x the CIE ion fractions they reference, protosolar abundances.
- Radiation: chi_PE/chi_LW dust-attenuated per TreeCol pixel; chi_H2 (eqs. 51-53, b5 = 3) and chi_C (self + H2 cross
  shielding as in the G17 reference) are photon-weighted pixel sums, live in the RHS; CR attenuation eqs. 54-55.
- Coefficients the paper only cites (G3, W01b/c, D11 16.25, M21, H2 collisional
  dissociation, H collisional ionization, CO table, C shielding, CR q_H2 fit, C+ + e) were taken from the authors' public implementation (pyathena
  microphysics/cool.py, commit f188a84; Athena++ shielding.cpp).
- Choices where the text is not explicit: sigma_PE = 1.0e-21 Z'_d (CR section; confirmed by the PDR figure);
  sigma_LW = 2.0e-21 Z'_d (`c.n20_sig_lw`; inferred from the paper's PDR figure and reproduces B2/B3; the
  text's absorption value 1.5e-21 puts the CO transitions 1.3-1.5x deeper); zeta_pi,C^ISRF = 3.5e-10 (Table 2; text 3.43e-10); dust heating = local UV
  + unattenuated-FUV-equal optical with tau_PE/1.87 + 6 K blackbody (B2 recipe, used everywhere; GIZMO floors T_bb
  at T_CMB(z)); k_gr T-dependent (eq. 16) in GIZMO, constant 3e-17 Z'_d in the benchmarks (as the paper's tests);
  x_e switches from eq. 27 to 28 sharply at 2e4 K; heating is not multiplied by (1-S).
- Line trapping: only CO (LVG, N_eff = n_CO / max(|dv/dr|, v_th/100 pc), |dv/dr| = mean |dv_i/dx_i| from the
  velocity gradients, `dvdr_cgs`); CII, CI, OI, H2 and the H lines are optically thin, as stated in Sec. 3.2 of
  the paper and done in the reference implementation. The CII/LVG trapping of the other networks is not used.
- GIZMO mapping: Z'_g = x_O,tot / 3.2e-4 (cell gas-phase O); Z'_d = cell DGR; chi = G0/1.7, chi_LW = G0_LW/1.7;
  zeta_pi,H and its heating from the M1 rates (rt_phot_HI / rt_heat_HI), H2 PI with q = 4.4 eV; N_C pixel
  columns are zero (TreeCol has no C column), so C self-shielding is off in GIZMO (H2 cross-shielding kept).
- Not ported: adaptive ray tracing / six-ray (TreeCol supplies the columns), grain destruction (App. E; GIZMO
  zeroes the DGR above 1e5 K), He/metal photoionization, OPT/IR radiation transfer. Driver: CVODE on 3 unknowns
  instead of the paper's explicit sub-cycling; TNOCHEM hot bypass as other nets (T from the net-20 relation).
- lambda slots: 0 gas-dust, 1 H2, 2 nebular, 6 CO, 10 CR (heat), 11 PE (heat), 12 OI, 13 CI, 15 CII, 16 grain
  rec., 24 Lya, 25 CIE x S, 27 ff; lambda_chem: 0 H2 cd, 1-3 H2 diss/pump/form (heat), 4 H ci, 5 H rr, 6 PI (heat).
- Benchmarks B1-B4: /orion/ptmp/uli/orion/chem_closeout/net20/bench/out/RESULTS20.md (sigma_LW 1.5e-21 run kept in out_sig1.5e-21).

## D-30. (2026-10-23) nets 5, 17, 3: H+ photoionization stays restoring above x_H+ = 1, both backends
`ph38 * abHI` uses the conserved HI fraction with the unclipped H+ (`1 - 2 x_H2 - y_H+ - ...`) when that is
negative (validate_input clips y_H+ to 1 and abHI is floored at 0, so the term was flat above 1). Under a strong
M1 field (Gamma_HI ~ 1e-7 s^-1) the equilibrium lies within ~1e-5 of x = 1; CVODE's unclipped state drifted
above 1 to the 1 + EPS_MAX reject wall, h collapsed to ~1e-276 s and the dense DQ Jacobian produced NaN trial
states (B2 cells 3782, 9151). Unchanged whenever that HI fraction is >= 0 or ph38 = 0 (no RT, no TreeRay-PI,
no comoving UVB). C++ chem_rhs_net5/17/3, Fortran rate_eq_simple/ulli/mol (net 3 saves y(ihp) before
validate_input). Not changed: unbuilt rate_eq.F, rate_eq_GMC.F, rate_eq_primordial.F; D+ (`ph38 * abDI`)
and He (`abheI`) photoionization, collisional ionization.

## D-31. (2026-10-23) C++ driver fails loudly instead of collapsing
- Non-finite abundances, energy, yn or dt on entry: full cell context, return 101 (fatal). Previously the
  state reached return 100 only after the substep had been halved below 1e-8 s.
- validate_input no longer prints per NaN; it counts, and the driver prints one line per solve with the
  number of NaN trial states CVODE rejected, the substep and the CVODE step size at the first one.
- CVodeSetMinStep(10 DBL_EPSILON x (t_start + dt)) before every CVode call; a failure with h at that floor
  prints the cell context (first per solve) and a per-solve count, and is rejected like any failed substep
  (halve, retry; fatal 100 below 1e-8 s). Making it fatal turned a solve that the retry completes into an
  abort (net 17, n 1, T0 1e6 K, G0 1e4, Z 3: CVODE at the He++ reject wall, flag -10). Below it t + h does not advance t; CVODE's first-step lower
  bound is 100 uround |tout|. A floor of 1e-12 x substep was tried and rejected: it clamped legitimate steps in
  hot dense gas (net 17, T0 1e6 K, n 1e6, dt 1e14 s; 99 one-zone states changed). Fortran DVODE path unchanged.
## D-32. SGChem rate updates ch9, ch17, ch18, both backends
Provenance: Arepo SGChem `cheminmo.F:347-349` (ch9) and `:451-470` (yfac, ch17, ch18).
* ch9 (He+ + H2 -> He + H + H+): 3.7e-14 exp(-35/T) -> 1.26e-13 exp(-22.5/T), same T < 3 K floor.
* ch17: the fixed 0.677 becomes yfac = 0.677 - 0.054 (abhe - 0.079)/0.021, abhe = /chem_comp/
  at CHEMINMO (ABHE, 0.1 for nets 5/15/16/17; per-cell abhe does not enter the tables).
* ch18: 0.96 ch17 + 0.677 gs -> 0.96 heii_rec_B + yfac gs (the old form counted ground-state
  photons twice and included 0.96 x dielectronic).
Not ported: SGChem chtab(148) (our slot 148 is a different reaction; neither is read here) and
the SGCHEM_GONG17_TEST branch. abhe reaches the C++ builder as `ChemBuildConfig.abhe` ->
`ChemGlobals.abhe` (`extract_rates.py` GLOBALS_MAP); the generator now reads the Fortran next to
it (`CHEMCOOL_FORTRAN_DIR` overrides). Consumers: ch9/ch17 in nets 3/15/16/17, ch18 in nets 3/17
(zeroed while 2 NH x_HI < 1.6e17, i.e. always with dl = 0); net 5 reads none of them.

## D-32. T_CMB cooling floor in nets 5 and 16, both backends
Provenance: SGChem `rate_eq_simple.F:459-470`, `rate_eq_gong17.F:875-889`.
* ydot(T) = -(ylam + ylam_chem - ylam_floor), ylam_floor = cool_func at T_CMB(z) = 2.726 (1+z) with
  iflag_cmb = 1, at every z (as SGChem). Nets 3/17 keep their z > 0 gate (= SGChem rate_eq_primordial);
  SGChem net 15 has no floor, ours neither.
* cool_func publishes `lambda` only for the iflag_cmb = 0 call, so the floor call no longer overwrites the
  diagnostic rates (also affected nets 3/17 at z > 0 before).

## D-33. CMB heating of dust, all networks, both backends
Provenance: SGChem `cool_util.F:780-804` (calc_dust_temp).
* isrf_heat = max(c_isrf chi_isrf - 4.68e-31 2.726^6, 0) yn G0_dust/1.7 + 4.68e-31 T_CMB(z)^6 yn, then x DGR
  (our DGR-multiplied convention); c_isrf = 5.8e-24 (isrf_option 1) or 5.73e-24. The z = 0 CMB share is moved
  out of the ISRF coefficient and the CMB term follows T_CMB(z).

## D-34. Grain recombination cooling (SGChem rates(29)), all networks, both backends
Provenance: SGChem `cool_func.F:1239-1248`, `coolinmo.F` cl75/cl76.
* cl75 = 0.74 / tfix^0.068, cl76 = 4.65e-30 tfix^0.94 phi_pah (tfix = min(T, 1e4)); NCLTAB 67 -> 76.
* Added to rates(12) (PE slot) as DGR cl76 PEvar0^cl75 x_e n^2 wherever the BT94 PE rate is evaluated
  (zero when PE is zero, as SGChem). Not added with CHEMCOOL_PE_K23, which carries WD01 recombination cooling.

## D-35. BT94 photoelectric fits held at T = 1e4 K above 1e4 K, all networks, both backends
Provenance: SGChem `coolinmo.F:639-651`. cl16 = sqrt(tfix)/phi_pah, cl17 = 3.7e-2 (tfix/1e4)^0.7.

## D-36. phi_pah = 0.5, both backends
Provenance: SGChem `init_chemistry_parameters.F:86`. Was 1.0 (gadget_tnt_new value). Enters cl16 (PE
charging parameter) and cl76 (D-34). Set in `chemcool.cc`, `chem_backend_cxx.cc` and the golden harnesses.

## D-37. PE electron cut-off x_e n < 1e-9 G_dust cl16, both backends
Provenance: SGChem `cool_func.F:909`. Was 1e-20. Our DGR < 1e-8 cut (SGChem: DGR = 0) is kept.

## D-38. No table extrapolation above tmax = 1e8 K in cool_func and net 3, both backends
Provenance: SGChem `cool_func.F:217-219` (dtemp = 0). cool_func and rate_eq_mol (net 3) set dtemp =
temp - temptab(nmd) above tmax, i.e. linear extrapolation of every cl/ch table with the last slope; now held at
the tmax value, as the other rate_eq index paths already were.

## KEPT deliberately (audit KEEP items — listed so they are not "fixed" later)

* **R1/R4 — the outer substep loop is kept literally.** Safety factors 0.1
  (species) / `DTCOOL_SCALE_FACTOR = 0.3` (energy), `ctrl_floor` pool gating
  (`eps_ctrl = 1e-6` × elemental pool), `dt = min(dt_est, t_end - t_start)`.
  Successful substeps *continue* the integrator (no re-init) ≡ DVODE
  `istate=2`, so `check_abundances` clips modify the driver's `y` (the final
  answer / the next `y_old`) but do **not** feed back into the integrator's
  trajectory. That clip-no-feedback subtlety (audit §3.1, "Critical subtlety")
  is reproduced exactly. Failures restore `y` *and* `tdust`, halve `dt`, and
  `CVodeReInit` ≡ `istate=1`.
* **R4 — tolerances.** `RTOL = 1e-4` (the "0.1%" comment in the Fortran is
  stale — not "fixed"). Per-species `ATOL` via `CVodeSVtolerances`: H2 1e-7,
  H+ 1e-12, CO 1e-14, and for net 17 He+ 1e-14, He++ 1e-14, D+ 1e-14, HD 1e-12.
  `MXSTEP = 200000` via `CVodeSetMaxNumSteps`.
* **R4 — `ATOL(energy) = 0` is kept as 0**, i.e. pure relative control on the
  energy component, exactly as `ATOL_TMP = 0d0`
  (`chemcool_consts.h:572`, `evolve_abundances.F:755`).
  *Justification for keeping it rather than mapping it to a small positive
  floor:* a zero `atol` component is legal in both DVODE and CVODE; it is only
  unsafe if that component can pass through zero, because the error test then
  has no scale. Here the energy component cannot: it is an internal-energy
  **density**, floored at `rho*All.MinEgySpec` before entry
  (`chemcool.cc:577-581`), bounded below physically by the CMB floor, and
  strictly positive throughout. Substituting any positive `atol_E` would set an
  absolute energy scale in `erg cm^-3` — a quantity that spans ~20 decades
  across the (n, T) domain — so a single constant would be simultaneously
  meaningless in diffuse gas and dominant in dense gas, and would change the
  accepted step sequence in a resolution-dependent way. Pure relative control
  is the only scale-free choice, and it is what has been validated.
  CVODE raises `CV_TOO_MUCH_ACC`/`-6` if a pure-relative component ever
  vanishes; that returns negative and lands in the normal restore/halve path,
  so the failure mode is contained rather than silent.
* **R4 — numerical Jacobian.** `MF = 22` maps to CVODE BDF + `SUNLinSol_Dense`
  with the internal difference-quotient Jacobian. There is no analytic
  Jacobian to port: `jac.F` is a dummy that DVODE never calls with `MF=22`.
* **R6 — the TNOCHEM bypass is ported wholesale**: 5e5 K threshold, molecules
  zeroed, hot-CIE ion set (`H+=1`, `He+=0`, `He++=abhe`, `D+=abundD`), explicit
  energy-only forward-Euler subcycle at `dt = 0.1|e/edot|`, 100000-iteration
  budget, mid-step hand-back to the stiff path when T drops below threshold,
  the `1e-10*t_end` relative completion tolerance (WEDGEFIX 3), and the
  `ipar`-clear semantics (RHS return value ignored inside the bypass loop).
* **R7 — failure taxonomy and codes.** `dt < 1e-8 s` → return 100 (≡ ABORT(100),
  with the same state dump); substep budget 9e6 exhausted → return 1
  (≡ ABORT(0)); negative or bad-zero `dt` → return 2 (≡ ABORT(1)/(2)). The
  harness maps any nonzero to the `chemcool_fatal` exit path.
* **R5 — photorate freshness.** `calc_photo` runs exactly once per cell call;
  `phrates`, `G_dust`, `chi_isrf`, `fshield_CO` and the dust factors are frozen
  for the whole call (snapshotted into `ChemCell` by `chem_bind_cell()`), while
  H2/HD self-shielding and the Doppler `b` are recomputed at every RHS
  evaluation from the live `abh2`/`T`, and `tdust` is re-solved every RHS
  evaluation and rolled back with `y` on retries.
* **net-17 specifics kept:** the 25-pass `eqtol = 1e-3` algebraic sub-iteration
  for H⁻/H2⁺/HeH⁺ with the 2026-07-07 WEDGEFIX 1/2 clamps; the
  `2*NH*abHI < 1.6e17 → ch18 = 0` optically-thin switch (always taken with
  `dl=0`); `abundances(2) = abo*(1-abhp)`; the second `cool_func` call at
  `T_CMB` when `redshift > 0`.
* **R10 is NOT applied, and Gate 3 decided to keep it that way — see D-19.**
  `lambda`/`lambda_chem` keep the Fortran
  "last RHS evaluation wins" semantics so the A/B stays comparable. The audit's
  recommendation (a dedicated post-solve RHS evaluation at the accepted state)
  is a Gate-3 change.

---

## Open / deferred

* **R2 (per-cell context) is only half done.** `ChemCell` exists and the C++
  RHS is reentrant, but the Fortran leaves still read COMMON, so the Gate-2
  backend is single-threaded. Closing this requires Gate 3.
* **R10** diagnostic-consistency change — deferred (above).
* **`iflag_atom=1` cl6 branch** (`coolatom`, 81 points) is ported in
  `chem_spline.cc` but not golden-tested; production uses `iflag_atom=2`.
* **Gate 4** (`CHEMCOOL_CXX` compile flag in canonical, default OFF, Fortran
  retained as the fallback) is untouched — Uli's standing requirement that the
  Fortran path remains the reference implementation still holds.

## Uli ruling 2026-07-29: "we trust the Glover code when in doubt"
- D-20 NL97 Gamma_CHx normalization (audit provenance flag): KEEP the code-heritage
  value (2.94e-10, as in Glover's own code) — do NOT adopt the GC12 paper
  reinterpretation. Both backends unchanged.
- D-21 co_cool vib/rot table column (cool_util.F:131, Gate-3 finding): inherited
  from the Glover code — KEEP as-coded in both backends. Do not re-flag.
Standing principle for all future paper-vs-code discrepancies in the chemistry:
the Glover CODE is the reference implementation.
