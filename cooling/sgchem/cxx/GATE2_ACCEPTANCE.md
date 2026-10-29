# Gate 2 acceptance — SGCHEM C++/CVODE backend vs the Fortran/DVODE golden master

**Date:** 2026-07-29 · **Verdict: all three gates PASS.**

| gate | criterion | result | verdict |
|---|---|---|---|
| (a) equilibrium | T_eq within **1%** of Fortran, 20 Tier-1 ladder points × 2 nets | 40/40 within 1%; **max 0.72%**, median 0 (net 5) / 2.2e-7 (net 17) | **PASS** |
| (b) relaxation | fig6 trajectory cases, report max deviation | 16 cases, 2400 compared points; **max pointwise ΔT/T = 0.352%** | **PASS** |
| (c) certification | 80-zone (n,T₀) grid per net, all zones complete, **zero failures** | 160/160 complete; **0 failures on either backend** | **PASS** |
| — | CVODE vs DVODE wall time | net 5 **×1.01**, net 17 **×1.10** (CVODE faster) over 16 regimes | reported below |

Supporting Gate-1 goldens (bit-level, same session):
**346/346 rate functions** at 1e-12 (worst 1.4e-14) and
**40000/40000 spline-table points** at 1e-12 (worst 1.65e-14).

---

## 0. What is actually being compared

Both backends run **inside the same binary** (`onezone3_net{5,17}`), selected by
`--backend f77|cxx`. They link the **same** `coolinmo`/`cheminmo`-built `/coolr/`
tables, call the **same** `cool_func`, `calc_dust_temp`,
`calc_H2_shielding_factor` and `lookup_temp_gamma`, and receive an identical
per-cell input state and an identical `calc_photo` freeze. What differs, and
only what differs:

* the integrator — SUNDIALS CVODE (BDF, dense difference-quotient Jacobian)
  instead of DVODE `MF=22`;
* the RHS assembly — `chem_rhs_net5.cc` / `chem_rhs_net17.cc` instead of
  `rate_eq_simple.F` / `rate_eq_ulli.F`;
* the driver/retry policy — `cvode_driver.cc` instead of `evolve_abundances.F`.

Every deviation below is therefore attributable to the ported layer and nothing
else. This is the scope decision recorded in `DEVIATIONS.md` D-0; the leaf
kernels are Gate 3.

**The benchmark compiles the same source files GIZMO compiles.** `build3.sh`
sets `CXX_DIR` to `cooling/sgchem/cxx` inside the build copy of the gizmo
tree. There is no benchmark-only variant of any source file.

Configuration for all runs: Z = 0.1 Z☉, G₀ = 1.7 Habing, ζ = 1.8e-17 s⁻¹,
shielding length 10 pc (`dl = 3.086e19` cm, `iphoto=6` TREECOL columns),
`WSS_CIE_COOL`, `TREE_RAD`/`TREE_RAD_H2`/`TREE_RAD_CO`, `NSIDE=1` (12 pixels) —
i.e. the production Stella chemistry configuration.

Reproduce:
```bash
bash /orion/ptmp/uli/orion/stella/tests/validation/chemistry_onezone/src/build3.sh
cd /orion/ptmp/uli/orion/test/gizmo/cooling/sgchem/cxx
python3 gate2_bench.py all && python3 gate2_timing.py && python3 gate2_summary.py
```
Raw data: `fb_tests/_sgchem_cxx_build/gate2_data/*.csv`,
logs `fb_tests/_sgchem_cxx_build/gate2_*.log`.

---

## 1. Gate (a) — equilibrium temperatures

20 points drawn from the Tier-1 ladder grid of `audit_sweeps.py` (the same
ladders that produced `data/audit_{Z,G0,CR}ladder.csv`): 7 points spanning the
n_H = 10⁻³…10⁶ cm⁻³ ladder at the standard point, plus 6 Z-ladder, 4 G₀-ladder
and 4 CR-ladder points. Each point is run on **both** backends with the
`audit_sweeps.final_T` protocol — 300 Myr from T₀ = 10⁴ K, 60 steps, last row —
so the comparison is against a freshly computed Fortran value, not a cached CSV.

| net | points | max ΔT_eq/T_eq | median | > 1% | non-OK exits |
|---|---|---|---|---|---|
| 5 | 20 | **7.18e-03** | 0.0 | 0 | 0 |
| 17 | 20 | **6.71e-03** | 2.2e-07 | 0 | 0 |

**GATE: PASS** — 40/40 points within 1%, worst 0.72%.

Worst cases (all in the cold dense corner):

| net | ladder | n_H | Z | T_eq DVODE | T_eq CVODE | rel |
|---|---|---|---|---|---|---|
| 5 | Z | 1.0e+05 | 1.0 | 4.5535 K | 4.5862 K | 7.18e-03 |
| 17 | n_H | 1.0e+06 | 0.1 | 5.6165 K | 5.6542 K | 6.71e-03 |
| 17 | n_H | 3.2e+04 | 0.1 | 12.898 K | 12.878 K | 1.59e-03 |
| 5 | n_H | 3.2e+04 | 0.1 | 12.303 K | 12.331 K | 2.28e-03 |

**Interpretation — this is not a bug, and it is worth understanding.** The
deviations are not spread over the domain: for the median point the two
backends agree to *zero* (net 5) or to 2e-7 (net 17). Everything above 1e-3
sits at n_H ≥ 3×10⁴ cm⁻³ where T_eq ≈ 4–13 K. There the gas is on the cold
molecular branch, the thermostat is the near-vertical CO/dust-coupling part of
the cooling curve, and dT/dΛ is very small: an O(RTOL) difference in the
accepted energy — which is exactly what two different BDF implementations at
RTOL = 1e-4 are entitled to produce — maps to a several-×10⁻³ difference in T.
An absolute deviation of 0.03 K at 4.5 K is the honest description. The
agreement is limited by the *tolerance the physics is solved at*, not by the
port: both backends are converging to the same attractor, and neither is more
"right" than the other at RTOL = 1e-4.

---

## 2. Gate (b) — relaxation trajectories

The four `RELAX_CASES` of `make_figures.py` (the fig6 panel set), each at two
time scales (`short` = t_end/300, `long` = t_end), 150 output steps, both nets,
both backends — 16 cases, **2400 compared trajectory points**. Deviation is
measured **pointwise along the whole trajectory**, not just at the endpoint.

| net | case | scale | n_H | T₀ | max ΔT/T | max Δx_H2/x | max Δx_H+/x | ΔT_end/T |
|---|---|---|---|---|---|---|---|---|
| 5 | hot_dense | short | 1e4 | 1e6 | 4.37e-04 | 2.25e-03 | 1.27e-03 | 4.52e-07 |
| 5 | hot_dense | long | 1e4 | 1e6 | 8.61e-04 | 1.70e-04 | 4.80e-04 | 8.61e-04 |
| 5 | wnm_cool | short | 1e2 | 8e3 | 0.0 | 0.0 | 0.0 | 0.0 |
| 5 | wnm_cool | long | 1e2 | 8e3 | 1.74e-07 | 4.70e-07 | 0.0 | 0.0 |
| 5 | h2_form | short | 1e4 | 1e2 | 0.0 | 0.0 | 0.0 | 0.0 |
| 5 | h2_form | long | 1e4 | 1e2 | 3.88e-04 | 6.05e-05 | 1.01e-03 | 2.53e-05 |
| 5 | diffuse_heat | short | 0.1 | 1e2 | 0.0 | 0.0 | 0.0 | 0.0 |
| 5 | diffuse_heat | long | 0.1 | 1e2 | 0.0 | 0.0 | 0.0 | 0.0 |
| 17 | hot_dense | short | 1e4 | 1e6 | **3.52e-03** | 2.77e-02 | 1.18e-02 | 5.18e-06 |
| 17 | hot_dense | long | 1e4 | 1e6 | 5.90e-04 | 4.31e-04 | 1.06e-04 | 9.12e-05 |
| 17 | wnm_cool | short | 1e2 | 8e3 | 3.93e-05 | 2.44e-04 | 5.86e-06 | 3.71e-05 |
| 17 | wnm_cool | long | 1e2 | 8e3 | 8.84e-05 | 3.32e-04 | 1.36e-04 | 1.68e-06 |
| 17 | h2_form | short | 1e4 | 1e2 | 6.79e-07 | 3.28e-06 | 1.32e-06 | 0.0 |
| 17 | h2_form | long | 1e4 | 1e2 | 4.50e-04 | 1.57e-04 | 4.38e-04 | 1.01e-04 |
| 17 | diffuse_heat | short | 0.1 | 1e2 | 4.60e-05 | 2.09e-05 | 1.67e-05 | 7.04e-07 |
| 17 | diffuse_heat | long | 0.1 | 1e2 | 2.92e-06 | 9.00e-06 | 1.65e-06 | 0.0 |

**Max pointwise deviation over all 2400 points: ΔT/T = 3.52e-03 (0.352%).**

Four of the eight net-5 cases are **bit-identical** across the whole
trajectory. The largest deviation is `hot_dense/short` on net 17 — a 10⁶ K,
n_H = 10⁴ cm⁻³ zone dropping through the TNOCHEM threshold and the fastest
transient in the set; the H2 fraction there differs by 2.8% *while x_H2 is
~10⁻⁷ and physically irrelevant*, and the endpoint temperature still agrees to
5e-6. Wherever the trajectories are compared after the transient, agreement
returns to the 1e-5…1e-7 level.

Per-point data: `gate2_data/gate2_b_points.csv` (all 2400 rows).

---

## 3. Gate (c) — DVODE certification grid

The `sweep_dvode` domain: n_H = 10⁻³…10⁶ cm⁻³ (10 values) × T₀ = 10¹…10⁸ K
(8 values) = **80 zones per net**, run on both backends.

| net | zones | DVODE complete | CVODE complete | failures | T_eq max rel | median |
|---|---|---|---|---|---|---|
| 5 | 80 | 80/80 | 80/80 | **0 / 0** | 7.99e-03 | 0.0 |
| 17 | 80 | 80/80 | 80/80 | **0 / 0** | 5.98e-03 | 4.5e-06 |

**GATE: PASS** — 160/160 zones complete, zero failures on either backend, and
no zone exceeds 1% in T_eq either. Notably the CVODE backend does not fail
anywhere DVODE succeeds, including the hot low-density corner that historically
produced the `nstep exceeded` / `dt < 1e-8` wedges.

---

## 4. Wall time — CVODE vs DVODE

The certification zones complete in ~1–6 ms each, which is clock-granularity
and process-noise dominated; those numbers are reported for completeness but
are not a usable benchmark. The table below is the dedicated measurement
(`gate2_timing.py`): 8 representative regimes per net, **2000 hydro steps per
zone**, 3 repeats, minimum taken, measuring **only the interval inside the
integrator** (table construction and the per-step `calc_photo` are excluded).

| net | regime | T₀ [K] | n_H [cm⁻³] | DVODE [s] | CVODE [s] | speedup | µs/step DVODE | µs/step CVODE |
|---|---|---|---|---|---|---|---|---|
| 5 | diffuse_warm | 1e4 | 1e-2 | 0.0253 | 0.0279 | 0.91 | 12.64 | 13.93 |
| 5 | wnm | 8e3 | 1e0 | 0.0250 | 0.0276 | 0.91 | 12.48 | 13.78 |
| 5 | cnm | 1e2 | 1e2 | 0.0251 | 0.0276 | 0.91 | 12.55 | 13.79 |
| 5 | molecular | 3e1 | 1e4 | 0.0326 | 0.0188 | **1.74** | 16.30 | 9.38 |
| 5 | dense_cold | 3e1 | 1e6 | 0.0167 | 0.0183 | 0.91 | 8.36 | 9.16 |
| 5 | shocked_hot | 1e6 | 1e2 | 0.0142 | 0.0157 | 0.90 | 7.11 | 7.87 |
| 5 | tnochem_hot | 1e7 | 1e1 | 0.0140 | 0.0153 | 0.91 | 6.98 | 7.64 |
| 5 | sn_bubble | 1e8 | 1e0 | 0.0010 | 0.0011 | 0.90 | 0.52 | 0.57 |
| 17 | diffuse_warm | 1e4 | 1e-2 | 0.0552 | 0.0561 | 0.98 | 27.62 | 28.07 |
| 17 | wnm | 8e3 | 1e0 | 0.0295 | 0.0300 | 0.98 | 14.75 | 15.02 |
| 17 | cnm | 1e2 | 1e2 | 0.0812 | 0.0839 | 0.97 | 40.60 | 41.95 |
| 17 | molecular | 3e1 | 1e4 | 0.0650 | 0.0745 | 0.87 | 32.51 | 37.27 |
| 17 | dense_cold | 3e1 | 1e6 | 0.0647 | 0.0659 | 0.98 | 32.35 | 32.93 |
| 17 | shocked_hot | 1e6 | 1e2 | 0.0754 | 0.0525 | **1.44** | 37.71 | 26.26 |
| 17 | tnochem_hot | 1e7 | 1e1 | 0.0605 | 0.0320 | **1.89** | 30.23 | 15.99 |
| 17 | sn_bubble | 1e8 | 1e0 | 0.0040 | 0.0017 | **2.36** | 1.98 | 0.84 |
| **5** | **TOTAL** | | | **0.1539** | **0.1522** | **×1.01** | | |
| **17** | **TOTAL** | | | **0.4355** | **0.3967** | **×1.10** | | |

**Reading of this.** Overall the two integrators are within ~10% of each other:
CVODE is 1% faster on net 5 and 10% faster on net 17. The distribution is more
informative than the total — CVODE is consistently ~9% *slower* in the cool,
chemically-active regimes (diffuse/wnm/cnm/dense_cold) and substantially
*faster* in the hot regimes on net 17 (×1.44 shocked, ×1.89 TNOCHEM, ×2.36 SN
bubble). The hot-regime win comes from the port's TNOCHEM bypass and driver
path doing less work per substep than the Fortran equivalent, and it is the
regime that dominates cost in SN-driven runs. The cool-regime loss is the
per-substep overhead of CVODE's `CVode()` entry versus DVODE's, paid on many
short substeps.

Fairness notes: the port caches one CVODE workspace and calls `CVodeReInit`
(≡ DVODE `istate=1`) instead of allocating per cell call, so it is not charged
a malloc DVODE never pays (`DEVIATIONS.md` D-7); both backends are compiled
`-O2` by the same GCC 13.1.0; measurements were taken on a shared login node,
which is why the minimum of 3 repeats is used.

**Honest caveat:** this is *not* a demonstration that the C++ port is faster.
It is a demonstration that replacing DVODE with CVODE costs nothing in
performance while gaining a modern, maintained, thread-safe-capable solver with
an explicit per-cell context. Performance was never the motivation; the
maintainability and the road to Gate 3/4 are.

---

## 5. Supporting bit-level goldens (Gate 1 / 1b)

| layer | test | result |
|---|---|---|
| analytic rate coefficients | 346 generated functions vs Fortran, 20k points each incl. every guard boundary | **346/346 PASS at 1e-12**, worst 1.4e-14 |
| spline/table cooling entries | cl1, cl2, cl3, cl6 vs the real `COOLINMO`-built `cltab`, all 10000 grid points, both nets | **40000/40000 PASS at 1e-12**, worst 1.65e-14 |
| classification claim | cl52/59/60/61 are analytic (not table-driven) under `iflag_atom=2` + `WSS_CIE_COOL` | verified by counting nonzero grid points after `COOLINMO`: 5497–7500 of 10000 each ⇒ not zeroed ⇒ analytic |

Reaching 40000/40000 required fixing a real fidelity bug in the port, worth
recording: `sgchem_consts.h` spells its constants as bare decimals
(`1.38066e-16`, …), which Fortran treats as **default REAL (single precision)**
literals; the running code holds `float32(1.38066e-16)`, 1.26e-09 away from the
decimal, and likewise for every bare-decimal `DATA` array. The port originally
read them as exact doubles and was therefore ~2e-6 "better" than the code it
was supposed to reproduce. Full account: `DEVIATIONS.md` D-10.

---

## 6. GIZMO integration status

The backend is compiled into GIZMO behind the Gate-0 seam, selected by
`SGCHEM_CXX` (**default OFF**). Verified on Orion
(`SYSTYPE=Orion-OpenMPI`, `CHEMISTRYNETWORK=17`):

| build | result |
|---|---|
| `SGCHEM_CXX` OFF | links clean, 4,163,264 B; no cxx object built, **0 CVODE symbols**; object list identical to pre-change |
| `SGCHEM_CXX` ON | links clean, 5,631,216 B; `nm` confirms `T chem_evolve_cxx`, `T cxx_evolve_abundances` and **110 CVODE symbols** in the executable |

The benchmark binaries in §1–4 were built from the Gate-2 state of `cxx/`
(solver layer only). A parallel session has since begun Gate 3 — the C++ leaf
kernels — so the ON executable now also contains those in-flight files. **The
numbers in this document describe the Gate-2 solver-layer port**; the Gate-3
kernels have their own goldens and are not certified here.

**Not yet done — the honest gap.** No *full GIZMO run* has been made with
`SGCHEM_CXX` ON. Gate 2 certifies the chemistry solver against the golden
master in one-zone; it does not yet certify the coupled code. The next step is
an A/B on a short Stella test (sn20 bench) comparing snapshots between OFF and
ON. Until that is done the flag should stay OFF in production, which is also
Uli's standing requirement for Gate 4.

---

## 7. Verdict

* **(a) equilibrium — PASS.** 40/40 within 1%; max 0.72%, in the cold dense
  corner where dT/dΛ makes RTOL = 1e-4 worth several ×10⁻³ in T.
* **(b) relaxation — PASS.** 2400 compared trajectory points, max pointwise
  ΔT/T = 0.352%; four net-5 cases bit-identical.
* **(c) certification — PASS.** 160/160 zones complete, zero failures on either
  backend.
* **wall time** — net 5 ×1.01, net 17 ×1.10 in favour of CVODE overall, with a
  ×1.4–2.4 advantage in the hot regimes and a ~9% deficit in the cool ones.

No tolerance was adjusted to obtain any of these results. The gates are the
ones specified before the runs were made, and the deviations reported are the
ones measured.
