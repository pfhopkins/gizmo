# INDEPENDENT ADVERSARIAL AUDIT — SGCHEM C++/CVODE port (2026-07-29)

Auditor: independent session (Uli directive "do an audit on that port", scope
extended same day to "audit against Simon Glover's network" = provenance).
Method: every claim re-verified by RUNNING and READING, never by trusting.
All reproductions were made in an isolated scratchpad; **nothing in the
canonical tree or the fb_tests artifacts was modified by this audit** (this
file is the only write).

Audit baseline: cxx/ snapshot taken 15:13:55 (RHS/driver files stable since
~10:00; Gate-3 kernel files churning 13:28–15:13 DURING the audit — see F2).
Fortran reference: canonical `rate_eq_ulli.F` / `rate_eq_simple.F` /
`evolve_abundances.F` / `validate_rates.F` as of today (contain the 2026-07-28
NH + Si+-electron fixes).

---

## Verdict table (per acceptance claim)

| # | claim (source) | verdict | evidence |
|---|---|---|---|
| 1 | Gate 1: 346/346 rates at 1e-12 (STATUS.md) | **CONFIRMED — reproduced** | independent rerun: `346 PASS, 0 FAIL of 346 rates (tol 1e-12)`; regenerated `rates_gen.{h,cc}`, `rates.json`, `rates_registry.inc` **byte-identical** to checked-in |
| 2 | Gate 1b: splines 40000/40000, worst 1.65e-14 (STATUS.md) | **CONFIRMED — reproduced** | independent rerun: 40000/40000 PASS both nets, worst 1.653e-14; cl52/59/60/61 "analytic under iflag_atom=2+WSS" classification check reproduced (5497–7500 nonzero points each) |
| 3 | Gate 2(a): equilibria 40/40 <1%, max 0.72% (GATE2_ACCEPTANCE) | **CONFIRMED for the certified binary; numbers superseded** | current-state rerun: max 0.39% (net5) / 0.18% (net17) — still passes the 1% gate, but the published table no longer describes any extant binary (F2) |
| 4 | Gate 2(b): relaxation max pointwise 0.352% | **CONFIRMED / superseded** | current-state rerun: max 0.125%; my independent spot-run of 2 cases matches the regenerated CSV to all printed digits |
| 5 | Gate 2(c): 160/160 complete, 0 failures, "no zone exceeds 1% in T_eq either" | **PARTIALLY REFUTED at current state** | 160/160 complete + 0 failures still holds (formal gate passes). The side-claim "no zone exceeds 1%" is now **false**: 14 zones at n_H=1e6 reach ΔT_eq/T up to **1.31%** (see F4) |
| 6 | Gate 4 A: OFF binary byte-identical to pristine, 7 snapshot pairs | **CONFIRMED — independently re-verified** | h5py dataset-level walk of snapshot_006 pair: all datasets + all header attrs identical (raw md5 differs — HDF5 container metadata only, claim is about content and holds) |
| 7 | Gate 4 B: AGB twin p99 4.35e-6, max 5.57e-6 | **CONFIRMED — recomputed exactly** | my recomputation from the run artifacts reproduces every digit (snap001: med 0, p99 4.35e-6, max 5.57e-6; snap006: 1.64e-6/2.11e-6) |
| 8 | Gate 4 C: pisub twin med 7.8e-8, p99 8.1e-5, max 1.61e-4 | **CONFIRMED — recomputed exactly** | snap005: 7.62e-8/8.14e-5/1.61e-4; snap007: 7.83e-8/5.3e-5/1.02e-4 |
| 9 | dispatch switches at symbol level (GATE4) | **CONFIRMED** | `nm` on gate4 binaries: OFF = 0 `CVode` T-syms, no `chem_evolve_cxx`; ON = 110 CVode T-syms + `chem_evolve_cxx` (Fortran `evolve_abundances_` correctly still present in both — D-0 scope) |
| 10 | RHS/driver is a faithful port incl. 2026-07-28 fixes (chem_rhs headers) | **CONFIRMED by line audit** | see §RHS below: NH fix, Si+ electron closure, MSM96 float32 constants, all clamps of model-doc §5.5 items 1–11/16–17 verified term-by-term in both nets; two nits (F8) |
| 11 | D-15: "OpenMP loop over cells race-free — proven" | **OVERSTATED** | true only for the `chem_backend_cxx.cc` purecxx path (`thread_local tl_cell`). The **GIZMO entry path** (`chem_gizmo_backend.cc → chem_backend_c_api.cc`) uses **static `g_cell`/`g_tab`** — not thread-local, OPENMP-unsafe (which chem_gizmo_backend.cc itself admits). D-15 as written invites a wrong conclusion about the shipped path (F5) |
| 12 | D-12: "what passes the benchmark is literally what GIZMO compiles" | **WEAKENED at current state** | `build3.sh` (the documented benchmark build) **no longer links** — Gate-3 made the RHS call `cxx_*` kernels not in its TU list. The 15:15 bench binaries were built some other way; GIZMO's wildcard compiles 13 TUs incl. two convenience layers the benchmark never exercises (F3) |
| 13 | GATE4: "rates_gen.cc deliberately NOT compiled (verified)" | **SUPERSEDED/FALSE today** | `CHEMCXX_OBJS` is now a wildcard: rates_gen.cc IS compiled and linked — and must be (chem_tables_build.cc references `rate_*`; link fails without it — demonstrated) |
| 14 | D-11: `clean` removes cxx/*.o unconditionally | **CONFIRMED** | explicit `rm -f cooling/sgchem/cxx/*.o` on the second clean line (initial suspicion withdrawn) |

---

## Findings, ranked

### F1 (HIGH — ON-path functional gap): `lambda[]` / `lambda_chem[]` are never published back to the COMMON block under SGCHEM_CXX

`chem_evolve_cxx()` publishes only `COOLR.tdust` after the solve
(`chem_gizmo_backend.cc:100`). The C++ path accumulates the 28 radiative rates
in `ChemCell.lambda` (`chem_cool_func.cc:637`) and the chemical rates in
`ChemCell.lambda_chem` (RHS files) — **nothing copies either back to
`COOLR.lambda` / `COOLR.lambda_chem`**, which is exactly what
`chem_read_outputs()` reads (`chem_interface.cc:737-738`).

Consequences with the flag ON:
* `do_sgchem_step`'s `cooling_rate` sum is computed from stale COMMON data.
  Currently harmless in production **only by accident**: the sole live caller
  (`cooling.cc:153`, mode 0) discards the return value, and modes 1/2/3 have
  no callers.
* Under `OUTPUT_INDIVIDUAL_COOLRATES`, `CellP.Lambda[]` / `LambdaChem[]`
  snapshots are stale garbage.
* Under `RADTRANSFER`, the M1 energy coupling
  (`Lambda_RadiativeCooling_toRHDBins`, sgchem.cc:359-435) is fed from
  `chem_out.lambda[..]` — **the M1 RT ↔ chemistry coupling is silently broken
  in any SGCHEM_CXX+RADTRANSFER build** at the current Gate-3 state.

History note: at the Gate-2 state the Fortran `cool_func` (called through the
shims) wrote `COOLR.lambda` as a side effect, so the radiative part was
accidentally maintained; `lambda_chem` was stale under ON even then (the C++
RHS never wrote it). The Gate-3 kernel port removed the accidental cover.
Neither Gate-4 twin could catch this (no OUTPUT_INDIVIDUAL_COOLRATES, no
RADTRANSFER in those configs).

**Fix before any default-ON:** mirror the tdust line — publish `c.lambda` and
`c.lambda_chem` into COOLR in `chem_evolve_cxx()` — or `#error` on
`SGCHEM_CXX && (RADTRANSFER || OUTPUT_INDIVIDUAL_COOLRATES)`.

### F2 (HIGH — process): the certified state is not pinned to anything, and it moved during certification — and during this audit

* `cooling/sgchem/cxx/` is **untracked in git** (`??` in git status). No
  acceptance document records a commit hash; there is nothing the claims can
  be attached to.
* Three copies exist (canonical cxx/, fb_tests build copy, compiled binaries)
  and were **mutually inconsistent at audit time** (canonical changed twice
  during the audit: `chem_backend_c_api.cc`, `chem_calc_photo.cc`,
  `chem_gizmo_backend.cc`; build copy differed from canonical in 2 files).
* The gate2 CSVs on disk were **regenerated at 15:15–15:20 with Gate-3-state
  binaries** and no longer match the GATE2_ACCEPTANCE.md tables (which
  describe the 12:13 run). Gate-4's ON binary (built 12:13) predates the
  14:12–15:13 kernel churn.
* My fresh rebuild of "the same sources" 40 minutes after the 15:15 binaries
  produced a *third* set of numbers in the cold corner (cxx T_eq 12.88769 vs
  12.89335 vs f77 12.89831) — live numerical drift across Gate-3 states.

**Fix:** commit cxx/ (and the golden/bench scripts), stamp every acceptance doc
with the SHA it certifies, and re-run all gates once at that frozen SHA.
Until then "certified" means "was true of some binary that no longer exists".

### F3 (MEDIUM): every documented reproduce path is broken post-relocation

Verified by running them as documented:
* `golden/build.sh` → `can't open /orion/.../fb_tests/_sgchem_cxx_build/extract_rates.py`
  ($TOP still points at the pre-relocation home).
* `golden/test_rates.cc` includes `../rates_gen.h` / `../rates_registry.inc` —
  also the pre-relocation layout.
* `src/build3.sh` → **link failure**: `undefined reference to cxx_cool_func,
  cxx_calc_dust_temp, cxx_lookup_temp_gamma, chem_tables_build, rate_cst87…`
  (its 6-TU list is the Gate-2 set; the RHS now unconditionally calls the
  Gate-3 kernels; also needs `rates_gen.cc`).

The goldens **do** pass once the paths are patched (claims 1/2 above) — the
results are solid, the recipes are stale. STATUS.md "Reproduce everything"
steps 3–4 currently do not work.

### F4 (MEDIUM): cold-dense-corner T_eq deviation grew past the 1% bound at current state — plus a benchmark-harness tdust asymmetry that lands exactly there

Regenerated certification CSV (15:19, Gate-3 binaries): 14 zones, all at
n_H = 1e6 cm⁻³ (T_eq ≈ 5.6 K), with ΔT_eq/T up to **1.156e-2 (net 5)** and
**1.309e-2 (net 17)** — above the 1% criterion used in gate (a), and
contradicting the acceptance doc's "no zone exceeds 1%". Curiously the cxx
backend lands on a T0-independent value (5.58222 / 5.62031 for all eight T0)
while f77 scatters ±0.03 K — cxx sits on a cleaner attractor; which backend is
"right" is genuinely open, but the number now exceeds the stated bound.

Contributing suspect found by code read: at the Gate-3 state the two backends
disagree about **tdust carry across hydro steps in the one-zone harness**.
f77 carries tdust in `COOLR.tdust` naturally; the cxx path re-binds
`c.tdust = COOLR.tdust` every step (`chemcxx_init` per step), but nothing in
`onezone3.c` publishes the cxx tdust back (no `chemcxx_tdust()` call) and the
C++ leaves no longer write COOLR as a side effect → cxx re-enters every step
with the stale initial guess (10 K). The GIZMO ON path is NOT affected
(chem_evolve_cxx does round-trip tdust); this is a harness defect that
contaminates precisely the dust-coupled corner. Ruled out by direct test:
`CHEMCXX_TABLES=fortran` and `CHEMCXX_LEGACY_LAMBDA=1` both leave the corner
result bit-identical — the drift is in the kernel/tdust layer, not tables or
the lambda deviation.

**Fix:** onezone3 must carry tdust for the cxx backend (publish after evolve,
as GIZMO does) before Gate-3 certification numbers are meaningful in that
corner.

### F5 (LOW-MEDIUM — doc accuracy): thread-safety claim (D-15) does not cover the shipped path

`cvode_driver.cc` workspace + stats are `thread_local` (true), and the
purecxx harness path keeps `thread_local tl_cell`. But the **GIZMO entry
path** goes through `chem_backend_c_api.cc`'s `static ChemTables g_tab; static
ChemCell g_cell;` — shared mutable per-cell state. D-15's "an OpenMP loop over
cells is race-free — proven" is proven only for a path GIZMO does not use.
The operative warning ("do not combine with OPENMP") is present in
chem_gizmo_backend.cc and remains the truth. Rewrite D-15 to say so.

### F6 (LOW — doc): D-19 is referenced but does not exist; the KEPT register contradicts the code default

DEVIATIONS.md cites "D-19" twice (R10 bullet, cvode_driver comment) — there is
no D-19 section. Worse, the KEPT bullet says "R10 is NOT applied … lambda keeps
the Fortran last-RHS-evaluation-wins semantics", while `cvode_driver.cc`'s
**default** DOES apply the R10 redesign (extra RHS evaluation at the accepted
state; legacy only via `CHEMCXX_LEGACY_LAMBDA=1`). The acceptance numbers were
produced with the non-legacy default. Measured: in one-zone the toggle is
bit-neutral for the trajectory (verified — cold corner identical), but under
GIZMO ON the extra eval changes the published `DustTemp` and the next step's
Newton seed, so "diagnostic-only" is not exactly true. Write the missing D-19,
fix the KEPT bullet, and decide the default consciously.

### F7 (LOW — hygiene): production-binary behavior depends on undocumented environment variables

`CHEMCXX_TABLES=fortran` flips table provenance, `CHEMCXX_LEGACY_LAMBDA` flips
driver semantics, `CHEMCXX_DATADIR` redirects data files, `CHEMCXX_VERBOSE`
gates CVODE errors — all read inside the ON path of a science binary. A stray
sbatch export silently changes physics with only an easily-lost stderr line.
Before default-ON: freeze these to compile-time choices or log the active
values loudly at startup (stdout header, once).

### F8 (LOW — fidelity nit): two default-REAL literals in `cdex`/`fpheat` are ported as exact doubles

`rate_eq_simple.F` and `rate_eq_ulli.F` both compute
`cdex = (1.4 * dexp(…) * abH2 + 1.0 * dexp(…) * abHI) * 1d-12 …` — the `1.4`
is a **default-REAL (float32) literal** (= 1.39999997615814209…), exactly the
class the port's own D-10/D-16 doctrine says must be reproduced by float32
rounding. `chem_rhs_net5.cc:141` / `chem_rhs_net17.cc:438` use the exact
double `1.4` (≈1.7e-8 relative on the UV-pumping heating fraction). The MSM96
constants three lines up WERE float32'd — the treatment is inconsistent.
Negligible physically; fix for doctrine-consistency (or log it in DEVIATIONS).

### F9 (INFO): wildcard `CHEMCXX_OBJS` — measured behavior of the stray-file hazard

`CHEMCXX_OBJS = $(patsubst %.cc,%.o,$(wildcard cooling/sgchem/cxx/*.cc))`:
any `.cc` dropped in cxx/ is compiled at `-std=c++14` and **linked into the
science binary**. A non-compiling half-finished Gate-3 file breaks the build
loudly (acceptable); a compilable-but-wrong file is included silently; a file
with `main()` breaks the link. `golden/` is excluded (wildcard not recursive)
— the test mains are safe where they are. This is a deliberate, documented
choice (D-11) and defensible during active development, but a default-ON
production flag should NOT be wildcard-fed. Note also GATE4's "rates_gen.cc
deliberately NOT compiled (verified)" is now false (claim 13 above).

### F10 (INFO): things independently verified clean

* **Dispatch/seam:** single `#ifdef` at the solve call; TNOCHEM bypass, PHOTOION
  ionized-cell skip, net-17 ionized-molecule clamp, He/D pool rescales, Z cap,
  energy floor, mode 1/2/3 semantics, TREE_RAY/RADTRANSFER column branches all
  live UPSTREAM of the dispatch in shared code — preserved by construction.
  mode==2 returns before the COMMON commit in both paths.
* **Frozen-photorate contract (R5):** `CALC_PHOTO_WRAPPER` owns `/current_photo/`
  under both settings; `chemcxx_init()` is called once per cell call and
  snapshots phrates/G_dust/chi_isrf into ChemCell; H2/HD shielding + tdust
  re-evaluated per RHS. Matches the audit contract.
* **Per-cell state:** `chem_bind_cell` re-binds every field per call (incl.
  `abhe` per-cell → VARIABLE_XH_AND_ABHE flows; `X_H_chem` has no solver-layer
  consumer — correctly not bound); `column_est` hard-zeroed (D-5, correct);
  no cross-cell leakage found in the single-threaded path.
* **COOLR layout duplicates** (chem_gizmo_backend.cc vs chem_backend_fortran.cc
  vs chem_interface.cc): textually identical including the OUTPUT_SHIELD_FAC /
  WSS_CIE_COOL conditionals. A maintenance hazard (4+ copies) but no active
  mismatch; the pisub twin exercised the conditional layout.
* **Restart compatibility:** no `SGCHEM_CXX` in any declarations/ header; no
  struct layout changes; no solver state persisted → restart files are
  ON/OFF-interchangeable.
* **SUNDIALS pinning:** 6.7.0 confirmed in the frozen install's
  `sundials_config.h`; static libs; absolute default path is Orion-specific
  (documented override). Version is recorded in Makefile comment + GATE4.
* **Sensitivity note for future A/Bs:** at the n=3.16e4 point a 7e-6 input
  perturbation moves f77 T_eq by 2.7e-4 (≈40× amplification) — differences
  below ~1e-3 in the cold corners are within the RTOL=1e-4 attractor width and
  should not be over-interpreted (supports the port authors' own reading).

---

## RHS line-audit detail (Gate-2 layer, stable files)

`chem_rhs_net17.cc` vs `rate_eq_ulli.F`, term-by-term: species unpack & floors;
**Si+ metal-electron closure `abe += abundsi` present in both** (.F:172 region,
cc:80); **NH = 0.5·dl·yn assigned BEFORE `calc_H2_shielding_factor` in both**
(the 2026-07-28 canonical fixes are in the port); yntot/ekn/NaN guards; cs2<0
clamp (net 17 only — net 5 correctly lacks it, as coded); MSM96 two-branch CID
with float32 magic constants 10^(−1.164408f)/(−1.174242f); ch24/ch173,
ch138/137, ch157/30, ch160/159 1e-100 floors + geometric blends; H2+ ncrit;
ph4 CMB (GP98 + Coppola) terms; ph5 LTE cap incl. T<3.15e3 branch and G0=0
early-out; ch18 optically-thin zero at 2·NH·abHI<1.6e17; h_gr WD01 φ branches
+ exp(−T/3e4); cr1/cr1a/cr2/cr6 = crtab(1,1,2,6); 25-iter eqtol=1e-3
equilibrium loop with WEDGEFIX 1 (abhehp clamp) and WEDGEFIX 2 (abhehp_old
update) and the abM-persists-past-loop Fortran scoping; all 7 ydot blocks
(cst(1)→cst[0], cst(75)→cst[74] correctly 0-based); NL97 CO block (k0=5e-16,
k1=5e-10, β logic); rates_chem(1..22) coefficients; abundances(1..16) mapping
incl. `abo·(1−abhp)` and SiII/HD cooling slots deliberately unset; CMB second
cool_func call; final NaN sweep. **No unlogged deviation found** beyond F8.

`chem_rhs_net5.cc` vs `rate_eq_simple.F`: **`NH = yn·dl` full-dl convention
kept as-coded** (not harmonized with net 17 — correct per the port contract);
**`0.5*cr2` fudge present**; ncrinv = 2·abh2·(ch6−ch5)+ch5; pump factor 8.5
(net 5) vs 6.94 (net 17) both correct per source; rates_chem(6)=0 below 100 K;
lambda_chem copies unconditional (matches the Fortran, whose `#ifdef` is
commented out); ph42 M1 add-on in ylam_chem but not in lambda_chem (matches);
`pdv_term` dropped (D-5, hard-zeroed in production — verified sgchem.cc).

Driver (`cvode_driver.cc` vs `evolve_abundances.F`): safety factors 1e-1 /
DTCOOL 0.3; ctrl_floor pool gating (CO/D+/HD/He+/He++ pools, eps_ctrl=1e-6);
ATOLs match `sgchem_consts.h` (H2 1e-7, H+ 1e-12, CO/He+/He++/D+ 1e-14,
HD 1e-12, TMP 0); NSTEP 9e6; MaxNumSteps 200000; TNOCHEM bypass verbatim
(threshold, HOTIONFIX ion set, 0.1·|e/edot| Euler subcycle, 1e5 budget with
accept-on-exhaustion, WEDGEFIX 3 completion tolerance in both loops, mid-step
handback); check_abundances NaN/negative/excess logic incl. clip-vs-reject
thresholds and per-net max_abundance sets; dt<1e-8→100, negative→2, zero-dt
with t_end==t_start → silent return WITHOUT copy-out (Fortran quirk faithfully
kept); clip-no-feedback continuation (ReInit only on rejection = istate
semantics); `NOSTOP_WHEN_BELOW_MINTIMESTEP` not set in production → the port's
unconditional code-100 is equivalent. Documented deviations D-1/D-2/D-3 (
recoverable returns instead of ipar+zero-ydot / tmin-ydot / net-5 ABORT)
verified as described.

Model-doc §5.5 checklist: items 1–11, 16, 17 verified in the ported layer
(item 5's net-5/net-17 asymmetry replaced by unified recoverable return =
logged D-3); items 12–15 live in the Gate-3 kernels (outside the certified
Gate-2 scope; D-13's CO vib-index quirk and D-16's literal traps are logged
there); items 18–19 are C-side/shared (item 19 = RADTRANSFER lambda scrub —
**see F1**: the scrub runs, but on stale data under ON).

---

## Provenance audit vs Glover's published network (scope extension)

**This is a separate axis from port fidelity**: a transcription error inherited
from the original Fortran would propagate through a bit-perfect port. Sources
actually fetched at digit level: Glover, Federrath, Mac Low & Klessen 2010
(0907.4081, Tables A1/photo/CR), Glover & Mac Low 2007a (astro-ph/0605120),
Glover & Clark 2012 (1102.0670), Weingartner & Draine 2001 preprint
(astro-ph/0105237) + the published-table reproduction in Gong, Ostriker &
Wolfire 2017 (1610.09023), Glover & Abel 2008 (0803.1768). NL97 primary
(ApJ 482, 796) unreachable (no arXiv, ADS 403) — marked UNVERIFIABLE where it
is the only source.

### Summary: ~25 coefficient sets checked digit-by-digit

**MATCHES-GLOVER, digit-exact** (transcription clean): H2 grain formation
(HM79 form, exact after (T/100)→T unit unfolding vs G10 R165, incl. f_A and
sticking); H2 CID by H low-n + both ncr fits (G10 R9/R10, MSM96 factor-10 ncr
folded in); H2 CID by H2 low-n MKM98 fit; H2 photodissociation base 3.3e-11
per Habing (=5.6e-11 at Draine, G10 R168) and the full DB96 shielding fit
(0.965/(1+x/b5)² + 0.035·e^{−8.5e-4√(1+x)}/√(1+x), x=N_H2/5e14); HII case-A
AND case-B F92 fits (all six constants each); HI collisional ionization A97
9-coefficient poly; WD01 grain-assisted recombination for H+/He+/C+/Si+ —
all four ions digit-exact against the **published** WD01 table; the CR
ionization ratio set (H=1, H2 2.0/0.037/0.22/6.5e-4, C 3.83, O 5.67, He 1.09,
CO 6.52 + the crphot set incl. the CO T^1/2·x_H2/x_CO^1/2 factor, applied
correctly at the use site); NL97 k0=5e-16, k1=5e-10 and the β expression;
C+ radiative recombination (NP97 3-branch incl. the positive-exponent middle
branch); OII recombination; Lyman-α (Cen 92); the full Glover & Abel 2008
5-partner H2 cooling coefficient tables (~40 coefficients); C+ fine structure
(HM89/WBV96/WB02 stack); gas-grain HM89 3.8e-33 form; photoelectric heating =
Wolfire 2003's numbers (4.9e-2/3.7e-2/1.3e-24 — NOT BT94's, exactly as the
code comment claims); plus chtab(21,22,23,24,30,42) incidentally.

**DIFFERS — the one substantive physics flag:**

* **NL97 Γ_CHx field normalization.** `rate_eq_ulli.F:573` (and Glover's own
  `rate_eq_simple.F:222`): `Γ_CHx = 2.94e-10 × G0(Habing)` — i.e. NL97's
  5e-10 divided by 1.7, reading NL97's rate as Draine-normalized. Glover &
  Clark 2012 states NL97's 5e-10·G0 is **Habing**-normalized and itself
  re-calibrates to 1e-9·G0(Habing) to preserve NL97's Γ_CHx/Γ_CO = 5. Our
  ratio is 2.94e-10/1.235e-10 = **2.38, not 5** → β biased toward 1 → CO forms
  somewhat faster in FUV-exposed gas (shielded gas unaffected, β→1 there
  anyway). Factor 1.7 low vs raw NL97, 3.4 vs GC12's calibration. This is
  inherited verbatim from Glover's own code — a Glover-code-vs-Glover-paper
  inconsistency, NOT a local transcription error. **Physics decision for Uli**,
  same class as the D-13 CO index quirk.

**DIFFERS — small, correctly-attributed vintage choices (≤5%, no action
required beyond awareness):** Γ_CO base 1.235e-10 Habing = 2.1e-10 Draine vs
published 2.0e-10 (vDB88; ours matches Lee+96); C photoionization 3.0e-10 vs
G10's 3.1e-10 (UMIST99/vD88 vs Verner+96); H2+e CID uses the newer TT02a fit
vs G10's printed ST99 (deliberate, cited).

**Flag-choice + label bug:** production uses `iflag_3bh2a=1` (ABN02-inverse)
and `iflag_3bh2b=1` (PSS83) for the high-density H2 CID limits where G10
prints LS83/SK87 — all variants exist as options in Glover's own code (matters
only at n≳ncr AND T≳few×10³ K; A3 in the model doc already asks Uli to confirm
ABN02). **Bookkeeping bug found by this check:** `sgchem.cc:45` comments
iflag_3bh2a=1 as "(PSS83)" and `cxx/rates.json` ch1 carries "Inverse of PSS83
three-body rate" — for channel *a*, flag 1 is the ABN02 inverse; PSS83 is
flag 2. Expressions extracted correctly; labels wrong in two places.

**Auditor's trap noted for posterity:** the arXiv *preprint* of WD01 tabulates
different H+/He+ fit rows (H+ C0=3.436…) than the published ApJ table
(C0=12.25…); our code matches the published version — checking against the
preprint would produce a false mismatch.

**UNVERIFIABLE-SOURCE:** Si+ radiative recombination broken-power-law (Glover's
private fit to Nahar 2000 — self-documented with error bounds); NK93 CO
cooling table DATA blocks (formula verified as NK93 eq. 5; primary not
fetchable); Si/Si+ CR ratios (cited L67/L78, not in G10's printed table).

---

## Ship-gate recommendation (what must happen before SGCHEM_CXX could ever default ON)

1. **F1**: publish `lambda`/`lambda_chem` to COOLR in `chem_evolve_cxx` (or
   compile-time forbid SGCHEM_CXX with RADTRANSFER/OUTPUT_INDIVIDUAL_COOLRATES),
   then re-run a twin WITH OUTPUT_INDIVIDUAL_COOLRATES to prove the diagnostics.
2. **F2**: commit cxx/ + scripts; stamp acceptance docs with the SHA; freeze a
   Gate-3 state and re-certify Gates 1/1b/2/4 against it in one session.
3. **F4**: fix the onezone3 tdust carry for the cxx backend; re-run the
   certification grid; require the cold-corner deviation back under the same
   1% bound gate (a) uses, or re-justify the bound.
4. **F3**: repair `golden/build.sh`, `test_rates.cc` includes, `build3.sh` TU
   list (or generate it from the same wildcard the Makefile uses).
5. **F7**: remove or startup-log the CHEMCXX_* environment switches.
6. **F5/F6**: correct D-15, write the missing D-19, align the R10 default with
   the register.
7. Runtime gap (their own honest caveat, still open): no full-physics Stella
   run (SF + feedback + SNe + TNOCHEM traffic) has been made ON vs OFF — the
   Gate-4 boxes are near-equilibrium. An sn20-class A/B remains the missing
   certification before any default flip.
8. Provenance: bring the NL97 Γ_CHx normalization (factor 1.7–3.4 vs
   NL97/GC12) to Uli as a physics decision; fix the "PSS83" mislabels in
   sgchem.cc:45 and rates.json ch1.
9. Timing footnote: the regenerated (Gate-3-state) timing CSV shows net 5 at
   ×0.82 (CVODE slower) vs the accepted ×1.01 — performance parity should be
   re-measured at the frozen SHA too (same F2 class; the claim was hedged as
   not-a-speedup anyway).

Bottom line: the **port quality is high** — every bit-level claim I attacked
survived independent reproduction, and the RHS/driver line audit found nothing
unlogged beyond a literal-rounding nit. What does NOT yet meet the bar is the
**certification process** (unpinned moving sources, broken reproduce paths,
one real ON-path wiring gap in F1, and acceptance documents that describe
binaries which no longer exist). Fix the process, not the port.
