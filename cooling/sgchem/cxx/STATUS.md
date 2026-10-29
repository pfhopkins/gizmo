# SGCHEM C++/CVODE port — STATUS (2026-07-29)

**Gate 3 COMPLETE and PASSED: the C++ backend contains ZERO Fortran** —
no shims, no `/coolr/` COMMON views, no libgfortran. Gate 2 (solver layer) is
complete and the backend is compiled INTO GIZMO: `SGCHEM` alone selects it (the default);
the legacy Fortran backend needs `SGCHEM_FORTRAN`; `SGCHEM_CXX` is an `#error` (flip 2026-10-29).

- Gate-3 results + PASS/FAIL: **`GATE3_ACCEPTANCE.md`** (this directory)
- Gate-2 results + PASS/FAIL: **`GATE2_ACCEPTANCE.md`**
- Every conscious deviation: **`DEVIATIONS.md`**

**2026-10-29 (branch chem/fortran-free):** until then the GIZMO C++ build still
ran the Fortran init (COOLINMO, CHEMINMO, INIT_TOLERANCES, LOAD_H2_TABLE,
INIT_TEMPERATURE_LOOKUP), read its COMMON scalars and linked all sgchem `.F` objects
and libgfortran; "ZERO Fortran" held only for the standalone golden binary. Now
`sgchem_init` fills `ChemInitParams` and `chem_cxx_tables_init` builds everything in
C++; the Makefile links no `.F`, no Fortran runtime, and neither `chem_backend_fortran.cc`
nor `chem_backend_c_api.cc` (golden harness only). Identity proof:
`golden/build_init_identity.sh` (config, cell template and tables bitwise identical to
the old init). Frozen Fortran reference data and tools: `fortran_reference/`.

Gate-3 headline: **38,918,708 golden comparisons, 0 failures, worst rel
4.2e-13**; `cool_func` bit-exact over 10.7M rate comparisons; the pure C++
binary links with `g++` alone and `nm` finds no Fortran symbol; 64 concurrent
OpenMP solves reproduce the serial answers bit-for-bit; the Gate-2 acceptance
re-run on the pure backend passes all three gates.

## Layout (as of the 2026-07-29 relocation)

| what | where |
|---|---|
| **port source (single source of truth)** | `/orion/ptmp/uli/orion/test/gizmo/cooling/sgchem/cxx/` |
| build copy of the gizmo tree | `/orion/ptmp/uli/orion/fb_tests/_sgchem_cxx_build/gizmo/` |
| SUNDIALS 6.7.0 static build | `/orion/ptmp/uli/orion/fb_tests/_sgchem_cxx_build/sundials/install` |
| goldens (rates + splines) | `/orion/ptmp/uli/orion/fb_tests/_sgchem_cxx_build/golden/` |
| benchmark binaries | `/orion/ptmp/uli/orion/fb_tests/_sgchem_cxx_build/bin/` |
| benchmark CSVs + logs | `/orion/ptmp/uli/orion/fb_tests/_sgchem_cxx_build/gate2_data/`, `gate2_*.log` |
| one-zone driver source | `stella/tests/validation/chemistry_onezone/src/onezone3.c` + `src/build3.sh` |

**Standing rule: NEVER compile inside `test/gizmo`.** rsync to the fb_tests
build dir and build there:
```
rsync -a --exclude='.git' --exclude='*.o' --exclude='output*' --exclude='Config.sh' \
      /orion/ptmp/uli/orion/test/gizmo/ \
      /orion/ptmp/uli/orion/fb_tests/_sgchem_cxx_build/gizmo/
```
`stella` is a figures-only repo; `stella/tests/sgchem_cxx` is now just a
symlink and is NOT the home of this code.

## GIZMO integration (the point of the exercise)

The C++/CVODE chemistry sits behind the Gate-0 seam
(`cooling/sgchem/chem_interface.h`, `ChemInputs`/`ChemOutputs`). In
`do_sgchem_step()` the single solver call is now:

```c
#ifndef SGCHEM_FORTRAN
    chem_evolve_cxx(&timestep, &dl_cgs, &yn, &divv, &energy, abundances, &column_est);
#else
    EVOLVE_ABUNDANCES(&timestep, &dl_cgs, &yn, &divv, &energy, abundances, &column_est);
#endif
```

(Historical, before the 2026-10-29 flip: the C++ backend was opt-in via `SGCHEM_CXX`.) Now `SGCHEM_FORTRAN` is documented in `Template_Config.sh` and **defaults OFF**: with
the flag absent the C++ backend is built; with it nothing C++ is built — no cxx object is compiled, no SUNDIALS symbol
is linked, and the Fortran DVODE path remains the reference implementation.
Details of every canonical-tree change: `DEVIATIONS.md` D-11.

**Both builds verified on Orion** (`SYSTYPE=Orion-OpenMPI`, modules
`gcc openmpi gsl hwloc fftw-mpi hdf5-mpi`, `CHEMISTRYNETWORK=17`):

| build | result |
|---|---|
| `SGCHEM_FORTRAN` (was `SGCHEM_CXX` OFF) | links clean, 4,163,264 B, **0 CVODE symbols**, object list identical to pre-change |
| default `SGCHEM` (was `SGCHEM_CXX` ON) | links clean, 5,631,216 B, 13 cxx objects, **110 CVODE symbols**; `nm` confirms `T chem_evolve_cxx` and `T cxx_evolve_abundances` |

(The ON size/object count grew from the Gate-2 figure of 7 objects / 4.64 MB
because a parallel session has begun Gate 3 — the C++ leaf kernels
`chem_cool_func.cc`, `chem_cool_util.cc`, `chem_calc_photo.cc`,
`chem_calc_temp.cc`, `chem_tables_build.cc` — and those now compile in too.
`CHEMCXX_OBJS` is wildcarded so new files are picked up automatically; the
Fortran shims in `chem_backend_fortran.cc` were renamed `fort_*` by that work
and are retained as the golden reference implementations.)

To build the ON variant without disturbing `Config.sh`:
```
sed 's/^#SGCHEM_FORTRAN  */SGCHEM_FORTRAN   /' Config.sh > Config_fortran.sh
make CONFIG=Config_fortran.sh EXEC=GIZMO_fortran -j16   # legacy Fortran; plain Config.sh = C++
```

SUNDIALS link line (Makefile, under the flag):
```
-L$(SUNDIALS_DIR)/lib64 -lsundials_cvode -lsundials_nvecserial \
  -lsundials_sunmatrixdense -lsundials_sunlinsoldense \
  -lsundials_sunnonlinsolnewton -lsundials_generic
```
`SUNDIALS_DIR` defaults to the fb_tests build; override on the make command
line for other machines. The six port sources get `-std=c++14 -I.` via their
own pattern rule (SUNDIALS' `sundials_context.hpp` needs C++14; the rest of
GIZMO keeps its SYSTYPE standard).

## Gate status

- **Gate 0** — chemistry seam frozen in canonical. Bit-identity PROVEN: 7/7
  snapshot pairs byte-identical vs pristine. Infra: `fb_tests/_gate0_build/`.
- **Gate 1** — rate layer machine-extracted: `rates_gen.{h,cc}` (346 functions),
  `rates.json`, `rates_registry.inc`, `RATE_AUDIT.md`. Goldens vs Fortran:
  **346/346 PASS at 1e-12** (worst 1.4e-14), 20k points/rate incl. every guard
  boundary. Reproduce: `bash <build>/golden/build.sh`.
- **Gate 1b** — the 8 `kind=TABLE` entries resolved and closed: cl1/cl2/cl3/cl6
  are genuine splines and are **ported** (`chem_spline.cc` + `chem_tables_data.h`
  from `tables_extract.py`); cl52/59/60/61 are NOT table-driven in the
  production config (`iflag_atom=2` + `WSS_CIE_COOL`) and need no port — that
  claim is *tested*, not asserted. Goldens vs the real `COOLINMO`-built `cltab`:
  **40000/40000 PASS at 1e-12** (worst 1.65e-14, both nets).
  Reproduce: `bash <build>/golden/build_splines.sh`.
- **Gate 2** — solver layer ported, compiled into GIZMO, and benchmarked.
  All three acceptance gates PASS; see `GATE2_ACCEPTANCE.md`.
- **Gate 3** — leaf kernels + table builders ported; the path is Fortran-free.
  38,918,708 golden comparisons, 0 failures, worst 4.2e-13. Acceptance re-run
  on the pure backend: (a) 40/40 within 1%, max 9.5e-3; (b) 2400 trajectory
  points, max 8.8e-4; (c) 160/160 zones, zero failures. See
  `GATE3_ACCEPTANCE.md`. Reproduce: `bash golden/build_gate3.sh`,
  `bash golden/build_purecxx.sh`.

## Files in this directory

| file | role |
|---|---|
| `chem_types.h` | `ChemCell` (per-cell context, audit R2) + `ChemTables` (view over `/coolr/`) + kernel interfaces |
| `chem_rhs_common.h` | `validate_input` on a local copy (R3) + shared grain-recombination term |
| `chem_rhs_net5.cc` | `rate_eq_simple.F` port (net 5) |
| `chem_rhs_net17.cc` | `rate_eq_ulli.F` port (net 17) + `comp_equilib` |
| `cvode_driver.cc` | `evolve_abundances.F` port on SUNDIALS CVODE (BDF + dense DQ Jacobian) |
| `chem_spline.cc` | `spline.F` port + cl1/cl2/cl3/cl6 table construction |
| `chem_backend_fortran.cc` | **golden harness only** after Gate 3: binds `ChemTables`/`ChemCell` to the COMMONs and exposes the `fort_*` reference kernels |
| `chem_cool_func.cc`, `chem_cool_util.cc`, `chem_calc_photo.cc`, `chem_calc_temp.cc` | Gate-3 leaf kernels (`cool_func.F`, `cool_util.F`, `calc_photo.F`, `calc_temp.F`+`compute_gamma.F`, `wss_z_collis.F`) |
| `chem_tables_build.cc` | Gate-3 table layer (`coolinmo.F`+`cheminmo.F`+`photoinit_ism.F`+`const_rates.F`+the file readers) |
| `chem_backend_cxx.cc` | the pure-C++ backend (immutable tables + per-cell context + thread-safe C API) |
| `golden/` | Gate-3 goldens and build scripts |
| `chem_backend_c_api.cc` | plain-C entry points (`chemcxx_init/evolve/stats/tdust`) |
| `chem_gizmo_backend.cc` | **the GIZMO entry point** `chem_evolve_cxx()` behind the Gate-0 seam |
| `rates_gen.{h,cc}`, `rates.json`, `rates_registry.inc`, `extract_rates.py` | Gate-1 rate layer |
| `tables_extract.py`, `chem_tables_data.h` | Fortran `DATA` arrays, extracted with correct literal KINDS (DEVIATIONS D-10) |
| `gate2_bench.py`, `gate2_timing.py`, `gate2_summary.py` | the acceptance benchmark |
| `RATE_AUDIT.md`, `DEVIATIONS.md`, `GATE2_ACCEPTANCE.md`, `STATUS.md` | documentation |

## Reproduce everything

```bash
# 1. refresh the build copy (never build in test/gizmo)
rsync -a --exclude='.git' --exclude='*.o' --exclude='output*' --exclude='Config.sh' \
      /orion/ptmp/uli/orion/test/gizmo/ \
      /orion/ptmp/uli/orion/fb_tests/_sgchem_cxx_build/gizmo/

# 2. GIZMO, both ways
cd /orion/ptmp/uli/orion/fb_tests/_sgchem_cxx_build/gizmo
module purge && module load gcc openmpi gsl hwloc fftw-mpi hdf5-mpi
export SYSTYPE=Orion-OpenMPI
make clean && make -j16                # SGCHEM_FORTRAN line absent/present in Config.sh

# 3. one-zone benchmark driver (compiles the SAME cxx/ sources)
bash /orion/ptmp/uli/orion/stella/tests/validation/chemistry_onezone/src/build3.sh

# 4. Gate-1/1b goldens
bash /orion/ptmp/uli/orion/fb_tests/_sgchem_cxx_build/golden/build_splines.sh

# 5. Gate-3 goldens: leaf kernels + table builder + Fortran-free proof
CX=/orion/ptmp/uli/orion/test/gizmo/cooling/sgchem/cxx
bash $CX/golden/build_gate3.sh        # phases A(tables) B(kernels/F tables) C(kernels/C++ tables) D(iflag_atom=1)
bash $CX/golden/build_purecxx.sh      # links with g++ only; nm/ldd audit; OpenMP determinism

# 6. acceptance gates on the Gate-3 backend
bash $CX/golden/build_gate2_rerun.sh  # dual-backend onezone3 with the pure C++ backend
cd /orion/ptmp/uli/orion/fb_tests/_sgchem_cxx_build
python3 $CX/gate2_bench.py all && python3 $CX/gate2_timing.py && python3 $CX/gate2_summary.py
```
Artefacts: `fb_tests/_sgchem_cxx_build/gate3/gate3_golden_net{5,17}.txt`,
`purecxx/purecxx_net{5,17}.txt`, `gate3_summary.txt`, `gate3_timing.log`.

## Gate 3 — leaf kernels + table builders (2026-07-29)

| file | ports |
|---|---|
| `chem_cool_func.cc` | `cool_func.F` (all 28 rates, WSS CIE path, He-state override, the `rates(3)` zeroing) + `get_coolrates_wssCIE` (`wss_z_collis.F`) |
| `chem_cool_util.cc` | `cool_util.F`: `co_cool`, `h2o_rot_cool`, `h2o_vib_cool`, `three_level_pops`, `compute_stim`, `compute_h2_opacity`, `calc_dust_temp` (with the 2026-07 bounded-Newton fix) |
| `chem_calc_photo.cc` | `calc_photo.F`: `calc_photo`, `chi`, DB96 H2 self-shielding incl. the `iphoto=6` TREECOL path, HD shielding, CO/H2 line shielding, dust attenuation |
| `chem_calc_temp.cc` | `calc_temp.F` (`lookup_temp_gamma`, `calc_spec`, `calc_gamma_temp`) + `compute_gamma.F` |
| `chem_tables_build.cc` | `coolinmo.F` + `cheminmo.F` + `photoinit_ism.F` + `const_rates.F` + the `wss_z_collis.txt` reader + `load_H2_table` + `init_temperature_lookup` — `ChemTables` is **built**, not viewed |
| `chem_backend_cxx.cc` | the pure-C++ backend: immutable process-global tables, per-cell `ChemCell`, thread-safe C API |
| `chem_backend_fortran.cc` | **golden harness only** now — binds `ChemTables` to the COMMONs and exposes the `fort_*` reference wrappers |
| `golden/test_gate3.cc`, `golden/build_gate3.sh` | goldens: A tables, B kernels on Fortran tables, C kernels on C++ tables, D `iflag_atom=1` |
| `golden/test_purecxx.cc`, `golden/build_purecxx.sh` | Fortran-free link + `nm`/`ldd` audit + OpenMP determinism |
| `golden/build_gate2_rerun.sh` | rebuilds `onezone3` with the Gate-3 backend |

Audit **R2 closed** (ChemTables immutable-after-init; ChemCell per cell; CVODE
workspace + stats `thread_local`; 64 concurrent solves bit-identical to serial)
and **R10 applied** (one extra RHS evaluation at the accepted state so the
published `lambda` breakdown describes the state that was returned;
`CHEMCXX_LEGACY_LAMBDA=1` restores the Fortran semantics).

Six bugs were found *by* the Gate-3 goldens — five Fortran default-REAL literal
kinds (the DEVIATIONS D-10 class) and one silent build bug where the net-5
objects picked up the canonical net-17 `GIZMO_config.h`. All fixed and
documented in DEVIATIONS D-16.

## Next steps

1. **Gate 4, the remaining wiring.** A GIZMO C++-backend build still links
   the Fortran objects because `do_sgchem_step()` uses
   `chem_commit_to_fortran()` + `CALC_PHOTO_WRAPPER` as the INPUT transport
   (and because the Fortran fallback is deliberately retained). Closing it:
   fill `ChemCell` directly from `struct ChemInputs` and call
   `cxx_calc_photo()`, i.e. route `chem_gizmo_backend.cc` at
   `chem_backend_cxx.cc` instead of `chem_backend_c_api.cc`. Everything needed
   already exists and is tested. `OPENMP` + the C++ backend becomes legal at
   that point — the C++ side is already safe.
2. **Runtime validation in GIZMO.** The flag builds and the one-zone benchmark
   passes, but no full GIZMO run has yet been made with the C++ backend.
   Next: an A/B on a short Stella test (e.g. the sn20 bench) comparing
   snapshots between OFF and ON.
3. **Flip the default only after (1) and (2)**; the Fortran path stays
   available as the fallback (Uli's standing requirement).
4. **Network 15** (NL99) RHS + driver ported 2026-10-19 (DEVIATIONS D-22);
   `chem_gizmo_backend.cc` accepts 5/15/17 and `#error`s on the rest.
   **Networks other than 5/17** in the C++ `calc_photo` are not ported
   (DEVIATIONS D-14); GIZMO still uses the Fortran `CALC_PHOTO_WRAPPER`.
5. **Flag to Uli separately from the port:** `co_cool` indexes the CO
   *vibrational* table with the *rotational* column (DEVIATIONS D-13, latent
   Fortran bug, reproduced literally), and the dense-cold T_eq spread of ~1%
   traces to `calc_dust_temp`'s own 1% Newton convergence criterion
   (`cool_util.F:972`), not to the port.

## Ground rules

Never compile inside `test/gizmo`. The Fortran path is the reference
implementation and stays the default. Reference docs:
`stella/CHEMICAL_MODEL_NETS_5_17.md`, `stella/SGCHEM_CONTROL_FLOW_AUDIT.md`,
`stella/SGCHEM_CXX_CVODE_ASSESSMENT.md`.
