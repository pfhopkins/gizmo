"""Pop III one-zone collapse: pristine gas (net 17, C++ SGChem, z = 17, primordial He/H = 0.079) in free fall.
The driver evaluates the code's Pop III criterion (galaxy_sf/resolvedism_popIII.h) and the Remy-Ruyer+14 DGR
(declarations/resolvedism_dgr_rr14.h) along the track. Checks: Z = 0 loiters at 150-250 K near n = 1e3 (Bromm+02);
the Pop III criterion first passes at n_H = 1e6-3e6 (the design density), while the Stella Jeans criterion
(M_J < 32 Msun) only passes above 1e9; gas at 1e-3 Zsun is not pristine and fragments at the CMB floor (n < 1e6).
Needs SUNDIALS_DIR (CVODE >= 6) and CHEMCXX_DATADIR (the SGChem tables); skipped without them.
"""

import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).parent
REPO = TEST_DIR.parent.parent
SKIP_SOURCES = {"chem_gizmo_backend", "chem_backend_fortran", "chem_backend_c_api"}
SUNDIALS_LIBS = ["-lsundials_cvode", "-lsundials_nvecserial", "-lsundials_sunmatrixdense", "-lsundials_sunlinsoldense",
                 "-lsundials_sunnonlinsolnewton", "-lsundials_generic"]


def _env_dir(name):
    d = os.environ.get(name)
    if not d or not Path(d).is_dir():
        pytest.skip(f"{name} not set to a directory")
    return Path(d)


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    sun = _env_dir("SUNDIALS_DIR")
    libdir = next(p for p in (sun / "lib64", sun / "lib") if p.is_dir())
    obj = tmp_path_factory.mktemp("popIII_onezone_obj")
    # the chemistry sources include ../../../GIZMO_config.h: build a copy next to the zoom's chemistry flags
    shutil.copytree(REPO / "cooling" / "sgchem", obj / "cooling" / "sgchem")
    shutil.copytree(REPO / "declarations", obj / "declarations")
    (obj / "GIZMO_config.h").write_text("#define SGCHEM\n#define CHEMISTRYNETWORK 17\n#define CO_SHIELDING\n#define WSS_CIE_COOL\n#define RT_USE_TREECOL_FOR_NH 12\n#define NEED_HEALPIX\n")
    cxx = obj / "cooling" / "sgchem" / "cxx"
    flags = ["c++", "-O2", "-std=c++14", f"-I{cxx}", f"-I{REPO}", f"-I{sun / 'include'}"]
    srcs = [s for s in sorted(cxx.glob("*.cc")) if s.stem not in SKIP_SOURCES]
    srcs.append(TEST_DIR / "popIII_onezone_driver.cc")

    def compile_one(s):
        o = obj / (s.stem + ".o")
        r = subprocess.run(flags + ["-c", str(s), "-o", str(o)], capture_output=True, text=True)
        assert r.returncode == 0, f"compile failed: {s.name}\n{r.stderr[-3000:]}"
        return str(o)

    with ThreadPoolExecutor(max_workers=4) as ex:
        objs = list(ex.map(compile_one, srcs))
    exe = obj / "popIII_onezone_driver"
    r = subprocess.run(["c++"] + objs + [f"-L{libdir}", f"-Wl,-rpath,{libdir}"] + SUNDIALS_LIBS + ["-lm", "-o", str(exe)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return exe


def run_track(exe, tmp_path, Z, dgr="rr14", n_end=1e10):
    data = _env_dir("CHEMCXX_DATADIR")
    out = tmp_path / f"Z{Z}_{dgr}.csv"
    r = subprocess.run([str(exe), str(data), str(Z), dgr, str(out), "17", str(n_end)], capture_output=True, text=True,
                       env=dict(os.environ, ABHE="0.079"), timeout=600)
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
    assert r.returncode == 0 and line, r.stdout + r.stderr
    res = dict(re.findall(r"(\w+)=(\S+)", line[0]))
    print(line[0])
    assert int(res["rc"]) == 0, "chemistry solver failed"
    return {k: float(v) for k, v in res.items() if k not in ("dgr",)}


def test_pristine_gas_forms_popIII_at_design_density(driver, tmp_path):
    r = run_track(driver, tmp_path, 0.0)
    assert r["pristine"] == 1
    assert 150 <= r["Tmin_1e2_1e4"] <= 250, "Z = 0 loitering minimum near n = 1e3"
    assert 1e6 <= r["n_popIII"] <= 3e6, "Pop III criterion should first pass just above n_H = 1e6"
    assert r["n_MJ32"] > 1e9, "the Stella Jeans criterion (M_J < 32 Msun) should only pass above 1e9 in pristine gas"


def test_enriched_gas_is_not_popIII(driver, tmp_path):
    r = run_track(driver, tmp_path, 1e-3)
    assert r["pristine"] == 0 and r["n_popIII"] < 0
    assert 0 < r["n_MJ32"] < 1e6, "1e-3 Zsun gas (RR14 DGR) fragments at the CMB floor via the Stella criterion"


def test_linear_dgr_delays_fragmentation_at_1e_3_zsun(driver, tmp_path):
    r = run_track(driver, tmp_path, 1e-3, dgr="lin")
    assert r["n_MJ32"] > 1e7, "linear DGR: grain H2 formation heating keeps 1e-3 Zsun gas warm (design report: 2.7e7)"
