"""SGChem one-zone parity: chemistry tables, cool_func rates and one-zone runs of every network against the frozen
reference (nets 3/5/15/16/17: the SGChem Fortran; nets 18/20: the C++ backend when it was introduced), plus serial vs
OpenMP-concurrent solves bit-identical (nets 5, 17). Network 5 with ionising photons on molecular gas (euv_h2.cc): the
H2 channel adds exactly its rate and heating, and a molecular cell in an EUV field loses its H2 and ionises; without the
H2 channel (r = 0) it does not.

Standalone: no GIZMO run. Needs mpicxx, SUNDIALS_DIR (6.x or 7.x), KOKKOS_PATH, the four chemistry tables
(SGCHEM_TABLES_DIR, else installed with scripts/get_stella_tables.sh) and the frozen reference data
(SGCHEM_REFERENCE_DATA, checked against cooling/sgchem/cxx/fortran_reference/data.sha256).
Pass criteria are those of reference.py (tables 1e-6, cool_func 1e-10, one-zone final T 1e-2).
"""

import re
import subprocess
import sys
from os import environ, path, makedirs

import pytest

HERE = path.dirname(path.abspath(__file__))
ROOT = path.dirname(path.dirname(HERE))
REFDIR = path.join(ROOT, "src", "cooling", "sgchem", "cxx", "fortran_reference")
CHEM_TABLES = ("energy-table-x.txt", "eh2-table.txt", "H2-cooling-ratios.dat", "wss_z_collis.txt")


def chemistry_tables_dir(net):
    """Directory with the four chemistry tables, their content hashes checked against galaxy_sf/stella_tables_check.h."""
    d = environ.get("SGCHEM_TABLES_DIR")
    if not d:
        out = subprocess.run(["bash", path.join(ROOT, "scripts", "get_stella_tables.sh")], capture_output=True, text=True,
                             env={**environ, "PYTHON": environ.get("PYTHON", sys.executable)})
        assert out.returncode == 0, f"get_stella_tables.sh failed:\n{out.stdout}\n{out.stderr}"
        d = [l for l in out.stdout.splitlines() if " holds " in l][-1].split()[1]
    hdr = open(path.join(ROOT, "src", "galaxy_sf", "stella_tables_check.h")).read()
    for f in CHEM_TABLES + (("sgchem_net18_tables.bin",) if net == 18 else ()):
        want = re.search(r'\{"' + re.escape(f) + r'",\s*"([0-9a-f]+)"\}', hdr).group(1)
        got = subprocess.run([sys.executable, "-I", path.join(ROOT, "scripts", "stella_tables_hash.py"), path.join(d, f)],
                             capture_output=True, text=True).stdout.split()
        assert got and got[0] == want, f"{d}/{f}: hash {got[:1]}, expected {want}"
    return d


@pytest.fixture(scope="module")
def env():
    missing = [v for v in ("SUNDIALS_DIR", "KOKKOS_PATH", "SGCHEM_REFERENCE_DATA") if not environ.get(v)]
    if missing:
        pytest.skip(f"set {', '.join(missing)}")
    data = environ["SGCHEM_REFERENCE_DATA"]
    chk = subprocess.run(["sha256sum", "-c", "--quiet", path.join(REFDIR, "data.sha256")], cwd=data, capture_output=True, text=True)
    assert chk.returncode == 0, f"reference data in {data} do not match data.sha256:\n{chk.stdout}{chk.stderr}"
    return environ


@pytest.mark.parametrize("net", (3, 5, 15, 16, 17, 18, 20))
def test_sgchem_onezone(net, env):
    env = {**env, "CHEMCXX_DATADIR": chemistry_tables_dir(net)}
    out = path.join(HERE, "output", f"net{net}")
    makedirs(out, exist_ok=True)
    b = subprocess.run(["bash", path.join(HERE, "build_onezone_dump.sh"), str(net), out], capture_output=True, text=True, env=env)
    assert b.returncode == 0, f"build/run failed:\n{b.stdout[-4000:]}\n{b.stderr[-4000:]}"
    ref = path.join(env["SGCHEM_REFERENCE_DATA"], f"fortran_net{net}")
    if not path.isdir(ref):
        ref = path.join(env["SGCHEM_REFERENCE_DATA"], f"cxx_net{net}")
    c = subprocess.run([sys.executable, path.join(REFDIR, "reference.py"), "compare", ref, path.join(out, "data")],
                       capture_output=True, text=True)
    print(c.stdout)
    assert c.returncode == 0, f"net {net} differs from {ref}:\n{c.stdout[-4000:]}"
    if net in (5, 17):
        t = subprocess.run([path.join(out, "threads_identity"), env["CHEMCXX_DATADIR"]], capture_output=True, text=True,
                           env={**env, "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "4")})
        print(t.stdout)
        assert t.returncode == 0 and "bit-identical to serial: YES" in t.stdout, t.stdout[-3000:]


@pytest.mark.parametrize("nH", (10.0, 100.0))
def test_sgchem_net5_euv_h2(nH, env):
    """Gamma_HI = 1e-9 s^-1, <h nu> = 18.6 eV, r = sigma_H2/sigma_HI = 1.34 (stella_dev DEVIATIONS D-56), WSS CIE cooling;
    G0 = 1.7 as stella_dev's test, and G0 = 1e-4 (shielded), where only the H2 channel clears the H2"""
    tables = chemistry_tables_dir(5)
    out = path.join(HERE, "output", "net5")
    if not path.exists(path.join(out, "euv_h2")):
        makedirs(out, exist_ok=True)
        b = subprocess.run(["bash", path.join(HERE, "build_onezone_dump.sh"), "5", out], capture_output=True, text=True,
                           env={**env, "CHEMCXX_DATADIR": tables})
        assert b.returncode == 0, f"build failed:\n{b.stdout[-4000:]}\n{b.stderr[-4000:]}"
    run = lambda r, g0: subprocess.run([path.join(out, "euv_h2"), tables, "1", "1e-9", "18.6", str(r), str(nH), str(g0)],
                                       capture_output=True, text=True)
    for g0 in (1.7, 1e-4):  # unshielded, and shielded from the FUV (G0 = 0 would switch every photo-rate off)
        t = run(1.34, g0)
        print(f"G0 = {g0}:", [l for l in t.stdout.splitlines() if l.startswith(("RHS", "RESULT", "EVOL "))])
        assert t.returncode == 0 and "RHS" in t.stdout and "EVOL PASS" in t.stdout, t.stdout[-3000:]
    t0 = run(0.0, 1e-4)  # without the H2 channel (the network before this change) shielded molecular gas stays molecular
    print("no H2 channel, G0 = 1e-4:", [l for l in t0.stdout.splitlines() if l.startswith("RESULT")])
    assert "EVOL FAIL" in t0.stdout
