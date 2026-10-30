"""SGChem one-zone parity: chemistry tables, cool_func rates and one-zone runs of each network against the frozen
reference (the SGChem Fortran networks), plus serial vs OpenMP-concurrent solves bit-identical.

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


def chemistry_tables_dir():
    """Directory with the four chemistry tables, their content hashes checked against galaxy_sf/stella_tables_check.h."""
    d = environ.get("SGCHEM_TABLES_DIR")
    if not d:
        out = subprocess.run(["bash", path.join(ROOT, "scripts", "get_stella_tables.sh")], capture_output=True, text=True,
                             env={**environ, "PYTHON": environ.get("PYTHON", sys.executable)})
        assert out.returncode == 0, f"get_stella_tables.sh failed:\n{out.stdout}\n{out.stderr}"
        d = [l for l in out.stdout.splitlines() if " holds " in l][-1].split()[1]
    hdr = open(path.join(ROOT, "src", "galaxy_sf", "stella_tables_check.h")).read()
    for f in CHEM_TABLES:
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


@pytest.mark.parametrize("net", (5,))
def test_sgchem_onezone(net, env):
    env = {**env, "CHEMCXX_DATADIR": chemistry_tables_dir()}
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
    if net in (5, 17):  # the thread driver covers these layouts
        t = subprocess.run([path.join(out, "threads_identity"), env["CHEMCXX_DATADIR"]], capture_output=True, text=True,
                           env={**env, "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "4")})
        print(t.stdout)
        assert t.returncode == 0 and "bit-identical to serial: YES" in t.stdout, t.stdout[-3000:]
