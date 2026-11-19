"""One electron fraction and one hydrogen mass fraction, consistent with the chemistry (audit row 19 and core C6/C8), behind
SGCHEM_ELECTRONS_AND_HYDROGEN_FRACTION_FROM_NETWORK (off by default; Config_stella_default.sh + the flag), net 5.

Definitions the flag makes every site use:
- electrons per H = the net-5 right-hand side's abe = x(H+) + (x_C - x(CO)) + x_Si (C in CO is neutral); stored as CellP.Ne by the
  cooling path and returned by ThermalProperties (it took H+ only);
- Get_Gas_Ionized_Fraction = the H ionised fraction x(H+), what the CR consumers use as 1 - n_H0 (it returned Ne - x(CO));
- HYDROGEN_MASSFRAC = 1/(1 + 4 ABHE) = 0.714285..., what the chemistry and mu use (it was 0.7065).
For every cell of the small Stella box after one cooling call: Ne = abe_rhs, ThermalProperties' ne = Ne, Get_Gas_Ionized_Fraction =
x(H+) (all to 1e-12 relative), and HYDROGEN_MASSFRAC = 1/1.4 (1e-15)."""

import os
import shutil
import subprocess
import sys
import tempfile
from os.path import abspath, dirname, join

import pytest

from gizmo.test import build_gizmo_for_test, link_test_driver, mpi_launch_command

TEST_NAME = "sgchem_unified_fractions"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
DRIVER = join(TEST_DIR, "output", f"{TEST_NAME}_driver")


def test_sgchem_unified_fractions():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO; nothing to validate from external snapshots")
    cwd = os.getcwd(); os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME, post_build=lambda: link_test_driver(join(TEST_DIR, f"{TEST_NAME}_driver.cc"), DRIVER))
    finally:
        os.chdir(cwd)
    run = tempfile.mkdtemp(prefix=f"{TEST_NAME}_"); os.makedirs(join(run, "output")); os.symlink(REPO, join(run, "gizmo"))
    subprocess.run([sys.executable, join(REPO, "test", "sgchem_cooling_path", "make_sgchem_cooling_path_ics.py"), join(run, f"{TEST_NAME}_ics.hdf5")], check=True)
    r = subprocess.run(mpi_launch_command(1) + [DRIVER, join(TEST_DIR, f"{TEST_NAME}.params")], capture_output=True, text=True, cwd=run)
    with open(join(TEST_DIR, "output", "driver.txt"), "w") as f:
        f.write(r.stdout + r.stderr)
    shutil.rmtree(run, ignore_errors=True)
    xh = [float(l.split()[1]) for l in r.stdout.splitlines() if l.startswith("XH")]
    rows = [[float(v) for v in l.split()[1:]] for l in r.stdout.splitlines() if l.startswith("EL")]
    assert xh and rows, "driver did not finish:\n" + (r.stdout + r.stderr)[-3000:]
    rel = lambda a, b: abs(a - b) / max(abs(b), 1e-300)
    e_rhs = max(rel(ne, abe) for ne, abe, xi, hp, nt in rows)
    e_ion = max(rel(xi, hp) for ne, abe, xi, hp, nt in rows)
    e_thm = max(rel(nt, ne) for ne, abe, xi, hp, nt in rows)
    print(f"{len(rows)} cells: Ne vs RHS abe {e_rhs:.2e}, ionised fraction vs H+ {e_ion:.2e}, ThermalProperties ne vs Ne {e_thm:.2e}, X_H {xh[0]:.15f}")
    assert e_rhs < 1e-12 and e_ion < 1e-12 and e_thm < 1e-12
    assert abs(xh[0] - 1 / 1.4) < 1e-15
