"""stella_sfr_log test: sfr.txt (rows on full-sync steps) books the Stella stars formed since the previous row. Stella cloud
production flags and parameters, cloud IC from STELLA_CLOUD_IC (a MakeCloud cloud, e.g. gmc_Z1_m1.0, 12377 cells),
2.7 Myr, 2 ranks. See README.md."""
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "stella_sfr_log"
TEST_DIR = Path(__file__).parent
TIME_MAX = 0.0028
UNIT_MASS_MSUN = 1.0e10  # stella_cloud.params: 1.989e43 g
UNIT_TIME_YR = 3.085678e21 / 1.0e5 / 3.15576e7  # kpc / (km/s)


@pytest.fixture(scope="module")
def run():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    if not os.environ.get("STELLA_CLOUD_IC"):
        pytest.skip("set STELLA_CLOUD_IC to a Stella cloud IC (HDF5)")
    T = get_stellar_evolution_tables()["StellarTablesFile"]
    get_cooling_tables(str(TEST_DIR))
    build_gizmo_for_test(TEST_NAME)
    d = TEST_DIR / "output"
    shutil.rmtree(d, ignore_errors=True)
    (d / "output").mkdir(parents=True)
    ic = Path(os.environ["STELLA_CLOUD_IC"])
    os.symlink(ic, d / ic.name)
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    sub = {"InitCondFile": ic.name[:-5] if ic.name.endswith(".hdf5") else ic.name, "StellarTablesFile": T,
           "SGChemTablesPath": os.environ.get("SGCHEM_TABLES_DIR", os.path.dirname(T))}
    p = [f"{l.split()[0]} {sub[l.split()[0]]}" if l.split() and l.split()[0] in sub else l
         for l in (TEST_DIR / f"{TEST_NAME}.params").read_text().splitlines()]
    (d / f"{TEST_NAME}.params").write_text("\n".join(p) + "\n")
    with open(d / "run.out", "w") as f:
        subprocess.run(mpi_launch_command(2) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params"], cwd=d, stdout=f,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    return d / "output"


def test_sfr_log_counts_stella_stars(run):
    """bar (written before the run): summed over the rows up to the last one (at TimeMax), column 5 equals the collected
    masses of all stars in single_star_births.txt (ok rows) to 1e-3, and column 4 times the interval since the previous
    row gives the same mass to 1e-3"""
    sfr = np.loadtxt(run / "sfr.txt", comments="#", ndmin=2)
    b = np.loadtxt(run / "single_star_births.txt", comments="#", ndmin=2)
    assert abs(sfr[-1, 0] / TIME_MAX - 1) < 1e-9, "no sfr.txt row at TimeMax"
    m_formed = b[b[:, 8] == 1, 5].sum()
    m_logged = sfr[:, 4].sum() * UNIT_MASS_MSUN
    dt_yr = np.diff(np.concatenate([[0.0], sfr[:, 0]])) * UNIT_TIME_YR
    m_rate = np.sum(sfr[:, 3] * dt_yr)
    print(f"{len(sfr)} sfr.txt rows ({np.count_nonzero(sfr[:, 4])} non-zero); stars formed {int((b[:, 8] == 1).sum())}, "
          f"{m_formed:.3f} Msun; column 5 {m_logged:.3f}, column 4 x dt {m_rate:.3f}")
    assert m_formed > 1.0, "the cloud formed no stars"
    assert abs(m_logged - m_formed) < 1e-3 * m_formed, "sfr.txt column 5 misses Stella stars"
    assert abs(m_rate - m_formed) < 1e-3 * m_formed, "sfr.txt column 4 is not the formed mass per time"
