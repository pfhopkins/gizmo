"""single_stars_radiation test: IMF-sampled single stars as sources of the HII-region loop, with the ionising photon
rate of the stellar-evolution tables. See README.md."""
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "single_stars_radiation"
TEST_DIR = Path(__file__).parent
X_H = 0.76  # HYDROGEN_MASSFRAC


@pytest.fixture(scope="module")
def built():
    tables = get_stellar_evolution_tables()
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME)
    return tables


def run(tag, tables, exe="GIZMO"):
    import sys
    sys.path.insert(0, str(TEST_DIR))
    from make_single_stars_radiation_ics import make_single_stars_radiation_ics
    d = TEST_DIR / f"run_{tag}"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        if d.exists():
            shutil.rmtree(d)
        (d / "output").mkdir(parents=True)
        L = make_single_stars_radiation_ics(str(d / "output" / "snapshot_000.hdf5"))
        params = Path(TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"])
        params = params.replace("BoxSize                            45.0", f"BoxSize                            {L}")
        (d / f"{TEST_NAME}.params").write_text(params)
        get_cooling_tables(str(TEST_DIR))
        for f in ("TREECOOL", "spcool_tables"):
            os.symlink(TEST_DIR / f, d / f)
        with open(d / f"test_{TEST_NAME}.out", "w") as out:
            subprocess.run(mpi_launch_command(2) + [str(TEST_DIR / exe), f"{TEST_NAME}.params", "2", "0"],
                           cwd=d, stdout=out, stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    return None, np.atleast_2d(np.loadtxt(d / "output" / "HIIheating.txt"))


def table_lum(tables_file, name, log_age, M=35.0, Z=0.014):
    """a table luminosity of a star at a (Z, M) grid node: the log value linear in log age, as the code"""
    with h5py.File(tables_file, "r") as F:
        Zt, Mt, la = F["Z"][()], F["M_init"][()], F["log_age_yr"][()]
        iz, im = int(np.argmin(np.abs(Zt - Z))), int(np.argmin(np.abs(Mt - M)))
        tpms = F["t_PMS"][()].reshape(len(Zt), len(Mt))[iz, im]
        v = F[name][iz, im, :].astype(np.float64)
    return 10 ** np.interp(log_age(tpms), la, v)


def expected(tables_file, t_code):
    from make_single_stars_radiation_ics import STAR_AGE_YR, PC_KMS_IN_YR
    la = lambda tpms: np.log10(max(STAR_AGE_YR + t_code * PC_KMS_IN_YR - tpms, 100.0))
    get = lambda n: table_lum(tables_file, n, la)
    nuv, nuv_lo = get("log_L_NUV"), get("log_L_NUV_lo")
    bol = get("log_L_ion_tot") + get("log_L_FUV_M1") + (nuv - nuv_lo) + nuv_lo + get("log_L_OPT_NIR")
    return bol, nuv - nuv_lo, get("log_Q_ion")


def test_hii_photon_rate(built):
    _, hii = run("default", built)
    rows = hii[hii[:, 1] > 0]  # steps with an ionising source: exactly one (the remnant emits nothing)
    assert len(rows) > 0 and np.all(rows[:, 1] == 1)
    for t, ndot in zip(rows[:, 0], rows[:, 2]):  # the loop's log reports Ndot / X, to 6 digits
        _, _, q = expected(built["StellarTablesFile"], t)
        assert abs(ndot * X_H / q - 1) < 1e-5, (ndot * X_H, q)
