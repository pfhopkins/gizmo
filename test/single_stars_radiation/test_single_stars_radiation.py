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


def run(tag, tables, exe="GIZMO", stars=None):
    import sys
    sys.path.insert(0, str(TEST_DIR))
    from make_single_stars_radiation_ics import make_single_stars_radiation_ics
    d = TEST_DIR / f"run_{tag}"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        if d.exists():
            shutil.rmtree(d)
        (d / "output").mkdir(parents=True)
        L = make_single_stars_radiation_ics(str(d / "output" / "snapshot_000.hdf5"), stars=stars)
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


RHO_UNIT = 1.989e33 / 3.085678e18 ** 3  # g/cm^3 per code density (Msun/pc^3)


def reference_onset_yr(tables_file, M, rho_birth, Z=0.014):
    """compute_single_star_S_ly: no ionising photons before min(t_KH, t_ff(birth density)) (at most 1 Myr, only
    with a birth density) and during the pre-main sequence"""
    with h5py.File(tables_file, "r") as F:
        Zt, Mt = F["Z"][()], F["M_init"][()]
        tpms = F["t_PMS"][()].reshape(len(Zt), len(Mt))[int(np.argmin(np.abs(Zt - Z))), int(np.argmin(np.abs(Mt - M)))]
    t_on = tpms
    if rho_birth > 0:
        t_ff = np.sqrt(3 * np.pi / (32 * 6.672e-8 * rho_birth * RHO_UNIT)) / 3.155e7
        t_on = max(t_on, min(1.5e7 * M ** -2.1, t_ff, 1e6))
    return t_on


def test_ionising_onset(built):
    """10 Msun stars whose onset is set by the pre-main sequence (no birth density, and a dense birth), by the
    Kelvin-Helmholtz time and by the free-fall time: each 0.5% before its onset is dark, 0.5% after it ionises"""
    n_to_rho = lambda n: n * 1.6726e-24 / 0.76 / RHO_UNIT
    t_kh = 1.5e7 * 10.0 ** -2.1
    rho_ff = 3 * np.pi / (32 * 6.672e-8 * (0.957 * t_kh * 3.155e7) ** 2) / RHO_UNIT  # t_ff between t_PMS and t_KH
    cases = [0.0, n_to_rho(1e7), n_to_rho(1e2), rho_ff]
    onsets = [reference_onset_yr(built["StellarTablesFile"], 10.0, r) for r in cases]
    assert len({round(t) for t in onsets}) == 3  # the three branches give three onsets
    stars, k = [], 0
    for r, t in zip(cases, onsets):
        for f in (0.995, 1.005):
            stars.append((10.0, 10.0, f * t, (0.2 + 0.08 * k, 0.5, 0.5), 0, r)); k += 1
    _, hii = run("onset", built, stars=stars)
    assert len(hii) > 0 and np.all(hii[:, 1] == len(cases)), hii[:, 1]  # exactly the stars past their onset
