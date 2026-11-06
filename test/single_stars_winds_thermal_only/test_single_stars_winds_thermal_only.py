"""single_stars_winds_thermal_only: GALSF_SINGLE_STARS_WINDS_THERMAL_ONLY gives the mass-return mode no kick: the whole
1/2 M v^2 of winds and AGB ejecta lands as heat, mass and metals as usual. 12^3 cells of 1 Msun, SGChem, 2 ranks, about
100 yr; a 35 Msun star halfway through its life, or a 3 Msun star at its AGB death.

Bars, written before the runs: the radial gas momentum about the star is below 1/4 of the booked sqrt(2 M E) (with a kick it is
about 1; the hydro response of the heated gas grows as v t / 2R, about 0.1 here); the heat row equals the ejecta energy row to 1 %;
the kinetic row is below 1e-3 of it; the mass delivered to the gas equals the mass taken from the star to 1e-12.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "single_stars_winds_thermal_only"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402
from stella_table_bands import Tables  # noqa: E402

L, N_SIDE = 12.0, 12
ERG_PER_MSUN_KMS2 = 1.989e43


@pytest.fixture(scope="module")
def built():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    tables = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    build_gizmo_for_test(TEST_NAME)
    return tables


def run(tag, tables, M, age):
    d = TEST_DIR / f"output_{tag}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    x0 = np.full(3, 0.5 * L + 0.17)
    make_treecol_pixels_ics(str(d / "snapshot_000.hdf5"), L, lattice(N_SIDE, L), [(M, age, x0)], hsml=2.0)
    params = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    for k, v in (("StellarTablesFile", tables["StellarTablesFile"]), ("SGChemTablesPath", os.environ.get("SGCHEM_TABLES_DIR", os.path.dirname(tables["StellarTablesFile"])))):
        params = params.replace(f"{k:35s}SET_BY_TEST", f"{k:35s}{v}")
    params = params.replace("OutputDir                          output", f"OutputDir                          {d.name}")
    (d / "run.params").write_text(params)
    with open(d / "run.out", "w") as out:
        subprocess.run(mpi_launch_command(2, 1) + [str(TEST_DIR / "GIZMO"), str(d / "run.params"), "2", "0"], cwd=TEST_DIR, stdout=out, stderr=subprocess.STDOUT, check=True)
    lines = (d / "ledger.txt").read_text().splitlines()
    head = [t for t in lines[0].split()[2:] if t.endswith(("_Msun", "_erg"))]
    row = np.atleast_2d(np.loadtxt(d / "ledger.txt"))[-1]
    b = {k: row[j + 1] for j, k in enumerate(head)}
    with h5py.File(sorted(d.glob("snapshot_*.hdf5"))[-1]) as f:
        dr = f["PartType0/Coordinates"][:] - x0
        rhat = dr / np.linalg.norm(dr, axis=1)[:, None]
        p_rad = (f["PartType0/Masses"][:] * (f["PartType0/Velocities"][:] * rhat).sum(1)).sum()
    return b, p_rad


def check(b, p_rad):
    M, E = b["M_single_star_ejecta_Msun"], b["E_single_star_ejecta_erg"]
    p_booked = np.sqrt(2 * M * E / ERG_PER_MSUN_KMS2)
    heat, kin = b["E_fb_thermal_now_erg"] + b["E_fb_thermal_deferred_erg"], b["E_fb_kinetic_erg"]
    print(f"M_ej {M:.4e} Msun, E_ej {E:.4e} erg, p_rad/p_booked {p_rad / p_booked:.4f}, heat/E_ej {heat / E:.5f}, kinetic/E_ej {kin / E:.3e}, M_to_gas/M_ej-1 {b['M_fb_to_gas_Msun'] / M - 1:.2e}")
    assert M > 0 and E > 0
    assert abs(p_rad) < 0.25 * p_booked
    assert abs(heat / E - 1) < 0.01
    assert abs(kin) < 1e-3 * E
    assert abs(b["M_fb_to_gas_Msun"] / M - 1) < 1e-12


def test_wind_is_heat(built):
    tab = Tables(built["StellarTablesFile"])
    lM, lZ = np.log10(35.0), np.log10(0.0142)
    check(*run("wind", built, 35.0, tab._i2(tab.t_pms, lM, lZ) + 0.5 * tab._i2(tab.life, lM, lZ)))


def test_agb_death_is_heat(built):
    tab = Tables(built["StellarTablesFile"])
    lM, lZ = np.log10(3.0), np.log10(0.0142)
    check(*run("agb", built, 3.0, tab._i2(tab.t_pms, lM, lZ) + tab._i2(tab.life, lM, lZ) + 10.0))
