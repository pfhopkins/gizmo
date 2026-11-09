"""sgchem_rt_slab: a slab lit by one star through the moment solver (reports/design/design_m1_coupling.md), the IC of
treecol_pixels test_slab_shielding: n_H ~ 1e3 slab, N_H ~ 1e22 through it, 40 Msun star 3 pc in front of the face.
  test_bins_attenuate: on the axis the 6-13.6 eV energy density equals L / (4 pi r^2 c) exp(-tau) of the chemistry's law (20 %) for tau < 3.
  test_attenuation_with_depth: the ratio to analytic stays flat inside the slab: dust acts once.
  test_shielding: after 0.15 Myr the cells deep in the slab hold over 30 x the H2 of the lit face and x_H2 > 1e-2: H2 and CO
    columns act with the transported field, dust in the bins only.
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

TEST_NAME = "sgchem_rt_slab"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics  # noqa: E402
from stella_table_bands import Tables  # noqa: E402
from test_treecol_pixels import slab_ics  # noqa: E402

L, Z0, ZSTAR, SIGMA = 8.0, 4.5, 1.5, 1.337e-21
PC, MSUN, MP, C = 3.085678e18, 1.989e33, 1.6726e-24, 2.9979e10
RESULT = {}


@pytest.fixture(scope="module")
def result():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    if "x" in RESULT:
        return RESULT
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    build_gizmo_for_test(TEST_NAME)
    tab = Tables(t["StellarTablesFile"])
    lM, lZ = np.log10(40.0), np.log10(0.0142)
    RESULT["L_uv"] = 10 ** tab._i3(tab.t3["log_L_FUV_total"], lM, lZ, np.log10(0.3 * tab._i2(tab.life, lM, lZ)))
    d = TEST_DIR / "run_slab"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    make_treecol_pixels_ics(str(d / "output" / "snapshot_000.hdf5"), L, slab_ics(L, Z0),
                            [(40.0, tab._i2(tab.t_pms, lM, lZ) + 0.3 * tab._i2(tab.life, lM, lZ), np.array([4.0, 4.0, ZSTAR]))], gas_mass=1.0, hsml=0.6)
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    T = t["StellarTablesFile"]
    p = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    p = p.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {T}")
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    (d / f"{TEST_NAME}.params").write_text(p)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(4) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    with h5py.File(d / "output" / "snapshot_003.hdf5", "r") as F:
        g = F["PartType0"]
        RESULT.update(x=g["Coordinates"][()], E=g["PhotonEnergy"][()], m=g["Masses"][()], rho=g["Density"][()], ab=g["SGChemTracAbund"][()])
    return RESULT


def axis(r):
    x = r["x"]
    return (np.abs(x[:, 0] - 4.0) < 1.2) & (np.abs(x[:, 1] - 4.0) < 1.2) & (x[:, 2] > Z0)


def field_over_analytic(r):
    """6-13.6 eV energy density on the axis, per slice, over L / (4 pi r^2 c) exp(-tau) of the chemistry's law"""
    a = axis(r); z = r["x"][a, 2]
    u = (r["E"][a, 0] + r["E"][a, 2]) * r["rho"][a] / r["m"][a] * MSUN * 1e10 / PC ** 3  # 8-13.6, 6-8 eV
    n = r["rho"][a].mean() * MSUN / PC ** 3 / (1.4 * MP)
    edges = np.linspace(Z0, L, 10); mid = 0.5 * (edges[1:] + edges[:-1])
    ub = np.array([u[(z >= lo) & (z < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])])
    tau = SIGMA * 0.0142 / 0.014 * n * (mid - Z0) * PC
    return tau, ub / (r["L_uv"] / (4 * np.pi * ((mid - ZSTAR) * PC) ** 2 * C) * np.exp(-tau))


def test_bins_attenuate(result):
    tau, ratio = field_over_analytic(result)
    print("tau", tau, "field / analytic", ratio)
    assert np.all(np.abs(ratio[tau < 3] - 1) < 0.2)


def test_attenuation_with_depth(result):
    """dust acts once: inside the slab the ratio to analytic stays flat with depth (a second attenuation would fall as exp(-tau))"""
    tau, ratio = field_over_analytic(result)
    inner = (tau > 1) & (tau < 6)
    assert np.all(np.abs(ratio[inner] / ratio[inner].mean() - 1) < 0.25), ratio


def test_shielding(result):
    r = result; a = axis(r); z = r["x"][:, 2]
    x = r["ab"][:, 0]
    face, deep = x[a & (z < Z0 + 0.4)].mean(), x[a & (z > Z0 + 2.0)].mean()
    print(f"x_H2 face {face:.2e}, deep {deep:.2e}")
    assert deep >= 30 * face and deep > 1e-2
