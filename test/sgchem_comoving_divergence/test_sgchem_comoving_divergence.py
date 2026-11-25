"""sgchem_comoving_divergence: in a comoving run the chemistry gets the physical velocity divergence, Hubble flow included.

The sgchem_uv_background box (sgchem_box lattice, n_H = 100 cm^-3 physical, 8000 K, solar metals) at a = 0.25, at rest in comoving
coordinates: pure Hubble flow; periodic (no edges, so no peculiar flow), Omega_M the box's own matter (check_omega), Omega_L 0;
Delta ln a = 5.7e-5; networks 5 and 20. TEST_SGCHEM_PRINT_CELL_VELOCITY_DIVERGENCE prints what the solver gets.
Derivation: physical velocity v = a dx/dt + H a x, so dv_k/dr_k = g_k + H with g_k = (dv_pec,k/dx_k) / a^2 [code velocity
units]; div v = sum_k g_k + 3 H, net 20 LVG dv/dr = (1/3) sum_k |g_k + H| (stella_dev chem_interface.cc:308-318).
test_hubble_rate: the printed H(a) is H0 sqrt(Omega_M / a^3 + (1 - Omega_M) / a^2) (curvature as hubble_function, Omega_L 0).
test_divergence_is_hubble_flow: every cell step: div v = sum_k g_k + 3 H and, at network 20, dv/dr = mean_k |g_k + H|, to 1e-6 of H.
test_mass_and_energy: the gas mass is constant to 1e-12; E_th + E_kin changes by the ledger's chemistry and floor rows plus the
  adiabatic Hubble term -2 int E_th dln a (trapezoid over the snapshots), to 2e-5 of E_th at the start.
"""
import glob
import re
import shutil
import sys
from os import environ, path, symlink

import h5py
import numpy as np
import pytest

from gizmo.test import build_and_run_test, get_cooling_tables, default_omp_threads, variant_suffix

HERE = path.dirname(path.abspath(__file__))
ROOT = path.dirname(path.dirname(HERE))
NAME = "sgchem_comoving_divergence"
sys.path.insert(0, path.join(ROOT, "test", "sgchem_box"))
sys.path.insert(0, path.join(ROOT, "test", "sgchem_uv_background"))
NETS = (5, 20)
H0 = 3.2407789e-18  # HubbleParam 1 [s^-1]
UL, UM, UV, G_CGS = 3.086e18, 1.989e33, 1.0e5, 6.672e-8  # params units; GRAVITY_G_CGS (constants.h): check_omega allows 1e-2 absolute
MSUN, KMS2 = 1.989e33, 1e10
LINE = re.compile(r"SGCHEM_DIVV: ID=(\d+) a=(\S+) H=(\S+) divv=(\S+) dvdr=(\S+) g=(\S+) (\S+) (\S+)")


@pytest.fixture(scope="module")
def runs():
    if not environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    from test_sgchem_box import chemistry_tables_dir
    from make_ics import make_ics
    tables = chemistry_tables_dir()
    if environ.get("SPCOOL_TABLES") and not path.exists(path.join(HERE, "spcool_tables")):
        symlink(environ["SPCOOL_TABLES"], path.join(HERE, "spcool_tables"))
    get_cooling_tables(HERE)
    make_ics(path.join(HERE, f"{NAME}_ics.hdf5"))
    with h5py.File(path.join(HERE, f"{NAME}_ics.hdf5"), "r") as f:
        G, H0c = G_CGS * UM / (UL * UV * UV), H0 * UL / UV  # code units as begrun.cc
        om = f["PartType0/Masses"][:].sum() / f["Header"].attrs["BoxSize"] ** 3 / (3 * H0c * H0c / (8 * np.pi * G))
    over = {"SGChemTablesPath": tables, "Omega0": om, "OmegaBaryon": om, "OmegaLambda": 0}
    out = {"omega": om}
    for net in NETS:
        flags = (f"CHEMISTRYNETWORK={net}",)
        build_and_run_test(NAME, 2, default_omp_threads(), extra_config_flags=flags, param_overrides=over)
        d = path.join(HERE, f"output_CHEMISTRYNETWORK_{net}")
        shutil.copy(path.join(HERE, f"test_{NAME}{variant_suffix(flags)}.out"), path.join(d, f"test_{NAME}.out"))  # the harness names the log after the variant
        out[net] = d
    return out


def divv_lines(d):
    rows = [m.groups() for m in map(LINE.search, open(path.join(d, f"test_{NAME}.out"))) if m]
    return np.array([[float(x) for x in r] for r in rows])


@pytest.mark.parametrize("net", NETS)
def test_hubble_rate(runs, net):
    r = divv_lines(runs[net])
    assert len(r) > 0
    a, h = r[:, 1], r[:, 2]
    h_exp = H0 * np.sqrt(runs["omega"] / a**3 + (1 - runs["omega"]) / a**2)
    print(f"net {net}: H(a) max relative deviation {np.max(np.abs(h / h_exp - 1)):.2e}")
    assert np.all(np.abs(h / h_exp - 1) < 1e-6)


@pytest.mark.parametrize("net", NETS)
def test_divergence_is_hubble_flow(runs, net):
    r = divv_lines(runs[net])
    assert len(np.unique(r[:, 0])) == 512
    h, divv, dvdr, g = r[:, 2], r[:, 3], r[:, 4], r[:, 5:8]
    e_div = np.max(np.abs(divv - g.sum(1) - 3 * h) / h)
    print(f"net {net}: {len(r)} cell steps, 3H = {3 * h[0]:.6e} s^-1, peculiar div v up to {np.abs(g.sum(1)).max():.2e}, "
          f"max |div v - sum g - 3H| / H = {e_div:.2e}")
    assert e_div < 1e-6
    if net == 20:
        e_dv = np.max(np.abs(dvdr - np.abs(g + h[:, None]).mean(1)) / h)
        print(f"net 20: max |dv/dr - mean|g + H|| / H = {e_dv:.2e}")
        assert e_dv < 1e-6


@pytest.mark.parametrize("net", NETS)
def test_mass_and_energy(runs, net):
    d = runs[net]
    t, m, eth, ekin = [], [], [], []
    for s in sorted(glob.glob(path.join(d, "snapshot_*.hdf5"))):
        with h5py.File(s) as f:
            a = f["Header"].attrs["Time"]
            g = f["PartType0"]
            mm = g["Masses"][:]
            v = g["Velocities"][:] * np.sqrt(a)  # physical peculiar
            t.append(a), m.append(mm.sum())
            eth.append(np.sum(mm * g["InternalEnergy"][:]) * MSUN * KMS2)
            ekin.append(0.5 * np.sum(mm * np.sum(v**2, axis=1)) * MSUN * KMS2)
    t, m, eth, ekin = map(np.array, (t, m, eth, ekin))
    assert len(t) >= 10
    assert np.all(np.abs(m / m[0] - 1) < 1e-12), m / m[0] - 1
    with open(path.join(d, "ledger.txt")) as f:
        names = f.readline().split("(")[0].split()[1:]
    led = np.loadtxt(path.join(d, "ledger.txt"))
    col = {n: led[:, k] for k, n in enumerate(names)}
    assert np.allclose(col["time"], t, rtol=1e-12, atol=0)
    booked = col["E_chem_heating_erg"] - col["E_chem_cooling_erg"] + col["E_kick_floor_erg"]
    lna = np.log(t)
    hubble = np.concatenate(([0], np.cumsum(-2 * 0.5 * (eth[1:] + eth[:-1]) * np.diff(lna))))
    res = (eth + ekin) - (eth[0] + ekin[0]) - (booked - booked[0]) - hubble
    print(f"net {net}: E_th {eth[0]:.6e} -> {eth[-1]:.6e} erg, E_kin {ekin[-1]:.3e}, booked {booked[-1]:.6e}, Hubble {hubble[-1]:.3e}, "
          f"max |residual| / E_th(0) = {np.max(np.abs(res)) / eth[0]:.2e}")
    assert np.max(np.abs(res)) < 2e-5 * eth[0]
