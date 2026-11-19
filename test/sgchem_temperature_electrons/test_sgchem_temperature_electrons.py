"""sgchem_temperature_electrons: the stored SGChem temperature counts the particles the network counts.

Derivation: the network's temperature (cooling/sgchem/cxx/chem_rhs_net17.cc:82-98, the same C++ code in stella_dev) is
T = T(ekn) with ekn = e / (k_B n_tot), n_tot = (1 + x_He + x_e - x_H2) n_H and x_e = x_H+ + x_He+ + 2 x_He++ + (x_C - x_CO) + x_D+ + x_Si
per H (net 5: x_H+ + (x_C - x_CO) + x_Si); for atomic gas T(ekn) = (gamma - 1) ekn, gamma = 5/3. The stored Temperature
(snapshot, dust 1e5 K gates) used x_e = x_H+ only: in ionised He gas n_tot is 2.25 instead of 2.1 per H, T 7 % high.

Problem: the sgchem_box lattice at n_H = 1 cm^-3, network 17, solar metals without depletion (SGChemDustToGasRatio 0), start
fully ionised H with x_He+ = x_He++ = 0.05 (SGCHEM_KEEP_IC_ABUNDANCES), about 5e4 K, 1 kyr, 2 ranks.
test_temperature_counts_all_electrons: at both snapshots after the start Temperature = (2/3) e / (k_B n_tot) to 1e-3, with
  x_C, x_Si from the snapshot metals (C/12, Si/28 per H nucleus, H mass fraction 1/1.4).
  The snapshot ElectronAbundance is that same x_e to 1e-6 (Ne and the temperature count the same electrons).
test_mass_and_energy: gas mass constant to 1e-12; thermal energy change = ledger chemistry heating - cooling + floor to 1e-6.
"""
import glob
import sys
from os import environ, path

import h5py
import numpy as np
import pytest

from gizmo.test import build_and_run_test, get_cooling_tables, default_omp_threads

HERE = path.dirname(path.abspath(__file__))
ROOT = path.dirname(path.dirname(HERE))
NAME = "sgchem_temperature_electrons"
sys.path.insert(0, path.join(ROOT, "test", "sgchem_box"))
ABHE, MP, KB, MSUN = 0.1, 1.6726e-24, 1.380649e-16, 1.989e33
SCALE = 100.0 ** (1.0 / 3.0)  # sgchem_box lattice from n_H = 100 to 1
T0 = 5.0e4
X0 = {"H2": 0.0, "H+": 1.0, "CO": 0.0, "He+": 0.05, "He++": 0.05, "D+": 2.6e-5, "HD": 0.0}  # net-17 order


@pytest.fixture(scope="module")
def run():
    if not environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    from test_sgchem_box import chemistry_tables_dir
    from make_sgchem_box_ics import make_ics
    tables = chemistry_tables_dir()
    get_cooling_tables(HERE)
    ic = path.join(HERE, f"{NAME}_ics.hdf5")
    make_ics(ic)
    with h5py.File(ic, "r+") as f:
        n = len(f["PartType0/ParticleIDs"])
        f["PartType0/Coordinates"][...] *= SCALE
        f["Header"].attrs["BoxSize"] = f["Header"].attrs["BoxSize"] * SCALE
        xe = X0["H+"] + X0["He+"] + 2 * X0["He++"] + X0["D+"]
        f["PartType0/InternalEnergy"][...] = 1.5 * KB * T0 * (1 + ABHE + xe) / ((1 + 4 * ABHE) * MP) / 1e10
        f["PartType0/SGChemTracAbund"] = np.tile(list(X0.values()), (n, 1))
    build_and_run_test(NAME, 2, default_omp_threads(), param_overrides={"SGChemTablesPath": tables})
    return sorted(glob.glob(path.join(HERE, "output", "snapshot_*.hdf5")))


def state(s):
    with h5py.File(s, "r") as f:
        g = f["PartType0"]
        return {k: g[k][()] for k in ("Masses", "InternalEnergy", "Temperature", "SGChemTracAbund", "Metallicity", "ElectronAbundance")}


def test_temperature_counts_all_electrons(run):
    assert len(run) == 3
    for s in run[1:]:
        g = state(s)
        ab, Z = g["SGChemTracAbund"], g["Metallicity"]
        x_c, x_si = Z[:, 2] / 12.0 * (1 + 4 * ABHE), Z[:, 7] / 28.0 * (1 + 4 * ABHE)
        xe = ab[:, 1] + ab[:, 3] + 2 * ab[:, 4] + np.maximum(x_c - ab[:, 2], 0) + ab[:, 5] + x_si
        n_tot_per_mass = (1 + ABHE + xe - ab[:, 0]) / ((1 + 4 * ABHE) * MP)
        t_net = (2.0 / 3.0) * g["InternalEnergy"] * 1e10 / (KB * n_tot_per_mass)
        r = g["Temperature"] / t_net
        print(f"{path.basename(s)}: T {g['Temperature'].mean():.5g} K, network T {t_net.mean():.5g} K, ratio {r.min():.5f}..{r.max():.5f}, "
              f"x_He+ {ab[:, 3].mean():.3g}, x_He++ {ab[:, 4].mean():.3g}")
        assert np.all(np.abs(r - 1) < 1e-3)
        dne = np.max(np.abs(g["ElectronAbundance"] / xe - 1))
        print(f"  ElectronAbundance vs the network's electrons per H: max relative difference {dne:.2e}")
        assert dne < 1e-6  # Ne and the temperature count the same electrons (float output)


def test_mass_and_energy(run):
    a, b = state(run[0]), state(run[-1])
    assert abs(b["Masses"].sum() / a["Masses"].sum() - 1) < 1e-12
    led = np.atleast_2d(np.loadtxt(path.join(HERE, "output", "ledger.txt")))
    names = [c for c in open(path.join(HERE, "output", "ledger.txt")).readline().split()[1:] if c != "(cumulative;"]
    booked = led[-1, names.index("E_chem_heating_erg")] - led[-1, names.index("E_chem_cooling_erg")] + led[-1, names.index("E_kick_floor_erg")]
    de = (np.sum(b["Masses"] * b["InternalEnergy"]) - np.sum(a["Masses"] * a["InternalEnergy"])) * MSUN * 1e10
    print(f"thermal energy change {de:.6e} erg, booked {booked:.6e} erg, relative residual {de / booked - 1:.2e}")
    assert abs(de / booked - 1) < 1e-6
