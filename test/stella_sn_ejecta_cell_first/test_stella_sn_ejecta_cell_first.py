"""stella_sn_ejecta_cell_first: a Stella supernova that finds no gas becomes an ejecta cell before any kernel growth.

A 12 Msun star (the cell with ID 1, GALSF_RESOLVEDISM_ISOLATED_FB_TEST) at the centre of a uniform 20 pc box explodes at
TestSNTime0 = 1e-6. Until TestFBNoGasUntilTime = 3e-6 the feedback weighting walk of the star finds no gas (an empty
bubble). Two builds:
  - ejecta_cell (GALSF_RESOLVEDISM_SN_SPAWN=1): the SN is turned into one new gas cell on the first weighting pass,
    holding the ejecta mass and metals; the star keeps its kernel (no growth) and is left with the remnant mass.
  - kernel_growth (no SN_SPAWN): the fallback still works: the kernel grows (x1.4 per try), the event is retried while
    no gas is in reach, and once gas is in reach the ejecta reach the gas and the star is left with the remnant mass.
"""

import os
import re
import shutil
import subprocess
import tempfile
from os.path import abspath, dirname, join

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "stella_sn_ejecta_cell_first"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
NRANKS = 4
UNIT_MASS_MSUN = 1.0e10

KPC, MSUN, M_P, K_B = 3.085678e21, 1.989e33, 1.6726e-24, 1.3807e-16
BOX_PC, N_H, T_GAS, Z_SUN, M_CELL = 20.0, 10.0, 1.0e4, 0.1, 1.0


def make_ics(fname):
    """Uniform lattice, gas only; the cell nearest the centre has ID 1 and becomes the star."""
    x_z = Z_SUN * 0.014
    x_h = 1.0 - 0.275 - x_z
    m_gas = N_H * (BOX_PC * 3.085678e18) ** 3 * M_P / x_h / MSUN
    ng = int(round((m_gas / M_CELL) ** (1.0 / 3.0)))
    n = ng**3
    box = BOX_PC * 1e-3
    g = (np.arange(ng) + 0.5) * box / ng
    pos = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    ids = np.arange(n, dtype=np.uint32) + 2
    ids[np.argmin(np.sum((pos - 0.5 * box) ** 2, axis=1))] = 1
    ea = np.zeros((n, 27), dtype=np.float32)
    ea[:, 1] = 0.275
    for k, x in {2: 2.53e-3, 3: 7.41e-4, 4: 6.13e-3, 5: 1.34e-3, 6: 7.57e-4, 7: 7.12e-4, 8: 3.31e-4, 9: 6.87e-5, 10: 1.38e-3}.items():
        ea[:, k] = x * Z_SUN
    ea[:, 0] = 1.0 - ea.sum(axis=1)
    u = 1.5 * K_B * T_GAS / (0.6 * M_P) / 1e10
    with h5py.File(fname, "w") as f:
        h = f.create_group("Header")
        h.attrs["BoxSize"] = box
        h.attrs["NumPart_ThisFile"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.int32)
        h.attrs["NumPart_Total"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_Metals"] = 27
        h.attrs["Flag_DoublePrecision"] = 0
        for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Feedback"):
            h.attrs[k] = 1
        p = f.create_group("PartType0")
        p["Coordinates"] = pos
        p["Velocities"] = np.zeros((n, 3), dtype=np.float32)
        p["Masses"] = np.full(n, m_gas / n * MSUN / 1.989e43, dtype=np.float32)
        p["ParticleIDs"] = ids
        p["InternalEnergy"] = np.full(n, u, dtype=np.float32)
        p["ElementAbundance"] = ea
        p["MagneticField"] = np.zeros((n, 3), dtype=np.float32)


def masses(snap):
    """Gas mass, gas metal mass and the mass of star ID 1, in Msun (float64 sums)."""
    with h5py.File(snap, "r") as f:
        mg = f["PartType0/Masses"][:].astype(np.float64) * UNIT_MASS_MSUN
        zg = f["PartType0/Metallicity"][:, 0].astype(np.float64)
        ms = None
        if "PartType4" in f:
            ids, m4 = f["PartType4/ParticleIDs"][:], f["PartType4/Masses"][:].astype(np.float64) * UNIT_MASS_MSUN
            ms = float(m4[ids == 1].sum()) if np.any(ids == 1) else None
    return mg.sum(), (mg * zg).sum(), ms


def run_variant(flags, tag):
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME, extra_config_flags=flags)
    finally:
        os.chdir(cwd)
    run = tempfile.mkdtemp(prefix=f"{TEST_NAME}_{tag}_")
    os.makedirs(join(run, "output"))
    os.symlink(REPO, join(run, "gizmo"))
    make_ics(join(run, f"{TEST_NAME}_ics.hdf5"))
    shutil.copy(join(TEST_DIR, f"{TEST_NAME}.params"), run)
    with open(join(run, "run.log"), "w") as log:
        r = subprocess.run(mpi_launch_command(NRANKS) + [join(TEST_DIR, "GIZMO"), f"{TEST_NAME}.params"],
                           stdout=log, stderr=subprocess.STDOUT, cwd=run)
    out = join(TEST_DIR, f"output_{tag}")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    for f in ("run.log", "output/SNinfo.txt"):
        if os.path.exists(join(run, f)):
            shutil.copy(join(run, f), out)
    text = open(join(run, "run.log")).read()
    assert r.returncode == 0, f"GIZMO exited {r.returncode}, see {out}/run.log"
    snaps = sorted(join(run, "output", s) for s in os.listdir(join(run, "output")) if s.startswith("snapshot_"))
    with h5py.File(snaps[-1], "r") as f:
        assert abs(f["Header"].attrs["Time"] / 5.0e-6 - 1) < 1e-6, "run stopped before TimeMax"
    m0 = masses(snaps[0])
    m1 = masses(snaps[-1])
    sninfo = np.loadtxt(join(run, "output", "SNinfo.txt"), ndmin=2) if os.path.exists(join(run, "output", "SNinfo.txt")) else np.zeros((0, 20))
    shutil.rmtree(run, ignore_errors=True)
    return text, m0, m1, sninfo, out


GROW = re.compile(r"RESOLVEDISM FB: event ID=1 found no gas in kernel; growing KernelRadius to ([0-9.eE+-]+) \(try (\d)\)")


@pytest.mark.skipif(bool(os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN")), reason="builds GIZMO")
def test_sn_without_gas_makes_ejecta_cell_first():
    text, (mg0, mz0, _), (mg1, mz1, ms1), _, out = run_variant(("GALSF_RESOLVEDISM_SN_SPAWN=1",), "ejecta_cell")
    dec = re.findall(r"SPAWNSN_DECIDE: star=1 Mkern=([0-9.eE+-]+) Msun < [0-9.]+\*Mej=([0-9.eE+-]+) -> payload stored \(rem=([0-9.eE+-]+)\)", text)
    exe = re.findall(r"SPAWNSN_EXEC: star=1 -> cell m=([0-9.eE+-]+) Msun", text)
    grow = GROW.findall(text)
    print(f"DECIDE {dec}, EXEC {exe}, kernel growth lines {len(grow)}; gas dM {mg1 - mg0:.4f} dMZ {mz1 - mz0:.4f}, star {ms1}")
    assert len(dec) == 1 and len(exe) == 1, f"expected one ejecta-cell decision and one new cell, got {len(dec)}/{len(exe)} ({out}/run.log)"
    assert not grow, f"kernel grew {len(grow)} times before the ejecta cell was made ({out}/run.log)"
    mej, rem = float(dec[0][1]), float(dec[0][2])
    assert float(dec[0][0]) == 0.0, "the weighting walk was meant to find no gas"
    assert abs(float(exe[0]) / mej - 1) < 1e-3, f"ejecta cell holds {exe[0]} Msun, ejecta mass {mej}"
    assert ms1 is not None and abs(ms1 - rem) < 1e-3 * mej, f"star mass {ms1} Msun after the SN, remnant {rem}"
    assert mz1 - mz0 > 0.1, f"gas metal mass grew by only {mz1 - mz0:.3g} Msun: the ejecta metals did not reach the gas"


@pytest.mark.skipif(bool(os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN")), reason="builds GIZMO")
def test_sn_without_gas_kernel_growth_fallback():
    text, (mg0, mz0, _), (mg1, mz1, ms1), sninfo, out = run_variant((), "kernel_growth")
    grow = GROW.findall(text)
    print(f"kernel growth lines {len(grow)}; SNinfo rows {len(sninfo)} status {sninfo[:, 19].tolist() if len(sninfo) else []}; "
          f"gas dM {mg1 - mg0:.4f} dMZ {mz1 - mz0:.4f}, star {ms1}")
    assert "SPAWNSN" not in text, "no ejecta cells without GALSF_RESOLVEDISM_SN_SPAWN"
    assert grow, f"no kernel growth while the SN found no gas ({out}/run.log)"
    h = np.array([float(g[0]) for g in grow if g[1] == "1"])
    h2 = np.array([float(g[0]) for g in grow if g[1] == "2"])
    assert len(h) and len(h2) and np.allclose(h2[: len(h)] / h[: len(h2)], 1.4, rtol=1e-5), "kernel does not grow by 1.4 per try"
    row = sninfo[(sninfo[:, 6] == 1) & (sninfo[:, 19] == 0)] if len(sninfo) else sninfo
    assert len(row) == 1, f"expected one delivered SN row for star 1 in SNinfo, got {len(row)}"
    mej = float(row[0, 9])
    assert np.any((sninfo[:, 6] == 1) & (sninfo[:, 19] == 2)), "no retried (no gas reached) row before the SN was delivered"
    assert ms1 is not None and abs(ms1 - (row[0, 7] - mej)) < 1e-3 * mej, f"star mass {ms1} Msun after the SN, expected {row[0, 7] - mej}"
    assert mz1 - mz0 > 0.1, f"gas metal mass grew by only {mz1 - mz0:.3g} Msun: the ejecta metals did not reach the gas"
