"""stella_imf_accretion_floor: star formation in gas drained to the IMF-accretion floor creates no mass.

Uniform n_H = 200 cm^-3 box of 1 Msun cells (12^3, 4 ranks), Stella production model with the forced-SF density
cutoff. nHcutoffSF and CritPhysDensity are far below the box density and the SF cooldown is off, so every cell forms
one star per step and the box drains its gas down to the accretion floor (1% of the merge mass). Masses are float32 as
in Gadget ICs, so drained donors reach the floor exactly. TreeDomainUpdateFrequency 2 keeps the decomposition, and with
it cell merging, off as in a deep zoom sub-step, so the drained cells stay. Stars drawn there cannot collect their mass.
Checks:
  - failed draws happen and are undone (gas back to the parent cell), never filled from nowhere;
  - total baryonic mass (gas + stars, double precision, mass_balance_steps.txt) is conserved to 1e-14;
  - every star holds its drawn IMF mass to within 1% (SFinfo.txt: 0.99 M_drawn <= M_acc <= M_drawn).
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

TEST_NAME = "stella_imf_accretion_floor"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
OUT = join(TEST_DIR, "output")
NRANKS = 4

PC, MSUN, M_P = 3.085678e18, 1.989e33, 1.6726e-24
X_H = 0.7065  # code HYDROGEN_MASSFRAC for net 5
N_H, NG, M_CELL, Z_SUN, T_IC = 200.0, 12, 1.0, 0.1, 0.001

# solar element mass fractions C..Zn (Asplund+09, as resolvedism_solar_massfrac)
SOLAR = np.array([2.4765351227e-03, 7.2545495820e-04, 6.0028592850e-03, 5.2841265250e-07, 1.3157897125e-03,
                  3.0605839462e-05, 7.4124358876e-04, 5.8255839166e-05, 6.9620919184e-04, 6.0990559675e-06,
                  3.2376421296e-04, 8.5877929305e-06, 7.6870616805e-05, 3.2093705781e-06, 6.7169384016e-05,
                  4.8646655463e-08, 3.2681466346e-06, 3.3215539907e-07, 1.7387424601e-05, 1.1327606908e-05,
                  1.3528499188e-03, 4.4118818564e-06, 7.4619379442e-05, 7.5396909229e-07, 1.8184871682e-06])


def metallicity(zfrac):
    """Metallicity[28] = (Z, H, He, C..Zn), the init.cc solar-scaled pattern."""
    el = SOLAR * zfrac
    he = (1 - X_H) + (0.2485 - (1 - X_H)) * zfrac
    return np.concatenate([[el.sum(), 1 - el.sum() - he, he], el])


def make_ics(fname):
    L = (NG**3 * M_CELL * MSUN / (N_H * M_P / X_H)) ** (1 / 3) / PC
    dx = L / NG
    g = (np.arange(NG) + 0.5) * dx
    x = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    ng = len(x)
    met = metallicity(Z_SUN)
    with h5py.File(fname, "w") as f:
        p0 = f.create_group("PartType0")
        p0["Coordinates"] = x * 1e-3
        p0["Velocities"] = np.zeros((ng, 3))
        p0["ParticleIDs"] = np.arange(1, ng + 1, dtype=np.uint64)
        p0["Masses"] = np.full(ng, M_CELL * MSUN / 1.989e43, dtype=np.float32)  # float32 as in Gadget ICs: drained donors reach the floor exactly
        p0["InternalEnergy"] = np.full(ng, 1.0)  # ~100 K
        p0["KernelMaxRadius"] = np.full(ng, 2.0 * dx * 1e-3)
        p0["Metallicity"] = np.tile(met, (ng, 1))
        p0["ElementAbundance"] = np.tile(met[1:], (ng, 1))
        p0["SGChemTracAbund"] = np.tile([0.0, 1.0e-4, 0.0], (ng, 1))
        p0["MagneticField"] = np.zeros((ng, 3))
        h = f.create_group("Header")
        npart = np.array([ng, 0, 0, 0, 0, 0])
        h.attrs["NumPart_ThisFile"] = npart.astype(np.int32)
        h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = T_IC
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = L * 1e-3
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_DoublePrecision"] = 0
        h.attrs["Flag_Metals"] = len(met)
        for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Feedback"):
            h.attrs[k] = 1
    return L


def mbary_trace(path):
    """(tag, M_bary, dM_bary) per logged checkpoint of mass_balance_steps.txt."""
    rows = [ln.split() for ln in open(path) if ln.strip() and not ln.startswith("#")]
    return [(r[1], float(r[3]), float(r[4])) for r in rows]


def test_stella_imf_accretion_floor():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO; nothing to validate from external snapshots")
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME)
    finally:
        os.chdir(cwd)
    run = tempfile.mkdtemp(prefix=f"{TEST_NAME}_")
    os.makedirs(join(run, "output"))
    os.symlink(REPO, join(run, "gizmo"))  # SGCHEM and the stellar tables open their data under gizmo/
    L = make_ics(join(run, f"{TEST_NAME}_ics.hdf5"))
    params = open(join(TEST_DIR, f"{TEST_NAME}.params")).read()
    params = re.sub(r"(?m)^BoxSize\s.*$", f"BoxSize {L * 1e-3:.10g}", params)
    open(join(run, f"{TEST_NAME}.params"), "w").write(params)
    with open(join(run, "run.log"), "w") as log:
        r = subprocess.run(mpi_launch_command(NRANKS) + [join(TEST_DIR, "GIZMO"), f"{TEST_NAME}.params", "2"],
                           stdout=log, stderr=subprocess.STDOUT, cwd=run)
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.copytree(join(run, "output"), OUT, ignore=shutil.ignore_patterns("snapshot_*", "restartfiles"))
    shutil.copy(join(run, "run.log"), join(OUT, "run.log"))
    text = open(join(run, "run.log")).read()

    n_fallback = len(re.findall(r"IMF accretion (failed for star|found no donor mass for star)", text))
    n_failed = sum(int(x) for x in re.findall(r"RESOLVEDISM IMF: (\d+) failed draws", text))
    n_spawn = sum(int(x) for x in re.findall(r"RESOLVEDISM SFR: spawned (\d+) stars", text))
    trace = mbary_trace(join(OUT, "mass_balance_steps.txt"))
    m0, m1 = trace[0][1], trace[-1][1]
    d_imf = sum(d for tag, _, d in trace if tag == "post_imf_accrete")
    sf = np.loadtxt(join(OUT, "SFinfo.txt"), usecols=(5, 6), ndmin=2)  # M_drawn, M_acc [Msun]
    heavy = int(np.sum(sf[:, 1] > sf[:, 0] + 1e-4))
    light = int(np.sum(sf[:, 1] < 0.99 * sf[:, 0] - 1e-4))
    print(f"stars spawned {n_spawn}, failed draws undone {n_failed}, fallback stars {n_fallback}; "
          f"M_bary {m0:.16e} -> {m1:.16e} (rel {(m1 - m0) / m0:+.3e}), sum dM at post_imf_accrete {d_imf:+.3e}; "
          f"SFinfo rows {len(sf)}, M_acc > M_drawn {heavy}, M_acc < 0.99 M_drawn {light}")
    assert r.returncode == 0, f"GIZMO exited {r.returncode}, see {OUT}/run.log"
    assert n_fallback == 0, f"{n_fallback} stars were given mass without donors"
    assert abs(m1 - m0) <= 1e-14 * m0, f"baryonic mass changed by {(m1 - m0) / m0:+.3e} (IMF accretion: {d_imf:+.3e} code)"
    assert heavy == 0 and light == 0, f"{heavy} stars heavier, {light} stars more than 1% lighter than their drawn IMF mass"
    assert n_failed > 0, "no failed draw: the drained-floor case was not reached"
    shutil.rmtree(run, ignore_errors=True)
