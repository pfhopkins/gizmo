"""sf_cutoff_pi_guard: the forced-SF density cutoff (GALSF_RESOLVEDISM_SF_INSTANT_CUTOFF) does not fire in ionised cells.

A 30 Msun Stella star (in the IC) sits at the centre of a uniform n_H = 200 cm^-3 box of 1 Msun cells (24^3, 4 ranks).
nHcutoffSF = 150, so every cell is above the cutoff; FacSfThreshMJ = 1e-6 makes the Jeans test fail everywhere, so every
new star comes from the forced path. Photoionisation runs before star formation in each step, so the star's HII region
is marked before the first SF pass. Checks:
  - the HII region exists (>= 50 ionised cells; radius R from the ionised volume);
  - no new star is born inside R - 1.5 dx (the forced path used to bypass the HII-region veto);
  - new stars are born outside R + 1.5 dx (the forced path itself works).
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

TEST_NAME = "sf_cutoff_pi_guard"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
OUT = join(TEST_DIR, "output")
NRANKS = 4

PC, MSUN, M_P = 3.085678e18, 1.989e33, 1.6726e-24
X_H = 0.7065  # code HYDROGEN_MASSFRAC for net 5, as used for n_H in the cutoff
N_H, NG, M_CELL, Z_SUN = 200.0, 24, 1.0, 0.1
STAR_MASS, STAR_AGE_YR, T_IC = 30.0, 1.0e5, 0.001
CODE_TIME_YR = 1e3 * PC / 1e5 / (365.25 * 86400)
STAR_ID = NG**3 + 1

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


def box_pc():
    rho = N_H * M_P / X_H
    return (NG**3 * M_CELL * MSUN / rho) ** (1 / 3) / PC


def make_ics(fname):
    L = box_pc()
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
        p0["Masses"] = np.full(ng, M_CELL * MSUN / 1.989e43)
        p0["InternalEnergy"] = np.full(ng, 1.0)  # ~100 K
        p0["KernelMaxRadius"] = np.full(ng, 2.0 * dx * 1e-3)
        p0["Metallicity"] = np.tile(met, (ng, 1))
        p0["ElementAbundance"] = np.tile(met[1:], (ng, 1))
        p0["SGChemTracAbund"] = np.tile([0.0, 1.0e-4, 0.0], (ng, 1))
        p0["MagneticField"] = np.zeros((ng, 3))
        p4 = f.create_group("PartType4")  # off-lattice centre: the star is not on top of a cell
        m_code = STAR_MASS * MSUN / 1.989e43
        p4["Coordinates"] = np.full((1, 3), 0.5 * L * 1e-3)
        p4["Velocities"] = np.zeros((1, 3))
        p4["ParticleIDs"] = np.array([STAR_ID], dtype=np.uint64)
        p4["Masses"] = np.array([m_code])
        p4["KernelMaxRadius"] = np.array([2.0 * dx * 1e-3])
        p4["StellarFormationTime"] = np.array([T_IC - STAR_AGE_YR / CODE_TIME_YR])
        p4["SampleIMFStellarMasses"] = np.array([[STAR_MASS, 0.0, 0.0, 0.0, m_code]])
        p4["IMFSampledFlag"] = np.array([1], dtype=np.int32)
        p4["BirthMetallicity"] = np.array([met[0]])
        p4["FormationDensity"] = np.array([0.0])  # no PMS delay
        p4["Metallicity"] = met[None, :]
        p4["ElementAbundance"] = met[None, 1:]
        h = f.create_group("Header")
        npart = np.array([ng, 0, 0, 0, 1, 0])
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


def test_sf_cutoff_pi_guard():
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
    os.makedirs(OUT, exist_ok=True)
    shutil.copy(join(run, "run.log"), join(OUT, "run.log"))
    text = open(join(run, "run.log")).read()
    assert r.returncode == 0, f"GIZMO exited {r.returncode}, see {OUT}/run.log"
    assert not re.search(r"nHcutoffSF.*(not used|ignored)", text), "nHcutoffSF not read"
    n_ion = [int(x) for x in re.findall(r"RESOLVEDISM PI: (\d+) ionized", text)]
    assert n_ion and n_ion[0] >= 50, f"no HII region at the first front solve ({n_ion[:1]}), see {OUT}/run.log"
    dx = L / NG
    r_hii = (3 * n_ion[0] * dx**3 / (4 * np.pi)) ** (1 / 3)
    snaps = sorted(s for s in os.listdir(join(run, "output")) if s.startswith("snapshot_"))
    with h5py.File(join(run, "output", snaps[-1]), "r") as f:
        ids = f["PartType4/ParticleIDs"][:]
        pos = f["PartType4/Coordinates"][:] * 1e3
    centre = pos[ids == STAR_ID][0]
    d = pos[ids != STAR_ID] - centre
    d -= L * np.round(d / L)
    r_new = np.linalg.norm(d, axis=1)
    inside, outside = int(np.sum(r_new < r_hii - 1.5 * dx)), int(np.sum(r_new > r_hii + 1.5 * dx))
    print(f"ionised cells {n_ion[0]} (R = {r_hii:.2f} pc, dx = {dx:.2f} pc); new stars {len(r_new)}: "
          f"inside R-1.5dx {inside}, outside R+1.5dx {outside}")
    assert inside == 0, f"{inside} stars formed inside the HII region through the density cutoff"
    assert outside > 0, "the density cutoff formed no stars in neutral gas"
    shutil.rmtree(run, ignore_errors=True)
