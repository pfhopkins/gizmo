"""The energy step trace's E_pot without the particles' own PM mesh self-potential (PM_EXACT_SELF_POTENTIAL_DIAG, diagnostics only).

One 20 Msun particle in a periodic box L = 2e5 AU with PMGRID = 32 (cell 6250 AU, r_s = 1.25 cells), moving at (1.5, 1.1, 0.7) km/s
so that over 1e-5 code time it crosses about half a cell per axis and its CIC weights sweep sub-cell offsets.

Derivation. Alone in the box, the tree adds nothing (the walk skips r = 0) and PM_Potential is entirely the particle's own mesh
potential: its CIC deposit convolved with the deconvolved Green's function smth(k) of pmforce_periodic, read back with the same
weights, phi_self = G/(pi L) m sum_{a,b} w_a w_b g(a - b), g(d) = sum_{k != 0} smth(k) cos(2 pi k.d / PMGRID). The trace's
E_pot = 1/2 m (Potential - phi_self) must then vanish at every sync, up to the single-precision FFT (fftw_real float): bar
|E_pot| <= 1e-5 x 1/2 m G m / (sqrt(pi) r_s), set before running. The analytic r -> 0 limit -G m / (sqrt(pi) r_s) used before is
the continuum value; the CIC value differs by sub-cell position, which the deliberate break (analytic term) must show.
Mass and energy step traces must close."""

import os

import h5py
import numpy as np

from gizmo.test import build_and_run_test, clean_test_outputs
from gizmo.conservation_trace import keep_and_check_traces

TEST_NAME = "pm_self_potential"
TEST_DIR = f"test/{TEST_NAME}"
G = 6.672e-8 * 1.989e43 / (3.085678e21 * 1e10)  # code units (kpc, km/s, 1e10 Msun), constants.h values
AU = 1.0 / 206264.806 / 1e3
L, NGRID = 2e5 * AU, 32
M = 20.0


def _write_ics(path):
    cell = L / NGRID
    x = np.array([[10.3, 17.6, 5.85]]) * cell
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = [0, 1, 0, 0, 0, 0]; h.attrs["NumPart_Total"] = [0, 1, 0, 0, 0, 0]
        h.attrs["NumPart_Total_HighWord"] = [0] * 6; h.attrs["MassTable"] = [0.0] * 6
        h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = L
        h.attrs["NumFilesPerSnapshot"] = 1; h.attrs["Omega0"] = 0.0; h.attrs["OmegaLambda"] = 0.0; h.attrs["HubbleParam"] = 1.0
        for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals", "Flag_Feedback"): h.attrs[k] = 0
        g = f.create_group("PartType1")
        g["Coordinates"] = x; g["Velocities"] = np.array([[1.5, 1.1, 0.7]]); g["Masses"] = np.array([M / 1e10])
        g["ParticleIDs"] = np.array([1], dtype=np.uint64)


def test_pm_self_potential():
    _write_ics(f"{TEST_DIR}/{TEST_NAME}_ics.hdf5")
    clean_test_outputs(TEST_NAME)
    build_and_run_test(TEST_NAME, 1, 1)
    m = M / 1e10; r_s = 1.25 * L / NGRID
    scale = 0.5 * m * G * m / (np.sqrt(np.pi) * r_s)
    epot = []
    for l in open(f"{TEST_DIR}/output/energy_step_trace.txt"):
        if l.startswith("SYNC"):
            kv = dict(tok.split("=", 1) for tok in l.split()[1:] if "=" in tok)
            epot.append(float(kv["E_pot"]))
    assert len(epot) >= 10
    worst = max(abs(e) for e in epot) / scale
    print(f"{len(epot)} syncs: max |E_pot| = {worst:.3e} of 1/2 m G m/(sqrt(pi) r_s) = {scale:.3e}")
    assert worst <= 1e-5
    rep = keep_and_check_traces(os.path.join(TEST_DIR, "output"), os.path.join(TEST_DIR, "output"))
    assert rep.ok, rep.summary()
