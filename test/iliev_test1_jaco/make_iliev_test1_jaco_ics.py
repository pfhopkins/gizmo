#!/usr/bin/env python3
"""Initial conditions for iliev_test1_jaco: Iliev et al. (2006) test 1 scaled to n_H = 100 cm^-3. A static uniform hydrogen
medium (GIZMO's X = 0.76, no metals) on a 32^3 glass in a 6 pc periodic box, with one ionizing source (a type-4
particle at the centre; RT_ILIEV_TEST1 sets its rate to 5e48 photons/s)."""

import os
import urllib.request

import h5py
import numpy as np

BOX_SIZE = 6.0  # pc
N_H = 100.0  # cm^-3
X_H = 0.76  # GIZMO's HYDROGEN_MASSFRAC without METALS
PC_CM, MSUN_G, PROTONMASS_CGS = 3.0857e18, 1.989e33, 1.6726e-24


def make_iliev_test1_jaco_ics(output_file="iliev_test1_jaco_ics.hdf5", glass="glass_32.hdf5"):
    glass_file = os.path.join(os.path.dirname(output_file) or ".", glass)
    if not os.path.exists(glass_file):
        urllib.request.urlretrieve(f"https://users.flatironinstitute.org/~mgrudic/glass/{glass}", glass_file)
    with h5py.File(glass_file, "r") as f:
        pos = f["Coordinates"][:].astype(np.float64)
    pos = (pos - pos.min(axis=0)) / (pos.max(axis=0) - pos.min(axis=0)).max() * BOX_SIZE
    ngas = len(pos)
    rho = N_H * PROTONMASS_CGS / X_H / (MSUN_G / PC_CM**3)  # Msun/pc^3
    m_gas = rho * BOX_SIZE**3 / ngas
    with h5py.File(output_file, "w") as f:
        h = f.create_group("Header")
        npart = np.array([ngas, 0, 0, 0, 1, 0], dtype=np.int32)
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = BOX_SIZE
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 0.0
        h.attrs["OmegaLambda"] = 0.0
        h.attrs["HubbleParam"] = 1.0
        h.attrs["Flag_Sfr"] = 0
        h.attrs["Flag_Cooling"] = 0
        h.attrs["Flag_StellarAge"] = 0
        h.attrs["Flag_Metals"] = 0
        h.attrs["Flag_Feedback"] = 0
        h.attrs["Flag_DoublePrecision"] = 0
        g = f.create_group("PartType0")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=np.zeros((ngas, 3), dtype=np.float32))
        g.create_dataset("Masses", data=np.full(ngas, m_gas, dtype=np.float32))
        g.create_dataset("ParticleIDs", data=np.arange(1, ngas + 1, dtype=np.int32))
        s = f.create_group("PartType4")
        s.create_dataset("Coordinates", data=np.full((1, 3), 0.5 * BOX_SIZE))
        s.create_dataset("Velocities", data=np.zeros((1, 3), dtype=np.float32))
        s.create_dataset("Masses", data=np.array([1e-3 * m_gas], dtype=np.float32))
        s.create_dataset("ParticleIDs", data=np.array([ngas + 1], dtype=np.int32))


if __name__ == "__main__":
    make_iliev_test1_jaco_ics()
