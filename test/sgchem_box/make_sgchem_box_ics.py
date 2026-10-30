"""Uniform periodic gas box for the SGChem tests: 8^3 cells, n_H = 100 cm^-3, T = 8000 K, solar metals, at rest."""
import h5py
import numpy as np

N, L = 8, 10.0                                  # cells per side, box side [pc]
NH, T0 = 100.0, 8000.0                          # [cm^-3], [K]
XH, MP, KB, PC, MSUN = 1 / 1.4, 1.6726e-24, 1.3807e-16, 3.086e18, 1.989e33


def make_ics(fname="sgchem_box_ics.hdf5"):
    x = (np.arange(N) + 0.5) * L / N
    pos = np.array(np.meshgrid(x, x, x, indexing="ij")).reshape(3, -1).T
    n = len(pos)
    rho_cgs = NH * MP / XH
    m = rho_cgs * (L * PC) ** 3 / MSUN / n
    mu = 1.4 / 1.1                              # neutral atomic gas, He/H = 0.1
    u = 1.5 * KB * T0 / (mu * MP) / 1e10        # [(km/s)^2]
    z = np.array([0.0142, 0.27, 2.53e-3, 7.41e-4, 6.13e-3, 1.34e-3, 7.57e-4, 7.12e-4, 3.31e-4, 6.87e-5, 1.38e-3])  # FIRE-3 solar slots
    with h5py.File(fname, "w") as f:
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = h.attrs["NumPart_Total"] = [n, 0, 0, 0, 0, 0]
        h.attrs["NumPart_Total_HighWord"] = [0] * 6
        h.attrs["MassTable"] = [0.0] * 6
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = L
        h.attrs["NumFilesPerSnapshot"] = 1
        g = f.create_group("PartType0")
        g["Coordinates"] = pos
        g["Velocities"] = np.zeros((n, 3))
        g["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint32)
        g["Masses"] = np.full(n, m)
        g["InternalEnergy"] = np.full(n, u)
        g["Metallicity"] = np.tile(z, (n, 1))


if __name__ == "__main__":
    make_ics()
