"""Uniform lattice sphere of radius R = 16 spacings (spacing, cell mass and composition of the sgchem_cooling_path IC), the cell
with ID 1 at its centre; with --pair one more cell 1e-3 spacings from the centre cell. Code units kpc, 1e10 Msun, km/s."""
import sys
import h5py
import numpy as np

D, NR, M = 6.5e-5, 16, 9.376713e-11
ABUNDANCE = [7.2360104e-01, 2.7500001e-01, 2.5300001e-04, 7.4099997e-05, 6.1300001e-04, 1.3400000e-04, 7.5700002e-05,
             7.1199996e-05, 3.3100001e-05, 6.8700001e-06, 1.3800000e-04] + [0.0] * 16
BOX = 4 * NR * D


def make_ics(path, pair):
    g = np.arange(-NR, NR + 1)
    idx = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    idx = idx[(idx ** 2).sum(1) <= NR ** 2]
    idx = idx[np.argsort((idx ** 2).sum(1), kind="stable")]  # centre first
    pos = idx * D + 0.5 * BOX
    if pair:
        pos = np.vstack([pos, pos[0] + [1e-3 * D, 0, 0]])
    n = len(pos)
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        for k, v in dict(NumPart_ThisFile=[n, 0, 0, 0, 0, 0], NumPart_Total=[n, 0, 0, 0, 0, 0],
                         NumPart_Total_HighWord=[0] * 6, MassTable=[0.0] * 6).items():
            h.attrs[k] = np.array(v, dtype=np.float64 if k == "MassTable" else np.int32)
        for k, v in dict(Time=0.0, Redshift=0.0, BoxSize=BOX, NumFilesPerSnapshot=1, Flag_DoublePrecision=0).items():
            h.attrs[k] = v
        p = f.create_group("PartType0")
        p["Coordinates"] = pos
        p["Velocities"] = np.zeros((n, 3), dtype=np.float32)
        p["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint32)
        p["Masses"] = np.full(n, M, dtype=np.float32)
        p["InternalEnergy"] = np.full(n, 0.20298721, dtype=np.float32)
        p["MagneticField"] = np.tile(np.array([1e-8, 0, 0], dtype=np.float32), (n, 1))
        p["ElementAbundance"] = np.tile(np.array(ABUNDANCE, dtype=np.float32), (n, 1))
    return n


if __name__ == "__main__":
    print(make_ics(sys.argv[1], "--pair" in sys.argv))
