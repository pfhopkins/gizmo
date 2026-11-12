"""4^3 gas lattice in a 1 ckpc/h comoving box at z = 6.5, pristine, ~100 K (one-zone UVB shielding driver; the driver sets density and state)."""
import sys
import numpy as np
import h5py

N, BOX = 4, 1.0  # code units (ckpc/h)


def main(path):
    x = (np.arange(N) + 0.5) * BOX / N
    pos = np.array(np.meshgrid(x, x, x, indexing="ij")).reshape(3, -1).T
    n = len(pos)
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        for k, v in dict(NumPart_ThisFile=[n, 0, 0, 0, 0, 0], NumPart_Total=[n, 0, 0, 0, 0, 0], NumPart_Total_HighWord=[0] * 6).items():
            h.attrs[k] = np.array(v, dtype=np.int32)
        h.attrs["MassTable"] = np.zeros(6)
        for k, v in dict(Time=1 / 7.5, Redshift=6.5, BoxSize=BOX, NumFilesPerSnapshot=1, Flag_DoublePrecision=0).items():
            h.attrs[k] = v
        g = f.create_group("PartType0")
        g["Coordinates"] = pos
        g["Velocities"] = np.zeros((n, 3), dtype=np.float32)
        g["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint32)
        g["Masses"] = np.full(n, 4.35e-9, dtype=np.float32)
        g["InternalEnergy"] = np.full(n, 1.0, dtype=np.float32)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sgchem_uvb_shielding_ics.hdf5")
