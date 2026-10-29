"""4^3 gas lattice at the density, composition, field and temperature of the Stella dense-box harness IC."""
import sys
import numpy as np
import h5py

N, BOX = 4, 0.00026  # code units (kpc); spacing 6.5e-5 as in the 20^3 harness box
ABUNDANCE = [7.2360104e-01, 2.7500001e-01, 2.5300001e-04, 7.4099997e-05, 6.1300001e-04, 1.3400000e-04, 7.5700002e-05,
             7.1199996e-05, 3.3100001e-05, 6.8700001e-06, 1.3800000e-04] + [0.0] * 16


def main(path):
    x = (np.arange(N) + 0.5) * BOX / N
    pos = np.array(np.meshgrid(x, x, x, indexing="ij")).reshape(3, -1).T
    n = len(pos)
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        for k, v in dict(NumPart_ThisFile=[n, 0, 0, 0, 0, 0], NumPart_Total=[n, 0, 0, 0, 0, 0],
                         NumPart_Total_HighWord=[0] * 6, MassTable=[0.0] * 6).items():
            h.attrs[k] = np.array(v, dtype=np.float64 if k == "MassTable" else np.int32)
        for k, v in dict(Time=0.0, Redshift=0.0, BoxSize=BOX, NumFilesPerSnapshot=1, Flag_DoublePrecision=0).items():
            h.attrs[k] = v
        g = f.create_group("PartType0")
        g["Coordinates"] = pos
        g["Velocities"] = np.zeros((n, 3), dtype=np.float32)
        g["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint32)
        g["Masses"] = np.full(n, 9.376713e-11, dtype=np.float32)
        g["InternalEnergy"] = np.full(n, 0.20298721, dtype=np.float32)
        g["MagneticField"] = np.tile(np.array([1e-8, 0, 0], dtype=np.float32), (n, 1))
        g["ElementAbundance"] = np.tile(np.array(ABUNDANCE, dtype=np.float32), (n, 1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sgchem_cooling_path_ics.hdf5")
