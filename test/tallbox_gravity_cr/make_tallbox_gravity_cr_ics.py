"""Tallbox (stella_tallbox geometry: 1 kpc in x/y, z in [0, 20] kpc, midplane z = 10 kpc): a uniform gas slab of Sigma = 10 Msun/pc^2,
16 x 16 cells in x/y and 2 layers at the midplane, and probe particles of type 3 (mass 1e-20 code) above it at heights Z_PROBES
(kpc above the midplane). Code units kpc, 1e10 Msun, km/s."""
import sys
import h5py
import numpy as np

L, NXY, SIGMA = 1.0, 16, 1e7 / 1e10  # Sigma = 10 Msun/pc^2 = 1e7 Msun/kpc^2
Z_PROBES = (1.0, 5.0, 9.5)
ABUNDANCE = [7.2360104e-01, 2.7500001e-01, 2.5300001e-04, 7.4099997e-05, 6.1300001e-04, 1.3400000e-04, 7.5700002e-05,
             7.1199996e-05, 3.3100001e-05, 6.8700001e-06, 1.3800000e-04] + [0.0] * 16


def make_ics(path):
    g = (np.arange(NXY) + 0.5) * L / NXY
    x, y, z = np.meshgrid(g, g, 10.0 + np.array([-0.5, 0.5]) * L / NXY / 2, indexing="ij")
    pos = np.stack([x.ravel(), y.ravel(), z.ravel()], 1)
    n = len(pos); m = SIGMA * L * L / n
    probes = np.array([[0.5 * L + 0.013, 0.5 * L + 0.007, 10.0 + zp] for zp in Z_PROBES])
    np_ = len(probes)
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        for k, v in dict(NumPart_ThisFile=[n, 0, 0, np_, 0, 0], NumPart_Total=[n, 0, 0, np_, 0, 0],
                         NumPart_Total_HighWord=[0] * 6, MassTable=[0.0] * 6).items():
            h.attrs[k] = np.array(v, dtype=np.float64 if k == "MassTable" else np.int32)
        for k, v in dict(Time=0.0, Redshift=0.0, BoxSize=L, NumFilesPerSnapshot=1, Flag_DoublePrecision=0).items():
            h.attrs[k] = v
        p = f.create_group("PartType0")
        p["Coordinates"] = pos; p["Velocities"] = np.zeros((n, 3)); p["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint32)
        p["Masses"] = np.full(n, m); p["InternalEnergy"] = np.full(n, 150.0)
        p["MagneticField"] = np.tile(np.array([1e-8, 0, 0]), (n, 1))
        p["ElementAbundance"] = np.tile(np.array(ABUNDANCE, dtype=np.float32), (n, 1))
        q = f.create_group("PartType3")
        q["Coordinates"] = probes; q["Velocities"] = np.zeros((np_, 3)); q["ParticleIDs"] = np.arange(n + 1, n + np_ + 1, dtype=np.uint32)
        q["Masses"] = np.full(np_, 1e-20)
    return n


if __name__ == "__main__":
    print(make_ics(sys.argv[1]))
