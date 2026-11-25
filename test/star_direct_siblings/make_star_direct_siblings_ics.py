"""Four Type-4 stars at rest, G = 1. Three share ParticleID 7 and carry the child numbers and generations the
star-formation routine gives the first three stars spawned from one cell (child 1, 2, 4; generation 1, 2, 3); the
fourth has its own ID. Unequal masses, no symmetry, separations of order 1 (softening 1e-4)."""
import h5py
import numpy as np

BOX = 100.0
POS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.3, 0.4, 2.0]]) + 0.5 * BOX
MASS = np.array([1.0, 2.0, 0.5, 1.5])
IDS = np.array([7, 7, 7, 9], dtype=np.uint32)
CHILD = np.array([1, 2, 4, 0], dtype=np.uint32)
GENERATION = np.array([1, 2, 3, 0], dtype=np.uint32)


def analytic_accelerations(pos=POS, mass=MASS):
    """Newtonian pair sum, G = 1."""
    acc = np.zeros_like(pos)
    for i in range(len(pos)):
        for j in range(len(pos)):
            if i != j:
                d = pos[j] - pos[i]
                acc[i] += mass[j] * d / np.sum(d * d) ** 1.5
    return acc


def make_star_direct_siblings_ics(fname):
    n = len(MASS)
    npart = np.array([0, 0, 0, 0, n, 0])
    with h5py.File(fname, "w") as f:
        h = f.create_group("Header")
        h.attrs["BoxSize"] = BOX
        h.attrs["NumPart_ThisFile"] = npart.astype(np.int32)
        h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_DoublePrecision"] = 1
        for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Feedback", "Flag_Metals"):
            h.attrs[k] = 0
        p = f.create_group("PartType4")
        p["Coordinates"] = POS
        p["Velocities"] = np.zeros((n, 3))
        p["Masses"] = MASS
        p["ParticleIDs"] = IDS
        p["ParticleChildIDsNumber"] = CHILD
        p["ParticleIDGenerationNumber"] = GENERATION


if __name__ == "__main__":
    import sys
    make_star_direct_siblings_ics(sys.argv[1] if len(sys.argv) > 1 else "star_direct_siblings_ics.hdf5")
