"""The initial condition of test/stella_dense_sf (a dense cold clump of 500 cells of 1 Msun at n_H = 1e6 cm^-3 in a
16 pc box of diffuse warm gas), optionally with one old 1 Msun star (type 4) 2 pc from a box corner, far from the clump."""
import sys
from os.path import abspath, dirname, join

import h5py
import numpy as np

sys.path.insert(0, join(dirname(abspath(__file__)), "..", "stella_dense_sf"))
from make_stella_dense_sf_ics import make_stella_dense_sf_ics  # noqa: E402

STAR_POS_KPC, STAR_MASS_MSUN = (0.002, 0.002, 0.002), 1.0


def make_sgchem_cr_zero_before_first_star_ics(fname, with_star=False):
    make_stella_dense_sf_ics(fname)
    if not with_star:
        return
    with h5py.File(fname, "r+") as f:
        h = f["Header"]
        n_gas = int(h.attrs["NumPart_ThisFile"][0])
        h.attrs["NumPart_ThisFile"] = np.array([n_gas, 0, 0, 0, 1, 0], dtype=np.int32)
        h.attrs["NumPart_Total"] = np.array([n_gas, 0, 0, 0, 1, 0], dtype=np.uint32)
        p = f.create_group("PartType4")
        p["Coordinates"] = np.array([STAR_POS_KPC], dtype=np.float32)
        p["Velocities"] = np.zeros((1, 3), dtype=np.float32)
        p["Masses"] = np.array([STAR_MASS_MSUN * 1.989e33 / 1.989e43], dtype=np.float32)
        p["ParticleIDs"] = np.array([n_gas + 1], dtype=np.uint32)


if __name__ == "__main__":
    make_sgchem_cr_zero_before_first_star_ics(sys.argv[1] if len(sys.argv) > 1 else "sgchem_cr_zero_before_first_star_ics.hdf5",
                                              with_star=len(sys.argv) > 2 and sys.argv[2] == "star")
