"""Gas lattice with a central hole around one heavy star without a single-star record: all gas sits far outside its kernel from the star."""
import numpy as np
import h5py
import sys
sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/treecol_pixels")
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402
N, M_STAR, HOLE = 40, 2.0e5, 0.45
RHO = 100.0 * 1.4 * 1.6726e-24 * 3.085678e18 ** 3 / 1.989e33  # Msun / pc^3


def make(fname):
    dx = (1.0 / RHO) ** (1.0 / 3.0); L = N * dx
    p = lattice(N, L); c = L / 2 + 0.31 * dx
    keep = np.sqrt(((p - c) ** 2).sum(1)) > HOLE * L
    make_treecol_pixels_ics(fname, L, p[keep], [(M_STAR, 1e9, (c, c, c))], T=10.0, hsml=2 * dx)
    with h5py.File(fname, "a") as F:
        F["PartType4/SingleStarDrawnMass"][...] = [0.0]
    return L, c, dx
