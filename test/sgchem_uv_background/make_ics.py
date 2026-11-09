"""The sgchem_box lattice at a = 0.25 (z = 3), comoving: physical n_H = 100 cm^-3 in a 10 pc box, so 40 comoving pc."""
import sys
from os import path
import h5py

A = 0.25
sys.path.insert(0, path.join(path.dirname(path.abspath(__file__)), "..", "sgchem_box"))
from make_sgchem_box_ics import make_ics as make_box_ics  # noqa: E402


def make_ics(fname):
    make_box_ics(fname)
    with h5py.File(fname, "r+") as f:
        f["PartType0/Coordinates"][...] /= A
        f["Header"].attrs["BoxSize"] = f["Header"].attrs["BoxSize"] / A
        f["Header"].attrs["Time"] = A
        f["Header"].attrs["Redshift"] = 1 / A - 1
