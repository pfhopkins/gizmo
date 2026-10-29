"""Uniform, static, isothermal gas on an N^3 lattice in a periodic box, for the turbulent_box test.

Units are pc - km/s - Msun. The total mass is set so the virial parameter at the target Mach number
is alpha_vir, using the box convention alpha_vir = 5 sigma_3D^2 L / (6 G M) (Federrath & Klessen 2012),
i.e. the Bertoldi & McKee sphere value with R = L/2 and sigma_1D^2 = sigma_3D^2 / 3. Defaults give
M = 3875 Msun. A uniform B_z gives plasma beta `beta` (used only by MAGNETIC builds).
G comes from gizmo.units, which mirrors GIZMO's constants.h.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python_src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gizmo.units import G_CODE  # noqa: E402
import turb_driving_common as common  # noqa: E402


def make_ics(out, n=32, box_size=10.0, mach=10.0, alpha_vir=2.0, beta=1.0):
    sigma = mach * common.CS
    mtot = 5 * sigma**2 * box_size / (6 * G_CODE * alpha_vir)
    x = (np.arange(n) + 0.5) * box_size / n
    pos = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)
    bz = common.b_field_gauss(mtot / box_size**3, beta)
    common.write_ics(out, pos, mtot / len(pos), box_size, bz)
    return mtot, bz


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--box_size", type=float, default=10.0)
    ap.add_argument("--mach", type=float, default=10.0)
    ap.add_argument("--alpha_vir", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--out", default="turbulent_box_ics.hdf5")
    a = ap.parse_args()
    m, bz = make_ics(a.out, a.n, a.box_size, a.mach, a.alpha_vir, a.beta)
    print(f"wrote {a.out}: {a.n}^3 cells, M = {m:.6g} Msun, mean density = {m / a.box_size**3:.6g} Msun/pc^3, "
          f"B_z = {bz * 1e6:.4g} uG (beta = {a.beta:g})")
