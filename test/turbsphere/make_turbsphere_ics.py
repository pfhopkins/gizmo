"""Uniform static sphere in a periodic box of 10 R with 1e-3-density ambient gas, for the turbsphere test.

This is MakeCloud's default geometry, which GravAccel_GMCTurbInit (STARFORGE_GMC_TURBINIT) assumes:
R = 0.1 BoxSize and a cloud holding 0.808 of the gas mass. All particles have the same mass; both
cloud and ambient sit on lattices, the ambient one 10x coarser. The cloud mass gives
alpha_vir = 5 sigma_3D^2 R / (3 G M) at the target Mach number; with R = 5 pc this is the sphere
inscribed in turbulent_box, with the same mass and sigma. A uniform B_z gives plasma beta `beta` in
the cloud; the ambient has the same field, as MakeCloud's does (MAGNETIC builds only).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python_src"))
sys.path.insert(0, os.path.join(HERE, "..", "turbulent_box"))
from gizmo.units import G_CODE  # noqa: E402
import turb_driving_common as common  # noqa: E402

DENSITY_CONTRAST = 1e-3
# as STARFORGE_GMC_TURBINIT_CLOUD_RADIUS_IN_BOXSIZE and _CLOUD_MASS_FRACTION in declarations/allvars.h
CLOUD_RADIUS_IN_BOXSIZE = 0.1
CLOUD_MASS_FRACTION = 0.808


def lattice(n, spacing, center):
    x = (np.arange(n) - (n - 1) / 2) * spacing
    return np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3) + center


def make_ics(out, n_cloud=32768, radius=5.0, mach=10.0, alpha_vir=2.0, beta=1.0):
    box = radius / CLOUD_RADIUS_IN_BOXSIZE
    sigma = mach * common.CS
    m_cloud = 5 * sigma**2 * radius / (3 * G_CODE * alpha_vir)
    d = radius * (4 * np.pi / (3 * n_cloud)) ** (1 / 3)  # lattice spacing giving ~n_cloud points in the sphere
    n = int(np.ceil(2 * radius / d)) + 2
    cloud = lattice(n, d, box / 2)
    cloud = cloud[np.linalg.norm(cloud - box / 2, axis=1) < radius]
    mass = m_cloud / len(cloud)

    # same particle mass, so 10x the spacing for 1e-3 the density; snapped to fit the box periodically
    n_amb = int(round(box / (d * DENSITY_CONTRAST ** (-1 / 3))))
    ambient = (np.stack(np.meshgrid(*[np.arange(n_amb)] * 3, indexing="ij"), axis=-1).reshape(-1, 3) + 0.5) * box / n_amb
    ambient = ambient[np.linalg.norm(ambient - box / 2, axis=1) >= radius + 0.5 * box / n_amb]

    pos = np.concatenate([cloud, ambient])
    rho_cloud = m_cloud / (4 / 3 * np.pi * radius**3)
    bz = common.b_field_gauss(rho_cloud, beta)
    common.write_ics(out, pos, mass, box, bz)
    return len(cloud), len(ambient), m_cloud, bz


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_cloud", type=int, default=32768)
    ap.add_argument("--radius", type=float, default=5.0)
    ap.add_argument("--mach", type=float, default=10.0)
    ap.add_argument("--alpha_vir", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--out", default="turbsphere_ics.hdf5")
    a = ap.parse_args()
    nc, na, m, bz = make_ics(a.out, a.n_cloud, a.radius, a.mach, a.alpha_vir, a.beta)
    print(f"wrote {a.out}: {nc} cloud + {na} ambient cells, M_cloud = {m:.6g} Msun, "
          f"cloud mass fraction {nc / (nc + na):.4f} (GravAccel_GMCTurbInit assumes {CLOUD_MASS_FRACTION}), "
          f"B_z = {bz * 1e6:.4g} uG (beta = {a.beta:g} in the cloud)")
