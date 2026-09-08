#!/usr/bin/env python3
"""Generate ICs for the shu_jets test: the shu1977 singular isothermal sphere with
solid-body rotation added.

The base ICs are the ones used by the shu1977 test (a truncated SIS at 2x the
critical density, 10 Msun of gas in 128000 cells), which start at rest.  Here we
spin them up about the z axis to a target

    beta = E_rot / |E_grav|

so that the collapse forms a rotationally supported disk, giving the jets
launched by SINGLE_STAR_FB_JETS a well-defined axis.

E_grav is evaluated with the spherical-shell formula, exact for a spherically
symmetric mass distribution (which the SIS ICs are, by construction):

    E_grav = -G * sum_i M(<r_i) * m_i / r_i
"""

import os
import urllib.request

import h5py
import numpy as np

import os as _os
import sys as _sys
# Run standalone (the test invokes this as a subprocess from the test directory), so python_src
# is not on the path the way it is under pytest.
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  "..", "harness"))
from gizmo.units import G_CODE, UNIT_LENGTH_IN_CM, UNIT_MASS_IN_G, UNIT_VELOCITY_IN_CM_PER_S  # noqa: E402

# Unit system of shu_jets.params: pc, Msun, km/s. G comes from gizmo.units, which mirrors
# GIZMO's constants.h. This file previously used G_CGS = 6.674e-8 -- the modern CODATA value --
# where the code integrates with 6.672e-8, a 3.0e-4 error in every G-derived quantity in the IC.

BASE_IC_URL = "https://users.flatironinstitute.org/~mgrudic/gizmo_tests/shu1977/shu1977_ics.hdf5"

BETA_TARGET = 0.1  # E_rot / |E_grav|


def gravitational_binding_energy(coords, masses, center):
    """E_grav for a spherically symmetric distribution, via the shell formula."""
    r = np.linalg.norm(coords - center, axis=1)
    order = np.argsort(r)
    r, m = r[order], masses[order]
    m_enclosed = np.cumsum(m) - m  # mass strictly interior to each particle
    nonzero = r > 0
    return -G_CODE * np.sum(m_enclosed[nonzero] * m[nonzero] / r[nonzero])


def make_shu_jets_ics(output_file="shu_jets_ics.hdf5", base_ic_file=None, beta=BETA_TARGET):
    """Write rotating ICs, spun up to E_rot/|E_grav| = beta about the z axis."""
    out_dir = os.path.dirname(output_file) or "."
    if base_ic_file is None:
        base_ic_file = os.path.join(out_dir, "shu1977_ics.hdf5")
    if not os.path.exists(base_ic_file):
        print(f"Downloading {BASE_IC_URL}...")
        urllib.request.urlretrieve(BASE_IC_URL, base_ic_file)

    with h5py.File(base_ic_file, "r") as F:
        coords = F["PartType0/Coordinates"][:].astype(np.float64)
        masses = F["PartType0/Masses"][:].astype(np.float64)

    center = np.average(coords, weights=masses, axis=0)
    dx = coords - center
    radius = np.linalg.norm(dx, axis=1)

    e_grav = gravitational_binding_energy(coords, masses, center)
    # solid-body rotation about z: E_rot = 0.5 * I_zz * Omega^2
    i_zz = np.sum(masses * (dx[:, 0] ** 2 + dx[:, 1] ** 2))
    omega = np.sqrt(2 * beta * abs(e_grav) / i_zz)

    vel = np.zeros_like(dx)
    vel[:, 0] = -omega * dx[:, 1]
    vel[:, 1] = omega * dx[:, 0]

    # copy the base ICs wholesale and swap in the rotating velocity field, so that
    # every other field (internal energy, metallicity, IDs, ...) stays identical
    with h5py.File(base_ic_file, "r") as src, h5py.File(output_file, "w") as dst:
        for name in src:
            src.copy(name, dst)
        for key, value in src["Header"].attrs.items():
            dst["Header"].attrs[key] = value
        del dst["PartType0/Velocities"]
        dst["PartType0"].create_dataset("Velocities", data=vel.astype(np.float32))

    m_total, r_max = masses.sum(), radius.max()
    e_rot = 0.5 * i_zz * omega**2
    # Peebles spin parameter, for reference; |E| ~ |E_grav| for a cold core
    lam = (
        omega * i_zz * np.sqrt(abs(e_grav)) / (G_CODE * m_total ** (5.0 / 2.0))
    )
    print(f"Wrote {output_file} with {len(coords)} gas cells")
    print(f"  M_gas       = {m_total:.4g} Msun, R = {r_max:.4g} pc, m_cell = {masses[0]:.6g} Msun")
    print(f"  E_grav      = {e_grav:.4g}, E_rot = {e_rot:.4g}  (code units: Msun (km/s)^2)")
    print(f"  beta        = {e_rot / abs(e_grav):.4g}  (target {beta})")
    # code time unit = UnitLength/UnitVelocity = 1 pc / (1 km/s) = 0.9778 Myr
    print(f"  Omega       = {omega:.6g} (km/s)/pc = {omega / 0.9777922:.6g} 1/Myr")
    print(f"  v_max       = {omega * r_max * 1e3:.4g} m/s at the outer edge")
    print(f"  lambda_Peebles ~ {lam:.4g}")


if __name__ == "__main__":
    make_shu_jets_ics()
