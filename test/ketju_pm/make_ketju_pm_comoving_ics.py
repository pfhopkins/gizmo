#!/usr/bin/env python3
"""ICs for test_ketju_pm_comoving (comoving, a = 0.5, h = 1): the physical geometry of make_ketju_pm_ics.py. A 20 Msun star at the
box centre (on a PM cell boundary, so the mesh is mirror-symmetric about it); two 1 + 1 Msun binaries (a = 50 AU, e = 0.5, each its
own KETJU region) R_OUT = 20000 AU physical along +x and +y on circular orbits about it; single 2 Msun stars at the mirror positions
with mirrored velocities. Comoving box 4e5 AU (2e5 AU physical). Gadget IC convention: comoving positions, velocities =
peculiar / sqrt(a)."""
import numpy as np
from make_ketju_pm_ics import kepler_state, AU, G, R_OUT, write

A0, L_COM = 0.5, 4e5 * AU


def state():
    M = 20.0; v = np.sqrt(G * (M + 2.0) / R_OUT)
    xs, vs = kepler_state(1, 1, 50 * AU, 0.5, 0.7)
    m, x, u = [M], [np.zeros(3)], [np.zeros(3)]
    for e_r, e_v in (([1, 0, 0], [0, 1, 0]), ([0, 1, 0], [-1, 0, 0])):
        e_r, e_v = np.array(e_r, float), np.array(e_v, float)
        for xi, vi in zip(xs, vs):  # binary at +r, internal orbit tilted out of the plane
            m.append(1.0); x.append(R_OUT * e_r + np.array([xi[0], 0.3 * xi[1], xi[1]])); u.append(v * e_v + np.array([vi[0], 0.3 * vi[1], vi[1]]))
        m.append(2.0); x.append(-R_OUT * e_r); u.append(-v * e_v)  # single star at the mirror position
    return np.array(m), np.array(x), np.array(u)


def main(name="ketju_pm_comoving_ics.hdf5"):
    m, x, u = state()
    write(name, m, x / A0, u / np.sqrt(A0), L_COM, time=A0, comoving=True)


if __name__ == "__main__":
    main()
