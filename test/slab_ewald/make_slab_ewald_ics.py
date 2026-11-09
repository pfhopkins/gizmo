#!/usr/bin/env python3
"""ICs for slab_ewald: on a 1 x 1 x 4 box (mid-plane z = 2), one heavy source (type 1), a random cloud of
light sources (type 1), and massless tracers (type 2) spread over |z - 2| < 2.5 * L so that both the
tabulated and the far-field (|dz| > 2 L) branches of the correction are probed."""

import numpy as np
import h5py

N_CLOUD, N_TRACER = 500, 300
M_HEAVY, M_CLOUD = 0.5, 0.5


def make_slab_ewald_ics(output_file="slab_ewald_ics.hdf5", seed=42):
    rng = np.random.default_rng(seed)
    heavy = np.array([[0.93, 0.07, 2.05]])  # near a corner, so its nearest image wraps in x and y
    cloud = np.column_stack([rng.random(N_CLOUD), rng.random(N_CLOUD), 2.0 + 0.15 * rng.standard_normal(N_CLOUD)])
    tracer = np.column_stack([rng.random(N_TRACER), rng.random(N_TRACER), 2.0 + rng.uniform(-1.95, 1.95, N_TRACER)])
    src = np.vstack([heavy, cloud])
    m_src = np.concatenate([[M_HEAVY], np.full(N_CLOUD, M_CLOUD / N_CLOUD)])
    with h5py.File(output_file, "w") as f:
        h = f.create_group("Header")
        npart = np.array([0, len(src), len(tracer), 0, 0, 0], dtype=np.int32)
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.37
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = 1.0
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_DoublePrecision"] = 1
        for name, x, m, id0 in (("PartType1", src, m_src, 1), ("PartType2", tracer, np.full(len(tracer), 1e-12), 1 + len(src))):
            p = f.create_group(name)
            p["Coordinates"] = x
            p["Velocities"] = np.zeros_like(x)
            p["Masses"] = m
            p["ParticleIDs"] = np.arange(id0, id0 + len(x), dtype=np.uint32)


if __name__ == "__main__":
    make_slab_ewald_ics()
