#!/usr/bin/env python3
"""Generate a collocated-particle IC to exercise the GPU tree-build's
random-octant fallback.

The IC has two populations:
  - 100 uniformly-random gas particles in a 10 pc box.
  - 8 clusters of 16 particles each, every cluster at a single shared (x,y,z)
    coordinate.  Cluster centers are placed on a coarse grid inside the box
    so that distinct clusters have distinct Morton keys but particles within
    a cluster are bit-identical in position.

Total particles: 100 + 8*16 = 228.  Small enough to inspect by hand, large
enough to ensure the BFS hits the LCP >= 126 collocation branch repeatedly
(both at the cluster level and through the random-octant recursion).

Units mirror gmc_cooling / poisson_box (1 pc / 1 Msun / 1 km/s).
"""

import os
import numpy as np
import h5py


def make_gravtree_collocated_ics(
    output_file="gravtree_collocated_ics.hdf5",
    BoxSize=10.0,
    N_random=100,
    N_clusters=8,
    N_per_cluster=16,
    seed=42,
):
    rng = np.random.default_rng(seed)

    # Random component: uniform in box.
    pos_rand = rng.uniform(0.0, BoxSize, size=(N_random, 3)).astype(np.float64)

    # Collocated clusters: 8 cluster centers on a coarse 2x2x2 grid, each
    # offset slightly from cell corners so they don't lie exactly on tree
    # split boundaries.  All particles in a cluster share the cluster center.
    cluster_centers = []
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                cx = (ix + 0.5) * BoxSize / 2 + 0.137  # arbitrary offsets
                cy = (iy + 0.5) * BoxSize / 2 + 0.211
                cz = (iz + 0.5) * BoxSize / 2 + 0.073
                cluster_centers.append((cx, cy, cz))
                if len(cluster_centers) == N_clusters:
                    break
            if len(cluster_centers) == N_clusters:
                break
        if len(cluster_centers) == N_clusters:
            break
    cluster_centers = np.array(cluster_centers[:N_clusters], dtype=np.float64)

    pos_colloc = np.zeros((N_clusters * N_per_cluster, 3), dtype=np.float64)
    for c, ctr in enumerate(cluster_centers):
        pos_colloc[c * N_per_cluster:(c + 1) * N_per_cluster] = ctr

    pos = np.vstack([pos_rand, pos_colloc])
    Ngas = len(pos)

    # Velocities: zero.
    vel = np.zeros((Ngas, 3), dtype=np.float32)

    # Equal masses; total mass scaled to give n_H ~ 1 /cc analogously to
    # poisson_box but at this smaller box size.
    M_total = 3.26e4 * (BoxSize / 100.0) ** 3
    masses = np.full(Ngas, M_total / Ngas, dtype=np.float32)

    # Internal energy for T ~ 1e4 K (same recipe as poisson_box).
    u_therm = 206.8
    internal_energy = np.full(Ngas, u_therm, dtype=np.float32)

    # IDs.
    ids = np.arange(1, Ngas + 1, dtype=np.uint32)

    # Smoothing length: mean spacing of random component.
    spacing = BoxSize / (N_random ** (1.0 / 3.0))
    h_guess = spacing * (32.0 / (4.0 * np.pi / 3.0)) ** (1.0 / 3.0)
    hsml = np.full(Ngas, h_guess, dtype=np.float32)

    # Solar metallicity (11 species); same vector as poisson_box.
    Z_solar = np.array(
        [0.02, 0.28, 2.53e-3, 7.41e-4, 6.13e-3, 1.20e-3,
         5.91e-4, 6.83e-4, 4.09e-4, 6.44e-5, 1.17e-3],
        dtype=np.float32,
    )

    with h5py.File(output_file, "w") as f:
        npart = np.array([Ngas, 0, 0, 0, 0, 0], dtype=np.int32)
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6, dtype=np.float64)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = BoxSize
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 0.0
        h.attrs["OmegaLambda"] = 0.0
        h.attrs["HubbleParam"] = 1.0
        h.attrs["Flag_Sfr"] = 0
        h.attrs["Flag_Cooling"] = 0
        h.attrs["Flag_StellarAge"] = 0
        h.attrs["Flag_Metals"] = 11
        h.attrs["Flag_Feedback"] = 0
        h.attrs["Flag_DoublePrecision"] = 0

        g = f.create_group("PartType0")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=vel)
        g.create_dataset("Masses", data=masses)
        g.create_dataset("ParticleIDs", data=ids)
        g.create_dataset("InternalEnergy", data=internal_energy)
        g.create_dataset("SmoothingLength", data=hsml)
        g.create_dataset("Metallicity", data=np.tile(Z_solar, (Ngas, 1)))

    print(f"Wrote {output_file}")
    print(f"  N_gas         = {Ngas} (random={N_random}, clusters={N_clusters}*{N_per_cluster})")
    print(f"  BoxSize       = {BoxSize} pc")
    print(f"  cluster centers (first 3):")
    for c in cluster_centers[:3]:
        print(f"    {c}")


if __name__ == "__main__":
    outdir = os.path.dirname(os.path.abspath(__file__))
    make_gravtree_collocated_ics(os.path.join(outdir, "gravtree_collocated_ics.hdf5"))
