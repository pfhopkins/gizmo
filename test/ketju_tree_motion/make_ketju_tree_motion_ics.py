"""ICs for test/ketju_tree_motion (kpc, km/s, 1e10 Msun): 256 1 + 1 Msun binaries 1000 AU apart in a Plummer sphere.
Circular orbits in ketju_tree_motion_ics.hdf5; at a tenth of the circular speed in ketju_tree_motion_merge_ics.hdf5, where the pairs collide."""
import h5py
import numpy as np

G = 43007.1
AU = 1.0 / 206264.806 / 1e3
N_BIN, M_STAR, SEP, B = 256, 1e-10, 1000 * AU, 1.08e-3


def make(name, orbit_speed):
    rng = np.random.default_rng(20261104)
    r = B / np.sqrt(np.clip(rng.uniform(size=N_BIN), 1e-3, 0.99) ** (-2.0 / 3.0) - 1)
    n = rng.normal(size=(N_BIN, 3)); n /= np.linalg.norm(n, axis=1)[:, None]
    com = r[:, None] * n
    Mtot = 2 * N_BIN * M_STAR
    q = np.zeros(N_BIN)
    for i in range(N_BIN):
        while True:
            x, y = rng.uniform(), rng.uniform() * 0.1
            if y < x * x * (1 - x * x) ** 3.5:
                q[i] = x
                break
    speed = q * np.sqrt(2 * G * Mtot) * (r * r + B * B) ** -0.25
    d = rng.normal(size=(N_BIN, 3)); d /= np.linalg.norm(d, axis=1)[:, None]
    vcom = speed[:, None] * d
    a = rng.normal(size=(N_BIN, 3)); a /= np.linalg.norm(a, axis=1)[:, None]
    b = np.cross(a, rng.normal(size=(N_BIN, 3))); b /= np.linalg.norm(b, axis=1)[:, None]
    vorb = orbit_speed * 0.5 * np.sqrt(G * 2 * M_STAR / SEP)
    x = np.empty((2 * N_BIN, 3)); v = np.empty((2 * N_BIN, 3))
    x[::2] = com + 0.5 * SEP * a; x[1::2] = com - 0.5 * SEP * a
    v[::2] = vcom + vorb * b; v[1::2] = vcom - vorb * b
    x -= x.mean(0); v -= v.mean(0)
    with h5py.File(name, "w") as f:
        h = f.create_group("Header"); n_all = len(x)
        h.attrs["NumPart_ThisFile"] = [0, 0, 0, 0, n_all, 0]; h.attrs["NumPart_Total"] = [0, 0, 0, 0, n_all, 0]
        h.attrs["NumPart_Total_HighWord"] = [0] * 6; h.attrs["MassTable"] = [0.0] * 6
        h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = 0.3; h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 0.0; h.attrs["OmegaLambda"] = 0.0; h.attrs["HubbleParam"] = 1.0
        for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals", "Flag_Feedback"): h.attrs[k] = 0
        g = f.create_group("PartType4")
        g["Coordinates"] = x; g["Velocities"] = v; g["Masses"] = np.full(n_all, M_STAR); g["ParticleIDs"] = np.arange(1, n_all + 1, dtype=np.uint64)


if __name__ == "__main__":
    make("ketju_tree_motion_ics.hdf5", 1.0)
    make("ketju_tree_motion_merge_ics.hdf5", 0.1)
