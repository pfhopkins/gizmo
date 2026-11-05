"""gmc_cooling cloud (gas as in test/gmc_cooling) plus 16 Type-4 stars in its centre: 4 binaries (a = 100-400 AU)
and 8 single stars within 0.3 pc, 1-30 Msun, cloud-centre velocity plus 1 km/s dispersion. Units pc, Msun, km/s."""
import sys
import h5py
import numpy as np

G = 4.300917e-3  # pc (km/s)^2 / Msun
AU = 1 / 206264.806


def make_ketju_cloud_restart_ics(gas_ic, out, seed=7):
    rng = np.random.default_rng(seed)
    with h5py.File(gas_ic, "r") as f:
        gas = {k: np.array(v) for k, v in f["PartType0"].items()}
        header = dict(f["Header"].attrs)
    c = np.average(gas["Coordinates"], axis=0, weights=gas["Masses"]); vc = np.average(gas["Velocities"], axis=0, weights=gas["Masses"])
    x, v, m = [], [], []
    for _ in range(4):  # circular binaries, random orientation
        m1, m2 = rng.uniform(2, 30, 2); a = rng.uniform(100, 400) * AU
        p = c + rng.normal(0, 0.15, 3); u = rng.normal(0, 1, 3)
        n = rng.normal(size=3); n /= np.linalg.norm(n); e1 = np.cross(n, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
        vrel = np.sqrt(G * (m1 + m2) / a)
        for mm, s in ((m1, m2 / (m1 + m2)), (m2, -m1 / (m1 + m2))):
            x.append(p + s * a * e1); v.append(vc + u + s * vrel * e2); m.append(mm)
    for _ in range(8):
        x.append(c + rng.normal(0, 0.15, 3)); v.append(vc + rng.normal(0, 1, 3)); m.append(rng.uniform(1, 20))
    x, v, m = np.array(x), np.array(v), np.array(m)
    ngas = len(gas["Masses"]); nst = len(m)
    with h5py.File(out, "w") as f:
        h = f.create_group("Header")
        for k, val in header.items():
            h.attrs[k] = val
        npart = np.array([ngas, 0, 0, 0, nst, 0])
        h.attrs["NumPart_ThisFile"] = npart; h.attrs["NumPart_Total"] = npart; h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=int)
        h.attrs["MassTable"] = np.zeros(6)
        g = f.create_group("PartType0")
        for k, val in gas.items():
            g.create_dataset(k, data=val)
        s = f.create_group("PartType4")
        s.create_dataset("Coordinates", data=x); s.create_dataset("Velocities", data=v); s.create_dataset("Masses", data=m)
        s.create_dataset("ParticleIDs", data=np.arange(nst, dtype=gas["ParticleIDs"].dtype) + gas["ParticleIDs"].max() + 1)


if __name__ == "__main__":
    make_ketju_cloud_restart_ics(sys.argv[1], sys.argv[2])
