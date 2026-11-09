"""ICs of the sn_driven_box test: make_box writes the gas box of the driving stage; add_stars puts pre-placed stars, with
table masses that explode and ages that make them die across a window, into a snapshot of the driven box."""
from pathlib import Path
import h5py
import numpy as np

MSUN, PC, MP, XH = 1.989e33, 3.085678e18, 1.6726e-24, 0.75
LENGTH, MASS = 1e-3, 1e-10  # kpc per pc, 1e10 Msun per Msun
YR_PER_CODE_TIME = 3.085678e21 / 1e5 / 3.155e7
Z_GAS = np.loadtxt(Path(__file__).parent / "solar_metallicity.txt")


def make_box(fname, L_pc, n_H, cell_msun=1.0, seed=7):
    """gas at 100 K at rest in a jittered lattice: returns the cell count"""
    rng = np.random.default_rng(seed)
    rho = n_H * MP / XH * PC**3 / MSUN
    N = int(round(L_pc / (cell_msun / rho) ** (1 / 3)))
    d = L_pc / N
    g = (np.arange(N) + 0.5) * d
    p = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    p = (p + rng.uniform(-0.2, 0.2, p.shape) * d) % L_pc
    n = len(p)
    m = rho * L_pc**3 / n
    u = 1.5 * 1.380649e-16 * 100 / (1.27 * MP) / 1e10
    with h5py.File(fname, "w") as F:
        h = F.create_group("Header")
        a = np.array([n, 0, 0, 0, 0, 0])
        h.attrs["NumPart_ThisFile"] = a.astype(np.int32)
        h.attrs["NumPart_Total"] = a.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"], h.attrs["Redshift"], h.attrs["BoxSize"] = 0.0, 0.0, L_pc * LENGTH
        h.attrs["NumFilesPerSnapshot"], h.attrs["HubbleParam"] = 1, 1.0
        h.attrs["Omega0"], h.attrs["OmegaLambda"] = 0.0, 0.0
        for k in ("Flag_Sfr", "Flag_StellarAge", "Flag_Feedback", "Flag_DoublePrecision"):
            h.attrs[k] = 1
        h.attrs["Flag_Metals"] = len(Z_GAS)  # the code reads this many columns
        h.attrs["Flag_Cooling"] = 0
        s = F.create_group("PartType0")
        s["Coordinates"] = p * LENGTH
        s["Velocities"] = np.zeros((n, 3))
        s["ParticleIDs"] = np.arange(2, n + 2, dtype=np.uint32)
        s["Masses"] = np.full(n, m * MASS)
        s["InternalEnergy"] = np.full(n, u)
        s["MagneticField"] = np.tile([0.0, 0.0, 2e-6], (n, 1))
        s["SGChemTracAbund"] = np.tile([0.0, 1e-4, 0.0], (n, 1))
        s["Metallicity"] = np.tile(Z_GAS, (n, 1))
    return n


def events(tables_file, n_sn, window_yr, seed=11):
    """masses from the table with an exploding remnant at solar Z (Kroupa-weighted), positions in box units, deaths over the window"""
    rng = np.random.default_rng(seed)
    with h5py.File(tables_file) as T:
        Zt, Mt = T["Z"][()], T["M_init"][()]
        iz = int(np.argmin(abs(Zt - 0.014)))
        shape = (len(Zt), len(Mt))
        life = (T["lifetime_yr"][()] + T["t_PMS"][()]).reshape(shape)[iz]
        rem = T["remnant_type"][()].reshape(shape)[iz]
    ok = np.array([i for i, m in enumerate(Mt) if 8.5 <= m <= 40 and rem[i] == 2])
    w = Mt[ok] ** -2.3
    idx = rng.choice(ok, n_sn, p=w / w.sum())
    t = np.sort(rng.uniform(0.05 * window_yr, window_yr, n_sn))
    return Mt[idx], life[idx] - t, rng.uniform(0, 1, (n_sn, 3))


def add_stars(fname, masses, ages_yr, pos_box):
    with h5py.File(fname, "r+") as F:
        h = F["Header"]
        t, n = float(h.attrs["Time"]), len(masses)
        for k in ("NumPart_ThisFile", "NumPart_Total"):
            a = h.attrs[k].copy()
            a[4] = n
            h.attrs[k] = a
        for k in ("Flag_Sfr", "Flag_StellarAge", "Flag_Feedback"):
            h.attrs[k] = 1
        first = int(F["PartType0/ParticleIDs"][()].max()) + 1
        s = F.create_group("PartType4")
        s["Coordinates"] = pos_box * float(h.attrs["BoxSize"])
        s["Velocities"] = np.zeros((n, 3))
        s["ParticleIDs"] = np.arange(first, first + n, dtype=np.uint32)
        s["Masses"] = masses * MASS
        s["StellarFormationTime"] = t - ages_yr / YR_PER_CODE_TIME
        s["Metallicity"] = np.tile(Z_GAS, (n, 1))
        s["SingleStarDrawnMass"] = masses
        s["SingleStarBirthMetallicity"] = np.full(n, 0.014)
        s["SingleStarBirthDensity"] = np.full(n, 10.0)
        s["SingleStarEventBits"] = np.zeros(n)
        s["SingleStarLastWindLogAge"] = np.log10(ages_yr)
