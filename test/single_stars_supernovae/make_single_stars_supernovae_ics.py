"""Lattice of 1 Msun gas cells at n_H = 10 cm^-3, T = 100 K, solar metallicity (total Z, or the FIRE layout of 11 slots with n_metals=11), at rest, with four
single stars at rest-frame speed 10 km/s, each alone in its region and older than its table lifetime, so each dies in
the first step: a core-collapse SN (20 Msun), a failed SN (40 Msun), a pulsational pair-instability SN (150 Msun) and a
pair-instability SN (320 Msun). Units pc, Msun, km/s. Written as a snapshot (the run starts from it with start flag 2,
which reads the stellar ages, metallicities and star records)."""
import numpy as np
import h5py

PC_KMS_IN_YR = 3.085678e13 / 3.155e7  # time unit [yr], with the code's year (constants.h)
SOLAR = np.array([0.0142, 0.2703, 2.53e-3, 7.41e-4, 6.13e-3, 1.34e-3, 7.57e-4, 7.12e-4, 3.31e-4, 6.87e-5, 1.38e-3])
STARS = [(20.0, (0.25, 0.25, 0.25)), (40.0, (0.75, 0.25, 0.75)), (150.0, (0.25, 0.75, 0.75)), (320.0, (0.75, 0.75, 0.25))]


def make_single_stars_supernovae_ics(fname, tables_file, n=32, n_H=10.0, T=100.0, stars=STARS, z=0.014, n_metals=1, before_death_yr=None):
    rho = n_H * 1.6726e-24 / 0.7 * 3.0857e18 ** 3 / 1.989e33  # Msun / pc^3
    dx = (1.0 / rho) ** (1.0 / 3.0)
    L = n * dx
    g = (np.arange(n) + 0.5) * dx
    pos = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    N = len(pos)
    u = 1.5 * 1.380649e-16 * T / (1.27 * 1.6726e-24) / 1e10
    Zg = np.tile(SOLAR * z / SOLAR[0], (N, 1)); Zg[:, 1] = SOLAR[1]
    with h5py.File(tables_file, "r") as F:  # ages: 1% past PMS + lifetime at the table node nearest in Z
        Zt, Mt = F["Z"][()], F["M_init"][()]
        iz = int(np.argmin(np.abs(np.log10(Zt) - np.log10(z))))
        life = (F["lifetime_yr"][()] + F["t_PMS"][()]).reshape(len(Zt), len(Mt))[iz]
    death = [np.interp(np.log10(m), np.log10(Mt), life) for m, _ in stars]
    ages = [(1.01 * t if before_death_yr is None else t - before_death_yr) / PC_KMS_IN_YR for t in death]  # past death, or this long before it
    ns = len(stars)
    with h5py.File(fname, "w") as F:
        h = F.create_group("Header")
        npart = np.array([N, 0, 0, 0, ns, 0])
        h.attrs["NumPart_ThisFile"] = npart; h.attrs["NumPart_Total"] = npart
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32); h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = L; h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = h.attrs["OmegaLambda"] = h.attrs["HubbleParam"] = 0.0
        h.attrs["Flag_Sfr"] = 1; h.attrs["Flag_Cooling"] = 0; h.attrs["Flag_StellarAge"] = 1; h.attrs["Flag_Metals"] = n_metals
        h.attrs["Flag_Feedback"] = 1; h.attrs["Flag_DoublePrecision"] = 1
        p = F.create_group("PartType0")
        p["Coordinates"] = pos; p["Velocities"] = np.zeros((N, 3)); p["ParticleIDs"] = np.arange(1, N + 1, dtype=np.uint32)
        p["Masses"] = np.ones(N); p["InternalEnergy"] = np.full(N, u); p["Metallicity"] = Zg[:, :n_metals]
        p["SmoothingLength"] = np.full(N, 2.0 * dx)  # a start guess (start flag 2 reads it)
        s = F.create_group("PartType4")
        s["Coordinates"] = np.array([np.asarray(x) * L + 0.37 * dx for _, x in stars])
        s["Velocities"] = np.array([[10.0, -6.0, 3.0]] * ns); s["ParticleIDs"] = np.arange(N + 1, N + ns + 1, dtype=np.uint32)
        s["Masses"] = np.array([m for m, _ in stars]); s["StellarFormationTime"] = -np.array(ages)
        Zs = np.tile(SOLAR * z / SOLAR[0], (ns, 1)); Zs[:, 1] = SOLAR[1]
        s["Metallicity"] = Zs[:, :n_metals]; s["SingleStarDrawnMass"] = np.array([m for m, _ in stars]); s["SingleStarBirthMetallicity"] = np.full(ns, z)
    return L, dx
