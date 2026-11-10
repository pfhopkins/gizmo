"""Lattice of 1 Msun gas cells at n_H = 10 cm^-3, T = 100 K, Z = 0.014 (total metallicity), at rest, with three single
stars, each alone in its region (units pc, Msun, km/s): a 35 Msun star 1 Myr old (main-sequence winds), a 3 Msun star
just past its table lifetime (AGB death, white-dwarf remnant), and a white dwarf of drawn mass 3 Msun, 1 Gyr old, whose
AGB death is already done (event bit 2), which explodes as a Type Ia SN (`which` picks a subset). Written as a snapshot (start flag 2)."""
import numpy as np
import h5py

PC_KMS_IN_YR = 3.085678e13 / 3.155e7  # time unit [yr], with the code's year (constants.h)


def make_single_stars_winds_agb_ia_ics(fname, tables_file, n=24, n_H=10.0, T=100.0, z=0.014, M=35.0, v_radial=0.0, last_wind_log_age=None, which=("wind", "agb", "wd"), n_metals=1, wind_age_yr=1.0e6, gas_metals=True):
    """v_radial [km/s]: gas moves radially out of the star, Gaussian of 3 cells; last_wind_log_age: winds returned up to this table age;
    n_metals 11: the FIRE layout (total, He, C, N, O, Ne, Mg, Si, S, Ca, Fe), solar scaled to z; wind_age_yr: age of the wind star;
    gas_metals False: metal-free gas"""
    solar = np.array([0.0142, 0.2703, 2.53e-3, 7.41e-4, 6.13e-3, 1.34e-3, 7.57e-4, 7.12e-4, 3.31e-4, 6.87e-5, 1.38e-3])
    Zv = (solar * z / solar[0])[:n_metals]
    if n_metals > 1:
        Zv[1] = solar[1]
    rho = n_H * 1.6726e-24 / 0.7 * 3.0857e18 ** 3 / 1.989e33
    dx = (1.0 / rho) ** (1.0 / 3.0)
    L = n * dx
    g = (np.arange(n) + 0.5) * dx
    pos = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    N = len(pos)
    u = 1.5 * 1.380649e-16 * T / (1.27 * 1.6726e-24) / 1e10
    with h5py.File(tables_file, "r") as F:
        Zt, Mt = F["Z"][()], F["M_init"][()]
        iz, im = int(np.argmin(np.abs(Zt - z))), int(np.argmin(np.abs(Mt - 3.0)))
        n2 = (len(Zt), len(Mt))
        death3 = (F["lifetime_yr"][()] + F["t_PMS"][()]).reshape(n2)[iz, im]
        wd3 = F["remnant_mass"][()].reshape(n2)[iz, im]
    stars = [  # drawn mass, mass, age [yr], position, event bits
        (M, M, wind_age_yr, (0.25, 0.25, 0.25), 0),
        (3.0, 3.0, 1.001 * death3, (0.75, 0.75, 0.25), 0),
        (3.0, wd3, 1.0e9, (0.25, 0.75, 0.75), 2)]
    stars = [st for k, st in zip(("wind", "agb", "wd"), stars) if k in which]
    ns = len(stars)
    r = pos - (np.asarray(stars[0][3]) * L + 0.37 * dx)
    r_abs = np.linalg.norm(r, axis=1)
    vel = v_radial * np.exp(-(r_abs / (3.0 * dx)) ** 2)[:, None] * r / r_abs[:, None]
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
        p["Coordinates"] = pos; p["Velocities"] = vel; p["ParticleIDs"] = np.arange(1, N + 1, dtype=np.uint32)
        p["Masses"] = np.ones(N); p["InternalEnergy"] = np.full(N, u); p["Metallicity"] = np.tile(Zv if gas_metals else 0 * Zv, (N, 1))
        p["SmoothingLength"] = np.full(N, 2.0 * dx)
        s = F.create_group("PartType4")
        s["Coordinates"] = np.array([np.asarray(x) * L + 0.37 * dx for _, _, _, x, _ in stars])
        s["Velocities"] = np.array([[5.0, 2.0, -1.0]] * ns); s["ParticleIDs"] = np.arange(N + 1, N + ns + 1, dtype=np.uint32)
        s["Masses"] = np.array([m for _, m, _, _, _ in stars]); s["StellarFormationTime"] = -np.array([a for _, _, a, _, _ in stars]) / PC_KMS_IN_YR
        s["Metallicity"] = np.tile(Zv, (ns, 1)); s["SingleStarDrawnMass"] = np.array([md for md, _, _, _, _ in stars])
        s["SingleStarBirthMetallicity"] = np.full(ns, z); s["SingleStarEventBits"] = np.array([b for _, _, _, _, b in stars], dtype=float)
        if last_wind_log_age is not None:
            s["SingleStarLastWindLogAge"] = np.full(ns, last_wind_log_age)
    return L
