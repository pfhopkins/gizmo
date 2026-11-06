"""Stars in the gradient cells of mechfb_healpix_weights, hydro frozen. Units pc, Msun, km/s. A snapshot (start flag 2).
Events: 'sn' 20 Msun past its lifetime; 'wind' 35 Msun at 1 Myr, winds owed since 0.7 Myr; 'agb' 3 Msun past its lifetime; 'moving' as wind,
gas and star moving; 'overlap' two wind stars 0.5 pc apart; 'ray' wind star with all cells on one side along x; 'rp' 7 Msun star, radiation only."""
import sys
from pathlib import Path
import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent / "mechfb_healpix_weights"))
from make_mechfb_healpix_weights_ics import cells, PC_KMS_IN_YR  # noqa: E402

KIND = {"sn": 20.0, "wind": 35.0, "moving": 35.0, "overlap": 35.0, "ray": 35.0, "agb": 3.0, "rp": 7.0}


def lifetime(tables_file, M, z):
    with h5py.File(tables_file, "r") as F:
        Zt, Mt = F["Z"][()], F["M_init"][()]
        iz = int(np.argmin(np.abs(np.log10(Zt) - np.log10(z))))
        life = (F["lifetime_yr"][()] + F["t_PMS"][()]).reshape(len(Zt), len(Mt))[iz]
    return np.interp(np.log10(M), np.log10(Mt), life)


def make_ics(fname, tables_file, event, L=20.0, seed=3, z=0.014, T=100.0):
    rng = np.random.default_rng(seed)
    M = KIND[event]
    pos = cells(L, rng)
    i0 = int(np.argmin(np.sum((pos - 0.5 * L) ** 2, axis=1)))
    pos = (np.delete(pos, i0, axis=0) - pos[i0] + 0.5 * L) % L
    if event == "ray":
        pos = np.column_stack([0.5 * L + 0.05 + 0.1 * np.arange(60.0), np.full(60, 0.5 * L), np.full(60, 0.5 * L)])
    N = len(pos)
    vel = np.zeros((N, 3)); star_v = np.zeros((1, 3))
    if event == "moving":
        vel = np.array([3.0, -2.0, 1.0]) + rng.normal(0, 1.0, (N, 3)); star_v[0] = [-5.0, 4.0, 2.0]
    if event in ("sn", "agb"):
        age = 1.01 * lifetime(tables_file, M, z); last = np.log10(age / 1.01)
    elif event == "rp":
        age = 1e6; last = -1.0
    else:
        age = 1e6; last = np.log10(7e5)
    spos = np.array([[0.5 * L] * 3] + ([[0.5 * L + 0.5, 0.5 * L, 0.5 * L]] if event == "overlap" else []))
    Ns = len(spos)
    u = 1.5 * 1.380649e-16 * T / (1.27 * 1.6726e-24) / 1e10
    with h5py.File(fname, "w") as F:
        h = F.create_group("Header"); npart = np.array([N, 0, 0, 0, Ns, 0])
        h.attrs["NumPart_ThisFile"] = npart; h.attrs["NumPart_Total"] = npart
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32); h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = L; h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = h.attrs["OmegaLambda"] = h.attrs["HubbleParam"] = 0.0
        h.attrs["Flag_Sfr"] = 1; h.attrs["Flag_Cooling"] = 0; h.attrs["Flag_StellarAge"] = 1; h.attrs["Flag_Metals"] = 1
        h.attrs["Flag_Feedback"] = 1; h.attrs["Flag_DoublePrecision"] = 1
        p = F.create_group("PartType0")
        p["Coordinates"] = pos; p["Velocities"] = vel; p["ParticleIDs"] = np.arange(1, N + 1, dtype=np.uint32)
        p["Masses"] = np.ones(N); p["InternalEnergy"] = np.full(N, u); p["Metallicity"] = np.full((N, 1), z)
        p["SmoothingLength"] = np.full(N, 1.0)
        s = F.create_group("PartType4")
        s["Coordinates"] = spos; s["Velocities"] = np.repeat(star_v, Ns, axis=0); s["ParticleIDs"] = np.arange(N + 1, N + 1 + Ns, dtype=np.uint32)
        s["Masses"] = np.full(Ns, M); s["StellarFormationTime"] = np.full(Ns, -age / PC_KMS_IN_YR); s["Metallicity"] = np.full((Ns, 1), z)
        s["SingleStarDrawnMass"] = np.full(Ns, M); s["SingleStarBirthMetallicity"] = np.full(Ns, z); s["SingleStarLastWindLogAge"] = np.full(Ns, last)
    return L
