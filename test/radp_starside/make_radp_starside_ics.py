"""Uniform dusty box for the star-side radiation pressure test: gas only, particle ID 1 at the centre becomes the star
(GALSF_RESOLVEDISM_ISOLATED_FB_TEST + TestStarMass). Code units kpc, 1e10 Msun, km/s."""
import sys
import h5py
import numpy as np

out, box_pc, nH, nside, Zsun = sys.argv[1], 6.0, 300.0, 14, 1.0
PC, MP, MSUN = 3.085678e18, 1.6726e-24, 1.989e33
L = box_pc / 1000.0
X_metals = Zsun * 0.014
X_He = 0.275
X_H = 1.0 - X_He - X_metals
m_tot = nH * (box_pc * PC) ** 3 * MP / X_H / MSUN
N = nside ** 3
ax = (np.arange(nside) + 0.5) * (L / nside)
pos = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
ids = np.arange(N, dtype=np.uint32) + 2
ids[np.argmin(((pos - L / 2) ** 2).sum(1))] = 1
T = 100.0
u = 1.5 * 1.3807e-16 * T / (1.22 * MP) / 1e10
EA = np.zeros((N, 27), dtype=np.float32)
EA[:, 1] = X_He
for k, x in {2: 2.53e-3, 3: 7.41e-4, 4: 6.13e-3, 5: 1.34e-3, 6: 7.57e-4, 7: 7.12e-4, 8: 3.31e-4, 9: 6.87e-5, 10: 1.38e-3}.items():
    EA[:, k] = x * Zsun
EA[:, 0] = 1.0 - EA.sum(axis=1)
B = np.zeros((N, 3), dtype=np.float32); B[:, 0] = 1e-8
with h5py.File(out, "w") as f:
    h = f.create_group("Header")
    h.attrs.update(BoxSize=L, NumPart_ThisFile=np.array([N, 0, 0, 0, 0, 0], dtype=np.int32), NumPart_Total=np.array([N, 0, 0, 0, 0, 0], dtype=np.uint32),
                   NumPart_Total_HighWord=np.zeros(6, dtype=np.uint32), MassTable=np.zeros(6), Time=0.0, Redshift=0.0, NumFilesPerSnapshot=1,
                   Flag_Sfr=1, Flag_Cooling=0, Flag_StellarAge=1, Flag_Metals=27, Flag_Feedback=1, Flag_DoublePrecision=0, HubbleParam=1.0,
                   ComovingIntegrationOn=0, UnitLength_In_CGS=3.085678e21, UnitMass_In_CGS=1.989e43, UnitVelocity_In_CGS=1e5)
    g = f.create_group("PartType0")
    g["Coordinates"] = pos; g["Velocities"] = np.zeros((N, 3), dtype=np.float32)
    g["Masses"] = np.full(N, m_tot / N / 1e10, dtype=np.float32); g["ParticleIDs"] = ids
    g["InternalEnergy"] = np.full(N, u, dtype=np.float32); g["ElementAbundance"] = EA; g["MagneticField"] = B
print(f"{N} cells of {m_tot / N:.2f} Msun")
