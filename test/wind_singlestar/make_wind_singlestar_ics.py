#!/usr/bin/env python3
"""Generate ICs for wind_singlestar test: a 100 Msun ZAMS star at the center of a
uniform gas box. Total gas mass is 50000 Msun, box size set so the ambient hydrogen
number density is n_H = 100 cm^-3 assuming GIZMO's default solar metallicity
(X_H = 0.70)."""

import os
import urllib.request
import numpy as np
import h5py


# GIZMO's default solar metallicity table: [Z_total, He, C, N, O, Ne, Mg, Si, S, Ca, Fe]
Z_SOLAR = np.array(
    [0.02, 0.28, 2.53e-3, 7.41e-4, 6.13e-3, 1.20e-3, 5.91e-4, 6.83e-4, 4.09e-4, 6.44e-5, 1.17e-3],
    dtype=np.float32,
)
X_H = float(1.0 - Z_SOLAR[0] - Z_SOLAR[1])  # hydrogen mass fraction = 0.70

# Physical constants
M_PROTON_G = 1.6726e-24
PC_IN_CM = 3.0857e18
MSUN_IN_G = 1.989e33

RES = 128


def make_wind_singlestar_ics(output_file="wind_singlestar_ics.hdf5"):
    # Download glass file for particle positions
    glass_url = f"https://users.flatironinstitute.org/~mgrudic/glass/glass_{RES}.hdf5"
    glass_file = os.path.join(os.path.dirname(output_file) or ".", f"glass_{RES}.hdf5")
    if not os.path.exists(glass_file):
        print(f"Downloading {glass_url}...")
        urllib.request.urlretrieve(glass_url, glass_file)

    with h5py.File(glass_file, "r") as f:
        pos_glass = f["Coordinates"][:]
    Ngas = len(pos_glass)

    # Gas mass and box size: total gas mass = 50000 Msun, ambient n_H = 100 cm^-3
    M_gas_total = 50000.0  # Msun
    n_H = 100.0  # cm^-3
    rho_cgs = n_H * M_PROTON_G / X_H  # g/cm^3
    rho_code = rho_cgs * (PC_IN_CM**3) / MSUN_IN_G  # Msun/pc^3
    box_volume = M_gas_total / rho_code  # pc^3
    BoxSize = box_volume ** (1.0 / 3.0)  # pc
    m_gas = M_gas_total / Ngas

    # Rescale glass positions to [0, BoxSize)
    pos_min = pos_glass.min(axis=0)
    pos_max = pos_glass.max(axis=0)
    glass_size = (pos_max - pos_min).max()
    pos_gas = (pos_glass - pos_min) / (glass_size * (1.0 + 1e-6)) * BoxSize
    vel_gas = np.zeros((Ngas, 3), dtype=np.float32)

    # Star: 100 Msun, zero age (ZAMS)
    M_star = 100.0

    pos_star = np.array([[BoxSize / 2, BoxSize / 2, BoxSize / 2]])
    vel_star = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    ids_gas = np.arange(1, Ngas + 1, dtype=np.int32)
    ids_star = np.array([Ngas + 1], dtype=np.int32)

    with h5py.File(output_file, "w") as f:
        npart = np.array([Ngas, 0, 0, 0, 0, 1], dtype=np.int32)
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = BoxSize
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 0.0
        h.attrs["OmegaLambda"] = 0.0
        h.attrs["HubbleParam"] = 1.0
        h.attrs["Flag_Sfr"] = 0
        h.attrs["Flag_Cooling"] = 0
        h.attrs["Flag_StellarAge"] = 1
        h.attrs["Flag_Metals"] = 11
        h.attrs["Flag_Feedback"] = 0
        h.attrs["Flag_DoublePrecision"] = 0

        # PartType0 (Gas)
        g = f.create_group("PartType0")
        g.create_dataset("Coordinates", data=pos_gas.astype(np.float64))
        g.create_dataset("Velocities", data=vel_gas)
        g.create_dataset("Masses", data=np.full(Ngas, m_gas, dtype=np.float32))
        g.create_dataset("ParticleIDs", data=ids_gas)
        g.create_dataset("Metallicity", data=np.tile(Z_SOLAR, (Ngas, 1)))

        # PartType5 (Sink / Star)
        s = f.create_group("PartType5")
        s.create_dataset("Coordinates", data=pos_star.astype(np.float64))
        s.create_dataset("Velocities", data=vel_star)
        s.create_dataset("Masses", data=np.array([M_star], dtype=np.float32))
        s.create_dataset("ParticleIDs", data=ids_star)
        s.create_dataset("Metallicity", data=Z_SOLAR.reshape(1, -1))
        s.create_dataset("StellarFormationTime", data=np.array([0.0], dtype=np.float32))
        # Main-sequence star (stage 5)
        s.create_dataset("ProtoStellarStage", data=np.array([5], dtype=np.int32))
        s.create_dataset(
            "ProtoStellarRadius_inSolar",
            data=np.array([max(10.0 * M_star, 5.24 * M_star ** (1.0 / 3.0))], dtype=np.float32),
        )
        s.create_dataset("StarLuminosity_Solar", data=np.array([1.0], dtype=np.float32))
        s.create_dataset("ZAMS_Mass", data=np.array([M_star], dtype=np.float32))
        s.create_dataset("ProtoStellarAge", data=np.array([0.0], dtype=np.float32))
        s.create_dataset("Sink_Formation_Mass", data=np.array([M_star], dtype=np.float32))

    print(f"Wrote {output_file} with {Ngas} gas particles and 1 star particle")
    print(f"  BoxSize = {BoxSize:.4f} pc")
    print(f"  Gas mass per particle = {m_gas:.6e} Msun  (total = {M_gas_total} Msun)")
    print(f"  Ambient n_H = {n_H} cm^-3  (X_H = {X_H:.3f})")
    print(f"  Star: M = {M_star} Msun (ZAMS, zero age)")


if __name__ == "__main__":
    make_wind_singlestar_ics()
