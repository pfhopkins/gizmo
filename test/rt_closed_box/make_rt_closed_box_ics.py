#!/usr/bin/env python3
"""Initial conditions for rt_closed_box: a static, uniform, periodic box of dusty gas (32^3 glass) holding a uniform
radiation field in every band of the build, optionally with one star at the centre. The file is a snapshot (read with
restart flag 2), because the photon energies are only read from snapshots: a flag-0 start resets them."""

import os
import urllib.request

import h5py
import numpy as np

BOX_SIZE = 8.0  # pc
N_H = 1.0e3  # cm^-3
X_H = 0.76  # GIZMO's HYDROGEN_MASSFRAC
T_GAS = 10.0  # K
T_RAD = 10.0  # K, initial IR radiation and dust temperature
Z_SOLAR = 0.02  # GIZMO's All.SolarAbundances[0]: dust opacities at their solar values
# All.SolarAbundances for the 11 species of a COOL_METAL_LINES_BY_SPECIES build: Z, He, C, N, O, Ne, Mg, Si, S, Ca, Fe
Z_SOLAR_SPECIES = (0.02, 0.28, 3.26e-3, 1.32e-3, 8.65e-3, 2.22e-3, 9.31e-4, 1.08e-3, 6.44e-4, 1.01e-4, 1.73e-3)
# initial energy density per band [eV cm^-3], in bin order: photoelectric, NUV, optical-NIR, IR; a STARFORGE build
# (RT_CHEM_PHOTOION) puts the ionizing band first
U_BANDS_EV = np.array([20.0, 20.0, 20.0, 30.0])
U_BANDS_EV_IONIZING = np.array([20.0, 20.0, 20.0, 20.0, 30.0])
PC_CM, MSUN_G, PROTONMASS_CGS, BOLTZMANN_CGS, EV_ERG = 3.085678e18, 1.989e33, 1.6726e-24, 1.38066e-16, 1.602177e-12
RHO_UNIT = MSUN_G / PC_CM**3  # g cm^-3
EGY_DENSITY_UNIT = RHO_UNIT * 1e10  # erg cm^-3, with km/s velocities


def ms_radius_solar(m):
    """ps_radius_MS_in_solar (galaxy_sf/stellar_evolution.cc), m in Msun"""
    return ((1.715359 * m**2.5 + 6.597788 * m**6.5 + 10.08855 * m**11 + 1.012495 * m**19 + 0.07490166 * m**19.5)
            / (0.01077422 + 3.082234 * m**2 + 17.84778 * m**8.5 + m**18.5 + 0.00022582 * m**19.5))


def make_rt_closed_box_ics(output_file="rt_closed_box_ics.hdf5", glass="glass_32.hdf5", n_h=N_H, t_gas=T_GAS,
                           t_rad=T_RAD, u_bands_ev=U_BANDS_EV, metallicity=Z_SOLAR, neutral_hydrogen=False,
                           star_mass=None, source_mass=None, h2_per_neutral_h=None, inset=False):
    """metallicity: total metal mass fraction, or one per species (it must match the build's NUM_METAL_SPECIES).
    neutral_hydrogen: write the H ionization state of an RT_CHEM_PHOTOION build (neutral). star_mass [Msun]: add a
    main-sequence STARFORGE star (Type 5, its sink properties set as a flag-2 start reads them) at the box centre.
    source_mass [Msun]: add the ionizing source of a non-GALSF RT_CHEM_PHOTOION build (Type 4) near the box centre, offset
    from the glass cell that sits exactly at the centre: the injection skips a cell at zero distance while the density
    loop's kernel sum counts it, which would lose 1/(kernel sum) of the luminosity.
    h2_per_neutral_h: write the H2 fraction per neutral H that a COOL_MOLECFRAC_NONEQM snapshot stores.
    inset: put the gas cube in the middle of a box 1/0.8 times larger, so no cell lies within 10% of the box edge, where
    RT_ISRF_BACKGROUND resets the radiation field every kick (rt_apply_boundary_conditions), for builds that keep the
    background on. The cube's surface has no neighbours outside it, so nothing crosses it."""
    z = np.atleast_1d(np.asarray(metallicity, dtype=np.float32))
    glass_file = os.path.join(os.path.dirname(output_file) or ".", glass)
    if not os.path.exists(glass_file):
        urllib.request.urlretrieve(f"https://users.flatironinstitute.org/~mgrudic/glass/{glass}", glass_file)
    with h5py.File(glass_file, "r") as f:
        pos = f["Coordinates"][:].astype(np.float64) * BOX_SIZE  # periodic unit glass
    box = BOX_SIZE / 0.8 if inset else BOX_SIZE
    if inset:
        pos += 0.1 * box
    ngas = len(pos)
    rho = n_h * PROTONMASS_CGS / X_H / RHO_UNIT  # code units
    m_gas = rho * BOX_SIZE**3 / ngas
    v_cell = m_gas / rho
    u_gas = BOLTZMANN_CGS * t_gas / ((5.0 / 3.0 - 1.0) * 1.27 * PROTONMASS_CGS) / 1e10  # (km/s)^2, neutral atomic
    e_bands = np.asarray(u_bands_ev) * EV_ERG / EGY_DENSITY_UNIT * v_cell  # code energy per cell
    with h5py.File(output_file, "w") as f:
        h = f.create_group("Header")
        npart = np.array([ngas, 0, 0, 0, 0 if source_mass is None else 1, 0 if star_mass is None else 1], dtype=np.int32)
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = box
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 0.0
        h.attrs["OmegaLambda"] = 0.0
        h.attrs["HubbleParam"] = 1.0
        for flag in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Feedback", "Flag_DoublePrecision"):
            h.attrs[flag] = 0
        h.attrs["Flag_Metals"] = len(z)
        g = f.create_group("PartType0")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=np.zeros((ngas, 3), dtype=np.float32))
        g.create_dataset("Masses", data=np.full(ngas, m_gas, dtype=np.float32))
        g.create_dataset("ParticleIDs", data=np.arange(1, ngas + 1, dtype=np.uint32))
        g.create_dataset("InternalEnergy", data=np.full(ngas, u_gas, dtype=np.float32))
        g.create_dataset("Density", data=np.full(ngas, rho, dtype=np.float32))
        g.create_dataset("KernelMaxRadius", data=np.full(ngas, 2.0 * v_cell ** (1 / 3), dtype=np.float32))
        g.create_dataset("PhotonEnergy", data=np.tile(e_bands, (ngas, 1)).astype(np.float32))
        g.create_dataset("IRBand_Radiation_Temperature", data=np.full(ngas, t_rad, dtype=np.float32))
        g.create_dataset("Dust_Temperature", data=np.full(ngas, t_rad, dtype=np.float32))
        g.create_dataset("Metallicity", data=np.tile(z, (ngas, 1)))
        if h2_per_neutral_h is not None:
            g.create_dataset("MolecularMassFraction", data=np.full(ngas, h2_per_neutral_h, dtype=np.float32))
        if neutral_hydrogen:
            g.create_dataset("NeutralHydrogenAbundance", data=np.full(ngas, 1.0, dtype=np.float32))
            g.create_dataset("HII", data=np.full(ngas, 1e-6, dtype=np.float32))
            g.create_dataset("ElectronAbundance", data=np.full(ngas, 1e-6, dtype=np.float32))
        if star_mass is not None:
            m = np.array([star_mass], dtype=np.float32)
            s = f.create_group("PartType5")
            s.create_dataset("Coordinates", data=np.full((1, 3), 0.5 * box))
            s.create_dataset("Velocities", data=np.zeros((1, 3), dtype=np.float32))
            s.create_dataset("Masses", data=m)
            s.create_dataset("ParticleIDs", data=np.array([ngas + 1], dtype=np.uint32))
            s.create_dataset("Metallicity", data=z.reshape(1, -1))
            s.create_dataset("Sink_Mass", data=m)
            s.create_dataset("Sink_InitialMass", data=m)
            s.create_dataset("ZAMS_Mass", data=m)
            s.create_dataset("ProtoStellarStage", data=np.array([5], dtype=np.int32))  # main sequence
            s.create_dataset("ProtoStellarRadius_inSolar", data=np.array([ms_radius_solar(star_mass)], dtype=np.float32))
            s.create_dataset("Sink_Radius", data=np.array([1e-4], dtype=np.float32))  # pc: no cell can be captured
        if source_mass is not None:
            s = f.create_group("PartType4")
            s.create_dataset("Coordinates", data=np.full((1, 3), 0.5 * box) + np.array([[0.1, 0.07, 0.03]]) * BOX_SIZE / 32)
            s.create_dataset("Velocities", data=np.zeros((1, 3), dtype=np.float32))
            s.create_dataset("Masses", data=np.array([source_mass], dtype=np.float32))
            s.create_dataset("ParticleIDs", data=np.array([ngas + 2], dtype=np.uint32))
    return output_file


if __name__ == "__main__":
    make_rt_closed_box_ics()
