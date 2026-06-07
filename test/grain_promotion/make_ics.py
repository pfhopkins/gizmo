"""IC generator for Phase 17g grain-promotion smoke test.

2D periodic box (10 cm × 10 cm):
  - 8×8 = 64 gas particles (Type 0) on a uniform grid
  - 1 grain particle (Type 3) at centre, mass=0.02g >> threshold=0.01g, ~1.3x gas particle mass

CGS units; UnitLength=UnitMass=UnitVelocity=1.
"""

import os
import numpy as np
import h5py

BOX      = 10.0          # cm  (periodic side length)
N_SIDE   = 8             # gas particles per side (total: N_SIDE**2)
GAS_DENS = 0.01          # g/cm² (2D surface density)
GRAIN_MASS = 0.02        # g  (~1.3x gas particle mass, 2x threshold=0.01g; prevents mass-ratio splits)
GRAIN_SIZE = 0.01        # cm (grain radius)


def make_ics(outfile=None):
    if outfile is None:
        outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "grain_promotion_ics.hdf5")
    N_gas  = N_SIDE * N_SIDE
    dx     = BOX / N_SIDE

    # Uniform gas grid
    ix, iy = np.meshgrid(np.arange(N_SIDE), np.arange(N_SIDE))
    gas_x  = (ix.ravel() + 0.5) * dx          # cm
    gas_y  = (iy.ravel() + 0.5) * dx
    gas_z  = np.zeros(N_gas)
    gas_pos = np.column_stack([gas_x, gas_y, gas_z])

    gas_mass = np.full(N_gas, GAS_DENS * dx * dx, dtype=np.float64)
    gas_vel  = np.zeros((N_gas, 3), dtype=np.float64)
    gas_rho  = np.full(N_gas, GAS_DENS, dtype=np.float64)
    gas_u    = np.full(N_gas, 1.0e8, dtype=np.float64)   # warm gas

    # KernelMaxRadius: h = sqrt(Nngb * m / (pi * rho))  (2D kernel)
    Nngb = 16
    gas_h = np.sqrt(Nngb * gas_mass[0] / (np.pi * GAS_DENS)) * np.ones(N_gas)

    # One grain particle at box centre, Type 3
    grain_pos = np.array([[BOX / 2, BOX / 2, 0.0]], dtype=np.float64)
    grain_vel = np.zeros((1, 3), dtype=np.float64)
    grain_h   = np.array([gas_h[0]], dtype=np.float64)

    ids_gas   = np.arange(1, N_gas + 1, dtype=np.uint32)
    ids_grain = np.array([N_gas + 1], dtype=np.uint32)
    child_ids = np.zeros(N_gas + 1, dtype=np.uint32)
    gen_ids   = np.zeros(N_gas + 1, dtype=np.uint32)

    with h5py.File(outfile, "w") as f:
        hdr = f.create_group("Header")
        npart = np.array([N_gas, 0, 0, 1, 0, 0], dtype=np.int32)
        hdr.attrs["NumPart_ThisFile"]       = npart
        hdr.attrs["NumPart_Total"]          = npart.astype(np.uint32)
        hdr.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        hdr.attrs["MassTable"]              = np.zeros(6, dtype=np.float64)
        hdr.attrs["Time"]                   = np.float64(0.0)
        hdr.attrs["Redshift"]               = np.float64(0.0)
        hdr.attrs["BoxSize"]                = np.float64(BOX)
        hdr.attrs["NumFilesPerSnapshot"]    = np.int32(1)
        hdr.attrs["Omega0"]                 = np.float64(1.0)
        hdr.attrs["OmegaLambda"]            = np.float64(0.0)
        hdr.attrs["HubbleParam"]            = np.float64(1.0)
        hdr.attrs["Flag_Sfr"]               = np.int32(0)
        hdr.attrs["Flag_Feedback"]          = np.int32(0)
        hdr.attrs["Flag_Cooling"]           = np.int32(0)
        hdr.attrs["Flag_StellarAge"]        = np.int32(0)
        hdr.attrs["Flag_Metals"]            = np.int32(0)
        hdr.attrs["Flag_DoublePrecision"]   = np.int32(0)
        hdr.attrs["Flag_IC_Info"]           = np.int32(0)
        hdr.attrs["Flag_AgeTracers"]        = np.int32(0)

        # Gas (PartType0)
        g = f.create_group("PartType0")
        g.create_dataset("Coordinates",              data=gas_pos)
        g.create_dataset("Velocities",               data=gas_vel)
        g.create_dataset("Masses",                   data=gas_mass)
        g.create_dataset("Density",                  data=gas_rho)
        g.create_dataset("InternalEnergy",           data=gas_u)
        g.create_dataset("KernelMaxRadius",           data=gas_h)
        g.create_dataset("ParticleIDs",              data=ids_gas)
        g.create_dataset("ParticleChildIDsNumber",   data=child_ids[:N_gas])
        g.create_dataset("ParticleIDGenerationNumber", data=gen_ids[:N_gas])

        # Grain (PartType3)
        g3 = f.create_group("PartType3")
        g3.create_dataset("Coordinates",             data=grain_pos)
        g3.create_dataset("Velocities",              data=grain_vel)
        g3.create_dataset("Masses",                  data=np.array([GRAIN_MASS], dtype=np.float64))
        g3.create_dataset("KernelMaxRadius",          data=grain_h)
        g3.create_dataset("GrainSize",               data=np.array([GRAIN_SIZE], dtype=np.float64))
        g3.create_dataset("ParticleIDs",             data=ids_grain)
        g3.create_dataset("ParticleChildIDsNumber",  data=child_ids[N_gas:])
        g3.create_dataset("ParticleIDGenerationNumber", data=gen_ids[N_gas:])

    print(f"Written {N_gas} gas + 1 grain particle (mass={GRAIN_MASS:.1e} g) to {outfile}")
    return N_gas


if __name__ == "__main__":
    make_ics()
