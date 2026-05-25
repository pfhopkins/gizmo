"""Generate the ionneutral_drag test IC.

1D periodic box of two interleaved Type=0 fluids:
  FluidType = FLUID_DEFAULT (0) -> neutrals, v_x = 0,    B = 0
  FluidType = FLUID_ION     (2) -> ions,     v_x = +1e-4, B = B0_x

Uniform density, equal mass per particle per fluid, equal internal energy.
Periodic 1D, box length L=1, N particles per fluid.

Analytic decay: d(Δv)/dt = -γ_AD (ρ_i + ρ_n) Δv. With the units chosen in
ionneutral_drag.params (UnitLength=1cm, UnitVelocity=1cm/s, UnitMass=1.76e-15g)
γ_AD_code = γ_AD_cgs · UnitDensity · UnitTime ≈ 1.00, and ρ_i+ρ_n = 1 code,
so t_damp = 1 code time and Δv(t) = 1e-4 · exp(-t).

Run with: HYDRO_MULTIFLUID + HYDRO_MULTIFLUID_IONNEUTRAL + MAGNETIC.
"""

import numpy as np
import h5py


# FluidID enum values must match declarations/multifluid_helpers.h
FLUID_DEFAULT = 0
FLUID_ION     = 2

# Test parameters
N_PER_FLUID  = 32
LBOX         = 1.0
DV0          = 1.0e-4    # initial v_ion - v_neutral
U_INTERNAL   = 1.0e-6    # internal energy per unit mass; small (pressure-free regime)
B0_X         = 1.0e-6    # ion x-field (code units, with UnitMagneticField=1 Gauss)


def make_IC(fname="ionneutral_drag_ics.hdf5"):
    Ntotal = 2 * N_PER_FLUID
    m_part = 0.5 / N_PER_FLUID  # per-fluid total mass = 0.5 -> ρ_per_fluid = 0.5 in code

    # Interleave ions and neutrals on a uniform 1D lattice.
    dx = LBOX / Ntotal
    x = (np.arange(Ntotal) + 0.5) * dx
    # k%2==0 -> neutral, k%2==1 -> ion
    is_ion = (np.arange(Ntotal) % 2).astype(np.int32)
    fluid_type = np.where(is_ion == 1, FLUID_ION, FLUID_DEFAULT).astype(np.int32)

    coords = np.column_stack([x, np.zeros(Ntotal), np.zeros(Ntotal)]).astype(np.float64)
    vels   = np.zeros_like(coords)
    vels[:, 0] = np.where(is_ion == 1, DV0, 0.0)

    bvec = np.zeros_like(coords)
    bvec[:, 0] = np.where(is_ion == 1, B0_X, 0.0)

    masses = np.full(Ntotal, m_part, dtype=np.float64)
    uvec   = np.full(Ntotal, U_INTERNAL, dtype=np.float64)
    ids    = np.arange(1, Ntotal + 1, dtype=np.uint64)

    with h5py.File(fname, "w") as F:
        h = F.create_group("Header")
        h.attrs["NumPart_ThisFile"]      = np.array([Ntotal, 0, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumPart_Total"]         = np.array([Ntotal, 0, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"]             = np.zeros(6)
        h.attrs["Time"]                  = 0.0
        h.attrs["Redshift"]              = 0.0
        h.attrs["BoxSize"]               = LBOX
        h.attrs["NumFilesPerSnapshot"]   = 1
        h.attrs["Omega0"]                = 0.0
        h.attrs["OmegaLambda"]           = 0.0
        h.attrs["HubbleParam"]           = 1.0
        h.attrs["Flag_Sfr"]              = 0
        h.attrs["Flag_Cooling"]          = 0
        h.attrs["Flag_StellarAge"]       = 0
        h.attrs["Flag_Metals"]           = 0
        h.attrs["Flag_Feedback"]         = 0
        h.attrs["Flag_DoublePrecision"]  = 1

        p = F.create_group("PartType0")
        p.create_dataset("Coordinates",    data=coords)
        p.create_dataset("Velocities",     data=vels)
        p.create_dataset("ParticleIDs",    data=ids)
        p.create_dataset("Masses",         data=masses)
        p.create_dataset("InternalEnergy", data=uvec)
        p.create_dataset("MagneticField",  data=bvec)
        p.create_dataset("FluidType",      data=fluid_type)

    return fname


if __name__ == "__main__":
    f = make_IC()
    print(f"Wrote {f}: {2*N_PER_FLUID} particles, {N_PER_FLUID} ions + {N_PER_FLUID} neutrals.")
    print(f"  initial Δv = v_ion - v_neutral = {DV0:.1e} (code velocity)")
    print(f"  analytic t_damp = 1 (code time) with γ_AD_code = 1 and ρ_total = 1")
