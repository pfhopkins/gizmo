"""IC generator for the CD21 H/He compression smoke test.

Builds a 1D periodic box of H/He fluid evaluated through the Chabrier &
Debras 2021 (Y=0.275) EOS, loaded via the existing EOS_ANEOS dispatch by
running cms_to_sesame.py end-to-end. A small adiabatic compression is
driven by a single-mode sinusoidal velocity perturbation; the test only
checks that the run completes with finite, positive (rho, P, u). Physics
validation (Jupiter/Saturn polytrope vs published interior models) is
deferred.

Units: CGS throughout (UnitLength=UnitMass=UnitVelocity=1 -> code = CGS).

CompositionType = 0 in EOS_ANEOS-only builds (single AneosTable0).
"""

import os
import sys
import numpy as np
import h5py


N_PARTICLES = 128
BOX_LEN     = 10.0          # cm  (matches cd21_hhe_compression.params BoxSize)
RHO_BG      = 1.0e-2        # g/cm^3 (giant-planet outer envelope-ish)
T_BG        = 1.0e3         # K
V_AMPL      = 1.0e3         # cm/s — small enough to stay adiabatic + linear
COMP_TYPE   = 0
Y_HE        = 0.275

# Matter properties for u_from_T fallback. Cv ≈ R/(mu*(gamma-1)) with mu≈1.23,
# gamma≈5/3 in this regime. Used only if SesameTable.internal_energy() is
# queried outside the table; otherwise the table itself supplies u(rho,T).
CV_HHE      = 1.0e8         # erg/g/K (approximate)


def main(out=None):
    here = os.path.dirname(os.path.abspath(__file__))
    if out is None:
        out = os.path.join(here, "compression_ics.hdf5")

    # Particle positions: uniform 1D lattice along x. Y=Z fixed at half-box.
    x = (np.arange(N_PARTICLES) + 0.5) * BOX_LEN / N_PARTICLES
    pos = np.zeros((N_PARTICLES, 3), dtype=np.float64)
    pos[:, 0] = x
    pos[:, 1] = 0.5 * BOX_LEN
    pos[:, 2] = 0.5 * BOX_LEN

    # Single-mode sinusoidal velocity along x to drive compression.
    vel = np.zeros((N_PARTICLES, 3), dtype=np.float64)
    vel[:, 0] = V_AMPL * np.sin(2.0 * np.pi * x / BOX_LEN)

    # Equal masses chosen so mean density = RHO_BG. With BOX_SPATIAL_DIMENSION=1,
    # GIZMO interprets the simulation as 1D so per-particle "volume" = dx;
    # mass = rho * dx (lengths in y/z drop out of the 1D kernel sum).
    dx = BOX_LEN / N_PARTICLES
    masses = np.full(N_PARTICLES, RHO_BG * dx, dtype=np.float64)

    # Internal energy from analytic ideal-gas-like relation; the actual u
    # consumed at run-time is reset by the InitGasTemp param (set to 1000 K
    # in the .params file), so this just needs to be positive + finite.
    u = np.full(N_PARTICLES, CV_HHE * T_BG, dtype=np.float64)

    comp = np.full(N_PARTICLES, COMP_TYPE, dtype=np.int32)
    ids = np.arange(1, N_PARTICLES + 1, dtype=np.uint32)

    with h5py.File(out, "w") as f:
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"]      = np.array([N_PARTICLES, 0, 0, 0, 0, 0], dtype=np.int32)
        h.attrs["NumPart_Total"]         = np.array([N_PARTICLES, 0, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumPart_Total_HighWord"]= np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"]             = np.zeros(6, dtype=np.float64)
        h.attrs["Time"]                  = 0.0
        h.attrs["Redshift"]              = 0.0
        h.attrs["BoxSize"]               = float(BOX_LEN)
        h.attrs["NumFilesPerSnapshot"]   = 1
        h.attrs["Omega0"]                = 0.0
        h.attrs["OmegaLambda"]           = 0.0
        h.attrs["HubbleParam"]           = 1.0
        h.attrs["Flag_Sfr"]              = 0
        h.attrs["Flag_Feedback"]         = 0
        h.attrs["Flag_Cooling"]          = 0
        h.attrs["Flag_StellarAge"]       = 0
        h.attrs["Flag_Metals"]           = 0
        h.attrs["Flag_DoublePrecision"]  = 1

        g = f.create_group("PartType0")
        g.create_dataset("Coordinates",     data=pos)
        g.create_dataset("Velocities",      data=vel)
        g.create_dataset("Masses",          data=masses)
        g.create_dataset("InternalEnergy",  data=u)
        g.create_dataset("ParticleIDs",     data=ids)
        g.create_dataset("CompositionType", data=comp)
        g.attrs["Y_helium"]   = float(Y_HE)
        g.attrs["RhoBackground"] = float(RHO_BG)
        g.attrs["T_background"]  = float(T_BG)

    print(f"Wrote {out}: N={N_PARTICLES}  rho={RHO_BG:.3e} g/cm^3  "
          f"T={T_BG:.3e} K  v_ampl={V_AMPL:.3e} cm/s  Y={Y_HE}")


if __name__ == "__main__":
    main()
