"""Generate the Biermann battery growth-rate IC.

2D periodic box of gas with orthogonal sinusoidal density and internal-energy
perturbations:
    rho(x,y) = rho_0 * (1 + a * sin(2 pi x / L))      => grad n_e || x-hat
    u(x,y)   = u_0   * (1 + b * sin(2 pi y / L))      => grad T_e || y-hat
B = 0 initially. The Biermann EMF
    E'_Bier = -(k_B/e)[grad T_e + (T_e/n_e) grad n_e]
has a nonzero curl pointing along z-hat, producing
    dB_z/dt|_Bier = -(c k_B / (e n_e)) (grad n_e) x (grad T_e),
which is what the test compares to.

Code units: standard GIZMO kpc/1e10 Msun/km-s. We pick rho_0 such that n_H ~ 1
cm^-3 in the unperturbed state and u_0 such that T ~ 10^4 K, both with small
amplitudes a=b=1e-2 so the test is solidly in the linear regime.
"""

import numpy as np
import h5py

# ---- physical constants & GIZMO unit conversions ---------------------------
PROTONMASS_CGS = 1.67262178e-24
BOLTZMANN_CGS = 1.38065e-16
HYDROGEN_MASSFRAC = 0.76
GAMMA = 5.0 / 3.0

# Custom unit system; see biermann_growth.params for rationale.
UnitLength_cm = 1.0
UnitMass_g = PROTONMASS_CGS
UnitVel_cms = 1.18e7
UnitDensity_cgs = UnitMass_g / UnitLength_cm**3
UnitEnergyPerMass_cgs = UnitVel_cms**2

# ---- target physical state at the unperturbed mean -------------------------
nH_target_cgs = 1.0               # cm^-3 (fully ionized hydrogen)
T_target_K = 1.0e6

rho_cgs = nH_target_cgs * PROTONMASS_CGS / HYDROGEN_MASSFRAC
rho0_code = rho_cgs / UnitDensity_cgs
u_cgs = T_target_K * BOLTZMANN_CGS / ((GAMMA - 1.0) * PROTONMASS_CGS)
u0_code = u_cgs / UnitEnergyPerMass_cgs

# ---- IC layout -------------------------------------------------------------
def make_IC(N_1D=64, Lbox=1.0, amp_rho=1e-4, amp_u=1e-4, seed=0,
            fname="biermann_growth_ics.hdf5"):
    """Lay down a 2D regular lattice and write the GIZMO HDF5 IC file."""

    DIMS = 2
    x0 = np.arange(0.0, Lbox, Lbox / N_1D) + 0.5 * Lbox / N_1D  # cell centers in [0,L)
    xv, yv = np.meshgrid(x0, x0, indexing="xy")
    xv = xv.flatten(); yv = yv.flatten()
    zv = 0.0 * xv
    Ngas = xv.size

    k = 2.0 * np.pi / Lbox
    rho = rho0_code * (1.0 + amp_rho * np.sin(k * xv))
    u   = u0_code   * (1.0 + amp_u   * np.sin(k * yv))

    # Equal-mass particles -> mass = rho_0 * V_cell (uniform), and density
    # gradient is established at the kernel-density-estimation step from the
    # particle layout. To keep particle masses uniform AND get rho ~ rho_0(1+a sin),
    # we equivalently re-space particles slightly along x. For the *initial*
    # gradient we instead vary the per-particle mass on a uniform lattice — this
    # is the simplest IC and is sufficient for an early-time growth-rate check.
    cell_volume = (Lbox / N_1D) ** DIMS  # 2D cell area; box "thickness" cancels
    mass = rho * cell_volume

    vx = 0.0 * xv; vy = 0.0 * xv; vz = 0.0 * xv
    bx = 0.0 * xv; by = 0.0 * xv; bz = 0.0 * xv
    ids = np.arange(1, Ngas + 1, dtype=np.uint64)

    with h5py.File(fname, "w") as F:
        h = F.create_group("Header")
        h.attrs["NumPart_ThisFile"] = np.array([Ngas, 0, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumPart_Total"]    = np.array([Ngas, 0, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = Lbox
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 0.0
        h.attrs["OmegaLambda"] = 0.0
        h.attrs["HubbleParam"] = 1.0
        h.attrs["Flag_Sfr"] = 0
        h.attrs["Flag_Cooling"] = 0
        h.attrs["Flag_StellarAge"] = 0
        h.attrs["Flag_Metals"] = 0
        h.attrs["Flag_Feedback"] = 0
        h.attrs["Flag_DoublePrecision"] = 1

        p = F.create_group("PartType0")
        coords = np.column_stack([xv, yv, zv]).astype(np.float64)
        vels   = np.column_stack([vx, vy, vz]).astype(np.float64)
        bvec   = np.column_stack([bx, by, bz]).astype(np.float64)
        p.create_dataset("Coordinates", data=coords)
        p.create_dataset("Velocities", data=vels)
        p.create_dataset("ParticleIDs", data=ids)
        p.create_dataset("Masses", data=mass.astype(np.float64))
        p.create_dataset("InternalEnergy", data=u.astype(np.float64))
        p.create_dataset("MagneticField", data=bvec)

    return fname, dict(
        Lbox=Lbox, N_1D=N_1D, amp_rho=amp_rho, amp_u=amp_u,
        rho0_code=rho0_code, u0_code=u0_code,
        nH0_cgs=nH_target_cgs, T0_K=T_target_K, k=k,
    )


# ---- analytic prediction for the early-time Biermann growth ---------------
def analytic_dBz_dt_cgs_at_origin(amp_rho, amp_u, T0_K, k_phys_cgs):
    """At (x=0, y=0): grad n_e = a * n_e0 * k * x-hat,
                     grad T_e = b * T_e0 * k * y-hat.
    dB_z/dt = -(c k_B / (e n_e0)) * (a*n_e0*k) * (b*T_e0*k)
            = -(c k_B / e) * a * b * T_e0 * k^2  [Gauss / s]
    Note this is independent of n_e0 (n_e cancels)."""
    C_LIGHT = 2.99792458e10
    BOLTZMANN = 1.38065e-16
    ELECTRON_CHARGE_ESU = 4.80320425e-10
    return -(C_LIGHT * BOLTZMANN / ELECTRON_CHARGE_ESU) * amp_rho * amp_u * T0_K * k_phys_cgs**2


if __name__ == "__main__":
    fname, info = make_IC()
    print(f"Wrote {fname} with {info['N_1D']}^2 gas particles.")
    print(f"  Lbox = {info['Lbox']} (code) = {info['Lbox']*UnitLength_cm:.3e} cm")
    print(f"  mean n_H = {info['nH0_cgs']} cm^-3, mean T = {info['T0_K']} K")
    k_phys = info['k'] / UnitLength_cm
    dBzdt = analytic_dBz_dt_cgs_at_origin(info['amp_rho'], info['amp_u'],
                                          info['T0_K'], k_phys)
    print(f"  Analytic dB_z/dt at (0,0) = {dBzdt:.3e} Gauss/s")
