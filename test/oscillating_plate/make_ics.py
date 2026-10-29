"""IC for test/oscillating_plate: the clamped oscillating plate of Gray, Monaghan & Swift (2001), as set up by Zhang et
al. (2023, sec. 6.1). A 2D plate (plane strain) of length 0.2 and thickness H = 0.02, density 1000, bulk modulus
3.25e6 and shear modulus 7.15e5 (Young's modulus 2.0e6, Poisson ratio 0.3975), H/dp = 10 cells across on a square
lattice, is clamped over a further length 0.02 at x < 0 and given the transverse velocity of the first bending mode,
v_y(x) = 0.01 c_0 f(x)/f(L) (c_0 = sqrt(K/rho)). The clamp cells carry ParticleID 0, which BOX_BND_PARTICLES holds
fixed. In 2D the Tillotson A is the 2D bulk modulus K + mu/3, so that the in-plane response matches plane strain.
Writes oscillating_plate_ics.hdf5 in the working directory."""
import numpy as np
import h5py

L, H, RHO_0, K, MU, V_FRAC, N_Y, L_CLAMP = 0.2, 0.02, 1000.0, 3.25e6, 7.15e5, 0.01, 10, 0.02
KL = 1.875                                        # first root of cos(kL) cosh(kL) = -1
X0, Y0 = 0.1, 0.2                                 # offset into positive coordinates


def mode_shape(x):
    k = KL / L
    return (np.cos(KL) + np.cosh(KL)) * (np.cosh(k * x) - np.cos(k * x)) + (np.sin(KL) - np.sinh(KL)) * (np.sinh(k * x) - np.sin(k * x))


def main(path="oscillating_plate_ics.hdf5"):
    dp = H / N_Y
    n_clamp, n_plate = int(round(L_CLAMP / dp)), int(round(L / dp))
    v_tip = V_FRAC * np.sqrt(K / RHO_0)
    ix, iy = np.meshgrid(np.arange(-n_clamp, n_plate), np.arange(N_Y), indexing="ij")
    x, y = ((ix + 0.5) * dp).ravel(), ((iy + 0.5) * dp).ravel()
    N = len(x)
    vy = np.where(x > 0, v_tip * mode_shape(x) / mode_shape(L), 0.0)
    ids = np.arange(1, N + 1, dtype=np.uint32)
    ids[x < 0] = 0
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        for key, val in [("NumPart_ThisFile", np.array([N, 0, 0, 0, 0, 0], np.int32)), ("NumPart_Total", np.array([N, 0, 0, 0, 0, 0], np.uint32)),
                         ("NumPart_Total_HighWord", np.zeros(6, np.uint32)), ("MassTable", np.zeros(6)), ("Time", 0.0), ("Redshift", 0.0),
                         ("BoxSize", 1.0), ("NumFilesPerSnapshot", 1), ("Omega0", 0.0), ("OmegaLambda", 0.0), ("HubbleParam", 1.0),
                         ("Flag_DoublePrecision", 1)]:
            h.attrs[key] = val
        for key in ["Flag_Sfr", "Flag_Feedback", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals"]:
            h.attrs[key] = 0
        g = f.create_group("PartType0")
        g["Coordinates"] = np.c_[X0 + x, Y0 + y, np.zeros(N)]
        g["Velocities"] = np.c_[np.zeros(N), vy, np.zeros(N)]
        g["Masses"] = np.full(N, RHO_0 * dp * dp)
        g["InternalEnergy"] = np.ones(N)
        g["ParticleIDs"] = ids
        g["CompositionType"] = np.zeros(N, np.int32)

if __name__ == "__main__":
    main()
