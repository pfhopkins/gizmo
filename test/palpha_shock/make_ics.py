"""IC for test/palpha_shock: 1600 cells of porous basalt (matrix density 2.7 g/cc, distention 1.275) in a 1D periodic box
of length 200 cm; the left half moves at +3 km/s and the right half at -3 km/s, so they collide at x = 100 (and separate
at the periodic boundary, far enough away that the tension there does not reach the measured region). Writes palpha_shock_ics.hdf5 in the working directory."""
import numpy as np
import h5py

N, L, RHO_MATRIX, ALPHA_0, U_P, U_INIT = 1600, 200.0, 2.7, 1.275, 3.0e4, 1.0e6

def main(path="palpha_shock_ics.hdf5"):
    rho = RHO_MATRIX / ALPHA_0
    dx = L / N
    x = (np.arange(N) + 0.5) * dx
    v = np.where(x < 0.5 * L, U_P, -U_P)
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        for k, val in [("NumPart_ThisFile", np.array([N, 0, 0, 0, 0, 0], np.int32)), ("NumPart_Total", np.array([N, 0, 0, 0, 0, 0], np.uint32)),
                       ("NumPart_Total_HighWord", np.zeros(6, np.uint32)), ("MassTable", np.zeros(6)), ("Time", 0.0), ("Redshift", 0.0),
                       ("BoxSize", L), ("NumFilesPerSnapshot", 1), ("Omega0", 0.0), ("OmegaLambda", 0.0), ("HubbleParam", 1.0),
                       ("Flag_DoublePrecision", 1)]:
            h.attrs[k] = val
        for k in ["Flag_Sfr", "Flag_Feedback", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals"]:
            h.attrs[k] = 0
        g = f.create_group("PartType0")
        g["Coordinates"] = np.c_[x, np.zeros(N), np.zeros(N)]
        g["Velocities"] = np.c_[v, np.zeros(N), np.zeros(N)]
        g["Masses"] = np.full(N, rho * dx)
        g["InternalEnergy"] = np.full(N, U_INIT)
        g["ParticleIDs"] = np.arange(1, N + 1, dtype=np.uint32)
        g["CompositionType"] = np.zeros(N, np.int32)

if __name__ == "__main__":
    main()
