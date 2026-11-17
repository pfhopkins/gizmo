"""IC for test/elastic_tension_wave: 400 cells of compact Tillotson basalt (2.7 g/cc) in a 1D periodic box of length 200 cm,
with u = 1e6 erg/g (cold: the pressure swings into tension through the wave) and a standing longitudinal wave of
velocity amplitude 30 cm/s and wavelength L/8 (50 cells per wavelength). Writes elastic_tension_wave_ics.hdf5 in the
working directory."""
import numpy as np
import h5py

N, L, RHO_0, U_INIT, AMPLITUDE, N_WAVES = 400, 200.0, 2.7, 1.0e6, 30.0, 8

def main(path="elastic_tension_wave_ics.hdf5"):
    dx = L / N
    x = (np.arange(N) + 0.5) * dx
    k = 2 * np.pi * N_WAVES / L
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        for key, val in [("NumPart_ThisFile", np.array([N, 0, 0, 0, 0, 0], np.int32)), ("NumPart_Total", np.array([N, 0, 0, 0, 0, 0], np.uint32)),
                         ("NumPart_Total_HighWord", np.zeros(6, np.uint32)), ("MassTable", np.zeros(6)), ("Time", 0.0), ("Redshift", 0.0),
                         ("BoxSize", L), ("NumFilesPerSnapshot", 1), ("Omega0", 0.0), ("OmegaLambda", 0.0), ("HubbleParam", 1.0),
                         ("Flag_DoublePrecision", 1)]:
            h.attrs[key] = val
        for key in ["Flag_Sfr", "Flag_Feedback", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals"]:
            h.attrs[key] = 0
        g = f.create_group("PartType0")
        g["Coordinates"] = np.c_[x, np.zeros(N), np.zeros(N)]
        g["Velocities"] = np.c_[AMPLITUDE * np.sin(k * x), np.zeros(N), np.zeros(N)]
        g["Masses"] = np.full(N, RHO_0 * dx)
        g["InternalEnergy"] = np.full(N, U_INIT)
        g["ParticleIDs"] = np.arange(1, N + 1, dtype=np.uint32)
        g["CompositionType"] = np.zeros(N, np.int32)

if __name__ == "__main__":
    main()
