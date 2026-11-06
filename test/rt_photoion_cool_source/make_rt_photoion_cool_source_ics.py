"""16^3 uniform gas (n_H ~ 100 cm^-3, 20 pc box, units pc / Msun / km/s) plus one 30 Msun source at the centre."""
import h5py
import numpy as np


def make_ics(output_file="rt_photoion_cool_source_ics.hdf5", n=16, box=20.0):
    x = (np.arange(n) + 0.5) * box / n
    pos_gas = np.array(np.meshgrid(x, x, x, indexing="ij")).reshape(3, -1).T
    ngas = len(pos_gas)
    m_gas = 28244.6 / ngas
    with h5py.File(output_file, "w") as f:
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = [ngas, 0, 0, 0, 1, 0]
        h.attrs["NumPart_Total"] = [ngas, 0, 0, 0, 1, 0]
        h.attrs["NumPart_Total_HighWord"] = [0] * 6
        h.attrs["MassTable"] = [0.0] * 6
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = box
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_DoublePrecision"] = 0
        g = f.create_group("PartType0")
        g["Coordinates"] = pos_gas
        g["Velocities"] = np.zeros((ngas, 3))
        g["Masses"] = np.full(ngas, m_gas)
        g["ParticleIDs"] = np.arange(1, ngas + 1, dtype=np.uint32)
        g["InternalEnergy"] = np.full(ngas, 0.0)
        s = f.create_group("PartType4")
        s["Coordinates"] = np.array([[box / 2] * 3])
        s["Velocities"] = np.zeros((1, 3))
        s["Masses"] = np.array([30.0])
        s["ParticleIDs"] = np.array([ngas + 1], dtype=np.uint32)


if __name__ == "__main__":
    make_ics()
