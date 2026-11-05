"""ICs for the tallbox_gravity test: collisionless particles in a box 1 x 1 x 8 (type 1, G = 1).

slab  jittered-grid slab of three layers around z = 4 plus one layer hugging each z face (z = 0.03 and 7.97),
      random masses. Pair separations stay above 0.03, well outside the 0.001 softening, so every pair
      force is Newtonian and a plain direct sum is the reference.
sheet one exact 16 x 16 lattice sheet of total mass 1 (surface density 1) at z = 4, and six probe
      particles of negligible mass above and below it (a quarter cell off the lattice sites so that no
      separation sits on the half-box wrap), for the analytic sheet field.
"""
import h5py
import numpy as np

LXY, LZ, NG = 1.0, 8.0, 16
SPACING = LXY / NG
Z0 = 4.0
SHEET_PROBE_HEIGHTS = (0.2, 0.5, 1.0)
PROBE_MASS = 1e-9


def _write(path, pos, mass, vel=None):
    n = len(mass)
    with h5py.File(path, "w") as F:
        h = F.create_group("Header")
        h.attrs["NumPart_ThisFile"] = [0, n, 0, 0, 0, 0]
        h.attrs["NumPart_Total"] = [0, n, 0, 0, 0, 0]
        h.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
        h.attrs["MassTable"] = [0, 0, 0, 0, 0, 0]
        h.attrs["BoxSize"] = LXY
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["NumFilesPerSnapshot"] = 1
        g = F.create_group("PartType1")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=np.zeros((n, 3)) if vel is None else vel)
        g.create_dataset("Masses", data=mass)
        g.create_dataset("ParticleIDs", data=np.arange(1, n + 1, dtype=np.uint32))


def _layer(rng, z, jitter_z):
    ij = np.stack(np.meshgrid(np.arange(NG), np.arange(NG), indexing="ij"), -1).reshape(-1, 2)
    xy = (ij + 0.5 + rng.uniform(-0.2, 0.2, ij.shape)) * SPACING
    return np.column_stack([xy, z + rng.uniform(-jitter_z, jitter_z, len(xy))])


def make_slab_ics(path, seed=7):
    rng = np.random.default_rng(seed)
    layers = [_layer(rng, Z0 + (k - 1) * SPACING, 0.1 * SPACING) for k in range(3)]
    layers += [_layer(rng, 0.03, 0.005), _layer(rng, LZ - 0.03, 0.005)]
    pos = np.vstack(layers)
    _write(path, pos, rng.uniform(0.5, 1.5, len(pos)) / len(pos))


def make_sheet_ics(path):
    ij = np.stack(np.meshgrid(np.arange(NG), np.arange(NG), indexing="ij"), -1).reshape(-1, 2)
    sheet = np.column_stack([(ij + 0.5) * SPACING, np.full(len(ij), Z0)])
    probes = [[(NG // 2 + 0.25) * SPACING, (NG // 2 + 0.25) * SPACING, Z0 + s * h]
              for h in SHEET_PROBE_HEIGHTS for s in (1, -1)]
    pos = np.vstack([sheet, probes])
    mass = np.concatenate([np.full(len(sheet), LXY * LXY / len(sheet)), np.full(len(probes), PROBE_MASS)])
    _write(path, pos, mass)
