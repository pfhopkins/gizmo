"""Write a minimal GADGET-2 *binary* (ICFormat 1) IC containing only the blocks that a
legacy IC actually carries: header, Coordinates, Velocities, ParticleIDs, Masses,
InternalEnergy. Deliberately NO Temperature block.

This reproduces the shape of real legacy ICs -- e.g. the MUSIC zoom ICs used for the FIRE
suite (`ic_agora_m12i_ref13_rad4-chull.ics`), which are gadget2 binary and carry no
Temperature. With ICFormat 1 there are no block labels, so a reader cannot probe for a
missing block: it just freads sequentially and runs off the end of the file.
"""

import struct
import numpy as np

HEADER_SIZE = 256


def _block(fh, payload_bytes):
    """Write one Fortran-style record: 4-byte size, payload, 4-byte size."""
    fh.write(struct.pack("i", len(payload_bytes)))
    fh.write(payload_bytes)
    fh.write(struct.pack("i", len(payload_bytes)))


def make_binary_ic(outfile, ngas=512, boxsize=1.0, u0=0.05, seed=42):
    rng = np.random.default_rng(seed)

    # a uniform-random gas cube filling the box, at rest
    pos = rng.uniform(0.0, boxsize, size=(ngas, 3)).astype(np.float32)
    vel = np.zeros((ngas, 3), dtype=np.float32)
    ids = np.arange(1, ngas + 1, dtype=np.uint32)
    mass = np.full(ngas, 1.0 / ngas, dtype=np.float32)
    u = np.full(ngas, u0, dtype=np.float32)

    npart = [ngas, 0, 0, 0, 0, 0]
    # MassTable all zero -> the Masses block is present and is read per-particle
    masstab = [0.0] * 6

    h = b""
    h += struct.pack("6i", *npart)                 # NumPart_ThisFile
    h += struct.pack("6d", *masstab)               # MassTable
    h += struct.pack("d", 0.0)                     # Time
    h += struct.pack("d", 0.0)                     # Redshift
    h += struct.pack("i", 0)                       # flag_sfr
    h += struct.pack("i", 0)                       # flag_feedback
    h += struct.pack("6i", *npart)                 # NumPart_Total
    h += struct.pack("i", 0)                       # flag_cooling
    h += struct.pack("i", 1)                       # NumFilesPerSnapshot
    h += struct.pack("d", boxsize)                 # BoxSize
    h += struct.pack("d", 0.0)                     # Omega0
    h += struct.pack("d", 0.0)                     # OmegaLambda
    h += struct.pack("d", 1.0)                     # HubbleParam
    h += struct.pack("i", 0)                       # flag_stellarage
    h += struct.pack("i", 0)                       # flag_metals
    h += struct.pack("6I", *([0] * 6))             # NumPart_Total_HighWord
    h += struct.pack("i", 0)                       # flag_entropy_instead_u
    h += b"\0" * (HEADER_SIZE - len(h))            # pad to exactly 256 bytes
    assert len(h) == HEADER_SIZE, len(h)

    with open(outfile, "wb") as fh:
        _block(fh, h)
        _block(fh, pos.tobytes())
        _block(fh, vel.tobytes())
        _block(fh, ids.tobytes())
        _block(fh, mass.tobytes())
        _block(fh, u.tobytes())
        # NOTE: no Temperature block -- that is the whole point of this test

    return outfile


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "read_ic_binary_ics"
    make_binary_ic(out)
    print(f"wrote {out}")
