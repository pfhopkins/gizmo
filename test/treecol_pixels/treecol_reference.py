"""Direct-sum references for the treecol_pixels tests: HEALPix RING pixel of a direction (Gorski+05, as
healpix vec2pix_ring), per-pixel columns and band fluxes summed over all particles."""
import numpy as np

C_LIGHT, PC = 2.9979e10, 3.085678e18  # constants.h


def ring_pixel(d, nside=1):
    d = np.atleast_2d(np.asarray(d, dtype=np.float64))
    r = np.sqrt((d ** 2).sum(1)); z = d[:, 2] / r
    tt = np.mod(np.arctan2(d[:, 1], d[:, 0]), 2 * np.pi) / (0.5 * np.pi)
    out = np.empty(len(d), dtype=int)
    for i in range(len(d)):
        za = abs(z[i])
        if za <= 2.0 / 3.0:
            t1, t2 = nside * (0.5 + tt[i]), nside * z[i] * 0.75
            jp, jm = int(t1 - t2), int(t1 + t2)
            ir = nside + 1 + jp - jm; kshift = 1 - (ir & 1)
            ip = ((jp + jm - nside + kshift + 1) // 2) % (4 * nside)
            out[i] = nside * (nside - 1) * 2 + (ir - 1) * 4 * nside + ip
        else:
            tp = tt[i] - int(tt[i]); tmp = nside * np.sqrt(3.0 * (1.0 - za))
            jp, jm = int(tp * tmp), int((1.0 - tp) * tmp); ir = jp + jm + 1
            ip = int(tt[i] * ir) % (4 * ir)
            out[i] = 2 * ir * (ir - 1) + ip if z[i] > 0 else 12 * nside * nside - 2 * ir * (ir + 1) + ip
    return out


def direct_columns(target, pos, m, npix=12, r_max=0.0):
    """per-pixel sum of m / (4 pi / npix r^2) over the sources (excluding the target itself)"""
    d = pos - target; r2 = (d ** 2).sum(1); ok = r2 > 0
    if r_max > 0:
        ok &= r2 < r_max ** 2
    col = np.zeros(npix)
    np.add.at(col, ring_pixel(d[ok], int(np.sqrt(npix / 12))), m[ok] / (4 * np.pi / npix * r2[ok]))
    return col


def direct_energy_density(target, pos, L_cgs, npix=12, r_floor=0.0):
    """per-pixel energy density [erg/cm^3] of point sources L_cgs [erg/s] at pos [pc]"""
    d = pos - target; r2 = np.maximum((d ** 2).sum(1), r_floor ** 2)
    u = np.zeros(npix)
    np.add.at(u, ring_pixel(d, int(np.sqrt(npix / 12))), L_cgs / (4 * np.pi * C_LIGHT * r2 * PC ** 2))
    return u
