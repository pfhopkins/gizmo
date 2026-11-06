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


def shell_of(r2, nshell, shield):
    """log-radial shell of r^2: nshell shells over 3 dex below the shielding length, clamped"""
    s = nshell - 1 + np.floor(0.5 * np.log10(r2 / shield ** 2) * nshell / 3.0).astype(int)
    return np.clip(s, 0, nshell - 1)


def direct_shell_sweep(target, gas_pos, m_dust, star_pos, L_cgs, sigma_nh, shield, nshell=12, npix=12, r_floor=0.0):
    """per-pixel energy density [erg/cm^3] of point sources L_cgs [erg/s] at star_pos [pc] behind the dust of the gas
    cells: per (pixel, shell) sources attenuated by the optical depth of the interior shells plus half their own;
    sigma_nh = optical depth per unit dust mass column m_dust / (4 pi / npix r^2)"""
    nside = int(np.sqrt(npix / 12))
    d = gas_pos - target; r2 = (d ** 2).sum(1); ok = (r2 > 0) & (r2 < shield ** 2)
    col = np.zeros((npix, nshell))
    np.add.at(col, (ring_pixel(d[ok], nside), shell_of(r2[ok], nshell, shield)), m_dust[ok] / (4 * np.pi / npix * r2[ok]))
    d = star_pos - target; r2 = np.maximum((d ** 2).sum(1), r_floor ** 2)
    src = np.zeros((npix, nshell))
    np.add.at(src, (ring_pixel(d, nside), shell_of(r2, nshell, shield)), L_cgs / (4 * np.pi * C_LIGHT * r2 * PC ** 2))
    dtau = sigma_nh * col
    tau = np.cumsum(dtau, axis=1) - 0.5 * dtau
    return (src * np.exp(-tau)).sum(1)
