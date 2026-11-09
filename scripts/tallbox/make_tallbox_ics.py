#!/usr/bin/env python3
"""ICs for a stratified, shearing local patch of a galactic disk (solar-neighbourhood defaults, cf. TIGRESS R8).

Code units: pc, Msun, km/s. Box L x L x (BOX_LONG_Z * L), mid-plane at z = BOX_LONG_Z * L / 2.
Gas: isothermal hydrostatic slab in the external stellar + DM potential (GRAVITY_ANALYTIC_STRATIFIED_DISK form) plus
its own self-gravity, supported by sigma_eff^2 = c_s^2 + sigma_turb^2, placed in layers of near-isotropic spacing;
a hot, tenuous floor fills the rest of the box. Velocities: background shear v_y = -q Omega (x - L/2) plus a
solenoidal Gaussian random field with E(k) ~ k^4 exp(-2 k^2 / k_p^2), peaked at wavelength --lambda_peak.

    python make_tallbox_ics.py --mgas 1e3 -o tallbox_m1e3.hdf5
"""

import argparse
import numpy as np
import h5py

G = 4.30092e-3           # pc (km/s)^2 / Msun
KB_OVER_MP = 0.008254    # k_B / m_p in (km/s)^2 / K
RHO_CODE_TO_NH = 1.989e33 / (3.0857e18) ** 3 * 0.70 / 1.6726e-24  # code density -> n_H [cm^-3] at X = 0.70


def vertical_profile(z, sigma_gas, sigma_star, z_star, rho_dm, R0, sigma_eff, n_iter=60):
    """isothermal HSE density rho(z) (z >= 0 grid from 0), normalised to surface density sigma_gas, with gas self-gravity"""
    dz = z[1] - z[0]
    phi_ext = 2 * np.pi * G * sigma_star * (np.sqrt(z**2 + z_star**2) - z_star) + 2 * np.pi * G * rho_dm * R0**2 * np.log1p(z**2 / R0**2)
    rho = np.exp(-phi_ext / sigma_eff**2)
    for _ in range(n_iter):
        rho *= sigma_gas / (2 * np.trapezoid(rho, z))
        g_self = 2 * np.pi * G * 2 * np.concatenate([[0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * dz)])  # 2 pi G Sigma(<|z|)
        phi_self = np.concatenate([[0], np.cumsum(0.5 * (g_self[1:] + g_self[:-1]) * dz)])
        rho_new = np.exp(-(phi_ext + phi_self) / sigma_eff**2)
        rho_new *= sigma_gas / (2 * np.trapezoid(rho_new, z))
        rho = 0.5 * rho + 0.5 * rho_new
    return rho


def place_layers(L, zmid, zgrid, rho, mgas, rng):
    """layers of near-isotropic spacing (m/rho)^(1/3); returns positions [N,3] for the half-slab z >= zmid, mirrored"""
    cum = np.concatenate([[0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * np.diff(zgrid))]) * L * L  # mass above mid-plane below z
    pos, m_done = [], 0.0
    while True:
        z_lo = np.interp(m_done, cum, zgrid)
        rho_here = np.interp(z_lo, zgrid, rho)
        n_side = max(1, int(round(L / (mgas / rho_here) ** (1 / 3))))
        n_layer = n_side * n_side
        if m_done + n_layer * mgas > cum[-1]:
            break
        z_c = np.interp(m_done + 0.5 * n_layer * mgas, cum, zgrid)
        g = (np.arange(n_side) + 0.5) / n_side * L
        x, y = [a.ravel() for a in np.meshgrid(g, g, indexing="ij")]
        off = rng.random(2) * L / n_side  # random per-layer offset breaks vertical alignment
        jit = (rng.random((n_layer, 2)) - 0.5) * 0.2 * L / n_side
        pos.append(np.column_stack([(x + off[0] + jit[:, 0]) % L, (y + off[1] + jit[:, 1]) % L, np.full(n_layer, z_c)]))
        m_done += n_layer * mgas
    p = np.vstack(pos)
    lower = p.copy(); lower[:, 2] = -lower[:, 2]
    lower[:, :2] = (lower[:, :2] + 0.5 * L / np.sqrt(len(p) / max(1, len(pos)))) % L  # offset mirror half
    out = np.vstack([p, lower]); out[:, 2] += zmid
    return out


def turbulent_field(L, Lz, n, lambda_peak, rng):
    """solenoidal Gaussian random velocity field on an n x n x nz periodic grid, unit 1D rms"""
    nz = int(round(n * Lz / L))
    kx = 2 * np.pi * np.fft.fftfreq(n, L / n)
    kz = 2 * np.pi * np.fft.fftfreq(nz, Lz / nz)
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing="ij")
    k2 = KX**2 + KY**2 + KZ**2; k2[0, 0, 0] = 1
    kp = 2 * np.pi / lambda_peak
    amp = np.sqrt(k2 * np.exp(-2 * k2 / kp**2))  # P_3D ~ k^2 exp(-2k^2/kp^2) -> E(k) ~ k^4 exp(-2k^2/kp^2)
    v = [np.fft.fftn(rng.standard_normal(KX.shape)) * amp for _ in range(3)]
    kdotv = (KX * v[0] + KY * v[1] + KZ * v[2]) / k2
    v = [vi - Ki * kdotv for vi, Ki in zip(v, (KX, KY, KZ))]
    v = np.array([np.real(np.fft.ifftn(vi)) for vi in v])
    v -= v.mean(axis=(1, 2, 3), keepdims=True)
    return v / np.sqrt((v**2).sum(axis=0).mean() / 3), nz


def interp_periodic(field, pos, L, Lz, zlo):
    """trilinear interpolation of a periodic grid field [3,n,n,nz] at positions"""
    n, nz = field.shape[1], field.shape[3]
    u = np.column_stack([pos[:, 0] / L * n, pos[:, 1] / L * n, (pos[:, 2] - zlo) / Lz * nz]) - 0.5
    i0 = np.floor(u).astype(int); f = u - i0
    out = np.zeros((len(pos), 3))
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                w = (f[:, 0] if a else 1 - f[:, 0]) * (f[:, 1] if b else 1 - f[:, 1]) * (f[:, 2] if c else 1 - f[:, 2])
                ii, jj, kk = (i0[:, 0] + a) % n, (i0[:, 1] + b) % n, (i0[:, 2] + c) % nz
                out += w[:, None] * field[:, ii, jj, kk].T
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mgas", type=float, default=1e3, help="gas cell mass [Msun]")
    ap.add_argument("--L", type=float, default=1000.0, help="box side [pc]")
    ap.add_argument("--long_z", type=float, default=8, help="BOX_LONG_Z")
    ap.add_argument("--sigma_gas", type=float, default=10.0, help="gas surface density [Msun/pc^2]")
    ap.add_argument("--sigma_star", type=float, default=42.0, help="StratDisk_SigmaStar [Msun/pc^2]")
    ap.add_argument("--z_star", type=float, default=245.0, help="StratDisk_ZStar [pc]")
    ap.add_argument("--rho_dm", type=float, default=0.0064, help="StratDisk_RhoDM [Msun/pc^3]")
    ap.add_argument("--R0", type=float, default=8000.0, help="StratDisk_R0 [pc]")
    ap.add_argument("--omega", type=float, default=0.028, help="BOX_SHEARING_OMEGA_BOX_CENTER [km/s/pc]")
    ap.add_argument("--q", type=float, default=1.0, help="BOX_SHEARING_Q")
    ap.add_argument("--T", type=float, default=1e4, help="initial disk temperature [K]")
    ap.add_argument("--sigma_turb", type=float, default=7.0, help="initial 1D turbulent dispersion (mass-weighted) [km/s]")
    ap.add_argument("--lambda_peak", type=float, default=100.0, help="turbulent power peak wavelength [pc]")
    ap.add_argument("--n_floor", type=float, default=1e-4, help="hot floor density n_H [cm^-3]")
    ap.add_argument("--T_floor", type=float, default=1e6, help="hot floor temperature [K]")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-o", "--output", default="tallbox_ics.hdf5")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    L, Lz = a.L, a.L * a.long_z
    zmid, mgas = 0.5 * Lz, a.mgas
    mu_warm, mu_hot = 1.27, 0.59
    cs2 = KB_OVER_MP * a.T / mu_warm
    sigma_eff = np.sqrt(cs2 + a.sigma_turb**2)
    zgrid = np.linspace(0, zmid, 200001)
    rho = vertical_profile(zgrid, a.sigma_gas, a.sigma_star, a.z_star, a.rho_dm, a.R0, sigma_eff)
    rho_floor = a.n_floor / RHO_CODE_TO_NH
    disk = rho > rho_floor
    zgrid_d, rho_d = zgrid[disk], rho[disk]
    pos_disk = place_layers(L, zmid, zgrid_d, rho_d, mgas, rng)
    # hot floor: uniform random over the box outside the disk region
    z_edge = zgrid_d[-1]
    n_floor = int(round(rho_floor * L * L * (Lz - 2 * z_edge) / mgas))
    pos_floor = np.column_stack([rng.random(n_floor) * L, rng.random(n_floor) * L, rng.random(n_floor) * (Lz - 2 * z_edge)])
    pos_floor[:, 2] = np.where(pos_floor[:, 2] < zmid - z_edge, pos_floor[:, 2], pos_floor[:, 2] + 2 * z_edge)
    pos = np.vstack([pos_disk, pos_floor])
    is_disk = np.arange(len(pos)) < len(pos_disk)

    zlo = zmid - 1.5 * max(0.5 * L, 4 * np.sqrt(np.trapezoid(rho * zgrid**2, zgrid) / np.trapezoid(rho, zgrid)))
    vt, nz = turbulent_field(L, 2 * (zmid - zlo), 64, a.lambda_peak, rng)
    vel = np.zeros_like(pos)
    vel[is_disk] = a.sigma_turb * interp_periodic(vt, pos[is_disk], L, 2 * (zmid - zlo), zlo)
    vel[is_disk] -= vel[is_disk].mean(axis=0)
    vel[is_disk] *= a.sigma_turb / np.sqrt((vel[is_disk] ** 2).sum(axis=1).mean() / 3)
    vel[:, 1] += -a.q * a.omega * (pos[:, 0] - 0.5 * L)
    u = np.where(is_disk, 1.5 * KB_OVER_MP * a.T / mu_warm, 1.5 * KB_OVER_MP * a.T_floor / mu_hot)

    H = np.sqrt(np.trapezoid(rho * zgrid**2, zgrid) / np.trapezoid(rho, zgrid))
    print(f"N_gas = {len(pos)} ({len(pos_disk)} disk, {n_floor} floor); mid-plane n_H = {rho[0] * RHO_CODE_TO_NH:.3g} cm^-3; "
          f"rms height = {H:.0f} pc; disk edge |z| = {z_edge:.0f} pc; sigma_eff = {sigma_eff:.2f} km/s; "
          f"Sigma_disk = {len(pos_disk) * a.mgas / L**2:.2f} Msun/pc^2")

    with h5py.File(a.output, "w") as f:
        h = f.create_group("Header")
        npart = np.array([len(pos), 0, 0, 0, 0, 0], dtype=np.int32)
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = L
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_DoublePrecision"] = 0
        p = f.create_group("PartType0")
        p["Coordinates"] = pos.astype(np.float32)
        p["Velocities"] = vel.astype(np.float32)
        p["Masses"] = np.full(len(pos), mgas, dtype=np.float32)
        p["InternalEnergy"] = u.astype(np.float32)
        p["ParticleIDs"] = np.arange(1, len(pos) + 1, dtype=np.uint32)


if __name__ == "__main__":
    main()
