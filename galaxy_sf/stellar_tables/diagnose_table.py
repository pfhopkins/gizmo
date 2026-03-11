#!/usr/bin/env python3
"""Diagnostic plots for stellar_tables_unified.hdf5"""
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

plt.rcParams.update({'font.size': 9, 'figure.dpi': 150})

hdf = 'stellar_tables_unified.hdf5'
outdir = 'diagnostics'
import os; os.makedirs(outdir, exist_ok=True)

with h5py.File(hdf, 'r') as f:
    Z      = f['Z'][:]
    M      = f['M_init'][:]
    log_age = f['log_age_yr'][:]
    elems  = [e.decode() for e in f['elements'][:]]

    lifetime  = f['lifetime_yr'][:]
    logL      = f['logL'][:]
    logTeff   = f['logTeff'][:]
    logR      = f['logR_cm'][:]
    log_Mdot  = f['log_Mdot'][:]
    v_wind    = f['v_wind'][:]
    M_current = f['M_current'][:]
    phase     = f['phase'][:]
    surface   = f['surface_abundances'][:]
    log_Q     = f['log_Q_ion'][:]
    log_Lbol  = f['log_L_bol'][:]

    rem_type  = f['remnant_type'][:]
    rem_mass  = f['remnant_mass'][:]
    M_preSN   = f['M_preSN'][:]
    M_CO      = f['M_CO_core'][:]
    M_He      = f['M_He_core'][:]

    yields    = f['net_yields'][:]
    yield_src = f['yield_source'][:]
    track_src = f['track_source'][:]

    # phase transitions
    t_MS_end  = f['phase_transitions/t_MS_end'][:]
    t_AGB_s   = f['phase_transitions/t_AGB_start'][:]
    t_AGB_e   = f['phase_transitions/t_AGB_end'][:]
    t_WR_s    = f['phase_transitions/t_WR_start'][:]

PHASE_NAMES = {0:'PMS', 1:'MS', 2:'HG', 3:'RGB', 4:'CHeB', 5:'AGB', 6:'post-AGB/WR'}
REM_NAMES   = {0:'WD', 1:'ECSN', 2:'NS', 3:'BH', 4:'PPISN', 5:'PISN'}
REM_COLORS  = {0:'grey', 1:'cyan', 2:'dodgerblue', 3:'black', 4:'orange', 5:'red'}

# Pick a few representative metallicities
iz_lo  = np.argmin(np.abs(Z - 0.0001))   # very low Z
iz_smc = np.argmin(np.abs(Z - 0.002))    # SMC-like
iz_sol = np.argmin(np.abs(Z - 0.014))    # solar
iz_hi  = np.argmin(np.abs(Z - 0.04))     # super-solar
iz_set = [(iz_lo, f'Z={Z[iz_lo]:.4f}'),
          (iz_smc, f'Z={Z[iz_smc]:.4f}'),
          (iz_sol, f'Z={Z[iz_sol]:.4f}'),
          (iz_hi, f'Z={Z[iz_hi]:.4f}')]

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Lifetimes
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
for iz, label in iz_set:
    mask = lifetime[iz] > 0
    ax.plot(M[mask], lifetime[iz, mask] / 1e6, label=label, lw=1.5)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('M_init [Msun]'); ax.set_ylabel('Lifetime [Myr]')
ax.set_title('Stellar Lifetimes')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 500); ax.set_ylim(1, 2e4)
fig.tight_layout(); fig.savefig(f'{outdir}/01_lifetimes.png'); plt.close()

# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Mass loss (total wind mass lost = M_init - M_preSN)
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for iz, label in iz_set:
    mask = M_preSN[iz] > 0
    dm = M[mask] - M_preSN[iz, mask]
    frac = dm / M[mask]
    ax1.plot(M[mask], dm, label=label, lw=1.5)
    ax2.plot(M[mask], frac, label=label, lw=1.5)
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('M_init'); ax1.set_ylabel('M_init - M_preSN [Msun]')
ax1.set_title('Total Wind Mass Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.set_xscale('log')
ax2.set_xlabel('M_init'); ax2.set_ylabel('Fraction of mass lost to winds')
ax2.set_title('Wind Mass Loss Fraction'); ax2.legend(); ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1)
fig.tight_layout(); fig.savefig(f'{outdir}/02_wind_mass_loss.png'); plt.close()

# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Peak Mdot in MS, AGB, post-AGB/WR phases
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ip, (phase_val, phase_name) in enumerate([(1, 'MS'), (5, 'AGB'), (6, 'post-AGB/WR')]):
    ax = axes[ip]
    for iz, label in iz_set:
        peak_mdot = np.full(len(M), -99.0)
        for im in range(len(M)):
            mask = phase[iz, im] == phase_val
            if np.any(mask):
                peak_mdot[im] = np.max(log_Mdot[iz, im, mask])
        valid = peak_mdot > -30
        if np.any(valid):
            ax.plot(M[valid], peak_mdot[valid], label=label, lw=1.5)
    ax.set_xscale('log')
    ax.set_xlabel('M_init'); ax.set_ylabel('Peak log(Mdot) [Msun/yr]')
    ax.set_title(f'Peak Mass Loss Rate — {phase_name}')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 500)
fig.tight_layout(); fig.savefig(f'{outdir}/03_peak_mdot_by_phase.png'); plt.close()

# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Wind velocity in different phases
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
iz = iz_sol
for phase_val, phase_name, ls in [(1, 'MS', '-'), (5, 'AGB', '--'), (6, 'WR/post-AGB', ':')]:
    peak_vw = np.full(len(M), 0.0)
    for im in range(len(M)):
        mask = (phase[iz, im] == phase_val) & (v_wind[iz, im] > 0)
        if np.any(mask):
            peak_vw[im] = np.max(v_wind[iz, im, mask])
    valid = peak_vw > 0
    if np.any(valid):
        ax.plot(M[valid], peak_vw[valid], label=phase_name, lw=1.5, ls=ls)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('M_init'); ax.set_ylabel('Peak v_wind [km/s]')
ax.set_title(f'Peak Wind Velocity by Phase (Z={Z[iz_sol]:.4f})')
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f'{outdir}/04_wind_velocity.png'); plt.close()

# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Surface abundances evolution for a few masses at solar Z
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
iz = iz_sol
mass_test = [3, 10, 20, 40, 80, 200]
for ip, Mt in enumerate(mass_test):
    ax = axes.flat[ip]
    im = np.argmin(np.abs(M - Mt))
    age = 10**log_age
    alive = M_current[iz, im] > 0.01
    for ie, elem in enumerate(['H', 'He', 'C', 'N', 'O']):
        idx = elems.index(elem)
        y = surface[iz, im, :, idx]
        valid = alive & (y > 1e-10)
        if np.any(valid):
            ax.plot(age[valid] / 1e6, y[valid], label=elem, lw=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Age [Myr]')
    ax.set_ylabel('Surface mass fraction')
    ax.set_title(f'M={M[im]:.0f} Msun')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-5, 1)
fig.suptitle(f'Surface Abundance Evolution (Z={Z[iz]:.4f})', fontsize=12)
fig.tight_layout(); fig.savefig(f'{outdir}/05_surface_abundances.png'); plt.close()

# ═══════════════════════════════════════════════════════════════
# FIGURE 6: Remnant types and masses
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Remnant type map
rem_img = rem_type[iz_sol]
for rt, name in REM_NAMES.items():
    mask = rem_type[iz_sol] == rt
    if np.any(mask):
        ax1.scatter(M[mask], np.full(np.sum(mask), Z[iz_sol]),
                   c=REM_COLORS[rt], label=f'{name} ({np.sum(mask)})', s=20, zorder=3)
# Now do all Z
for iz in range(len(Z)):
    for rt in REM_NAMES:
        mask = rem_type[iz] == rt
        if np.any(mask):
            ax1.scatter(M[mask], np.full(np.sum(mask), Z[iz]),
                       c=REM_COLORS[rt], s=8, alpha=0.5, zorder=2)
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('M_init [Msun]'); ax1.set_ylabel('Z')
ax1.set_title('Remnant Type Map (all Z)')
# Custom legend
from matplotlib.lines import Line2D
handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor=REM_COLORS[rt],
           label=f'{REM_NAMES[rt]}', markersize=8) for rt in sorted(REM_NAMES)]
ax1.legend(handles=handles, fontsize=8)
ax1.grid(True, alpha=0.3)

# Remnant mass vs M_init
for iz, label in iz_set:
    mask = rem_mass[iz] > 0
    ax2.plot(M[mask], rem_mass[iz, mask], label=label, lw=1.5)
ax2.plot(M, M, 'k--', alpha=0.3, label='M_rem = M_init')
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('M_init [Msun]'); ax2.set_ylabel('Remnant Mass [Msun]')
ax2.set_title('Remnant Mass vs Initial Mass')
ax2.legend(); ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.5, 500); ax2.set_ylim(0.3, 100)
fig.tight_layout(); fig.savefig(f'{outdir}/06_remnants.png'); plt.close()

# ═══════════════════════════════════════════════════════════════
# FIGURE 7: SN type census and explosion energy map
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Census per Z
for iz in range(len(Z)):
    for rt in [1, 2, 3, 4, 5]:
        n = np.sum(rem_type[iz] == rt)
        if n > 0:
            ax1.scatter(Z[iz], n, c=REM_COLORS[rt], s=40, zorder=3)
ax1.set_xscale('log')
ax1.set_xlabel('Z'); ax1.set_ylabel('Count')
ax1.set_title('SN Type Census per Z')
ax1.legend(handles=[Line2D([0],[0], marker='o', color='w', markerfacecolor=REM_COLORS[rt],
           label=REM_NAMES[rt], markersize=8) for rt in [1,2,3,4,5]], fontsize=8)
ax1.grid(True, alpha=0.3)

# Yield source map
ysrc_names = {1: 'Karakas', 2: 'Limongi25', 3: 'PARSEC_v2', 4: 'CL18'}
ysrc_colors = {0: 'white', 1: 'green', 2: 'blue', 3: 'red', 4: 'grey'}
for ys, name in ysrc_names.items():
    for iz in range(len(Z)):
        mask = yield_src[iz] == ys
        if np.any(mask):
            ax2.scatter(M[mask], np.full(np.sum(mask), Z[iz]),
                       c=ysrc_colors[ys], s=8, alpha=0.6)
handles2 = [Line2D([0],[0], marker='o', color='w', markerfacecolor=ysrc_colors[ys],
            label=name, markersize=8) for ys, name in ysrc_names.items()]
ax2.legend(handles=handles2, fontsize=8)
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('M_init [Msun]'); ax2.set_ylabel('Z')
ax2.set_title('Yield Source Map')
ax2.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f'{outdir}/07_sn_types_yield_sources.png'); plt.close()

# ═══════════════════════════════════════════════════════════════
# FIGURE 8: Net yields vs mass (O, Fe, C, Si)
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ip, elem in enumerate(['O', 'Fe', 'C', 'Si']):
    ax = axes.flat[ip]
    ie = elems.index(elem)
    for iz, label in iz_set:
        y = yields[iz, :, ie]
        # Split positive and negative
        pos = np.where(y > 0, y, np.nan)
        neg = np.where(y < 0, -y, np.nan)
        ax.plot(M, pos, label=f'{label} (+)', lw=1.5)
        ax.plot(M, neg, label=f'{label} (-)', lw=1, ls='--', alpha=0.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('M_init [Msun]')
    ax.set_ylabel(f'|Net Yield| [Msun]')
    ax.set_title(f'Net {elem} Yield')
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 500); ax.set_ylim(1e-6, 100)
fig.suptitle('Explosion Net Yields', fontsize=12)
fig.tight_layout(); fig.savefig(f'{outdir}/08_net_yields.png'); plt.close()

# ═══════════════════════════════════════════════════════════════
# FIGURE 9: Wind yields (cumulative mass loss × surface) for MS, AGB, WR
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
iz = iz_sol
age = 10**log_age
dt = np.zeros(len(log_age))
dt[1:] = age[1:] - age[:-1]

for ip, elem in enumerate(['H', 'He', 'C', 'N', 'O', 'Fe']):
    ax = axes.flat[ip]
    ie = elems.index(elem)

    for phase_val, pname, ls in [(1, 'MS', '-'), (5, 'AGB', '--'), (6, 'WR', ':')]:
        wind_yield = np.zeros(len(M))
        for im in range(len(M)):
            pmask = phase[iz, im] == phase_val
            mdot_lin = np.where(log_Mdot[iz, im] > -30,
                               10**log_Mdot[iz, im], 0.0)
            surf = surface[iz, im, :, ie]
            # wind yield = integral of |Mdot| * X_surface * dt
            wy = np.sum(mdot_lin[pmask] * surf[pmask] * dt[pmask])
            wind_yield[im] = wy
        valid = wind_yield > 1e-12
        if np.any(valid):
            ax.plot(M[valid], wind_yield[valid], label=pname, lw=1.5, ls=ls)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('M_init'); ax.set_ylabel(f'Wind {elem} [Msun]')
    ax.set_title(f'Wind {elem} Yield by Phase')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 500)

fig.suptitle(f'Cumulative Wind Yields by Phase (Z={Z[iz]:.4f})', fontsize=12)
fig.tight_layout(); fig.savefig(f'{outdir}/09_wind_yields_by_phase.png'); plt.close()

# ═══════════════════════════════════════════════════════════════
# FIGURE 10: HR diagram tracks at solar Z for a few masses
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 8))
iz = iz_sol
mass_show = [1, 3, 5, 10, 20, 40, 80, 150, 300]
cmap = plt.cm.plasma
for i, Mt in enumerate(mass_show):
    im = np.argmin(np.abs(M - Mt))
    alive = M_current[iz, im] > 0.01
    lt = logTeff[iz, im, alive]
    ll = logL[iz, im, alive]
    if len(lt) < 2:
        continue
    c = cmap(i / len(mass_show))
    ax.plot(lt, ll, color=c, lw=1.2, label=f'{M[im]:.0f} Msun')
    ax.scatter(lt[0], ll[0], color=c, s=30, zorder=5, marker='o')
    ax.scatter(lt[-1], ll[-1], color=c, s=30, zorder=5, marker='x')

ax.invert_xaxis()
ax.set_xlabel('log(Teff / K)'); ax.set_ylabel('log(L / L_sun)')
ax.set_title(f'HR Diagram (Z={Z[iz]:.4f})')
ax.legend(fontsize=7, loc='lower left'); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f'{outdir}/10_HR_diagram.png'); plt.close()

# ═══════════════════════════════════════════════════════════════
# FIGURE 11: Ionizing photon rate and bolometric luminosity
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
iz = iz_sol
for Mt in [5, 10, 20, 40, 80, 150]:
    im = np.argmin(np.abs(M - Mt))
    alive = M_current[iz, im] > 0.01
    a = 10**log_age[alive] / 1e6
    q = log_Q[iz, im, alive]
    lb = log_Lbol[iz, im, alive]
    valid_q = q > 30
    valid_l = lb > 30
    ax1.plot(a[valid_q], q[valid_q], label=f'{M[im]:.0f}', lw=1)
    ax2.plot(a[valid_l], lb[valid_l], label=f'{M[im]:.0f}', lw=1)

ax1.set_xscale('log'); ax1.set_xlabel('Age [Myr]')
ax1.set_ylabel('log Q_ion [photons/s]'); ax1.set_title('Ionizing Photon Rate')
ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)
ax2.set_xscale('log'); ax2.set_xlabel('Age [Myr]')
ax2.set_ylabel('log L_bol [erg/s]'); ax2.set_title('Bolometric Luminosity')
ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)
fig.suptitle(f'Radiation (Z={Z[iz]:.4f})', fontsize=12)
fig.tight_layout(); fig.savefig(f'{outdir}/11_radiation.png'); plt.close()

# ═══════════════════════════════════════════════════════════════
# FIGURE 12: Phase durations
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
iz = iz_sol
ms_dur = np.where(t_MS_end[iz] > 0, t_MS_end[iz], 0)
agb_dur = np.where((t_AGB_e[iz] > 0) & (t_AGB_s[iz] > 0), t_AGB_e[iz] - t_AGB_s[iz], 0)
wr_dur = np.where((t_WR_s[iz] > 0) & (lifetime[iz] > 0), lifetime[iz] - t_WR_s[iz], 0)

mask_ms = ms_dur > 0
mask_agb = agb_dur > 0
mask_wr = wr_dur > 0

if np.any(mask_ms):
    ax.plot(M[mask_ms], ms_dur[mask_ms] / 1e6, label='MS', lw=2)
if np.any(mask_agb):
    ax.plot(M[mask_agb], agb_dur[mask_agb] / 1e6, label='AGB', lw=2)
if np.any(mask_wr):
    ax.plot(M[mask_wr], wr_dur[mask_wr] / 1e6, label='WR/post-AGB', lw=2)
ax.plot(M[lifetime[iz] > 0], lifetime[iz, lifetime[iz] > 0] / 1e6, 'k--', alpha=0.4, label='Total')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('M_init [Msun]'); ax.set_ylabel('Duration [Myr]')
ax.set_title(f'Phase Durations (Z={Z[iz]:.4f})')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 500); ax.set_ylim(1e-3, 2e4)
fig.tight_layout(); fig.savefig(f'{outdir}/12_phase_durations.png'); plt.close()

# ═══════════════════════════════════════════════════════════════
# FIGURE 13: Core masses and pre-SN mass
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
iz = iz_sol
mask = M > 5
ax.plot(M[mask], M_preSN[iz, mask], 'k-', label='M_preSN', lw=2)
mask_he = M_He[iz] > 0
mask_co = M_CO[iz] > 0
if np.any(mask_he):
    ax.plot(M[mask_he], M_He[iz, mask_he], 'r-', label='M_He_core', lw=1.5)
if np.any(mask_co):
    ax.plot(M[mask_co], M_CO[iz, mask_co], 'b-', label='M_CO_core', lw=1.5)
ax.plot(M[mask], rem_mass[iz, mask], 'g--', label='M_remnant', lw=1.5)
ax.plot(M, M, 'k:', alpha=0.3, label='M_init')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('M_init [Msun]'); ax.set_ylabel('Mass [Msun]')
ax.set_title(f'Stellar Structure at Death (Z={Z[iz]:.4f})')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xlim(5, 500); ax.set_ylim(0.5, 500)
fig.tight_layout(); fig.savefig(f'{outdir}/13_core_masses.png'); plt.close()

print(f"Wrote 13 diagnostic figures to {outdir}/")
