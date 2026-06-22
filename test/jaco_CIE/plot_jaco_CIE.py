"""Plot jaco CIE test results against CHIANTI reference data."""
import numpy as np
import matplotlib.pyplot as plt
import sys

dat = np.loadtxt("test/jaco_CIE_results.dat")
T = dat[:, 0]
xHp = dat[:, 1]
xHep = dat[:, 2]
xHepp = dat[:, 3]
conv = dat[:, 4].astype(int)

y = 0.0994
xH = np.clip(1.0 - xHp, 1e-6, None)
xHe = np.clip(y - xHep - xHepp, 1e-6, None)
xe = xHp + xHep + 2*xHepp

fig, ax = plt.subplots(figsize=(8, 6))
ax.loglog(T, np.clip(xH, 1e-6, None), 'b-', lw=2, label='H')
ax.loglog(T, np.clip(xHp, 1e-6, None), color='orange', lw=2, label='H+')
ax.loglog(T, np.clip(xHe, 1e-6, None), 'g-', lw=2, label='He')
ax.loglog(T, np.clip(xHep, 1e-6, None), 'r-', lw=2, label='He+')
ax.loglog(T, np.clip(xHepp, 1e-6, None), 'm-', lw=2, label='He++')
ax.loglog(T, np.clip(xe, 1e-6, None), color='brown', lw=2, label='e-')

# Mark non-converged points
nc = conv == 0
if nc.any():
    ax.plot(T[nc], xHp[nc], 'kx', markersize=8, label='not converged')

# CHIANTI reference if available
try:
    chianti = np.load("/Users/mgrudic/code/jaco/tests/chianti_He_abundances.npy")
    cT = chianti[:, 0]
    mask = (cT >= T.min()) & (cT <= T.max())
    ax.loglog(cT[mask], np.clip(chianti[mask, 1]*y, 1e-6, None), 'g--', alpha=0.4, lw=1)
    ax.loglog(cT[mask], np.clip(chianti[mask, 2]*y, 1e-6, None), 'r--', alpha=0.4, lw=1)
    ax.loglog(cT[mask], np.clip(chianti[mask, 3]*y, 1e-6, None), 'm--', alpha=0.4, lw=1, label='CHIANTI (dashed)')
except Exception:
    pass

ax.set_xlabel('T (K)')
ax.set_ylabel(r'$x_i$')
ax.set_ylim(1e-4, 2)
ax.legend(fontsize=9, ncol=2)
ax.set_title('GIZMO jaco CIE: ion fractions vs Temperature')
fig.tight_layout()
fig.savefig('test/jaco_CIE.png', dpi=150)
plt.close(fig)
print("Saved test/jaco_CIE.png")
