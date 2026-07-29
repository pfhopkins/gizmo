"""Fibonacci IC + pre-computed h* (Python finds h s.t. N_eff(h)=32) written
to HDF5 as KernelMaxRadius. Tests whether GIZMO's iteration trap on shell
ICs is solved by a better initial guess alone, with CT fix already in place.
"""
import os, sys, numpy as np, h5py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from initial_conditions.eos_tools.eos_dispatch import Material
from initial_conditions.eos_tools.hse_solver import Zone, solve_hse
from initial_conditions.eos_tools.shell_placement import fibonacci_shells
from initial_conditions.eos_tools.sph_density import correct_u_for_sph_density
from initial_conditions.eos_tools.build_layered_body import write_hdf5

ol = Material.tillotson("olivine"); fe = Material.tillotson("iron")
zones = [Zone(ol, inner_mass_frac=0.32, T_profile="adiabatic"),
         Zone(fe, inner_mass_frac=0.0,  T_profile="adiabatic")]
prof = solve_hse(M_total=5.972e27, T_surface=300.0, P_surface=1e6,
                 zones=zones, n_steps=1500, verbose=False)
placed = fibonacci_shells(prof, zones, n_particles=2000)
correct_u_for_sph_density(placed, prof, zones, des_num_ngb=32, verbose=True)

# Compute h* per particle: h s.t. N_eff(h) = 32 using exact kernel sum.
x = placed["coords"]
def f_cubic(q):
    out = np.zeros_like(q)
    m1=q<0.5; m2=(q>=0.5)&(q<1.0)
    out[m1] = 1.0 + 6.0*(q[m1]-1.0)*q[m1]**2
    out[m2] = 2.0*(1.0-q[m2])**3
    return out

def find_h_star(i, target=32.0):
    h_lo, h_hi = 1e6, 8e8
    d = np.linalg.norm(x - x[i], axis=1)
    for _ in range(60):
        h = 0.5*(h_lo+h_hi)
        N = (32.0/3.0) * f_cubic(d/h).sum()
        if N < target: h_lo = h
        else: h_hi = h
        if (h_hi-h_lo)/h_lo < 1e-6: break
    return 0.5*(h_lo+h_hi)

n = len(x)
h_star = np.array([find_h_star(i) for i in range(n)])
print(f"h_star: min={h_star.min():.3e} max={h_star.max():.3e} median={np.median(h_star):.3e}")

# Write to HDF5 with KernelMaxRadius field
out = os.path.join(os.path.dirname(__file__), "hse_earth_hguess_ics.hdf5")
write_hdf5(out, placed, prof, box_size=10*prof['R'])
# Add KernelMaxRadius
with h5py.File(out, "a") as f:
    f["PartType0"].create_dataset("KernelMaxRadius", data=h_star.astype(np.float64))
print(f"Wrote {out} with KernelMaxRadius pre-computed.")
