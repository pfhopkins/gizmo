"""Single-material basalt sphere — clean test of GIZMO HSE.
~Earth-mass basalt body, glass-relaxed, SPH-corrected u, self-gravity ON.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from initial_conditions.eos_tools.eos_dispatch import Material
from initial_conditions.eos_tools.hse_solver import Zone, solve_hse
from initial_conditions.eos_tools.shell_placement import glass_relax
from initial_conditions.eos_tools.sph_density import correct_u_for_sph_density
from initial_conditions.eos_tools.build_layered_body import write_hdf5, hse_diagnostics

basalt = Material.tillotson("basalt")  # CT=2
zones = [Zone(basalt, inner_mass_frac=0.0, T_profile="adiabatic")]
M_total = 5.972e27
prof = solve_hse(M_total=M_total, T_surface=300.0, P_surface=1e6,
                 zones=zones, n_steps=1500, verbose=True)
print(f"R={prof['R']:.3e}  P_c={prof['P'][-1]:.3e}  T_c={prof['T'][-1]:.3e}")
hse_diagnostics(prof)
placed = glass_relax(prof, zones, n_particles=2000, n_sweeps=30)
correct_u_for_sph_density(placed, prof, zones, des_num_ngb=32, verbose=True)
write_hdf5(os.path.join(os.path.dirname(__file__), "hse_basalt_ics.hdf5"),
           placed, prof, box_size=10*prof['R'])
