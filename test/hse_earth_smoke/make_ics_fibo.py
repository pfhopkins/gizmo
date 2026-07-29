"""Plain Fibonacci IC + u-correction; default GIZMO h-iteration; CT fix in place."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from initial_conditions.eos_tools.eos_dispatch import Material
from initial_conditions.eos_tools.hse_solver import Zone, solve_hse
from initial_conditions.eos_tools.shell_placement import fibonacci_shells
from initial_conditions.eos_tools.sph_density import correct_u_for_sph_density
from initial_conditions.eos_tools.build_layered_body import write_hdf5

ol = Material.tillotson("olivine"); fe = Material.tillotson("iron")
zones = [Zone(ol, inner_mass_frac=0.32, T_profile="adiabatic"),
         Zone(fe, inner_mass_frac=0.0,  T_profile="adiabatic")]
prof = solve_hse(M_total=5.972e27, T_surface=300.0, P_surface=1e6, zones=zones, n_steps=1500)
placed = fibonacci_shells(prof, zones, n_particles=2000)
correct_u_for_sph_density(placed, prof, zones, des_num_ngb=32, verbose=True)
write_hdf5(os.path.join(os.path.dirname(__file__), "hse_earth_fibo_ics.hdf5"),
           placed, prof, box_size=10*prof['R'])
