"""Generate an Earth-like HSE IC using the Phase 17h builder.

Mantle (olivine, 68%) + iron core (32%); adiabatic; 2000 particles for a
fast smoke run. Output: hse_earth_smoke_ics.hdf5 in CGS code units
(UnitLength=UnitMass=UnitVelocity=1 -> code = CGS).
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from initial_conditions.eos_tools.eos_dispatch import Material
from initial_conditions.eos_tools.hse_solver import Zone, solve_hse
from initial_conditions.eos_tools.shell_placement import fibonacci_shells, glass_relax
from initial_conditions.eos_tools.sph_density import correct_u_for_sph_density
from initial_conditions.eos_tools.build_layered_body import write_hdf5, hse_diagnostics


def main():
    olivine = Material.tillotson("olivine")  # CompositionType = 5
    iron    = Material.tillotson("iron")     # CompositionType = 3
    zones = [
        Zone(olivine, inner_mass_frac=0.32, T_profile="adiabatic"),
        Zone(iron,    inner_mass_frac=0.00, T_profile="adiabatic"),
    ]
    M_earth = 5.972e27
    prof = solve_hse(M_total=M_earth, T_surface=300.0, P_surface=1e6,
                     zones=zones, n_steps=1500, verbose=True)
    print(f"  Solved R = {prof['R']:.3e} cm  P_c = {prof['P'][-1]:.3e}  T_c = {prof['T'][-1]:.3e}")
    hse_diagnostics(prof)

    placed = glass_relax(prof, zones, n_particles=2000, n_sweeps=30)
    # Glass placement breaks the radial step-function that traps GIZMO's
    # h-iteration (see hydro/density.cc:470 bracket-shrink termination).
    # Now ρ_GIZMO converges to N=32 properly and matches Python SPH; the
    # u-correction step adjusts u so EOS(ρ_SPH, u_corr) = P_HSE per particle.
    correct_u_for_sph_density(placed, prof, zones, des_num_ngb=32, verbose=True)
    R = prof["R"]
    out = os.path.join(os.path.dirname(__file__), "hse_earth_smoke_ics.hdf5")
    write_hdf5(out, placed, prof, box_size=10 * R)


if __name__ == "__main__":
    main()
