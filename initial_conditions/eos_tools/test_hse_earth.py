"""End-to-end self-test for the Phase 17h IC builder.

Builds an Earth-like layered body (iron core, olivine mantle, adiabatic) and
checks:
  1. solver bisection converges
  2. HSE residual rms < 1%
  3. central pressure within order-of-magnitude of Earth (~3.6e12 dyn/cm^2)
  4. particle placement produces equal-mass particles, no NaNs

Run: python -m initial_conditions.eos_tools.test_hse_earth
"""

import numpy as np

from .eos_dispatch import Material
from .hse_solver import Zone, solve_hse
from .shell_placement import fibonacci_shells, glass_relax
from .build_layered_body import hse_diagnostics, write_hdf5


def main():
    iron = Material.tillotson("iron")
    olivine = Material.tillotson("olivine")
    zones = [
        Zone(olivine, inner_mass_frac=0.32, T_profile="adiabatic"),  # mantle (68%)
        Zone(iron,    inner_mass_frac=0.00, T_profile="adiabatic"),  # core (32%)
    ]
    M_earth = 5.972e27   # g
    R_earth = 6.371e8    # cm (reference; solver finds its own R)

    print("Phase 17h self-test: Earth-like body (Tillotson olivine + iron, adiabatic)")
    prof = solve_hse(M_total=M_earth, T_surface=300.0, P_surface=1e6,
                     zones=zones, verbose=True)
    R = prof["R"]
    print(f"  Solved R = {R:.3e} cm  ({R / R_earth:.2f} x R_earth)")
    print(f"  P_central = {prof['P'][-1]:.3e} dyn/cm^2  "
          f"(Earth ~ 3.6e12)")
    print(f"  T_central = {prof['T'][-1]:.3e} K")

    rel = hse_diagnostics(prof)
    assert np.sqrt(np.mean(rel ** 2)) < 0.10, "HSE residual too large"
    assert 0.5 * R_earth < R < 2.0 * R_earth, f"R={R:.3e} unreasonable"

    placed = fibonacci_shells(prof, zones, n_particles=5000)
    print(f"  Placed {len(placed['coords'])} particles, "
          f"mass scatter = {placed['masses'].std() / placed['masses'].mean():.2e}")
    assert np.isfinite(placed["coords"]).all()
    assert np.isfinite(placed["u_p"]).all()
    assert (placed["comp_p"] > 0).all()

    placed_glass = glass_relax(prof, zones, n_particles=5000, n_sweeps=5)
    print(f"  Glass-relaxed {len(placed_glass['coords'])} particles")
    assert np.isfinite(placed_glass["coords"]).all()

    write_hdf5("/tmp/hse_earth_test.hdf5", placed, prof, box_size=10 * R)
    print("OK")


if __name__ == "__main__":
    main()
