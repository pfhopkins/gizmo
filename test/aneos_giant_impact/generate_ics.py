"""Generate ICs for the ANEOS giant-impact test via the Phase 17h HSE builder.

Earth-mass differentiated body, forsterite mantle (68% by mass) + iron core
(32%). The HSE solve uses Material.sesame() against the same Stewart S19/S20
.sesame tables that GIZMO reads at run-time, so the IC's pressure profile and
the runtime EOS are consistent at construction.

Adiabatic zones throughout: each zone runs on its own adiabat anchored to the
boundary T inherited from the layer outside it; outer surface is set at
T=2000 K (warm enough to sit comfortably inside the Stewart table grid).

CompositionType (EOS_ANEOS-only build):
    0 -> forsterite (mantle, AneosTable0)
    1 -> iron       (core,   AneosTable1)

Glass-relaxed placement + per-particle u-correction so EOS(rho_SPH, u) =
P_HSE_local, matching the recipe validated in test/hse_earth_smoke.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from initial_conditions.eos_tools.eos_dispatch import Material
from initial_conditions.eos_tools.hse_solver import Zone, solve_hse
from initial_conditions.eos_tools.shell_placement import (
    glass_relax, atmosphere_shell, merge_placements,
)
from initial_conditions.eos_tools.sph_density import correct_u_for_sph_density
from initial_conditions.eos_tools.build_layered_body import write_hdf5, hse_diagnostics


# Cv supplied as a fallback only — the SESAME u<->T conversion now bisects
# on the table itself (post-fix). Setting Cv to a sane order-of-magnitude
# value keeps any residual Cv-dependent code paths well-behaved.
CV_FORSTERITE = 1.0e7   # erg/g/K
CV_IRON       = 4.5e6   # erg/g/K

M_EARTH = 5.972e27   # g


def main(out=None, n_particles=2000, n_sweeps=30,
         n_atmo=1500, atmo_rho_frac=1.0e-3, atmo_R_outer_frac=3.0,
         atmo_rho_taper_power=5.0):
    here = os.path.dirname(os.path.abspath(__file__))
    if out is None:
        out = os.path.join(here, "aneos_giant_impact_ics.hdf5")

    forsterite_path = os.path.join(here, "forsterite.sesame")
    iron_path       = os.path.join(here, "iron.sesame")
    for tbl in (forsterite_path, iron_path):
        if not os.path.isfile(tbl):
            raise FileNotFoundError(
                f"{tbl} not found. Run setup_tables() in "
                "test_aneos_giant_impact.py first (downloads + converts the "
                "Stewart S19/S20 release tables)."
            )

    forsterite = Material.sesame("forsterite_S19", forsterite_path,
                                 composition_type=0, Cv=CV_FORSTERITE)
    iron       = Material.sesame("iron_S20", iron_path,
                                 composition_type=1, Cv=CV_IRON)
    zones = [
        Zone(forsterite, inner_mass_frac=0.32, T_profile="adiabatic"),
        Zone(iron,       inner_mass_frac=0.00, T_profile="adiabatic"),
    ]

    prof = solve_hse(M_total=M_EARTH, T_surface=2000.0, P_surface=1e6,
                     zones=zones, n_steps=1500, verbose=True)
    print(f"  Solved R = {prof['R']:.3e} cm  P_c = {prof['P'][-1]:.3e}  "
          f"T_c = {prof['T'][-1]:.3e}")
    hse_diagnostics(prof)

    body = glass_relax(prof, zones, n_particles=n_particles, n_sweeps=n_sweeps)
    correct_u_for_sph_density(body, prof, zones, des_num_ngb=32, verbose=True)

    # Add a low-density atmosphere shell to cure the SPH-isolated-body
    # surface pathology (Bern-SPH / Reinhardt-Stadel recipe). ρ_atmo is set
    # to atmo_rho_frac × surface ρ of the HSE profile; T matches the body's
    # surface T. Same material as the outermost body zone, so the runtime
    # ANEOS dispatch is identical.
    rho_surface = float(prof["rho"][0])
    T_surface_body = float(prof["T"][0])
    rho_atmo = atmo_rho_frac * rho_surface
    R_atmo = atmo_R_outer_frac * prof["R"]
    print(f"  Atmosphere: rho_inner={rho_atmo:.3e} g/cm^3 "
          f"({atmo_rho_frac:.0e} x rho_surface), taper=(R_in/r)^{atmo_rho_taper_power}, "
          f"T={T_surface_body:.1f} K, R_outer={R_atmo:.3e} cm, N={n_atmo}")
    atmo = atmosphere_shell(prof, zones, n_particles=n_atmo,
                            rho_atmo=rho_atmo, T_atmo=T_surface_body,
                            R_outer=R_atmo, rho_taper_power=atmo_rho_taper_power,
                            equal_mass_to=body)

    placed = merge_placements(body, atmo)

    box = 2.0 * R_atmo + 2.0 * prof["R"]   # comfortable padding around atmo
    write_hdf5(out, placed, prof, box_size=box)


if __name__ == "__main__":
    main()
