"""Generate ICs for the ANEOS giant-impact test via the HSE builder.

Earth-mass differentiated body, forsterite mantle (68% by mass) + iron core
(32%), on the same Stewart S19/S20 .sesame tables that GIZMO reads at run-time.
Adiabatic zones throughout (the first-law adiabat GIZMO itself follows), the
temperature continuous across the core-mantle boundary; the surface is set at
T=2000 K (warm enough to sit comfortably inside the Stewart table grid).

CompositionType (EOS_ANEOS-only build):
    0 -> forsterite (mantle, AneosTable0)
    1 -> iron       (core,   AneosTable1)

build_planet places the particles on SEAGen shells with the particle mass matched
to the spacing across the core-mantle boundary, computes the density the run's
kernel (cubic spline, DesNumNgb 32: this test's Config and params) gives at the
placed positions, sets each particle's mass so that this density equals the
profile's (calibrate_masses="particle": the iron core sits close to its cold
compression curve, where the few-percent density scatter of a shell placement has
no EOS state at any temperature), and sets u so that EOS(rho, u) = P_HSE. The build stops if any
particle below the free-surface layer has no valid EOS state at its hydrostatic
pressure. The IC records, for the test: the number of such substitutions (interior
and surface), the kernel density's deviation from the profile below the surface
layer, and the hydrostatic profile itself (group HSEProfile: r, P, rho, zone_id, and
the ParticleIDs of the free-surface layer).
"""

import os
import sys
import numpy as np
import h5py

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from initial_conditions.eos_tools.eos_dispatch import Material
from initial_conditions.eos_tools.hse_solver import Zone
from initial_conditions.eos_tools.build_layered_body import build_planet


# Cv is a fallback only: the SESAME u<->T conversion inverts the table itself.
CV_FORSTERITE = 1.0e7   # erg/g/K
CV_IRON       = 4.5e6   # erg/g/K

M_EARTH = 5.972e27   # g
KERNEL_CUBIC_SPLINE = 3   # the builder's id for GIZMO's default kernel (no KERNEL_FUNCTION in Config.sh)


def main(out=None, n_particles=2000, box_size=6.0e9):
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

    forsterite = Material.sesame("forsterite_S19", forsterite_path, composition_type=0, Cv=CV_FORSTERITE)
    iron       = Material.sesame("iron_S20", iron_path, composition_type=1, Cv=CV_IRON)
    zones = [
        Zone(forsterite, inner_mass_frac=0.32, T_profile="adiabatic"),
        Zone(iron,       inner_mass_frac=0.00, T_profile="adiabatic"),
    ]

    placed, prof = build_planet(zones, M_total=M_EARTH, T_surface=2000.0, P_surface=1e6, n_particles=n_particles, out=out,
                                kernel=KERNEL_CUBIC_SPLINE, des_num_ngb=32, calibrate_masses="particle", box_size=box_size)

    interior = ~placed["free_surface"]
    n_sub_interior = int(np.sum(placed["noroot"] & interior))
    n_sub_surface = int(np.sum(placed["noroot"] & ~interior))
    if n_sub_interior:
        raise RuntimeError(f"{n_sub_interior} particles below the free-surface layer have no valid EOS state at their hydrostatic pressure")
    dev = (placed["rho_sph"] / placed["rho_profile"] - 1.0)[interior]
    with h5py.File(out, "r+") as f:
        g = f["PartType0"]
        g.attrs["EOS_substitutions_interior"] = n_sub_interior
        g.attrs["EOS_substitutions_free_surface"] = n_sub_surface
        g.attrs["N_free_surface"] = int(np.sum(~interior))
        g.attrs["KernelDensityOverProfile_interior_median"] = float(np.median(dev))
        g.attrs["KernelDensityOverProfile_interior_maxabs"] = float(np.max(np.abs(dev)))
        for k, c in (("forsterite", 0), ("iron", 1)):
            mk = placed["masses"][placed["comp_p"] == c]
            g.attrs[f"MassPercentiles_p1_p50_p99_{k}"] = np.percentile(mk, [1, 50, 99])
        p = f.create_group("HSEProfile")   # outward order, for interpolation
        p.create_dataset("r", data=prof["r"][::-1]); p.create_dataset("P", data=prof["P"][::-1])
        p.create_dataset("rho", data=prof["rho"][::-1]); p.create_dataset("zone_id", data=prof["zone_id"][::-1])
        p.create_dataset("FreeSurfaceIDs", data=g["ParticleIDs"][:][placed["free_surface"]])
    print(f"  {len(placed['masses'])} particles ({np.sum(~interior)} in the free-surface layer); EOS substitutions: interior "
          f"{n_sub_interior}, free surface {n_sub_surface}; kernel density / profile - 1 below it: median {np.median(dev):+.4f}, "
          f"max |.| {np.max(np.abs(dev)):.4f}")


if __name__ == "__main__":
    main()
