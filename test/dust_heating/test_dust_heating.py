"""Protostellar dust heating: does the tree-summed rate match kappa_P(T_eff) L / (4 pi r^2)?

The uniform glass box is used rather than the power-law core because its answer is known exactly
in the optically thin limit. The core is the qualitative test and lives in make_dust_heating_ics.py;
see README.md for why its centred configuration must not be used to measure profiles.
"""

import os

import numpy as np
import pytest
import h5py

from gizmo.test import (build_and_run_test, default_mpi_ranks, default_omp_threads,
                        get_final_snapshot, variant_output_dir)

PC = 3.085678e18
MSUN = 1.989e33
LSUN = 3.828e33
RSUN = 6.957e10
AU = 1.496e13
SIGMA_SB = 5.6704e-5

# Semenov et al. 2003 Planck-mean dust opacity, T_dust < 160 K composition zone, as tabulated in
# radiation/rt_dust_opacity.cc. Duplicated here so the test checks the code against the published
# table rather than against the code's own copy of it.
_LOG_KAPPA = np.array([
    -1.909515215033498, -1.5017616295543856, -1.2610211141587906, -1.0643751130254193,
    -0.7661794028048912, -0.2485164276172981, 0.3936319485109052, 0.7185651396015793,
    1.003536500936941, 1.0703750048744685, 1.185414318657744, 1.4334392971521788,
    1.6154100043688104, 1.833331149478953, 2.2402919406422592])
_LOG_TRAD = np.arange(15) * (4.0 / 14)


def kappa_planck(T):
    """Planck-mean dust opacity in cm^2/g at Solar metallicity, cold-dust composition."""
    return 10 ** np.interp(np.clip(np.log10(np.atleast_1d(np.float64(T))), _LOG_TRAD[0],
                                   _LOG_TRAD[-1]), _LOG_TRAD, _LOG_KAPPA)


@pytest.mark.parametrize("extra_config_flags", [()], ids=["default"])
def test_dust_heating(num_mpi_ranks, num_omp_threads, extra_config_flags):
    test_name = "dust_heating"
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads, extra_config_flags)
    snap = get_final_snapshot(variant_output_dir(test_name, extra_config_flags))

    with h5py.File(snap, "r") as f:
        gas, sink = f["PartType0"], f["PartType5"]
        centre = sink["Coordinates"][:][0]
        r = np.linalg.norm(np.float64(gas["Coordinates"][:]) - centre, axis=1) * PC
        rho = np.float64(gas["Density"][:]) * (MSUN / PC**3)
        heat = np.float64(gas["DustHeatingRate"][:]) * ((MSUN / PC**3) * (100.0**2) / (PC / 100.0))
        T_dust = np.float64(gas["Dust_Temperature"][:])
        lum = float(sink["StarLuminosity_Solar"][:][0]) * LSUN
        rad = float(sink["ProtoStellarRadius_inSolar"][:][0])

    T_eff = 5780.0 * (lum / LSUN / rad**2) ** 0.25
    kappa = float(kappa_planck(T_eff)[0])

    # bin over a decade in radius, staying inside the box and outside the sink kernel
    edges = np.logspace(np.log10(900 * AU), np.log10(8000 * AU), 12)
    idx = np.digitize(r, edges)
    r_mid, ratio, T_mid = [], [], []
    for i in range(1, len(edges)):
        m = idx == i
        if m.sum() < 40:
            continue
        rc = np.sqrt(edges[i - 1] * edges[i])
        analytic = kappa * lum / (4 * np.pi * rc**2)
        r_mid.append(rc)
        ratio.append(np.median(heat[m] / rho[m]) / analytic)
        T_mid.append(np.median(T_dust[m]))
    r_mid, ratio, T_mid = np.array(r_mid), np.array(ratio), np.array(T_mid)
    assert len(r_mid) >= 6, f"only {len(r_mid)} usable radial bins"

    # 1. the tree sum reproduces kappa_P(T_eff) L / (4 pi r^2). The ~3% offset is a constant, not a
    #    scaling error, so the flatness bound is the tighter statement and the one that would catch
    #    a broken node aggregation or a lost MPI partial sum.
    assert np.all(np.abs(ratio - 1.0) < 0.15), (
        f"tree-summed heating rate off by more than 15%: ratio {ratio.min():.3f}-{ratio.max():.3f}")
    spread = ratio.std() / ratio.mean()
    assert spread < 0.05, (
        f"heating rate / analytic varies by {spread:.3f} across radius; a correct 1/r^2 sum with a "
        f"single source should be flat. A stale per-source accumulator shows up here.")

    # 2. the equilibrium dust temperature follows radiative equilibrium for this opacity law.
    #    T^4 kappa_P(T) ~ r^-2 gives d ln T / d ln r = -2/(4+beta); beta ~ 1.6 here -> about -0.36.
    slope = np.polyfit(np.log(r_mid), np.log(T_mid), 1)[0]
    assert -0.50 < slope < -0.25, (
        f"d ln T_dust / d ln r = {slope:.3f}, outside the range bracketing radiative equilibrium "
        f"(-0.363). Too steep suggests the absorber blend is thermalising early; too shallow "
        f"suggests the opacity is not tracking the radiation colour.")

    # 3. the solver is self-consistent: T_dust must be the root of 4 sigma kappa_P(T) T^4 = Gamma.
    #    This is what fails if kappa is evaluated at the wrong temperature on either side.
    for rc, Td in zip(r_mid[::3], T_mid[::3]):
        gamma = kappa * lum / (4 * np.pi * rc**2)
        T = 20.0
        for _ in range(200):
            T = (gamma / (4 * SIGMA_SB * float(kappa_planck(T)[0]))) ** 0.25
        assert 0.6 < Td / T < 1.6, (
            f"at {rc/AU:.0f} AU, T_dust={Td:.2f} K against radiative equilibrium {T:.2f} K")
