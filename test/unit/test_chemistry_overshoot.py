#!/usr/bin/env python3
"""
Test: under what conditions does the CHEMCOOL H+ abundance hit or exceed 1.0?

After SN thermal injection, a cold neutral particle can suddenly be at
T ~ 10^7 K. CHEMCOOL must evolve H+ from ~0 to ~1.0 (full ionization).
If the solver overshoots y(H+) > 1 + eps_max (1e-4), validate_input
rejects it and the simulation crashes.

This test maps out the parameter space where this happens:
- Pre-SN: cold (T ~ 10-100 K), neutral (H+ ~ 0), molecular (H2 ~ 0.1)
- Post-SN: hot (T ~ 10^6 - 10^8 K), should be fully ionized
- Variable metallicity from enrichment (Z = 0.001 to 0.5)

We can't call the Fortran DVODE solver here, but we can:
1. Characterize the initial conditions that the crash had
2. Compute the equilibrium H+ at the post-SN temperature
3. Estimate the stiffness ratio (how far H+ must jump)
4. Flag conditions where the jump is extreme enough to risk overshoot
"""

import sys
import numpy as np

# Physical constants
k_B = 1.381e-16     # erg/K
m_p = 1.673e-24     # g
eV_to_erg = 1.602e-12
E_ion_H = 13.6 * eV_to_erg  # hydrogen ionization energy

# Code units
UnitVel = 1e5        # cm/s
Msun = 1.989e33      # g

SOLAR_Z = 0.02

nfail = 0
npass = 0


def check(cond, msg, detail=""):
    global nfail, npass
    if cond:
        npass += 1
    else:
        nfail += 1
        print(f"  FAIL: {msg}")
        if detail:
            print(f"        {detail}")


def collisional_ionization_rate(T):
    """Collisional ionization rate coefficient for H (cm^3/s).
    Fit from Abel et al. 1997 / Cen 1992."""
    if T < 1e3:
        return 0.0
    T_eV = T * k_B / eV_to_erg
    return 5.85e-11 * T**0.5 * np.exp(-13.6 / T_eV) / (1.0 + T_eV**0.5)


def recombination_rate_B(T):
    """Case B recombination rate for H (cm^3/s).
    Fit from Hui & Gnedin 1997."""
    if T < 10:
        return 1e-10  # cap at low T
    lam = 2.0 * 157807.0 / T  # 2 * T_ion / T
    return 2.753e-14 * lam**1.5 / (1.0 + (lam / 2.740)**0.407)**2.242


def equilibrium_hp(T, n_H):
    """Equilibrium H+ fraction at temperature T and hydrogen density n_H.
    Balances collisional ionization vs recombination.
    Returns x_HII = n(H+) / n(H_total)."""
    k_ion = collisional_ionization_rate(T)
    k_rec = recombination_rate_B(T)
    if k_ion == 0:
        return 0.0
    # Equilibrium: k_ion * n_HI * n_e = k_rec * n_HII * n_e
    # With n_HI = (1-x)*n_H, n_HII = x*n_H, n_e ~ x*n_H (ignoring metals)
    # k_ion * (1-x) = k_rec * x
    # x = k_ion / (k_ion + k_rec)
    x = k_ion / (k_ion + k_rec)
    return min(x, 1.0)


def post_sn_temperature(E_sn_erg, M_gas_msun, N_ngb, mu=0.6):
    """Temperature after SN energy injection.
    mu ~ 0.6 for fully ionized H+He gas."""
    E_per = E_sn_erg / N_ngb
    M_gas = M_gas_msun * Msun
    u_specific = E_per / M_gas  # erg/g
    T = (5.0 / 3.0 - 1.0) * mu * m_p * u_specific / k_B
    return T


def ionization_timescale(T, n_H):
    """Timescale for collisional ionization at temperature T (seconds)."""
    k_ion = collisional_ionization_rate(T)
    if k_ion == 0:
        return np.inf
    # t_ion ~ 1 / (k_ion * n_e)
    # For initially neutral gas, n_e ~ 0, so first ionizations are slow
    # But once a few electrons exist, it's ~ 1/(k_ion * n_H) for runaway
    return 1.0 / (k_ion * n_H)


def stiffness_ratio(hp_initial, hp_equilibrium):
    """How many orders of magnitude H+ must jump."""
    if hp_initial <= 0 or hp_equilibrium <= 0:
        return 0.0
    return np.log10(hp_equilibrium / max(hp_initial, 1e-30))


# ============================================================

def test_crash_conditions():
    """Reproduce the exact conditions of the job 150573 crash."""
    print("\n--- Crash conditions (particle 3095686) ---")

    # Pre-SN state
    hp_pre = 2.1e-6
    h2_pre = 0.074
    T_pre = 13.0  # K
    n_H = 17.17   # cm^-3

    # Post-SN (1e51 erg into 32 neighbors of ~4 Msun)
    T_post = post_sn_temperature(1e51, 4.0, 32, mu=0.6)
    hp_eq = equilibrium_hp(T_post, n_H)
    stiff = stiffness_ratio(hp_pre, hp_eq)
    t_ion = ionization_timescale(T_post, n_H)

    print(f"  Pre-SN:  T={T_pre:.0f} K, H+={hp_pre:.2e}, H2={h2_pre:.3f}")
    print(f"  Post-SN: T={T_post:.2e} K")
    print(f"  Equilibrium H+ = {hp_eq:.6f}")
    print(f"  Stiffness: H+ must jump {stiff:.1f} orders of magnitude")
    print(f"  Ionization timescale: {t_ion:.2e} s = {t_ion/3.15e7:.2e} yr")
    print(f"  eps_max = 1e-4, overshoot was 1.4e-4")
    print()

    # The solver needs to go from 2e-6 to 1.0 — that's extreme
    check(stiff > 5.0,
          f"Stiffness ratio {stiff:.1f} should be > 5 (confirming extreme jump)")
    print("  done")


def test_parameter_space():
    """Map (T_post, Z) parameter space for overshoot risk.
    Flag conditions where H+ equilibrium is very close to 1.0 and
    the stiffness ratio is extreme."""
    print("\n--- Parameter space: T_post vs metallicity ---")
    print(f"  {'T_post':>10s} {'Z':>8s} {'Z/Zsun':>8s} {'hp_eq':>8s} {'stiff':>8s} {'t_ion_yr':>10s} {'risk':>6s}")

    hp_pre = 1e-6  # typical cold neutral gas

    temperatures = [1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8]
    metallicities = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    n_H = 17.0  # cm^-3

    high_risk = 0
    for T in temperatures:
        for Z in metallicities:
            hp_eq = equilibrium_hp(T, n_H)
            stiff = stiffness_ratio(hp_pre, hp_eq)
            t_ion = ionization_timescale(T, n_H)
            t_ion_yr = t_ion / 3.15e7

            # Risk: equilibrium H+ > 0.999 AND stiffness > 4
            risk = hp_eq > 0.999 and stiff > 4
            if risk:
                high_risk += 1
            risk_str = "HIGH" if risk else "ok"

            print(f"  {T:10.0e} {Z:8.4f} {Z/SOLAR_Z:8.1f} {hp_eq:8.5f} {stiff:8.1f} {t_ion_yr:10.2e} {risk_str:>6s}")

    print(f"\n  {high_risk} high-risk conditions (H+ -> 1.0, stiffness > 10^4)")
    # At least some conditions should be high risk
    check(high_risk > 0, "Should find high-risk conditions in parameter space")
    print("  done")


def test_sn_energy_vs_mass():
    """How does post-SN temperature vary with SN energy and gas cell mass?
    Higher energy or lower mass -> higher T -> more likely overshoot."""
    print("\n--- SN energy vs gas cell mass ---")
    print(f"  {'E_sn':>10s} {'M_gas':>8s} {'N_ngb':>6s} {'T_post':>10s} {'hp_eq':>8s} {'stiff':>8s}")

    hp_pre = 1e-6
    n_H = 17.0

    cases = [
        (5e50, 4.0, 32, "ECSN into 4 Msun"),
        (1e51, 4.0, 32, "CCSN/PPISN into 4 Msun"),
        (1e52, 4.0, 32, "PISN into 4 Msun"),
        (1e51, 1.0, 32, "CCSN into 1 Msun"),
        (1e51, 0.5, 32, "CCSN into 0.5 Msun"),
        (1e51, 4.0, 8,  "CCSN, few neighbors"),
    ]

    for E_sn, M_gas, N_ngb, label in cases:
        T = post_sn_temperature(E_sn, M_gas, N_ngb)
        hp_eq = equilibrium_hp(T, n_H)
        stiff = stiffness_ratio(hp_pre, hp_eq)
        print(f"  {E_sn:10.0e} {M_gas:8.1f} {N_ngb:6d} {T:10.2e} {hp_eq:8.5f} {stiff:8.1f}  {label}")

    print("  done")


def test_critical_overshoot_margin():
    """At what temperature does equilibrium H+ exceed 1 - eps_max?
    This is where the solver is most likely to overshoot past 1 + eps_max."""
    print("\n--- Critical temperature for H+ ~ 1.0 ---")

    eps_max = 1e-4
    n_H = 17.0

    # Binary search for T where hp_eq = 1 - eps_max
    T_lo, T_hi = 1e3, 1e9
    for _ in range(100):
        T_mid = np.sqrt(T_lo * T_hi)
        hp = equilibrium_hp(T_mid, n_H)
        if hp < 1.0 - eps_max:
            T_lo = T_mid
        else:
            T_hi = T_mid

    T_crit = np.sqrt(T_lo * T_hi)
    hp_crit = equilibrium_hp(T_crit, n_H)
    print(f"  H+ = 1 - eps_max at T ~ {T_crit:.2e} K")
    print(f"  hp_eq(T_crit) = {hp_crit:.8f}")
    print(f"  Above this T, any solver overshoot risks exceeding 1 + eps_max")
    print()

    # All SN should heat above this
    T_sn = post_sn_temperature(1e51, 4.0, 32)
    print(f"  Typical SN: T = {T_sn:.2e} K >> T_crit = {T_crit:.2e} K")
    check(T_sn > T_crit,
          f"SN temperature {T_sn:.0e} should exceed critical T {T_crit:.0e}")

    # Even weakest SN (ECSN)
    T_ecsn = post_sn_temperature(5e50, 4.0, 32)
    print(f"  ECSN:       T = {T_ecsn:.2e} K vs T_crit = {T_crit:.2e} K")
    check(T_ecsn > T_crit,
          f"ECSN temperature {T_ecsn:.0e} should exceed critical T {T_crit:.0e}")

    print("  done")


if __name__ == "__main__":
    print("Chemistry overshoot analysis: H+ near 1.0 after SN injection")

    test_crash_conditions()
    test_parameter_space()
    test_sn_energy_vs_mass()
    test_critical_overshoot_margin()

    print(f"\n{'='*60}")
    print(f"Results: {npass} passed, {nfail} FAILED")
    sys.exit(1 if nfail > 0 else 0)
