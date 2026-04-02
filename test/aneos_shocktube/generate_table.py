"""Generate an ideal-gas SESAME table for ANEOS shocktube test.

The table encodes a gamma=1.4 ideal gas with:
  P = (gamma-1) * rho * u
  u = Cv * T
  cs = sqrt(gamma * (gamma-1) * Cv * T)

The Sod shocktube operates in code units where the unit system
is such that rho ~ O(1), u ~ O(1), P ~ O(1). We choose Cv so
that the table covers the relevant range.

Since GIZMO's ANEOS dispatch converts to CGS before calling
aneos_compute and converts back, the table must be in CGS.
We set UNIT_LENGTH = UNIT_MASS = UNIT_VELOCITY = 1 (i.e., code
units = CGS) for this test, making the conversion trivial.
"""

import numpy as np

GAMMA = 1.4
CV = 1.0 / (GAMMA - 1.0)  # Cv = 1/(gamma-1) so that u = Cv*T and P = rho*T*(gamma-1)*Cv = rho*T

NRHO = 200
NT = 200
LOGRHO_MIN = -6.0  # 1e-6
LOGRHO_MAX = 4.0   # 1e4
LOGT_MIN = -4.0    # 1e-4
LOGT_MAX = 6.0     # 1e6

def generate():
    fname = "ideal_gas_gamma14.sesame"

    rho_arr = np.logspace(LOGRHO_MIN, LOGRHO_MAX, NRHO)
    T_arr = np.logspace(LOGT_MIN, LOGT_MAX, NT)

    with open(fname, 'w') as f:
        # Header: mat_id nrho nT ncols
        f.write(f"1400 {NRHO} {NT} 6\n")

        # Density grid
        for rho in rho_arr:
            f.write(f"{rho:.15e}\n")

        # Temperature grid
        for T in T_arr:
            f.write(f"{T:.15e}\n")

        # 2D data: rho T P u S cs
        for rho in rho_arr:
            for T in T_arr:
                u = CV * T
                P = (GAMMA - 1.0) * rho * u
                cs = np.sqrt(GAMMA * P / rho) if rho > 0 else 0
                # Entropy (arbitrary positive offset)
                S = CV * np.log(T) - (GAMMA - 1.0) * CV * np.log(rho) + 100.0
                S = max(S, 1e-30)
                f.write(f"{rho:.15e} {T:.15e} {P:.15e} {u:.15e} {S:.15e} {cs:.15e}\n")

    print(f"Generated {fname}: {NRHO}x{NT} grid, gamma={GAMMA}")

if __name__ == "__main__":
    generate()
