// chem_rhs_common.h -- shared RHS helpers for the net-5 / net-15 / net-17 ports.
//
// validate_input (validate_rates.F:9-77) ported per audit R3 REDESIGN:
// operates on a LOCAL COPY of the state vector (CVODE forbids mutating y in
// the RHS).  Semantics preserved exactly:
//   NaN                -> gross failure (caller returns CHEM_RHS_RECOVERABLE)
//   y < 0              -> clip to 0 on the copy (no flag)
//   y > 1  (non-H2, non-energy): clip to 1; flag only if > 1+EPS_MAX
//   H2 > 0.5           : clip to 0.5; flag only if > 0.5+EPS_MAX
// "flag" == Fortran ipar(1)=1 == the whole evaluation is rejected (the Fortran
// returns zero ydot and the driver rejects the substep; the CVODE port returns
// a recoverable error so CVODE retries with a smaller step).
#ifndef CHEM_RHS_COMMON_H
#define CHEM_RHS_COMMON_H

#include "chem_types.h"
#include <cstdio>

// NaN states seen by validate_input on this thread; the driver reports them once per solve
extern thread_local long chem_nan_states;

// returns 0 ok, 1 flagged (reject evaluation)
inline int chem_validate_input(double* y, int nspec, int itmp0 /*0-based energy slot*/,
                               int ih2_0 /*0-based H2 slot*/, long id_current)
{
    for (int i = 0; i < nspec; i++) {
        if (y[i] != y[i]) {
            (void)id_current;
            chem_nan_states++;
            return 1;
        }
        if (y[i] < 0.0) y[i] = 0.0;
        if (i != itmp0 && i != ih2_0) {
            if (y[i] > 1.0) {
                int flag = (y[i] > 1.0 + CHEM_EPS_MAX);
                y[i] = 1.0;
                if (flag) return 1;
            }
        } else if (i == ih2_0) {
            if (y[i] > 0.5) {
                int flag = (y[i] > 0.5 + CHEM_EPS_MAX);
                y[i] = 0.5;
                if (flag) return 1;
            }
        }
    }
    return 0;
}

// full validate_input sweep: clips every species like the Fortran, which keeps
// going after an excess flag.  Returns 0 ok, 1 excess flagged, -1 NaN.
inline int chem_validate_input_full(double* y, int nspec, int itmp0, int ih2_0,
                                    long id_current)
{
    int flag = 0;
    for (int i = 0; i < nspec; i++) {
        if (y[i] != y[i]) {
            (void)id_current;
            chem_nan_states++;
            return -1;
        }
        if (y[i] < 0.0) y[i] = 0.0;
        if (i != itmp0 && i != ih2_0) {
            if (y[i] > 1.0) {
                if (y[i] > 1.0 + CHEM_EPS_MAX) flag = 1;
                y[i] = 1.0;
            }
        } else if (i == ih2_0) {
            if (y[i] > 0.5) {
                if (y[i] > 0.5 + CHEM_EPS_MAX) flag = 1;
                y[i] = 0.5;
            }
        }
    }
    return flag;
}

// grain-surface H+ recombination rate (identical blocks in rate_eq_simple.F:232-253
// and rate_eq_ulli.F:575-594)
inline double chem_h_gr(double temp, double yn, double abe, double G_dust,
                        double dust_to_gas_ratio, double ch34, double ch35)
{
    double phi = (abe == 0.0) ? 1e20 : G_dust * std::sqrt(temp) / (yn * abe);
    double h_gr;
    if (phi < 1e-6) {
        h_gr = 1.225e-13 * dust_to_gas_ratio;
    } else {
        double hgrvar1 = 8.074e-6 * std::pow(phi, 1.378);
        double hgrvar2 = 1.0 + ch34 * std::pow(phi, ch35);
        h_gr = 1.225e-13 * dust_to_gas_ratio / (1.0 + hgrvar1 * hgrvar2);
    }
    return h_gr * std::exp(-temp / 3e4);
}

#endif
