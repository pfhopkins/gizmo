// chem_rhs_net20.cc -- network 20 (TIGRESS-NCR) RHS.  y = {x_H2, x_H+, e [erg cm^-3]}.
// Neutral H, C, O and electrons follow from conservation inside n20_solve_state, so the
// element and charge budgets close by construction.

#include "chem_net20.h"
#include "chem_rhs_common.h"
#include <cstdio>

int chem_rhs_net20(double /*t*/, const double* y_in, double* ydot, ChemCell& c)
{
    const int nspec = 3, itmp = 2;
    double y[3];
    for (int i = 0; i < nspec; i++) { y[i] = y_in[i]; ydot[i] = 0.0; }
    if (chem_validate_input(y, nspec, itmp, C20_IH2, c.id_current)) return CHEM_RHS_RECOVERABLE;

    N20State s;
    if (n20_solve_state(y[C20_IH2], y[C20_IHP], y[itmp], c, s)) return CHEM_RHS_RECOVERABLE;
    n20_source_terms(s, c, &ydot[C20_IH2], &ydot[C20_IHP], &ydot[itmp]);

    for (int i = 0; i < nspec; i++) {
        if (ydot[i] != ydot[i]) {
            std::fprintf(stderr, "chem_rhs_net20: NaN ydot[%d] (ID %ld, T=%g yn=%g)\n",
                         i, c.id_current, s.T, c.yn);
            for (int j = 0; j < nspec; j++) ydot[j] = 0.0;
            return CHEM_RHS_RECOVERABLE;
        }
    }
    return CHEM_RHS_OK;
}
