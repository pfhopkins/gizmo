// chem_net18.h -- network 18: non-equilibrium ions of H, He, C, N, O, Ne, Mg, Si, S, Fe with RTZ rates and
// cooling, all stages and the temperature integrated together in CVODE.  Provenance: DEVIATIONS.md D-51.
#ifndef CHEM_NET18_H
#define CHEM_NET18_H

#include "chem_types.h"
#include "chem_net18_layout.h"

// ChemCell::n18_flags
#define N18_FIXT 1      // hold T at n18_fixT (energy not evolved)
#define N18_NOH2 2      // H2 frozen
#define N18_NOCO 4      // CO frozen
#define N18_UVB  8      // HM12 UV background at c.redshift (RTZ self-shielding exp(-n_H/0.01))
#define N18_EXACT 16    // ionization/recombination/charge-exchange coefficients evaluated exactly (default: log-log table)

void n18_cell_defaults(ChemCell& c);

// y = ODE vector (N18_NODE species + energy density); T for that state
double n18_temperature(const double* y, const ChemCell& c);

// full RHS; 0 ok, CHEM_RHS_RECOVERABLE on bad input.  Fills c.lambda / c.lambda_chem.
int n18_rhs(const double* y, double* ydot, ChemCell& c);

// TracAbund <-> ODE (entry: rescale every element's stages to the cell's current pool)
void n18_trac_to_ode(const double* trac, double energy, double* y, ChemCell& c);   // also picks each element's reference stage
void n18_ode_to_trac(const double* y, double* trac, const ChemCell& c);

// evolve trac / energy over dt [s]; 0 ok, else CVODE failure code (as cxx_evolve_abundances)
int n18_evolve(double dt, double* trac, double* energy, ChemCell& c);

// solver statistics of the calling thread
void n18_stats(long* calls, long* steps, long* rhs, long* jac);

#endif
