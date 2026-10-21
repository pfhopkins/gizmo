// chem_net20.h -- network 20: TIGRESS-NCR thermochemistry (ApJS 264, 10; arXiv:2210.08024).
// Time-dependent H2 and H+ (+ energy); C+, CO, C, O, O+, MH+ and e- in equilibrium.
// Provenance and simplifications: DEVIATIONS.md D-29.
#ifndef CHEM_NET20_H
#define CHEM_NET20_H

#include "chem_types.h"

// algebraic state of a cell at given (x_H2, x_H+, energy density)
struct N20State {
    double T, xe, xH2, xHp, xHI;
    double xCp, xC, xCO, xO, xOp, xMHp;
    double xCtot, xOtot;
};

// one-time per-cell defaults of the net-20 fields
void n20_cell_defaults(ChemCell& c);

// fields, CR attenuation and dust heating from G0/G0_LW, the pixel columns and Z'_d (once per cell call)
void cxx_calc_photo_net20(ChemCell& c);

// joint T / x_e fixed point and equilibrium C, O, e species; 0 ok, 1 bad input
int n20_solve_state(double xH2, double xHp, double energy, ChemCell& c, N20State& s);

// d x_H2/dt, d x_H+/dt [s^-1] and d e/dt [erg cm^-3 s^-1]; fills c.lambda / c.lambda_chem and c.tdust
void n20_source_terms(const N20State& s, ChemCell& c, double* dxH2, double* dxHp, double* dedt);

#endif
