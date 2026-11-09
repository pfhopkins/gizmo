/* C++ implementation: Ulrich Steinwandel (uli@mpa-garching.mpg.de).
   Based on the SGChem Fortran networks by Simon Glover (GNU GPL, as distributed in Arepo). */
// chem_spline.cc -- port of the SGCHEM spline layer (spline.F) and of the
// four genuinely table/spline-driven cooling entries cl1, cl2, cl3, cl6.
//
// Gate-1 completion: RATE_AUDIT.md classified 8 cltab entries as kind=TABLE
// (cl1-3, cl6, cl52, cl59, cl60, cl61) because the extractor saw them assigned
// both in coolinmo's analytic temperature loop AND in a later residual loop.
// Reading the residual loop (coolinmo.F:1520-1590) resolves that:
//
//   cl1, cl2, cl3 : analytic below 100 K and above 1e4 K; a natural-cubic
//                   spline over the 41 mol_data.h points INSIDE [1e2, 1e4] K
//                   overwrites the (unassigned) interior.  Genuinely TABLE.
//   cl6           : iflag_atom=2 (the production setting, sgchem.cc /
//                   onezone drivers) -> spline over the 76-point ca2 DATA
//                   array, zero below 10^4.3 K.  Genuinely TABLE.
//   cl52, cl59,   : zeroed by the residual loop ONLY under iflag_atom==1
//   cl60, cl61      (coolinmo.F:1541-1554).  Under iflag_atom==2 the zeroing
//                   is additionally #ifndef WSS_CIE_COOL, and WSS_CIE_COOL is
//                   ON in every Stella build.  => in the production config
//                   these four keep their ANALYTIC main-loop values and are
//                   NOT table-driven at all.  Their formulas are already in
//                   RATE_AUDIT.md and are covered by the Gate-1 rate goldens.
//                   Nothing to port; see DEVIATIONS.md D-9.
//
// So the "8 TABLE splines" are 4 spline ports + 4 audit-classification
// artifacts.  This file implements the 4 and the shared spline machinery.

#include "chem_types.h"
#include "chem_tables_data.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// spline.F port.  Note the exact parameterisation: the cubic is evaluated in
// the NORMALIZED local coordinate dpt = (x - x_i)/(x_{i+1} - x_i), and the
// derivatives are those of the uniform-knot Bartels formulation -- i.e. the
// knot SPACING does not enter spline_derivatives at all.  That is not a
// standard non-uniform natural cubic spline; it is what the Fortran does, and
// reproducing it exactly is the point (both cl1-3 and cl6 use non-uniform
// knots, so a "correct" spline would give different numbers).
// ---------------------------------------------------------------------------
void chem_spline_derivatives(int nval, const double* values, double* deriv)
{
    const int n = nval;
    // augmented matrix Cc(n, n+1), row-major
    std::vector<double> Cc((size_t)n * (n + 1), 0.0);
    auto C = [&](int i, int j) -> double& { return Cc[(size_t)i * (n + 1) + j]; };

    for (int I = 0; I < n; I++) {
        if (I == 0) {
            C(0, 0) = 2.0; C(0, 1) = 1.0;
            C(0, n) = 3.0 * (values[1] - values[0]);
        } else if (I == n - 1) {
            C(I, I - 1) = 1.0; C(I, I) = 2.0;
            C(I, n) = 3.0 * (values[n - 1] - values[n - 2]);
        } else {
            C(I, I - 1) = 1.0; C(I, I) = 4.0; C(I, I + 1) = 1.0;
            C(I, n) = 3.0 * (values[I + 1] - values[I - 1]);
        }
    }
    for (int I = 1; I < n; I++) {
        double f = C(I, I - 1) / C(I - 1, I - 1);
        for (int J = 0; J <= n; J++) C(I, J) -= f * C(I - 1, J);
    }
    deriv[n - 1] = C(n - 1, n) / C(n - 1, n - 1);
    for (int I = n - 2; I >= 0; I--) {
        double sum = 0.0;
        for (int J = I + 1; J < n; J++) sum += C(I, J) * deriv[J];
        deriv[I] = (C(I, n) - sum) / C(I, I);
    }
}

void chem_spline_coefficients(int nval, const double* values, double* coef /*4*(nval-1)*/)
{
    std::vector<double> d(nval);
    chem_spline_derivatives(nval, values, d.data());
    for (int I = 0; I < nval - 1; I++) {
        coef[4 * I + 0] = values[I];
        coef[4 * I + 1] = d[I];
        coef[4 * I + 2] = 3.0 * (values[I + 1] - values[I]) - 2.0 * d[I] - d[I + 1];
        coef[4 * I + 3] = 2.0 * (values[I] - values[I + 1]) + d[I] + d[I + 1];
    }
}

void chem_spline_eval(int nval, const double* positions, const double* values,
                      int nnew, const double* new_positions, double* new_values)
{
    std::vector<double> coef(4 * (size_t)(nval - 1));
    chem_spline_coefficients(nval, values, coef.data());

    for (int I = 0; I < nnew; I++) {
        double pt = new_positions[I];
        int index = -1;
        double dpt = 0.0;
        if (pt < positions[0] || pt > positions[nval - 1]) {
            new_values[I] = 0.0;
        } else if (pt == positions[nval - 1]) {
            new_values[I] = values[nval - 1];
        } else {
            index = 0;
            // NB the Fortran scans ALL intervals and keeps the LAST match; with
            // strictly increasing knots that is the unique match.  `dpt` is
            // only updated when a match is found -- if no interval matches it
            // retains its previous value, exactly as the Fortran `dpt` (which
            // is a saved local, initialised to 0 once).  Reproduced literally.
            for (int J = 0; J < nval - 1; J++) {
                if (pt >= positions[J] && pt < positions[J + 1]) {
                    index = J;
                    dpt = (pt - positions[index]) /
                          (positions[index + 1] - positions[index]);
                }
            }
        }
        if (index != -1) {
            double a = coef[4 * index + 0], b = coef[4 * index + 1];
            double c = coef[4 * index + 2], d = coef[4 * index + 3];
            new_values[I] = a + b * dpt + c * dpt * dpt + d * dpt * dpt * dpt;
        }
    }
}

// ---------------------------------------------------------------------------
// The four table-driven cooling entries.
// temptab must be the coolinmo grid: temptab(i) = 10^((i-1)*dtlog),
// dtlog = log10(1e8)/(nmd-1), nmd = CHEM_NMD.
// out_cl1/2/3/6 are filled for all nmd entries.
// ---------------------------------------------------------------------------
void chem_build_cl_tables(const double* temptab, int nmd,
                          double* cl1, double* cl2, double* cl3, double* cl6,
                          int iflag_atom)
{
    const double kboltz = CHEM_KBOLTZ;
    const double opratio = 2.4;
    const double fortho = opratio / (1.0 + opratio);
    const double fpara  = 1.0 / (1.0 + opratio);
    const double h2e20 = 508.95, h2e31 = 852.5;

    // ---- analytic branches (coolinmo.F:247-300) ----
    for (int i = 0; i < nmd; i++) {
        double temp = temptab[i];
        double tau = temp * 1e-3 + 1.0;
        cl1[i] = cl2[i] = cl3[i] = 0.0;   // interior left unassigned by the .F

        if (temp < 5.0) {
            cl1[i] = 1e-60;
        } else if (temp < 1e2) {
            double h2n2 = 0.25 * (5.0 * std::exp(-h2e20 / temp) /
                                  (1.0 + 5.0 * std::exp(-h2e20 / temp)));
            double h2n3 = 0.75 * ((7.0 / 3.0) * std::exp(-h2e31 / temp) /
                                  (1.0 + (7.0 / 3.0) * std::exp(-h2e31 / temp)));
            double f = 2.94e-11 * h2e20 * h2n2 * kboltz +
                       4.76e-10 * h2e31 * h2n3 * kboltz;
            cl1[i] = (f > 1e-60) ? f : 1e-60;
        } else if (temp > 1e4) {
            cl1[i] = std::pow(10.0, -18.253);
        }

        {
            double q02 = std::pow(10.0, -9.121 - (3.983 / tau) - (1.844e-1 / (tau * tau)))
                         * 5.0 * std::exp(-h2e20 / temp);
            double q13 = std::pow(10.0, -9.493 - (4.435 / tau) + (2.707e-1 / (tau * tau)))
                         * (7.0 / 3.0) * std::exp(-h2e31 / temp);
            if (temp < 5.0)        cl2[i] = 1e-60;
            else if (temp < 1e2)   cl2[i] = (fpara * q02 * h2e20 + fortho * q13 * h2e31) * kboltz;
            else if (temp > 1e4)   cl2[i] = std::pow(10.0, -21.943);
        }
        {
            double q02 = std::pow(10.0, -9.946 - (2.688 / tau) + (2.020e-1 / (tau * tau)))
                         * 5.0 * std::exp(-h2e20 / temp);
            double q13 = std::pow(10.0, -9.785 - (3.024 / tau) + (2.930e-1 / (tau * tau)))
                         * (7.0 / 3.0) * std::exp(-h2e31 / temp);
            if (temp < 5.0)        cl3[i] = 1e-60;
            else if (temp < 1e2)   cl3[i] = (fpara * q02 * h2e20 + fortho * q13 * h2e31) * kboltz;
            else if (temp > 1e4)   cl3[i] = std::pow(10.0, -22.758);
        }
    }

    // ---- cl1-3 spline interior (coolinmo.F:1154-1168) ----
    {
        const int nh2 = 41;
        double h2_temp[41];
        for (int i = 0; i < nh2; i++) h2_temp[i] = std::pow(10.0, 2.0 + 5e-2 * i);
        std::vector<double> r1(nmd), r2(nmd), r3(nmd);
        chem_spline_eval(nh2, h2_temp, dat_h2_lte,     nmd, temptab, r1.data());
        chem_spline_eval(nh2, h2_temp, dat_h2_h_rate,  nmd, temptab, r2.data());
        chem_spline_eval(nh2, h2_temp, dat_h2_h2_rate, nmd, temptab, r3.data());
        for (int i = 0; i < nmd; i++) {
            if (temptab[i] >= 1e2 && temptab[i] <= 1e4) {
                cl1[i] = std::pow(10.0, r1[i]);
                cl2[i] = std::pow(10.0, r2[i]);
                cl3[i] = std::pow(10.0, r3[i]);
            }
        }
    }

    // ---- cl6 (coolinmo.F:1520-1590) ----
    std::vector<double> r0(nmd);
    if (iflag_atom == 1) {
        const int natom = 81;
        double ct[81];
        for (int i = 0; i < natom; i++) ct[i] = std::pow(10.0, 8.0 * i / 79.0);
        chem_spline_eval(natom, ct, dat_coolatom, nmd, temptab, r0.data());
        for (int i = 0; i < nmd; i++) cl6[i] = std::pow(10.0, r0[i]);
    } else if (iflag_atom == 2) {
        const int natom2 = 76;
        double ct[76];
        ct[0] = 1.0;                                   // ca2_temp(1) = tmin = 1
        for (int i = 1; i < natom2; i++)               // i(F) = i(C)+1
            ct[i] = std::pow(10.0, 4.25 + 0.05 * (i + 1));
        chem_spline_eval(natom2, ct, dat_ca2, nmd, temptab, r0.data());
        for (int i = 0; i < nmd; i++)
            cl6[i] = (temptab[i] < std::pow(10.0, 4.3)) ? 0.0
                                                        : std::pow(10.0, r0[i]);
    } else {
        for (int i = 0; i < nmd; i++) cl6[i] = 0.0;
    }
}
