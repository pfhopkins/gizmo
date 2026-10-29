// chem_calc_temp.cc -- Gate-3 port of calc_temp.F and compute_gamma.F.
//
//   lookup_temp_gamma  (calc_temp.F:159-211)  -- the T/gamma table lookup used
//                                                by rate_eq_* and the driver
//   calc_spec          (calc_temp.F:49-67)
//   calc_gamma_temp    (calc_temp.F:84-99)
//   calc_temp/calc_gamma are one-line wrappers around lookup_temp_gamma
//   compute_gamma      (compute_gamma.F:9-64)  -- the high-n gamma table
//
// The tables themselves (temp_table / gamma_table / eh2_table, from
// energy-table.txt and eh2-table.txt) are loaded in chem_tables_build.cc,
// which is the port of init_temperature_lookup.

#include "chem_types.h"
#include <cmath>
#include <algorithm>

// ===========================================================================
// lookup_temp_gamma (calc_temp.F:159-211)
// ===========================================================================
void cxx_lookup_temp_gamma(double abh2, double abhe, double ekn, double* temp, double* gamma,
                           const ChemTables& T)
{
    if (ekn < 1.0 || ekn > 5e4 || abh2 <= 0.0) {
        // For ekn > 50000 the H2 content of the gas is negligible.
        *temp  = (2.0 / 3.0) * ekn;
        *gamma = 5.0 / 3.0;
        return;
    }
    // table is indexed by the H2 share of heavy particles, exact for any abhe
    double xh2 = std::min(std::max(abh2 / (1.0 + abhe - abh2), 0.0), 1.0);
    int idx_en     = std::min((int)ekn, 49999);
    double diff_en = ekn - idx_en;
    double rx      = xh2 * (CHEM_NETAB_X - 1);
    int idx_x      = std::min((int)rx, CHEM_NETAB_X - 2);
    double diff_x  = rx - idx_x;
    int r1 = 50000 * idx_x + idx_en - 1, r2 = r1 + 50000;   // 0-based rows of blocks idx_x, idx_x+1
    double temp_1  = T.temp_table[r1]  + (T.temp_table[r1 + 1]  - T.temp_table[r1])  * diff_en;
    double gamma_1 = T.gamma_table[r1] + (T.gamma_table[r1 + 1] - T.gamma_table[r1]) * diff_en;
    double temp_2  = T.temp_table[r2]  + (T.temp_table[r2 + 1]  - T.temp_table[r2])  * diff_en;
    double gamma_2 = T.gamma_table[r2] + (T.gamma_table[r2 + 1] - T.gamma_table[r2]) * diff_en;
    *temp  = temp_1  + (temp_2  - temp_1)  * diff_x;
    *gamma = gamma_1 + (gamma_2 - gamma_1) * diff_x;
}

// ===========================================================================
// calc_spec (calc_temp.F:49-67) -- H2 contribution to the specific energy
// ===========================================================================
double cxx_calc_spec(const ChemTables& T, double temp)
{
    if (temp < 1.0 || temp > 5e4) return 1.5 * temp;
    int idx = (int)temp;
    double dtemp = temp - idx;
    return T.eh2_table[idx - 1] + dtemp * (T.eh2_table[idx] - T.eh2_table[idx - 1]);
}

// ===========================================================================
// calc_gamma_temp (calc_temp.F:84-99)
// ===========================================================================
double cxx_calc_gamma_temp(const ChemTables& T, double abh2, double temp,
                           double abhe)
{
    double en = cxx_calc_spec(T, temp);
    double ekn = (abh2 / (1.0 + abhe - abh2)) * en
               + ((1.0 + abhe - 2.0 * abh2) / (1.0 + abhe - abh2)) * 1.5 * temp;
    double t, g;
    cxx_lookup_temp_gamma(abh2, abhe, ekn, &t, &g, T);
    return g;
}

// ===========================================================================
// compute_gamma (compute_gamma.F:9-64) -- gamma(e/kn, fh2) from highn_gamma.h
// ===========================================================================
double cxx_compute_gamma(const ChemTables& T, double abh2, double ekn)
{
    double lekn = std::log10(ekn);
    int index; double scale;
    if (lekn <= 1.5) {
        index = 1; scale = 0.0;
    } else if (lekn >= 4.5) {
        index = 25; scale = 0.0;
    } else {
        index = (int)((lekn - 1.5) / 0.125) + 1;
        scale = (lekn - 1.5 - 0.125 * (index - 1)) / 0.125;
    }

    const double* h = T.hng;
    if (abh2 == 0.0) return 5.0 / 3.0;

    if (abh2 < 0.1) {
        double gl = 5.0 / 3.0;
        double gh = (scale > 0.0) ? h[index-1] + (h[index] - h[index-1]) * scale
                                  : h[index-1];
        return gl + (abh2 / 0.1) * (gh - gl);
    }
    if (abh2 == 0.5) {
        index = 100 + index;
        return (scale > 0.0) ? h[index-1] + (h[index] - h[index-1]) * scale
                             : h[index-1];
    }
    double scale_h2 = (abh2 / 0.1) - (int)(abh2 / 0.1);
    int index1 = index + 25 * ((int)(abh2 / 0.1) - 1);
    int index2 = index + 25 * (int)(abh2 / 0.1);
    double gl, gh;
    if (scale > 0.0) {
        gl = h[index1-1] + (h[index1] - h[index1-1]) * scale;
        gh = h[index2-1] + (h[index2] - h[index2-1]) * scale;
    } else {
        gl = h[index1-1];
        gh = h[index2-1];
    }
    return gl + (gh - gl) * scale_h2;
}
