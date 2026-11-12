/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
#ifndef SGCHEM_UVB_SHIELD_H
#define SGCHEM_UVB_SHIELD_H
/* UV-background shielding by absorber columns (SGCHEM_UVB_TREECOL_SHIELDING). Standalone. */
#include <math.h>
/* FG20 UVB at z=6.5 in 10 bands (13.6-16.3-19.7-24.59-29.9-35.4-40.8-46.3-51-54.42 eV-2 keV): absorber cross sections [cm^2] of HI, HeI, HeII
   and each band's fraction of the HI, HeI, HeII photoionisation and photoheating rates (Verner+96 cross sections) */
#define UVB_SHIELD_NBANDS 10
static const double uvb_band_sigma[UVB_SHIELD_NBANDS][3] = {{5.2758e-18, 0.0000e+00, 0.0000e+00}, {3.1271e-18, 0.0000e+00, 0.0000e+00}, {1.7763e-18, 0.0000e+00, 0.0000e+00}, {9.8995e-19, 6.5297e-18, 0.0000e+00}, {5.8335e-19, 4.7654e-18, 0.0000e+00}, {3.4596e-19, 3.4189e-18, 0.0000e+00}, {2.5052e-19, 2.7632e-18, 0.0000e+00}, {1.7903e-19, 2.1999e-18, 0.0000e+00}, {1.3624e-19, 1.8172e-18, 0.0000e+00}, {1.0888e-19, 1.5396e-18, 1.4188e-18}};
static const double uvb_band_gamma[UVB_SHIELD_NBANDS][3] = {{5.3842e-01, 0.0000e+00, 0.0000e+00}, {2.3627e-01, 0.0000e+00, 0.0000e+00}, {1.3486e-01, 0.0000e+00, 0.0000e+00}, {5.3904e-02, 5.1599e-01, 0.0000e+00}, {1.8099e-02, 2.1390e-01, 0.0000e+00}, {1.4828e-02, 2.1156e-01, 0.0000e+00}, {3.0618e-03, 4.8662e-02, 0.0000e+00}, {4.8675e-04, 8.6038e-03, 0.0000e+00}, {6.0200e-05, 1.1544e-03, 0.0000e+00}, {6.4644e-06, 1.3281e-04, 1.0000e+00}};
static const double uvb_band_heat[UVB_SHIELD_NBANDS][3] = {{1.3363e-01, 0.0000e+00, 0.0000e+00}, {2.3477e-01, 0.0000e+00, 0.0000e+00}, {2.6553e-01, 0.0000e+00, 0.0000e+00}, {1.7055e-01, 1.7071e-01, 0.0000e+00}, {8.0709e-02, 2.3742e-01, 0.0000e+00}, {8.8595e-02, 4.2772e-01, 0.0000e+00}, {2.1553e-02, 1.2954e-01, 0.0000e+00}, {4.0258e-03, 2.9265e-02, 0.0000e+00}, {5.6454e-04, 4.6960e-03, 0.0000e+00}, {6.8520e-05, 6.5206e-04, 1.0000e+00}};

/* UVB rate fractions left behind columns N[3] = (HI, HeI, HeII) [cm^-2]: f_gamma, f_heat per channel (HI, HeI, HeII); exactly 1 at N = 0 */
static inline void sgchem_uvb_shield_factors(const double N[3], double f_gamma[3], double f_heat[3])
{
    double sg[3] = {0, 0, 0}, ng[3] = {0, 0, 0}, sq[3] = {0, 0, 0}, nq[3] = {0, 0, 0};
    for(int k = 0; k < UVB_SHIELD_NBANDS; k++) {
        double att = exp(-(uvb_band_sigma[k][0] * N[0] + uvb_band_sigma[k][1] * N[1] + uvb_band_sigma[k][2] * N[2]));
        for(int c = 0; c < 3; c++) {sg[c] += uvb_band_gamma[k][c] * att; ng[c] += uvb_band_gamma[k][c]; sq[c] += uvb_band_heat[k][c] * att; nq[c] += uvb_band_heat[k][c];}
    }
    for(int c = 0; c < 3; c++) {f_gamma[c] = sg[c] / ng[c]; f_heat[c] = sq[c] / nq[c];}
}

#endif
