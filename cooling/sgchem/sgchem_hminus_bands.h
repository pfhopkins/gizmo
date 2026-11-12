/* Copyright (c) 2026 Ulrich Steinwandel
 * Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
#ifndef SGCHEM_HMINUS_BANDS_H
#define SGCHEM_HMINUS_BANDS_H
/* H- photodetachment by the stellar band fields (SGCHEM_HMINUS_PHOTODETACHMENT_FROM_STELLAR_BANDS):
 * k = c sum_b u_b <sigma/(h nu)>_b over the upstream bins RT_FREQ_BIN_OPTICAL_NIR (0.4133-3.444 eV), RT_FREQ_BIN_NUV less
 * RT_FREQ_BIN_FUV_6_8EV (3.444-6 eV) and FUV_6_8EV + PHOTOELECTRIC (6-13.6 eV, the G0 band), cross-section of Shapiro & Kang 1987
 * (sigma = 7.928e5 (nu - nu0)^1.5 / nu^3 cm^2, h nu0 = 0.755 eV), in-band shape a 3e4 K blackbody. Standalone. */
#include <math.h>

#define SGCHEM_HM_NU0_EV 0.755
#define SGCHEM_HM_BAND_SED_TEMP_K 3.0e4

static inline double sgchem_hminus_sigma(double nu_hz)
{
    const double nu0 = SGCHEM_HM_NU0_EV * 1.602176634e-12 / 6.62607015e-27;
    if(nu_hz <= nu0) {return 0;}
    return 7.928e5 * pow(nu_hz - nu0, 1.5) / (nu_hz * nu_hz * nu_hz);
}

/* <sigma/(h nu)> over [E_lo, E_hi] eV weighted by a blackbody u_nu at T [cm^2 erg^-1] (Simpson, 4000 intervals) */
static inline double sgchem_hminus_band_mean(double E_lo_eV, double E_hi_eV, double T)
{
    const double h = 6.62607015e-27, k = 1.380649e-16, eV = 1.602176634e-12;
    const int n = 4000; double w = 0, a = 0, dnu = (E_hi_eV - E_lo_eV) * eV / h / n;
    for(int i = 0; i <= n; i++) {
        double nu = E_lo_eV * eV / h + i * dnu, f = (i == 0 || i == n) ? 1 : ((i % 2) ? 4 : 2);
        double B = nu * nu * nu / expm1(h * nu / (k * T));
        w += f * B; a += f * B * sgchem_hminus_sigma(nu) / (h * nu);
    }
    return a / w;
}

/* rate [s^-1] from the band energy densities [erg cm^-3]: optical/NIR 0.4133-3.444 eV, NUV 3.444-6 eV, FUV 6-13.6 eV */
static inline double sgchem_hminus_rate_from_bands(double u_opt, double u_nuv, double u_fuv)
{
    static const double a_opt = sgchem_hminus_band_mean(0.4133, 3.444, SGCHEM_HM_BAND_SED_TEMP_K);
    static const double a_nuv = sgchem_hminus_band_mean(3.444, 6.0, SGCHEM_HM_BAND_SED_TEMP_K);
    static const double a_fuv = sgchem_hminus_band_mean(6.0, 13.6, SGCHEM_HM_BAND_SED_TEMP_K);
    return 2.99792458e10 * (u_opt * a_opt + u_nuv * a_nuv + u_fuv * a_fuv);
}
#endif
