/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* SGChem reads the moment solver's bins (GALSF_SINGLE_STARS_RT_TABLE_BANDS): G0, G0_LW and G0_dust; the band law matches
   rt_kappa, so the transported field arrives attenuated. Host only, per cell, no global state. */
#include <math.h>
#include "../../declarations/allvars.h"
#include "../../core/proto.h"
#include "chem_interface.h"
#include "sgchem_network.h"
#include "../../gravity/gravtree_treecol_pixels.h"
#include "../../radiation/rt_functions.h"

#if defined(SGCHEM) && defined(RADTRANSFER)

/* energy density of bin k [erg/cm^3] */
static inline double sgchem_rt_band_u_cgs(int i, int k) {return CellP[i].Rad_E_gamma[k] * (CellP[i].Density * All.cf_a3inv / P[i].Mass) * UNIT_PRESSURE_IN_CGS;}

/* the stellar field of cell i, as sgchem_treecol_field with the bins in place of the pixels: the part along the M1 flux goes
   to the pixel facing the source, the rest is isotropic; pixels are floored at 0.324e-2 SGChemBackgroundG0 / NPIX Habing,
   only floor pixels take the dust column (pix_att) */
struct ChemFUVField sgchem_rt_field(int i, double *fac_uv, double *pix_att)
{
    const int k_uv[3] = {RT_FREQ_BIN_GENERIC_USER_FREQ, RT_FREQ_BIN_PHOTOELECTRIC, RT_FREQ_BIN_LYMAN_WERNER}; /* disjoint 6-8, 8-11.2, 11.2-13.6 eV */
    double u_uv = 0, E_uv = 0, f_beam = 0; int pix_src = 0; Vec3<double> F = {};
    for(int k = 0; k < 3; k++) {u_uv += sgchem_rt_band_u_cgs(i, k_uv[k]); E_uv += CellP[i].Rad_E_gamma[k_uv[k]]; F += CellP[i].Rad_Flux[k_uv[k]];}
    double F_mag = sqrt(F.norm_sq());
    if(F_mag > 0 && E_uv > 0) {
        f_beam = DMIN(1., F_mag / (C_LIGHT_CODE_REDUCED * E_uv));
        pix_src = treecol_hpx_pixel(F * (-1.)); /* radiation flows away from its source */
    }
    double g0 = u_uv / HABING_EGY_DENSITY_6_13P6EV_CGS, g0_lw = sgchem_rt_band_u_cgs(i, RT_FREQ_BIN_LYMAN_WERNER) / HABING_EGY_DENSITY_11P2_13P6EV_CGS;
    double bg = 0.324e-2 * All.SGChemBackgroundG0, floor_pix = bg / NPIX, floor_lw = 0.1 * floor_pix * HABING_EGY_DENSITY_6_13P6EV_CGS / HABING_EGY_DENSITY_11P2_13P6EV_CGS;
    double g0_tot = 0, g0_lw_tot = 0, g_pix[NPIX];
    for(int k = 0; k < NPIX; k++) {
        double w = (1. - f_beam) / NPIX + ((k == pix_src) ? f_beam : 0.), s = g0 * w;
        pix_att[k] = (s < floor_pix) ? 1. : 0.;
        g_pix[k] = DMAX(s, floor_pix); g0_tot += g_pix[k];
        g0_lw_tot += DMAX(g0_lw * w, floor_lw);
    }
    for(int k = 0; k < NPIX; k++) {fac_uv[k] = (g0_tot > 0) ? g_pix[k] / g0_tot : 1. / NPIX;}
    struct ChemFUVField f = {g0_tot, g0_lw_tot, 0};
    f.G0_dust = chem_grain_heating_g0(g0_tot, sgchem_rt_band_u_cgs(i, RT_FREQ_BIN_NUV), sgchem_rt_band_u_cgs(i, RT_FREQ_BIN_OPTICAL_NIR), 1, 1, bg);
    return f;
}

#endif /* SGCHEM && RADTRANSFER */
