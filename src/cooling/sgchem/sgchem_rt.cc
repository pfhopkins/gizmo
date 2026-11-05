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

#ifdef RT_CHEM_PHOTOION
#if CHEMISTRYNETWORK == 5
static double sgchem_rt_r_h2 = 0; /* sigma_H2 per molecule / sigma_HI per atom over the ionising bins */
#endif

/* photon cross-sections as Arepo MRT H2_ionization_crosssection (15.2 eV threshold) and rt_chem.cc */
static double sgchem_rt_sigma_h2(double e)
{
    static const double Epw[12] = {15.20, 15.45, 15.70, 15.95, 16.20, 16.40, 16.65, 16.85, 17.00, 17.20, 17.65, 18.10};
    static const double Spw[11] = {0.09e-18, 1.15e-18, 3.0e-18, 5.0e-18, 6.75e-18, 8.0e-18, 9.0e-18, 9.5e-18, 9.8e-18, 10.1e-18, 9.8e-18};
    if(e < 15.2) {return 0;}
    if(e < 18.1) {for(int k = 0; k < 11; k++) {if(e < Epw[k+1]) {return Spw[k];}}}
    return 9.75e-18 * pow(18.1 / e, 3);
}

/* r over the ionising photons of his 4e4 K blackbody (rt_get_sigma); serial, once */
void sgchem_rt_init(void)
{
#if CHEMISTRYNETWORK == 5
    double s1 = 0, s2 = 0, kT = 4.0e4 * BOLTZMANN_CGS / ELECTRONVOLT_IN_ERGS, de = (500. - 13.6) / 19999.;
    for(int j = 1; j < 20000; j++) {
        double e = 13.6 + j * de, x = e / kT, w = (x < 500.) ? e * e / expm1(x) : 0, f = sqrt(e / 13.6 - 1.);
        s1 += w * 6.3e-18 * pow(13.6 / e, 4) * exp(4. - 4. * atan(f) / f) / (1. - exp(-2. * M_PI / f));
        s2 += w * sgchem_rt_sigma_h2(e);
    }
    sgchem_rt_r_h2 = s2 / s1;
    if(ThisTask == 0) {printf("SGCHEM: ionising photons on H2, sigma_H2 / sigma_HI = %.3f\n", sgchem_rt_r_h2);}
#endif
}

/* rates per atom [1/s] and heating [erg/s per atom] over all ionising bins, as CoolingRate; H2 by the ratio used in the absorbers below */
void sgchem_rt_ionising_rates(int i, struct ChemInputs *in)
{
    for(int k = 0; k < N_RT_FREQ_BINS; k++) {
        if(!RT_BAND_IS_IONIZING(k)) {continue;}
        double f = C_LIGHT_CGS / UNIT_LENGTH_IN_CGS * rt_ionizing_band_photon_number_density(i, k, CellP), g = f * All.rt_ion_sigma_HI[k];
        in->rt_phot_HI += g; in->rt_heat_HI += g * All.rt_ion_G_HI[k] * UNIT_ENERGY_IN_CGS;
#if CHEMISTRYNETWORK == 5
        in->rt_phot_H2 += sgchem_rt_r_h2 * g;
        in->rt_heat_H2 += sgchem_rt_r_h2 * g * DMAX(All.rt_nu_eff_eV[k] - 15.2, 0.) * ELECTRONVOLT_IN_ERGS;
#endif
#ifdef RT_CHEM_PHOTOION_HE
        in->rt_phot_HeI += f * All.rt_ion_sigma_HeI[k]; in->rt_heat_HeI += f * All.rt_ion_sigma_HeI[k] * All.rt_ion_G_HeI[k] * UNIT_ENERGY_IN_CGS;
        in->rt_phot_HeII += f * All.rt_ion_sigma_HeII[k]; in->rt_heat_HeII += f * All.rt_ion_sigma_HeII[k] * All.rt_ion_G_HeII[k] * UNIT_ENERGY_IN_CGS;
#endif
    }
}

/* the absorbers rt_kappa reads: what the network consumes photons on. Net 5: atomic H plus H2 by the same ratio as its rate,
   no He species, so the He fields stay empty and helium bins act on H. Net 17: atomic H, its He. */
void sgchem_rt_absorbers(int i, const double *ab)
{
    double xHI = DMAX(0, 1. - 2. * ab[IH2] - ab[IHP]);
    CellP[i].HII = ab[IHP];
#if CHEMISTRYNETWORK == 5
    CellP[i].HI = xHI + sgchem_rt_r_h2 * ab[IH2];
#ifdef RT_CHEM_PHOTOION_HE
    CellP[i].HeI = CellP[i].HeII = CellP[i].HeIII = 0;
#endif
#else
    CellP[i].HI = xHI;
#ifdef RT_CHEM_PHOTOION_HE
    CellP[i].HeII = ab[IHEP]; CellP[i].HeIII = ab[IHEPP]; CellP[i].HeI = DMAX(0, ABHE - ab[IHEP] - ab[IHEPP]);
#endif
#endif
}
#endif /* RT_CHEM_PHOTOION */

#endif /* SGCHEM && RADTRANSFER */
