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
    const int k_uv[2] = {RT_FREQ_BIN_NUV, RT_FREQ_BIN_PHOTOELECTRIC}; /* 6-8 and 8-13.6 eV; the Lyman-Werner bin lies inside the second */
    double u_uv = 0, E_uv = 0, f_beam = 0; int pix_src = 0; Vec3<double> F = {};
    for(int k = 0; k < 2; k++) {u_uv += sgchem_rt_band_u_cgs(i, k_uv[k]); E_uv += CellP[i].Rad_E_gamma[k_uv[k]]; F += CellP[i].Rad_Flux[k_uv[k]];}
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
#ifdef RT_OPTICAL_NIR
    f.G0_dust = chem_grain_heating_g0(g0_tot, 0, sgchem_rt_band_u_cgs(i, RT_FREQ_BIN_OPTICAL_NIR), 0, 1, bg); /* 3.4-6 eV ISRF-shaped */
#else
    f.G0_dust = g0_tot; /* ISRF-shaped */
#endif
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

#ifdef RT_CHEM_PHOTOION_HE
#define SGCHEM_RT_H0_TOP_EV 24.6 /* rt_chem.cc nu_vec */
#else
#define SGCHEM_RT_H0_TOP_EV 500.
#endif
/* r over the H0 bin's photons of his 4e4 K blackbody (rt_get_sigma); serial, once */
void sgchem_rt_init(void)
{
#if CHEMISTRYNETWORK == 5
    double s1 = 0, s2 = 0, kT = 4.0e4 * BOLTZMANN_CGS / ELECTRONVOLT_IN_ERGS, de = (SGCHEM_RT_H0_TOP_EV - 13.6) / 19999.;
    for(int j = 1; j < 20000; j++) {
        double e = 13.6 + j * de, x = e / kT, w = (x < 500.) ? e * e / expm1(x) : 0, f = sqrt(e / 13.6 - 1.);
        s1 += w * 6.3e-18 * pow(13.6 / e, 4) * exp(4. - 4. * atan(f) / f) / (1. - exp(-2. * M_PI / f));
        s2 += w * sgchem_rt_sigma_h2(e);
    }
    sgchem_rt_r_h2 = s2 / s1;
    if(ThisTask == 0) {printf("SGCHEM: ionising photons on H2, sigma_H2 / sigma_HI = %.3f\n", sgchem_rt_r_h2);}
#endif
}

/* rates per atom [1/s] and heating [erg/s per atom], as stella_dev: H and H2 from the H0 bin, HeI from He0, HeII from He1, heat = rate x
   (mean photon energy - threshold). With SGCHEM_RT_ABSORBERS_FOLLOW_THE_NETWORK every ionising bin counts, as CoolingRate */
static inline double sgchem_rt_gamma(int i, int k, double sigma) {return C_LIGHT_CGS / UNIT_LENGTH_IN_CGS * rt_ionizing_band_photon_number_density(i, k, CellP) * sigma;}
void sgchem_rt_ionising_rates(int i, struct ChemInputs *in)
{
#ifdef SGCHEM_RT_ABSORBERS_FOLLOW_THE_NETWORK
    for(int k = 0; k < N_RT_FREQ_BINS; k++) {
        if(!RT_BAND_IS_IONIZING(k)) {continue;}
        double g = sgchem_rt_gamma(i, k, All.rt_ion_sigma_HI[k]);
        in->rt_phot_HI += g; in->rt_heat_HI += g * All.rt_ion_G_HI[k] * UNIT_ENERGY_IN_CGS;
#if CHEMISTRYNETWORK == 5
        in->rt_phot_H2 += sgchem_rt_r_h2 * g;
        in->rt_heat_H2 += sgchem_rt_r_h2 * g * DMAX(All.rt_nu_eff_eV[k] - 15.2, 0.) * ELECTRONVOLT_IN_ERGS;
#endif
#ifdef RT_CHEM_PHOTOION_HE
        double ge = sgchem_rt_gamma(i, k, All.rt_ion_sigma_HeI[k]), gi = sgchem_rt_gamma(i, k, All.rt_ion_sigma_HeII[k]);
        in->rt_phot_HeI += ge; in->rt_heat_HeI += ge * All.rt_ion_G_HeI[k] * UNIT_ENERGY_IN_CGS;
        in->rt_phot_HeII += gi; in->rt_heat_HeII += gi * All.rt_ion_G_HeII[k] * UNIT_ENERGY_IN_CGS;
#endif
    }
#else
    double g = sgchem_rt_gamma(i, RT_FREQ_BIN_H0, All.rt_ion_sigma_HI[RT_FREQ_BIN_H0]), nu = All.rt_nu_eff_eV[RT_FREQ_BIN_H0];
    in->rt_phot_HI = g; in->rt_heat_HI = g * (nu - 13.6) * ELECTRONVOLT_IN_ERGS;
#if CHEMISTRYNETWORK == 5
    in->rt_phot_H2 = sgchem_rt_r_h2 * g; in->rt_heat_H2 = sgchem_rt_r_h2 * g * DMAX(nu - 15.2, 0.) * ELECTRONVOLT_IN_ERGS;
#endif
#ifdef RT_PHOTOION_MULTIFREQUENCY
    in->rt_phot_HeI = sgchem_rt_gamma(i, RT_FREQ_BIN_He0, All.rt_ion_sigma_HeI[RT_FREQ_BIN_He0]);
    in->rt_heat_HeI = in->rt_phot_HeI * (All.rt_nu_eff_eV[RT_FREQ_BIN_He0] - 24.6) * ELECTRONVOLT_IN_ERGS;
    in->rt_phot_HeII = sgchem_rt_gamma(i, RT_FREQ_BIN_He1, All.rt_ion_sigma_HeII[RT_FREQ_BIN_He1]);
    in->rt_heat_HeII = in->rt_phot_HeII * (All.rt_nu_eff_eV[RT_FREQ_BIN_He1] - 54.4) * ELECTRONVOLT_IN_ERGS;
#endif
#endif
}

/* what rt_kappa reads, as stella_dev: atomic H. With SGCHEM_RT_ABSORBERS_FOLLOW_THE_NETWORK what the network consumes photons on:
   net 5 atomic H plus H2 by the same ratio as its rate and no He, net 17 its He */
void sgchem_rt_absorbers(int i, const double *ab)
{
    double xHI = DMAX(0, 1. - 2. * ab[IH2] - ab[IHP]);
    CellP[i].HII = ab[IHP];
    CellP[i].HI = xHI;
#ifdef SGCHEM_RT_ABSORBERS_FOLLOW_THE_NETWORK
#if CHEMISTRYNETWORK == 5
    CellP[i].HI = xHI + sgchem_rt_r_h2 * ab[IH2];
#ifdef RT_CHEM_PHOTOION_HE
    CellP[i].HeI = CellP[i].HeII = CellP[i].HeIII = 0;
#endif
#elif defined(RT_CHEM_PHOTOION_HE)
    CellP[i].HeII = ab[IHEP]; CellP[i].HeIII = ab[IHEPP]; CellP[i].HeI = DMAX(0, ABHE - ab[IHEP] - ab[IHEPP]);
#endif
#endif
}
#endif /* RT_CHEM_PHOTOION */

#ifdef RT_INFRARED
/* the gas's radiative losses of this step go to the infrared bin; the solver's balance sets the dust temperature. Gas-dust exchange
   (0) is in rt_ir_lambdadust, CR and photoelectric heating (10, 11) are no cooling */
void sgchem_rt_cooling_to_infrared(int i, const double *lambda, double temp)
{
    double nH = CellP[i].nHcgs(), L = 0;
    for(int k = 0; k < 28; k++) {if(k != 0 && k != 10 && k != 11 && isfinite(lambda[k])) {L += lambda[k];}}
    CellP[i].Lambda_RadiativeCooling_toRHDBins[RT_FREQ_BIN_INFRARED] = L / (nH * nH);
    rt_ir_lambdadust(i, temp, P, CellP);
    CellP[i].DustTemp = CellP[i].Dust_Temperature;
}
#endif

#endif /* SGCHEM && RADTRANSFER */
