/* C++ implementation: Ulrich Steinwandel (uli@mpa-garching.mpg.de).
   Based on the SGChem Fortran networks by Simon Glover (GNU GPL, as distributed in Arepo). */
/* SGChem in GIZMO: one gas cell over one cooling step (inputs from the cell, CVODE solve, state back), and the
   DoCooling branch. Host only: the solver scratch is thread_local, each cell's result depends only on its inputs. */
#include "sgchem_hminus_bands.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "../../declarations/allvars.h"
#include "../cooling.h"
#include "../../core/proto.h"
#include "chem_interface.h"
#include "sgchem_network.h"
#include "sgchem_dust_to_gas.h"
#include "../../gravity/gravtree_treecol_pixels.h"
#ifdef COSMIC_RAY_FLUID
#include "../../eos/cosmic_ray_fluid/cosmic_ray_functions.h"
#endif
#if CHEMISTRYNETWORK == 18
#include "cxx/chem_net18_layout.h"
#endif

#ifdef SGCHEM

/* element pools per H from the FIRE-3 metal slots (0 Z, 1 He, 2 C, 3 N, 4 O, 5 Ne, 6 Mg, 7 Si, 8 S, 9 Ca, 10 Fe) */
static void sgchem_gas_abundances(int i, struct ChemInputs *in)
{
    const MyFloat *Z = P[i].Metallicity;
    double X_H = 1.0 / (1.0 + 4.0 * ABHE); /* the H reference of yn below */
#ifdef GALSF_ISMDUSTCHEM_MODEL
    const MyDouble *D = CellP[i].ISMDustChem_Dust_Metal; /* gas phase = metals minus the live dust */
    in->abundc  = DMAX(Z[2] - D[2], 0) / 12.0 / X_H;
    in->abundo  = DMAX(Z[4] - D[4], 0) / 16.0 / X_H;
    in->abundsi = DMAX(Z[7] - D[7], 0) / 28.0 / X_H;
#else
    double g_dep = DMAX(All.SGChemDustToGasRatio, 0);
    in->abundc  = (1 - DMIN(SGCHEM_DEPL_C  * g_dep, SGCHEM_DEPL_MAX)) * (Z[2] / 12.0) / X_H;
    in->abundo  = (1 - DMIN(SGCHEM_DEPL_O  * g_dep, SGCHEM_DEPL_MAX)) * (Z[4] / 16.0) / X_H;
    in->abundsi = (1 - DMIN(SGCHEM_DEPL_SI * g_dep, SGCHEM_DEPL_MAX)) * (Z[7] / 28.0) / X_H;
#endif
    in->abundN  = (Z[3] / 14.0) / X_H;
#if CHEMISTRYNETWORK == 18
    in->abundNe = (Z[5] / 20.0) / X_H; in->abundS = (Z[8] / 32.0) / X_H;
#ifdef GALSF_ISMDUSTCHEM_MODEL
    in->abundMg = DMAX(Z[6] - D[6], 0) / 24.0 / X_H; in->abundFe = DMAX(Z[10] - D[10], 0) / 56.0 / X_H;
#else
    in->abundMg = (Z[6] / 24.0) / X_H; in->abundFe = (Z[10] / 56.0) / X_H;
#endif
#endif
    double zm[12] = {Z[1], Z[2], Z[6], Z[4], Z[10], Z[7], 1.0 - Z[0] - Z[1], Z[3], Z[5], Z[8], Z[9], 0.0}; /* He C Mg O Fe Si H N Ne S Ca Zn (Zn not tracked) */
    for(int k = 0; k < 12; k++) {in->Zmass[k] = zm[k];}
    double Zmax = 10.0 * SGCHEM_SOLAR_Z, Zcell = Z[0]; /* solver inputs capped at 10 Zsun (the certified domain); stored metals untouched */
    if(Zcell > Zmax) {
        double s = Zmax / Zcell;
        in->abundc *= s; in->abundo *= s; in->abundsi *= s; in->abundN *= s;
#if CHEMISTRYNETWORK == 18
        in->abundNe *= s; in->abundMg *= s; in->abundS *= s; in->abundFe *= s;
#endif
        for(int k = 1; k < 12; k++) {if(k != 6) {in->Zmass[k] *= s;}}
    }
    in->abundM = SGCHEM_ABUNDM_SOLAR * DMIN(Zcell, Zmax) / SGCHEM_SOLAR_Z;
#if CHEMISTRYNETWORK == 16
    in->abundM = in->abundsi; /* Gong+17: M is silicon */
#endif
}

#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS
/* the gravity walk's pixel columns (dust-weighted, H2, CO) as chemistry inputs [cm^-2]; dust columns divided by the local
   dust-to-gas ratio (the chemistry multiplies it back), zero where there is no dust */
static void sgchem_treecol_columns(int i, struct ChemInputs *in)
{
    double to_cgs = UNIT_DENSITY_IN_CGS * UNIT_LENGTH_IN_CGS * All.cf_a2inv; /* code surface density to g cm^-2 */
    for(int k = 0; k < NPIX; k++)
    {
        double col_dust = TREECOL_PIXELS_COLUMN(CellP[i], TREECOL_PIXELS_DUST, k) * to_cgs / ((1.0 + 4.0 * ABHE) * PROTONMASS_CGS);
        in->column_density_projection[k] = (in->dust_to_gas_ratio > 1.0e-10) ? col_dust / in->dust_to_gas_ratio : 0;
        in->column_density_projection_h2[k] = TREECOL_PIXELS_COLUMN(CellP[i], TREECOL_PIXELS_H2, k) * to_cgs / (2.0 * PROTONMASS_CGS);
        in->column_density_projection_co[k] = TREECOL_PIXELS_COLUMN(CellP[i], TREECOL_PIXELS_CO, k) * to_cgs / (28.0 * PROTONMASS_CGS);
    }
}
#endif

#ifdef SGCHEM_UVB_TREECOL_SHIELDING
#include "sgchem_uvb_shield.h"

/* pixel mean of the UVB fractions; absorber columns = TreeCol gas column x this cell's neutral H, He0, He+ fractions */
static void sgchem_uvb_treecol_shielding(int i, double f_gamma[3], double f_heat[3])
{
    const double h_per_mass = 1.0 / ((1.0 + 4.0 * ABHE) * PROTONMASS_CGS);
    double x_hep = 0, x_hepp = 0;
#if (IHEP != UNUSED)
    x_hep = CellP[i].TracAbund[IHEP];
#endif
#if defined(IHEPP) && (IHEPP != UNUSED)
    x_hepp = CellP[i].TracAbund[IHEPP];
#endif
    double x[3] = {DMAX(1.0 - CellP[i].TracAbund[IHP], 0.0), DMAX(ABHE - x_hep - x_hepp, 0.0), DMAX(x_hep, 0.0)};
    for(int c = 0; c < 3; c++) {f_gamma[c] = f_heat[c] = 0;}
    for(int p = 0; p < TREECOL_PIXELS_N; p++) {
        double NH = TREECOL_PIXELS_COLUMN(CellP[i], TREECOL_PIXELS_GAS, p) * UNIT_SURFDEN_IN_CGS * All.cf_a2inv * h_per_mass;
        double N[3] = {x[0] * NH, x[1] * NH, x[2] * NH}, fg[3], fq[3];
        sgchem_uvb_shield_factors(N, fg, fq);
        for(int c = 0; c < 3; c++) {f_gamma[c] += fg[c]; f_heat[c] += fq[c];}
    }
    for(int c = 0; c < 3; c++) {f_gamma[c] /= TREECOL_PIXELS_N; f_heat[c] /= TREECOL_PIXELS_N;}
}
#endif

#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
/* the unattenuated stellar field of cell i from the gravity walk: each 6-13.6 eV pixel floored at
   0.324e-2 SGChemBackgroundG0 / NPIX Habing, the pixel weights fac_uv, G0 / G0_LW / band-resolved G0_dust */
struct ChemFUVField sgchem_treecol_field(int i, double *fac_uv)
{
    double floor_pix = 0.324e-2 * All.SGChemBackgroundG0 / NPIX * HABING_EGY_DENSITY_6_13P6EV_CGS, u_uv = 0, u_pix[NPIX];
    for(int k = 0; k < NPIX; k++) {u_pix[k] = DMAX(TREECOL_PIXELS_EGY_UV(CellP[i], k), floor_pix); u_uv += u_pix[k];}
    for(int k = 0; k < NPIX; k++) {fac_uv[k] = (u_uv > 0) ? u_pix[k] / u_uv : 1.0 / NPIX;}
    return chem_fuv_field_from_band_energy(u_uv, HABING_EGY_DENSITY_6_13P6EV_CGS, CellP[i].TreeColEgyLW, CellP[i].TreeColEgyNUV,
                                           CellP[i].TreeColEgyOPT, 1, 1, 0.324e-2 * All.SGChemBackgroundG0);
}
#endif

#ifdef OUTPUT_SGCHEM_RADIATION_FIELDS
/* snapshot value of a radiation block: cells from the pixel fields of the last walk, nothing stored for it but the shielding
   factor of the last solve; stars log10 [erg/s] (-100 when dark) */
double sgchem_radiation_field_for_snapshot(int i, int blocknr)
{
    if(blocknr == IO_SINGLE_STAR_LOG_LYMAN_Q) {double q = single_star_ionizing_photon_rate(i); return (q > 0) ? log10(q) : -100.0;} /* log10 photons/s */
    if(blocknr == IO_SINGLE_STAR_LOG_UV_LUM || blocknr == IO_SINGLE_STAR_LOG_LW_LUM)
    {
        double L[SINGLE_STAR_NBANDS], L_bol;
        if(!single_star_band_luminosities_cgs(i, L, &L_bol)) {return -100.0;}
        double l = (blocknr == IO_SINGLE_STAR_LOG_LW_LUM) ? L[SINGLE_STAR_BAND_LW] : L[SINGLE_STAR_BAND_PE] + L[SINGLE_STAR_BAND_68];
        return (l > 0) ? log10(l) : -100.0;
    }
    double fac[NPIX]; struct ChemFUVField f = sgchem_treecol_field(i, fac);
    switch(blocknr)
    {
        case IO_SGCHEM_G0_FUV: return f.G0;
        case IO_SGCHEM_G0_LW: return f.G0_LW;
        case IO_SGCHEM_G0_FUV_ATTEN: return f.G0 * CellP[i].ShieldFacDust;
        case IO_SGCHEM_SHIELDFAC_DUST: return CellP[i].ShieldFacDust;
        case IO_SGCHEM_G0_NUV: return CellP[i].TreeColEgyNUV / HABING_EGY_DENSITY_6_13P6EV_CGS;
        case IO_SGCHEM_G0_OPT: return CellP[i].TreeColEgyOPT / HABING_EGY_DENSITY_6_13P6EV_CGS;
        case IO_SGCHEM_CR_ZETA: return sgchem_cr_ion_rate(i, f.G0);
        default: return 0;
    }
}
#endif

#ifdef ISRF_AND_CR_RATE_SCALE_WITH_RECENT_SFR
double sgchem_recent_sfr_g0(void) {return DMAX(All.FactorG0 * All.SGChemG0, 0.324e-2);} /* G0 = G0_LW = G0_dust */
#endif

/* cosmic-ray ionisation rate per H [s^-1] of cell i under the field G0 [Habing] */
double sgchem_cr_ion_rate(int i, double G0)
{
#ifdef COSMIC_RAY_FLUID
    return Get_CosmicRayIonizationRate_cgs(i, P, CellP); /* the rate from the CR fluid (Cummings+16), radioactive terms included */
#endif
    double zeta = All.SGChemCosmicRayIonRate;
#if defined(CR_SCALE_WITH_G0) && defined(ISRF_AND_CR_RATE_SCALE_WITH_RECENT_SFR)
    zeta = All.FactorG0 * zeta; /* unclamped */
#elif defined(CR_SCALE_WITH_G0)
    zeta = DMAX(1e-21, DMIN(2e-16, (G0 / 1.7) * zeta)); /* with the stars' field */
#endif
#ifdef CR_BACKGROUND_IONRATE_ATTENUATED
    zeta = All.SGChemCosmicRayIonRate * CellP[i].TreeColCRAttenuation; /* replaces the scaling */
#endif
    return zeta;
}

#ifdef COSMIC_RAY_FLUID
/* the solve's CR ionisation heating de_heat [code energy], minus the radioactive share of zeta (those terms), comes out of
   the CR energy of cell i, capped at it */
static void sgchem_cr_ionisation_from_fluid(int i, const struct ChemInputs *in, double de_heat)
{
    double zeta_radio = 0;
#ifdef METALS
    zeta_radio += 1.e-21 * P[i].Metallicity[0] / All.SolarAbundances[0];
#if (NUM_METAL_SPECIES > 10)
    zeta_radio += 1.e-19 * P[i].Metallicity[10] / All.SolarAbundances[10];
#endif
#endif
    double f_cr = (in->cosmic_ray_ion_rate > 0) ? DMIN(DMAX(1. - zeta_radio / in->cosmic_ray_ion_rate, 0.), 1.) : 0;
    double de = f_cr * DMAX(de_heat, 0), ecr = 0;
    for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {ecr += CellP[i].CosmicRayEnergy[k];}
    if(!(de > 0 && ecr > 0)) {return;}
    if(de > ecr) {static long n_capped = 0; if(n_capped++ == 0) {printf("SGCHEM: CR ionisation heating %g exceeds e_CR %g (ID %llu); capped at e_CR (first of possibly more)\n", de, ecr, (unsigned long long) P[i].ID);} de = ecr;}
    double q = 1. - de / ecr;
    for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {CellP[i].CosmicRayEnergy[k] *= q; CellP[i].CosmicRayEnergyPred[k] *= q; CellP[i].CosmicRayFlux[k] *= q; CellP[i].CosmicRayFluxPred[k] *= q;}
#ifdef OUTPUT_MASS_METAL_ENERGY_LEDGER
    ledger_add(LEDGER_E_CR_IONISATION_FROM_FLUID, de); ledger_add(LEDGER_E_CHEM_CR_IONISATION, DMAX(de_heat, 0));
#endif
}
#endif

#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS_PHOTOION
/* rates from the swept fluxes: sigma_HI F per atom, with the excess over threshold as heat; a cell thick to ionising photons
   takes only those entering it, (1 - e^-tau) / tau (Abel & Wandelt 99): the tree has no source depletion, so without the
   limiter an over-ionised front cell strips its own shielding and the front runs away */
static void sgchem_treecol_photoion_rates(int i, struct ChemInputs *in, double rho_cgs)
{
    const double n_h = HYDROGEN_MASSFRAC * rho_cgs / PROTONMASS_CGS, l_cell = pow(P[i].Mass * UNIT_MASS_IN_CGS / rho_cgs, 1.0 / 3.0);
#ifdef TREECOL_SHELLS_ION_HELIUM
    const MyFloat *ab = CellP[i].TracAbund; const float *ph = CellP[i].TreeRayIonPhotonFlux, *en = CellP[i].TreeRayIonEnergyFlux;
    const double n_hi = DMAX(0.0, 1.0 - ab[IHP]) * n_h, n_hei = DMAX(0.0, ABHE - ab[IHEP] - ab[IHEPP]) * n_h, E_H = 13.6 * ELECTRONVOLT_IN_ERGS, E_He = 24.6 * ELECTRONVOLT_IN_ERGS;
    const double t0 = TREECOL_SHELLS_SIGMA_HI_B0 * n_hi * l_cell, t1 = (TREECOL_SHELLS_SIGMA_HI_B1 * n_hi + TREECOL_SHELLS_SIGMA_HEI_B1 * n_hei) * l_cell;
    const double f0 = (t0 > 1.0e-8) ? (1.0 - exp(-t0)) / t0 : 1.0, f1 = (t1 > 1.0e-8) ? (1.0 - exp(-t1)) / t1 : 1.0;
    in->rt_phot_HI = TREECOL_SHELLS_SIGMA_HI_B0 * f0 * ph[0] + TREECOL_SHELLS_SIGMA_HI_B1 * f1 * ph[1];
    in->rt_heat_HI = TREECOL_SHELLS_SIGMA_HI_B0 * f0 * DMAX(en[0] - E_H * ph[0], 0.0) + TREECOL_SHELLS_SIGMA_HI_B1 * f1 * DMAX(en[1] - E_H * ph[1], 0.0);
    in->rt_phot_HeI = TREECOL_SHELLS_SIGMA_HEI_B1 * f1 * ph[1]; in->rt_heat_HeI = TREECOL_SHELLS_SIGMA_HEI_B1 * f1 * DMAX(en[1] - E_He * ph[1], 0.0);
#else
    const double F_ph = CellP[i].TreeRayIonPhotonFlux[0], F_en = CellP[i].TreeRayIonEnergyFlux[0], sig = TREECOL_SHELLS_SIGMA_HI, x_h2 = CellP[i].TracAbund[IH2];
    if(F_ph <= 0) {return;}
    const double n_abs = (DMAX(0.0, 1.0 - CellP[i].TracAbund[IHP] - 2.0 * x_h2) + TREECOL_SHELLS_H2_PER_HI * x_h2) * n_h, tau = sig * n_abs * l_cell;
    const double f = sig * ((tau > 1.0e-8) ? (1.0 - exp(-tau)) / tau : 1.0);
    in->rt_phot_HI = f * F_ph; in->rt_heat_HI = f * DMAX(F_en - 13.6 * ELECTRONVOLT_IN_ERGS * F_ph, 0.0);
    in->rt_phot_H2 = TREECOL_SHELLS_H2_PER_HI * in->rt_phot_HI;
    in->rt_heat_H2 = TREECOL_SHELLS_H2_PER_HI * f * DMAX(F_en - 15.2 * ELECTRONVOLT_IN_ERGS * F_ph, 0.0);
#endif
}
#endif

static thread_local int sgchem_network_skipped = 0; /* the last cell step ran no network (photoionised or test pin) */

/* evolves cell i from specific energy u_code over dt_code [code units]; writes TracAbund, Temp, DustTemp, Ne; returns the new u */
static double sgchem_cell_step(int i, double u_code, double dt_code)
{
    struct ChemInputs in = {};
    struct ChemOutputs out;
    double rho = CellP[i].Density * All.cf_a3inv, divv = 0;
    for(int k = 0; k < 3; k++) {divv += CellP[i].Gradients.Velocity[k][k];}
    divv *= All.cf_a2inv;
#if CHEMISTRYNETWORK == 20
    {double g = 0; for(int k = 0; k < 3; k++) {g += fabs(CellP[i].Gradients.Velocity[k][k] * All.cf_a2inv);} in.dvdr_cgs = g / 3.0 * UNIT_VEL_IN_CGS / UNIT_LENGTH_IN_CGS;}
#endif
    sgchem_gas_abundances(i, &in);
    in.abundD = All.SGChemDeuteriumAbundance;
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
    struct ChemFUVField fuv = sgchem_treecol_field(i, in.fac_uv); /* the stars' field from the gravity walk */
#ifdef RT_TREECOL_STELLAR_PIXELS_PREATTENUATED
    for(int k = 0; k < NPIX; k++) {in.pix_att[k] = (TREECOL_PIXELS_EGY_UV(CellP[i], k) < 0.324e-2 * All.SGChemBackgroundG0 / NPIX * HABING_EGY_DENSITY_6_13P6EV_CGS) ? 1. : 0.;}
#endif
#elif defined(RADTRANSFER)
    struct ChemFUVField fuv = sgchem_rt_field(i, in.fac_uv, in.pix_att); /* the stars' field from the moment bins */
#elif defined(ISRF_AND_CR_RATE_SCALE_WITH_RECENT_SFR)
    struct ChemFUVField fuv; fuv.G0 = fuv.G0_LW = fuv.G0_dust = sgchem_recent_sfr_g0(); /* uniform, scaled with the recent SFR */
#else
    struct ChemFUVField fuv = sgchem_uniform_field(All.SGChemG0, All.SGChemG0_LW, All.SGChemBackgroundG0); /* uniform, unattenuated */
#endif
    double uvb[6] = {0, 0, 0, 0, 0, 0}; /* UV background (gJH0, gJHe0, gJHep, epsH0, epsHe0, epsHep), comoving runs only */
    if(All.ComovingIntegrationOn) {
        get_uvb_rates(&uvb[0], &uvb[1], &uvb[2], &uvb[3], &uvb[4], &uvb[5]);
        double G0_uvb = uvb[0] / 2.29e-10, G0_old = fuv.G0; /* Gamma_HI as a Habing field; the LW share is 10 % */
        if(fuv.G0 < G0_uvb) {fuv.G0 = G0_uvb;}
        if(fuv.G0_LW < 0.1 * G0_uvb) {fuv.G0_LW = 0.1 * G0_uvb;}
        fuv.G0_dust += fuv.G0 - G0_old; /* a raise counts ISRF-shaped */
        in.redshift = 1. / All.Time - 1.;
    }
    in.G0 = fuv.G0; in.G0_LW = fuv.G0_LW; in.G0_dust = fuv.G0_dust;
#ifdef SGCHEM_HMINUS_PHOTODETACHMENT_FROM_STELLAR_BANDS
    { /* stellar 6-13.6 eV = pixel energy above the floor; NUV (3.444-6 eV) and optical (0.4133-3.444 eV) are stellar only */
        double floor_pix = 0.324e-2 * All.SGChemBackgroundG0 / NPIX * HABING_EGY_DENSITY_6_13P6EV_CGS, u_fuv_star = 0;
        for(int k = 0; k < NPIX; k++) {u_fuv_star += DMAX(TREECOL_PIXELS_EGY_UV(CellP[i], k) - floor_pix, 0.);}
        in.hm_detach_stellar = sgchem_hminus_rate_from_bands(CellP[i].TreeColEgyOPT, CellP[i].TreeColEgyNUV, u_fuv_star);
        in.G0_hm_isrf = DMAX(in.G0 - u_fuv_star / HABING_EGY_DENSITY_6_13P6EV_CGS, 0.);
    }
#endif
#if defined(RADTRANSFER) && defined(RT_CHEM_PHOTOION)
    sgchem_rt_ionising_rates(i, &in);
#endif
    in.cosmic_ray_ion_rate = sgchem_cr_ion_rate(i, fuv.G0);
#ifdef COSMIC_RAY_FLUID
    in.cr_energy_density = Get_CosmicRayEnergyDensity_cgs(i, P, CellP); /* hadronic and Coulomb heating by the CR fluid */
#endif
    in.id_current = (int) P[i].ID;
    in.dust_to_gas_ratio = sgchem_cell_dust_to_gas_ratio(P, CellP, i);
#if CHEMISTRYNETWORK == 18 && !defined(GALSF_ISMDUSTCHEM_MODEL)
    {double yd = DMIN(DMAX(in.dust_to_gas_ratio, 0), 1); /* RTZ depletion scaled with the DGR */
     in.abundc *= 1 - (1 - n18_depl[2]) * yd; in.abundN *= 1 - (1 - n18_depl[3]) * yd; in.abundo *= 1 - (1 - n18_depl[4]) * yd; in.abundNe *= 1 - (1 - n18_depl[5]) * yd;
     in.abundMg *= 1 - (1 - n18_depl[6]) * yd; in.abundsi *= 1 - (1 - n18_depl[7]) * yd; in.abundS *= 1 - (1 - n18_depl[8]) * yd; in.abundFe *= 1 - (1 - n18_depl[9]) * yd;}
#endif
#if defined(RADTRANSFER) && defined(RT_INFRARED)
    in.tdust = CellP[i].Dust_Temperature; /* the solver owns it */
#else
    in.tdust = CellP[i].DustTemp;
#endif
    double energy = rho * u_code; /* energy density */
    if(energy < rho * All.MinEgySpec) {energy = rho * All.MinEgySpec;}
    rho *= UNIT_DENSITY_IN_CGS;
    double timestep = dt_code * UNIT_TIME_IN_CGS;
    energy *= UNIT_ENERGY_IN_CGS / pow(UNIT_LENGTH_IN_CGS, 3);
    divv *= UNIT_VEL_IN_CGS / UNIT_LENGTH_IN_CGS;
    double *ab = in.abundances;
    for(int k = 0; k < TRAC_NUM; k++) {ab[k] = CellP[i].TracAbund[k];}
    in.abhe = chem_cxx_abhe_default(); in.X_H_chem = chem_cxx_xh_default();
    double yn = rho / ((1.0 + 4.0 * ABHE) * PROTONMASS_CGS), dl = 0;
    in.rpar[0] = yn; in.rpar[1] = dl; in.rpar[2] = divv;
    double abh2 = ab[IH2], abco = ab[ICO], abhd = 0, temp;
#if CHEMISTRYNETWORK == 17
    abhd = ab[IHD];
#endif
#if CHEMISTRYNETWORK == 18
    temp = chem_cxx_net18_temp(ab, energy, yn, in.abhe);
#else
    temp = chem_cxx_calc_temp(abh2, in.abhe, energy / (BOLTZMANN_CGS * (1.0 + ABHE - abh2 + ab[IHP]) * yn));
#endif
    if(temp > 1.0e5) {in.dust_to_gas_ratio = 0;} /* hot gas: grains sputtered */
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS
    sgchem_treecol_columns(i, &in);
#endif
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS_PHOTOION
    sgchem_treecol_photoion_rates(i, &in, rho);
#endif
#ifdef RADTRANSFER
    in.NH_cell = 0.5 * yn * cbrt(P[i].Mass / (CellP[i].Density * All.cf_a3inv)) * UNIT_LENGTH_IN_CGS; /* unresolved half-cell column; cell size from the physical density is physical */
#endif

    int skip = 0; out.tdust = in.tdust;
#ifdef GALSF_FB_PHOTOIONIZATION_PER_DIRECTION
    {  /* photoionised cell: ionised abundances, no network, energy at least that of 1e4 K ionised gas */
        double e_HII = 1.0e4 * 1.5 * BOLTZMANN_CGS * yn * (2.0 + 2.0 * ABHE + in.abundD + in.abundc);
        int ionised = (All.Ti_Current < CellP[i].PhotoionizedUntil);
        if(ionised) {
            skip = 1; sgchem_set_ionized(ab, &in);
#ifdef OUTPUT_MASS_METAL_ENERGY_LEDGER
            if(energy < e_HII) {ledger_add(LEDGER_PI_HEATING, (e_HII - energy) / rho * P[i].Mass * UNIT_MASS_IN_CGS / UNIT_ENERGY_IN_CGS);}
#endif
            energy = DMAX(energy, e_HII);
        }
#ifdef TEST_PHOTOIONIZATION_FIXED_RATE_TWO_TEMPERATURE_GAS
        if(All.TestPhotoionNeutralTemp > 0) {skip = 1; energy = ionised ? e_HII : All.TestPhotoionNeutralTemp * 1.5 * BOLTZMANN_CGS * yn * (1.0 + ABHE);} /* StarBench: ionised at 1e4 K, neutral isothermal, no network */
#endif
    }
#endif
    sgchem_network_skipped = skip;
    if(!skip) {
    if(All.ComovingIntegrationOn) { /* the background is a floor under the local photoionisation and heating rates */
        double *rate[6] = {&in.rt_phot_HI, &in.rt_phot_HeI, &in.rt_phot_HeII, &in.rt_heat_HI, &in.rt_heat_HeI, &in.rt_heat_HeII};
#ifdef SGCHEM_UVB_TREECOL_SHIELDING
        {double f_gamma[3], f_heat[3]; sgchem_uvb_treecol_shielding(i, f_gamma, f_heat); /* the G0 floor above stays unshielded */
         for(int c = 0; c < 3; c++) {uvb[c] *= f_gamma[c]; uvb[3 + c] *= f_heat[c];}}
#endif
        for(int k = 0; k < 6; k++) {if(*rate[k] < uvb[k]) {*rate[k] = uvb[k];}}
        in.irad_heat = 1;
    }
    chem_cxx_begin_cell(&in, temp, abh2, abhd, abco);
#if CHEMISTRYNETWORK == 17
    if(ab[IHP] > 0.99) {ab[IH2] = 0; ab[ICO] = 0; ab[IHD] = 0;} /* fully ionised: no molecules (stiffness) */
#endif
#ifdef TEST_SGCHEM_PRINT_INPUT_POOL_VIOLATIONS /* species above the pools they draw from, as handed to the solver (before the clamp) */
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 17 || CHEMISTRYNETWORK == 20
    {
        int viol = 0; const double rtol = 1.0e-3;
        if(ab[ICO] > DMIN(in.abundc, in.abundo) * (1.0 + rtol) + 1.0e-20) {viol |= 1;}
        if(2.0 * ab[IH2] + ab[IHP] > 1.0 + rtol) {viol |= 2;}
#if CHEMISTRYNETWORK == 17
        if(ab[IHEP] + ab[IHEPP] > in.abhe * (1.0 + rtol) + 1.0e-20) {viol |= 4;}
        if(ab[IDP] + ab[IHD] > in.abundD * (1.0 + rtol) + 1.0e-20) {viol |= 8;}
#endif
        if(viol) {printf("SGCHEM_INPUT_VIOLATION: ID=%llu viol=%d yn=%.4e T=%.4e pools abundc=%.6e abundo=%.6e abhe=%.6e abundD=%.6e H2=%.6e H+=%.6e CO=%.6e\n",
                         (unsigned long long) P[i].ID, viol, yn, temp, in.abundc, in.abundo, in.abhe, in.abundD, ab[IH2], ab[IHP], ab[ICO]);}
    }
#endif
#endif
    sgchem_clamp_pools(ab, &in);
    chem_cxx_evolve_cell(timestep, dl, yn, divv, &energy, ab);
    chem_cxx_read_outputs(&out);
#ifdef COSMIC_RAY_FLUID
    sgchem_cr_ionisation_from_fluid(i, &in, out.cr_ion_heat * (P[i].Mass / (CellP[i].Density * All.cf_a3inv)) * pow(UNIT_LENGTH_IN_CGS, 3) / UNIT_ENERGY_IN_CGS);
#endif
#ifdef TEST_TREECOL_SHELLS_PHOTOION_FIXED_RATE_TWO_TEMPERATURE_GAS
    if(All.TestTreeRayNeutralTemp > 0) /* StarBench: T(x) = T_n + (1e4 K - T_n) x, no losses */
    {
        double x = DMIN(DMAX(ab[IHP], 0.0), 1.0);
        energy = (All.TestTreeRayNeutralTemp + (1.0e4 - All.TestTreeRayNeutralTemp) * x) * 1.5 * BOLTZMANN_CGS * yn * (1.0 + ABHE + x);
    }
#endif
    }
#if CHEMISTRYNETWORK == 18
    chem_cxx_net18_finish(ab, energy, &temp);
#elif CHEMISTRYNETWORK == 20
    chem_cxx_net20_finish(ab, energy, &temp);
#else
    temp = chem_cxx_calc_temp(ab[IH2], in.abhe, energy / (BOLTZMANN_CGS * (1.0 + ABHE - ab[IH2] + ab[IHP]) * yn));
#endif
    if(temp < All.MinGasTemp && temp > 0) {energy *= All.MinGasTemp / temp; temp = All.MinGasTemp;}
#ifdef TEST_RT_STARBENCH_SGCHEM_TEMPERATURE_FROM_IONISED_FRACTION
    {double x = DMIN(DMAX(ab[IHP], 0.), 1.); temp = All.InitGasTemp + (1.e4 - All.InitGasTemp) * x; energy = temp * 1.5 * BOLTZMANN_CGS * yn * (1. + ABHE + x);} /* StarBench two-temperature gas */
#endif
    for(int k = 0; k < TRAC_NUM; k++) {CellP[i].TracAbund[k] = ab[k];}
    CellP[i].Temp = temp;
    CellP[i].DustTemp = out.tdust;
#ifdef OUTPUT_INDIVIDUAL_COOLRATES
    for(int k = 0; k < 28; k++) {CellP[i].SGChemLambda[k] = (float) (out.lambda[k] / yn / yn);}
    for(int k = 0; k < 6; k++) {CellP[i].SGChemLambdaChem[k] = (float) (out.lambda_chem[k] / yn / yn);}
#endif
#ifdef OUTPUT_CHEM_PHOTORATES
    for(int k = 0; k < 6; k++) {CellP[i].SGChemPhotoRates[k] = (float) out.photorates[k];}
#endif
#ifdef OUTPUT_SGCHEM_RADIATION_FIELDS
    CellP[i].ShieldFacDust = (float) out.fac_shield_dust;
#endif
    CellP[i].Ne = sgchem_electrons(ab, &in);
#if defined(RADTRANSFER) && defined(RT_INFRARED)
    sgchem_rt_cooling_to_infrared(i, out.lambda, temp);
#endif
    return energy * pow(UNIT_LENGTH_IN_CGS, 3) / UNIT_ENERGY_IN_CGS / (rho / UNIT_DENSITY_IN_CGS);
}

#ifdef SGCHEM_TEMPERATURE_FROM_CHEMISTRY_AT_SNAPSHOT_RESTART
/* gas temperature [K] from the cell's energy and abundances, no solve: a snapshot carries no Temp, and the first force pass reads it */
double sgchem_startup_temperature(int i)
{
    double rho = CellP[i].Density * All.cf_a3inv, energy = DMAX(rho * CellP[i].InternalEnergy, rho * All.MinEgySpec);
    rho *= UNIT_DENSITY_IN_CGS; energy *= UNIT_ENERGY_IN_CGS / pow(UNIT_LENGTH_IN_CGS, 3);
    double ab[TRAC_NUM]; for(int k = 0; k < TRAC_NUM; k++) {ab[k] = CellP[i].TracAbund[k];}
    double yn = rho / ((1.0 + 4.0 * ABHE) * PROTONMASS_CGS), temp;
#if CHEMISTRYNETWORK == 18
    temp = chem_cxx_net18_temp(ab, energy, yn, chem_cxx_abhe_default());
#else
    temp = chem_cxx_calc_temp(ab[IH2], chem_cxx_abhe_default(), energy / (BOLTZMANN_CGS * (1.0 + ABHE - ab[IH2] + ab[IHP]) * yn));
#endif
    if(temp < All.MinGasTemp && temp > 0) {temp = All.MinGasTemp;}
    return temp;
}
#endif

/* electrons per H from the cell's current abundances: the state the first force pass reads, before any solve */
double sgchem_startup_electrons(int i)
{
    struct ChemInputs in; memset(&in, 0, sizeof(in));
    sgchem_gas_abundances(i, &in);
    double ab[TRAC_NUM]; for(int k = 0; k < TRAC_NUM; k++) {ab[k] = CellP[i].TracAbund[k];}
#if CHEMISTRYNETWORK == 20
    return ab[IHP]; /* its electrons come from the last solve */
#else
    return sgchem_electrons(ab, &in);
#endif
}

/* DoCooling branch for SGCHEM: returns the new u; its cached Ne, HI, Gamma, mean molecular weight and
   Temperature follow the chemistry, so its Temperature is the chemistry temperature */
double sgchem_docooling(int i, double u_old, double dt, double *ne_eval)
{
    double u_before = CellP[i].InternalEnergy, u, u_solve, u_in;
#ifndef COOLING_OPERATOR_SPLIT
    if(CellP[i].CoolingIsOperatorSplitThisTimestep == 0)
    { /* unsplit: half the hydro heating before the solve, a corrected half after (as for grackle) */
        double du = dt * CellP[i].DtInternalEnergy / ((UNIT_SPECEGY_IN_CGS/UNIT_TIME_IN_CGS) * (PROTONMASS_CGS/HYDROGEN_MASSFRAC));
        u_old += 0.5*du;
        u = u_solve = sgchem_cell_step(i, u_in = u_old, dt);
        double r=u/u_old; if(r>1) {r=1/r;} if(fabs(r-1)>1.e-4) {r=(r-1)/log(r);} r=DMAX(0,DMIN(r,1));
        du *= 0.5*r; if(du<-0.5*u) {du=-0.5*u;} u+=du;
    } else
#endif
    {u = u_solve = sgchem_cell_step(i, u_in = u_old, dt);}
#ifdef TEST_PHOTOIONIZATION_FIXED_RATE_TWO_TEMPERATURE_GAS
    if(All.TestPhotoionNeutralTemp > 0) {u = u_solve;} /* StarBench: the pinned energy */
#endif
#ifdef TEST_TREECOL_SHELLS_PHOTOION_FIXED_RATE_TWO_TEMPERATURE_GAS
    if(All.TestTreeRayNeutralTemp > 0) {u = u_solve;} /* the pinned energy */
#endif
    if(!isfinite(u)) {printf("SGCHEM: non-finite energy after cooling, ID=%llu u=%g, keeping %g\n", (unsigned long long) P[i].ID, u, u_before); u = u_solve = u_before;}
    u = DMAX(u, All.MinEgySpec);
#ifdef OUTPUT_MASS_METAL_ENERGY_LEDGER
    if(!sgchem_network_skipped) {double dE = (u_solve - u_in) * P[i].Mass; ledger_add(dE > 0 ? LEDGER_CHEM_HEATING : LEDGER_CHEM_COOLING, fabs(dE));} /* the solve only */
#endif
    *ne_eval = CellP[i].Ne;
    CellP[i].HI = DMAX(0, 1. - CellP[i].TracAbund[IHP]); /* neutral H nuclei (atomic and molecular) per H */
#if defined(RADTRANSFER) && defined(RT_CHEM_PHOTOION)
    sgchem_rt_absorbers(i, CellP[i].TracAbund); /* with the moment solver HI holds what absorbs in the ionising bins */
#endif
    CellP[i].Gamma = CellP[i].gamma_eos_value();
    if(u_solve > 0) {CellP[i].MeanMolecularWeight = CellP[i].Temp / ((CellP[i].Gamma - 1.) * U_TO_TEMP_UNITS * u_solve);}
    CellP[i].Temperature = (u == u_solve) ? CellP[i].Temp : CellP[i].gas_temperature_from_u(u);
#ifdef EOS_ANCHOR_INTERNALENERGY_IN_DRIFTS
    CellP[i].u_anchor = u;
#endif
#ifdef TEST_SGCHEM_PRINT_THERMAL_PROPERTIES
    {double mu, ne, nh0, nhp, T = ThermalProperties(u, CellP[i].Density * All.cf_a3inv, i, &mu, &ne, &nh0, &nhp, P, CellP), fmol = Get_Gas_Molecular_Mass_Fraction(i, T, nh0, ne, 0., P, CellP);
    printf("THERMALPROPS %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", T, mu, ne, nh0, nhp, CellP[i].Temp, CellP[i].TracAbund[IH2], CellP[i].TracAbund[IHP], fmol);}
#endif
    return u;
}

#endif /* SGCHEM */
