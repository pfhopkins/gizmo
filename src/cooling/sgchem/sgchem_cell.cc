/* C++ implementation: Ulrich Steinwandel (uli@mpa-garching.mpg.de).
   Based on the SGChem Fortran networks by Simon Glover (GNU GPL, as distributed in Arepo). */
/* SGChem in GIZMO: one gas cell over one cooling step (inputs from the cell, CVODE solve, state back), and the
   DoCooling branch. Host only: the solver scratch is thread_local, each cell's result depends only on its inputs. */
#include <stdio.h>
#include <math.h>
#include "../../declarations/allvars.h"
#include "../../core/proto.h"
#include "chem_interface.h"
#include "sgchem_network.h"
#include "sgchem_dust_to_gas.h"
#include "../../gravity/gravtree_treecol_pixels.h"
#if CHEMISTRYNETWORK == 18
#include "cxx/chem_net18_layout.h"
#endif

#ifdef SGCHEM

/* element pools per H from the FIRE-3 metal slots (0 Z, 1 He, 2 C, 3 N, 4 O, 5 Ne, 6 Mg, 7 Si, 8 S, 9 Ca, 10 Fe), as stella_dev chem_fill_gas_abundances */
static void sgchem_gas_abundances(int i, struct ChemInputs *in)
{
    const MyFloat *Z = P[i].Metallicity;
    double X_H = 1.0 / (1.0 + 4.0 * ABHE); /* the H reference of yn below */
    double g_dep = DMAX(All.SGChemDustToGasRatio, 0);
    in->abundc  = (1 - DMIN(SGCHEM_DEPL_C  * g_dep, SGCHEM_DEPL_MAX)) * (Z[2] / 12.0) / X_H;
    in->abundo  = (1 - DMIN(SGCHEM_DEPL_O  * g_dep, SGCHEM_DEPL_MAX)) * (Z[4] / 16.0) / X_H;
    in->abundsi = (1 - DMIN(SGCHEM_DEPL_SI * g_dep, SGCHEM_DEPL_MAX)) * (Z[7] / 28.0) / X_H;
    in->abundN  = (Z[3] / 14.0) / X_H;
#if CHEMISTRYNETWORK == 18
    in->abundNe = (Z[5] / 20.0) / X_H; in->abundS = (Z[8] / 32.0) / X_H;
    in->abundMg = (Z[6] / 24.0) / X_H; in->abundFe = (Z[10] / 56.0) / X_H;
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

#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
/* the unattenuated stellar field of cell i from the gravity walk, as stella_dev: each 6-13.6 eV pixel floored at
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
        case IO_SGCHEM_CR_ZETA: return All.SGChemCosmicRayIonRate;
        default: return 0;
    }
}
#endif

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
#else
    struct ChemFUVField fuv = sgchem_uniform_field(All.SGChemG0, All.SGChemG0_LW, All.SGChemBackgroundG0); /* uniform, unattenuated */
#endif
    in.G0 = fuv.G0; in.G0_LW = fuv.G0_LW; in.G0_dust = fuv.G0_dust;
    in.cosmic_ray_ion_rate = All.SGChemCosmicRayIonRate;
    in.id_current = (int) P[i].ID;
    in.dust_to_gas_ratio = sgchem_dust_to_gas_ratio(P[i].Metallicity[0]);
#if CHEMISTRYNETWORK == 18
    {double yd = DMIN(DMAX(in.dust_to_gas_ratio, 0), 1); /* RTZ depletion scaled with the DGR */
     in.abundc *= 1 - (1 - n18_depl[2]) * yd; in.abundN *= 1 - (1 - n18_depl[3]) * yd; in.abundo *= 1 - (1 - n18_depl[4]) * yd; in.abundNe *= 1 - (1 - n18_depl[5]) * yd;
     in.abundMg *= 1 - (1 - n18_depl[6]) * yd; in.abundsi *= 1 - (1 - n18_depl[7]) * yd; in.abundS *= 1 - (1 - n18_depl[8]) * yd; in.abundFe *= 1 - (1 - n18_depl[9]) * yd;}
#endif
    in.tdust = CellP[i].DustTemp;
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

    chem_cxx_begin_cell(&in, temp, abh2, abhd, abco);
#if CHEMISTRYNETWORK == 17
    if(ab[IHP] > 0.99) {ab[IH2] = 0; ab[ICO] = 0; ab[IHD] = 0;} /* fully ionised: no molecules (stiffness) */
#endif
    sgchem_clamp_pools(ab, &in);
    chem_cxx_evolve_cell(timestep, dl, yn, divv, &energy, ab);
    chem_cxx_read_outputs(&out);
#if CHEMISTRYNETWORK == 18
    chem_cxx_net18_finish(ab, energy, &temp);
#elif CHEMISTRYNETWORK == 20
    chem_cxx_net20_finish(ab, energy, &temp);
#else
    temp = chem_cxx_calc_temp(ab[IH2], in.abhe, energy / (BOLTZMANN_CGS * (1.0 + ABHE - ab[IH2] + ab[IHP]) * yn));
#endif
    if(temp < All.MinGasTemp && temp > 0) {energy *= All.MinGasTemp / temp; temp = All.MinGasTemp;}
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
    return energy * pow(UNIT_LENGTH_IN_CGS, 3) / UNIT_ENERGY_IN_CGS / (rho / UNIT_DENSITY_IN_CGS);
}

/* DoCooling branch for SGCHEM (as stella_dev): returns the new u; his cached Ne, HI, Gamma, mean molecular weight and
   Temperature follow the chemistry, so his Temperature is the chemistry temperature */
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
    if(!isfinite(u)) {printf("SGCHEM: non-finite energy after cooling, ID=%llu u=%g, keeping %g\n", (unsigned long long) P[i].ID, u, u_before); u = u_solve = u_before;}
    u = DMAX(u, All.MinEgySpec);
#ifdef OUTPUT_MASS_METAL_ENERGY_LEDGER
    {double dE = (u_solve - u_in) * P[i].Mass; ledger_add(dE > 0 ? LEDGER_CHEM_HEATING : LEDGER_CHEM_COOLING, fabs(dE));} /* the solve only */
#endif
    *ne_eval = CellP[i].Ne;
    CellP[i].HI = DMAX(0, 1. - CellP[i].TracAbund[IHP]); /* neutral H nuclei (atomic and molecular) per H */
    CellP[i].Gamma = CellP[i].gamma_eos_value();
    if(u_solve > 0) {CellP[i].MeanMolecularWeight = CellP[i].Temp / ((CellP[i].Gamma - 1.) * U_TO_TEMP_UNITS * u_solve);}
    CellP[i].Temperature = (u == u_solve) ? CellP[i].Temp : CellP[i].gas_temperature_from_u(u);
#ifdef EOS_ANCHOR_INTERNALENERGY_IN_DRIFTS
    CellP[i].u_anchor = u;
#endif
    return u;
}

#endif /* SGCHEM */
