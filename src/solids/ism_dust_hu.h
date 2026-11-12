/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* Dust processes of Hu et al. 2019, 2023, 2026 (GALSF_ISMDUSTCHEM_HU_MODEL) on the fields of the ISM dust model, which keep
   their transport, snapshots and restart: thermal and nonthermal sputtering per MRN size bin with the dust-gas drift of the
   cell, growth below 300 K, operator-split after the chemistry; replaces update_dust_processes. Only dust fields change,
   never the metals. Host and device. */
#ifndef ISM_DUST_HU_H
#define ISM_DUST_HU_H
#ifdef GALSF_ISMDUSTCHEM_HU_MODEL
#include "ism_dust_hu_kernels.h"
#include "../cooling/sgchem/sgchem_config.h"
#include "../cooling/sgchem/sgchem_consts.h"
#include "../cooling/sgchem/chem_interface.h"

#define ISM_DUST_HU_FSUB 0.1         /* subcycle factor (Hu+19 eq 13) */
#define ISM_DUST_HU_NSUB_MAX 100000
#define ISM_DUST_HU_XE_METALS 1.4e-4 /* electrons from C+ at solar metallicity */

/* the cell as the kernels see it: SGChem's temperature and ionisation */
KOKKOS_INLINE_FUNCTION void ism_dust_hu_env(int i, const struct particle_data *pp, const struct gas_cell_data *cell, struct dustk_env *e)
{
    const MyFloat *Z = pp[i].Metallicity;
    double Z_rel = DMAX(Z[0], 0) / SGCHEM_SOLAR_Z, accel_to_kms_yr = UNIT_VEL_IN_KMS / UNIT_TIME_IN_YR;
    e->nH = cell[i].Density * All.cf_a3inv * UNIT_DENSITY_IN_NHCGS * DMAX(1.0 - Z[0] - Z[1], 0);
    e->T = cell[i].Temp;
    e->x_e = DMAX(cell[i].TracAbund[IHP], 0) + ISM_DUST_HU_XE_METALS * Z_rel;
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
    double u_uv = 0; for(int k = 0; k < NPIX; k++) {u_uv += cell[i].TreeColEgyUV[k];}
    e->G0 = DMAX(u_uv / HABING_EGY_DENSITY_6_13P6EV_CGS, 0.324e-2 * All.SGChemBackgroundG0);
#else
    e->G0 = All.SGChemG0;
#endif
    for(int k = 0; k < 3; k++) {e->ahyd[k] = cell[i].HydroAccel[k] * accel_to_kms_yr;}
    e->divv = (pp[i].Particle_DivVel * All.cf_a2inv + (All.ComovingIntegrationOn ? 3 * All.cf_hubble_a : 0)) / UNIT_TIME_IN_YR; /* physical, Hubble flow included */
#ifdef GALSF_ISMDUSTCHEM_HU_NO_BETATRON
    e->betatron = 0;
#else
    e->betatron = 1;
#endif
}

KOKKOS_INLINE_FUNCTION void update_dust_processes_hu(int i, double dtime_gyr, struct particle_data *pp, struct gas_cell_data *cell)
{
    MyDouble *D = cell[i].ISMDustChem_Dust_Metal; /* slots 2 C, 4 O, 6 Mg, 7 Si, 10 Fe */
    const MyFloat *Z = pp[i].Metallicity;
    double dC = D[2], dS[4] = {D[4], D[6], D[7], D[10]}, total0 = dC + dS[0] + dS[1] + dS[2] + dS[3], dt_yr = dtime_gyr * 1.e9;
    if(total0 <= 0) {for(int b = 0; b < GALSF_ISMDUSTCHEM_HU_SIZE_BINS; b++) {for(int k = 0; k < 3; k++) {cell[i].ISMDustChem_Hu_Vrel[b][k] = 0;}} return;}
    double frac[4]; /* silicate mass fractions of O, Mg, Si, Fe */
    for(int k = 0; k < 4; k++) {frac[k] = All.ISMDustChem_SilicateNumberOfAtomsTable[k] * All.ISMDustChem_AtomicMassTable[All.ISMDustChem_SilicateMetallicityFieldIndexTable[k]] / All.ISMDustChem_EffectiveSilicateDustAtomicWeight;}
    struct dustk_env env; ism_dust_hu_env(i, pp, cell, &env);

    /* thermal + nonthermal sputtering per MRN size bin (Hu+19 eqs 2, 12, 20) */
    double YthC = dustk_Yth_C(env.T), YthS = dustk_Yth_Sil(env.T), fC = 0, fS = 0;
    for(int b = 0; b < GALSF_ISMDUSTCHEM_HU_SIZE_BINS; b++) {
        double a_b, w_b; dustk_size_bin(GALSF_ISMDUSTCHEM_HU_SIZE_BINS, b, &a_b, &w_b);
        double v[3] = {cell[i].ISMDustChem_Hu_Vrel[b][0], cell[i].ISMDustChem_Hu_Vrel[b][1], cell[i].ISMDustChem_Hu_Vrel[b][2]}, InC, InS;
        dustk_step_vrel(&env, a_b, dt_yr, v, &InC, &InS, ISM_DUST_HU_FSUB, ISM_DUST_HU_NSUB_MAX);
        for(int k = 0; k < 3; k++) {cell[i].ISMDustChem_Hu_Vrel[b][k] = v[k];}
        fC += w_b * exp(-(dt_yr * 3.0 * env.nH * YthC / a_b + InC));
        fS += w_b * exp(-(dt_yr * 3.0 * env.nH * YthS / a_b + InS));
    }
    dC *= fC; for(int k = 0; k < 4; k++) {dS[k] *= fS;}
    double total1 = dC + dS[0] + dS[1] + dS[2] + dS[3], growth = 0;

    /* growth (Hu+23/26 eq 4): T < 300 K, total-Z single (1-f), CO-bound C and O excluded, exact logistic */
    double r0 = dustk_growth_rate0(env.nH, env.T, All.DustGrowthTimescaleGyr, All.DustGrowthCoulombFactor);
    if(r0 > 0) {
        double Zp = DMAX(Z[0], 0) / SGCHEM_SOLAR_Z, C_CO = 0, O_CO = 0;
#if (ICO != UNUSED)
        double xco = DMAX(cell[i].TracAbund[ICO], 0) / (1.0 + 4.0 * ABHE); C_CO = 12.0 * xco; O_CO = 16.0 * xco;
#endif
        double C1 = dustk_grow_carbon(dC, DMAX(Z[2], 0), C_CO, Zp, r0, dt_yr), S0 = dS[2] / frac[2], Fe_free = DMAX(dS[3] - S0 * frac[3], 0);
        double pool[4] = {DMAX(Z[4] - O_CO, 0), DMAX(Z[6], 0), DMAX(Z[7], 0), DMAX(Z[10] - Fe_free, 0)};
        double dSil = dustk_grow_silicate(S0, pool, frac, Zp, r0, dt_yr) - S0;
        growth = C1 - dC; dC = C1;
        if(dSil > 0) {for(int k = 0; k < 4; k++) {dS[k] += dSil * frac[k];} growth += dSil;}
    }

    /* dust stays inside its element pool; silicate scales as a whole */
    dC = DMIN(dC, DMAX(Z[2], 0));
    {double pool[4] = {DMAX(Z[4], 0), DMAX(Z[6], 0), DMAX(Z[7], 0), DMAX(Z[10], 0)}, f = 1;
     for(int k = 0; k < 4; k++) {if(dS[k] > 0) {f = DMIN(f, pool[k] / dS[k]);}}
     if(f < 1) {for(int k = 0; k < 4; k++) {dS[k] *= f;}}}

    D[2] = dC; D[4] = dS[0]; D[6] = dS[1]; D[7] = dS[2]; D[10] = dS[3];
    double total2 = dC + dS[0] + dS[1] + dS[2] + dS[3];
    D[0] = total2;
    double Fe_free = DMAX(dS[3] - dS[2] / frac[2] * frac[3], 0), keep = total1 / total0, trim = (total1 + growth > 0) ? total2 / (total1 + growth) : 0;
    for(int k = 0; k < NUM_ISMDUSTCHEM_SOURCES; k++) {cell[i].ISMDustChem_Dust_Source[k] *= keep;}
    cell[i].ISMDustChem_Dust_Source[0] += growth; /* 0 = gas-dust accretion */
    for(int k = 0; k < NUM_ISMDUSTCHEM_SOURCES; k++) {cell[i].ISMDustChem_Dust_Source[k] *= trim;}
    int iron = All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_FreeIron_Index];
    cell[i].ISMDustChem_Dust_Species[All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_Carb_Index]] = dC;
    cell[i].ISMDustChem_Dust_Species[All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_Sil_Index]] = total2 - dC - ((iron >= 0) ? Fe_free : 0);
    if(iron >= 0) {cell[i].ISMDustChem_Dust_Species[iron] = Fe_free;}
}

/* fresh start: the Milky Way grain-locked fractions of the cell's own C and Si, silicate capped by O, Mg, Fe (from supernovae) */
inline void ism_dust_hu_seed_cell(int i, const struct particle_data *pp, struct gas_cell_data *cell)
{
    const MyFloat *Z = pp[i].Metallicity;
    static const int slot[4] = {4, 6, 7, 10};
    double frac[4], sil = SGCHEM_DEPL_SI * DMAX(Z[7], 0);
    for(int k = 0; k < 4; k++) {frac[k] = All.ISMDustChem_SilicateNumberOfAtomsTable[k] * All.ISMDustChem_AtomicMassTable[slot[k]] / All.ISMDustChem_EffectiveSilicateDustAtomicWeight;}
    sil /= frac[2];
    for(int k = 0; k < 4; k++) {if(k != 2) {sil = DMIN(sil, SGCHEM_DEPL_MAX * DMAX(Z[slot[k]], 0) / frac[k]);}}
    for(int k = 0; k < NUM_ISMDUSTCHEM_ELEMENTS; k++) {cell[i].ISMDustChem_Dust_Metal[k] = 0;}
    for(int k = 0; k < NUM_ISMDUSTCHEM_SOURCES; k++) {cell[i].ISMDustChem_Dust_Source[k] = 0;}
    for(int k = 0; k < NUM_ISMDUSTCHEM_SPECIES; k++) {cell[i].ISMDustChem_Dust_Species[k] = 0;}
    cell[i].ISMDustChem_Dust_Metal[2] = SGCHEM_DEPL_C * DMAX(Z[2], 0);
    for(int k = 0; k < 4; k++) {cell[i].ISMDustChem_Dust_Metal[slot[k]] = sil * frac[k];}
    cell[i].ISMDustChem_Dust_Metal[0] = cell[i].ISMDustChem_Dust_Metal[2] + sil;
    cell[i].ISMDustChem_Dust_Source[2] = cell[i].ISMDustChem_Dust_Metal[0];
    cell[i].ISMDustChem_Dust_Species[All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_Sil_Index]] = sil;
    cell[i].ISMDustChem_Dust_Species[All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_Carb_Index]] = cell[i].ISMDustChem_Dust_Metal[2];
}

#endif
#endif
