/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* Five-element ISM dust model (GALSF_ISMDUSTCHEM_FIVE_ELEMENT_MODEL) on the fields of the ISM dust model, which keep their
   transport, snapshots and restart: dust in C (carbon) and O, Mg, Si, Fe (FeMgSiO4), thermal sputtering and growth on one
   grain size, operator-split after the cooling; replaces update_dust_processes. Only dust fields change, never the metals. */
#ifndef ISM_DUST_FIVE_ELEMENT_H
#define ISM_DUST_FIVE_ELEMENT_H
#ifdef GALSF_ISMDUSTCHEM_FIVE_ELEMENT_MODEL
#include "ism_dust_five_element_kernels.h"

#define DUST5_NSLOT 5
KOKKOS_INLINE_FUNCTION int dust5_slot(int k) {return (k == 0) ? 2 : (k == 1) ? 4 : (k == 2) ? 6 : (k == 3) ? 7 : 10;} /* FIRE slots C, O, Mg, Si, Fe */

/* silicate mass fractions of O, Mg, Si, Fe */
KOKKOS_INLINE_FUNCTION void dust5_silicate_fractions(double frac[4])
{
    for(int k = 0; k < 4; k++) {frac[k] = All.ISMDustChem_SilicateNumberOfAtomsTable[k] * All.ISMDustChem_AtomicMassTable[All.ISMDustChem_SilicateMetallicityFieldIndexTable[k]] / All.ISMDustChem_EffectiveSilicateDustAtomicWeight;}
}

/* write the five elements back: total, species (carbon, silicate, free iron) and sources follow; kept = surviving share of the old dust */
KOKKOS_INLINE_FUNCTION void dust5_store(int i, const double d[5], const double frac[4], double kept, double grown, struct gas_cell_data *cell)
{
    MyDouble *D = cell[i].ISMDustChem_Dust_Metal;
    double total = 0, before = 0;
    for(int k = 0; k < DUST5_NSLOT; k++) {D[dust5_slot(k)] = d[k]; total += d[k];}
    D[0] = total;
    for(int k = 0; k < NUM_ISMDUSTCHEM_SOURCES; k++) {cell[i].ISMDustChem_Dust_Source[k] *= kept; before += cell[i].ISMDustChem_Dust_Source[k];}
    cell[i].ISMDustChem_Dust_Source[0] += grown; /* 0 = gas-dust accretion */
    double trim = (before + grown > 0) ? total / (before + grown) : 0;
    for(int k = 0; k < NUM_ISMDUSTCHEM_SOURCES; k++) {cell[i].ISMDustChem_Dust_Source[k] *= trim;}
    int iron = All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_FreeIron_Index];
    double Fe_free = (iron >= 0) ? DMAX(d[4] - d[3] / frac[2] * frac[3], 0) : 0;
    cell[i].ISMDustChem_Dust_Species[All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_Carb_Index]] = d[0];
    cell[i].ISMDustChem_Dust_Species[All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_Sil_Index]] = total - d[0] - Fe_free;
    if(iron >= 0) {cell[i].ISMDustChem_Dust_Species[iron] = Fe_free;}
}

KOKKOS_INLINE_FUNCTION void update_dust_processes_five_element(int i, double dtime_gyr, struct particle_data *pp, struct gas_cell_data *cell)
{
    double d[5], z[5], frac[4], total0 = 0;
    for(int k = 0; k < DUST5_NSLOT; k++) {d[k] = cell[i].ISMDustChem_Dust_Metal[dust5_slot(k)]; z[k] = pp[i].Metallicity[dust5_slot(k)]; total0 += d[k];}
    if(total0 <= 0) {return;}
    dust5_silicate_fractions(frac);
    double rho = cell[i].Density * All.cf_a3inv, nH = rho * UNIT_DENSITY_IN_NHCGS, dt_yr = dtime_gyr * 1.e9;
#ifdef SGCHEM
    double T = cell[i].Temp, T_dust = cell[i].DustTemp;
#else
    double ne = 1, nh0 = 0, nhp = 0, mu = 1, T = ThermalProperties(cell[i].InternalEnergyPred, rho, i, &mu, &ne, &nh0, &nhp, pp, cell), T_dust = DMIN(T, 100.);
#endif
    dust5_sputter(d, nH, T, dt_yr);
    double total1 = d[0] + d[1] + d[2] + d[3] + d[4];
    double grown = dust5_grow(d, z, frac, nH, T, T_dust, dt_yr);
    dust5_clamp(d, z);
    dust5_store(i, d, frac, total1 / total0, grown, cell);
}

/* fresh start: the seed of the five-element model, from supernovae */
inline void ism_dust_five_element_seed_cell(int i, const struct particle_data *pp, struct gas_cell_data *cell)
{
    double d[5], z[5], frac[4];
    for(int k = 0; k < DUST5_NSLOT; k++) {z[k] = pp[i].Metallicity[dust5_slot(k)];}
    dust5_silicate_fractions(frac);
    dust5_seed(d, z, frac, pp[i].Metallicity[0]);
    for(int k = 0; k < NUM_ISMDUSTCHEM_ELEMENTS; k++) {cell[i].ISMDustChem_Dust_Metal[k] = 0;}
    for(int k = 0; k < NUM_ISMDUSTCHEM_SOURCES; k++) {cell[i].ISMDustChem_Dust_Source[k] = (k == 2) ? 1 : 0;}
    dust5_store(i, d, frac, 1, 0, cell);
}

#endif
#endif
