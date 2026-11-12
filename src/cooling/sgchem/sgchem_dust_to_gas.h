/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* Dust-to-gas ratio of a gas cell for SGChem, shared by the chemistry input and the tree dust column. */
#ifndef SGCHEM_DUST_TO_GAS_H
#define SGCHEM_DUST_TO_GAS_H
#include "sgchem_config.h"
#include "sgchem_dgr_rr14.h"

/* relative to the Milky Way at solar metallicity, scaled with the cell metallicity Z, capped at 10 */
#ifdef SGCHEM_DGR_INDEPENDENT_OF_METALLICITY
KOKKOS_INLINE_FUNCTION double sgchem_dust_to_gas_ratio(double Z) {return fmin(All.SGChemDustToGasRatio, 10.0);} /* the same at every Z */
#else
#ifdef SGCHEM_DGR_REMY_RUYER14 /* Remy-Ruyer+14 below 0.26 Zsun, linear above */
KOKKOS_INLINE_FUNCTION double sgchem_dust_to_gas_ratio(double Z) {return fmin(All.SGChemDustToGasRatio * Z / SGCHEM_SOLAR_Z * sgchem_dgr_rr14_over_linear(Z / SGCHEM_SOLAR_Z), 10.0);}
#else
KOKKOS_INLINE_FUNCTION double sgchem_dust_to_gas_ratio(double Z) {return fmin(All.SGChemDustToGasRatio * Z / SGCHEM_SOLAR_Z, 10.0);}
#endif
#endif

#ifdef GALSF_ISMDUSTCHEM_HU_MODEL
#define SGCHEM_LIVE_DUST_MILKY_WAY (1.24e-3 + 3.5e-3) /* the solar seed: carbonaceous + silicate (Dwek 2005, Hu+23/26) */
#else
#define SGCHEM_LIVE_DUST_MILKY_WAY 0.01
#endif

/* dust-to-gas ratio of gas cell i: the live dust of GALSF_ISMDUSTCHEM_MODEL, else from Z */
KOKKOS_INLINE_FUNCTION double sgchem_cell_dust_to_gas_ratio(const struct particle_data *pp, const struct gas_cell_data *cell, int i)
{
#ifdef GALSF_ISMDUSTCHEM_MODEL
    return fmin(cell[i].ISMDustChem_Dust_Metal[0] / SGCHEM_LIVE_DUST_MILKY_WAY, 10.0);
#else
    return sgchem_dust_to_gas_ratio(pp[i].Metallicity[0]);
#endif
}

#endif
