/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* Dust-to-gas ratio of a gas cell for SGChem, shared by the chemistry input and the tree dust column. */
#ifndef SGCHEM_DUST_TO_GAS_H
#define SGCHEM_DUST_TO_GAS_H
#include "sgchem_config.h"

/* relative to the Milky Way at solar metallicity, scaled with the cell metallicity Z, capped at 10 */
#ifdef SGCHEM_DGR_INDEPENDENT_OF_METALLICITY
KOKKOS_INLINE_FUNCTION double sgchem_dust_to_gas_ratio(double Z) {return fmin(All.SGChemDustToGasRatio, 10.0);} /* the same at every Z */
#else
KOKKOS_INLINE_FUNCTION double sgchem_dust_to_gas_ratio(double Z) {return fmin(All.SGChemDustToGasRatio * Z / SGCHEM_SOLAR_Z, 10.0);}
#endif

#endif
