/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* GALSF_RESOLVEDISM_TREECOL_PIXELS=N: per-direction columns (N HEALPix pixels) and stellar band fluxes accumulated in the
   gravity walk for the SGChem chemistry. Shared by the node-moment paths and the host and device walks. */
#ifndef GRAVTREE_TREECOL_PIXELS_H
#define GRAVTREE_TREECOL_PIXELS_H
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS

#include "../cooling/sgchem/sgchem_consts.h"
#include "../cooling/sgchem/sgchem_dust_to_gas.h"

/* node species masses: dust-weighted gas (dust-to-gas ratio x mass), H2 and CO */
#define TREECOL_PIXELS_DUST 0
#define TREECOL_PIXELS_H2   1
#define TREECOL_PIXELS_CO   2

/* a particle's species masses [code units]; zero for non-gas. H nuclei per mass as the chemistry, 1/((1+4 ABHE) m_p) */
KOKKOS_INLINE_FUNCTION void treecol_pixels_species_masses(const struct particle_data *pp, const struct gas_cell_data *cell, int i, double *m)
{
    m[TREECOL_PIXELS_DUST] = m[TREECOL_PIXELS_H2] = m[TREECOL_PIXELS_CO] = 0;
    if(pp[i].Type != 0) {return;}
    double dgr = (cell[i].Temp > 1.0e5) ? 0 : fmax(sgchem_dust_to_gas_ratio(pp[i].Metallicity[0]), 0.); /* hot gas: grains sputtered */
    m[TREECOL_PIXELS_DUST] = dgr * pp[i].Mass;
    double m_H = pp[i].Mass / (1.0 + 4.0 * ABHE);
    m[TREECOL_PIXELS_H2] = 2.0 * cell[i].TracAbund[IH2] * m_H;
    m[TREECOL_PIXELS_CO] = 28.0 * cell[i].TracAbund[ICO] * m_H;
}

#endif
#endif
