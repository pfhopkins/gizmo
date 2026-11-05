/* C++ implementation: Ulrich Steinwandel (uli@mpa-garching.mpg.de).
   Based on the SGChem Fortran networks by Simon Glover (GNU GPL, as distributed in Arepo). */
/* SGChem chemistry-private switches, derived from the build flags (preprocessor only; the C++ backend sees GIZMO_config.h only).
   With GALSF_RESOLVEDISM_TREECOL_PIXELS the gravity walk gives per-pixel dust, H2 and CO columns (and, with
   ..._STELLAR_BAND_FLUXES, the stellar field per pixel); without it the chemistry is optically thin (no dust, H2 or CO
   shielding) under a uniform field. TEST_SGCHEM_ONEZONE_WITH_12_PIXEL_COLUMNS gives the standalone
   one-zone drivers the 12-pixel input layout of the frozen references; GIZMO builds reject it (sgchem.cc). */
#ifndef SGCHEM_CONFIG_H
#define SGCHEM_CONFIG_H

#ifdef SGCHEM
#define WSS_CIE_COOL   /* WSS CIE metal-line cooling above 1e4 K from the cell metals (nets 5, 15) */
#define CO_SHIELDING   /* Visser+09 CO self- and H2 cross-shielding */
#ifdef METALS
#define GALSF_RESOLVEDISM_METALS_INDIVIDUAL /* N and the CIE element masses come from the cell metals */
#endif
#define SGCHEM_CELL_GRAIN_HEATING_FIELD /* the cell call sets G0_dust (uniform field from parameters, or the tree field) */
#ifdef GALSF_SINGLE_STARS_RT_TABLE_BANDS
#define RT_TREECOL_STELLAR_PIXELS_PREATTENUATED /* transported stellar field arrives attenuated: the chemistry attenuates only the floor pixels (all chemistry files read this header) */
#endif
#define SGCHEM_SOLAR_Z  0.014  /* Asplund+09 present-day solar Z: the chemistry's metallicity unit, as stella_dev */
#define SGCHEM_DEPL_C   0.50   /* grain-locked fractions at the Milky Way dust-to-gas ratio, scaled with SGChemDustToGasRatio */
#define SGCHEM_DEPL_SI  0.82
#define SGCHEM_DEPL_O   0.217
#define SGCHEM_DEPL_MAX 0.9
#endif

#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS
#define NPIX GALSF_RESOLVEDISM_TREECOL_PIXELS /* the chemistry's pixel arrays hold 12 (static_assert in the backend) */
#define RT_TREECOL_DUST_WEIGHTED        /* per-pixel dust column */
#define RT_TREECOL_H2_CO                /* per-pixel H2 and CO columns */
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
#define GALSF_RESOLVEDISM_G0_VARIABLE   /* per-pixel FUV weights */
#endif
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS
#define RT_TREECOL_STELLAR_PIXELS_PREATTENUATED /* the walk attenuated the stars' pixels; floor pixels keep the column */
#endif
#endif

#ifdef TEST_SGCHEM_ONEZONE_WITH_12_PIXEL_COLUMNS
#define NPIX 12
#define NSIDE 1
#define RT_TREECOL_HEALPIX
#define RT_TREECOL_DUST_WEIGHTED        /* per-pixel dust column */
#define RT_TREECOL_H2_CO                /* per-pixel H2 and CO columns */
#define GALSF_RESOLVEDISM_G0_VARIABLE   /* per-pixel FUV weights */
#endif

#endif /* SGCHEM_CONFIG_H */
