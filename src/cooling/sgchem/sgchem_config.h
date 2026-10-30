/* C++ implementation: Ulrich Steinwandel (uli@mpa-garching.mpg.de).
   Based on the SGChem Fortran networks by Simon Glover (GNU GPL, as distributed in Arepo). */
/* SGChem chemistry-private switches, derived from the build flags (preprocessor only; the C++ backend sees GIZMO_config.h only).
   The per-pixel tree columns and fluxes (TreeCol extension) are not in this tree yet: without them the chemistry is optically
   thin (no dust, H2 or CO shielding) under a uniform field. TEST_SGCHEM_ONEZONE_WITH_12_PIXEL_COLUMNS gives the standalone
   one-zone drivers the 12-pixel input layout of the frozen references; GIZMO builds reject it (sgchem.cc). */
#ifndef SGCHEM_CONFIG_H
#define SGCHEM_CONFIG_H

#ifdef SGCHEM
#define WSS_CIE_COOL   /* WSS CIE metal-line cooling above 1e4 K from the cell metals (nets 5, 15) */
#define CO_SHIELDING   /* Visser+09 CO self- and H2 cross-shielding */
#ifdef METALS
#define GALSF_RESOLVEDISM_METALS_INDIVIDUAL /* N and the CIE element masses come from the cell metals */
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
