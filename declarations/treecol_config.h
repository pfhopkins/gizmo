/* Tree column densities (TreeCol): RT_USE_TREECOL_FOR_NH = number of angular bins, 6 = cube faces, 12*NSIDE^2 = HEALPix.
   Preprocessor only: also included by the chemistry (C++ and Fortran), which see the raw Config.sh flags. */
#ifndef GIZMO_TREECOL_CONFIG_H
#define GIZMO_TREECOL_CONFIG_H

#if defined(TREE_RAD)
#error "TREE_RAD was removed: use RT_USE_TREECOL_FOR_NH=12*NSIDE^2 (HEALPix tree columns), e.g. RT_USE_TREECOL_FOR_NH=12"
#endif
#if defined(TREE_RAD_H2) || defined(TREE_RAD_CO)
#error "TREE_RAD_H2 and TREE_RAD_CO were removed: H2 and CO tree columns are on automatically for CHEMCOOL with HEALPix RT_USE_TREECOL_FOR_NH"
#endif
#if defined(NSIDE)
#error "NSIDE was removed: use RT_USE_TREECOL_FOR_NH=12*NSIDE^2"
#endif

/* variable ISRF needs the HEALPix columns; M1 with CHEMCOOL needs the H2/CO columns for line self-shielding */
#if defined(GALSF_RESOLVEDISM_G0_VARIABLE) && !defined(RT_USE_TREECOL_FOR_NH)
#define RT_USE_TREECOL_FOR_NH 12
#endif
#if (defined(RADTRANSFER) || defined(RT_M1) || defined(RT_OTVET) || defined(RT_FLUXLIMITEDDIFFUSION) || defined(RT_LOCALRAYGRID)) && defined(CHEMCOOL) && !defined(RT_USE_TREECOL_FOR_NH)
#define RT_USE_TREECOL_FOR_NH 12
#endif

#ifdef RT_USE_TREECOL_FOR_NH
#if (RT_USE_TREECOL_FOR_NH != 6)
#define RT_TREECOL_HEALPIX
#if (RT_USE_TREECOL_FOR_NH == 12)
#define NSIDE 1
#elif (RT_USE_TREECOL_FOR_NH == 48)
#define NSIDE 2
#elif (RT_USE_TREECOL_FOR_NH == 108)
#define NSIDE 3
#elif (RT_USE_TREECOL_FOR_NH == 192)
#define NSIDE 4
#elif (RT_USE_TREECOL_FOR_NH == 300)
#define NSIDE 5
#elif (RT_USE_TREECOL_FOR_NH == 432)
#define NSIDE 6
#elif (RT_USE_TREECOL_FOR_NH == 588)
#define NSIDE 7
#elif (RT_USE_TREECOL_FOR_NH == 768)
#define NSIDE 8
#else
#error "RT_USE_TREECOL_FOR_NH must be 6 (cube faces) or 12*NSIDE^2 with NSIDE = 1..8 (HEALPix)"
#endif
#define NPIX RT_USE_TREECOL_FOR_NH
#endif

/* deposit m/(4pi/NPIX r^2) for 0<r<ShieldingLength, unsoftened, stored for gas cells only; no SigmaEff */
#if defined(RT_TREECOL_HEALPIX) && (defined(GALSF_RESOLVEDISM) || defined(CHEMCOOL))
#define RT_TREECOL_SHIELDINGLENGTH
#define RT_TREECOL_GAS_TARGETS_ONLY
#else
#define RT_TREECOL_EFFECTIVE_COLUMN_FOR_NH /* SigmaEff from the gas bins feeds evaluate_NH_from_GradRho */
#endif

/* Stella with a transported FUV field (TreeRay walk or M1): stellar pixel fluxes arrive dust-attenuated, the chemistry attenuates only the floor pixels */
#if defined(RT_TREECOL_HEALPIX) && defined(CHEMCOOL) && (defined(TREE_RAY) || (defined(GALSF_RESOLVEDISM) && (defined(RADTRANSFER) || defined(RT_M1) || defined(RT_OTVET) || defined(RT_FLUXLIMITEDDIFFUSION) || defined(RT_LOCALRAYGRID))))
#define RT_TREECOL_STELLAR_PIXELS_PREATTENUATED
#endif

/* column species: gas always; per-cell-DGR-weighted gas with the variable or GHOSDT ISRF (or Stella M1); H2 and CO for CHEMCOOL */
#if defined(RT_TREECOL_HEALPIX) && (defined(GALSF_RESOLVEDISM_G0_VARIABLE) || (defined(GHOSDT_G0_CR) && defined(CHEMCOOL)) || defined(RT_TREECOL_STELLAR_PIXELS_PREATTENUATED))
#define RT_TREECOL_DUST_WEIGHTED
#endif
#if defined(RT_TREECOL_HEALPIX) && defined(CHEMCOOL)
#define RT_TREECOL_H2_CO
#endif
#define TREECOL_GAS 0
#ifdef RT_TREECOL_DUST_WEIGHTED
#define TREECOL_DUST 1
#define TREECOL_N_GAS_DUST 2
#else
#define TREECOL_N_GAS_DUST 1
#endif
#ifdef RT_TREECOL_H2_CO
#define TREECOL_H2 (TREECOL_N_GAS_DUST)
#define TREECOL_CO (TREECOL_N_GAS_DUST+1)
#define TREECOL_NSPECIES (TREECOL_N_GAS_DUST+2)
#else
#define TREECOL_NSPECIES (TREECOL_N_GAS_DUST)
#endif
#endif /* RT_USE_TREECOL_FOR_NH */

#if defined(GALSF_RESOLVEDISM_G0_VARIABLE) && !defined(RT_TREECOL_HEALPIX)
#error "GALSF_RESOLVEDISM_G0_VARIABLE needs HEALPix tree columns: RT_USE_TREECOL_FOR_NH=12*NSIDE^2"
#endif

#endif /* GIZMO_TREECOL_CONFIG_H */
