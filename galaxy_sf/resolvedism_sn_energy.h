/* Copyright (c) 2026 Ulrich Steinwandel
 * Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
#ifndef RESOLVEDISM_SN_ENERGY_H
#define RESOLVEDISM_SN_ENERGY_H
/* Supernova explosion energies [erg], shared by the deposit, the ledgers and the logs. */

/* PISN: Heger & Woosley 2002 (astro-ph/0107037) via the Wise+12a fit, E = [5.0 + 1.304 (M_He - 64)] 1e51, M_He = (13/24)(M - 20); He core held to 64-133 Msun */
static inline double resolvedism_pisn_energy_erg(double M_zams)
{
    double M_He = (13.0 / 24.0) * (M_zams - 20.0);
    if(M_He < 64.0) {M_He = 64.0;}
    if(M_He > 133.0) {M_He = 133.0;}
    return (5.0 + 1.304 * (M_He - 64.0)) * 1.0e51;
}

/* core-collapse-type death: 1e51 for explosive types (PISN from the He core with GALSF_RESOLVEDISM_PISN_ENERGY_FROM_HELIUM_CORE), 0 for failed SNe and direct BHs */
static inline double resolvedism_sn_energy_erg(int is_pisn, int explodes, double M_zams)
{
    if(!explodes) {return 0;}
#ifdef GALSF_RESOLVEDISM_PISN_ENERGY_FROM_HELIUM_CORE
    if(is_pisn) {return resolvedism_pisn_energy_erg(M_zams);}
#endif
    return 1.0e51;
}
#endif
