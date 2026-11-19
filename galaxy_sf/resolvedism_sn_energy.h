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

/* by death type: 1e51 for every explosion, 0 for failed SNe and direct collapse; with RESOLVED_ISM_VARIABLE_SN_ENERGY the PISN
 * energy from the He core and ECSN 1e50 (Kitaura, Janka & Hillebrandt 2006) */
static inline double resolvedism_sn_energy_erg(int is_pisn, int is_ecsn, int explodes, double M_zams)
{
    if(!explodes) {return 0;}
#ifdef RESOLVED_ISM_VARIABLE_SN_ENERGY
    if(is_pisn) {return resolvedism_pisn_energy_erg(M_zams);}
    if(is_ecsn) {return 1.0e50;}
#else
    (void)is_pisn; (void)is_ecsn; (void)M_zams;
#endif
    return 1.0e51;
}
#endif
