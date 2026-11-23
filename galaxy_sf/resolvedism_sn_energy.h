/* Copyright (c) 2026 Ulrich Steinwandel
 * Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
#ifndef RESOLVEDISM_SN_ENERGY_H
#define RESOLVEDISM_SN_ENERGY_H
/* Supernova explosion energies [erg], shared by the deposit, the ledgers and the logs. */

/* by death type: 1e51 for every explosion, 0 for failed SNe and direct collapse; with RESOLVED_ISM_VARIABLE_SN_ENERGY
 * (Uli 2026-11-19) PISN 1e52 and ECSN 5e50 */
static inline double resolvedism_sn_energy_erg(int is_pisn, int is_ecsn, int explodes)
{
    if(!explodes) {return 0;}
#ifdef RESOLVED_ISM_VARIABLE_SN_ENERGY
    if(is_pisn) {return 1.0e52;}
    if(is_ecsn) {return 5.0e50;}
#else
    (void)is_pisn; (void)is_ecsn;
#endif
    return 1.0e51;
}
#endif
