/* transport_subcycle.h - data structures and declarations for transport subcycling
 *
 * When TRANSPORT_SUBCYCLE is enabled, RT and/or CR transport can take multiple smaller
 * sub-steps per hydro step. Each sub-step does a full MPI-aware flux exchange
 * (code_block_xchange pattern) to correctly handle cross-domain interactions.
 *
 * Written by Mike Grudic for GIZMO.
 */

#ifndef TRANSPORT_SUBCYCLE_H
#define TRANSPORT_SUBCYCLE_H

#ifdef TRANSPORT_SUBCYCLE

/* function declarations */
void transport_subcycle_exchange_fluxes(void);  /* full code_block_xchange RT flux computation */
void transport_subcycle_kick(void);
void transport_subcycle_prepare_topology(void); /* build active list + ghost pool + shared CSR just before the subcycle loop (run.cc) */

#endif // TRANSPORT_SUBCYCLE
#endif // TRANSPORT_SUBCYCLE_H
