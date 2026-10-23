#ifndef RESOLVEDISM_DEPLETION_H
#define RESOLVEDISM_DEPLETION_H
/* MW grain-locked mass fraction of each element at DGRnormalized = 1, shared by the dust seed and the no-dust chemistry input. */
#define RESOLVEDISM_DEPL_C   0.50   /* gas C/H 1.4e-4 of Asplund09 2.8e-4 (SGChem CarbAbund, gizmo2017 abundc) */
#define RESOLVEDISM_DEPL_SI  0.82   /* Hu+23 MW silicate 3.5e-3 (FeMgSiO4) on Asplund09 Si */
#define RESOLVEDISM_DEPL_O   0.217  /* O held by that silicate at solar Si/O */
#define RESOLVEDISM_DEPL_MAX 0.9    /* cap on the locked fraction of any element */
#endif
