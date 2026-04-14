/* cooling_tables.h — Consolidated cooling table data structure.
 *
 * Holds all cooling/ionization table data that per-particle cooling functions
 * need to access. With GPU offload, the struct is __managed__ so device kernels
 * can access it directly. Pointer members point to separately allocated managed
 * memory (Kokkos SharedSpace on GPU, mymalloc on CPU).
 *
 * This replaces ~15 individual static variables that were scattered at file scope
 * in cooling.cc, making the cooling tables accessible from any translation unit
 * via the header (required for cooling_functions.h to work across TUs).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef COOLING_TABLES_H
#define COOLING_TABLES_H

#define NCOOLTAB  2000 /* number of entries in cooling rate tables (temperature grid) */

struct cooling_tables_t {
    /* Temperature grid */
    double Tmin, Tmax, deltaT;

    /* Cooling/ionization rate tables [NCOOLTAB+1] entries each, indexed by temperature */
    double *BetaH0;      /* collisional ionization cooling: HI */
    double *BetaHep;     /* collisional ionization cooling: HeII */
    double *Betaff;      /* free-free (bremsstrahlung) cooling */
    double *AlphaHp;     /* recombination rate: HII */
    double *AlphaHep;    /* recombination rate: HeII */
    double *Alphad;      /* dielectronic recombination: HeII */
    double *AlphaHepp;   /* recombination rate: HeIII */
    double *GammaeH0;    /* collisional ionization rate: HI */
    double *GammaeHe0;   /* collisional ionization rate: HeI */
    double *GammaeHep;   /* collisional ionization rate: HeII */

    /* UV background parameters (set by IonizeParams from TREECOOL table) */
    double J_UV;         /* UV intensity normalization */
    double gJH0;         /* photoionization rate: HI */
    double gJHep;        /* photoionization rate: HeII */
    double gJHe0;        /* photoionization rate: HeI */
    double epsH0;        /* photoheating rate: HI */
    double epsHep;       /* photoheating rate: HeII */
    double epsHe0;       /* photoheating rate: HeI */

#ifdef COOL_METAL_LINES_BY_SPECIES
    /* Metal-line cooling tables (interpolation grids over nH, T, redshift) */
    float *SpCoolTable0; /* cooling table at redshift z0 */
    float *SpCoolTable1; /* cooling table at redshift z1 (for interpolation; cosmological only) */
#endif
};

/* Global cooling tables instance. Defined in cooling.cc.
   With GPU offload: __managed__ static (accessible from device).
   Without: plain static. */
extern struct cooling_tables_t CoolTables;

#endif /* COOLING_TABLES_H */
