#ifndef ANEOS_H
#define ANEOS_H

#include "../GIZMO_config.h"

#ifdef EOS_ANEOS

#include <stdio.h>
#include <stdlib.h>

#ifndef ANEOS_MAX_MATERIALS
#define ANEOS_MAX_MATERIALS 7
#endif

/* Storage for one ANEOS material table (SESAME-format, pre-computed rho-T grid) */
struct aneos_table {
    int    mat_id;        /* SESAME material number */
    int    nrho;          /* number of density grid points */
    int    nT;            /* number of temperature grid points */
    int    loaded;        /* 0 = empty slot, 1 = loaded */

    /* Grid axes in log10 space (CGS units: rho in g/cm^3, T in K) */
    double *logrho;       /* log10(rho), length nrho */
    double *logT;         /* log10(T),   length nT   */

    /* Tabulated 2D arrays, row-major [irho * nT + iT], stored in log10 of CGS values */
    double *log_pressure; /* log10(P [dyn/cm^2]) */
    double *log_energy;   /* log10(u [erg/g])    */
    double *log_entropy;  /* log10(S [erg/g/K])  */
    double *log_csound;   /* log10(cs [cm/s])    */
    int    *phase;        /* phase flag (integer, no interpolation) */
    int     has_csound;   /* 1 if sound speed column present in table */
    int     has_phase;    /* 1 if phase column present in table */

    /* Grid bounds and spacing */
    double logrho_min, logrho_max;
    double logT_min, logT_max;
    double d_logrho;      /* uniform spacing in log10(rho) */
    double d_logT;        /* uniform spacing in log10(T)   */
    double inv_d_logrho;  /* 1/d_logrho for fast indexing */
    double inv_d_logT;    /* 1/d_logT   for fast indexing */
};

/* Global table array -- one per material slot */
extern struct aneos_table ANEOS_Tables[ANEOS_MAX_MATERIALS];
extern int ANEOS_Num_Tables_Loaded;

/* ---- Public API ---- */

/* Read a SESAME-format table file into slot mat_index.
   Call once per material from the main thread at startup. */
int aneos_read_table(const char *filename, int mat_index);

/* Free all allocated table memory. */
int aneos_cleanup(void);

/* Main EOS call: given (rho, u) in CGS, compute all thermodynamic quantities.
   T_guess is in/out: pass the previous temperature as initial guess,
   returns the converged temperature.
   phase_out may be NULL if not needed. Returns 0 on success. Thread-safe. */
int aneos_compute(int mat_index,
                  double rho_cgs, double u_cgs,
                  double *T_guess,
                  double *press_cgs,
                  double *cs_cgs,
                  double *entropy_cgs,
                  double *cv_cgs,
                  double *gruneisen,
                  int    *phase_out);

/* Direct lookup from (rho, T) in CGS, no temperature inversion.
   Available for diagnostics and internal use. Thread-safe. */
int aneos_lookup_rhoT(int mat_index,
                      double rho_cgs, double T_cgs,
                      double *press_cgs, double *u_cgs,
                      double *entropy_cgs, double *cs_cgs,
                      int *phase_out);

#endif /* EOS_ANEOS */
#endif /* ANEOS_H */
