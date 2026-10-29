/* chem_net18_layout.h -- network 18 (RTZ metal ions) species layout; plain C, shared by GIZMO and the C++ backend.
 * TracAbund (n_X/n_H): 0 H2, 1 H+, 2 CO, 3-5 He0-He2, then every stage 0..Z of C N O Ne Mg Si S Fe.
 * ODE: 0 H2, 1 H+, 2 CO, then Z slots per element He..Fe (all stages but one reference stage, which follows
 * from the element pool); energy last.  Electrons follow from the ion charges. */
#ifndef CHEM_NET18_LAYOUT_H
#define CHEM_NET18_LAYOUT_H

#define N18_NEL   10      /* H He C N O Ne Mg Si S Fe */
#define N18_NTRAC 113
#define N18_NODE  104     /* ODE species; energy density is slot N18_NODE */
#define N18_T_H2  0
#define N18_T_HP  1
#define N18_T_CO  2
#define N18_T_HE  3

static const int    n18_Z[N18_NEL]     = {1, 2, 6, 7, 8, 10, 12, 14, 16, 26};
static const int    n18_toff[N18_NEL]  = {-1, 3, 6, 13, 21, 30, 41, 54, 69, 86};   /* TracAbund slot of stage 0 */
static const int    n18_ooff[N18_NEL]  = {-1, 3, 5, 11, 18, 26, 36, 48, 62, 78};   /* first ODE slot of each element */
static const double n18_amass[N18_NEL] = {1.008, 4.0026, 12.0107, 14.0067, 15.9994, 20.1797, 24.305, 28.0855, 32.065, 55.854};
/* RTZ solar abundances per H and gas-phase fractions at MW dust-to-gas (rtz_main.c initialize_elements) */
static const double n18_zsun[N18_NEL]  = {1.0, 8.51e-2, 2.69e-4, 6.76e-5, 4.90e-4, 8.51e-5, 3.98e-5, 3.24e-5, 1.32e-5, 3.16e-5};
static const double n18_depl[N18_NEL]  = {1.0, 1.0, 0.5, 0.6, 0.73, 1.0, 0.16, 0.1, 1.0, 0.01};

/* electrons per H from a TracAbund vector */
static inline double n18_electrons_trac(const double *ab)
{
    double xe = ab[N18_T_HP];
    int e, j;
    for(e = 1; e < N18_NEL; e++) for(j = 1; j <= n18_Z[e]; j++) xe += j * ab[n18_toff[e] + j];
    return xe;
}

/* free particles per H (atoms, ions, molecules, electrons) from a TracAbund vector; abhe if He is unset */
static inline double n18_particles_trac(const double *ab, double abhe)
{
    double xHI = 1.0 - 2.0 * ab[N18_T_H2] - ab[N18_T_HP];
    double n = (xHI > 0 ? xHI : 0) + ab[N18_T_HP] + ab[N18_T_H2] + ab[N18_T_CO], he = 0;
    int e, j;
    for(j = 0; j < 3; j++) he += ab[N18_T_HE + j];
    n += (he > 0) ? he : abhe;
    for(e = 2; e < N18_NEL; e++) for(j = 0; j <= n18_Z[e]; j++) n += ab[n18_toff[e] + j];
    return n + n18_electrons_trac(ab);
}

#endif
