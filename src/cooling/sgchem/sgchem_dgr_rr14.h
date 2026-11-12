/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
#ifndef SGCHEM_DGR_RR14_H
#define SGCHEM_DGR_RR14_H
#include <math.h>
#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION static inline
#endif
/* Remy-Ruyer+14 (arXiv:1312.3442, Table 1, X_CO,Z broken power law) dust-to-gas ratio over the linear law, both 1 at Zsun:
 * log(G/D) = 2.21 + 1.00 (8.69 - x) above x = 8.10, 0.96 + 3.10 (8.69 - x) below, x = 12 + log(O/H) = 8.69 + log(Z/Zsun) */
KOKKOS_INLINE_FUNCTION double sgchem_dgr_rr14_over_linear(double Z_over_Zsun)
{
    if(Z_over_Zsun <= 0) {return 1.0;} /* DGR = 0 either way */
    double dx = -log10(Z_over_Zsun); /* 8.69 - x */
    if(8.69 - dx > 8.10) {return 1.0;}
    return pow(10.0, (2.21 + 1.00 * dx) - (0.96 + 3.10 * dx));
}
#endif
