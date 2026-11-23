/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* Placement of the n >= 2 stars a parent cell formed in one call (sfr_collect_gas.cc and the unit test). Without it they keep the
   parent's position to the bit, and a snapshot restart soon after such a birth stops on coincident positions. */
#ifndef SINGLE_STAR_BIRTH_OFFSETS_H
#define SINGLE_STAR_BIRTH_OFFSETS_H
#include <math.h>

#define SINGLE_STAR_BIRTH_OFFSET_FRACTION 0.01 /* of the parent's kernel radius */

/* star k (mass m[k], its own uniforms u[2k], u[2k+1]) moves by SINGLE_STAR_BIRTH_OFFSET_FRACTION h along a direction from its
   uniforms, less the mass-weighted mean of these moves, so the group's centre of mass stays put; velocities are not touched
   (momentum unchanged). Offsets go to d[3k..3k+2]. */
static inline void single_star_birth_offsets(int n, const double *m, const double *u, double h, double *d)
{
    double M = 0, D[3] = {0, 0, 0};
    for(int k = 0; k < n; k++) {
        double mu = 2.0 * u[2*k] - 1.0, phi = 2.0 * M_PI * u[2*k+1], s = sqrt(fmax(0.0, 1.0 - mu * mu)), r = SINGLE_STAR_BIRTH_OFFSET_FRACTION * h;
        d[3*k] = r * s * cos(phi); d[3*k+1] = r * s * sin(phi); d[3*k+2] = r * mu;
        for(int c = 0; c < 3; c++) {D[c] += m[k] * d[3*k+c];}
        M += m[k];
    }
    for(int c = 0; c < 3; c++) {D[c] /= M;}
    for(int k = 0; k < n; k++) {for(int c = 0; c < 3; c++) {d[3*k+c] -= D[c];}}
}

#endif
