/* Copyright (c) 2026 Ulrich Steinwandel
 * Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
#ifndef RESOLVEDISM_SF_DRAW_H
#define RESOLVEDISM_SF_DRAW_H
/* Stella star-formation draw: Kroupa mass draw and the per-event decision. Pure functions, shared by sfr_eff.cc and unit tests. */
#include <math.h>
#include <vector>

/* IMF accretion donor floor: a gas cell gives mass only above it; m_merge = All.MinMassForParticleMerger */
static inline double resolvedism_imf_donor_floor(double m_merge) {return 0.01 * m_merge;}

/* mass-weighted Kroupa M dN/dM (slopes -0.3 below 0.5 Msun, -1.3 above) by rejection from M^-0.3; uniform() returns [0,1) */
template<class U> static inline double kroupa_draw_mass_weighted(U &&uniform, double M_min, double M_max)
{
    const double A = 0.126512;
    double M;
    do {
        M = pow((pow(M_max, 0.7) - pow(M_min, 0.7)) * uniform() + pow(M_min, 0.7), 1.0 / 0.7);
    } while(uniform() > ((M < 0.5 ? 2.0 * A * pow(M, -0.3) : A * pow(M, -1.3)) / (2.0 * A * pow(M, -0.3))));
    return M;
}

/* one SF event of expected stellar mass X: M0 is the first mass-weighted draw (>= M_lo), draw() gives further ones, accept() a uniform.
   n = ceil(X/M_lo) sub-events of X/n keep each acceptance (X/n)/M <= 1, so every formed star follows the number IMF. Formed masses go to out. */
template<class D, class A> static inline int resolvedism_sf_event(double X, double M_lo, double M0, D &&draw, A &&accept, std::vector<double> &out)
{
    out.clear();
    long n = (X > M_lo) ? (long)ceil(X / M_lo) : 1;
    for(long k = 0; k < n; k++) {
        double M = (k == 0) ? M0 : draw();
        if(accept() < (X / n) / M) {out.push_back(M);}
    }
    return (int)out.size();
}

#endif
