/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* One single-star formation event: the decision of which stars form, as a pure function, shared by sfr_eff.cc and the unit tests. */
#ifndef SINGLE_STAR_SF_EVENT_H
#define SINGLE_STAR_SF_EVENT_H
#include <math.h>
#include <vector>

/* one event of expected stellar mass X [Msun]: M0 is the first mass-weighted draw (>= M_lo), draw(k) gives draw k >= 1,
   accept(k) the uniform of draw k. n = ceil(X/M_lo) sub-events of X/n keep each acceptance (X/n)/M <= 1, so every formed star
   follows the number IMF; n = 1 is the single draw. Formed masses go to out; returns their number. */
template<class D, class A> static inline int single_star_sf_event(double X, double M_lo, double M0, D &&draw, A &&accept, std::vector<double> &out)
{
    out.clear();
    long n = (X > M_lo) ? (long)ceil(X / M_lo) : 1;
    for(long k = 0; k < n; k++)
    {
        double M = (k == 0) ? M0 : draw(k);
        if(accept(k) < (X / n) / M) {out.push_back(M);}
    }
    return (int)out.size();
}
#endif
