#ifndef RESOLVEDISM_STAR_AGE_H
#define RESOLVEDISM_STAR_AGE_H
/* true time since birth [yr]; evaluate_stellar_age_Gyr floors at 1e-5 Gyr, which biases rate integrals over the first 10 kyr */
static inline double resolvedism_star_age_yr(int i) {return DMAX(evaluate_time_since_t_initial_in_Gyr(P[i].StellarAge), 0.0) * 1.0e9;}
#endif
