#ifndef RESOLVEDISM_RNG_H
#define RESOLVEDISM_RNG_H

/* Rank-independent random streams for Stella draws: seeded from particle identity and integer time, never from rank or call order. */

/* Draw labels: one fixed constant per kind of draw, so different draws of the same particle at the same time are independent */
#define RESOLVEDISM_RNG_DRAW_IMF_MASS 0x494D464D41535331ULL /* IMF mass draw (incl. forced-SF redraw) */
#define RESOLVEDISM_RNG_DRAW_SF_ACCEPT 0x5346414343455054ULL /* LYRA SF acceptance */
#define RESOLVEDISM_RNG_DRAW_IA 0x5459504549414E4EULL       /* Type Ia hazard */
#define RESOLVEDISM_RNG_DRAW_STOCH_IMF 0x53544F4348494D46ULL /* STOCHASTIC_IMF massive-star draw */

static inline unsigned long long resolvedism_rng_mix64(unsigned long long z)
{
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* stream state for particle i at the current integer time: (ID, child number, generation, Ti_Current, draw label) */
static inline unsigned long long resolvedism_rng_seed(int i, unsigned long long draw_label)
{
    unsigned long long s = resolvedism_rng_mix64(draw_label);
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
    if(All.RNGRealisation) {s = resolvedism_rng_mix64(s ^ (unsigned long long)All.RNGRealisation);} /* ResolvedISM_RNGRealisation: independent realisation; 0 keeps the default streams */
#endif
    s = resolvedism_rng_mix64(s ^ (unsigned long long)P[i].ID);
    s = resolvedism_rng_mix64(s ^ (unsigned long long)P[i].ID_child_number);
    s = resolvedism_rng_mix64(s ^ (unsigned long long)P[i].ID_generation);
    s = resolvedism_rng_mix64(s ^ (unsigned long long)All.Ti_Current);
    return s;
}

/* next uniform in [0,1) from the stream (splitmix64) */
static inline double resolvedism_rng_uniform(unsigned long long *state)
{
    *state += 0x9E3779B97F4A7C15ULL;
    return get_random_number_keyed(*state);
}

#endif
