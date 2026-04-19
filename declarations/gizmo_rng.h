/* gizmo_rng.h — Stateful CPU pseudo-random number generator for GIZMO.
 *
 * Implements xoshiro256** (Blackman & Vigna 2019), a 256-bit (4×uint64_t)
 * generator that is fast, serializable with a fixed-size state, passes
 * BigCrush, and requires no dynamic allocation.  It replaces the global
 * `gsl_rng *random_generator` and `gsl_rng *StRng` that previously depended
 * on GSL.
 *
 * Seeding uses SplitMix64 to expand a 64-bit seed into the four-word state
 * (standard practice for xoshiro initialization; avoids poor coverage of
 * zero-heavy seeds).
 *
 * Serialization: the state is exactly 4×8 = 32 bytes, written/read directly
 * by restart.cc via `byten(rng.s, sizeof(rng.s), modus)`.  This changes the
 * restart-file format relative to the old GSL-ranlxd1 state; existing restart
 * files must be discarded and runs restarted from initial conditions.
 *
 * Usage (mirrors gsl_rng interface):
 *   gizmo_rng_t rng;
 *   gizmo_rng_init(&rng, seed);
 *   double u = gizmo_rng_uniform(&rng);      // [0, 1)
 *   double g = gizmo_rng_gaussian(&rng);     // standard normal
 *   double g_s = gizmo_rng_gaussian(&rng) * sigma;  // N(0, sigma^2)
 *
 * Save / restore state (replacing gsl_rng_state / gsl_rng_size):
 *   gizmo_rng_t saved = rng;   // copy by value
 *   rng = saved;               // restore by value
 *
 * Not GPU-callable (stateful pointer semantics don't map to massively-parallel
 * kernels).  For GPU use, see declarations/gpu_rng.h (counter-based, stateless).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef GIZMO_RNG_H
#define GIZMO_RNG_H

#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 256-bit state for xoshiro256**. */
typedef struct { uint64_t s[4]; } gizmo_rng_t;


/* SplitMix64 step — used only during seeding. */
static inline uint64_t gizmo_rng_sm64_(uint64_t x)
{
    uint64_t z = x + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/* Seed the generator from a 64-bit integer. */
static inline void gizmo_rng_init(gizmo_rng_t *r, uint64_t seed)
{
    r->s[0] = gizmo_rng_sm64_(seed);
    r->s[1] = gizmo_rng_sm64_(r->s[0]);
    r->s[2] = gizmo_rng_sm64_(r->s[1]);
    r->s[3] = gizmo_rng_sm64_(r->s[2]);
}

/* xoshiro256** core — returns a well-mixed 64-bit output. */
static inline uint64_t gizmo_rng_next_(gizmo_rng_t *r)
{
    uint64_t res = r->s[1] * 5;
    res = ((res << 7) | (res >> 57)) * 9;  /* rotl(s1*5, 7) * 9 */
    uint64_t t  = r->s[1] << 17;
    r->s[2] ^= r->s[0];
    r->s[3] ^= r->s[1];
    r->s[1]  ^= r->s[2];
    r->s[0]  ^= r->s[3];
    r->s[2]  ^= t;
    r->s[3]   = (r->s[3] << 45) | (r->s[3] >> 19);  /* rotl(s3, 45) */
    return res;
}

/* Uniform double in [0, 1). Uses upper 53 bits for full mantissa coverage. */
static inline double gizmo_rng_uniform(gizmo_rng_t *r)
{
    return (double)(gizmo_rng_next_(r) >> 11) * (1.0 / 9007199254740992.0);
}

/* Standard-normal draw via Box-Muller. Consumes two uniform draws. */
static inline double gizmo_rng_gaussian(gizmo_rng_t *r)
{
    double u1 = gizmo_rng_uniform(r);
    double u2 = gizmo_rng_uniform(r);
    if(u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* Uniform draw in [low, high). */
static inline double gizmo_rng_range(gizmo_rng_t *r, double low, double high)
{
    return low + (high - low) * gizmo_rng_uniform(r);
}

/* Poisson draw with mean lambda.
 * lambda < 30: Knuth's exact algorithm (product of exponentials).
 * lambda >= 30: Normal approximation (accurate to ~0.1% for large means). */
static inline unsigned int gizmo_rng_poisson(gizmo_rng_t *r, double lambda)
{
    if(lambda <= 0) return 0;
    if(lambda < 30.0) {
        double L = exp(-lambda), p = 1.0;
        unsigned int k = 0;
        do { k++; p *= gizmo_rng_uniform(r); } while(p > L && k < 10000);
        return k - 1;
    }
    double x = gizmo_rng_gaussian(r) * sqrt(lambda) + lambda;
    return (x > 0) ? (unsigned int)(x + 0.5) : 0;
}

#ifdef __cplusplus
}
#endif

#endif /* GIZMO_RNG_H */
