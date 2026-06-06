/* gpu_rng.h — GPU-safe counter-based random number generator for GIZMO.
 *
 * Stateless RNG suitable for use inside Kokkos parallel_for lambdas. Built on
 * the SplitMix64 finalizer mixed with a 64-bit (key, counter) pair. Each call
 * takes an explicit (key, counter) and returns a deterministic output —
 * independent of thread id, rank count, or call ordering. Reproducible
 * across runs with the same (key, counter) inputs.
 *
 * Intended usage:
 *   - key:     per-particle identity, typically `P[i].ID` (MyIDType → uint64).
 *   - counter: a per-call stream identifier, typically constructed as
 *              `(All.Ti_Current << 8) | tag`, where `tag` is a small
 *              per-kernel nibble selecting an independent sub-stream
 *              (e.g. 0 = SIDM scatter direction, 1 = CBE basis split,
 *              2 = sink stochastic accretion, ...). This guarantees that
 *              different kernels sample independent streams from the same
 *              particle without correlation across kernel calls on the same
 *              timestep, and different timesteps give fresh streams.
 *
 * Statistical quality: SplitMix64 passes BigCrush. Sufficient for all
 * stochastic physics in GIZMO (scattering directions, stochastic accretion,
 * basis splits, Monte-Carlo event draws).
 *
 * Not cryptographic. Not a replacement for gsl_rng in host-only contexts
 * that use specific gsl distributions (chi-squared, Poisson, etc.) —
 * extend this header if those are needed on GPU.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef GIZMO_GPU_RNG_H
#define GIZMO_GPU_RNG_H

#include <stdint.h>
#include <math.h>

/* KOKKOS_INLINE_FUNCTION when compiling a GPU TU; plain inline otherwise. */
#ifdef GIZMO_GPU_COMPILER
  #ifndef KOKKOS_INLINE_FUNCTION
    #define KOKKOS_INLINE_FUNCTION inline
  #endif
  #define GIZMO_RNG_INLINE KOKKOS_INLINE_FUNCTION
#else
  #define GIZMO_RNG_INLINE static inline
#endif


/* SplitMix64 finalizer (Steele, Lea, Flood 2014). Takes a 64-bit state and
   emits a well-mixed 64-bit output. Good enough for independent streams
   when the input combines (key, counter) uniquely. */
GIZMO_RNG_INLINE uint64_t gizmo_gpu_splitmix64(uint64_t x)
{
    uint64_t z = x + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}


/* Combine (key, counter) into a single 64-bit seed by interleaving halves.
   Avoids trivial collisions when key or counter are small. */
GIZMO_RNG_INLINE uint64_t gizmo_gpu_rng_mix(uint64_t key, uint64_t counter)
{
    /* Rotate key by 32, XOR with counter, then mix a second round so
       the high bits of either input influence all bits of the output. */
    uint64_t k = (key << 32) | (key >> 32);
    return gizmo_gpu_splitmix64(k ^ counter);
}


/* Uniform 64-bit integer draw. */
GIZMO_RNG_INLINE uint64_t gizmo_gpu_rand_u64(uint64_t key, uint64_t counter)
{
    return gizmo_gpu_splitmix64(gizmo_gpu_rng_mix(key, counter));
}


/* Uniform double in [0, 1). Uses upper 53 bits (double mantissa width). */
GIZMO_RNG_INLINE double gizmo_gpu_rand_double(uint64_t key, uint64_t counter)
{
    uint64_t h = gizmo_gpu_rand_u64(key, counter);
    return (double)(h >> 11) * (1.0 / 9007199254740992.0); /* 2^53 */
}


/* Standard-normal draw via Box-Muller. Consumes two uniforms derived from
   consecutive counter values. Caller should not reuse those counters. */
GIZMO_RNG_INLINE double gizmo_gpu_rand_gaussian(uint64_t key, uint64_t counter)
{
    double u1 = gizmo_gpu_rand_double(key, 2 * counter);
    double u2 = gizmo_gpu_rand_double(key, 2 * counter + 1);
    /* Guard against log(0) when u1 happens to be exactly zero. */
    if(u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}


/* Uniform draw in [low, high). */
GIZMO_RNG_INLINE double gizmo_gpu_rand_range(uint64_t key, uint64_t counter, double low, double high)
{
    return low + (high - low) * gizmo_gpu_rand_double(key, counter);
}


/* Compile-time FNV-1a 64-bit hash of a loop-name string. Used to derive a
 * per-loop salt that gets XOR-mixed into the counter base of every RNG draw
 * inside that loop. Rationale: counter-based RNG keyed only on
 * (Ti_Current, ID_pair, tag) gives identical streams across DIFFERENT loops
 * that happen to draw on the same (Ti, ID_pair, tag) — i.e., correlated
 * outcomes that the original time-advancing GSL RNG could never produce by
 * construction. Adding a loop-domain salt restores stream independence
 * across loops without changing parity within a loop (oracle and production
 * both see the same salted stream → oracle compare unaffected).
 *
 * Use as:
 *   static constexpr uint64_t MY_LOOP_RNG_SALT = gizmo_loop_rng_salt("my_loop");
 *   uint64_t counter = ((uint64_t)Ti_Current << 32) ^ MY_LOOP_RNG_SALT ^ tag;
 *
 * Constexpr → resolved at compile time, zero runtime cost, human-legible
 * (the source-of-truth string "my_loop" lives next to the constant).
 *
 * FNV-1a 64-bit reference: Glenn Fowler / Phong Vo / Landon Curt Noll,
 * IETF draft-eastlake-fnv-17 §3. Offset basis and prime are the canonical
 * 64-bit values. */
constexpr uint64_t gizmo_loop_rng_salt(const char *s)
{
    uint64_t h = 0xcbf29ce484222325ULL; /* FNV-1a 64 offset basis */
    while(*s) { h ^= (uint64_t)(unsigned char)(*s++); h *= 0x100000001b3ULL; }
    return h;
}


/* Poisson draw with mean lambda.
 *   lambda < 30: Knuth's exact algorithm — draws uniforms at counter+1, +2, ...
 *                Callers must reserve counter..counter+~3*lambda+1 in their
 *                stream to avoid counter collisions with sibling draws.
 *   lambda >= 30: Normal approximation — single Gaussian draw (two uniforms
 *                 at counter*2 and counter*2+1 via gizmo_gpu_rand_gaussian).
 *
 * Useful for stochastic IMF sampling, sink formation, etc. on GPU. */
GIZMO_RNG_INLINE uint64_t gizmo_gpu_rand_poisson(uint64_t key, uint64_t counter, double lambda)
{
    if(lambda <= 0) return 0;
    if(lambda < 30.0) {
        double L = exp(-lambda), p = 1.0;
        uint64_t k = 0, c = counter;
        do { k++; c++; p *= gizmo_gpu_rand_double(key, c); } while(p > L && k < 10000);
        return k - 1;
    }
    double x = gizmo_gpu_rand_gaussian(key, counter) * sqrt(lambda) + lambda;
    return (x > 0) ? (uint64_t)(x + 0.5) : 0;
}

#endif /* GIZMO_GPU_RNG_H */
