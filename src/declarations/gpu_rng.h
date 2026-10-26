/* gpu_rng.h — GPU-safe counter-based random number generator for GIZMO.
 *
 * Stateless RNG suitable for use inside Kokkos parallel_for lambdas. Built on
 * the SplitMix64 finalizer mixed with a 64-bit (key, counter) pair. Each call
 * takes an explicit (key, counter) and returns a deterministic output —
 * independent of thread id, rank count, or call ordering. Reproducible
 * across runs with the same (key, counter) inputs.
 *
 * Intended usage:
 *   - key:     whose draw it is. For one particle, gizmo_rng_particle_event_key(P[i])
 *              (identity, type, single-precision position).
 *   - counter: which draw, gizmo_rng_event_counter(SALT, event, draw_index): SALT =
 *              gizmo_loop_rng_salt("<call site>") names the kind of event, `event` is
 *              when it happens (Ti_Current, or 0 for a draw that must not change with
 *              time), `draw_index` which of the event's numbers this is. Each field is
 *              folded in separately (gizmo_rng_combine); additive or shifted counters
 *              are not used for new draws, since they collide (draw k of one stream is
 *              draw k-1 of the next).
 *   The device pair draws (SIDM, AGS, sink feed, radiation-pressure pair gate) build
 *   their counters as (Ti_Current << 32) ^ SALT ^ tag, keyed on both particles.
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


/* Fold one more value into a running hash. A tuple (identity, call site, event, draw, ...) is hashed one field at
   a time, which keeps distinct tuples apart; additive offsets do not (with ID+k, draw k of one particle is draw k-1
   of the next). */
GIZMO_RNG_INLINE uint64_t gizmo_rng_combine(uint64_t state, uint64_t value)
{
    return gizmo_gpu_splitmix64(state ^ value);
}


/* The stochastic identity of a particle: its ID, the child number a split assigns, and the split generation.
   Split daughters keep their parent's ID, so all three are needed to tell them apart. */
GIZMO_RNG_INLINE uint64_t gizmo_rng_particle_identity(uint64_t id, uint64_t child_number, uint64_t generation)
{
    return gizmo_rng_combine(gizmo_rng_combine(gizmo_rng_combine(0, generation), child_number), id);
}


/* The counter of one random event: what kind of event (`site_salt`, gizmo_loop_rng_salt of a name per call
   site), when (`event`: the integer time Ti_Current, or 0 for a draw that must not change with time), and which
   of the numbers the event needs (`draw_index`). */
GIZMO_RNG_INLINE uint64_t gizmo_rng_event_counter(uint64_t site_salt, uint64_t event, uint64_t draw_index)
{
    return gizmo_rng_combine(gizmo_rng_combine(site_salt, event), draw_index);
}


/* Compile-time FNV-1a 64-bit hash of a loop-name string. Used to derive a
 * per-loop salt that gets XOR-mixed into the counter base of every RNG draw
 * inside that loop. Rationale: counter-based RNG keyed only on
 * (Ti_Current, ID_pair, tag) gives identical streams across DIFFERENT loops
 * that happen to draw on the same (Ti, ID_pair, tag) — i.e., correlated
 * outcomes that the original time-advancing GSL RNG could never produce by
 * construction. Adding a loop-domain salt restores stream independence
 * across loops without changing parity within a loop.
 *
 * Use as:
 *   static constexpr uint64_t MY_LOOP_RNG_SALT = gizmo_loop_rng_salt("my_loop");
 *   uint64_t counter = gizmo_rng_event_counter(MY_LOOP_RNG_SALT, Ti_Current, draw_index);
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


/*! Mix one position into 64 bits, for keying a draw where identifiers cannot.
 *
 * Identifiers are not unique -- every spawned wind cell carries one stamped value and an input can
 * hold duplicates of its own -- so a key built only from identifiers hands every member of a
 * degenerate group the same stream, and they all draw the same number and act together. A position
 * separates them, because distinct particles occupy distinct points.
 *
 * It only needs to separate distinct positions, not to be a good hash of the geometry: it feeds a
 * counter-based RNG whose own mixing does that work. Bit-casting the doubles keeps it exact, so two
 * evaluations reading the same coordinates agree bit for bit -- which is what lets a symmetric pair
 * draw agree on both sides. */
GIZMO_RNG_INLINE uint64_t gizmo_position_mix(double x, double y, double z)
{
    union { double d; uint64_t u; } cx, cy, cz;
    cx.d = x; cy.d = y; cz.d = z;
    uint64_t h = cx.u * 0x9E3779B97F4A7C15ULL;
    h ^= (cy.u + 0xBF58476D1CE4E5B9ULL + (h << 6) + (h >> 2));
    h ^= (cz.u + 0x94D049BB133111EBULL + (h << 6) + (h >> 2));
    h ^= h >> 31;
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

/* The random-number key of one particle that stays fixed while it moves: its identity and its type. It
   changes only when the particle's type, child number or generation does, and so serves draws that must
   not change from step to step. Works on any particle structure carrying these fields. */
template <class Particle>
GIZMO_RNG_INLINE uint64_t gizmo_rng_particle_identity_key(const Particle &p)
{
    return gizmo_rng_combine(gizmo_rng_particle_identity((uint64_t) p.ID, (uint64_t) p.ID_child_number, (uint64_t) p.ID_generation), (uint64_t) p.Type);
}

/* The random-number key of one particle at the current instant: its identity key and its position at
   single precision. The position separates particles that share an identity (spawned wind cells all carry
   one ID; an initial condition can hold duplicates); at single precision, the round-off by which the same
   particle's position differs between runs on different numbers of ranks changes the key only rarely (when
   it crosses a single-precision boundary). */
template <class Particle>
GIZMO_RNG_INLINE uint64_t gizmo_rng_particle_event_key(const Particle &p)
{
    return gizmo_rng_combine(gizmo_rng_particle_identity_key(p), gizmo_position_mix((float) p.Pos[0], (float) p.Pos[1], (float) p.Pos[2]));
}

#endif /* GIZMO_GPU_RNG_H */
