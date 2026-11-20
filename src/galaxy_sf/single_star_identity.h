/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* Identity of a particle created from a parent (a spawned star, a split cell, an SN ejecta cell) and of a merged cell, in one
   place: shared by sfr_eff.cc, merge_split.cc and the unit test (templates on any struct with the identity fields).
   The generation counter rolls over after 30, so (ID, child number, generation) repeats after 31 children of one particle.
   Structs with ID_wraps and ID_lineage (Stella builds, GALSF_SINGLE_STARS_FROM_IMF) also count the roll-overs and, once a
   lineage has rolled over, give each new particle a hash of its parent's state at creation (stella_dev 18150a30d, 5c11a92a9).
   Both enter the random-number key only when nonzero, so particles whose lineage never rolled over keep their streams. */
#ifndef SINGLE_STAR_IDENTITY_H
#define SINGLE_STAR_IDENTITY_H
#include <stdint.h>
#include <type_traits>
#include <utility>

/* host and device (the radiation-pressure walk draws with these keys): as GIZMO_RNG_INLINE in declarations/gpu_rng.h */
#ifdef GIZMO_GPU_COMPILER
#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif
#define SINGLE_STAR_IDENTITY_INLINE KOKKOS_INLINE_FUNCTION
#else
#define SINGLE_STAR_IDENTITY_INLINE static inline
#endif

template <class T, class = void> struct single_star_has_identity_wraps : std::false_type {};
template <class T> struct single_star_has_identity_wraps<T, std::void_t<decltype(std::declval<T &>().ID_wraps), decltype(std::declval<T &>().ID_lineage)>> : std::true_type {};

SINGLE_STAR_IDENTITY_INLINE uint64_t single_star_identity_mix64(uint64_t x) /* splitmix64 finaliser */
{
    x += 0x9E3779B97F4A7C15ULL; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL; x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL; return x ^ (x >> 31);
}

/* lineage key of a particle created now from parent p (call before p's counter advances): 0 until p's lineage has rolled
   over, then a hash of p's whole identity state, which does not repeat */
template <class Particle> SINGLE_STAR_IDENTITY_INLINE uint64_t single_star_lineage_of(const Particle &p)
{
    if(p.ID_wraps == 0 && p.ID_lineage == 0) {return 0;}
    uint64_t s = single_star_identity_mix64(0x4C494E4541474531ULL ^ (uint64_t)p.ID_lineage);
    s = single_star_identity_mix64(s ^ (uint64_t)p.ID_child_number); s = single_star_identity_mix64(s ^ (uint64_t)p.ID_generation);
    s = single_star_identity_mix64(s ^ (uint64_t)p.ID_wraps);
    return s ? s : 1;
}

/* child c of parent p: GIZMO's split rule (child number = parent's + 2^generation; the parent's generation advances and rolls
   over after 30; the child takes the parent's new generation), plus the roll-over count and lineage key where present */
template <class Particle> SINGLE_STAR_IDENTITY_INLINE void single_star_identity_from_parent(Particle &p, Particle &c)
{
    c.ID_child_number = p.ID_child_number + (decltype(p.ID_child_number))(1 << ((int)p.ID_generation));
    if constexpr(single_star_has_identity_wraps<Particle>::value) {c.ID_lineage = (decltype(c.ID_lineage))single_star_lineage_of(p);}
    p.ID_generation = p.ID_generation + 1;
    if(p.ID_generation > 30) {
        p.ID_generation = 0;
        if constexpr(single_star_has_identity_wraps<Particle>::value) {p.ID_wraps++;}
    }
    c.ID_generation = p.ID_generation;
    if constexpr(single_star_has_identity_wraps<Particle>::value) {c.ID_wraps = p.ID_wraps;}
}

/* merge of i into j: j takes i's identity (with its lineage key) when swap_ids; the roll-over count never drops below either's */
template <class Particle> SINGLE_STAR_IDENTITY_INLINE void single_star_identity_merge(const Particle &i, Particle &j, int swap_ids)
{
    if(swap_ids) {j.ID = i.ID; j.ID_child_number = i.ID_child_number; j.ID_generation = i.ID_generation;}
    if constexpr(single_star_has_identity_wraps<Particle>::value) {
        if(swap_ids) {j.ID_lineage = i.ID_lineage;}
        if(i.ID_wraps > j.ID_wraps) {j.ID_wraps = i.ID_wraps;}
    }
}

/* the part of a particle's identity beyond (ID, child number, generation, type), mixed into its random-number key when nonzero */
template <class Particle> SINGLE_STAR_IDENTITY_INLINE uint64_t single_star_identity_extra(const Particle &p)
{
    if constexpr(single_star_has_identity_wraps<Particle>::value) {
        if(p.ID_wraps == 0 && p.ID_lineage == 0) {return 0;}
        uint64_t s = single_star_identity_mix64(0x5752415053ULL ^ (uint64_t)p.ID_wraps);
        s = single_star_identity_mix64(s ^ (uint64_t)p.ID_lineage);
        return s ? s : 1;
    } else {
        (void)p; return 0;
    }
}

#endif
