/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* Identity of a particle created from a parent (a spawned star, a split cell, an SN ejecta cell) and of a merged cell, in one
   place: shared by sfr_eff.cc, merge_split.cc and the unit test (templates on any struct with the identity fields). */
#ifndef SINGLE_STAR_IDENTITY_H
#define SINGLE_STAR_IDENTITY_H
#include <stdint.h>

/* child c of parent p: GIZMO's split rule (child number = parent's + 2^generation; the parent's generation advances and rolls
   over after 30; the child takes the parent's new generation) */
template <class Particle> static inline void single_star_identity_from_parent(Particle &p, Particle &c)
{
    c.ID_child_number = p.ID_child_number + (decltype(p.ID_child_number))(1 << ((int)p.ID_generation));
    p.ID_generation = p.ID_generation + 1;
    if(p.ID_generation > 30) {p.ID_generation = 0;}
    c.ID_generation = p.ID_generation;
}

/* merge of i into j: j takes i's identity when swap_ids */
template <class Particle> static inline void single_star_identity_merge(const Particle &i, Particle &j, int swap_ids)
{
    if(swap_ids) {j.ID = i.ID; j.ID_child_number = i.ID_child_number; j.ID_generation = i.ID_generation;}
}

/* the part of a particle's identity beyond (ID, child number, generation, type), mixed into its random-number key when nonzero */
template <class Particle> static inline uint64_t single_star_identity_extra(const Particle &) {return 0;}

#endif
