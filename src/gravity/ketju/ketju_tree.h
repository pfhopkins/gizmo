/* KETJU <-> gravity tree: the leaf-pair force the walk adds (so the negative half-kicks remove exactly that),
   and the chain-sphere node opening shared by the device and host walks */
#ifndef KETJU_TREE_H
#define KETJU_TREE_H
#if defined(KETJU_REGULARIZATION)

/* 1 if a node (side len) overlaps a sphere (radius 0 = none): the walk opens nodes around a chain member, so its partners
   are leaf pairs; the LET sender does the same for a receiver's chains. d is the (wrapped) node-centre minus sphere-centre offset. */
KOKKOS_INLINE_FUNCTION int ketju_node_overlaps_chain_sphere(double dx, double dy, double dz, double sphere_radius, double len)
{
    if(!(sphere_radius > 0)) {return 0;}
    double reach = sphere_radius + 0.866025404 * len; /* node bounding radius = len*sqrt(3)/2 */
    return (dx*dx + dy*dy + dz*dz < reach*reach) ? 1 : 0;
}

#ifdef TEST_KETJU_WALK_OPENINGS
#include <Kokkos_Core.hpp>
extern long long ketju_walk_openings;
#define KETJU_WALK_OPENING_NOTE Kokkos::atomic_add(&ketju_walk_openings, 1LL)
#else
#define KETJU_WALK_OPENING_NOTE
#endif

#ifdef KETJU_TREE_PAIR_KERNEL
/* scalar factors of one leaf-pair interaction, through the walk's own pair kernel (gravtree_force_kernel.h) */
static inline void tree_pair_gravity_factors(double r, double r2, double mass, double h, double h_p, int ptype, int ptype_sec,
                                             double zeta, double zeta_sec, double pmass, int ags_bitflag_primary,
                                             double &fac_accel, double &fac_pot, double &fac_tidal, double &fac2_tidal)
{
    grav_force_pair_t o = grav_force_pair(r, r2, mass, h, h_p, ptype, ptype_sec, pmass, zeta, zeta_sec, ags_bitflag_primary);
    fac_accel = o.fac_accel; fac_pot = 0; fac_tidal = 0; fac2_tidal = 0;
#ifdef EVALPOTENTIAL
    fac_pot = o.fac_pot;
#endif
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
    fac_tidal = o.fac_tidal; fac2_tidal = o.fac2_tidal;
#endif
}
#endif

#endif
#endif
