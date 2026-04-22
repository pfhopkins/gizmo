/* gpu_gravity_tree.cc — Step 13 Phase 3
 *
 * See gpu_gravity_tree.h for design notes.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef OPENMP_GPU_OFFLOAD
#include <Kokkos_Core.hpp>
#endif

/* GPU All mirror precedes allvars.h so nvc++ sees `All` (=All_dev) for
 * eagerly-parsed templates referencing it. Matches density_gpu.cc include order. */
#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "gpu_gravity_tree.h"

#ifdef OPENMP_GPU_OFFLOAD

static struct gpu_gravity_tree_soa_t soa_ = {0};
static int soa_capacity_ = 0;
static int soa_valid_    = 0;

static void free_arrays_(void)
{
    if(soa_.center)   {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.center);   soa_.center   = NULL;}
    if(soa_.len)      {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.len);      soa_.len      = NULL;}
    if(soa_.s)        {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.s);        soa_.s        = NULL;}
    if(soa_.mass)     {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.mass);     soa_.mass     = NULL;}
    if(soa_.sibling)  {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.sibling);  soa_.sibling  = NULL;}
    if(soa_.nextnode) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.nextnode); soa_.nextnode = NULL;}
    if(soa_.bitflags) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.bitflags); soa_.bitflags = NULL;}
    if(soa_.maxsoft)  {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.maxsoft);  soa_.maxsoft  = NULL;}
    if(soa_.N_part)   {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.N_part);   soa_.N_part   = NULL;}
    soa_.nnodes = 0;
}

static int alloc_arrays_(int n)
{
    soa_.center   = (Vec3<MyFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyFloat>));
    soa_.len      = (MyFloat       *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyFloat));
    soa_.s        = (Vec3<MyFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyFloat>));
    soa_.mass     = (MyFloat       *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyFloat));
    soa_.sibling  = (int           *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(int));
    soa_.nextnode = (int           *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(int));
    soa_.bitflags = (unsigned int  *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(unsigned int));
    soa_.maxsoft  = (MyFloat       *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyFloat));
    soa_.N_part   = (long          *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(long));
    if(!soa_.center || !soa_.len || !soa_.s || !soa_.mass || !soa_.sibling ||
       !soa_.nextnode || !soa_.bitflags || !soa_.maxsoft || !soa_.N_part) {
        printf("gpu_gravity_tree: kokkos_malloc failed for %d nodes\n", n);
        return 0;
    }
    return 1;
}

static void seed_from_aos_(int n, struct NODE *Nodes_host, struct extNODE * /*Extnodes_host*/)
{
    /* Copy the fields the walk reads. Kept as a straight host-side loop;
     * SharedSpace pages migrate device-side on first kernel touch. The Vec3
     * fields (center, s) currently live inside Nodes[] as Vec3<MyFloat>;
     * straight assignment works because the SoA mirror uses the same type. */
    for(int k = 0; k < n; k++) {
        soa_.center[k]   = Nodes_host[k].center;
        soa_.len[k]      = Nodes_host[k].len;
        soa_.s[k]        = Nodes_host[k].u.d.s;
        soa_.mass[k]     = Nodes_host[k].u.d.mass;
        soa_.sibling[k]  = Nodes_host[k].u.d.sibling;
        soa_.nextnode[k] = Nodes_host[k].u.d.nextnode;
        soa_.bitflags[k] = Nodes_host[k].u.d.bitflags;
        soa_.maxsoft[k]  = Nodes_host[k].maxsoft;
        soa_.N_part[k]   = Nodes_host[k].N_part;
    }
    soa_.nnodes = n;
}

extern "C" void gpu_gravity_tree_acquire(int min_nodes,
                                          struct NODE    *Nodes_host,
                                          struct extNODE *Extnodes_host)
{
    if(min_nodes <= 0) {min_nodes = 1;}

    if(soa_capacity_ >= min_nodes && soa_.center) {
        if(soa_valid_) {
            /* Mirror is in sync — fast path, no copy. */
            return;
        }
        /* Reseed in place. */
        if(Nodes_host) {seed_from_aos_(min_nodes, Nodes_host, Extnodes_host);}
        soa_valid_ = 1;
        return;
    }

    /* Need fresh allocation: capacity grew or first acquire. */
    free_arrays_();
    if(!alloc_arrays_(min_nodes)) {endrun(913101);}
    soa_capacity_ = min_nodes;
    if(Nodes_host) {seed_from_aos_(min_nodes, Nodes_host, Extnodes_host);}
    soa_valid_ = 1;
}

extern "C" void gpu_gravity_tree_invalidate(void)
{
    soa_valid_ = 0;
}

extern "C" void gpu_gravity_tree_release(void)
{
    free_arrays_();
    soa_capacity_ = 0;
    soa_valid_    = 0;
}

extern "C" struct gpu_gravity_tree_soa_t *gpu_gravity_tree_soa(void) {return soa_valid_ ? &soa_ : NULL;}
extern "C" int gpu_gravity_tree_capacity(void)                       {return soa_capacity_;}
extern "C" int gpu_gravity_tree_valid(void)                          {return soa_valid_;}

#endif /* OPENMP_GPU_OFFLOAD */
