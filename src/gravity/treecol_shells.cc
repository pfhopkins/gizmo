/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS: after the gravity walk, the global cap on the attenuated stellar bands. */
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <Kokkos_Core.hpp>
#include "../declarations/gpu_all_mirror.h"  /* before allvars.h */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../declarations/gpu_stage_launch.h"
#include "../system/loop_routes.h"
#include "gravtree_treecol_pixels.h"

#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS
/* Each target integrates its rays alone, so the implied absorption sum_cells sigma N_H F can exceed what the stars emit.
   Above the emission (root node, 6-13.6 eV) the active cells' bands scale down; never up (per band set) */
void treecol_shells_photon_cap(void)
{
    const double L_emit = (Nodes[All.TreeNodeIndexBase].stellar_lum[RT_FREQ_BIN_PHOTOELECTRIC] + Nodes[All.TreeNodeIndexBase].stellar_lum[RT_FREQ_BIN_GENERIC_USER_FREQ]) * UNIT_LUM_IN_CGS;
    if(!(L_emit > 0)) {return;} /* no light: nothing absorbed; the root is the same on every rank */
    const double to_NH = UNIT_MASS_IN_CGS * HYDROGEN_MASSFRAC / PROTONMASS_CGS;
    /* both loops are one Kokkos kernel each over the active list, on the device or on the host execution space (gizmo_stage_on_device;
       the host space runs the list in order, so its sum is that of a plain loop) */
    const int n = (int) ActiveParticleList.size(); int on_device = gizmo_stage_on_device(n);
    int *act = gizmo_stage_index_alloc(&on_device, n, "treecol_shells_active"); /* the active list where the kernel can read it */
    if(n > 0) {memcpy(act, ActiveParticleList.data(), (size_t) n * sizeof(int));}
    if(on_device) {GIZMO_GPU_ENSURE_ALL_FRESH();}
    const int *ka = act;
    struct particle_data *kp = P; struct gas_cell_data *kc = CellP;
    double abs_glob = 0;
    double abs_loc = gizmo_stage_sum<double>("treecol_shells_abs", on_device, n, KOKKOS_LAMBDA(int a, double &sum) {
        const int i = ka[a];
        if(kp[i].Type != 0 || kp[i].Mass <= 0) {return;}
        double m[TREECOL_PIXELS_NSPECIES], u_uv = 0; treecol_pixels_species_masses(kp, kc, i, m);
        for(int k = 0; k < TREECOL_PIXELS_N; k++) {u_uv += TREECOL_PIXELS_EGY_UV(kc[i], k);}
        sum += u_uv * TREECOL_SHELLS_SIGMA_UV * m[TREECOL_PIXELS_DUST] * to_NH * C_LIGHT_CGS; /* F = u c, physical */
    });
    loop_route_count("treecol_shells_abs", on_device ? LOOP_ROUTE_DEVICE_KERNEL : LOOP_ROUTE_HOST_KERNEL, n);
    MPI_Allreduce(&abs_loc, &abs_glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(abs_glob <= L_emit || abs_glob <= 0) {gizmo_stage_index_free(on_device, act); return;}
    const double f = L_emit / abs_glob;
    gizmo_stage_for("treecol_shells_scale", on_device, n, KOKKOS_LAMBDA(int a) {
        const int i = ka[a];
        if(kp[i].Type != 0) {return;}
        for(int k = 0; k < TREECOL_PIXELS_N; k++) {kc[i].TreeColEgyUV[k] = treecol_pixel_store(TREECOL_PIXELS_EGY_UV(kc[i], k) * f, 1.0, TREECOL_PIXELS_LOG16_MIN_EGY);}
        kc[i].TreeColEgyLW *= f; kc[i].TreeColEgyNUV *= f; kc[i].TreeColEgyOPT *= f;
    });
    loop_route_count("treecol_shells_scale", on_device ? LOOP_ROUTE_DEVICE_KERNEL : LOOP_ROUTE_HOST_KERNEL, n);
    gizmo_stage_index_free(on_device, act);
    if(ThisTask == 0) {printf("TreeCol shells: photon cap engaged, f=%g (L_emit=%g, L_abs_implied=%g erg/s)\n", f, L_emit, abs_glob);}
}
#endif
