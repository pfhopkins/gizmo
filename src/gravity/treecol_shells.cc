/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS: after the gravity walk, the global cap on the attenuated stellar bands. */
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "gravtree_treecol_pixels.h"

#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS
/* Each target integrates its rays alone, so the implied absorption sum_cells sigma N_H F can exceed what the stars emit.
   Above the emission (root node, 6-13.6 eV) the active cells' bands scale down; never up (stella_dev, per band set) */
void treecol_shells_photon_cap(void)
{
    const double L_emit = (Nodes[All.TreeNodeIndexBase].stellar_lum[RT_FREQ_BIN_PHOTOELECTRIC] + Nodes[All.TreeNodeIndexBase].stellar_lum[RT_FREQ_BIN_GENERIC_USER_FREQ]) * UNIT_LUM_IN_CGS;
    if(!(L_emit > 0)) {return;} /* no light: nothing absorbed; the root is the same on every rank */
    const double to_NH = UNIT_MASS_IN_CGS * HYDROGEN_MASSFRAC / PROTONMASS_CGS;
    double abs_loc = 0, abs_glob = 0;
    for(int i : ActiveParticleList)
    {
        if(P[i].Type != 0 || P[i].Mass <= 0) {continue;}
        double m[TREECOL_PIXELS_NSPECIES], u_uv = 0; treecol_pixels_species_masses(P, CellP, i, m);
        for(int k = 0; k < TREECOL_PIXELS_N; k++) {u_uv += TREECOL_PIXELS_EGY_UV(CellP[i], k);}
        abs_loc += u_uv * TREECOL_SHELLS_SIGMA_UV * m[TREECOL_PIXELS_DUST] * to_NH * C_LIGHT_CGS / All.cf_a2inv; /* F = u c */
    }
    MPI_Allreduce(&abs_loc, &abs_glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(abs_glob <= L_emit || abs_glob <= 0) {return;}
    const double f = L_emit / abs_glob;
    for(int i : ActiveParticleList)
    {
        if(P[i].Type != 0) {continue;}
        for(int k = 0; k < TREECOL_PIXELS_N; k++) {CellP[i].TreeColEgyUV[k] = treecol_pixel_store(TREECOL_PIXELS_EGY_UV(CellP[i], k) * f, 1.0, TREECOL_PIXELS_LOG16_MIN_EGY);}
        CellP[i].TreeColEgyLW *= f; CellP[i].TreeColEgyNUV *= f; CellP[i].TreeColEgyOPT *= f;
    }
    if(ThisTask == 0) {printf("TreeCol shells: photon cap engaged, f=%g (L_emit=%g, L_abs_implied=%g erg/s)\n", f, L_emit, abs_glob);}
}
#endif
