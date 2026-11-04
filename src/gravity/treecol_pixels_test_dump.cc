/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* TEST_DUMP_TREECOL_PIXELS_AND_EXIT (test only): after the gravity computation of step TEST_DUMP_TREECOL_PIXELS_AT_STEP (default 0),
   write the gravity-tree radiation inputs and results as text to <OutputDir>/treecol_dump_<what>_<rank>.txt and stop.
   Read by test/treecol_pixels. */
#include <mpi.h>
#include <stdio.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "gravtree_treecol_pixels.h"
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
#include "../cooling/sgchem/sgchem_network.h"
#endif

#ifdef TEST_DUMP_TREECOL_PIXELS_AND_EXIT
#ifndef TEST_DUMP_TREECOL_PIXELS_AT_STEP
#define TEST_DUMP_TREECOL_PIXELS_AT_STEP 0
#endif

static FILE *treecol_dump_open(const char *what)
{
    char fn[1024]; snprintf(fn, sizeof(fn), "%s/treecol_dump_%s_%d.txt", All.OutputDir, what, ThisTask);
    FILE *fd = fopen(fn, "w"); if(!fd) {printf("treecol dump: cannot open %s\n", fn); endrun(9817301);}
    return fd;
}

void test_dump_treecol_pixels_and_exit(void)
{
    if(All.NumCurrentTiStep < TEST_DUMP_TREECOL_PIXELS_AT_STEP) {return;}
#ifdef RT_USE_GRAVTREE
    FILE *fd = treecol_dump_open("stars"); /* ID, time, band luminosities [erg/s] as the tree sources see them, position */
    for(int i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 4) {continue;}
        double lum[N_RT_FREQ_BINS]; rt_get_source_luminosity(i, 1, lum, P, CellP);
        fprintf(fd, "%llu %.17g", (unsigned long long) P[i].ID, All.Time);
        for(int k = 0; k < N_RT_FREQ_BINS; k++) {fprintf(fd, " %.17g", lum[k] * UNIT_LUM_IN_CGS);}
        fprintf(fd, " %.17g %.17g %.17g\n", P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
    }
    fclose(fd);
    if(ThisTask == 0)
    {
        fd = treecol_dump_open("root_lum"); /* the root node's band luminosities [erg/s] */
        for(int k = 0; k < N_RT_FREQ_BINS; k++) {fprintf(fd, "%.17g ", Nodes[All.TreeNodeIndexBase].stellar_lum[k] * UNIT_LUM_IN_CGS);}
        fprintf(fd, "\n"); fclose(fd);
        fd = treecol_dump_open("bins"); /* bin indices: PE LW 6-8 NUV OPT */
        fprintf(fd, "%d %d %d %d %d", RT_FREQ_BIN_PHOTOELECTRIC, RT_FREQ_BIN_LYMAN_WERNER, RT_FREQ_BIN_GENERIC_USER_FREQ, RT_FREQ_BIN_NUV, RT_FREQ_BIN_OPTICAL_NIR);
#ifdef RT_FREQ_BIN_ION_ENERGY
        fprintf(fd, " %d %d", RT_FREQ_BIN_ION_ENERGY, RT_FREQ_BIN_ION_PHOTONS);
#endif
        fprintf(fd, "\n");
        fclose(fd);
    }
#endif
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS
    FILE *fc = treecol_dump_open("cells"); /* ID, species masses [code units], pixel columns [code], band energy densities [erg/cm^3], G0 G0_LW G0_dust, position */
    for(int i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 0) {continue;}
        double m[TREECOL_PIXELS_NSPECIES]; treecol_pixels_species_masses(P, CellP, i, m);
        fprintf(fc, "%llu", (unsigned long long) P[i].ID);
        for(int k = 0; k < TREECOL_PIXELS_NSPECIES; k++) {fprintf(fc, " %.17g", m[k]);}
        for(int k = 0; k < TREECOL_PIXELS_NCOLUMNS; k++) {for(int j = 0; j < GALSF_RESOLVEDISM_TREECOL_PIXELS; j++) {fprintf(fc, " %.9g", TREECOL_PIXELS_COLUMN(CellP[i], k, j));}}
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
        for(int j = 0; j < GALSF_RESOLVEDISM_TREECOL_PIXELS; j++) {fprintf(fc, " %.9g", TREECOL_PIXELS_EGY_UV(CellP[i], j));}
        fprintf(fc, " %.9g %.9g %.9g %.9g", CellP[i].TreeColEgyLW, CellP[i].TreeColEgyNUV, CellP[i].TreeColEgyOPT, CellP[i].TreeColEgy68);
        {double fac_uv[NPIX]; struct ChemFUVField f = sgchem_treecol_field(i, fac_uv); fprintf(fc, " %.17g %.17g %.17g", f.G0, f.G0_LW, f.G0_dust);} /* chemistry inputs */
#endif
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS_PHOTOION
        fprintf(fc, " %.9g %.9g", CellP[i].TreeRayIonPhotonFlux, CellP[i].TreeRayIonEnergyFlux);
#endif
        fprintf(fc, " %.17g %.17g %.17g\n", P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
    }
    fclose(fc);
    if(ThisTask == 0)
    {
        fc = treecol_dump_open("root_species"); /* the root node's species masses [code units] */
        for(int k = 0; k < TREECOL_PIXELS_NSPECIES; k++) {fprintf(fc, "%.17g ", Nodes[All.TreeNodeIndexBase].treecol_species_mass[k]);}
        fprintf(fc, "\n"); fclose(fc);
    }
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    if(ThisTask == 0) {printf("TREECOL DUMP written\n"); fflush(stdout);}
    endrun(0);
}

#endif
