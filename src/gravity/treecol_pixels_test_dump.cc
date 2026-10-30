/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* TEST_DUMP_TREECOL_PIXELS_AND_EXIT (test only): after the first gravity computation, write the gravity-tree radiation inputs
   and results as text to <OutputDir>/treecol_dump_<what>_<rank>.txt and stop. Read by test/treecol_pixels. */
#include <mpi.h>
#include <stdio.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"

#ifdef TEST_DUMP_TREECOL_PIXELS_AND_EXIT

static FILE *treecol_dump_open(const char *what)
{
    char fn[1024]; snprintf(fn, sizeof(fn), "%s/treecol_dump_%s_%d.txt", All.OutputDir, what, ThisTask);
    FILE *fd = fopen(fn, "w"); if(!fd) {printf("treecol dump: cannot open %s\n", fn); endrun(9817301);}
    return fd;
}

void test_dump_treecol_pixels_and_exit(void)
{
#ifdef RT_USE_GRAVTREE
    FILE *fd = treecol_dump_open("stars"); /* ID, time, band luminosities [erg/s] as the tree sources see them */
    for(int i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 4) {continue;}
        double lum[N_RT_FREQ_BINS]; rt_get_source_luminosity(i, 1, lum, P, CellP);
        fprintf(fd, "%llu %.17g", (unsigned long long) P[i].ID, All.Time);
        for(int k = 0; k < N_RT_FREQ_BINS; k++) {fprintf(fd, " %.17g", lum[k] * UNIT_LUM_IN_CGS);}
        fprintf(fd, "\n");
    }
    fclose(fd);
    if(ThisTask == 0)
    {
        fd = treecol_dump_open("root"); /* the root node's band luminosities [erg/s] */
        for(int k = 0; k < N_RT_FREQ_BINS; k++) {fprintf(fd, "%.17g ", Nodes[All.TreeNodeIndexBase].stellar_lum[k] * UNIT_LUM_IN_CGS);}
        fprintf(fd, "\n"); fclose(fd);
        fd = treecol_dump_open("bins"); /* bin indices: PE LW 6-8 NUV OPT */
        fprintf(fd, "%d %d %d %d %d\n", RT_FREQ_BIN_PHOTOELECTRIC, RT_FREQ_BIN_LYMAN_WERNER, RT_FREQ_BIN_GENERIC_USER_FREQ, RT_FREQ_BIN_NUV, RT_FREQ_BIN_OPTICAL_NIR);
        fclose(fd);
    }
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    if(ThisTask == 0) {printf("TREECOL DUMP written\n"); fflush(stdout);}
    endrun(0);
}

#endif
