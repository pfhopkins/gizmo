/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../../declarations/allvars.h"
#include "../../core/proto.h"
#include "rumi_lib.h"

/*! \file rumi_coupling.cc
 *  \brief Type-4 stars and the collisional N-body library RUMI++ (STARS_INTEGRATED_BY_RUMI): RUMI++ start-up on GIZMO's
 *  Kokkos and tasks 0..RumiNumRanks-1, units (pc, Msun, km/s) and G from All.G. Star-star pairs leave the tree; until RUMI++
 *  integrates the stars they feel each other by the exact direct sum (star_direct_gravity.cc).
 */

#ifdef STARS_INTEGRATED_BY_RUMI

#define RUMI_COUPLING_API_VERSION 3 /* rumi_lib.h interface these calls are written for (Makefile.rumi checks librumi.a) */
static_assert(RUMI_LIB_API_VERSION == RUMI_COUPLING_API_VERSION, "rumi_lib.h is not the RUMI++ interface version rumi_coupling.cc expects");
#ifndef RUMI_LIB_ID_MK
#define RUMI_LIB_ID_MK "unknown"
#endif

namespace {

double UL = 1, UM = 1, UV = 1;           /* GIZMO code unit in pc, Msun, km/s */
MPI_Comm RumiComm = MPI_COMM_NULL;       /* tasks 0..RumiNumRanks-1 */
int rumi_rank = 0;                        /* this task runs RUMI++ */

double rumi_G(void) {return All.G * UL * UV * UV / UM;} /* G in pc^3 Msun^-1 (km/s)^2 */

void rumi_atexit(void) {rumi_lib::kokkos_finalize();}

} // namespace


/* begrun: units; on the RUMI++ ranks GIZMO's Kokkos (RUMI++ reuses it), the sub-communicator and the RUMI++ configuration */
void rumi_init(void)
{
    if(All.ComovingIntegrationOn) {if(ThisTask == 0) {printf("STARS_INTEGRATED_BY_RUMI: comoving integration is not supported\n");} endrun(9301);}
    UL = UNIT_LENGTH_IN_PC; UM = UNIT_MASS_IN_SOLAR; UV = UNIT_VEL_IN_KMS;
    if(All.RumiLengthUnitPc > 0) {UL = All.RumiLengthUnitPc;}
    if(All.RumiMassUnitMsun > 0) {UM = All.RumiMassUnitMsun;}
    if(All.RumiVelUnitKms > 0) {UV = All.RumiVelUnitKms;}
    int nr = All.RumiNumRanks; if(nr < 1) {nr = 1;} if(nr > NTask) {nr = NTask;}
    rumi_rank = (ThisTask < nr);
    MPI_Comm_split(MPI_COMM_WORLD, rumi_rank ? 0 : MPI_UNDEFINED, ThisTask, &RumiComm);
    int ok = 1;
    if(rumi_rank) {
        if(rumi_lib::kokkos_initialize(0, -1) != 0) {ok = 0;}
        else {
            atexit(rumi_atexit); /* endrun(0) calls MPI_Finalize and exit directly */
#ifdef RUMI_LIB_HAS_MPI
            if(NTask > 1 && rumi_lib::attach_comm(MPI_Comm_c2f(RumiComm)) != 0) {ok = 0;} /* unattached, its collectives would run on MPI_COMM_WORLD */
#else
            if(nr > 1 && rumi_lib::attach_comm(MPI_Comm_c2f(RumiComm)) != 0) {ok = 0;}
#endif
            if(ok && rumi_lib::configure(All.RumiConfigFile, ThisTask == 0) != 0) {ok = 0;}
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(!ok) {endrun(9302);}
    if(ThisTask == 0) {
        printf("RUMI++: %s (built against %s), Kokkos %s on %d rank(s); G = %.10e pc^3/Msun (km/s)^2 (All.G = %.10e); units pc %.10g Msun %.10g km/s %.10g\n",
               rumi_lib::build_id(), RUMI_LIB_ID_MK, rumi_lib::backend_name(), nr, rumi_G(), All.G, UL, UM, UV);
        if(strcmp(rumi_lib::build_id(), RUMI_LIB_ID_MK) != 0) {printf("RUMI++: the linked library is not the one Makefile.rumi checked\n"); endrun(9307);}
    }
}

void rumi_finalize(void)
{
    if(!rumi_rank) return;
    rumi_lib::kokkos_finalize();
}

#endif /* STARS_INTEGRATED_BY_RUMI */
