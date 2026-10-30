#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "stellar_evolution_tables.h"
#include <mpi.h>

/* Single-star stellar-evolution tables (stella-tables, github.com/Ulli1991/stella-tables): lifetimes, tracks, band
   luminosities, winds, yields, remnants. Installed outside the repository by scripts/get_stella_tables.sh; the
   parameter StellarTablesFile names the main table, the wind table is read from the same directory. Both must have the
   content hashes compiled in (stellar_tables_check.cc). */

#ifdef GALSF_SINGLE_STARS_EVOLUTION_TABLES

/* the wind table sits next to the main table */
static std::string stellar_wind_tables_path(void)
{
    std::string p = All.StellarTablesFile;
    size_t s = p.find_last_of('/');
    return ((s == std::string::npos) ? std::string("") : p.substr(0, s + 1)) + "stellar_wind_cumulative.hdf5";
}

/* rank 0 checks both files; a mismatch is a controlled stop at start-up */
void stellar_evolution_tables_init(void)
{
    if(ThisTask != 0) {return;}
    double t0 = MPI_Wtime();
    std::vector<std::string> paths = {All.StellarTablesFile, stellar_wind_tables_path()};
    int bad = stellar_tables_verify(paths);
    printf("STELLAR TABLES: hash check took %.2f s\n", MPI_Wtime() - t0); fflush(stdout);
    if(bad) {
        gizmo_request_controlled_stop(7801, "stellar evolution tables missing or not the version these sources expect; install with scripts/get_stella_tables.sh", __FILE__, __LINE__, __FUNCTION__);
    }
}

#endif /* GALSF_SINGLE_STARS_EVOLUTION_TABLES */
