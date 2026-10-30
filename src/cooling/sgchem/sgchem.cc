/* C++ implementation: Ulrich Steinwandel (uli@mpa-garching.mpg.de).
   Based on the SGChem Fortran networks by Simon Glover (GNU GPL, as distributed in Arepo). */
/* SGChem chemistry in GIZMO: start-up (table check, network tables). C++ only, host only (thread_local solver state). */
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <mpi.h>
#include "../../declarations/allvars.h"
#include "../../core/proto.h"
#include "chem_interface.h"
#include "../../galaxy_sf/stella_tables_check.h"

#ifdef SGCHEM

static_assert(SGCHEM_NUM_TRACKED_SPECIES == TRAC_NUM, "SGCHEM_NUM_TRACKED_SPECIES (precompiler_logic.h) must equal TRAC_NUM (sgchem_consts.h)");

#ifdef TEST_SGCHEM_ONEZONE_WITH_12_PIXEL_COLUMNS
#error "TEST_SGCHEM_ONEZONE_WITH_12_PIXEL_COLUMNS is for the standalone one-zone drivers (test/sgchem_onezone) only"
#endif

/* target of the backend's fatal errors (one rank, possibly inside the threaded cooling loop): abort the whole job, visibly */
extern "C" void sgchem_fatal_(int *code)
{
    printf("SGCHEM FATAL: code %d on task %d\n", *code, ThisTask);
    fflush(stdout);
    MPI_Abort(MPI_COMM_WORLD, 78100 + *code);
}

/* rank 0 checks the installed tables against the hashes these sources expect; a mismatch stops the run */
static void sgchem_tables_check(void)
{
    int bad = 0;
    if(ThisTask == 0) {
        std::vector<std::string> paths;
        for(const char *f : {"energy-table-x.txt", "eh2-table.txt", "H2-cooling-ratios.dat", "wss_z_collis.txt"}) {paths.push_back(std::string(All.SGChemTablesPath) + "/" + f);}
#if CHEMISTRYNETWORK == 18
        paths.push_back(std::string(All.SGChemTablesPath) + "/sgchem_net18_tables.bin");
#endif
        bad = stella_tables_verify(paths);
    }
    MPI_Bcast(&bad, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(bad) {
        if(ThisTask == 0) {printf("SGCHEM: chemistry tables in '%s' missing or not the version these sources expect; install with scripts/get_stella_tables.sh and set SGChemTablesPath\n", All.SGChemTablesPath); fflush(stdout);}
        endrun(78001);
        gizmo_exit_bad_stop_if_requested("sgchem:tables");
    }
}

void sgchem_init(void)
{
    if(ThisTask == 0) {printf("SGCHEM: network %d, tables from %s\n", CHEMISTRYNETWORK, All.SGChemTablesPath); fflush(stdout);}
    sgchem_tables_check();
    struct ChemInitParams p;
    p.iflag_mn    = 4;   /* H- + H+ mutual neutralisation (CDG99) */
    p.iflag_ad    = 6;   /* H- + H associative detachment */
    p.iflag_atom  = 2;   /* SD93 atomic cooling */
    p.iflag_3bh2a = 5;   /* H2 + H: CID (MSM96), three-body formation (G07) */
    p.iflag_3bh2b = 4;   /* H2 + H2: 1/8 of the H2 + H rates */
    p.iflag_h3pra = 1;   /* H3+ dissociative recombination (GP98) */
    p.iphoto      = 0;   /* no tree columns in this tree yet: optically thin */
    p.irad_heat   = 0;
    p.isrf_option = 1;   /* Draine (1978) ISRF shape */
    p.no_chem     = 0;
    p.tables_dir  = All.SGChemTablesPath;
    p.h2_form_ex  = 0.84;
    p.h2_form_kin = 0.12;
    p.deff        = 1.0;
    p.phi_pah     = 0.5;
    p.AV_conversion_factor = 5.348e-22; /* N_H to A_V (Bohlin+ 1978) */
    p.AV_ext      = 0.0;
    p.dm_density  = 0.0;
    p.pdv_term    = 0.0;
    p.redshift    = 0.0;
    p.abundD      = All.SGChemDeuteriumAbundance;
    p.G0          = 0.0; /* no external field in this tree yet */
    p.G0_LW       = 0.0;
    p.cosmic_ray_ion_rate = All.SGChemCosmicRayIonRate;
    p.cr_energy_density = 0.0;
    p.dust_to_gas_ratio = All.SGChemDustToGasRatio;
    chem_cxx_tables_init(&p);
    if(ThisTask == 0) {printf("SGCHEM: chemistry tables built\n"); fflush(stdout);}
    if(ThisTask == 0) {printf("SGCHEM: hydrogen mass fraction %g, MinEgySpec %.8g\n", (double) HYDROGEN_MASSFRAC, All.MinEgySpec); fflush(stdout);}
}

/* fresh-start cell state, as stella_dev init.cc: no H2 or CO, x(H+) = 1e-4, dust at 10 K, gas at 100 K until the first solve */
void sgchem_init_cell_state(int i)
{
    for(int k = 0; k < TRAC_NUM; k++) {CellP[i].TracAbund[k] = 0;}
    CellP[i].TracAbund[IHP] = 1.0e-4;
    CellP[i].DustTemp = 10.0;
    CellP[i].Temp = 100.0;
}

#endif /* SGCHEM */
