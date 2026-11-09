/* SkyNet external nuclear network library wrapper.
   SkyNet (Lippuner & Roberts 2017, ApJS 233, 18) is a C++ library for
   nuclear reaction networks with flexible species sets, REACLIB rates,
   and built-in implicit ODE solvers.

   This wrapper is only compiled when NUCLEAR_NETWORK_SOLVER == 1.
   Required link flags:
     -lskynet -lhdf5_cpp -lhdf5 -lgsl -lgslcblas -lboost_filesystem -lboost_serialization -lgfortran

   The SkyNet data directory (containing reaclib, winvn, webnucleo_nuc XML)
   must be specified via NuclearNetworkDataFile in the GIZMO parameter file.
*/

#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "nuclear.h"

#ifdef NUCLEAR_NETWORK
#if defined(NUCLEAR_NETWORK_SOLVER) && (NUCLEAR_NETWORK_SOLVER == 1)

#include <Network/ReactionNetwork.hpp>
#include <Network/NetworkOptions.hpp>
#include <Network/NSEOptions.hpp>
#include <NuclearData/NuclideLibrary.hpp>
#include <NuclearData/Nuclide.hpp>
#include <Reactions/REACLIBReactionLibrary.hpp>
#include <Reactions/REACLIB.hpp>
#include <DensityProfiles/ConstantFunction.hpp>

/* species index mapping: GIZMO aprox13 index -> SkyNet nuclide vector index */
static int skynet_species_map[NUM_NUCLEAR_SPECIES];
static int skynet_n_species = 0;

/* SkyNet objects — read-only after init */
static ReactionNetwork *skynet_network = nullptr;


void nuclear_skynet_init(void)
{
    if (ThisTask == 0) {
        printf("SkyNet nuclear network: initializing\n");
        printf("  Data directory: %s\n", All.NuclearNetworkDataFile);
    }

    std::string datadir(All.NuclearNetworkDataFile);
    std::string webnucleofile = datadir + "/webnucleo_nuc_v2.0.xml";
    std::string reaclibfile   = datadir + "/reaclib";

    /* build restricted species list for aprox13 */
    std::vector<std::string> species_names;
    for (int k = 0; k < NUM_NUCLEAR_SPECIES; k++) {
        species_names.push_back(Nuclide::GetName(nuclear_aprox13_Z(k), nuclear_aprox13_A(k)));
    }

    /* create nuclide library */
    NuclideLibrary fullLib = NuclideLibrary::CreateFromWebnucleoXML(webnucleofile);
    NuclideLibrary nuclib = NuclideLibrary::CreateRestrictedLibrary(fullLib, species_names);
    skynet_n_species = nuclib.NumNuclides();

    if (ThisTask == 0) {
        printf("  %d species in network\n", skynet_n_species);
    }

    /* build species mapping: GIZMO index -> SkyNet vector position */
    const std::vector<Nuclide> &nuclides = nuclib.Nuclides();
    for (int k = 0; k < NUM_NUCLEAR_SPECIES; k++) {
        skynet_species_map[k] = -1;
        for (int s = 0; s < skynet_n_species; s++) {
            if (nuclides[s].A() == nuclear_aprox13_A(k) && nuclides[s].Z() == nuclear_aprox13_Z(k)) {
                skynet_species_map[k] = s;
                break;
            }
        }
        if (skynet_species_map[k] < 0 && ThisTask == 0) {
            printf("  WARNING: species Z=%d A=%d not found in SkyNet library\n",
                   nuclear_aprox13_Z(k), nuclear_aprox13_A(k));
        }
    }

    /* configure network options */
    NetworkOptions netOpts;
    netOpts.IsSelfHeating = false; /* T provided externally by GIZMO */

    /* load REACLIB reaction rates and build the reaction library */
    REACLIB reaclib(reaclibfile);
    REACLIBReactionLibrary reaclibLib(reaclib, "JINA REACLIB",
        ReactionTypeStruct::Strong, true,
        LeptonModeStruct::TreatAllAsDecay, "CF88+JINA",
        nuclib, netOpts, false);

    /* assemble the reaction library vector */
    ReactionLibs reacLibs;
    reacLibs.push_back(&reaclibLib);

    /* create the network */
    skynet_network = new ReactionNetwork(nuclib, reacLibs, nullptr, nullptr, netOpts);

    MPI_Barrier(MPI_COMM_WORLD);
    if (ThisTask == 0) printf("SkyNet initialization complete.\n");
}


int nuclear_skynet_solve(const struct nuclear_input *in, struct nuclear_output *out)
{
    if (!skynet_network) return -1;

    double rho_cgs = in->rho * UNIT_DENSITY_IN_CGS;
    double T9 = in->T / 1.0e9;
    double dt_s = in->dt * UNIT_TIME_IN_CGS;

    /* map GIZMO mass fractions to SkyNet molar abundances Y = X/A */
    std::vector<double> Y_in(skynet_n_species, 0.0);
    for (int k = 0; k < NUM_NUCLEAR_SPECIES; k++) {
        if (skynet_species_map[k] >= 0) {
            Y_in[skynet_species_map[k]] = in->X[k] / (double)nuclear_aprox13_A(k);
        }
    }

    /* constant T and rho over this timestep */
    ConstantFunction T9_const(T9);
    ConstantFunction rho_const(rho_cgs);

    /* evolve */
    const NetworkOutput &result = skynet_network->Evolve(
        Y_in, 0.0, dt_s, &T9_const, &rho_const, "", dt_s);

    /* map back to GIZMO mass fractions */
    const std::vector<double> &Y_final = result.FinalY();
    double sum = 0;
    for (int k = 0; k < NUM_NUCLEAR_SPECIES; k++) {
        if (skynet_species_map[k] >= 0 && skynet_species_map[k] < (int)Y_final.size()) {
            out->X[k] = Y_final[skynet_species_map[k]] * (double)nuclear_aprox13_A(k);
        } else {
            out->X[k] = in->X[k];
        }
        if (out->X[k] < 0) out->X[k] = 0;
        sum += out->X[k];
    }
    if (sum > 0) { for (int k = 0; k < NUM_NUCLEAR_SPECIES; k++) out->X[k] /= sum; }

    nuclear_compute_ye_abar(out->X, &out->Ye, &out->Abar);

    /* energy: binding energy difference */
    double de_cgs = 0;
    for (int k = 0; k < NUM_NUCLEAR_SPECIES; k++) {
        de_cgs += (out->X[k] - in->X[k]) * nuclear_aprox13_BE_per_A(k) * 1.602176634e-6 * 6.02214076e23;
    }
    out->de = de_cgs / UNIT_SPECEGY_IN_CGS;
    out->edot = (dt_s > 0) ? de_cgs / dt_s / (UNIT_SPECEGY_IN_CGS / UNIT_TIME_IN_CGS) : 0;
    out->burning_timescale = dt_s / UNIT_TIME_IN_CGS;

    return 0;
}


#endif /* NUCLEAR_NETWORK_SOLVER == 1 */
#endif /* NUCLEAR_NETWORK */
