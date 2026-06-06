/* XNet external nuclear network library wrapper (solver slot 2).
   XNet (Hix & Meyer; starkiller-astro) is a Fortran 2003 nuclear reaction
   network with GPU support (CUDA/HIP) and multi-zone batching, designed
   for coupling to hydrodynamics codes (used in Castro/AMReX).

   This wrapper is compiled when NUCLEAR_NETWORK_SOLVER == 2.
   XNet must be built as a library (libxnet.a) from the starkiller-astro
   source with Fortran-C interoperability bindings enabled.

   Status: stub implementation. XNet's Fortran full_net() interface needs
   to be wrapped via iso_c_binding to create a callable C API. The GPU
   batching mode (processing multiple zones simultaneously) is the primary
   reason to prefer XNet over SkyNet for large-scale production runs.

   See: https://github.com/starkiller-astro/XNet
*/

#include <cstdio>
#include <cstring>
#include <cmath>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "nuclear.h"

#ifdef NUCLEAR_NETWORK
#if defined(NUCLEAR_NETWORK_SOLVER) && (NUCLEAR_NETWORK_SOLVER == 2)

/* XNet Fortran-C interface declarations.
   These would be provided by a libxnet header or defined here via extern "C".
   The XNet source already uses iso_c_binding for GPU interop routines;
   the full_net() subroutine needs a similar C binding added.

   extern "C" {
       void xnet_init_c(const char *datadir, int *nspecies);
       void xnet_solve_c(int nzones, double *rho, double *T9, double *dt,
                         double *Y_in, double *Y_out, double *energy_released);
   }
*/

static int xnet_initialized = 0;
static int xnet_species_map[NUM_NUCLEAR_SPECIES];


void nuclear_torch_init(void)
{
    if (ThisTask == 0) {
        printf("XNet nuclear network wrapper: initializing\n");
        printf("  Data directory: %s\n", All.NuclearNetworkDataFile);
        printf("  NOTE: XNet requires a libxnet.a built with C bindings from\n");
        printf("        https://github.com/starkiller-astro/XNet\n");
        printf("  Using stub implementation until XNet library is linked.\n");
    }

    /* When XNet is available as a library:
       xnet_init_c(All.NuclearNetworkDataFile, &xnet_nspecies);
       // build species mapping from XNet's internal ordering to GIZMO's aprox13
    */

    for (int k = 0; k < NUM_NUCLEAR_SPECIES; k++) {
        xnet_species_map[k] = k;
    }

    xnet_initialized = 1;
    MPI_Barrier(MPI_COMM_WORLD);
    if (ThisTask == 0) printf("XNet initialization complete (stub mode).\n");
}


int nuclear_torch_solve(const struct nuclear_input *in, struct nuclear_output *out)
{
    if (!xnet_initialized) return -1;

    /* When XNet library is linked:
       double rho_cgs = in->rho * UNIT_DENSITY_IN_CGS;
       double T9 = in->T / 1.0e9;
       double dt_s = in->dt * UNIT_TIME_IN_CGS;

       // pack Y = X/A
       double Y_in[NUM_NUCLEAR_SPECIES], Y_out[NUM_NUCLEAR_SPECIES];
       for (int k = 0; k < NUM_NUCLEAR_SPECIES; k++)
           Y_in[xnet_species_map[k]] = in->X[k] / nuclear_aprox13_A(k);

       double energy_released;
       xnet_solve_c(1, &rho_cgs, &T9, &dt_s, Y_in, Y_out, &energy_released);

       for (int k = 0; k < NUM_NUCLEAR_SPECIES; k++)
           out->X[k] = Y_out[xnet_species_map[k]] * nuclear_aprox13_A(k);
       out->de = energy_released / UNIT_SPECEGY_IN_CGS;
    */

    /* stub: no-op until XNet library is linked */
    memcpy(out->X, in->X, sizeof(out->X));
    out->Ye = in->Ye;
    nuclear_compute_ye_abar(out->X, &out->Ye, &out->Abar);
    out->de = 0; out->edot = 0; out->burning_timescale = 1.0e30;
    {static int warned = 0; if (!warned && ThisTask == 0) {
        printf("WARNING: XNet solver stub called — build and link libxnet for actual burning\n"); warned = 1;}}
    return 0;
}


#endif /* NUCLEAR_NETWORK_SOLVER == 2 */
#endif /* NUCLEAR_NETWORK */
