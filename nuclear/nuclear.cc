/* Nuclear reaction network driver, initialization, and EOS coupling for GIZMO.
   Follows the cooling gather-dispatch-scatter pattern for GPU compatibility.
   All per-particle work operates on compact arrays — never touches P[]/CellP[] globals.

   Contents:
     - InitNuclearNetwork(): one-time setup (called from begrun.cc)
     - nuclear_parent_routine(): main driver (called from run.cc)
     - CallNuclearNetwork(): dispatcher to selected solver
     - nuclear_compute_ye_abar(): composition -> Ye, Abar
     - nuclear_update_ye_abar(): write Ye/Abar to cell data
     - nuclear_fixup_mass_fractions(): post-hydro clamp+renormalize
*/

#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "nuclear.h"

#ifdef NUCLEAR_NETWORK


/* Update Ye and Abar in compact cell data from network output. */
void nuclear_update_ye_abar(int j, const struct nuclear_output *out,
                            struct particle_data *pp, struct gas_cell_data *cell)
{
#ifdef EOS_CARRIES_YE
    cell[j].Ye = out->Ye;
#endif
#ifdef EOS_CARRIES_ABAR
    cell[j].Abar = out->Abar;
#endif
}


/* =========================================================================
   Initialization (called once from core/begrun.cc)
   ========================================================================= */
void InitNuclearNetwork(void)
{
    if (ThisTask == 0) {
        printf("Initializing nuclear reaction network (solver=%d, nspecies=%d)\n",
               NUCLEAR_NETWORK_SOLVER, NUM_NUCLEAR_SPECIES);
    }
    /* set defaults for parameters not specified in the paramfile */
    if (All.NuclearBurningFloor_T <= 0) All.NuclearBurningFloor_T = 1.0e8;
    if (All.NuclearNSE_T_threshold <= 0) All.NuclearNSE_T_threshold = 6.0e9;

#if !defined(NUCLEAR_NETWORK_SOLVER) || (NUCLEAR_NETWORK_SOLVER == 0)
    if (ThisTask == 0) {
        printf("  Built-in aprox13 alpha-chain (%d species)\n", NUM_NUCLEAR_SPECIES);
        printf("  T_burn_floor = %.2e K, T_NSE = %.2e K\n",
               All.NuclearBurningFloor_T, All.NuclearNSE_T_threshold);
    }
#elif (NUCLEAR_NETWORK_SOLVER == 1)
    nuclear_skynet_init();
#elif (NUCLEAR_NETWORK_SOLVER == 2)
    nuclear_torch_init();
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    if (ThisTask == 0) printf("Nuclear network initialization complete.\n");
}


/* =========================================================================
   CallNuclearNetwork — dispatch to the selected solver.
   Pure function: no global state, thread-safe.
   ========================================================================= */
int CallNuclearNetwork(const struct nuclear_input *in, struct nuclear_output *out)
{
    memset(out, 0, sizeof(*out));
#if !defined(NUCLEAR_NETWORK_SOLVER) || (NUCLEAR_NETWORK_SOLVER == 0)
    return nuclear_aprox13_solve(in, out);
#elif (NUCLEAR_NETWORK_SOLVER == 1)
    return nuclear_skynet_solve(in, out);
#elif (NUCLEAR_NETWORK_SOLVER == 2)
    return nuclear_torch_solve(in, out);
#else
    return -1;
#endif
}


/* =========================================================================
   Pack / unpack helpers for the gather-dispatch-scatter pattern.
   These operate only on compact_P[j] / compact_Cell[j].
   ========================================================================= */
static inline struct nuclear_input pack_nuclear_input(int j,
        struct particle_data *pp, struct gas_cell_data *cell)
{
    struct nuclear_input in;
    in.rho = cell[j].Density * All.cf_a3inv;
    in.T   = cell[j].Temperature;
    in.dt  = get_particle_timestep_in_physical(j, pp);
#ifdef EOS_CARRIES_YE
    in.Ye  = cell[j].Ye;
#else
    in.Ye  = 0.5;
#endif
    for (int k = 0; k < NUM_NUCLEAR_SPECIES; k++) {
        in.X[k] = pp[j].Metallicity[NUCLEAR_SPECIES_OFFSET_IN_METALLICITY + k];
    }
    return in;
}

static inline void unpack_nuclear_output(int j, const struct nuclear_output *out,
        struct particle_data *pp, struct gas_cell_data *cell)
{
    for (int k = 0; k < NUM_NUCLEAR_SPECIES; k++) {
        pp[j].Metallicity[NUCLEAR_SPECIES_OFFSET_IN_METALLICITY + k] = out->X[k];
    }
    nuclear_update_ye_abar(j, out, pp, cell);
    if (out->de != 0.0) {
        /* cap energy injection to prevent EOS blowup: limit to 2x current internal energy per step.
           if nuclear burning would release more, the timestep limiter (via burning_timescale) will
           ensure smaller steps next time so the energy is deposited gradually. */
        double u_current = DMAX(cell[j].InternalEnergy, All.MinEgySpec);
        double de_max = 0.25 * u_current; /* limit to 25% energy change per hydro step for stability */
        double de_apply = out->de;
        if (de_apply > de_max) de_apply = de_max;
        if (de_apply < -0.9 * u_current) de_apply = -0.9 * u_current; /* don't cool below 10% of current */
        cell[j].InternalEnergy     += de_apply;
        cell[j].InternalEnergyPred += de_apply;
        cell[j].InternalEnergy      = DMAX(cell[j].InternalEnergy, All.MinEgySpec);
        cell[j].InternalEnergyPred  = DMAX(cell[j].InternalEnergyPred, All.MinEgySpec);
    }
    cell[j].NuclearEnergyGenerationRate = out->edot;
    cell[j].NuclearBurningTimescale     = out->burning_timescale;
#ifdef NUCLEAR_NETWORK_NEUTRINOS
    for (int f = 0; f < 3; f++) {
        cell[j].NeutrinoLuminosity[f] = out->nu_lum[f];
        cell[j].NeutrinoMeanEnergy[f] = out->nu_emean[f];
    }
#endif
}


/* =========================================================================
   nuclear_fixup_mass_fractions — clamp negatives and renormalize.
   Can operate on compact arrays (pp[j]) or global P[j].
   ========================================================================= */
void nuclear_fixup_mass_fractions(int j, struct particle_data *pp, struct gas_cell_data *cell)
{
    double sum = 0;
    for (int k = 0; k < NUM_NUCLEAR_SPECIES; k++) {
        double Xk = pp[j].Metallicity[NUCLEAR_SPECIES_OFFSET_IN_METALLICITY + k];
        if (Xk < 0) Xk = 0;
        if (Xk > 1) Xk = 1;
        pp[j].Metallicity[NUCLEAR_SPECIES_OFFSET_IN_METALLICITY + k] = Xk;
        sum += Xk;
    }
    if (sum > 0 && sum != 1.0) {
        double inv_sum = 1.0 / sum;
        for (int k = 0; k < NUM_NUCLEAR_SPECIES; k++) {
            pp[j].Metallicity[NUCLEAR_SPECIES_OFFSET_IN_METALLICITY + k] *= inv_sum;
        }
    }
}


/* =========================================================================
   nuclear_parent_routine — main driver, called from core/run.cc.
   Gather-dispatch-scatter pattern identical to cooling_parent_routine.
   ========================================================================= */
void nuclear_parent_routine(void)
{
    PRINT_STATUS("Nuclear burning update");

    /* Step 1: determine active gas particles eligible for burning */
    std::vector<int> burn_indices;
    burn_indices.reserve(ActiveParticleList.size());
    for (int i : ActiveParticleList) {
        if (P[i].Type != 0 || P[i].Mass <= 0) continue;
        double T_threshold = (All.NuclearBurningFloor_T > 0) ? All.NuclearBurningFloor_T : 1.0e8;
        if (CellP[i].Temperature < T_threshold) continue;
        burn_indices.push_back(i);
    }
    int N_active = (int) burn_indices.size();
    if (N_active == 0) return;

    /* Step 2: Gather — allocate compact arrays, copy active particle data */
    struct particle_data *compact_P = (struct particle_data *) mymalloc("nuclear_P",
            N_active * sizeof(struct particle_data));
    struct gas_cell_data *compact_Cell = (struct gas_cell_data *) mymalloc("nuclear_Cell",
            N_active * sizeof(struct gas_cell_data));
    for (int j = 0; j < N_active; j++) {
        compact_P[j]    = P[burn_indices[j]];
        compact_Cell[j] = CellP[burn_indices[j]];
    }

    /* Step 2b: fixup mass fractions on compact arrays before burning */
    for (int j = 0; j < N_active; j++) {
        nuclear_fixup_mass_fractions(j, compact_P, compact_Cell);
    }

    /* Step 3: Dispatch — run nuclear burning on compact arrays */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int j = 0; j < N_active; j++) {
        if (compact_P[j].Type != 0 || compact_Cell[j].Mass <= 0) continue;
        struct nuclear_input  in  = pack_nuclear_input(j, compact_P, compact_Cell);
        if (in.dt <= 0) continue;
        struct nuclear_output out;
        int ierr = CallNuclearNetwork(&in, &out);
        if (ierr == 0) {
            unpack_nuclear_output(j, &out, compact_P, compact_Cell);
        }
    }

    /* Step 4: Scatter — copy modified data back to global arrays */
    for (int j = 0; j < N_active; j++) {
        int i = burn_indices[j];
        CellP[i] = compact_Cell[j];
        P[i]     = compact_P[j];
    }

    myfree(compact_Cell);
    myfree(compact_P);
}


#endif /* NUCLEAR_NETWORK */
