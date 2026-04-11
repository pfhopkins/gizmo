/* Nuclear reaction network interface for GIZMO.
   Provides infrastructure for coupling nuclear burning to the hydro solver,
   following the Grackle-like interface pattern with GPU-compatible
   gather-dispatch-scatter memory layout.

   Supports: built-in aprox13 alpha-chain (default), SkyNet, or Torch
   external libraries (conditionally compiled).
*/

#ifndef NUCLEAR_NETWORK_H
#define NUCLEAR_NETWORK_H

#ifdef NUCLEAR_NETWORK

#include "../GIZMO_config.h"
#include "../declarations/allvars.h"

#ifndef NUCLEAR_NETWORK_NSPECIES
#define NUCLEAR_NETWORK_NSPECIES 13
#endif
#define NUM_NUCLEAR_SPECIES NUCLEAR_NETWORK_NSPECIES

/* ---------------------------------------------------------------------------
   Species indices for the aprox13 alpha-chain network.
   These are the 13 alpha-particle nuclei from He4 to Ni56.
   For external networks with different species lists, a mapping is built
   at initialization time.
   --------------------------------------------------------------------------- */
enum nuclear_species_aprox13 {
    NUCLEAR_HE4  = 0,
    NUCLEAR_C12  = 1,
    NUCLEAR_O16  = 2,
    NUCLEAR_NE20 = 3,
    NUCLEAR_MG24 = 4,
    NUCLEAR_SI28 = 5,
    NUCLEAR_S32  = 6,
    NUCLEAR_AR36 = 7,
    NUCLEAR_CA40 = 8,
    NUCLEAR_TI44 = 9,
    NUCLEAR_CR48 = 10,
    NUCLEAR_FE52 = 11,
    NUCLEAR_NI56 = 12
};

/* atomic numbers and mass numbers for the 13 species */
static constexpr int nuclear_aprox13_A[] = {4, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56};
static constexpr int nuclear_aprox13_Z[] = {2,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28};

/* binding energies per nucleon [MeV] (from AME2020) — used for energy generation */
static constexpr double nuclear_aprox13_BE_per_A[] = {
    7.07392,  /* He4  */
    7.68014,  /* C12  */
    7.97621,  /* O16  */
    8.03224,  /* Ne20 */
    8.26069,  /* Mg24 */
    8.44774,  /* Si28 */
    8.49311,  /* S32  */
    8.51986,  /* Ar36 */
    8.55128,  /* Ca40 */
    8.53350,  /* Ti44 */
    8.57232,  /* Cr48 */
    8.60956,  /* Fe52 */
    8.64278   /* Ni56 */
};


/* ---------------------------------------------------------------------------
   Input/output structs for per-particle network evaluation.
   Pure value types — no pointers, GPU-buffer-safe, memcpy-safe.
   --------------------------------------------------------------------------- */

/* Self-contained input for one particle's burn step.
   Populated during gather phase from compact_P[j] / compact_Cell[j]. */
struct nuclear_input {
    double rho;         /* density [code units] */
    double T;           /* temperature [K] */
    double Ye;          /* electron fraction */
    double dt;          /* timestep [code units] */
    double X[NUM_NUCLEAR_SPECIES]; /* mass fractions in */
};

/* Self-contained output — results written back during scatter phase. */
struct nuclear_output {
    double X[NUM_NUCLEAR_SPECIES]; /* mass fractions out */
    double Ye;          /* updated electron fraction */
    double Abar;        /* updated mean atomic weight [amu] */
    double de;          /* total specific energy change over dt [code units] */
    double edot;        /* specific energy generation rate [code units / time] */
    double burning_timescale; /* shortest burning timescale [code units], for dt control */
#ifdef NUCLEAR_NETWORK_NEUTRINOS
    double nu_lum[3];   /* neutrino luminosity per flavor [code units] */
    double nu_emean[3]; /* mean neutrino energy per flavor [MeV] */
#endif
};


/* ---------------------------------------------------------------------------
   Public API — called from the main GIZMO code
   --------------------------------------------------------------------------- */

/* Initialize the nuclear network: load rate tables, init external libraries.
   Called once at startup from core/begrun.cc. */
void InitNuclearNetwork(void);

/* Main driver: gather active particles, dispatch burning, scatter results.
   Called each timestep from core/run.cc. */
void nuclear_parent_routine(void);

/* Pure function: evaluate the nuclear network for a single particle.
   No global state access — only reads from the const rate tables and
   the input struct. Thread-safe, GPU-kernel-safe. */
KOKKOS_FUNCTION int CallNuclearNetwork(const struct nuclear_input *in, struct nuclear_output *out);

/* Update Ye and Abar in compact cell data from network output.
   Called per-particle after CallNuclearNetwork in the dispatch loop. */
KOKKOS_FUNCTION void nuclear_update_ye_abar(int j, const struct nuclear_output *out,
                            struct particle_data *pp, struct gas_cell_data *cell);

/* Clamp mass fractions to [0,1] and renormalize to sum=1.
   Called after hydro advection to fix passive scalar positivity violations. */
KOKKOS_FUNCTION void nuclear_fixup_mass_fractions(int j, struct particle_data *pp, struct gas_cell_data *cell);


/* Ye/Abar computation from mass fractions — used by all solvers */
KOKKOS_FUNCTION void nuclear_compute_ye_abar(const double X[NUM_NUCLEAR_SPECIES], double *Ye_out, double *Abar_out);

#ifdef NUCLEAR_NETWORK_NEUTRINOS
/* Neutrino emission estimate from thermal processes (pair, plasmon). Pure function. */
void nuclear_neutrino_emission(double rho_cgs, double T9, double Ye,
                               double nu_lum[3], double nu_emean[3]);
/* Neutrino opacity for RT bands */
double nuclear_neutrino_opacity(int i, int k_freq, struct particle_data *pp, struct gas_cell_data *cell);
double nuclear_neutrino_absorb_frac(int i, int k_freq, struct particle_data *pp, struct gas_cell_data *cell);
/* Ye feedback from neutrino absorption after RT kick */
void nuclear_neutrino_ye_feedback(int i, double dt_code, struct particle_data *pp, struct gas_cell_data *cell);
#endif

/* ---------------------------------------------------------------------------
   Solver-specific entry points (dispatched by CallNuclearNetwork)
   --------------------------------------------------------------------------- */

#if !defined(NUCLEAR_NETWORK_SOLVER) || (NUCLEAR_NETWORK_SOLVER == 0)
/* Built-in aprox13 alpha-chain network */
KOKKOS_FUNCTION int nuclear_aprox13_solve(const struct nuclear_input *in, struct nuclear_output *out);
/* NSE equilibrium composition lookup */
KOKKOS_FUNCTION void nuclear_nse_composition(double rho_cgs, double T9, double Ye, double X_out[NUM_NUCLEAR_SPECIES]);
KOKKOS_FUNCTION int nuclear_check_nse(double T9);
#endif

#if defined(NUCLEAR_NETWORK_SOLVER) && (NUCLEAR_NETWORK_SOLVER == 1)
/* SkyNet external library wrapper */
int nuclear_skynet_solve(const struct nuclear_input *in, struct nuclear_output *out);
void nuclear_skynet_init(void);
#endif

#if defined(NUCLEAR_NETWORK_SOLVER) && (NUCLEAR_NETWORK_SOLVER == 2)
/* Torch external library wrapper */
int nuclear_torch_solve(const struct nuclear_input *in, struct nuclear_output *out);
void nuclear_torch_init(void);
#endif

#endif /* NUCLEAR_NETWORK */
#endif /* NUCLEAR_NETWORK_H */
