#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"

/* In this file we implement functions for computing ISM chemical rates and abundances under
various simplifying steady-state assumptions. Most prescriptions follow Kim et al. 2023ApJS..264...10K
and references therein.
 */

#ifdef SIMPLE_STEADYSTATE_CHEMISTRY

/* All GPU-callable function bodies are now in simple_chemistry.h (single source of truth).
   Include with non-inline linkage to provide externally-visible symbols. */
#undef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION
#include "simple_chemistry.h"

/* ---- Host-only helpers below (not GPU-callable) ---- */

#define NUM_RECOMB_TABLE_IONS 12
const char *ion_names[NUM_RECOMB_TABLE_IONS] = {"H+", "He+", "C+", "Na+", "Mg+", "Si+", "S+", "K+", "Ca+", "Mn+", "Fe+", "Ca++"};

int ion_name_to_index(const char *ion_name, struct gas_cell_data *cell)
{
    for (int i = 0; i < NUM_RECOMB_TABLE_IONS; i++)
    {
        if (strcmp(ion_name, ion_names[i]) == 0)
        {
            return i;
        }
    }
    return 0;
}

/* String-based wrapper for alpha_recomb_grain (host convenience) */
MyFloat alpha_recomb_grain(int i, MyFloat temp, MyFloat x_elec, MyFloat shieldfac, const char *ion_name, struct particle_data *pp, struct gas_cell_data *cell)
{
    int j = ion_name_to_index(ion_name, cell);
    return alpha_recomb_grain(i, temp, x_elec, shieldfac, j, pp, cell);
}

/* Fraction of C atoms in CO: Kim 2023 Eq 25 */
MyFloat f_CO(int i, MyFloat temp, MyFloat x_elec, MyFloat shieldfac, MyFloat nHp, struct particle_data *pp, struct gas_cell_data *cell)
{
    MyFloat xi_cr16 = Get_CosmicRayIonizationRate_cgs(i, pp, cell) / 1e-16, Zd = pp[i].Metallicity[0] / All.SolarAbundances[0];
    MyFloat G0 = get_FUV_G0(i, shieldfac, 0, pp, cell);
    MyFloat n_COcrit = pow(4e3 * Zd / (xi_cr16 * xi_cr16), cbrt(G0)) * (50 * xi_cr16 / pow(Zd, 1.4));
    MyFloat nHcgs = cell[i].nHcgs();
    MyFloat f_CO = 0.5 * cell[i].MolecularMassFraction * (1 - DMAX(f_Cplus(i, temp, x_elec, shieldfac, pp, cell), f_Oplus(nHp))) / (1 + pow(n_COcrit / nHcgs, 2));
    return f_CO;
}

#endif
