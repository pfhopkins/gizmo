/* dm_cooling_tables.h — Consolidated table data for HYDRO_MULTIFLUID_DM_COOLING.
 *
 * Mirrors the cooling/cooling_tables.h (CoolTables) pattern: a single struct
 * holding the temperature grid + rate tables, defined as __managed__ in
 * cooling/cooling.cc so device kernels can read it directly.  Pointer members
 * point to UVM-managed buffers allocated via Kokkos::kokkos_malloc<SharedSpace>.
 *
 * The dm-cooling chain in sidm/dm_fluid_functions.h is device-callable
 * (KOKKOS_INLINE_FUNCTION) for the on-the-fly cooling rate evaluation;
 * the per-temperature table-builder helpers (dm_g_integral et al.) stay
 * host-only because they run once at startup in dm_MakeCoolingTable.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef DM_COOLING_TABLES_H
#define DM_COOLING_TABLES_H

#define NCOOLTAB_DM  2000  /* number of entries in dm cooling rate tables */

struct dm_cooling_tables_t {
    double Tmin;          /* log10(T_min/K), set in dm_MakeCoolingTable */
    double Tmax;          /* log10(T_max/K) */
    double deltaT;        /* (Tmax-Tmin)/NCOOLTAB_DM */
    double *BetaH0;       /* collisional excitation cooling [NCOOLTAB_DM+1] */
    double *Betaff;       /* free-free (bremsstrahlung) cooling [NCOOLTAB_DM+1] */
    double *AlphaHp;      /* recombination rate [NCOOLTAB_DM+1] */
    double *AlphaHpRate;  /* recombination cooling rate [NCOOLTAB_DM+1] */
    double *GammaeH0;     /* collisional ionization rate [NCOOLTAB_DM+1] */
};

/* Defined in cooling/cooling.cc as __managed__ under GIZMO_GPU_COMPILER.
   Other TUs include sidm/dm_fluid_functions.h, which forward-declares the
   instance via `extern struct dm_cooling_tables_t DMCoolTables;`. */

/* Table builder: allocates DMCoolTables buffers in Kokkos SharedSpace
   (UVM) and populates them. Body lives in cooling/cooling.cc (GPU_OBJS,
   has Kokkos_Core.hpp). Called once at startup from core/begrun.cc. This
   declaration is the ONLY symbol core/begrun.cc needs to see for ADM
   cooling -- letting begrun.cc include this minimal header instead of
   sidm/dm_fluid_functions.h keeps Kokkos CUDA headers out of host TUs. */
void InitCool_dm(void);

#endif /* DM_COOLING_TABLES_H */
