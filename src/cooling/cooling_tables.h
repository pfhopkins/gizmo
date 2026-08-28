/* cooling_tables.h — Consolidated cooling table data structure.
 *
 * Holds all cooling/ionization table data that per-particle cooling functions
 * need to access. With GPU offload, the struct is __managed__ so device kernels
 * can access it directly. Pointer members point to separately allocated managed
 * memory (Kokkos SharedSpace on GPU, mymalloc on CPU).
 *
 * This replaces ~15 individual static variables that were scattered at file scope
 * in cooling.cc, making the cooling tables accessible from any translation unit
 * via the header (required for cooling_functions.h to work across TUs).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef COOLING_TABLES_H
#define COOLING_TABLES_H

#define NCOOLTAB  2000 /* number of entries in cooling rate tables (temperature grid) */

struct cooling_tables_t {
    /* Temperature grid */
    double Tmin, Tmax, deltaT;

    /* Cooling/ionization rate tables [NCOOLTAB+1] entries each, indexed by temperature */
    double *BetaH0;      /* collisional ionization cooling: HI */
    double *BetaHep;     /* collisional ionization cooling: HeII */
    double *Betaff;      /* free-free (bremsstrahlung) cooling */
    double *AlphaHp;     /* recombination rate: HII */
    double *AlphaHep;    /* recombination rate: HeII */
    double *Alphad;      /* dielectronic recombination: HeII */
    double *AlphaHepp;   /* recombination rate: HeIII */
    double *GammaeH0;    /* collisional ionization rate: HI */
    double *GammaeHe0;   /* collisional ionization rate: HeI */
    double *GammaeHep;   /* collisional ionization rate: HeII */

    /* UV background parameters (set by IonizeParams from TREECOOL table) */
    double J_UV;         /* UV intensity normalization */
    double gJH0;         /* photoionization rate: HI */
    double gJHep;        /* photoionization rate: HeII */
    double gJHe0;        /* photoionization rate: HeI */
    double epsH0;        /* photoheating rate: HI */
    double epsHep;       /* photoheating rate: HeII */
    double epsHe0;       /* photoheating rate: HeI */

#ifdef COOL_METAL_LINES_BY_SPECIES
    /* Metal-line cooling tables (interpolation grids over nH, T, redshift) */
    float *SpCoolTable0; /* cooling table at redshift z0 */
    float *SpCoolTable1; /* cooling table at redshift z1 (for interpolation; cosmological only) */
#endif
};

/* The global cooling tables instance is defined in cooling.cc, which owns it.
   Code inside that file reads it by name. Code anywhere else -- and in
   particular any function that may be compiled into a device kernel -- must
   NOT declare it, and must take the tables as data instead: see the table
   view below. Declaring the instance in another translation unit is a silent
   correctness bug, not a link error: a device symbol cannot be shared across
   translation units without relocatable device code, which this build does
   not use, so the declaration binds to storage the owner never fills. */

/* Table view -- how any function reaches table data it does not own.
 *
 * A device kernel cannot reach a table by name across translation units, so
 * the tables travel as data: the host fills this view from the live tables at
 * dispatch, the kernel captures it by value, and the functions below it take
 * it by pointer. Two properties follow, and both matter:
 *
 *   - The values are current by construction. The view is filled on the host
 *     immediately before each dispatch, so table contents that change during
 *     a run -- the radiation background scalars are rewritten every timestep --
 *     are picked up with no separate refresh step to forget.
 *
 *   - Adding a table does not touch a single call site. Every entry point
 *     takes this one view, whatever it happens to need out of it, so a new
 *     table is a new member here plus the code that reads it. Callers never
 *     name the tables they are passing through.
 *
 * The members are pointers to storage the owning module allocates and keeps
 * alive; the view neither owns nor frees anything. Each is present under the
 * same condition that compiles its table, and the view itself always exists,
 * so functions taking it keep one signature in every configuration. */
struct cooling_tables_t;
struct dm_cooling_tables_t;

struct PhysicsTablesView
{
#if !defined(CHIMES)
    const struct cooling_tables_t *cooling;
#endif
#ifdef HYDRO_MULTIFLUID_DM_COOLING
    const struct dm_cooling_tables_t *dm_cooling;
#endif
};

/* Fill a view from the live tables. Defined in cooling.cc, which owns them.
   Host-only: call it before a dispatch and let the kernel capture the result
   by value. */
struct PhysicsTablesView gizmo_physics_tables_view(void);

#endif /* COOLING_TABLES_H */
