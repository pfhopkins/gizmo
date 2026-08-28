#ifndef INLINE_FUNC
#ifdef INLINE
#define INLINE_FUNC inline
#else
#define INLINE_FUNC
#endif
#endif

/*!
 * This file contains the definitions for the cooling.c routines
 */

/*!
 * This file was originally part of the GADGET3 code developed by Volker Springel.
 * The code has been modified by Phil Hopkins and Mike Grudic for GIZMO. Essentially everything has been re-written at this point.
 */

/* Two forms of the same body. The plain name is the host entry: cooling.cc owns
   the tables and fills them in for you. The _impl form takes them as data, and is
   the one to call from a kernel in any other file, which cannot reach the tables
   by name -- see cooling_tables.h. */
struct PhysicsTablesView;
GIZMO_GPU_FUNCTION double ThermalProperties(double u, double rho, int target, double *mu_guess, double *ne_guess, double *nH0_guess, double *nHp_guess, double *nHe0_guess, double *nHep_guess, double *nHepp_guess, struct particle_data *pp, struct gas_cell_data *cell);
GIZMO_GPU_FUNCTION double ThermalProperties_impl(double u, double rho, int target, double *mu_guess, double *ne_guess, double *nH0_guess, double *nHp_guess, double *nHe0_guess, double *nHep_guess, double *nHepp_guess, struct particle_data *pp, struct gas_cell_data *cell, const struct PhysicsTablesView *tables);
GIZMO_GPU_FUNCTION double return_uvb_shieldfac(int target, double gamma_12, double nHcgs, double logT, struct gas_cell_data *cell);
GIZMO_GPU_FUNCTION double return_local_gammamultiplier(int target, struct gas_cell_data *cell);
GIZMO_GPU_FUNCTION double evaluate_Compton_heating_cooling_rate(int target, double T, double nHcgs, double n_elec, double shielding_factor_for_exgalbg, struct gas_cell_data *cell);
GIZMO_GPU_FUNCTION double get_background_radiation_temperature_for_emission_corrections(int target, struct gas_cell_data *cell);
void   InitCool(void);
#ifndef CHIMES
int    InitCoolMemory(void);
void   IonizeParams(void);
void   IonizeParamsFunction(void);
void   IonizeParamsTable(void);
//double INLINE_FUNC LogTemp(double u, double ne);
void   MakeCoolingTable(void);
void   ReadIonizeParams(const char *fname);
void   SetZeroIonization(void);
#endif
