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

double ThermalProperties(double u, double rho, int target, double *mu_guess, double *ne_guess, double *nH0_guess, double *nHp_guess, double *nHe0_guess, double *nHep_guess, double *nHepp_guess, struct particle_data *pp, struct gas_cell_data *cell);
double return_uvb_shieldfac(int target, double gamma_12, double nHcgs, double logT, struct gas_cell_data *cell);
double return_local_gammamultiplier(int target, struct gas_cell_data *cell);
double evaluate_Compton_heating_cooling_rate(int target, double T, double nHcgs, double n_elec, double shielding_factor_for_exgalbg, struct gas_cell_data *cell);
double get_background_radiation_temperature_for_emission_corrections(int target, struct gas_cell_data *cell);
void   InitCool(void);
#ifndef CHIMES
void   InitCoolMemory(void);
void   IonizeParams(void);
void   IonizeParamsFunction(void);
void   IonizeParamsTable(void);
//double INLINE_FUNC LogTemp(double u, double ne);
void   MakeCoolingTable(void);
void   ReadIonizeParams(char *fname);
void   SetZeroIonization(void);
#endif
