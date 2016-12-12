#ifndef INLINE_FUNC
#ifdef INLINE
#define INLINE_FUNC inline
#else
#define INLINE_FUNC
#endif
#endif

/*
 * This file contains the definitions for the cooling.c routines
 *
 * This file was originally part of the GADGET3 code developed by
 *   Volker Springel (volker.springel@h-its.org). The code has been modified by
 *   Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */


double ThermalProperties(double u, double rho, double *ne_guess, double *nH0_pointer, double *nHeII_pointer, double *mu_pointer, int target);
void   InitCool(void);
void   InitCoolMemory(void);
void   IonizeParams(void);
void   IonizeParamsFunction(void);
void   IonizeParamsTable(void);
//double INLINE_FUNC LogTemp(double u, double ne);
void   MakeCoolingTable(void);
void   ReadIonizeParams(char *fname);
void   SetZeroIonization(void);
void   TestCool(void);

void   find_abundances_and_rates(double logT, double rho, double *ne_guess, int target, double shieldfac);
double convert_u_to_temp(double u, double rho, double *ne_guess, int target);
double CoolingRate(double logT, double rho, double *nelec, int target);
double CoolingRateFromU(double u, double rho, double *ne_guess, int target);
double DoCooling(double u_old, double rho, double dt, double *ne_guess, int target);
double GetCoolingTime(double u_old, double rho,  double *ne_guess, int target);
double DoInstabilityCooling(double m_old, double u, double rho, double dt, double fac, double *ne_guess, int target);
double get_mu(double T_guess, double rho, double *ne_guess, int target);
double yhelium(int target);
double calculate_shieldfac(double nHcgs, double NH_SS_z); 

#ifdef GRACKLE
void InitGrackle(void);
double CallGrackle(double u_old, double rho, double dt, double *ne_guess, int target, int mode);
#endif

