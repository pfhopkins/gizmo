#ifndef EOS_INTERFACE_H
#define EOS_INTERFACE_H

#include "../GIZMO_config.h"

#ifdef EOS_HELMHOLTZ
#define EOS_TABULATED
#define EOS_USES_CGS
#define EOS_CARRIES_YE
#define EOS_CARRIES_ABAR
#define EOS_PROVIDES_ENTROPY
#define EOS_PROVIDES_CV
#endif

#ifdef EOS_ANEOS
#define EOS_TABULATED
#define EOS_USES_CGS
#define EOS_PROVIDES_ENTROPY
#define EOS_PROVIDES_CV
#endif

#if defined(EOS_ANEOS) && defined(EOS_HELMHOLTZ)
#error "EOS_ANEOS and EOS_HELMHOLTZ are mutually exclusive"
#endif

struct eos_input
{
  double rho;         /* Density */
  double eps;         /* Specific internal energy */
#ifdef EOS_CARRIES_YE
  double Ye;          /* Electron fraction */
#endif
#ifdef EOS_CARRIES_ABAR
  double Abar;        /* Mean atomic weight (in atomic mass units) */
#endif
  double temp;        /* Temperature initial guess */
};

struct eos_output
{
  double press;       /* Pressure */
  double csound;      /* Sound speed */
  double temp;        /* Temperature (in Kelvin) */
#ifdef EOS_PROVIDES_ENTROPY
  double entropy;     /* Entropy (in CGS) */
#endif
#ifdef EOS_PROVIDES_CV
  double cv;          /* Specific heat at constant volume (in CGS) */
#endif
};

#ifdef EOS_TABULATED
int eos_init(char const * eos_table_fname);
int eos_cleanup();
#endif

/* Evaluating the EOS is eos_compute_P in eos/eos_functions.h -- a header body,
   because it has to be callable from a device kernel and a call cannot cross a
   translation unit without relocatable device code. It takes the table
   explicitly; eos/eos_interface.cc owns the table and hands it out through
   helm_table_view(), declared beside the HelmTable type in helmholtz.h.
   This header stays free of that type so that allvars.h, which includes it,
   does not pull the table definition into every translation unit. */

#endif
