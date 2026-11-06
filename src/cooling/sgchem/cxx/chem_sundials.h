/* C++ implementation: Ulrich Steinwandel (uli@mpa-garching.mpg.de).
   Based on the SGChem Fortran networks by Simon Glover (GNU GPL, as distributed in Arepo). */
// chem_sundials.h -- SUNDIALS version switch for the CVODE context (include after the SUNDIALS headers).
#ifndef CHEM_SUNDIALS_H
#define CHEM_SUNDIALS_H
#include <sundials/sundials_config.h>
#if SUNDIALS_VERSION_MAJOR >= 7
#define SGCHEM_SUN_COMM_NULL SUN_COMM_NULL   // 7.x: SUNContext_Create takes a communicator
#else
#define SGCHEM_SUN_COMM_NULL NULL            // 6.x
#endif
#endif
