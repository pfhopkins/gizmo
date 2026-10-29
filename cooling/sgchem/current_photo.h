#if CHEMISTRYNETWORK != 1
#define CPHOTO
      REAL G_dust, chi_isrf, phrates(npr)
      common /current_photo/ G_dust, chi_isrf, phrates
c$omp threadprivate(/current_photo/)
#endif
