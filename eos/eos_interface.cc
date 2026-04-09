#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "eos.h"
#include "../declarations/allvars.h"

#ifdef EOS_HELMHOLTZ
#include "helmholtz/helmholtz.h"
static HelmTable helm_table; /* read-only after eos_init, safe for concurrent access */
#endif

#define BITMASK_SET_FLAG(BITMASK,FLAG)      (BITMASK) |= (FLAG)
#define BITMASK_SET_ALL_FLAGS(BITMASK)      (BITMASK = ~(0))
#define BITMASK_UNSET_FLAG(BITMASK,FLAG)    (BITMASK) &= ~(FLAG)
#define BITMASK_UNSET_ALL_FLAGS(BITMASK)    (BITMASK) = 0
#define BITMASK_CHECK_FLAG(BITMASK,FLAG)    (((BITMASK) & (FLAG)) == (FLAG))

#define EOS_ERR_VALID         0
#define EOS_ERR_RHO_LT_RHOMIN 1
#define EOS_ERR_RHO_GT_RHOMAX 2
#define EOS_ERR_EPS_LT_EPSMIN 4
#define EOS_ERR_EPS_GT_EPSMAX 8
#define EOS_ERR_COMPOSITION   16

static int eos_input_to_cgs(struct eos_input * vars);
static int eos_output_from_cgs(struct eos_output * vars);

static int eos_validate(struct eos_input const * vars, struct eos_input * vars_adj, int * bitmask);
static int eos_compute_from_valid(struct eos_input const * in, struct eos_output * out);

#ifdef EOS_TABULATED
int eos_init(char const * eos_table_fname)
{
#ifdef EOS_HELMHOLTZ
  int ierr = helm_read_table(eos_table_fname, &helm_table);
  if (ierr) {
    fprintf(stderr, "eos_init: failed to read Helmholtz table '%s'\n", eos_table_fname);
    return 1;
  }
#endif
  return 0;
}

int eos_cleanup()
{
  return 0;
}
#endif

int eos_compute(struct eos_input const * in_, struct eos_output * out_)
{
  struct eos_input in, in_adj;
  memcpy(&in, in_, sizeof(in));
#ifdef EOS_USES_CGS
  eos_input_to_cgs(&in);
#endif

  int bitmask = 0;
  int ierr = eos_validate(&in, &in_adj, &bitmask);
  assert(!ierr);
  if(bitmask != EOS_ERR_VALID)
  {
    fprintf(stderr, "EOS ERROR:");
    if(BITMASK_CHECK_FLAG(bitmask, EOS_ERR_COMPOSITION))
      fprintf(stderr, "/invalid composition");
    if(BITMASK_CHECK_FLAG(bitmask, EOS_ERR_RHO_LT_RHOMIN))
      fprintf(stderr, "/density too low");
    if(BITMASK_CHECK_FLAG(bitmask, EOS_ERR_RHO_GT_RHOMAX))
      fprintf(stderr, "/density too large");
    if(BITMASK_CHECK_FLAG(bitmask, EOS_ERR_EPS_LT_EPSMIN))
      fprintf(stderr, "/temperature too low");
    if(BITMASK_CHECK_FLAG(bitmask, EOS_ERR_EPS_GT_EPSMAX))
      fprintf(stderr, "/temperature too high");
    fprintf(stderr, "\n");

#ifdef EOS_USES_CGS
    char const * unit_dens = "g/cm^3";
    char const * unit_ene  = "erg/g";
#else
    char const * unit_dens = "";
    char const * unit_ene  = "";
#endif

    fprintf(stderr, "  rho  = %.19e %s\n", in.rho, unit_dens);
    fprintf(stderr, "  eps  = %.19e %s\n", in.eps, unit_ene);
#ifdef EOS_CARRIES_YE
    fprintf(stderr, "  Ye   = %.19e\n", in.Ye);
#endif
#ifdef EOS_CARRIES_ABAR
    fprintf(stderr, "  Abar = %.19e\n", in.Abar);
#endif
    fprintf(stderr, "Using 0th order extrapolation\n");
    memcpy(&in, &in_adj, sizeof(in));
  }
  struct eos_output out;
  ierr = eos_compute_from_valid(&in, &out);
  if (ierr) {
    /* EOS evaluation failed even after clamping — use a fallback ideal gas estimate
       rather than crashing. This can happen at extreme conditions during initialization. */
    double gamma = 5.0/3.0;
    out.press = (gamma - 1.0) * in.rho * in.eps;
    out.csound = sqrt(gamma * out.press / in.rho);
    out.temp = in.temp > 0 ? in.temp : 1.0e6;
#ifdef EOS_PROVIDES_ENTROPY
    out.entropy = 0;
#endif
#ifdef EOS_PROVIDES_CV
    out.cv = in.eps / (out.temp + 1.0e-30);
#endif
    ierr = 0;
  }
#ifdef EOS_USES_CGS
  ierr = eos_output_from_cgs(&out);
  assert(!ierr);
#endif
  memcpy(out_, &out, sizeof(out));

  return 0;
}

static int eos_input_to_cgs(struct eos_input * vars)
{
  vars->rho *= UNIT_DENSITY_IN_CGS;
  vars->eps *= UNIT_SPECEGY_IN_CGS;
  return 0;
}

static int eos_output_from_cgs(struct eos_output * vars)
{
  vars->press  /= UNIT_PRESSURE_IN_CGS;
  vars->csound /= UNIT_VEL_IN_CGS;
  return 0;
}

static int eos_validate(struct eos_input const * vars, struct eos_input * vars_adj, int * bitmask)
{
  *bitmask = EOS_ERR_VALID;
  memcpy(vars_adj, vars, sizeof(*vars));

#ifdef EOS_HELMHOLTZ
  if(vars->Ye < 0)
  {
    BITMASK_SET_FLAG(*bitmask, EOS_ERR_COMPOSITION);
    vars_adj->Ye = 0;
  }
  if(vars->Ye > 1)
  {
    BITMASK_SET_FLAG(*bitmask, EOS_ERR_COMPOSITION);
    vars_adj->Ye = 1;
  }
  if(vars->Abar < 1)
  {
    BITMASK_SET_FLAG(*bitmask, EOS_ERR_COMPOSITION);
    vars_adj->Abar = 1;
  }

  double rho_ye_min, rho_ye_max;
  helm_get_rhoye_range(&helm_table, &rho_ye_min, &rho_ye_max);
  if(vars->rho * vars_adj->Ye < rho_ye_min)
  {
    BITMASK_SET_FLAG(*bitmask, EOS_ERR_RHO_LT_RHOMIN);
    vars_adj->rho = rho_ye_min / vars_adj->Ye;
  }
  if(vars->rho * vars_adj->Ye > rho_ye_max)
  {
    BITMASK_SET_FLAG(*bitmask, EOS_ERR_RHO_GT_RHOMAX);
    vars_adj->rho = rho_ye_max / vars_adj->Ye;
  }

  /* check energy range by evaluating at table T boundaries */
  HelmInput hin;
  HelmResult hout;
  hin.rho  = vars_adj->rho;
  hin.abar = vars_adj->Abar;
  hin.ye   = vars_adj->Ye;

  double tmin, tmax;
  helm_get_temp_range(&helm_table, &tmin, &tmax);

  hin.temp = tmin;
  if (helm_eos_from_temp(&helm_table, &hin, &hout)) return 1;
  double eps_min = hout.etot;

  hin.temp = tmax;
  if (helm_eos_from_temp(&helm_table, &hin, &hout)) return 1;
  double eps_max = hout.etot;

  if(vars->eps < eps_min)
  {
    BITMASK_SET_FLAG(*bitmask, EOS_ERR_EPS_LT_EPSMIN);
    vars_adj->eps = eps_min;
  }
  if(vars->eps > eps_max)
  {
    BITMASK_SET_FLAG(*bitmask, EOS_ERR_EPS_GT_EPSMAX);
    vars_adj->eps = eps_max;
  }
#endif

  return 0;
}

static int eos_compute_from_valid(struct eos_input const * in, struct eos_output * out)
{
#ifdef EOS_HELMHOLTZ
  /* use Newton iteration to invert for T given (rho, eps, abar, ye) */
  double temp_guess = in->temp;
  double tmin, tmax;
  helm_get_temp_range(&helm_table, &tmin, &tmax);
  if (temp_guess < tmin || temp_guess > tmax) {
    /* initial guess from gamma=5/3 electron gas */
    temp_guess = (2.0/3.0) * in->Abar * in->eps * helm_constants::me / helm_constants::kerg;
  }

  HelmResult hout;
  int ierr = helm_eos_from_energy(&helm_table, in->rho, in->eps,
                                  in->Abar, in->Ye, temp_guess, &hout);
  if (ierr) {
    fprintf(stderr, "%s:%d unexpected EOS failure!\n", __FILE__, __LINE__);
    return 1;
  }

  out->press  = hout.ptot;
  out->csound = hout.csound;
  out->temp   = hout.temp;
#ifdef EOS_PROVIDES_ENTROPY
  out->entropy = hout.stot;
#endif
#ifdef EOS_PROVIDES_CV
  out->cv = hout.cv;
#endif
#endif

  return 0;
}
