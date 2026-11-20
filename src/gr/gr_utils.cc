/* gr/gr_utils.cc -- host-side utilities of the GR module: the tabulated metric, the initial accelerations, failure reports, the
 * excision around the black hole.
 *
 * The device-callable physics is in gr/gr_functions.h (included through declarations/cell_data.h). */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "gr_kick_functions.h"
#include "../system/gpu_particles_arena.h" /* gpu_particles_uvm_alloc: shared-memory storage for the metric table */

#if (GR_METRIC<0)
/* Tabulated static spherically symmetric metric (TOV solutions): the file holds a header "# N log_scale" and N rows
   "r M(r) -g_tt r_iso" (areal radius, enclosed gravitational mass, -g_tt = exp(nu), isotropic radius), r linearly spaced from 0
   (log_scale 0) or logarithmically (1). Every rank reads the file, as in the C/C++ GR code; the table goes to shared memory so
   that get_covariant_metric can read it on the device through the All mirror. */
void gr_load_metric_table(void)
{
    /* every failure requests the controlled stop (endrun returns on the host) and returns at once, without publishing the table: begrun
       polls the stop right after this call, before any metric is evaluated */
    All.GR_MetricTable = NULL; All.GR_MetricTableN = 0;
    FILE *f = fopen(All.GR_MetricTableFile, "r");
    if(f == NULL) {printf("GR: cannot open the metric table '%s'\n", All.GR_MetricTableFile); endrun(9993002); return;}
    char line[1000]; int N = 0, logscale = 0;
    if(fgets(line, 999, f) == NULL || sscanf(line, "# %d %d", &N, &logscale) != 2 || N < 3) {printf("GR: bad header in the metric table '%s' (need '# N log' with N >= 3)\n", All.GR_MetricTableFile); fclose(f); endrun(9993003); return;}
    double *tab = (double *) gpu_particles_uvm_alloc(4*(size_t)N*sizeof(double), "gr_metric_table", 0);
    if(tab == NULL) {printf("GR: cannot allocate the metric table (%d rows)\n", N); fclose(f); endrun(9993004); return;}
    for(int i = 0; i < N; i++)
    {
        if(fgets(line, 999, f) == NULL || sscanf(line, "%le %le %le %le", tab + i, tab + N + i, tab + 2*N + i, tab + 3*N + i) != 4)
        {printf("GR: bad row %d in the metric table '%s'\n", i, All.GR_MetricTableFile); fclose(f); endrun(9993005); return;}
        /* columns r, M(r), -g_tt, r_iso: finite, radii non-negative and strictly increasing, M >= 0, positive lapse (r_iso is used, and
           checked, only with GR_METRIC_USE_ISOTROPIC_RADIUS: other tables may carry anything in that column, e.g. the density) */
        double r = tab[i], M = tab[N + i], mgtt = tab[2*N + i], riso = tab[3*N + i];
        int ok = isfinite(r) && isfinite(M) && isfinite(mgtt) && (r >= 0) && (M >= 0) && (mgtt > 0) && (i == 0 || r > tab[i-1]);
#ifdef GR_METRIC_USE_ISOTROPIC_RADIUS
        ok = ok && isfinite(riso) && (riso >= 0) && (i == 0 || riso > tab[3*N + i-1]);
        const char *radius_condition = "strictly increasing r_iso >= 0";
#else
        ok = ok && ((r == 0) || (2*M < r)); /* areal radius: g_rr = 1/(1 - 2M/r) needs every row outside its own horizon */
        const char *radius_condition = "2M < r (areal radius)";
#endif
        if(!ok) {printf("GR: row %d of the metric table '%s' is not valid (r=%g M=%g -g_tt=%g r_iso=%g: need finite values, strictly increasing radii >= 0, M >= 0, -g_tt > 0, %s)\n", i, All.GR_MetricTableFile, r, M, mgtt, riso, radius_condition); fclose(f); endrun(9993006); return;}
    }
    fclose(f);
    All.GR_MetricTable = tab; All.GR_MetricTableN = N; All.GR_MetricTableLog = logscale; /* (the spacing flag is informative: rows are found by bisection) */
    if(ThisTask == 0) {printf("GR: metric table '%s' loaded (%d radii, %s-spaced, r in [%g, %g], M(r_max) = %g)\n", All.GR_MetricTableFile, N, logscale ? "log" : "lin", tab[0], tab[N-1], tab[2*N-1]);}
}
#endif

#ifdef GENERAL_RELATIVITY

/* the first kick of every gas cell (over a nominal small step), so that HydroAccel/DtInternalEnergy hold the rates of the primitives from the initial fluxes */
void compute_initial_accelerations(void)
{
    /* the rates are finite differences over a short nominal interval: 1.1 MinSizeTimestep, or a small fraction of
       MaxSizeTimestep when MinSizeTimestep is 0 */
    double dt_0 = (All.MinSizeTimestep > 0) ? DMIN(All.MinSizeTimestep*1.1, All.MaxSizeTimestep) : 1.e-3*All.MaxSizeTimestep;
    for(int i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 0 || P[i].Mass <= 0) {continue;} /* (cells inside the excision radius were given zero mass in init) */
        gr_kick_impl(i, dt_0, P[i].Mass, P[i].Mass, 0, P, CellP);
    }
}

#ifdef PARTICLE_EXCISION

/* sets the mass to 0 inside GR_EXCISION_RADIUS horizon radii of the black hole (gr_hole_displacement for the Kerr metric) */
void check_boundary(int target)
{
#if (GR_METRIC==1)
    double r_min = gr_hole_displacement(P[target].Pos).norm();
    double a[3] = {0, 0, All.MetricParams[1]};
    double horizon = GR_HORIZON_RADIUS(All.MetricParams[0], a);
    if(r_min < GR_EXCISION_RADIUS*horizon) {P[target].Mass = 0;}
#endif
}

#endif /* PARTICLE_EXCISION */


#endif /* GENERAL_RELATIVITY */
