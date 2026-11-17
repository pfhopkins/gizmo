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
   (log_scale 0) or logarithmically (1). Every rank reads the file, as in the original code; the table goes to shared memory so
   that get_covariant_metric can read it on the device through the All mirror. */
void gr_load_metric_table(void)
{
    FILE *f = fopen(All.GR_MetricTableFile, "r");
    if(f == NULL) {printf("GR: cannot open the metric table '%s'\n", All.GR_MetricTableFile); endrun(9993002);}
    char line[1000]; int N = 0, logscale = 0;
    if(fgets(line, 999, f) == NULL || sscanf(line, "# %d %d", &N, &logscale) != 2 || N < 2) {printf("GR: bad header in the metric table '%s'\n", All.GR_MetricTableFile); endrun(9993003);}
    double *tab = (double *) gpu_particles_uvm_alloc(4*(size_t)N*sizeof(double), "gr_metric_table", 0);
    if(tab == NULL) {printf("GR: cannot allocate the metric table (%d rows)\n", N); endrun(9993004);}
    for(int i = 0; i < N; i++)
    {
        if(fgets(line, 999, f) == NULL || sscanf(line, "%le %le %le %le", tab + i, tab + N + i, tab + 2*N + i, tab + 3*N + i) != 4)
        {printf("GR: bad row %d in the metric table '%s'\n", i, All.GR_MetricTableFile); endrun(9993005);}
    }
    fclose(f);
    All.GR_MetricTable = tab; All.GR_MetricTableN = N; All.GR_MetricTableLog = logscale;
    if(ThisTask == 0) {printf("GR: metric table '%s' loaded (%d radii, %s-spaced, r in [%g, %g], M(r_max) = %g)\n", All.GR_MetricTableFile, N, logscale ? "log" : "lin", tab[0], tab[N-1], tab[2*N-1]);}
}
#endif

#ifdef GENERAL_RELATIVITY

/* the first kick of every gas cell (over a nominal small step), so that HydroAccel/DtInternalEnergy hold the rates of the primitives from the initial fluxes */
void compute_initial_accelerations(void)
{
    double dt_0 = DMIN(All.MinSizeTimestep*1.1, All.MaxSizeTimestep);
    for(int i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 0) {continue;}
        gr_kick_impl(i, dt_0, P[i].Mass, P[i].Mass, 0, P, CellP);
    }
}

/* prints the state of a failed conserved-to-primitive inversion (the con2prim itself never prints) */
void gr_con2prim_report_failure(const char *where, int code, const struct gr_conservatives *cons, const struct gr_primitives *prims, const struct gr_metric *gr)
{
    printf("GR con2prim failed in %s with code %d (task %d)\n", where, code, ThisTask);
    printf(" *dens        = %26.16e \n *sx          = %26.16e \n *sy          = %26.16e \n *sz          = %26.16e \n *tau         = %26.16e \n",
           cons->D, cons->S[0], cons->S[1], cons->S[2], cons->tau);
    printf(" *rho (guess) = %26.16e \n *velx        = %26.16e \n *vely        = %26.16e \n *velz        = %26.16e \n *epsilon     = %26.16e \n *pressure    = %26.16e \n",
           prims->rho0, prims->v[0], prims->v[1], prims->v[2], prims->u, prims->p);
    printf(" metric: alpha %g gdet %g beta %g %g %g gamma xx %g xy %g xz %g yy %g yz %g zz %g\n", gr->alpha, gr->gdet, gr->beta[0], gr->beta[1], gr->beta[2],
           gr->gamma[0][0], gr->gamma[0][1], gr->gamma[0][2], gr->gamma[1][1], gr->gamma[1][2], gr->gamma[2][2]);
    fflush(stdout);
}

#ifdef PARTICLE_EXCISION
/* damps the Eulerian velocity (and the field) of a cell approaching the horizon, as in the original GR code */
void smooth_conservatives_inside_horizon(int i, double r, double r_EH, double M)
{
    const struct gr_metric *gr = &(CellP[i].gr);
    double exponent = (0.5*sqrt(GR_EXCISION_RADIUS*M)*(r - r_EH)*(r - r_EH))/(r_EH - 0.4), Chi = exp(-exponent);
    P[i].Vel = Vcoord(Chi*Vphys(P[i].Vel, gr), gr);
    CellP[i].VelPred = Vcoord(Chi*Vphys(CellP[i].VelPred, gr), gr);
#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && (HYDRO_FIX_MESH_MOTION>=5)
    CellP[i].ParticleVel = Vcoord(Chi*Vphys(CellP[i].ParticleVel, gr), gr);
#endif
}

/* sets the mass to 0 inside GR_EXCISION_RADIUS horizon radii of the black hole (at the origin for the Kerr metric) */
void check_boundary(int target)
{
#if (GR_METRIC==1)
    double r_min = P[target].Pos.norm();
    double a[3] = {0, 0, All.MetricParams[1]};
    double horizon = EHradius(All.MetricParams[0], a);
    if(r_min < GR_EXCISION_RADIUS*horizon) {P[target].Mass = 0;}
#endif
}

#endif /* PARTICLE_EXCISION */


#endif /* GENERAL_RELATIVITY */
