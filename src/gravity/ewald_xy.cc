#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include "../declarations/allvars.h"
#include "../core/proto.h"

/* GRAVITY_PERIODIC_XY: Ewald correction for self-gravity periodic in x/y and open in z (stratified tall boxes), from upstream's
 * gravity/ewald_xy.cc (starforge_dev, used by stella_dev's tall box). The tree supplies the nearest-image force; the pair kernel
 * adds the difference between the doubly-periodic lattice sum and that nearest image (grav_pair_xy_ewald, gravtree_force_kernel.h)
 * from a table filled here by 2D Ewald summation (Parry 1975; Grzybowski, Gwozdz & Brodka 2000). Lattice: the images of a source at
 * nearest-image separation dr sit at dr + n e1 + m e2 with e1 = (L_x, -s), e2 = (0, L_y); s = Shearing_Box_Pos_Offset is the y-shift
 * NEAREST_XYZ applies across the x boundary (BOX_SHEARING; else 0). Reciprocal vectors: b1 = 2pi (1/L_x, 0), b2 = 2pi (s/(L_x L_y),
 * 1/L_y). With shear, tables sit on a grid of EWXY_NS values of s in [0, L_y); the two that bracket the current s are kept and
 * interpolated linearly, refilled as s advances. A table covers dx in [0, L_x/2], dy in [-L_y/2, L_y/2], dz in [0, EWXY_ZMAX_IN_L max(L_x, L_y)]; inversion and
 * z-reflection give the rest; beyond it the analytic sheet-minus-point-mass field. The table lives in Kokkos shared memory so the
 * device walk reads it; its pointer and sizes sit in All (device-visible). */

#ifdef GRAVITY_PERIODIC_XY

/* exact correction (unit mass, G=1) for separation (dx,dy,dz) = source - target and lattice shift s: out[0..2] = acceleration of the target from all
   images minus the nearest one; out[3] = the same for the potential */
void ewald_xy_exact(double dx, double dy, double dz, double s, double Lx, double Ly, double out[4])
{
    const double A = Lx * Ly, alpha = 2.0 / sqrt(A), rcut = 5.0 / alpha, kcut = 10.0 * alpha, sqrtpi = sqrt(M_PI);
    s -= Ly * floor(s / Ly + 0.5); /* the same lattice for s and s - L_y; keeps it near-rectangular */
    const double rx = -dx, ry = -dy, rz = -dz; /* r = target - source; psi(r) = sum_R 1/|r+R| (Ewald-regularised), psi_c = psi - 1/|r| */
    double g[3] = {0, 0, 0}, psi = 0; /* g = grad psi_c at r */

    const int nmax = (int)ceil(rcut / Lx) + 1; /* real-space sum */
    for(int n = -nmax; n <= nmax; n++)
    {
        const double Rx = n * Lx, Ry0 = -n * s;
        const int m0 = (int)floor(-(ry + Ry0) / Ly), mmax = (int)ceil(rcut / Ly) + 1;
        for(int m = m0 - mmax; m <= m0 + mmax + 1; m++)
        {
            const double px = rx + Rx, py = ry + Ry0 + m * Ly, pz = rz, rho2 = px*px + py*py + pz*pz;
            const bool self = (n == 0 && m == 0);
            if(!self && rho2 > rcut * rcut) {continue;}
            const double rho = sqrt(rho2), ar = alpha * rho, gauss = (2.0 * alpha / sqrtpi) * exp(-ar * ar);
            double f, dfdr;
            if(self) /* erfc(a r)/r - 1/r = -erf(a r)/r, smooth at r = 0 */
            {
                if(ar < 1.0e-4) {f = -2.0 * alpha / sqrtpi * (1.0 - ar * ar / 3.0); dfdr = 4.0 * alpha * alpha * alpha / (3.0 * sqrtpi) * rho;}
                else {f = -erf(ar) / rho; dfdr = erf(ar) / rho2 - gauss / rho;}
            }
            else {f = erfc(ar) / rho; dfdr = -erfc(ar) / rho2 - gauss / rho;}
            psi += f;
            if(rho > 0) {g[0] += dfdr * px / rho; g[1] += dfdr * py / rho; g[2] += dfdr * pz / rho;}
        }
    }

    const double b1x = 2.0 * M_PI / Lx, b2x = 2.0 * M_PI * s / A, b2y = 2.0 * M_PI / Ly; /* reciprocal-space sum */
    const int lmax = (int)ceil(kcut / b2y) + 1, hmax = (int)ceil(kcut / b1x) + 1;
    for(int l = -lmax; l <= lmax; l++)
    {
        const double ky = l * b2y, kx0 = l * b2x;
        const int h0 = (int)floor(-kx0 / b1x);
        for(int h = h0 - hmax; h <= h0 + hmax + 1; h++)
        {
            if(h == 0 && l == 0) {continue;}
            const double kx = kx0 + h * b1x, k = sqrt(kx*kx + ky*ky);
            if(k > kcut) {continue;}
            const double a = k / (2.0 * alpha), ep = exp(k * rz) * erfc(a + alpha * rz), em = exp(-k * rz) * erfc(a - alpha * rz);
            const double phase = kx * rx + ky * ry, c = cos(phase), sn = sin(phase), pre = M_PI / A;
            psi += pre * c / k * (ep + em);
            g[0] -= pre * sn * kx / k * (ep + em);
            g[1] -= pre * sn * ky / k * (ep + em);
            g[2] += pre * c * (ep - em);
        }
    }

    psi -= (2.0 * M_PI / A) * (rz * erf(alpha * rz) + exp(-alpha * alpha * rz * rz) / (alpha * sqrtpi)); /* k = 0 term */
    g[2] -= (2.0 * M_PI / A) * erf(alpha * rz);

    for(int d = 0; d < 3; d++) {out[d] = g[d];} /* target acceleration = grad psi(r), r = -dr; potential = -psi */
    out[3] = -psi;
}


/* collective: fill one table (shift s) in shared memory, split over the ranks */
static void ewald_xy_fill_table(float *tab, double s)
{
    const size_t npts = (size_t)(EWXY_NXH + 1) * (EWXY_NXY + 1) * (All.EwaldXY_nz + 1);
    std::vector<int> cnt(NTask), off(NTask);
    for(int task = 0; task < NTask; task++) {size_t beg = npts * task / NTask, end = npts * (task + 1) / NTask; off[task] = (int)(4 * beg); cnt[task] = (int)(4 * (end - beg));}
    std::vector<float> mine(cnt[ThisTask] + 1);
    for(size_t p = (size_t)off[ThisTask] / 4, q = 0; q < (size_t)cnt[ThisTask] / 4; p++, q++)
    {
        const int k = (int)(p % (All.EwaldXY_nz + 1)), j = (int)((p / (All.EwaldXY_nz + 1)) % (EWXY_NXY + 1)), i = (int)(p / ((size_t)(All.EwaldXY_nz + 1) * (EWXY_NXY + 1)));
        double out[4]; ewald_xy_exact(i * All.EwaldXY_dx, -0.5 * All.EwaldXY_Ly + j * All.EwaldXY_dy, k * All.EwaldXY_dz, s, All.EwaldXY_Lx, All.EwaldXY_Ly, out);
        for(int d = 0; d < 4; d++) {mine[4 * q + d] = (float)out[d];}
    }
    std::vector<float> all(4 * npts);
    MPI_Allgatherv(mine.data(), cnt[ThisTask], MPI_FLOAT, all.data(), cnt.data(), off.data(), MPI_FLOAT, MPI_COMM_WORLD);
    gizmo_kokkos_fence(); /* no walk reads the old contents any more */
    for(size_t n = 0; n < 4 * npts; n++) {tab[n] = all[n];}
    gizmo_kokkos_fence();
}

/* before each gravity walk: the table, or with shear the two that bracket the current offset. Communicates only where a table is
   filled: on the first call of a run (also after a restart), and with shear when the offset passes a point of its grid */
void ewald_xy_update(void)
{
#ifndef SELFGRAVITY_OFF
    static float *ewxy_tab[2] = {NULL, NULL}; /* not All's pointers: All comes back from restart files with stale ones */
    static int slot[2] = {-1, -1};
#ifdef BOX_SHEARING
    enum {EWXY_NTAB = 2};
#else
    enum {EWXY_NTAB = 1};
#endif
    if(!ewxy_tab[0])
    {
        if(All.ComovingIntegrationOn) {if(ThisTask == 0) {printf("GRAVITY_PERIODIC_XY does not support comoving integration\n");} endrun(1);}
        All.EwaldXY_Lx = boxSize_X; All.EwaldXY_Ly = boxSize_Y; All.EwaldXY_zmax = EWXY_ZMAX_IN_L * DMAX(boxSize_X, boxSize_Y);
        All.EwaldXY_dx = boxSize_X / EWXY_NXY; All.EwaldXY_dy = boxSize_Y / EWXY_NXY;
        All.EwaldXY_nz = (int)ceil(All.EwaldXY_zmax / DMIN(All.EwaldXY_dx, All.EwaldXY_dy)); All.EwaldXY_dz = All.EwaldXY_zmax / All.EwaldXY_nz;
        for(int t = 0; t < EWXY_NTAB; t++) {ewxy_tab[t] = (float *) gizmo_gpu_alloc_shared((size_t)4 * (EWXY_NXH + 1) * (EWXY_NXY + 1) * (All.EwaldXY_nz + 1) * sizeof(float), "ewald_xy_table");}
        if(ThisTask == 0) {printf("GRAVITY_PERIODIC_XY: Ewald table %d x %d x %d over |dz| <= %g\n", EWXY_NXH + 1, EWXY_NXY + 1, All.EwaldXY_nz + 1, All.EwaldXY_zmax);}
    }
#ifdef BOX_SHEARING
    const double ds = All.EwaldXY_Ly / EWXY_NS;
    double s = Shearing_Box_Pos_Offset; s -= All.EwaldXY_Ly * floor(s / All.EwaldXY_Ly);
    int lo = (int)floor(s / ds); if(lo >= EWXY_NS) {lo = EWXY_NS - 1;}
    if(slot[1] == lo || slot[0] == lo + 1) {float *t = ewxy_tab[0]; ewxy_tab[0] = ewxy_tab[1]; ewxy_tab[1] = t; int k = slot[0]; slot[0] = slot[1]; slot[1] = k;} /* a kept table changes its place */
    for(int t = 0; t < 2; t++) {if(slot[t] != lo + t) {ewald_xy_fill_table(ewxy_tab[t], (lo + t) * ds); slot[t] = lo + t;}} /* slot EWXY_NS is s = L_y, the lattice of slot 0 */
    All.EwaldXY_TableNext = ewxy_tab[1]; All.EwaldXY_WeightNext = s / ds - lo;
#else
    if(slot[0] < 0) {ewald_xy_fill_table(ewxy_tab[0], 0); slot[0] = 0;}
#endif
    All.EwaldXY_Table = ewxy_tab[0];
#ifdef BOX_SHEARING
    const size_t origin = 4 * (size_t)(EWXY_NXY / 2) * (All.EwaldXY_nz + 1) + 3; /* potential at dr = 0 */
    All.EwaldXY_SelfPot = (1 - All.EwaldXY_WeightNext) * ewxy_tab[0][origin] + All.EwaldXY_WeightNext * ewxy_tab[1][origin];
#else
    {double c0[4]; ewald_xy_exact(0, 0, 0, 0, All.EwaldXY_Lx, All.EwaldXY_Ly, c0); All.EwaldXY_SelfPot = c0[3];}
#endif
#endif
}

#endif
