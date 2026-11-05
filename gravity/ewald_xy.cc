#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include "../declarations/allvars.h"
#include "../core/proto.h"

/* GRAVITY_PERIODIC_XY: Ewald correction for self-gravity that is periodic (or shear-periodic, under BOX_SHEARING)
 * in x/y and open in z, for stratified boxes. The tree supplies the nearest-image force; ewald_xy_corr returns the
 * difference between the full doubly-periodic lattice sum and that nearest image, from tables filled by 2D Ewald
 * summation (Parry 1975; Grzybowski, Gwozdz & Brodka 2000).
 *
 * Lattice: the images of a source at nearest-image separation dr (source - target) sit at dr + n*e1 + m*e2, with
 * e1 = (L_x, -s), e2 = (0, L_y), where s = Shearing_Box_Pos_Offset is the y-shift NEAREST_XYZ applies across the
 * x-boundary (s = 0 without shear). Reciprocal vectors: b1 = 2pi (1/L_x, 0), b2 = 2pi (s/(L_x L_y), 1/L_y).
 *
 * The tables cover dx in [0, L_x/2], dy in [-L_y/2, L_y/2], dz in [0, EWXY_ZMAX_IN_L * max(L_x, L_y)]; the rest follows from
 * the lattice's inversion symmetry (C(-d) = -C(d)) and z-reflection (C_z odd, C_x, C_y and potential even in dz). Beyond that
 * the k != 0 terms are below ~exp(-4 pi) and the correction is the analytic sheet-minus-point-mass field. With shear
 * the correction depends on s; tables are kept on a grid of EWXY_NS values of s in [0, L_y) and interpolated
 * linearly between the two that bracket the current s, refilled collectively as s advances. */

#ifdef GRAVITY_PERIODIC_XY

#define EWXY_NXY 48        /* table cells across each of L_x, L_y */
#define EWXY_ZMAX_IN_L 2.0 /* table height in units of max(L_x, L_y) */
#ifdef BOX_SHEARING
#define EWXY_NS 64         /* s-grid cells across L_y */
#endif

namespace {
struct ewxy_table {int slot = -1; std::vector<float> v;}; /* 4 floats per node: acc_x, acc_y, acc_z, pot */
ewxy_table ewxy_tab[2]; /* [0] at the s-grid point below the current s, [1] above */
double ewxy_wslot = 0;  /* interpolation weight of ewxy_tab[1] */
int ewxy_nz = 0;
double ewxy_Lx, ewxy_Ly, ewxy_zmax, ewxy_dx, ewxy_dy, ewxy_dz;
#define EWXY_NXH (EWXY_NXY / 2) /* table cells across [0, L_x/2] */
inline size_t ewxy_index(int i, int j, int k) {return 4 * ((size_t)(i * (EWXY_NXY + 1) + j) * (ewxy_nz + 1) + k);}
}


/* exact correction (unit mass, G=1) for separation (dx,dy,dz) = source - target and lattice shift s:
   out[0..2] = acceleration of the target from all images minus the nearest one; out[3] = the same for the potential */
void ewald_xy_exact(double dx, double dy, double dz, double s, double Lx, double Ly, double out[4])
{
    const double A = Lx * Ly, alpha = 2.0 / sqrt(A), rcut = 5.0 / alpha, kcut = 10.0 * alpha, sqrtpi = sqrt(M_PI);
    s -= Ly * floor(s / Ly + 0.5); /* same lattice for s and s - L_y; keeps it near-rectangular */
    /* work with r = target - source = -dr; psi(r) = sum_R 1/|r+R| (Ewald-regularised), psi_c = psi - 1/|r| */
    const double rx = -dx, ry = -dy, rz = -dz;
    double g[3] = {0, 0, 0}, psi = 0; /* g = grad psi_c at r */

    /* real-space sum */
    const int nmax = (int)ceil(rcut / Lx) + 1;
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
            double f, dfdr; /* term and d(term)/drho */
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

    /* reciprocal-space sum */
    const double b1x = 2.0 * M_PI / Lx, b2x = 2.0 * M_PI * s / A, b2y = 2.0 * M_PI / Ly;
    const int lmax = (int)ceil(kcut / b2y) + 1;
    for(int l = -lmax; l <= lmax; l++)
    {
        const double ky = l * b2y, kx0 = l * b2x;
        const int h0 = (int)floor(-kx0 / b1x), hmax = (int)ceil(kcut / b1x) + 1;
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

    /* k = 0 term */
    psi -= (2.0 * M_PI / A) * (rz * erf(alpha * rz) + exp(-alpha * alpha * rz * rz) / (alpha * sqrtpi));
    g[2] -= (2.0 * M_PI / A) * erf(alpha * rz);

    /* target acceleration = grad psi(r) and r = -dr; potential = -psi */
    for(int d = 0; d < 3; d++) {out[d] = g[d];}
    out[3] = -psi;
}


/* analytic correction beyond the table: uniform sheet minus the nearest-image point mass */
static void ewald_xy_far(const double dr[3], double out[4])
{
    const double A = ewxy_Lx * ewxy_Ly, r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2], r = sqrt(r2), r3 = r2 * r;
    for(int d = 0; d < 3; d++) {out[d] = -dr[d] / r3;}
    out[2] += (dr[2] > 0 ? 1.0 : -1.0) * 2.0 * M_PI / A;
    out[3] = 2.0 * M_PI / A * fabs(dr[2]) + 1.0 / r;
}


static void ewald_xy_fill_table(ewxy_table &t, int slot, double s)
{
    const size_t npts = (size_t)(EWXY_NXH + 1) * (EWXY_NXY + 1) * (ewxy_nz + 1);
    std::vector<int> cnt(NTask), off(NTask);
    for(int task = 0; task < NTask; task++)
    {
        size_t beg = npts * task / NTask, end = npts * (task + 1) / NTask;
        off[task] = (int)(4 * beg); cnt[task] = (int)(4 * (end - beg));
    }
    std::vector<float> mine(cnt[ThisTask] + 1);
    for(size_t p = (size_t)off[ThisTask] / 4, q = 0; q < (size_t)cnt[ThisTask] / 4; p++, q++)
    {
        const int k = (int)(p % (ewxy_nz + 1)), j = (int)((p / (ewxy_nz + 1)) % (EWXY_NXY + 1)), i = (int)(p / ((size_t)(ewxy_nz + 1) * (EWXY_NXY + 1)));
        double out[4]; ewald_xy_exact(i * ewxy_dx, -0.5 * ewxy_Ly + j * ewxy_dy, k * ewxy_dz, s, ewxy_Lx, ewxy_Ly, out);
        for(int d = 0; d < 4; d++) {mine[4 * q + d] = (float)out[d];}
    }
    t.v.resize(4 * npts);
    MPI_Allgatherv(mine.data(), cnt[ThisTask], MPI_FLOAT, t.v.data(), cnt.data(), off.data(), MPI_FLOAT, MPI_COMM_WORLD);
    t.slot = slot;
}


/* collective: make sure the tables bracket the current shear offset. call before any tree walk that uses ewald_xy_corr */
void ewald_xy_update(void)
{
#ifndef SELFGRAVITY_OFF
    if(ewxy_nz == 0)
    {
        if(All.ComovingIntegrationOn) {if(ThisTask == 0) {printf("GRAVITY_PERIODIC_XY does not support comoving integration\n");} endrun(1);}
        ewxy_Lx = boxSize_X; ewxy_Ly = boxSize_Y; ewxy_zmax = EWXY_ZMAX_IN_L * DMAX(ewxy_Lx, ewxy_Ly);
        ewxy_dx = ewxy_Lx / EWXY_NXY; ewxy_dy = ewxy_Ly / EWXY_NXY;
        ewxy_nz = (int)ceil(ewxy_zmax / DMIN(ewxy_dx, ewxy_dy)); ewxy_dz = ewxy_zmax / ewxy_nz;
        if(ThisTask == 0) {printf("GRAVITY_PERIODIC_XY: Ewald tables %d x %d x %d over |dz| <= %g\n", EWXY_NXH + 1, EWXY_NXY + 1, ewxy_nz + 1, ewxy_zmax);}
    }
#ifdef BOX_SHEARING
    const double ds = ewxy_Ly / EWXY_NS;
    double s = Shearing_Box_Pos_Offset; s -= ewxy_Ly * floor(s / ewxy_Ly);
    int lo = (int)floor(s / ds); if(lo >= EWXY_NS) {lo = EWXY_NS - 1;}
    const int want[2] = {lo, lo + 1}; /* slot EWXY_NS is s = L_y, the same lattice as slot 0 but kept separate for simple indexing */
    if(ewxy_tab[0].slot != want[0] || ewxy_tab[1].slot != want[1])
    {
        ewxy_table next[2];
        for(int w = 0; w < 2; w++)
        {
            for(int o = 0; o < 2; o++) {if(ewxy_tab[o].slot == want[w]) {next[w] = std::move(ewxy_tab[o]); ewxy_tab[o].slot = -1; break;}}
            if(next[w].slot != want[w]) {ewald_xy_fill_table(next[w], want[w], want[w] * ds);}
        }
        ewxy_tab[0] = std::move(next[0]); ewxy_tab[1] = std::move(next[1]);
    }
    ewxy_wslot = s / ds - lo;
#else
    if(ewxy_tab[0].slot < 0) {ewald_xy_fill_table(ewxy_tab[0], 0, 0.0);}
    ewxy_wslot = 0;
#endif
#endif
}


/* correction for a source of unit mass at nearest-image separation dr = source - target: acc[0..2] and potential *pot (G=1) */
void ewald_xy_corr(const double dr[3], double acc[3], double *pot)
{
    double out[4];
    const double az = fabs(dr[2]);
    if(az >= ewxy_zmax) {ewald_xy_far(dr, out);}
    else
    {
        const double sx = (dr[0] < 0) ? -1. : 1.; /* C(-dx,-dy,dz) = (-C_x, -C_y, C_z, pot) */
        double u = sx * dr[0] / ewxy_dx, v = (sx * dr[1] + 0.5 * ewxy_Ly) / ewxy_dy, w = az / ewxy_dz;
        int i = (int)u, j = (int)v, k = (int)w;
        i = (i >= EWXY_NXH) ? EWXY_NXH - 1 : i;
        j = (j < 0) ? 0 : ((j >= EWXY_NXY) ? EWXY_NXY - 1 : j);
        k = (k >= ewxy_nz) ? ewxy_nz - 1 : k;
        u -= i; v -= j; w -= k;
        u = DMIN(DMAX(u, 0.), 1.); v = DMIN(DMAX(v, 0.), 1.);
        const double fu[2] = {1 - u, u}, fv[2] = {1 - v, v}, fw[2] = {1 - w, w};
        const double ft[2] = {1 - ewxy_wslot, ewxy_wslot};
        for(int d = 0; d < 4; d++) {out[d] = 0;}
        for(int t = 0; t < 2; t++)
        {
            if(ft[t] == 0) {continue;}
            const float *tab = ewxy_tab[t].v.data();
            for(int a = 0; a < 2; a++) for(int b = 0; b < 2; b++) for(int c = 0; c < 2; c++)
            {
                const double f = ft[t] * fu[a] * fv[b] * fw[c];
                const float *e = tab + ewxy_index(i + a, j + b, k + c);
                for(int d = 0; d < 4; d++) {out[d] += f * e[d];}
            }
        }
        out[0] *= sx; out[1] *= sx;
        if(dr[2] < 0) {out[2] = -out[2];} /* the z-component is odd in dz, the rest even */
    }
    for(int d = 0; d < 3; d++) {acc[d] = out[d];}
    if(pot) {*pot = out[3];}
}

#endif
