// dust_temperature.cc -- dust temperature solve (stella_dev golden/test_conservation.cc phase F, STL-164, STL-294, D-61),
// C++ only (gku has no Fortran backend: the DVODE comparisons of phase F are not applicable).
//   F1  production-like sequences in the dense shielded corner (nH 1e6, G0 0, Z 3), where gas falls below ~4 K inside
//       one solve: 5 starts, MinGasTemp 10 / 5 K applied after every solve as the cooling step does, solves of
//       100 yr .. 10 Myr up to 10 Myr in total (at most argv[2] solves per case, default 200). Bar: every solve rc 0,
//       no dust bisection, dust iterations <= 3e4 per solve.
//   F2  same dust root from tdust 3.8 K and 30 K (spread <= 1e-6) over T 3-30 K, x_HI 1e-3..0.5.
//   F3  Newton over Z 0.1-3, nH 1e4-1e6, T 5 K-1e4 K, G0 0 / 1.7 / 100: no bisection; cold start (min(T, 100 K) or
//       10 K) <= 30 and warm start (the previous root, gas 1 % warmer) <= 20 Newton iterations (Uli 2026-11-23).
//   F4  Newton slope against a central difference of the balance, relative <= 1e-6, at 0.5x, 1x, 2x the root (grid of F3).
// Usage: dust_temperature <chemistry tables dir> [max solves per F1 case]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

#include "chem_types.h"
#include "sgchem_consts.h"

extern "C" void sgchem_fatal_(int* code) { std::printf("  [sgchem_fatal %d]\n", *code); std::exit(1); }

#if CHEMISTRYNETWORK != 17 && CHEMISTRYNETWORK != 5
#error "dust_temperature covers networks 5 and 17"
#endif

extern thread_local long chem_dust_iters, chem_dust_bisections, chem_dust_bisect_evals;

static const int NET = CHEMISTRYNETWORK;
static const double ABHE_F = (double)0.1f;
static const double DL_SHIELD = 3.086e19;
static const double ZETA = 2e-16;
static ChemTables* g_cx = nullptr;
static double g_abundc = 0.0, g_abundsi = 0.0, g_abundD = 0.0, g_abundo = 0.0;

// stella_dev set_cell_commons(G0, Zs) + chem_bind_cell: inputs of a cell at G0 and Z (Zs solar units)
static void set_inputs(ChemCell& c, double G0, double Zs)
{
    chem_cxx_cell_defaults(c, *g_cx, NET);
    c.abhe = ABHE_F;
    c.iphoto = 6; c.irad_heat = 0; c.isrf_option = 1;
    for (int k = 0; k < CHEM_NPIX; k++) { c.pix_att[k] = 0.0; c.fac_uv[k] = 1.0 / CHEM_NPIX; }
    c.G0 = G0; c.G0_LW = G0; c.G0_dust = G0;
    c.cosmic_ray_ion_rate = ZETA; c.cr_energy_density = 0.0;
    c.dust_to_gas_ratio = Zs; c.redshift = 0.0;
    c.abundD = 2.6e-5; c.abundc = 1.4e-4 * Zs; c.abundo = 3.2e-4 * Zs; c.abundsi = 1.5e-5 * Zs;
    c.abundM = c.abundsi; c.abundN = 7.6e-5 * Zs;
    const double zsolar[12] = {0.2806, 2.07e-3, 5.91e-4, 5.49e-3, 1.10e-3, 6.83e-4,
                               0.7065, 8.36e-4, 1.41e-3, 4.09e-4, 6.44e-5, 1.10e-3};   // He C Mg O Fe Si H N Ne S Ca Zn
    for (int k = 0; k < 12; k++) c.Zmass[k] = (k == 0 || k == 6) ? zsolar[k] : zsolar[k] * Zs;
    g_abundc = c.abundc; g_abundsi = c.abundsi; g_abundD = c.abundD; g_abundo = c.abundo;
}

static double abe_of(const double* y)
{
    double cp = std::max(g_abundc - y[ICO], 0.0);
#if CHEMISTRYNETWORK == 17
    return y[IHP] + y[IHEP] + 2.0 * y[IHEPP] + cp + y[IDP] + g_abundsi;
#else
    return y[IHP] + cp + g_abundsi;
#endif
}
static double temp_of(double yn, double energy, const double* y)
{
    double yntot = (1.0 + ABHE_F - y[IH2] + abe_of(y)) * yn, t, g;
    cxx_lookup_temp_gamma(y[IH2], ABHE_F, energy / (yntot * CHEM_KBOLTZ), &t, &g, *g_cx);
    return t;
}
static double energy_at(double yn, double T0, const double* y)
{
    double yntot = (1.0 + ABHE_F - y[IH2] + abe_of(y)) * yn;
    double e = 1.5 * yntot * CHEM_KBOLTZ * T0;
    for (int k = 0; k < 60; k++) {
        double t = temp_of(yn, e, y);
        if (std::fabs(t - T0) < 1e-10 * T0) break;
        e *= T0 / t;
    }
    return e;
}
// the cooling step's final temperature (electrons = H+ only), for the MinGasTemp rescale at fixed composition
static double prod_temp(double nH, double en, const double* y)
{
    double ekn = en / (CHEM_KBOLTZ * (1.0 + ABHE_F - y[IH2] + y[IHP]) * nH), t, g;
    cxx_lookup_temp_gamma(y[IH2], ABHE_F, ekn, &t, &g, *g_cx);
    return t;
}

// stella_dev start states: 0 atomic, 1 molecular, 2 ionised, 3 mixed, 4 overdrawn (5% over every pool, then clamped)
static const int NSTART = 5;
static void initial_state(int start, double* y)
{
    const double C = g_abundc, O = g_abundo, He = ABHE_F, D = g_abundD;
    const double CO = std::min(C, O);
    for (int i = 0; i < TRAC_NUM; i++) y[i] = 0.0;
    (void)He; (void)D;
    switch (start) {
    case 0: y[IH2] = 1e-6; y[IHP] = 1e-4; break;
    case 1: y[IH2] = 0.49;  y[IHP] = 1e-6; y[ICO] = 0.5 * CO; break;
    case 2: y[IH2] = 0.0;   y[IHP] = 0.999; break;
    case 3: y[IH2] = 0.25;  y[IHP] = 0.3;  y[ICO] = 0.3 * CO; break;
    case 4: y[IH2] = 0.30;  y[IHP] = 0.42; y[ICO] = 1.05 * CO; break;
    }
#if CHEMISTRYNETWORK == 17
    switch (start) {
    case 0: y[IDP] = 1e-4 * D; break;
    case 1: y[IHD] = 0.5 * D; y[IDP] = 1e-6 * D; break;
    case 2: y[IHEP] = 0.5 * He; y[IHEPP] = 0.4 * He; y[IDP] = D; break;
    case 3: y[IHEP] = 0.5 * He; y[IHEPP] = 0.5 * He; y[IDP] = 0.5 * D; y[IHD] = 0.5 * D; break;
    case 4: y[IHEP] = 0.6 * He; y[IHEPP] = 0.45 * He; y[IDP] = 0.55 * D; y[IHD] = 0.5 * D; break;
    }
#endif
}
// pool clamp before a solve (sgchem_clamp_pools needs the GIZMO headers; here each over-filled pool is scaled to its total)
static void clamp_pools(double* y)
{
#if CHEMISTRYNETWORK == 17
    double h = 2.0 * y[IH2] + y[IHP] + y[IHD];
    if (h > 1.0) { y[IH2] /= h; y[IHP] /= h; y[IHD] /= h; }
    double he = y[IHEP] + y[IHEPP];
    if (he > ABHE_F) { y[IHEP] *= ABHE_F / he; y[IHEPP] *= ABHE_F / he; }
    double d = y[IDP] + y[IHD];
    if (d > g_abundD) { y[IDP] *= g_abundD / d; y[IHD] *= g_abundD / d; }
    if (y[IHP] > 0.99) { y[IH2] = 0.0; y[ICO] = 0.0; y[IHD] = 0.0; }
#else
    double h = 2.0 * y[IH2] + y[IHP];
    if (h > 1.0) { y[IH2] /= h; y[IHP] /= h; }
#endif
    y[ICO] = std::min(y[ICO], std::min(g_abundc, g_abundo));
}

// one solve as the test's run_cxx (no shielding length: dl 0, uniform columns from DL_SHIELD)
static int run_cxx(double G0, double Zs, double dt, double yn, double* e, double* y, double* td, long* iters, long* bis)
{
    ChemCell c; set_inputs(c, G0, Zs);
    double Tc = temp_of(yn, *e, y);
    const double NH = yn * DL_SHIELD;
    for (int k = 0; k < CHEM_NPIX; k++) {
        c.col_tot[k] = NH; c.col_h2[k] = std::max(y[IH2], 0.0) * NH; c.col_co[k] = std::max(y[ICO], 0.0) * NH;
    }
    c.yn = yn; c.dl = 0.0; c.divv = 0.0; c.tdust = *td;
#if CHEMISTRYNETWORK == 17
    cxx_calc_photo(Tc, y[IH2], y[IHD], y[ICO], c);
#else
    cxx_calc_photo(Tc, y[IH2], 0.0, y[ICO], c);
#endif
    const long i0 = chem_dust_iters, b0 = chem_dust_bisections;
    int rc = cxx_evolve_abundances(dt, 0.0, yn, 0.0, e, y, c);
    *iters = chem_dust_iters - i0; *bis = chem_dust_bisections - b0;
    *td = c.tdust;
    return rc;
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::printf("usage: %s <tables dir> [max solves per F1 case]\n", argv[0]); return 2; }
    const double cap = (argc > 2) ? std::atof(argv[2]) : 200.0;
    std::printf("=== DUST TEMPERATURE, NETWORK %d (D-61, STL-164, STL-294), C++ ===\n", NET);
    ChemBuildConfig cfg;
    cfg.net = NET;
    cfg.iflag_atom = 2; cfg.iflag_mn = 4; cfg.iflag_ad = 6;
    cfg.iflag_3bh2a = 5; cfg.iflag_3bh2b = 4; cfg.iflag_h3pra = 1;
    cfg.wss_cie = 1;
    cfg.deff = 1.0; cfg.phi_pah = 0.5;
    cfg.dust_to_gas_ratio = 0.1;   // stella_dev phase F: tables built at init_commons' set_cell_commons(1.7, 0.1)
    cfg.cosmic_ray_ion_rate = ZETA;
    cfg.abhe = ABHE_F;
    static ChemTables Cx;
    if (chem_tables_build(Cx, cfg, argv[1])) { std::printf("FATAL: chem_tables_build\n"); return 1; }
    g_cx = &Cx;
    int ok = 1;

    // F1
    std::printf("\nF1 -- dense shielded corner (nH 1e6, G0 0, Z 3): sequences with MinGasTemp after every solve, <= %.0f solves per case\n", cap);
    const double nH = 1e6;
    int nf1 = 0, nf1bad = 0; long f1max = 0, f1bis = 0;
    for (double Tf : {10.0, 5.0}) for (double dt : {3.156e9, 3.156e10, 3.156e11, 3.156e12, 3.156e14}) for (int st = 0; st < NSTART; st++) {
        ChemCell tmp; set_inputs(tmp, 0.0, 3.0);
        double y[TRAC_NUM + 1];
        initial_state(st, y); clamp_pools(y);
        const long nsteps = (long)std::min(cap, std::ceil(3.156e14 / dt));
        double en = energy_at(nH, Tf, y), td = 10.0, Tp = 0.0;
        int rc = 0; long nsol = 0, maxit = 0, nbis = 0;
        for (long n = 0; n < nsteps; n++) {
            long it = 0, bi = 0;
            rc = run_cxx(0.0, 3.0, dt, nH, &en, y, &td, &it, &bi);
            if (rc) break;
            maxit = std::max(maxit, it); nbis += bi;
            Tp = prod_temp(nH, en, y);
            if (Tp < Tf && Tp > 0) en *= Tf / Tp;
            nsol++;
        }
        const int bad = rc || (maxit > 30000) || nbis || !std::isfinite(Tp);
        nf1++; nf1bad += bad; f1max = std::max(f1max, maxit); f1bis += nbis;
        std::printf("  floor %4.0f K dt %9.3g start %d: %ld solves rc %d, T %.6g K, tdust %.6g K, x_H2 %.6g, dust iterations max %ld per solve, "
                    "bisections %ld%s\n", Tf, dt, st, nsol, rc, Tp, td, y[IH2], maxit, nbis, bad ? "  FAIL" : "");
        std::fflush(stdout);
    }
    std::printf("  GATE F1: %d of %d cases fail (rc != 0, bisection, > 3e4 dust iterations per solve); max %ld iterations per solve, %ld bisections\n",
                nf1bad, nf1, f1max, f1bis);
    ok &= (nf1bad == 0);

    // F2
    {
        ChemCell c; set_inputs(c, 0.0, 3.0);
        double y[TRAC_NUM + 1]; initial_state(1, y);
        const double NH = nH * DL_SHIELD;
        for (int k = 0; k < CHEM_NPIX; k++) { c.col_tot[k] = NH; c.col_h2[k] = y[IH2] * NH; c.col_co[k] = y[ICO] * NH; }
        c.yn = nH;
#if CHEMISTRYNETWORK == 17
        cxx_calc_photo(10.0, y[IH2], y[IHD], y[ICO], c);
#else
        cxx_calc_photo(10.0, y[IH2], 0.0, y[ICO], c);
#endif
        int nsame = 0; double worst = 0.0;
        for (double T : {3.0, 4.0, 5.0, 10.0, 30.0}) for (double aHI : {1e-3, 2e-2, 0.5}) {
            double v[2]; int k = 0;
            for (double t0 : {3.8, 30.0}) {
                double RH2 = 0.0, Rd = 0.0;
                c.tdust = t0; cxx_calc_dust_temp(nH, c.chi_isrf, T, aHI, &RH2, &Rd, 0, c); v[k++] = c.tdust;
            }
            const double sp = std::fabs(v[1] - v[0]) / std::min(v[0], v[1]);
            worst = std::max(worst, sp); nsame += (sp <= 1e-6);
            if (!(sp <= 1e-6))
                std::printf("  dust root T %.1f K x_HI %.0e: %.10g / %.10g (start 3.8 / 30 K), spread %.1e  FAIL\n", T, aHI, v[0], v[1], sp);
        }
        std::printf("\nF2 -- dust root: %d of 15 states give the same tdust from both starts (<= 1e-6), worst spread %.1e (chi_isrf %.4g): %s\n",
                    nsame, worst, c.chi_isrf, nsame == 15 ? "PASS" : "FAIL");
        ok &= (nsame == 15);
    }

    // F3 + F4
    {
        int ncall = 0, nbad = 0; long mxc = 0, mxw = 0, nbc = 0, mxev = 0;
        int nfd = 0, nfdbad = 0; double fdw = 0.0;
        for (double G0 : {0.0, 1.7, 100.0}) for (double Z : {0.1, 0.3, 1.0, 3.0}) {
            ChemCell c; set_inputs(c, G0, Z);
            c.chi_isrf = 1.0;
            for (double n : {1e4, 1e5, 1e6}) for (double T : {5.0, 10.0, 20.0, 50.0, 100.0, 1e3, 1e4}) for (double aHI : {1e-3, 0.5}) for (int s0 = 0; s0 < 2; s0++) {
                const double t0 = s0 ? 10.0 : std::min(T, 100.0);
                double RH2 = 0.0, Rd = 0.0;
                const long i0 = chem_dust_iters, e0 = chem_dust_bisect_evals, b0 = chem_dust_bisections;
                c.tdust = t0; cxx_calc_dust_temp(n, c.chi_isrf, T, aHI, &RH2, &Rd, 0, c);
                const long ev = chem_dust_bisect_evals - e0, nc_ = (chem_dust_iters - i0) - ev, bc = chem_dust_bisections - b0;
                const double tdc = c.tdust;
                const long wi0 = chem_dust_iters, we0 = chem_dust_bisect_evals, wb0 = chem_dust_bisections;
                c.tdust = tdc; cxx_calc_dust_temp(n, c.chi_isrf, 1.01 * T, aHI, &RH2, &Rd, 0, c);
                const long wev = chem_dust_bisect_evals - we0, wnc = (chem_dust_iters - wi0) - wev, wbc = chem_dust_bisections - wb0;
                const int bad = bc || wbc || nc_ > 30 || wnc > 20;
                ncall++; nbad += bad; mxc = std::max(mxc, nc_); nbc += bc + wbc;
                mxev = std::max(mxev, std::max(ev, wev)); mxw = std::max(mxw, wnc);
                if (bad && nbad <= 40)
                    std::printf("  dust Newton G0 %.1f Z %.1f nH %.0e T %.0f K x_HI %.0e start %.0f K: cold %ld, warm %ld Newton, bisection entries %ld, tdust %.10g  FAIL\n",
                                G0, Z, n, T, aHI, t0, nc_, wnc, bc + wbc, tdc);
                if (s0) continue;
                const ChemTables& TT = *c.tab;
                int itemp; double dtemp;
                if (T <= TT.tmin)      { itemp = 1;        dtemp = 0.0; }
                else if (T >= TT.tmax) { itemp = CHEM_NMD; dtemp = 0.0; }
                else { itemp = (int)(std::log10(T) / TT.dtlog) + 1; dtemp = T - TT.temptab[itemp - 1]; }
                double ch7  = (TT.chtab[itemp - 1][6] + dtemp * TT.dtchtab[itemp - 1][6]) * c.dust_to_gas_ratio;
                double cl14 = TT.cltab[itemp - 1][13] + dtemp * TT.dtcltab[itemp - 1][13];
                const double CMB_heat_z0 = 4.68e-31 * std::pow(2.726, 6.0);
                double isrf = std::max(0.0, ((c.isrf_option == 1) ? 5.8e-24 : 5.73e-24) * c.chi_isrf - CMB_heat_z0) * n * (c.G0_dust / (double)1.7f);
                isrf = (isrf + 4.68e-31 * chem_fpowi(2.726 * (1.0 + c.redshift), 6) * n) * c.dust_to_gas_ratio;
                for (double fac : {0.5, 1.0, 2.0}) {
                    const double td = std::min(1400.0, std::max(1.5, fac * tdc)), h = 1e-6 * td;
                    double F[3], dF = 0.0, Rd2, RH, dc, gd, hh;
                    const double tds[3] = {td, td + h, td - h};
                    for (int q = 0; q < 3; q++) {
                        double dFq;
                        cxx_dust_newton_terms(n, T, aHI, isrf, cl14, ch7, tds[q], c, &F[q], &dFq, &Rd2, &RH, &dc, &gd, &hh);
                        if (q == 0) dF = dFq;
                    }
                    const double fd = (F[1] - F[2]) / (2.0 * h);
                    const double rel = std::fabs(dF - fd) / std::max(std::fabs(dF), 1e-300);
                    nfd++; fdw = std::max(fdw, rel);
                    if (!(rel <= 1e-6)) {
                        nfdbad++;
                        if (nfdbad <= 20)
                            std::printf("  dust slope G0 %.1f Z %.1f nH %.0e T %.0f K x_HI %.0e tdust %.6g: Newton %.10e, central difference %.10e, rel %.1e  FAIL\n",
                                        G0, Z, n, T, aHI, td, dF, fd, rel);
                    }
                }
            }
        }
        std::printf("\nF3 -- dust Newton (STL-294): %d of %d states pass (no bisection, cold start <= 30 / warm start <= 20 Newton iterations); "
                    "max cold %ld, max warm %ld, bisection entries %ld, max bisection evals per call %ld: %s\n",
                    ncall - nbad, ncall, mxc, mxw, nbc, mxev, nbad == 0 ? "PASS" : "FAIL");
        std::printf("F4 -- dust slope (STL-294): %d of %d checks within 1e-6 of the central difference; worst %.1e: %s\n",
                    nfd - nfdbad, nfd, fdw, nfdbad == 0 ? "PASS" : "FAIL");
        ok &= (nbad == 0) && (nfdbad == 0);
    }
    chem_tables_free(Cx);
    std::printf("\n=== DUST TEMPERATURE NET %d: %s ===\n", NET, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
