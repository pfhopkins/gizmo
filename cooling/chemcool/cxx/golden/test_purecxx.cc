// test_purecxx.cc -- Gate-3 proof that the CHEMCOOL_CXX path is Fortran-free.
//
// This driver links ONLY the C++ translation units of the port plus SUNDIALS.
// No .F object, no libgfortran, no /coolr/ COMMON.  If any Fortran symbol were
// still reachable from the C++ path the link would fail; build_purecxx.sh then
// runs `nm` over the binary and asserts that no libgfortran / Fortran-mangled
// symbol is present, so the claim is checked rather than asserted.
//
// It also does real work: a one-zone relaxation and (optionally) an OpenMP
// loop over independent cells, to demonstrate the R2 thread-safety claim.

#include "chem_types.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#define PROTONMASS_D 1.6726e-24
#define KBOLTZ_D     1.3807e-16
#define SEC_PER_MYR  3.156e13

#ifndef PURE_NET
#define PURE_NET 5
#endif

static void set_cell(ChemCell& c, const ChemTables& T, int net,
                     double Zs, double G0, double zeta, double dl_shield)
{
    chem_cxx_cell_defaults(c, T, net);
    c.abundc  = 1.4e-4 * Zs;
    c.abundo  = 3.2e-4 * Zs;
    c.abundsi = 1.5e-5 * Zs;
    c.abundN  = 7.6e-5 * Zs;
    c.abundD  = 2.6e-5;
    c.G0 = G0; c.G0_LW = G0; c.G0_dust = G0;
    c.cosmic_ray_ion_rate = zeta;
    c.dust_to_gas_ratio = Zs;
    c.iphoto = (dl_shield > 0) ? 6 : 0;
    c.tdust = 10.0;
    double zsolar[12] = {0.7065, 2.07e-3, 5.91e-4, 5.49e-3, 1.10e-3, 6.83e-4,
                         0.2806, 8.36e-4, 1.41e-3, 4.09e-4, 6.44e-5, 1.10e-3};
    for (int k = 0; k < 12; k++)
        c.Zmass[k] = (k == 0 || k == 6) ? zsolar[k] : zsolar[k] * Zs;
}

// one cell, nsteps of dt, returns the final temperature
static double relax(const ChemTables& T, int net, double T0, double nH,
                    double Zs, double G0, double zeta, double t_end_Myr,
                    int nsteps, double dl_shield, int* rc_out)
{
    ChemCell c;
    set_cell(c, T, net, Zs, G0, zeta, dl_shield);
    c.yn = nH; c.dl = (dl_shield > 0) ? dl_shield : 1e18; c.divv = 0.0;

    int trac = c.trac_num;
    std::vector<double> ab(trac, 0.0);
    ab[CIH2] = 1e-6;
    ab[CIHP] = 1e-4;
    ab[CICO] = 0.0;
    if (net == 17) { ab[CIHEP] = 0.0; ab[CIHEPP] = 0.0;
                     ab[CIDP] = 1e-4 * c.abundD; ab[CIHD] = 0.0; }

    double abh2 = ab[CIH2];
    double ntot = nH * (1.0 + c.abhe - abh2);
    double energy = 1.5 * ntot * KBOLTZ_D * T0;

    double dt = t_end_Myr * SEC_PER_MYR / nsteps;
    int rc = 0;
    for (int s = 0; s < nsteps; s++) {
        double abhd = (net == 17) ? ab[CIHD] : 0.0;
        double NHcol = c.yn * c.dl;
        for (int k = 0; k < c.npix; k++) {
            c.col_tot[k] = NHcol;
            c.col_h2[k]  = ab[CIH2] * NHcol;
            c.col_co[k]  = ab[CICO] * NHcol;
        }
        cxx_calc_photo(T0, ab[CIH2], abhd, ab[CICO], c);
        rc = cxx_evolve_abundances(dt, c.dl, c.yn, c.divv, &energy, ab.data(), c);
        if (rc) break;
        ntot = nH * (1.0 + c.abhe - ab[CIH2]);
        double ekn = energy / (ntot * KBOLTZ_D);
        double tt, gg;
        cxx_lookup_temp_gamma(ab[CIH2], c.abhe, ekn, &tt, &gg, T);
        T0 = tt;
    }
    if (rc_out) *rc_out = rc;
    ntot = nH * (1.0 + c.abhe - ab[CIH2]);
    double ekn = energy / (ntot * KBOLTZ_D);
    double tt, gg;
    cxx_lookup_temp_gamma(ab[CIH2], c.abhe, ekn, &tt, &gg, T);
    return tt;
}

int main(int argc, char** argv)
{
    const char* datadir = (argc > 1) ? argv[1] : "gizmo/cooling/chemcool";
    int net = PURE_NET;

    ChemBuildConfig cfg;
    cfg.net = net; cfg.iflag_atom = 2;
    cfg.iflag_mn = 4; cfg.iflag_ad = 6;
    cfg.iflag_3bh2a = 5; cfg.iflag_3bh2b = 4; cfg.iflag_h3pra = 1;
#ifdef WSS_CIE_COOL
    cfg.wss_cie = 1;
#else
    cfg.wss_cie = 1;      // Stella production always has WSS_CIE_COOL on
#endif
    cfg.deff = 1.0; cfg.phi_pah = 0.5;
    cfg.dust_to_gas_ratio = 0.1; cfg.cosmic_ray_ion_rate = 3e-17;
    cfg.abhe = (double)0.1f;   // as coolinmo (ABHE default-REAL)

    if (chem_cxx_tables_init(cfg, datadir)) {
        std::printf("FATAL: table init failed (datadir=%s)\n", datadir);
        return 1;
    }
    const ChemTables& T = chem_cxx_tables();
    std::printf("PURE-C++ backend: tables built (net %d, nmd=%d, tmin=%g tmax=%g)\n",
                net, CHEM_NMD, T.tmin, T.tmax);

    // ---- serial reference sweep ----
    struct Case { double T0, nH, Zs, G0, zeta, tend; };
    Case cases[] = {
        {1e4,  1.0,   0.1, 1.7, 3e-17, 30.0},
        {1e3,  1e2,   0.1, 1.7, 3e-17, 30.0},
        {1e2,  1e4,   0.1, 1.7, 3e-17, 10.0},
        {3e5,  1e-2,  0.1, 1.7, 3e-17, 30.0},
        {50.0, 1e6,   0.1, 0.1, 3e-17, 1.0},
        {1e6,  1.0,   0.1, 1.7, 3e-17, 30.0},
        {5e3,  1e3,   0.1, 10.0,3e-17, 20.0},
        {2e4,  1e1,   0.1, 1.7, 3e-16, 30.0},
    };
    const int ncase = (int)(sizeof(cases)/sizeof(cases[0]));
    std::vector<double> serial(ncase);
    int bad = 0;
    for (int i = 0; i < ncase; i++) {
        int rc = 0;
        serial[i] = relax(T, net, cases[i].T0, cases[i].nH, cases[i].Zs,
                          cases[i].G0, cases[i].zeta, cases[i].tend, 200,
                          3e18, &rc);
        std::printf("  case %d: T0=%9.3g nH=%8.3g -> T=%12.6g   rc=%d\n",
                    i, cases[i].T0, cases[i].nH, serial[i], rc);
        if (rc || !std::isfinite(serial[i])) bad++;
    }

#ifdef _OPENMP
    // ---- the same cases run concurrently: results must be BIT-IDENTICAL ----
    int nth = omp_get_max_threads();
    std::vector<double> par(ncase, 0.0);
    std::vector<int> rcs(ncase, 0);
    const int REP = 8;
    std::vector<double> pr((size_t)ncase * REP, 0.0);
#pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < ncase * REP; j++) {
        int i = j % ncase;
        int rc = 0;
        pr[j] = relax(T, net, cases[i].T0, cases[i].nH, cases[i].Zs,
                      cases[i].G0, cases[i].zeta, cases[i].tend, 200,
                      3e18, &rc);
        rcs[i] = rc;
    }
    int mismatch = 0;
    for (int j = 0; j < ncase * REP; j++)
        if (pr[j] != serial[j % ncase]) mismatch++;
    std::printf("OpenMP: %d threads, %d concurrent solves, "
                "bit-identical to serial: %s (%d mismatches)\n",
                nth, ncase * REP, mismatch ? "NO" : "YES", mismatch);
    if (mismatch) bad++;
#else
    std::printf("OpenMP: not enabled in this build\n");
#endif

    chem_cxx_tables_shutdown();
    std::printf("PURE-C++ RESULT: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
