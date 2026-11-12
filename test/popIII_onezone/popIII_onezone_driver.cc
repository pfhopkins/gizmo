// One-zone free-fall collapse with the net-17 SGChem C++ backend (Omukai-style), for test_popIII_onezone.py.
// Per step: chemistry+cooling at fixed n over dt (cxx_evolve_abundances), then free-fall compression
// n *= exp(dt/t_ff), e *= (n'/n)^gamma. Evaluates the code's Pop III criterion (galaxy_sf/resolvedism_popIII.h)
// and DGR law (declarations/resolvedism_dgr_rr14.h) along the track.
// usage: popIII_onezone_driver DATADIR Z_solar DGRMODE(lin|rr14) OUT.csv [z=17] [n_end=1e12]
#include "chem_types.h"
#include "galaxy_sf/resolvedism_popIII.h"
#include "declarations/resolvedism_dgr_rr14.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

static const double MP = 1.6726e-24, KB = 1.3807e-16, G = 6.674e-8, MSUN = 1.989e33, YR = 3.156e7;

int main(int argc, char** argv)
{
    if (argc < 5) { std::fprintf(stderr, "usage: %s DATADIR Z DGRMODE OUT [z] [n_end]\n", argv[0]); return 1; }
    const char* datadir = argv[1];
    const double Z = atof(argv[2]);
    const int rr14 = (std::strcmp(argv[3], "rr14") == 0);
    const char* out = argv[4];
    const double zred = (argc > 5) ? atof(argv[5]) : 17.0;
    const double n_end = (argc > 6) ? atof(argv[6]) : 1e12;
    const int flag3b = getenv("FLAG3B") ? atoi(getenv("FLAG3B")) : 5;

    ChemBuildConfig cfg;
    cfg.net = 17; cfg.iflag_atom = 2; cfg.iflag_mn = 4; cfg.iflag_ad = 6;
    cfg.iflag_3bh2a = flag3b; cfg.iflag_3bh2b = 4; cfg.iflag_h3pra = 1;
    cfg.wss_cie = 1; cfg.deff = 1.0; cfg.phi_pah = 0.5;
    cfg.dust_to_gas_ratio = 1.0; cfg.cosmic_ray_ion_rate = 3e-17;
    cfg.abhe = (double)(float)(getenv("ABHE") ? atof(getenv("ABHE")) : 0.1);   // He/H by number, as compiled (float literal, coolinmo)
    if (chem_cxx_tables_init(cfg, datadir)) { std::printf("table init failed\n"); return 1; }
    const ChemTables& T = chem_cxx_tables();

    ChemCell c;
    chem_cxx_cell_defaults(c, T, 17);
    // gas-phase C/O/Si as chem_interface.cc (no dust model): (1 - depl) x Asplund09 x Z
    c.abundc  = (1 - 0.50)  * 2.69e-4 * Z;
    c.abundo  = (1 - 0.217) * 4.90e-4 * Z;
    c.abundsi = (1 - 0.82)  * 3.24e-5 * Z;
    c.abundN  = 6.76e-5 * Z;
    c.abundM  = c.abundsi;
    c.abundD  = 2.6e-5;
    c.G0 = 0; c.G0_LW = 0; c.G0_dust = 0;
    c.cosmic_ray_ion_rate = 1e-21;        // CR_SCALE_WITH_G0 floor at G0 = 0
    c.dust_to_gas_ratio = Z * (rr14 ? resolvedism_dgr_rr14_over_linear(Z) : 1.0);   // solar units, DGRnormalized = 1
    c.redshift = zred;
    c.iphoto = 0;
    const double zsolar[12] = {1.0 / (1.0 + 4.0 * cfg.abhe), 2.07e-3, 5.91e-4, 5.49e-3, 1.10e-3, 6.83e-4,
                               1.0 - 1.0 / (1.0 + 4.0 * cfg.abhe), 8.36e-4, 1.41e-3, 4.09e-4, 6.44e-5, 1.10e-3};
    for (int k = 0; k < 12; k++) c.Zmass[k] = (k == 0 || k == 6) ? zsolar[k] : zsolar[k] * Z;
    const double Tcmb = 2.726 * (1 + zred);
    c.tdust = Tcmb;

    std::vector<double> ab(c.trac_num, 0.0);
    ab[CIH2] = 2e-6; ab[CIHP] = 2e-4; ab[CICO] = 0;
    ab[CIHEP] = 0; ab[CIHEPP] = 0; ab[CIDP] = 2e-4 * c.abundD; ab[CIHD] = 0;

    const double XH = 1.0 / (1.0 + 4.0 * c.abhe);
    double n = 1.0, Ttemp = 1000.0;
    auto ntot_of = [&](double nn) { return nn * (1.0 + c.abhe - ab[CIH2] + ab[CIHP] + ab[CIHEP] + 2 * ab[CIHEPP] + c.abundc); };
    double e = 1.5 * ntot_of(n) * KB * Ttemp;   // refined below by the table
    for (int it = 0; it < 30; it++) {
        double tt, gg; cxx_lookup_temp_gamma(ab[CIH2], c.abhe, e / (ntot_of(n) * KB), &tt, &gg, T);
        e *= Ttemp / tt;
    }

    FILE* f = std::fopen(out, "w");
    std::fprintf(f, "nH,T,gamma,MJ_Msun,xH2,xHD,xe,tdust,tff_yr,"
                    "L_H2,L_HD,L_CII,L_CI,L_OI,L_SiII,L_dust,L_CO,L_H2CIE,L_atomic,L_compton,G_compress,"
                    "R_H2dust,R_H23b,rc\n");
    double t = 0; int step = 0, rc = 0;
    double MJ32 = -1, MJ2048 = -1, n_popIII = -1, T_popIII = 0, Tmin_loiter = 1e30;
    const popIII_sf_params pp = {1e-4 * 0.014, 1e6, 1e-3, 1000.};   // code defaults (PopIII_* parameters)
    const int pristine = popIII_gas_is_pristine(Z * 0.014, pp.Zcrit_abs);
    while (n < n_end && step < 20000) {
        double rho = n * MP / XH;
        double tff = std::sqrt(3 * M_PI / (32 * G * rho));
        double dt = 0.01 * tff;
        c.yn = n; c.dl = std::cbrt(MSUN / rho); c.divv = -1.0 / tff;
        cxx_calc_photo(Ttemp, ab[CIH2], ab[CIHD], ab[CICO], c);
        rc = cxx_evolve_abundances(dt, c.dl, n, c.divv, &e, ab.data(), c);
        if (rc) { std::fprintf(stderr, "Z=%g: evolve rc=%d at n=%g T=%g\n", Z, rc, n, Ttemp); break; }
        double gg;
        cxx_lookup_temp_gamma(ab[CIH2], c.abhe, e / (ntot_of(n) * KB), &Ttemp, &gg, T);
        // diagnostics at the post-chemistry state
        double cs2 = gg * (gg - 1) * e / rho;                  // adiabatic sound speed^2 (effective_soundspeed)
        double MJ = 2.9155697 * std::pow(cs2, 1.5) / (std::pow(G, 1.5) * std::sqrt(rho)) / MSUN;
        if (MJ32 < 0 && MJ < 32) MJ32 = n;
        if (MJ2048 < 0 && MJ < 2048) MJ2048 = n;
        if (pristine && n_popIII < 0 && popIII_sf_allowed(n, c.divv, ab[CIH2], MJ, 0, &pp)) {n_popIII = n; T_popIII = Ttemp;}
        if (n > 1e2 && n < 1e4) Tmin_loiter = std::min(Tmin_loiter, Ttemp);
        double abHI = std::max(0.0, 1 - 2 * ab[CIH2] - ab[CIHP] - ab[CIHD]);
        double xe = ab[CIHP] + ab[CIHEP] + 2 * ab[CIHEPP] + c.abundc + ab[CIDP] + c.abundsi;
        ChemCell cd = c; double RH2 = 0, Rdust = 0;
        cxx_calc_dust_temp(n, c.chi_isrf, Ttemp, abHI, &RH2, &Rdust, 0, cd);
        double R_dust = Rdust * abHI * n;                      // d xH2/dt from grains [s^-1]
        double k108 = (flag3b == 6) ? 6e-32 / std::pow(Ttemp, 0.25) + 2e-31 / std::sqrt(Ttemp)
                                    : 7.7e-31 / std::pow(Ttemp, 0.464);
        double abheI = std::max(0.0, c.abhe - ab[CIHEP] - ab[CIHEPP]);
        double R_3b = (k108 * abHI * abHI * abHI + k108 / 8 * abHI * abHI * ab[CIH2]
                       + 6.9e-32 * std::pow(Ttemp, -0.4) * abHI * abHI * abheI) * n * n;
        double Gcomp = gg * e / tff;                           // free-fall compressional heating [erg/cm^3/s]
        const double* L = c.lambda;
        std::fprintf(f, "%.6e,%.6e,%.5f,%.6e,%.6e,%.6e,%.6e,%.4e,%.6e,"
                        "%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%d\n",
                     n, Ttemp, gg, MJ, ab[CIH2], ab[CIHD], xe, cd.tdust, tff / YR,
                     L[1], L[18], L[15], L[13], L[12], L[16], L[0], L[6] + L[7] + L[8], L[19],
                     L[2] + L[24] + L[25] + L[26] + L[27], L[17], Gcomp, R_dust, R_3b, rc);
        // compression
        double nn = n * std::exp(dt / tff);
        e *= std::pow(nn / n, gg);
        n = nn; t += dt; step++;
        cxx_lookup_temp_gamma(ab[CIH2], c.abhe, e / (ntot_of(n) * KB), &Ttemp, &gg, T);
    }
    std::fclose(f);
    std::printf("RESULT Z=%g dgr=%s DGR=%.3e pristine=%d n_popIII=%.4e T_popIII=%.1f n_MJ32=%.4e n_MJ2048=%.4e Tmin_1e2_1e4=%.1f n_last=%.3e rc=%d\n",
                Z, rr14 ? "rr14" : "lin", c.dust_to_gas_ratio, pristine, n_popIII, T_popIII, MJ32, MJ2048, Tmin_loiter, n, rc);
    chem_cxx_tables_shutdown();
    return 0;
}
