// test_cr_ion_heat.cc -- the CR ionisation heating the network applies over a solve is reported (c.cr_ion_heat_int).
// The T_CMB floor call inside every RHS evaluation must not reset it.
// usage: test_cr_ion_heat <datadir> [nH]
#include "chem_types.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static const double KB = 1.3807e-16;

int main(int argc, char** argv)
{
    if (argc < 2) { std::printf("usage: %s datadir [nH]\n", argv[0]); return 2; }
    double nH = argc > 2 ? atof(argv[2]) : 100.0, zeta = 1e-15;
    ChemBuildConfig cfg;
    cfg.net = 5; cfg.iflag_atom = 2; cfg.iflag_mn = 4; cfg.iflag_ad = 6;
    cfg.iflag_3bh2a = 5; cfg.iflag_3bh2b = 4; cfg.iflag_h3pra = 1;
    cfg.wss_cie = 0; cfg.deff = 1.0; cfg.phi_pah = 0.5;
    cfg.dust_to_gas_ratio = 1.0; cfg.cosmic_ray_ion_rate = zeta; cfg.abhe = (double)0.1f;
    if (chem_cxx_tables_init(cfg, argv[1])) { std::printf("FATAL: tables (%s)\n", argv[1]); return 1; }
    const ChemTables& T = chem_cxx_tables();
    ChemCell c; chem_cxx_cell_defaults(c, T, 5);
    c.abundc = 1.4e-4; c.abundo = 3.2e-4; c.abundsi = 1.5e-5; c.abundN = 7.6e-5; c.abundD = 2.6e-5;
    c.G0 = c.G0_LW = c.G0_dust = 1.7;
    c.cosmic_ray_ion_rate = zeta; c.dust_to_gas_ratio = 1.0; c.iphoto = 0; c.tdust = 10.0;
    double zsolar[12] = {0.7065, 2.07e-3, 5.91e-4, 5.49e-3, 1.10e-3, 6.83e-4,
                         0.2806, 8.36e-4, 1.41e-3, 4.09e-4, 6.44e-5, 1.10e-3};
    for (int k = 0; k < 12; k++) c.Zmass[k] = zsolar[k];
    c.yn = nH; c.dl = 1e18; c.divv = 0.0;
    std::vector<double> ab(c.trac_num, 0.0);
    ab[CIH2] = 0.4; ab[CIHP] = 1e-3; ab[CICO] = 0.0;
    double Tg = 50.0, energy = 1.5 * nH * (1.0 + c.abhe - ab[CIH2]) * KB * Tg, dt = 3e9;
    cxx_calc_photo(Tg, ab[CIH2], 0.0, 0.0, c);
    int rc = cxx_evolve_abundances(dt, c.dl, c.yn, c.divv, &energy, ab.data(), c);
    if (rc) { std::printf("solver rc=%d\n", rc); return 1; }
    double expect = 3.2e-11 * nH * zeta * dt;   // 3.2e-11 erg per ionisation x rate x n_H x dt, order of magnitude
    std::printf("cr_ion_heat_int %.4e erg/cm3, order-of-magnitude expectation %.4e (ratio %.2f)\n", c.cr_ion_heat_int, expect, c.cr_ion_heat_int / expect);
    bool ok = c.cr_ion_heat_int > 0.1 * expect && c.cr_ion_heat_int < 10.0 * expect;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
