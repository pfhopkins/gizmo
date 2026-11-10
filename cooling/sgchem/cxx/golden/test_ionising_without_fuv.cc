// test_ionising_without_fuv.cc -- ionising rates reach the network when G0 = 0 (no FUV background, M1 ionising light only).
// usage: test_ionising_without_fuv <datadir> <net: 5|17>
#include "chem_types.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main(int argc, char** argv)
{
    if (argc < 3) { std::printf("usage: %s datadir net\n", argv[0]); return 2; }
    int net = atoi(argv[2]);
    ChemBuildConfig cfg;
    cfg.net = net; cfg.iflag_atom = 2; cfg.iflag_mn = 4; cfg.iflag_ad = 6;
    cfg.iflag_3bh2a = 5; cfg.iflag_3bh2b = 4; cfg.iflag_h3pra = 1;
    cfg.wss_cie = 0; cfg.deff = 1.0; cfg.phi_pah = 0.5;
    cfg.dust_to_gas_ratio = 1.0; cfg.cosmic_ray_ion_rate = 3e-17; cfg.abhe = (double)0.1f;
    if (chem_cxx_tables_init(cfg, argv[1])) { std::printf("FATAL: tables (%s)\n", argv[1]); return 1; }
    const ChemTables& T = chem_cxx_tables();
    ChemCell c; chem_cxx_cell_defaults(c, T, net);
    c.G0 = c.G0_LW = c.G0_dust = 0.0;
    c.cosmic_ray_ion_rate = 3e-17; c.dust_to_gas_ratio = 1.0; c.iphoto = 0; c.tdust = 10.0;
    c.yn = 100.0; c.dl = 1e18; c.divv = 0.0;
    const double G = 1e-12, H = 1e-24;
    c.rt_phot_HI = G; c.rt_heat_HI = H; c.rt_phot_HeI = 0.5 * G; c.rt_phot_HeII = 0.1 * G; c.rt_phot_H2 = 0.3 * G;
    c.rt_heat_HeI = 0.5 * H; c.rt_heat_HeII = 0.1 * H; c.rt_heat_H2 = 0.3 * H;
    cxx_calc_photo(100.0, 0.1, 0.0, 0.0, c);
    int fail = 0;
    auto chk = [&](const char* n, double got, double want) {
        bool ok = std::fabs(got - want) <= 1e-12 * std::fabs(want);
        std::printf("%-22s got %.6e want %.6e %s\n", n, got, want, ok ? "PASS" : "FAIL"); if (!ok) fail++; };
    chk("phrates[37] HI", c.phrates[37], G);
    chk("phrates[41] HI heat", c.phrates[41], H);
    if (net != 5) { chk("phrates[38] HeI", c.phrates[38], 0.5 * G); chk("phrates[39] HeII", c.phrates[39], 0.1 * G); }
    return fail ? 1 : 0;
}
