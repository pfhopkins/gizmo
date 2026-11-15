// h2_shield_db96.cc -- net 5 one-zone: H2 self-shielding of the Lyman-Werner rate.
//  1. cxx_calc_H2_shielding_factor against Draine & Bertoldi (1996) eq. 37 with b = sqrt(2kT/m_H2), written out here.
//  2. TreeCol pixels (iphoto 6): the factor is the pixel average with the field's weights fac_uv (isotropic and beamed,
//     as sgchem_rt_field sets them for the moment-solver bins and sgchem_treecol_field for the walk).
//  3. Applied once: the LW rate phrates[0] carries no f_shield (it does not change with the H2 columns), and the RHS
//     photodissociation term of dxH2/dt equals -phrates[0] * f_shield * xH2.
// usage: h2_shield_db96 <datadir>
#include "chem_types.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <initializer_list>

static double db96(double NH2, double T)
{
    const double kB = 1.380649e-16, mH2 = 2.0 * 1.6726e-24;
    double b5 = std::sqrt(2.0 * kB * T / mH2) / 1e5, x = NH2 / 5e14;
    return 0.965 / ((1.0 + x / b5) * (1.0 + x / b5)) + 0.035 / std::sqrt(1.0 + x) * std::exp(-8.5e-4 * std::sqrt(1.0 + x));
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::printf("usage: %s datadir\n", argv[0]); return 2; }
    ChemBuildConfig cfg;
    cfg.net = 5; cfg.iflag_atom = 2; cfg.iflag_mn = 4; cfg.iflag_ad = 6;
    cfg.iflag_3bh2a = 5; cfg.iflag_3bh2b = 4; cfg.iflag_h3pra = 1;
    cfg.wss_cie = 1; cfg.deff = 1.0; cfg.phi_pah = 0.5;
    cfg.dust_to_gas_ratio = 1.0; cfg.cosmic_ray_ion_rate = 3e-17; cfg.abhe = (double)0.1f;
    if (chem_cxx_tables_init(cfg, argv[1])) { std::printf("FATAL: tables (%s)\n", argv[1]); return 1; }
    const ChemTables& T = chem_cxx_tables();
    int fail = 0;

    // ---- 1. local column (iphoto 1): f(N_H2 = abh2 * NH, T) ----
    {
        double maxrel = 0;
        for (double Tg : {10.0, 100.0, 1000.0, 8000.0})
            for (double N : {1e12, 1e14, 1e15, 1e16, 1e18, 1e20, 1e22}) {
                ChemCell c; chem_cxx_cell_defaults(c, T, 5); c.iphoto = 1;
                double f = cxx_calc_H2_shielding_factor(Tg, 0.5, 2.0 * N, c), e = db96(N, Tg);
                maxrel = std::fmax(maxrel, std::fabs(f / e - 1));
                std::printf("DB96 T=%g N_H2=%.0e f=%.6e db96=%.6e\n", Tg, N, f, e);
            }
        std::printf("LOCAL maxrel %.2e %s\n", maxrel, maxrel < 1e-4 ? "PASS" : "FAIL");
        if (!(maxrel < 1e-4)) fail++;
    }

    // ---- 2. pixel columns (iphoto 6), isotropic and beamed weights ----
    {
        double maxrel = 0, Tg = 50.0;
        for (double f_beam : {0.0, 0.8}) {
            ChemCell c; chem_cxx_cell_defaults(c, T, 5); c.iphoto = 6; c.npix = CHEM_NPIX;
            double e = 0;
            for (int k = 0; k < CHEM_NPIX; k++) {
                c.col_h2[k] = std::pow(10.0, 13.0 + 0.8 * k);
                c.fac_uv[k] = (1.0 - f_beam) / CHEM_NPIX + (k == 3 ? f_beam : 0.0);
                e += c.fac_uv[k] * db96(c.col_h2[k], Tg);
            }
            double f = cxx_calc_H2_shielding_factor(Tg, 0.3, 0.0, c);
            maxrel = std::fmax(maxrel, std::fabs(f / e - 1));
            std::printf("PIXELS f_beam=%.1f f=%.6e weighted db96=%.6e\n", f_beam, f, e);
        }
        std::printf("PIXELS maxrel %.2e %s\n", maxrel, maxrel < 1e-4 ? "PASS" : "FAIL");
        if (!(maxrel < 1e-4)) fail++;
    }

    // ---- 3. once: phrates[0] free of f_shield, RHS term = -phrates[0] f_shield xH2 ----
    {
        double Tg = 50.0, nH = 100.0, y[4], d1[4], d2[4], ph[2], fs[2];
        y[CIH2] = 0.3; y[CIHP] = 1e-4; y[CICO] = 0.0;
        y[3] = 1.5 * nH * (1.0 + 0.1 - y[CIH2]) * 1.3806599982553653e-16 * Tg;
        for (int s = 0; s < 2; s++) {  // s = 1: H2 columns 100 x larger
            ChemCell c; chem_cxx_cell_defaults(c, T, 5);
            c.abundc = 1.4e-4; c.abundo = 3.2e-4; c.abundsi = 1.5e-5; c.abundN = 7.6e-5;
            c.G0 = c.G0_dust = 1.7; c.cosmic_ray_ion_rate = 3e-17; c.dust_to_gas_ratio = 1.0; c.tdust = 10.0;
            c.iphoto = 6; c.npix = CHEM_NPIX; c.yn = nH; c.dl = 0.0; c.divv = 0.0;
            for (int k = 0; k < CHEM_NPIX; k++) { c.col_h2[k] = (s ? 1e18 : 1e16) * (1 + k); c.col_tot[k] = 1e19; c.col_co[k] = 0; c.fac_uv[k] = 1.0 / CHEM_NPIX; c.pix_att[k] = 1.0; }
            c.NH_cell = 0;
            c.G0_LW = 1.7; cxx_calc_photo(Tg, y[CIH2], 0.0, 0.0, c); chem_rhs_net5(0, y, d1, c);
            double Trhs, gg, ntot = nH * (1.0 + c.abhe - y[CIH2] + y[CIHP] + c.abundc + c.abundsi);  // the RHS temperature (chem_rhs_net5.cc:57-61)
            cxx_lookup_temp_gamma(y[CIH2], c.abhe, y[3] / (ntot * 1.3806599982553653e-16), &Trhs, &gg, T);
            ph[s] = c.phrates[0]; fs[s] = cxx_calc_H2_shielding_factor(Trhs, y[CIH2], 0.0, c);
            c.G0_LW = 3.4; cxx_calc_photo(Tg, y[CIH2], 0.0, 0.0, c); chem_rhs_net5(0, y, d2, c);
            double dH2 = d2[CIH2] - d1[CIH2], e = -(c.phrates[0] - ph[s]) * fs[s] * y[CIH2];
            double rel = std::fabs(dH2 / e - 1);
            std::printf("ONCE cols x%g: phrates0=%.6e f_shield=%.4e d(dxH2/dt)=%.6e expect %.6e rel %.2e %s\n",
                        s ? 100.0 : 1.0, ph[s], fs[s], dH2, e, rel, rel < 1e-8 ? "PASS" : "FAIL");
            if (!(rel < 1e-8)) fail++;
        }
        double rel = std::fabs(ph[1] / ph[0] - 1);
        std::printf("ONCE phrates0 independent of the H2 columns: rel %.2e (f_shield ratio %.3e) %s\n", rel, fs[1] / fs[0],
                    rel < 1e-12 ? "PASS" : "FAIL");
        if (!(rel < 1e-12)) fail++;
    }
    std::printf("RESULT %s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
