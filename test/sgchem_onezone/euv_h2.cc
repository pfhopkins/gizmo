// euv_h2.cc -- net 5 one-zone: EUV photons on molecular gas.
//  1. RHS check: the photo terms add exactly -r*Gamma*xH2 to dxH2/dt and
//     nH*[Gamma(E-13.6)xHI + r*Gamma(E-15.2)xH2] eV to the heating.
//  2. Evolution: molecular cell (xH2=0.4999) in a constant EUV field must lose its
//     H2, ionise and settle near 1e4 K.
// usage: euv_h2 <datadir> <wss 0|1> <Gamma_HI [1/s]> <E_mean [eV]> <r=sigma_H2/sigma_HI> <nH> [G0, default 1.7; 0 = shielded]
#include "chem_types.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static const double KB = 1.3807e-16, EV = 1.60218e-12, MYR = 3.156e13;

static double G0_field = 1.7;

static void set_cell(ChemCell& c, const ChemTables& T, double nH)
{
    chem_cxx_cell_defaults(c, T, 5);
    c.abundc = 1.4e-4; c.abundo = 3.2e-4; c.abundsi = 1.5e-5; c.abundN = 7.6e-5; c.abundD = 2.6e-5;
    c.G0 = c.G0_LW = c.G0_dust = G0_field;
    c.cosmic_ray_ion_rate = 3e-17; c.dust_to_gas_ratio = 1.0; c.iphoto = 0; c.tdust = 10.0;
    double zsolar[12] = {0.7065, 2.07e-3, 5.91e-4, 5.49e-3, 1.10e-3, 6.83e-4,
                         0.2806, 8.36e-4, 1.41e-3, 4.09e-4, 6.44e-5, 1.10e-3};
    for (int k = 0; k < 12; k++) c.Zmass[k] = zsolar[k];
    c.yn = nH; c.dl = 1e18; c.divv = 0.0;
}

static void set_euv(ChemCell& c, double G, double E, double r, int on)
{
    c.rt_phot_HI = on ? G : 0.0;          c.rt_heat_HI = on ? G * (E - 13.6) * EV : 0.0;
    c.rt_phot_H2 = on ? r * G : 0.0;      c.rt_heat_H2 = on ? r * G * std::fmax(E - 15.2, 0.0) * EV : 0.0;
}

static double temp_of(const ChemTables& T, const ChemCell& c, const double* ab, double energy)
{
    double ntot = c.yn * (1.0 + c.abhe - ab[CIH2]), tt, gg;
    cxx_lookup_temp_gamma(ab[CIH2], c.abhe, energy / (ntot * KB), &tt, &gg, T);
    return tt;
}

int main(int argc, char** argv)
{
    if (argc < 7) { std::printf("usage: %s datadir wss Gamma E r nH\n", argv[0]); return 2; }
    const char* datadir = argv[1];
    int wss = atoi(argv[2]);
    double G = atof(argv[3]), E = atof(argv[4]), r = atof(argv[5]), nH = atof(argv[6]);
    if (argc > 7) G0_field = atof(argv[7]);

    ChemBuildConfig cfg;
    cfg.net = 5; cfg.iflag_atom = 2; cfg.iflag_mn = 4; cfg.iflag_ad = 6;
    cfg.iflag_3bh2a = 5; cfg.iflag_3bh2b = 4; cfg.iflag_h3pra = 1;
    cfg.wss_cie = wss; cfg.deff = 1.0; cfg.phi_pah = 0.5;
    cfg.dust_to_gas_ratio = 1.0; cfg.cosmic_ray_ion_rate = 3e-17; cfg.abhe = (double)0.1f;
    if (chem_cxx_tables_init(cfg, datadir)) { std::printf("FATAL: tables (%s)\n", datadir); return 1; }
    const ChemTables& T = chem_cxx_tables();
    int fail = 0;

    // ---- 1. RHS difference, EUV on minus off, at a partly molecular state ----
    {
        ChemCell c; set_cell(c, T, nH);
        double y[4], d_on[4], d_off[4];
        y[CIH2] = 0.3; y[CIHP] = 0.2; y[CICO] = 0.0;
        double Tg = 300.0, ntot = nH * (1.0 + c.abhe - y[CIH2]);
        y[3] = 1.5 * ntot * KB * Tg;
        double xHI = 1.0 - 2.0 * y[CIH2] - y[CIHP];
        set_euv(c, G, E, r, 1); cxx_calc_photo(Tg, y[CIH2], 0.0, 0.0, c); chem_rhs_net5(0, y, d_on, c);
        set_euv(c, G, E, r, 0); cxx_calc_photo(Tg, y[CIH2], 0.0, 0.0, c); chem_rhs_net5(0, y, d_off, c);
        double dH2 = d_on[CIH2] - d_off[CIH2], dHp = d_on[CIHP] - d_off[CIHP], dE = d_on[3] - d_off[3];
        double eH2 = -r * G * y[CIH2], eHp = G * xHI;
        double eE = nH * (G * (E - 13.6) * xHI + r * G * (E - 15.2) * y[CIH2]) * EV;
        double rel = std::fmax(std::fabs(dH2 / eH2 - 1), std::fmax(std::fabs(dHp / eHp - 1), std::fabs(dE / eE - 1)));
        std::printf("RHS  dxH2/dt %.6e (expect %.6e)  dxH+/dt %.6e (expect %.6e)  heat %.6e (expect %.6e) erg/cm3/s  maxrel %.2e  %s\n",
                    dH2, eH2, dHp, eHp, dE, eE, rel, rel < 1e-10 ? "PASS" : "FAIL");
        if (!(rel < 1e-10)) fail++;
    }

    // ---- 2. evolution of a molecular cell ----
    {
        ChemCell c; set_cell(c, T, nH); set_euv(c, G, E, r, 1);
        std::vector<double> ab(c.trac_num, 0.0);
        ab[CIH2] = 0.4999; ab[CIHP] = 2e-4; ab[CICO] = 0.0;
        double Tg = 20.0, energy = 1.5 * nH * (1.0 + c.abhe - ab[CIH2]) * KB * Tg;
        double t = 0, t_end = 1.0 * MYR, dt = 1e2;   // log-spaced substeps, as an operator-split step sequence
        double t_H2 = -1, t_HII = -1, T_last = 0;
        std::printf("EVOL nH=%g Gamma=%g E=%g r=%g wss=%d\n", nH, G, E, r, wss);
        int k = 0, rc = 0;
        while (t < t_end) {
            if (t + dt > t_end) dt = t_end - t;
            cxx_calc_photo(Tg, ab[CIH2], 0.0, ab[CICO], c);
            rc = cxx_evolve_abundances(dt, c.dl, c.yn, c.divv, &energy, ab.data(), c);
            if (rc) { std::printf("  solver rc=%d at t=%g\n", rc, t); fail++; break; }
            t += dt; Tg = temp_of(T, c, ab.data(), energy);
            if (t_H2 < 0 && ab[CIH2] < 1e-3) t_H2 = t;
            if (t_HII < 0 && ab[CIHP] > 0.99) t_HII = t;
            if (k++ % 4 == 0) std::printf("  t=%10.3e s  xH2=%10.3e  xH+=%10.6f  T=%10.1f K\n", t, ab[CIH2], ab[CIHP], Tg);
            dt = std::fmin(dt * 1.5, 0.01 * MYR);
            T_last = Tg;
        }
        {   // dominant cooling channels at the final state (cool_func rates[], lambda_chem[])
            double yy[4] = {ab[CIH2], ab[CIHP], ab[CICO], energy}, dd[4];
            int rr = chem_rhs_net5(0, yy, dd, c);
            std::printf("  final rhs rc=%d  edot=%.3e", rr, dd[3]);
            for (int q = 0; q < CHEM_NRATES; q++) if (std::fabs(c.lambda[q]) > 1e-3 * std::fabs(dd[3])) std::printf("  lam[%d]=%.2e", q, c.lambda[q]);
            for (int q = 0; q < 6; q++) if (std::fabs(c.lambda_chem[q]) > 1e-3 * std::fabs(dd[3])) std::printf("  chem[%d]=%.2e", q, c.lambda_chem[q]);
            std::printf("\n");
        }
        double t_ion = 1.0 / G;
        std::printf("RESULT xH2=%.3e xH+=%.6f T=%.0f K  t(xH2<1e-3)=%.3e s  t(xH+>0.99)=%.3e s  (1/Gamma=%.3e s)\n",
                    ab[CIH2], ab[CIHP], T_last, t_H2, t_HII, t_ion);
        bool ok = (ab[CIH2] < 1e-6) && (ab[CIHP] > 0.95) && (T_last > 5e3) && (T_last < 3e4);
        std::printf("EVOL %s (xH2<1e-6, xH+>0.95, 5e3<T<3e4 K)\n", ok ? "PASS" : "FAIL");
        if (!ok) fail++;
    }
    return fail ? 1 : 0;
}
