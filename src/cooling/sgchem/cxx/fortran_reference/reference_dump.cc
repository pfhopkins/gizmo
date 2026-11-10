// reference_dump.cc -- chemistry init tables, cool_func rates and one-zone trajectories in one format.
// The Fortran-free GIZMO entry points (chem_gizmo_backend.cc) on CVODE; data/ is compared with reference.py.
// Usage: reference_dump <outdir>   (CHEMCXX_DATADIR = installed chemistry tables)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cooling/sgchem/cxx/chem_gizmo_backend.cc"   // GIZMO glue; its statics are visible here


extern "C" void sgchem_fatal_(int* code) { std::printf("sgchem_fatal %d\n", *code); std::exit(2); }

static const double KB = 1.380649e-16;   // BOLTZMANN_CGS as chem_interface.cc uses it
static std::string g_out;

// ---------------------------------------------------------------------------
// sgchem_init parameters (default settings; All.* values fixed here)
static ChemInitParams init_params()
{
    ChemInitParams p;
    p.iflag_mn = 4; p.iflag_ad = 6; p.iflag_atom = 2;
    p.iflag_3bh2a = 5; p.iflag_3bh2b = 4; p.iflag_h3pra = 1;
    p.iphoto = 6; p.irad_heat = 0; p.isrf_option = 1; p.no_chem = 0;
    p.h2_form_ex = 0.84; p.h2_form_kin = 0.12; p.deff = 1.0; p.phi_pah = 0.5;
    p.AV_conversion_factor = 5.348e-22; p.AV_ext = 0.0; p.dm_density = 0.0;
    p.pdv_term = 0.0; p.redshift = 0.0; p.abundD = 2.6e-5; p.G0 = 1.7; p.G0_LW = 1.7;
    p.cosmic_ray_ion_rate = 2e-16; p.cr_energy_density = 0.0; p.dust_to_gas_ratio = 1.0;
    p.tables_dir = nullptr;   // CHEMCXX_DATADIR
    return p;
}

// ---------------------------------------------------------------------------
// one cell's inputs, as chem_fill_inputs would hand them over
static void fill_cell(ChemInputs& in, double yn, double dl, double Zs, double abh2, double abco, double tdust)
{
    std::memset(&in, 0, sizeof(in));
    in.rpar[0] = yn; in.rpar[1] = dl; in.rpar[2] = 0.0;
    in.abundc = 1.4e-4 * Zs; in.abundo = 3.2e-4 * Zs; in.abundsi = 1.5e-5 * Zs;
    in.abundN = 7.6e-5 * Zs; in.abundM = SGCHEM_ABUNDM_SOLAR * Zs; in.abundD = 2.6e-5;
    const double zsolar[12] = {0.2806, 2.07e-3, 5.91e-4, 5.49e-3, 1.10e-3, 6.83e-4, 0.7065, 8.36e-4, 1.41e-3, 4.09e-4, 6.44e-5, 1.10e-3};
    for (int k = 0; k < 12; k++) in.Zmass[k] = (k == 0 || k == 6) ? zsolar[k] : zsolar[k] * Zs;
    in.G0 = 1.7; in.G0_LW = 1.7; in.G0_dust = 1.7;
    in.cosmic_ray_ion_rate = 2e-16;
    in.dust_to_gas_ratio = Zs; in.tdust = tdust;
    in.id_current = 1; in.irad_heat = 0;
    in.abhe = chem_cxx_abhe_default(); in.X_H_chem = chem_cxx_xh_default();
#if defined(RT_TREECOL_DUST_WEIGHTED) || defined(RT_TREECOL_H2_CO)
    const double NH = yn * 3.086e19;   // 10 pc column in every pixel
    for (int k = 0; k < NPIX; k++) {
        in.column_density_projection[k] = NH;
#if defined(GALSF_RESOLVEDISM_G0_VARIABLE) || defined(RT_TREECOL_STELLAR_PIXELS_PREATTENUATED)
        in.fac_uv[k] = 1.0 / NPIX;
#endif
#ifdef RT_TREECOL_H2_CO
        in.column_density_projection_h2[k] = abh2 * NH;
        in.column_density_projection_co[k] = abco * NH;
#endif
    }
#else
    (void)abh2; (void)abco;
#endif
}


// ---------------------------------------------------------------------------
static const ChemTables& init_backend()
{
    ChemInitParams p = init_params();
    chem_cxx_tables_init(&p);
    return chem_cxx_tables();
}

// ---------------------------------------------------------------------------
static void dump_tables(const ChemTables& A)
{
    FILE* fb = std::fopen((g_out + "/tables.bin").c_str(), "wb");
    FILE* fi = std::fopen((g_out + "/tables.idx").c_str(), "w");
    size_t off = 0;
    auto put = [&](const char* name, const double* a, size_t n) {
        std::fwrite(a, sizeof(double), n, fb);
        std::fprintf(fi, "%s %zu %zu\n", name, off, n);
        off += n;
    };
#define X(f, n) put(#f, A.f, (size_t)(n));
    X(temptab, CHEM_NMD) X(crtab, CHEM_NCRTAB) X(crphot, CHEM_NCRPHOT) X(phtab, CHEM_NPHTAB) X(cst, CHEM_NCONST)
    put("chtab", &A.chtab[0][0], (size_t)CHEM_NMD * CHEM_NCHTAB);
    put("dtchtab", &A.dtchtab[0][0], (size_t)CHEM_NMD * CHEM_NCHTAB);
    put("cltab", &A.cltab[0][0], (size_t)CHEM_NMD * CHEM_NCLTAB);
    put("dtcltab", &A.dtcltab[0][0], (size_t)CHEM_NMD * CHEM_NCLTAB);
    X(co_temptab, CHEM_NTCO) X(co_colntab, CHEM_NCDCO) X(co_L0, CHEM_NTCO) X(dTco_L0, CHEM_NTCO)
    X(co_lte, CHEM_NTCO * CHEM_NCDCO) X(co_n05, CHEM_NTCO * CHEM_NCDCO) X(co_alp, CHEM_NTCO * CHEM_NCDCO)
    X(dTco_lte, CHEM_NTCO * CHEM_NCDCO) X(dTco_n05, CHEM_NTCO * CHEM_NCDCO) X(dTco_alp, CHEM_NTCO * CHEM_NCDCO)
    X(co_vib_temptab, CHEM_NTCO_VIB) X(co_vib_colntab, CHEM_NCDCO_VIB)
    X(co_vib_LTE_final, CHEM_NTCO_VIB * CHEM_NCDCO_VIB) X(dTco_vib_LTE, CHEM_NTCO_VIB * CHEM_NCDCO_VIB)
    X(h2o_temptab, CHEM_NTH2O) X(h2o_colntab, CHEM_NCDH2O)
    X(h2o_L0_ortho, CHEM_NTH2O) X(dTh2o_L0_ortho, CHEM_NTH2O) X(h2o_L0_para, CHEM_NTH2O) X(dTh2o_L0_para, CHEM_NTH2O)
    X(h2o_LTE_ortho, CHEM_NTH2O * CHEM_NCDH2O) X(dTh2o_LTE_ortho, CHEM_NTH2O * CHEM_NCDH2O)
    X(h2o_n05_ortho, CHEM_NTH2O * CHEM_NCDH2O) X(dTh2o_n05_ortho, CHEM_NTH2O * CHEM_NCDH2O)
    X(h2o_alp_ortho, CHEM_NTH2O * CHEM_NCDH2O) X(dTh2o_alp_ortho, CHEM_NTH2O * CHEM_NCDH2O)
    X(h2o_LTE_para, CHEM_NTH2O * CHEM_NCDH2O) X(dTh2o_LTE_para, CHEM_NTH2O * CHEM_NCDH2O)
    X(h2o_n05_para, CHEM_NTH2O * CHEM_NCDH2O) X(dTh2o_n05_para, CHEM_NTH2O * CHEM_NCDH2O)
    X(h2o_alp_para, CHEM_NTH2O * CHEM_NCDH2O) X(dTh2o_alp_para, CHEM_NTH2O * CHEM_NCDH2O)
    X(h2o_vib_temptab, CHEM_NTH2O_VIB) X(h2o_vib_colntab, CHEM_NCDH2O_VIB)
    X(h2o_vib_LTE_final, CHEM_NTH2O_VIB * CHEM_NCDH2O_VIB) X(dTh2o_vib_LTE, CHEM_NTH2O_VIB * CHEM_NCDH2O_VIB)
    X(h2_opac_temp, CHEM_NH2OP) X(h2_opac_column, CHEM_NH2OP) X(h2_opac, CHEM_NH2OP * CHEM_NH2OP)
    X(CO_self_shielding, CHEM_IMAX_COSS) X(CO_H2_shielding, CHEM_IMAX_COH2)
    X(chi_tab_mathis, CHEM_NCHI) X(chi_tab_black, CHEM_NCHI)
    X(temp_table, CHEM_ISIZE) X(gamma_table, CHEM_ISIZE) X(eh2_table, CHEM_ISIZE2) X(hng, CHEM_NHNG)
#ifdef WSS_CIE_COOL
    X(wss_C, CHEM_NWSS) X(wss_N, CHEM_NWSS) X(wss_O, CHEM_NWSS) X(wss_Ne, CHEM_NWSS) X(wss_Mg, CHEM_NWSS)
    X(wss_Si, CHEM_NWSS) X(wss_S, CHEM_NWSS) X(wss_Ca, CHEM_NWSS) X(wss_Fe, CHEM_NWSS)
    X(HeI_tbl, CHEM_NHE_TBL) X(HeII_tbl, CHEM_NHE_TBL)
#endif
#undef X
    double sc[6] = {A.dtlog, A.tmin, A.tmax, 0, 0, 0};
    sc[3] = g_tmpl.abhe; sc[4] = chem_cxx_xh_default(); sc[5] = g_tmpl.f_rsc;
    put("scalars:dtlog,tmin,tmax,abhe,X_H_chem,f_rsc", sc, 6);
    std::fclose(fb); std::fclose(fi);
}

// ---------------------------------------------------------------------------
// cool_func over (abundance mix, nH, T, CMB flag): ylam, rates[28], tdust out
static void dump_cool_func(const ChemTables& T)
{
    const double base[4][16] = {
        // H2   O      OH     H2O    CO     CI     CII    SiI    SiII   e-     H+     HI     HD     HeI    HeII   HeIII
        {0.49,  3.0e-5,1e-9,  1e-9,  1e-6,  1e-6,  1e-5,  1e-7,  1e-6,  1e-5,  1e-6,  0.02,  1e-9,  0.1,   0.0,   0.0},
        {1e-6,  3.2e-5,1e-12, 1e-12, 1e-12, 1e-8,  1.4e-5,1e-9,  1.5e-6,1.4e-5,1e-4,  0.999, 1e-12, 0.1,   0.0,   0.0},
        {0.0,   1e-9,  0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   1.1,   1.0,   1e-9,  0.0,   0.0,   0.0,   0.1},
        {0.1,   3.2e-4,1e-8,  1e-7,  3.0e-4,1.4e-4,1e-9,  1.5e-5,1e-10, 1e-9,  1e-10, 0.8,   1e-6,  0.05,  0.03,  0.02},
    };
    const double yns[] = {1e-2, 1.0, 1e2, 1e4, 1e6};
    FILE* f = std::fopen((g_out + "/cool_func.bin").c_str(), "wb");
    long nrec = 0;
    for (int a = 0; a < 4; a++)
        for (double yn : yns)
            for (int it = 0; it <= 160; it++) {
                double temp = std::pow(10.0, 0.5 + 0.05 * it);   // 3 K .. 3e8 K
                ChemInputs in; fill_cell(in, yn, 3e18, 1.0, base[a][0], base[a][4], 23.7);
                chem_cxx_begin_cell(&in, temp, base[a][0], 0.0, base[a][4]);
                ChemCell c = tl_cell; c.tdust = 23.7;
                double ylam = 0, rates[CHEM_NRATES];
                for (int k = 0; k < CHEM_NRATES; k++) rates[k] = 0.0;
                int rc = cxx_cool_func(temp, yn, 3e18, 0.0, 1.7, base[a], &ylam, rates, 1, c);
                double rec[4 + 2 + CHEM_NRATES];
                rec[0] = a; rec[1] = yn; rec[2] = temp; rec[3] = rc;
                rec[4] = ylam; rec[5] = c.tdust;
                for (int k = 0; k < CHEM_NRATES; k++) rec[6 + k] = rates[k];
                std::fwrite(rec, sizeof(double), 6 + CHEM_NRATES, f);
                nrec++;
            }
    std::fclose(f);
    std::printf("cool_func: %ld records of %d doubles\n", nrec, 6 + CHEM_NRATES);
    (void)T;
}

// ---------------------------------------------------------------------------
// one-zone evolutions as GIZMO steps a cell: entry T, photo rates, solve, carry tdust
static void dump_onezone()
{
    const double nHs[] = {1.0, 1e2, 1e4};
    const double T0s[] = {50.0, 8e3, 1e6};
    const double Zs_[] = {0.1, 1.0};
    const int NZSTEP = 30;
    const double dl = 3.086e18;
    FILE* f = std::fopen((g_out + "/onezone.csv").c_str(), "w");
    std::fprintf(f, "zone,step,t_s,nH,T0,Z,rc,T,energy,tdust");
    for (int k = 0; k < TRAC_NUM; k++) std::fprintf(f, ",y%d", k);
    std::fprintf(f, "\n");
    int zone = 0;
    for (double yn : nHs) for (double T0 : T0s) for (double Zs : Zs_) {
        double y[TRAC_NUM];
        for (int k = 0; k < TRAC_NUM; k++) y[k] = 0.0;
        y[IH2] = (T0 < 100) ? 0.25 : 1e-6;
        y[IHP] = (T0 < 100) ? 1e-4 : (T0 < 1e5 ? 0.05 : 0.99);
#if CHEMISTRYNETWORK == 3 || CHEMISTRYNETWORK == 15 || CHEMISTRYNETWORK == 16
        y[IC] = 1.4e-4 * Zs;
#endif
        double abe0 = y[IHP];
        double energy = 1.5 * KB * yn * (1.0 + ABHE - y[IH2] + abe0) * T0;
        double tdust = 10.0, t = 0.0;
        for (int s = 0; s < NZSTEP; s++) {
            double dt = 3.156e9 * std::pow(10.0, 0.2 * s);   // 100 yr .. ~1e8 yr per step
            double abh2 = y[IH2], abe = y[IHP];
#if CHEMISTRYNETWORK == 17
            double abhd = y[IHD];
#else
            double abhd = 0.0;
#endif
            double abco = y[ICO];
            double ekn = energy / (KB * (1.0 + ABHE - abh2 + abe) * yn);
            double temp, divv = 0.0, ddl = dl, dyn = yn, dtt = dt;
            ChemInputs in; fill_cell(in, yn, dl, Zs, abh2, abco, tdust);
            int rc = 0;
            temp = chem_cxx_calc_temp(abh2, in.abhe, ekn);
            chem_cxx_begin_cell(&in, temp, abh2, abhd, abco);
            rc = cxx_evolve_abundances(dtt, ddl, dyn, divv, &energy, y, tl_cell);
            tdust = tl_cell.tdust;
            t += dt;
            abh2 = y[IH2]; abe = y[IHP];
            ekn = energy / (KB * (1.0 + ABHE - abh2 + abe) * yn);
            temp = chem_cxx_calc_temp(abh2, in.abhe, ekn);
            std::fprintf(f, "%d,%d,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%.17g,%.17g", zone, s, t, yn, T0, Zs, rc, temp, energy, tdust);
            for (int k = 0; k < TRAC_NUM; k++) std::fprintf(f, ",%.17g", y[k]);
            std::fprintf(f, "\n");
        }
        zone++;
    }
    std::fclose(f);
    std::printf("onezone: %d zones x %d steps\n", zone, NZSTEP);
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::printf("usage: reference_dump <outdir>\n"); return 1; }
    g_out = argv[1];
    std::printf("backend: C++, CHEMISTRYNETWORK=%d\n", CHEMISTRYNETWORK);
    const ChemTables& T = init_backend();
    dump_tables(T);
    dump_cool_func(T);
    dump_onezone();
    return 0;
}
