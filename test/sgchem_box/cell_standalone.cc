// Standalone SGChem cell evolution for test_sgchem_box: the box cells' inputs (FIRE-3 metals, density, energy, the
// run's timesteps) through the C++ backend directly, written from the stella_dev input recipe (chem_interface.cc), not
// from the GIZMO glue, so the box run is checked against an independent assembly of the same inputs.
// Built by the test in the box build's src/ (same GIZMO_config.h), linked with its cooling/sgchem/cxx objects.
// stdin: UnitMass UnitLength UnitVelocity HubbleParam zeta DGR D/H MinGasTemp ndt dt[ndt] / 11 metal slots / then "id rho u" per cell
// stdout: id u T Ne x(H2) x(H+) x(CO) T_dust per cell
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include "chem_interface.h"
#include "sgchem_network.h"

extern "C" void sgchem_fatal_(int *code) { std::printf("FATAL %d\n", *code); std::exit(1); }

int main(int argc, char **argv)
{
    if(argc < 2) {std::printf("usage: cell_standalone <tables dir> < input\n"); return 2;}
    double UM, UL, UV, h, zeta, dgr, abD, Tmin; int ndt;
    if(std::scanf("%lf %lf %lf %lf %lf %lf %lf %lf %d", &UM, &UL, &UV, &h, &zeta, &dgr, &abD, &Tmin, &ndt) != 9) return 3;
    std::vector<double> dts(ndt); for(auto &d : dts) {if(std::scanf("%lf", &d) != 1) return 3;}
    double Z[11]; for(double &z : Z) {if(std::scanf("%lf", &z) != 1) return 3;}
    const double M = UM / h, L = UL / h, V = UV, rho_u = M / (L*L*L), t_u = L / V, e_u = M*V*V;

    ChemInitParams p = {};
    p.iflag_mn = 4; p.iflag_ad = 6; p.iflag_atom = 2; p.iflag_3bh2a = 5; p.iflag_3bh2b = 4; p.iflag_h3pra = 1;
    p.isrf_option = 1; p.tables_dir = argv[1]; p.h2_form_ex = 0.84; p.h2_form_kin = 0.12; p.deff = 1.0; p.phi_pah = 0.5;
    p.AV_conversion_factor = 5.348e-22; p.abundD = abD; p.cosmic_ray_ion_rate = zeta; p.dust_to_gas_ratio = dgr;
    chem_cxx_tables_init(&p);

    // element pools: gas-phase C/O/Si after Milky-Way depletion x DGR, per H with X_H = 1/(1+4 ABHE); DGR = DGR x Z/0.014
    const double XH = 1.0 / (1.0 + 4.0 * ABHE), g = std::max(dgr, 0.0);
    ChemInputs base = {};
    base.abundc  = (1 - std::min(0.50 * g, 0.9)) * (Z[2] / 12.0) / XH;
    base.abundo  = (1 - std::min(0.217 * g, 0.9)) * (Z[4] / 16.0) / XH;
    base.abundsi = (1 - std::min(0.82 * g, 0.9)) * (Z[7] / 28.0) / XH;
    base.abundN  = (Z[3] / 14.0) / XH;
    double zm[12] = {Z[1], Z[2], Z[6], Z[4], Z[10], Z[7], 1.0 - Z[0] - Z[1], Z[3], Z[5], Z[8], Z[9], 0.0};
    for(int k = 0; k < 12; k++) base.Zmass[k] = zm[k];
    base.abundM = 1.0e-7 * Z[0] / 0.014;   // Z below the 10 Zsun cap here
    base.abundD = abD; base.cosmic_ray_ion_rate = zeta;
    base.dust_to_gas_ratio = std::min(dgr * Z[0] / 0.014, 10.0);
    base.abhe = chem_cxx_abhe_default(); base.X_H_chem = chem_cxx_xh_default();

    long id; double rho_c, u;
    while(std::scanf("%ld %lf %lf", &id, &rho_c, &u) == 3) {
        double ab[TRAC_NUM] = {0}; ab[IHP] = 1.0e-4; double tdust = 10.0, T = 0;
        ChemInputs in = base;
        double rho = rho_c * rho_u, yn = rho / ((1.0 + 4.0 * ABHE) * PROTONMASS_CGS);
        for(double dt : dts) {
            in = base; in.id_current = (int) id; in.tdust = tdust;
            in.rpar[0] = yn; in.rpar[1] = 0; in.rpar[2] = 0;
            double energy = rho_c * u; energy *= e_u / std::pow(L, 3);   // same operation order as GIZMO
            for(int k = 0; k < TRAC_NUM; k++) in.abundances[k] = ab[k];
            double T0 = chem_cxx_calc_temp(ab[IH2], in.abhe, energy / (BOLTZMANN_CGS * (1.0 + ABHE - ab[IH2] + ab[IHP]) * yn));
            if(T0 > 1.0e5) in.dust_to_gas_ratio = 0;
            chem_cxx_begin_cell(&in, T0, ab[IH2], 0.0, ab[ICO]);
            sgchem_clamp_pools(in.abundances, &in);
            chem_cxx_evolve_cell(dt * t_u, 0, yn, 0, &energy, in.abundances);
            ChemOutputs out; chem_cxx_read_outputs(&out);
            for(int k = 0; k < TRAC_NUM; k++) ab[k] = in.abundances[k];
            T = chem_cxx_calc_temp(ab[IH2], in.abhe, energy / (BOLTZMANN_CGS * (1.0 + ABHE - ab[IH2] + ab[IHP]) * yn));
            if(T < Tmin && T > 0) {energy *= Tmin / T; T = Tmin;}
            tdust = out.tdust;
            u = energy * std::pow(L, 3) / e_u / (rho / rho_u);
        }
        std::printf("%ld %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", id, u, T, sgchem_electrons(ab, &in), ab[IH2], ab[IHP], ab[ICO], tdust);
    }
    return 0;
}
