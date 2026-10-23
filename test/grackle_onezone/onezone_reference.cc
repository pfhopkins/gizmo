// Reference solution for the grackle_onezone test: a standalone one-zone Grackle 3 integration set up exactly like GIZMO's
// InitGrackle/CallGrackle (cooling/grackle.cc, COOL_GRACKLE_APIVERSION=3): same units (GIZMO defaults, h=1), same
// chemistry_data choices, same initial abundances. A uniform static GIZMO box is N identical one-zone problems.
// Compiled by test_grackle_onezone.py against the same grackle.pc (pkg-config) that the GIZMO build uses.
//
// usage: onezone_reference <datafile> <chem 0-3> <metals 0/1> <rho_code> <u_code> <Z> <t_end_code> <n_out> <substeps_per_out>
//                          [L_shield_code [self_shielding_method]]
//   L_shield: H2 self-shielding length; GIZMO uses its Sobolev estimate, which for a uniform box (grad rho=0) is
//             KernelRadius/NumNgb^(1/3)
//   self_shielding_method: 3 if the table has /UVBRates/CrossSections (as GIZMO decides in InitGrackle), else 0
// prints one line per output: t u Ne(per H) HI HII HM HeI HeII HeIII H2I H2II DI DII HDI T   (mass fractions; code units)
#include <cstdio>
#include <cstdlib>
#include <grackle.h>

int main(int argc, char **argv)
{
    if(argc < 10 || argc > 12) {fprintf(stderr, "usage: %s datafile chem metals rho u Z t_end n_out substeps [L_shield]\n", argv[0]); return 2;}
    const char *datafile = argv[1]; int chem = atoi(argv[2]), metals = atoi(argv[3]);
    double rho = atof(argv[4]), u0 = atof(argv[5]), Z = atof(argv[6]), t_end = atof(argv[7]);
    int n_out = atoi(argv[8]), nsub = atoi(argv[9]); double L_shield = (argc >= 11) ? atof(argv[10]) : 0; int ssm = (argc >= 12) ? atoi(argv[11]) : 0;
    const double X = 0.76, GAMMA = 5./3.; // HYDROGEN_MASSFRAC, GAMMA_DEFAULT
    if(gr_check_consistency() != GR_SUCCESS) {fprintf(stderr, "gr_check_consistency failed\n"); return 1;}
    grackle_verbose = 0;

    code_units units; // GIZMO defaults: kpc, 1e10 Msun, km/s, HubbleParam=1, non-comoving
    double UL = 3.085678e21, UM = 1.989e43, UV = 1e5;
    units.comoving_coordinates = 0; units.length_units = UL; units.velocity_units = UV; units.time_units = UL/UV;
    units.density_units = UM/(UL*UL*UL); units.a_units = 1.0; units.a_value = 1.0;

    chemistry_data chemd; chemistry_data_storage rates;
    if(local_initialize_chemistry_parameters(&chemd) != GR_SUCCESS) return 1;
    chemistry_data *gc = &chemd;
    gc->use_grackle = 1; gc->with_radiative_cooling = 1; gc->grackle_data_file = datafile; gc->Gamma = GAMMA;
    gc->primordial_chemistry = chem; gc->UVbackground = 1; gc->cmb_temperature_floor = 1; gc->Compton_xray_heating = 1;
    gc->three_body_rate = 0; gc->cie_cooling = 0; gc->h2_optical_depth_approximation = 0; gc->LWbackground_intensity = 0; gc->LWbackground_sawtooth_suppression = 0;
    if(chem > 0) {gc->HydrogenFractionByMass = X; gc->self_shielding_method = ssm;}
    if(chem > 1) {gc->H2_self_shielding = 2;}
    if(metals) {gc->metal_cooling = 1; if(chem > 0) {gc->dust_chemistry = 1;} gc->photoelectric_heating = 1; gc->photoelectric_heating_rate = 8.5e-26;}
    if(local_initialize_chemistry_data(gc, &rates, &units) != GR_SUCCESS) {fprintf(stderr, "init failed\n"); return 1;}

    int dim[3] = {1,1,1}, st[3] = {0,0,0}, en[3] = {0,0,0};
    grackle_field_data f; gr_initialize_field_data(&f);
    f.grid_rank = 3; f.grid_dimension = dim; f.grid_start = st; f.grid_end = en; f.grid_dx = 0;
    double tiny = 1e-20;
    gr_float d = rho, u = u0, vx = 0, vy = 0, vz = 0, md = metals ? Z*rho : 0.02*rho;
    gr_float HI = X*d, HII = tiny*d, HM = tiny*d, HeI = (1-X)*d, HeII = tiny*d, HeIII = tiny*d;
    gr_float H2I = tiny*d, H2II = tiny*d, DI = gc->DeuteriumToHydrogenRatio*X*d, DII = tiny*d, HDI = tiny*d, L = L_shield;
    gr_float e = (HII + HeII/4. + HeIII/2.); // charge balance (GIZMO init.cc), e_density = n_e m_H
    double ne_tab = 1.0;                      // tabular: GIZMO keeps its own Ne (not evolved by Grackle)
    f.density = &d; f.internal_energy = &u; f.x_velocity = &vx; f.y_velocity = &vy; f.z_velocity = &vz; f.metal_density = &md;
    if(chem > 0) {f.HI_density = &HI; f.HII_density = &HII; f.HM_density = &HM; f.HeI_density = &HeI; f.HeII_density = &HeII; f.HeIII_density = &HeIII; f.e_density = &e;}
    if(chem > 1) {f.H2I_density = &H2I; f.H2II_density = &H2II; f.H2_self_shielding_length = &L;}
    if(chem > 2) {f.DI_density = &DI; f.DII_density = &DII; f.HDI_density = &HDI;}

    double t = 0, dt_out = t_end / n_out, dt = dt_out / nsub;
    for(int k = 0; k <= n_out; k++)
    {
        if(k > 0) for(int s = 0; s < nsub; s++) {if(local_solve_chemistry(gc, &rates, &units, &f, dt) != GR_SUCCESS) {fprintf(stderr, "solve failed\n"); return 1;} t += dt;}
        gr_float T = 0; local_calculate_temperature(gc, &rates, &units, &f, &T);
        double ne = chem > 0 ? e/(X*d) : ne_tab;
        printf("%.9e %.9e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n", t, (double)u, ne,
               HI/d, HII/d, HM/d, HeI/d, HeII/d, HeIII/d, H2I/d, H2II/d, DI/d, DII/d, HDI/d, (double)T);
    }
    local_free_chemistry_data(gc, &rates);
    return 0;
}
