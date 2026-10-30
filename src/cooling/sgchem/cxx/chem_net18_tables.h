/* C++ implementation: Ulrich Steinwandel (uli@mpa-garching.mpg.de).
   Based on the SGChem Fortran networks by Simon Glover (GNU GPL, as distributed in Arepo). */
// chem_net18_tables.h -- network 18 rate tables (RTZ), read at start-up from the installed sgchem_net18_tables.bin:
// the arrays below, in this order, as little-endian float64; content hash checked with the other chemistry tables.
#ifndef CHEM_NET18_TABLES_H
#define CHEM_NET18_TABLES_H
#include <cstddef>
static double n18_rrec[30][30][2];
static double n18_rnew[30][30][4];
static double n18_rfe[13][3];
static double n18_dr_c[9][27][9];
static double n18_dr_e[9][27][9];
static double n18_ci[27][27][5];
static double n18_ction[7][3][31];
static double n18_ctrec[6][4][31];
static double n18_cr[10][27];
static double n18_cr_uv[27];
static double n18_cr_uv_heat[27];
static double n18_dustrec[27][7];
static double n18_g0_heat[27];
static double n18_hit_logT[121];
static double n18_hit_6[7][121];
static double n18_hit_7[8][121];
static double n18_hit_8[9][121];
static double n18_hit_10[11][121];
static double n18_hit_12[13][121];
static double n18_hit_14[15][121];
static double n18_hit_16[17][121];
static double n18_hit_26[27][121];
static double n18_fs[27][160][8];
static double n18_uvb_z[60];
static double n18_uvb_pi_1[60][1];
static double n18_uvb_ph_1[60][1];
static double n18_uvb_pi_2[60][2];
static double n18_uvb_ph_2[60][2];
static double n18_uvb_pi_6[60][6];
static double n18_uvb_ph_6[60][6];
static double n18_uvb_pi_7[60][7];
static double n18_uvb_ph_7[60][7];
static double n18_uvb_pi_8[60][8];
static double n18_uvb_ph_8[60][8];
static double n18_uvb_pi_10[60][10];
static double n18_uvb_ph_10[60][10];
static double n18_uvb_pi_12[60][12];
static double n18_uvb_ph_12[60][12];
static double n18_uvb_pi_14[60][14];
static double n18_uvb_ph_14[60][14];
static double n18_uvb_pi_16[60][16];
static double n18_uvb_ph_16[60][16];
static double n18_uvb_pi_26[60][26];
static double n18_uvb_ph_26[60][26];
static const struct { double* p; std::size_t n; } n18_table_list[] = {
    {&n18_rrec[0][0][0], sizeof(n18_rrec) / sizeof(double)},
    {&n18_rnew[0][0][0], sizeof(n18_rnew) / sizeof(double)},
    {&n18_rfe[0][0], sizeof(n18_rfe) / sizeof(double)},
    {&n18_dr_c[0][0][0], sizeof(n18_dr_c) / sizeof(double)},
    {&n18_dr_e[0][0][0], sizeof(n18_dr_e) / sizeof(double)},
    {&n18_ci[0][0][0], sizeof(n18_ci) / sizeof(double)},
    {&n18_ction[0][0][0], sizeof(n18_ction) / sizeof(double)},
    {&n18_ctrec[0][0][0], sizeof(n18_ctrec) / sizeof(double)},
    {&n18_cr[0][0], sizeof(n18_cr) / sizeof(double)},
    {&n18_cr_uv[0], sizeof(n18_cr_uv) / sizeof(double)},
    {&n18_cr_uv_heat[0], sizeof(n18_cr_uv_heat) / sizeof(double)},
    {&n18_dustrec[0][0], sizeof(n18_dustrec) / sizeof(double)},
    {&n18_g0_heat[0], sizeof(n18_g0_heat) / sizeof(double)},
    {&n18_hit_logT[0], sizeof(n18_hit_logT) / sizeof(double)},
    {&n18_hit_6[0][0], sizeof(n18_hit_6) / sizeof(double)},
    {&n18_hit_7[0][0], sizeof(n18_hit_7) / sizeof(double)},
    {&n18_hit_8[0][0], sizeof(n18_hit_8) / sizeof(double)},
    {&n18_hit_10[0][0], sizeof(n18_hit_10) / sizeof(double)},
    {&n18_hit_12[0][0], sizeof(n18_hit_12) / sizeof(double)},
    {&n18_hit_14[0][0], sizeof(n18_hit_14) / sizeof(double)},
    {&n18_hit_16[0][0], sizeof(n18_hit_16) / sizeof(double)},
    {&n18_hit_26[0][0], sizeof(n18_hit_26) / sizeof(double)},
    {&n18_fs[0][0][0], sizeof(n18_fs) / sizeof(double)},
    {&n18_uvb_z[0], sizeof(n18_uvb_z) / sizeof(double)},
    {&n18_uvb_pi_1[0][0], sizeof(n18_uvb_pi_1) / sizeof(double)},
    {&n18_uvb_ph_1[0][0], sizeof(n18_uvb_ph_1) / sizeof(double)},
    {&n18_uvb_pi_2[0][0], sizeof(n18_uvb_pi_2) / sizeof(double)},
    {&n18_uvb_ph_2[0][0], sizeof(n18_uvb_ph_2) / sizeof(double)},
    {&n18_uvb_pi_6[0][0], sizeof(n18_uvb_pi_6) / sizeof(double)},
    {&n18_uvb_ph_6[0][0], sizeof(n18_uvb_ph_6) / sizeof(double)},
    {&n18_uvb_pi_7[0][0], sizeof(n18_uvb_pi_7) / sizeof(double)},
    {&n18_uvb_ph_7[0][0], sizeof(n18_uvb_ph_7) / sizeof(double)},
    {&n18_uvb_pi_8[0][0], sizeof(n18_uvb_pi_8) / sizeof(double)},
    {&n18_uvb_ph_8[0][0], sizeof(n18_uvb_ph_8) / sizeof(double)},
    {&n18_uvb_pi_10[0][0], sizeof(n18_uvb_pi_10) / sizeof(double)},
    {&n18_uvb_ph_10[0][0], sizeof(n18_uvb_ph_10) / sizeof(double)},
    {&n18_uvb_pi_12[0][0], sizeof(n18_uvb_pi_12) / sizeof(double)},
    {&n18_uvb_ph_12[0][0], sizeof(n18_uvb_ph_12) / sizeof(double)},
    {&n18_uvb_pi_14[0][0], sizeof(n18_uvb_pi_14) / sizeof(double)},
    {&n18_uvb_ph_14[0][0], sizeof(n18_uvb_ph_14) / sizeof(double)},
    {&n18_uvb_pi_16[0][0], sizeof(n18_uvb_pi_16) / sizeof(double)},
    {&n18_uvb_ph_16[0][0], sizeof(n18_uvb_ph_16) / sizeof(double)},
    {&n18_uvb_pi_26[0][0], sizeof(n18_uvb_pi_26) / sizeof(double)},
    {&n18_uvb_ph_26[0][0], sizeof(n18_uvb_ph_26) / sizeof(double)},
};
#endif
