#ifndef SINGLE_STAR_SN_ENERGY_H
#define SINGLE_STAR_SN_ENERGY_H
/* single-star supernova energy by death type [erg] (stella_dev F6): PISN from the He core (Heger & Woosley 2002 via the Wise+12a fit,
   M_He = (13/24)(M_ZAMS - 20) held to 64-133 Msun); ECSN 1e50 (Kitaura, Janka & Hillebrandt 2006); other explosions 1e51; failed SNe
   and direct collapse 0. One helper for the deposit, the ledger and the logs. */
static inline double single_star_sn_energy_erg(int is_pisn, int is_ecsn, int explodes, double M_zams)
{
    if(!explodes) {return 0;}
    if(is_pisn) {
        double M_He = (13. / 24.) * (M_zams - 20.);
        M_He = (M_He < 64.) ? 64. : ((M_He > 133.) ? 133. : M_He);
        return (5.0 + 1.304 * (M_He - 64.)) * 1.e51;
    }
    if(is_ecsn) {return 1.e50;}
    return 1.e51;
}
#endif
