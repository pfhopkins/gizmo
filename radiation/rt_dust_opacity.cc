// Dust opacity table and interpolation routine computed using
// https://github.com/mikegrudic/STARFORGE-methods-errata

#include "../declarations/allvars.h"
#include "math.h"
#include "stdio.h"

#define N_TRAD 15 // number of T_rad samples
#define N_TDUST 5 // number of T_dust zones
MyFloat Tdust_zones[N_TDUST] = {160, 275, 425, 680, 1500}; // demarcation of T_dust zones
MyFloat logTrad_table[N_TRAD] = {0.0,
                                 0.2857142857142857,
                                 0.5714285714285714,
                                 0.8571428571428571,
                                 1.1428571428571428,
                                 1.4285714285714284,
                                 1.7142857142857142,
                                 2.0,
                                 2.2857142857142856,
                                 2.571428571428571,
                                 2.8571428571428568,
                                 3.142857142857143,
                                 3.4285714285714284,
                                 3.714285714285714,
                                 4.0};
MyFloat log_kappadust_table[N_TDUST][N_TRAD] = {
    {-1.909515215033498, -1.5017616295543856, -1.2610211141587906,
     -1.0643751130254193, -0.7661794028048912, -0.2485164276172981,
     0.3936319485109052, 0.7185651396015793, 1.003536500936941,
     1.0703750048744685, 1.185414318657744, 1.4334392971521788,
     1.6154100043688104, 1.833331149478953, 2.2402919406422592},
    {-2.0584711507666156, -1.639523808954471, -1.3937562378746238,
     -1.217712452271845, -1.0035307263948003, -0.6537979484512214,
     -0.14444336531252724, 0.33781494089162734, 0.6204502553977504,
     0.7073792380378322, 0.817126602196553, 1.1620592165098858,
     1.4775329387335934, 1.7314724407802422, 2.122441393841703},
    {-2.092756656625387, -1.6688435106176933, -1.4197982288972344,
     -1.2491484838263678, -1.0526101054461765, -0.7239962293949143,
     -0.21897626943630635, 0.26791745490238816, 0.5498747399044497,
     0.6380011931553505, 0.7545581854919681, 1.1204828950445178,
     1.447717175478172, 1.6975481925473666, 2.066531977172171},
    {-2.466846959771108, -1.9983094494283768, -1.7094791411944892,
     -1.5454814504892613, -1.4403397374848839, -1.2856048236415571,
     -0.8297121104893529, -0.28151883033918534, 0.004258175582704628,
     0.12669094376131687, 0.31518820522515606, 0.8267768233252955,
     1.2296194436753016, 1.4691152621947785, 1.6952767195239868},
    {-3.7448342907548136, -3.298656729562377, -2.8766563714426696,
     -2.4999445785716605, -2.1658033063307927, -1.8743193706624732,
     -1.5588128589881036, -1.0606192296411485, -0.33965876540303136,
     0.5368140402181846, 1.1915541410857275, 1.4384491935887818,
     1.4792273276721044, 1.5197919949174217, 1.6603749111299055},
};

/* Returns the Planck-mean dust opacity tabulated for the Semenov 2003 5-layered
 porous shell dust model.

 Parameters
 ----------
 Trad: Radiation temperature in K
 Tdust: Dust temperature in K

 Returns
 -------
 kappa_dust: Planck-mean dust opacity in cm^2/g assuming Solar metallicity
 */
MyFloat dust_planck_mean_opacity(MyFloat Trad, MyFloat Tdust) {
    MyFloat logTmax = logTrad_table[N_TRAD - 1], logTmin = logTrad_table[0];
    MyFloat logT = log10(Trad);

    int Tdust_idx;
    for (Tdust_idx = 0; Tdust_idx < N_TDUST + 1; Tdust_idx++) {
        if (Tdust < Tdust_zones[Tdust_idx]) {
            break;
        }
    }
    if (Tdust_idx == N_TDUST + 1) { // Tdust is hot so dust is sublimated; return smol value
        return MIN_REAL_NUMBER;
    }

    if (logT >= logTmax) {
        return pow(10., log_kappadust_table[Tdust_idx][N_TRAD - 1]);
    }
    if (logT <= logTmin) {
        return pow(10., log_kappadust_table[Tdust_idx][0]);
    }

    MyFloat dlogT = logTrad_table[1] - logTrad_table[0];
    int Trad_idx = (int)(N_TRAD - 1) * logT / (logTmax - logTmin);
    MyFloat wt1 = 1 - (logT - logTrad_table[Trad_idx]) / dlogT, wt2 = 1 - wt1;
    MyFloat log_kappa = wt1 * log_kappadust_table[Tdust_idx][Trad_idx] +
                        wt2 * log_kappadust_table[Tdust_idx][Trad_idx + 1];
    return pow(10., log_kappa);
}
