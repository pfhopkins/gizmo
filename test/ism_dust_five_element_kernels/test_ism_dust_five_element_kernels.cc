/* Driver for the unit test of solids/ism_dust_five_element_kernels.h: one case per stdin line, results on stdout. */
#include <stdio.h>
#include <string.h>
#include "solids/ism_dust_five_element_kernels.h"

static const double FRAC[4] = {64.0 / 172.0, 24.0 / 172.0, 28.0 / 172.0, 56.0 / 172.0}; /* FeMgSiO4 with integer masses */

int main(void)
{
    char cmd[16];
    while(scanf("%15s", cmd) == 1) {
        double a[16] = {0}, d[5], z[5];
        if(!strcmp(cmd, "rate")) {
            if(scanf("%lf", &a[0]) != 1) {return 1;}
            printf("%.17g %.17g %.17g\n", dustk_Yth_C(a[0]), dustk_Yth_Sil(a[0]), dust5_sticking(a[0], 20.0));
        } else if(!strcmp(cmd, "destr")) {
            if(scanf("%lf %lf", &a[0], &a[1]) != 2) {return 1;}
            printf("%.17g\n", dust5_swept_mass_destroyed_fraction(a[0], 1.0, a[1]));
        } else if(!strcmp(cmd, "step")) { /* nH T Td dt_yr d[5] z[5] */
            for(int k = 0; k < 14; k++) {if(scanf("%lf", &a[k]) != 1) {return 1;}}
            for(int k = 0; k < 5; k++) {d[k] = a[4 + k]; z[k] = a[9 + k];}
            dust5_sputter(d, a[0], a[1], a[3]);
            dust5_grow(d, z, FRAC, a[0], a[1], a[2], a[3]);
            dust5_clamp(d, z);
            printf("%.17g %.17g %.17g %.17g %.17g\n", d[0], d[1], d[2], d[3], d[4]);
        } else if(!strcmp(cmd, "seed")) { /* Z z[5] */
            for(int k = 0; k < 6; k++) {if(scanf("%lf", &a[k]) != 1) {return 1;}}
            for(int k = 0; k < 5; k++) {z[k] = a[1 + k];}
            dust5_seed(d, z, FRAC, a[0]);
            printf("%.17g %.17g %.17g %.17g %.17g\n", d[0], d[1], d[2], d[3], d[4]);
        } else {return 1;}
    }
    return 0;
}
