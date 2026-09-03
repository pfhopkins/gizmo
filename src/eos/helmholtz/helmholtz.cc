/* C++ port of the Timmes & Swesty (2000) Helmholtz free energy EOS.
   Ported from helm_impl.f90 and helm_wrap.f90.
   Thread-safe: all mutable state is stack-local; the table is const after init.

   References: Timmes & Swesty, ApJS 126, 501 (2000)
               Yakovlev & Shalybkov, Sov. Sci. Rev. E. Astrophys. Space Phys. 7, 311 (1989) [Coulomb]
*/

#include "helmholtz.h"

#ifdef EOS_HELMHOLTZ

#include <cstdio>
/* Only the table reader lives here now. Every evaluation routine moved into
   helmholtz.h so that device translation units can call it; this file keeps
   the part that cannot go anywhere, which is the file I/O. */

/* =========================================================================
   Table I/O
   ========================================================================= */

int helm_read_table(const char *filename, HelmTable *tab)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "helm_read_table: cannot open '%s'\n", filename);
        return 1;
    }

    const int imax = HELM_IMAX;
    const int jmax = HELM_JMAX;

    /* grid parameters (must match table) */
    tab->tlo  =  3.0;  tab->thi  = 13.0;
    tab->tstp = (tab->thi - tab->tlo) / (double)(jmax - 1);
    tab->tstpi = 1.0 / tab->tstp;
    tab->dlo  = -12.0; tab->dhi  = 15.0;
    tab->dstp = (tab->dhi - tab->dlo) / (double)(imax - 1);
    tab->dstpi = 1.0 / tab->dstp;

    /* read the Helmholtz free energy table and its derivatives */
    for (int j = 0; j < jmax; j++) {
        tab->t[j] = pow(10.0, tab->tlo + j * tab->tstp);
        for (int i = 0; i < imax; i++) {
            tab->d[i] = pow(10.0, tab->dlo + i * tab->dstp);
            if (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf",
                    &tab->f[i][j],    &tab->fd[i][j],   &tab->ft[i][j],
                    &tab->fdd[i][j],  &tab->ftt[i][j],  &tab->fdt[i][j],
                    &tab->fddt[i][j], &tab->fdtt[i][j],  &tab->fddtt[i][j]) != 9) {
                fprintf(stderr, "helm_read_table: error reading free energy at i=%d j=%d\n", i, j);
                fclose(fp);
                return 1;
            }
        }
    }

    /* read the pressure derivative with density table */
    for (int j = 0; j < jmax; j++) {
        for (int i = 0; i < imax; i++) {
            if (fscanf(fp, "%lf %lf %lf %lf",
                    &tab->dpdf[i][j],  &tab->dpdfd[i][j],
                    &tab->dpdft[i][j], &tab->dpdfdt[i][j]) != 4) {
                fprintf(stderr, "helm_read_table: error reading dpdf at i=%d j=%d\n", i, j);
                fclose(fp);
                return 1;
            }
        }
    }

    /* read the electron chemical potential table */
    for (int j = 0; j < jmax; j++) {
        for (int i = 0; i < imax; i++) {
            if (fscanf(fp, "%lf %lf %lf %lf",
                    &tab->ef[i][j],  &tab->efd[i][j],
                    &tab->eft[i][j], &tab->efdt[i][j]) != 4) {
                fprintf(stderr, "helm_read_table: error reading ef at i=%d j=%d\n", i, j);
                fclose(fp);
                return 1;
            }
        }
    }

    /* read the number density table */
    for (int j = 0; j < jmax; j++) {
        for (int i = 0; i < imax; i++) {
            if (fscanf(fp, "%lf %lf %lf %lf",
                    &tab->xf[i][j],  &tab->xfd[i][j],
                    &tab->xft[i][j], &tab->xfdt[i][j]) != 4) {
                fprintf(stderr, "helm_read_table: error reading xf at i=%d j=%d\n", i, j);
                fclose(fp);
                return 1;
            }
        }
    }

    fclose(fp);

    /* precompute temperature and density deltas and their inverses */
    for (int j = 0; j < jmax - 1; j++) {
        double dth = tab->t[j+1] - tab->t[j];
        tab->dt_sav[j]  = dth;
        tab->dt2_sav[j] = dth * dth;
        tab->dti_sav[j] = 1.0 / dth;
        tab->dt2i_sav[j] = 1.0 / (dth * dth);
        tab->dt3i_sav[j] = tab->dt2i_sav[j] * tab->dti_sav[j];
    }
    for (int i = 0; i < imax - 1; i++) {
        double dd = tab->d[i+1] - tab->d[i];
        tab->dd_sav[i]  = dd;
        tab->dd2_sav[i] = dd * dd;
        tab->ddi_sav[i] = 1.0 / dd;
        tab->dd2i_sav[i] = 1.0 / (dd * dd);
        tab->dd3i_sav[i] = tab->dd2i_sav[i] * tab->ddi_sav[i];
    }

    tab->initialized = 1;
    return 0;
}

#endif /* EOS_HELMHOLTZ */
