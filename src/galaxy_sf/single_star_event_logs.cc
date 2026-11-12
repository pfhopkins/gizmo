/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "single_star_record.h"

/* Per-event logs of single stars (GALSF_SINGLE_STARS_EVENT_LOGS): SNinfo.txt, AGBinfo.txt and
 * WINDinfo.txt. A rank keeps its rows until the next snapshot or restart file, where rank 0 gathers them and writes them
 * sorted by time and ID, so the files do not depend on the rank count and no step between outputs has a collective. */
#ifdef GALSF_SINGLE_STARS_EVENT_LOGS
#if !defined(GALSF_SINGLE_STARS_FROM_IMF) || !defined(GALSF_SINGLE_STARS_EVOLUTION_TABLES) || !defined(GALSF_FB_MECHANICAL)
#error "GALSF_SINGLE_STARS_EVENT_LOGS needs single stars with table feedback (GALSF_SINGLE_STARS_FROM_IMF, GALSF_SINGLE_STARS_EVOLUTION_TABLES, GALSF_FB_MECHANICAL)"
#endif

static std::vector<struct single_star_log_row> ssl_rows;
static FILE *ssl_fd[3] = {NULL, NULL, NULL};

void single_star_event_log_push(const struct single_star_log_row *r) {ssl_rows.push_back(*r);}

static void ssl_open(void)
{
    static const char *name[3] = {"SNinfo.txt", "AGBinfo.txt", "WINDinfo.txt"};
    static const char *cols[3] = {
        "(1) time  (2) x  (3) y  (4) z  (5) u_ambient  (6) rho_ambient  (7) id  (8) m_star[Msun]  (9) remnant_type [1=ECSN,2=CCSN,3=FSN,4=PPISN,5=PISN,6=DBH,7=TypeIa]  (10) M_ej[Msun]  (11) Z_ej[Msun]  (12) C[Msun]  (13) O[Msun]  (14) Si[Msun]  (15) Fe[Msun]  (16) age_at_death[yr]  (17) lifetime_table[yr]  (18) timebin  (19) E_SN[erg]  (20) status [0=exploded+deposited,1=failed SN (no energy),2=no gas reached (nothing booked, retried)]  (21) n_H_ambient[cm^-3]  (22) T_ambient[K, mu=1.22]  (23) h_star[pc]  (24) NumNgb  (25) n_H_birth[cm^-3]  (26) n_H_birth_frozen[cm^-3, birth comoving density at the current a]",
        "(1) time  (2) x  (3) y  (4) z  (5) u_ambient  (6) rho_ambient  (7) id  (8) m_star[Msun]  (9) M_ej[Msun]  (10) Z_ej[Msun]  (11) C[Msun]  (12) O[Msun]  (13) Si[Msun]  (14) Fe[Msun]  (15) age_at_death[yr]  (16) lifetime_table[yr]  (17) timebin  (18) status [0=ejected+deposited,2=no gas reached (nothing booked, retried)]  (19) n_H_ambient[cm^-3]  (20) T_ambient[K, mu=1.22]  (21) h_star[pc]  (22) NumNgb  (23) n_H_birth[cm^-3]  (24) n_H_birth_frozen[cm^-3, birth comoving density at the current a]",
        "(1) time  (2) x  (3) y  (4) z  (5) id  (6) M_init[Msun]  (7) dM_wind[Msun]  (8) dp_wind[g*cm/s]  (9) Z_wind[Msun]  (10) timebin"};
    for(int f = 0; f < 3; f++) {
        if(ssl_fd[f]) {continue;}
        char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE];
        snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s%s", All.OutputDir, name[f]);
        ssl_fd[f] = fopen(buf, (RestartFlag == 0) ? "w" : "a");
        if(ssl_fd[f] && RestartFlag == 0) {fprintf(ssl_fd[f], "### %s\n", cols[f]);}
    }
}

static void ssl_print(const struct single_star_log_row *r)
{
    FILE *fd = ssl_fd[r->file];
    if(!fd) {return;}
    long long id = (long long) r->id;
    if(r->file == 0) {fprintf(fd, "%.10e  %.8e %.8e %.8e  %.6e %.6e  %lld  %.8g  %d  %.10g  %.10g  %.8e %.8e %.8e %.8e  %.6e  %.6e  %3d  %.4e  %d  %.4e %.4e %.4e %7.2f %.4e %.4e\n", r->t, r->x[0], r->x[1], r->x[2], r->u, r->rho, id, r->mstar, r->remnant, r->mej, r->zej, r->C, r->O, r->Si, r->Fe, r->age, r->life, r->timebin, r->E, r->status, r->nH, r->T, r->h_pc, r->nngb, r->nH_birth, r->nH_stale);}
    if(r->file == 1) {fprintf(fd, "%.10e  %.8e %.8e %.8e  %.6e %.6e  %lld  %.8g  %.10g  %.10g  %.8e %.8e %.8e %.8e  %.6e  %.6e  %3d  %d  %.4e %.4e %.4e %7.2f %.4e %.4e\n", r->t, r->x[0], r->x[1], r->x[2], r->u, r->rho, id, r->mstar, r->mej, r->zej, r->C, r->O, r->Si, r->Fe, r->age, r->life, r->timebin, r->status, r->nH, r->T, r->h_pc, r->nngb, r->nH_birth, r->nH_stale);}
    if(r->file == 2) {fprintf(fd, "%.10e  %.8e %.8e %.8e  %lld  %.8g  %.8e  %.6e  %.6e  %3d\n", r->t, r->x[0], r->x[1], r->x[2], id, r->mstar, r->mej, r->E, r->zej, r->timebin);}
}

/* collective: from the output routines only */
void single_star_event_logs_write(void)
{
    int nb = (int)(ssl_rows.size() * sizeof(struct single_star_log_row)), tot = 0;
    std::vector<int> counts(NTask), displs(NTask);
    MPI_Gather(&nb, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(ThisTask == 0) {for(int t = 0; t < NTask; t++) {displs[t] = tot; tot += counts[t];}}
    std::vector<struct single_star_log_row> all(tot / sizeof(struct single_star_log_row) + 1);
    MPI_Gatherv(ssl_rows.data(), nb, MPI_BYTE, all.data(), counts.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    ssl_rows.clear();
    if(ThisTask != 0) {return;}
    ssl_open();
    size_t n = tot / sizeof(struct single_star_log_row);
    std::stable_sort(all.begin(), all.begin() + n, [](const struct single_star_log_row &a, const struct single_star_log_row &b) {return (a.t != b.t) ? (a.t < b.t) : (a.id < b.id);});
    for(size_t k = 0; k < n; k++) {ssl_print(&all[k]);}
    for(int f = 0; f < 3; f++) {if(ssl_fd[f]) {fflush(ssl_fd[f]);}}
}

/* sfr_rism.txt: star-formation rate [Msun/yr] of the stars younger than 50, 40, 30, 20, 5 and 1 Myr (drawn masses) and the
   number of stars; collective with one reduction, called in the steps that build the tree only */
void single_star_sfr_windows_write(void)
{
    static const double window_myr[6] = {50., 40., 30., 20., 5., 1.};
    double loc[7] = {0, 0, 0, 0, 0, 0, 0}, glob[7];
    for(int i = 0; i < NumPart; i++) {
        if(P[i].Type != 4 || P[i].StarIdx <= 0 || P[i].Mass <= 0) {continue;}
        double M = SingleStars.M_drawn[single_star_slot(P[i].StarIdx)], age_myr = evaluate_stellar_age_Gyr(i) * 1.e3;
        loc[6] += 1;
        for(int w = 0; w < 6; w++) {if(age_myr <= window_myr[w]) {loc[w] += M;}}
    }
    MPI_Reduce(loc, glob, 7, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask != 0) {return;}
    static FILE *fd = NULL;
    if(!fd) {
        char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE]; snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s%s", All.OutputDir, "sfr_rism.txt");
        fd = fopen(buf, (RestartFlag == 0) ? "w" : "a");
        if(fd && RestartFlag == 0) {fprintf(fd, "# time N_stars SFR_50Myr SFR_40Myr SFR_30Myr SFR_20Myr SFR_5Myr SFR_1Myr [Msun/yr]\n");}
    }
    if(!fd) {return;}
    fprintf(fd, "%12.6f  %8d", All.Time, (int) glob[6]);
    for(int w = 0; w < 6; w++) {fprintf(fd, "  %12.6e", glob[w] / (window_myr[w] * 1.e6));}
    fprintf(fd, "\n"); fflush(fd);
}

#endif
