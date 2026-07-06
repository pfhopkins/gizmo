#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <ctype.h>

#include "../declarations/allvars.h"
#include "../mesh/kernel.h"   /* kernel_gravity: softened-potential metric in incode_total_energy */
#include "../core/proto.h"

/* mbary_step_checkpoint: always-on mass-balance leak locator.  Logs to
 * <OutputDir>/mass_balance_steps.txt one line per checkpoint where EITHER
 * M_bary (total baryonic mass) OR M_Z (total gas metal mass) deviates from
 * the previous call by more than 1e-14 fractional.  M_Z change between
 * pre/post_diff_unpack catches metal-diffusion leaks; M_bary change at any
 * tag catches mass leaks.  Use it to bracket suspect operations. */
void mbary_step_checkpoint(const char *tag)
{
    static double last_M_tot = -1.0;
    static double last_M_Z   = -1.0;
#ifdef GALSF_RESOLVEDISM_DUST
    static double last_M_dust = -1.0;
#endif
    static long long call_count = 0;
    int i;
    double M_gas_loc = 0, M_star_loc = 0, M_Z_loc = 0;
#ifdef GALSF_RESOLVEDISM_DUST
    double M_dust_loc = 0;
#endif
    for(i = 0; i < NumPart; i++) {
        if(P[i].Mass <= 0) continue;
        if(P[i].Type == 0) {
            M_gas_loc += P[i].Mass;
#if defined(METALS)
            M_Z_loc += P[i].Mass * P[i].Metallicity[0];  /* gas metal mass = Σ M·Met[0] */
#endif
#ifdef GALSF_RESOLVEDISM_DUST
            /* Dust is a PARTITION of the metals (Met[k] includes dust-locked atoms),
             * so a dust diffusion leak is invisible in M_Z and M_bary — it only
             * relabels dust-phase mass as gas-phase.  Track Σ M·ΣDust separately:
             * must be constant across post_diff_unpack; changes legitimately at
             * FB (yields/shock destruction) and sputtering checkpoints. */
            {int kd; double dsum = 0; for(kd = 0; kd < NUM_RESOLVEDISM_DUST; kd++) {dsum += CellP[i].Dust[kd];}
             M_dust_loc += P[i].Mass * dsum;}
#endif
        }
        else if(P[i].Type == 4) {
            M_star_loc += P[i].Mass;
        }
    }
    double M_gas = 0, M_star = 0, M_Z = 0;
    MPI_Allreduce(&M_gas_loc,  &M_gas,  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&M_star_loc, &M_star, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&M_Z_loc,    &M_Z,    1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#ifdef GALSF_RESOLVEDISM_DUST
    double M_dust = 0;
    MPI_Allreduce(&M_dust_loc, &M_dust, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    double M_tot = M_gas + M_star;
    if(ThisTask == 0) {
        int should_log = 0;
        double dM = 0.0, dZ = 0.0;
#ifdef GALSF_RESOLVEDISM_DUST
        double dDust = 0.0;
#endif
        if(last_M_tot < 0) { should_log = 1; }
        else {
            dM = M_tot - last_M_tot;
            dZ = M_Z   - last_M_Z;
            double frac_M = (last_M_tot > 0) ? fabs(dM) / last_M_tot : 0.0;
            double frac_Z = (last_M_Z   > 0) ? fabs(dZ) / last_M_Z   : 0.0;
            if(frac_M > 1.0e-14 || frac_Z > 1.0e-14) should_log = 1;
#ifdef GALSF_RESOLVEDISM_DUST
            dDust = M_dust - last_M_dust;
            double frac_D = (last_M_dust > 0) ? fabs(dDust) / last_M_dust : 0.0;
            if(frac_D > 1.0e-14) should_log = 1;
#endif
        }
        if(should_log) {
            char path[DEFAULT_PATH_BUFFERSIZE_TOUSE];
            snprintf(path, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s/mass_balance_steps.txt", All.OutputDir);
            FILE *f = fopen(path, "a");
            if(f) {
                if(ftell(f) == 0) {
                    fprintf(f, "### Per-checkpoint M_bary + M_Z delta trace for leak localization.\n");
                    fprintf(f, "###  Logs only when |delta|/value > 1e-14 for M_bary, M_Z, or M_dust (or first call).\n");
#ifdef GALSF_RESOLVEDISM_DUST
                    fprintf(f, "###  (1) call_num  (2) tag  (3) t  (4) M_bary[code]  (5) deltaM_bary[code]  (6) M_Z[code]  (7) deltaM_Z[code]  (8) M_dust[code]  (9) deltaM_dust[code]\n");
#else
                    fprintf(f, "###  (1) call_num  (2) tag  (3) t  (4) M_bary[code]  (5) deltaM_bary[code]  (6) M_Z[code]  (7) deltaM_Z[code]\n");
#endif
                }
#ifdef GALSF_RESOLVEDISM_DUST
                fprintf(f, "%8lld  %-22s  %.10e  %.16e  %+.16e  %.16e  %+.16e  %.16e  %+.16e\n",
                    call_count, tag, All.Time, M_tot, dM, M_Z, dZ, M_dust, dDust);
#else
                fprintf(f, "%8lld  %-22s  %.10e  %.16e  %+.16e  %.16e  %+.16e\n",
                    call_count, tag, All.Time, M_tot, dM, M_Z, dZ);
#endif
                fclose(f);
            }
        }
        last_M_tot = M_tot;
        last_M_Z   = M_Z;
#ifdef GALSF_RESOLVEDISM_DUST
        last_M_dust = M_dust;
#endif
        call_count++;
    }
}

/* diff_unpack_breakdown_log: writes per-step pair-flux Σ Dyield[0] and clamp-correction
 * Σ M·(post-pre-Dyield/Mass) to <OutputDir>/diff_unpack_breakdown.txt.  Pair-flux total
 * should be ~0 if diffusion solver is conservative; clamp contribution should be ~0 if
 * the DMAX(.., 0.0) floor never fires.  Pinpoints which mechanism leaks metal mass. */
void diff_unpack_breakdown_log(double dMZ_pair_loc, double dMZ_clamp_loc)
{
    double dMZ_pair = 0, dMZ_clamp = 0;
    MPI_Allreduce(&dMZ_pair_loc,  &dMZ_pair,  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&dMZ_clamp_loc, &dMZ_clamp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(ThisTask == 0) {
        char path[DEFAULT_PATH_BUFFERSIZE_TOUSE];
        snprintf(path, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s/diff_unpack_breakdown.txt", All.OutputDir);
        FILE *f = fopen(path, "a");
        if(f) {
            if(ftell(f) == 0) {
                fprintf(f, "### Per-step diffusion unpack breakdown of dM_Z contribution.\n");
                fprintf(f, "###   dM_Z_pair_flux: Σ_cells Dyield[0] (should be 0 if solver is conservative).\n");
                fprintf(f, "###   dM_Z_clamp:     Σ_cells M·(post_Met[0] − pre_Met[0] − Dyield[0]/M) — non-zero iff DMAX(,0) floor fires.\n");
                fprintf(f, "###   (1) time  (2) dM_Z_pair_flux[code]  (3) dM_Z_clamp[code]\n");
            }
            fprintf(f, "%.10e  %+.16e  %+.16e\n", All.Time, dMZ_pair, dMZ_clamp);
            fclose(f);
        }
    }
}

#if defined(GALSF_RESOLVEDISM_METALS_INDIVIDUAL) && defined(GALSF_RESOLVEDISM_ISOLATED_FB_TEST)
/* Mass-conservation diagnostic for isolated-FB tests.  Logs at each call:
 *   - Total baryonic mass (gas + stars)
 *   - Total per-element mass Σ_cells P[i].Mass × P[i].Metallicity[k]  for k=0..27
 *   - Per-cell Σ Met[1..27] stats (min/max/#violating |Σ−1|>tol)
 * Subtracting consecutive log lines reveals which operation (hydro/diffusion/
 * chemistry/FB/merge_split) leaks mass.  Tagged with `MCBAL` for grep. */
void resolvedism_mass_balance_log(const char *tag)
{
    int i, k;
    double M_gas_loc = 0, M_star_loc = 0;
    double M_per_elem_loc[NUM_METAL_SPECIES];
    for(k = 0; k < NUM_METAL_SPECIES; k++) M_per_elem_loc[k] = 0;
    double sigMet_min_loc = 1e30, sigMet_max_loc = -1e30;
    long n_viol_loc = 0;

    for(i = 0; i < NumPart; i++) {
        if(P[i].Mass <= 0) continue;
        if(P[i].Type == 0) {
            M_gas_loc += P[i].Mass;
            for(k = 0; k < NUM_METAL_SPECIES; k++)
                M_per_elem_loc[k] += P[i].Mass * P[i].Metallicity[k];
            double sigMet = 0;
            for(k = 1; k < NUM_METAL_SPECIES; k++) sigMet += P[i].Metallicity[k];
            if(sigMet < sigMet_min_loc) sigMet_min_loc = sigMet;
            if(sigMet > sigMet_max_loc) sigMet_max_loc = sigMet;
            if(fabs(sigMet - 1.0) > 1.0e-6) n_viol_loc++;
        } else if(P[i].Type == 4) {
            M_star_loc += P[i].Mass;
        }
    }

    double M_gas, M_star;
    double M_per_elem[NUM_METAL_SPECIES];
    double sigMet_min, sigMet_max;
    long n_viol;
    MPI_Reduce(&M_gas_loc,  &M_gas,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&M_star_loc, &M_star, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(M_per_elem_loc, M_per_elem, NUM_METAL_SPECIES, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sigMet_min_loc, &sigMet_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sigMet_max_loc, &sigMet_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&n_viol_loc, &n_viol, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if(ThisTask == 0) {
        double M_tot = (M_gas + M_star) * UNIT_MASS_IN_SOLAR;
        printf("MCBAL %s  t=%.6e  M_tot=%.10e  M_gas=%.10e  M_star=%.10e  "
               "ΣMet[1..27]: min=%.9e max=%.9e #viol=%ld\n",
            tag, All.Time, M_tot, M_gas*UNIT_MASS_IN_SOLAR, M_star*UNIT_MASS_IN_SOLAR,
            sigMet_min, sigMet_max, n_viol);
        printf("MCBAL %s  per-element[k=0..27]:", tag);
        for(k = 0; k < NUM_METAL_SPECIES; k++)
            printf(" %.6e", M_per_elem[k]*UNIT_MASS_IN_SOLAR);
        printf("\n");
        fflush(stdout);
    }
}
/* MCBAL_LOG is defined in proto.h so other translation units can call it. */
#endif


/*! \file run.c
 *  \brief  iterates over timesteps, main loop
 */
/*!
 * This file was originally part of the GADGET3 code developed by
 * Volker Springel. The code has been modified
 * heavily (adding/removing calls, re-ordering some routines, and
 * adding hooks to new elements such as particle splitting, as necessary)
 * for GIZMO by Phil Hopkins (phopkins@caltech.edu) and Mike Grudic (also
 * adding options needed for higher-order Runge-Kutta and Hermite integration)
 */


#ifndef KETJU_REGULARIZATION
#define KETJU_ETOT_PTYPE(i) (P[(i)].Type == 4)
#else
/* Stella: KETJU stars are PartType 4 (dead remnants included); Type 5 only with sinks enabled */
#ifdef SINK_PARTICLES
#define KETJU_ETOT_PTYPE(i) (P[(i)].Type == 4 || P[(i)].Type == 5)
#else
#define KETJU_ETOT_PTYPE(i) (P[(i)].Type == 4)
#endif
#endif
/* DIAGNOSTIC: direct double-precision total energy over Type-5 particles, evaluated at the
 * sync point at the top of the loop. Unsoftened PE (exact for r >> softening).
 *
 * KETJU subtlety: for chain members the host-KDK handoff leaves P.Vel half-a-kick off the true
 * t_sync velocity (the 2nd host half-kick and the negative half-kick do not perfectly cancel),
 * so P.Vel gives a spuriously large energy error at fast (e.g. pericenter) phases even though the
 * orbit/energy are actually conserved. We therefore use P.KetjuTrueVel — MSTAR's true end-of-step
 * velocity, captured before the negative half-kick — for KETJU-integrated particles, and plain
 * P.Vel for everyone else. (KE_raw from P.Vel is still printed for comparison.) */
static void incode_total_energy(void)
{
    /* MPI-aware: gather all Type-5 particles to rank 0 and do the exact double-precision
       KE + unsoftened PE there. Works for any NTask. */
    static double E0 = 0; static int have0 = 0; static double last_t = -1e30;
    /* NOTE: this routine now runs its gather + pair loop EVERY step (no early throttle): the
     * Hamiltonian-SWITCH LEDGER below must catch region-membership transitions at the step they occur
     * (the potential-convention switch of an inside-kernel pair must be evaluated at the separation
     * where it actually happened). Only the PRINT remains throttled to snapshot cadence. Cost is
     * O(N_type5^2) per step — the diagnostic already bails for N>4096. */
    /* FULL-SYNC GATE: the total energy is only unambiguous when EVERY particle's (pos,vel) are at the
     * same physical instant. In a block-timestep run that is exactly a full step (highest active bin ==
     * highest occupied bin): then chain members' MSTAR end-of-step state (KetjuTruePos/Vel) and all
     * non-members' P.Pos/P.Vel are genuinely simultaneous, so the cross-term PE(member,non-member) is
     * consistent. At a SUB-sync only some bins are current, so that cross-term mixes phases and injects a
     * bounded, non-secular ~1e-3 artifact (proven: cold-collapse structure is identical between KETJU and
     * Hermite — no real energy loss). We therefore capture E0 and report dE ONLY on full steps. The
     * ledger (below) still accumulates every step so membership switches are caught at their true
     * separation. */
    int full_sync = (All.HighestActiveTimeBin == All.HighestOccupiedTimeBin);
    int do_print = full_sync && !(have0 && (All.Time - last_t < All.TimeBetSnapshot - 1e-9));

    /* per particle, 11 doubles: pos[3], Vel[3], (unused), mass, integrated-flag.
     * With pure-MSTAR coupling the chain member's synchronized P.Vel and P.Pos are exactly
     * MSTAR's end-of-step state, so the plain sync-point values give the true energy. */
    const int W = 12;  /* [11] = particle ID (for the switch-ledger pair tracking across steps) */
    int nloc = 0, i, j;
    for(i = 0; i < NumPart; i++) if(KETJU_ETOT_PTYPE(i)) nloc++;
    double *sloc = (double*)malloc((nloc > 0 ? nloc : 1) * W * sizeof(double));
    int k = 0;
    for(i = 0; i < NumPart; i++) if(KETJU_ETOT_PTYPE(i)) {
        /* Use P.Pos/P.Vel by default. For a KETJU chain member these host-frame values are the
         * velocity-trick reconstruction: P.Pos is a LINEAR drift (chord) of the true curved internal
         * orbit and P.Vel the trick velocity, so at large member host steps they misrepresent the
         * internal separation/velocity and give a spurious (constant) PE/KE offset even though MSTAR
         * conserves the orbit exactly. Use MSTAR's synchronized true end-of-step state instead. */
        double px=P[i].Pos[0], py=P[i].Pos[1], pz=P[i].Pos[2];
        double vx=P[i].Vel[0], vy=P[i].Vel[1], vz=P[i].Vel[2];
        sloc[k*W+10]=0.0;
#ifdef KETJU_REGULARIZATION
        if(P[i].KetjuIntegrated) {
            px=P[i].KetjuTruePos[0]; py=P[i].KetjuTruePos[1]; pz=P[i].KetjuTruePos[2];
            vx=P[i].KetjuTrueVel[0]; vy=P[i].KetjuTrueVel[1]; vz=P[i].KetjuTrueVel[2];
        }
        sloc[k*W+10]=(P[i].KetjuIntegrated ? 1.0 : 0.0);
#endif
        sloc[k*W+0]=px; sloc[k*W+1]=py; sloc[k*W+2]=pz;
        sloc[k*W+3]=vx; sloc[k*W+4]=vy; sloc[k*W+5]=vz;
        /* [6] = kernel radius (compact-support h) for the softened-potential metric;
         * [7] = KETJU region tag (0 = not a chain member; same nonzero tag = same chain);
         * [8] = 1 if chain-internal pairs are UNSOFTENED (KetjuUseStarStarSoftening=0), else 0 */
        sloc[k*W+6]=ForceSoftening_KernelRadius(i);
        sloc[k*W+7]=0.0; sloc[k*W+8]=0.0;
#ifdef KETJU_REGULARIZATION
        sloc[k*W+7]=(double)P[i].KetjuRegionTag;
        sloc[k*W+8]=(All.KetjuUseStarStarSoftening ? 0.0 : 1.0);
#endif
        sloc[k*W+9]=P[i].Mass;
        sloc[k*W+11]=(double)P[i].ID;
        k++;
    }
    int *cnt = NULL, *disp = NULL;
    if(ThisTask == 0) { cnt = (int*)malloc(NTask*sizeof(int)); disp = (int*)malloc(NTask*sizeof(int)); }
    int sendcnt = nloc * W;
    MPI_Gather(&sendcnt, 1, MPI_INT, cnt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int ntot = 0; double *all = NULL;
    if(ThisTask == 0) {
        for(i = 0; i < NTask; i++) { disp[i] = ntot; ntot += cnt[i]; }
        all = (double*)malloc((ntot > 0 ? ntot : 1) * sizeof(double));
    }
    MPI_Gatherv(sloc, sendcnt, MPI_DOUBLE, all, cnt, disp, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    free(sloc);

    if(ThisTask == 0) {
        int n = ntot / W;
        if(n >= 2 && n <= 4096) {
            double KE = 0, PE = 0, PE_prop = 0; int n_int = 0;
            /* HAMILTONIAN-SWITCH LEDGER: when an inside-kernel pair changes potential convention
             * (softened <-> unsoftened) because its chain membership changed, the proper-metric energy
             * jumps by the REAL, one-time difference between the two Hamiltonians at that separation —
             * physics of the handover (host integrates softened, MSTAR unsoftened), not integration
             * error. Track each particle's tag between calls (this routine runs every step); when a
             * pair's convention flips, accumulate dPhi = phi_new(r) - phi_old(r) into `ledger`.
             * dEcorr = dE_prop - ledger then measures pure integration quality. */
            static double ledger = 0; static int prev_n = 0;
            static unsigned long long *prev_id = NULL; static int *prev_tag = NULL;
            for(i = 0; i < n; i++) {
                double v2=0; if(all[i*W+10] > 0.5) n_int++;
                for(j = 0; j < 3; j++) { v2 += all[i*W+3+j]*all[i*W+3+j]; }
                KE += 0.5*all[i*W+9]*v2;
            }
            /* precompute current-index -> previous-index map once (O(n^2)); O(1) lookups in the pair loop */
            int *cur2prev = (int*)malloc((n > 0 ? n : 1) * sizeof(int));
            for(i = 0; i < n; i++) {
                cur2prev[i] = -1;
                if(prev_id) for(int q = 0; q < prev_n; q++)
                    if(prev_id[q] == (unsigned long long)all[i*W+11]) { cur2prev[i] = q; break; }
            }
            for(i = 0; i < n; i++) for(int b = i+1; b < n; b++) {
                double dr2 = 0; for(j = 0; j < 3; j++){ double d = all[i*W+j]-all[b*W+j]; dr2 += d*d; }
                double r = sqrt(dr2), Gmm = All.G * all[i*W+9] * all[b*W+9];
                PE += -Gmm / r;   /* unsoftened metric (backward-compatible column) */
                /* PROPER metric = the Hamiltonian the run actually conserves: kernel-softened potential
                 * for every pair (as the tree applies), EXCEPT unsoftened for chain-internal pairs when
                 * the run integrates chains unsoftened (KetjuUseStarStarSoftening=0). */
                double h = (all[i*W+6] > all[b*W+6]) ? all[i*W+6] : all[b*W+6];
                double hinv = (h > 0) ? 1.0/h : 0;
                double pe_soft = (r >= h || h <= 0) ? (-Gmm / r) : Gmm * kernel_gravity(r*hinv, hinv, hinv*hinv*hinv, -1);
                int same_chain = (all[i*W+7] > 0.5) && (fabs(all[i*W+7]-all[b*W+7]) < 0.5);
                int unsoft_now = (same_chain && (all[i*W+8] > 0.5));
                PE_prop += unsoft_now ? (-Gmm / r) : pe_soft;
                /* ledger: compare this pair's convention to the previous step's (matched by ID) */
                if(prev_id && r < h) {  /* conventions only differ inside the kernel */
                    int pi = cur2prev[i], pb = cur2prev[b];
                    if(pi >= 0 && pb >= 0) {
                        int same_prev = (prev_tag[pi] > 0) && (prev_tag[pi] == prev_tag[pb]);
                        int unsoft_prev = (same_prev && (all[i*W+8] > 0.5));
                        if(unsoft_now != unsoft_prev) {
                            double pe_unsoft = -Gmm / r;
                            ledger += unsoft_now ? (pe_unsoft - pe_soft) : (pe_soft - pe_unsoft);
                        }
                    }
                }
            }
            /* store this step's tags for the next call */
            if(!prev_id) { prev_id = (unsigned long long*)malloc(4096*sizeof(unsigned long long)); prev_tag = (int*)malloc(4096*sizeof(int)); }
            for(i = 0; i < n && i < 4096; i++) { prev_id[i] = (unsigned long long)all[i*W+11]; prev_tag[i] = (int)(all[i*W+7] + 0.5); }
            prev_n = (n < 4096) ? n : 4096;
            free(cur2prev);

            double E = KE + PE, E_prop = KE + PE_prop;
            /* baseline E0 taken on the first FULL step only — an unambiguous, phase-consistent reference */
            static double E0_prop = 0; if(!have0 && full_sync) { E0 = E; E0_prop = E_prop; have0 = 1; }
            if(do_print) {
                printf("ETOT_INCODE: t=%.6f E=%.16g dE/|E0|=%.6e dEprop/|E0p|=%.6e dEcorr/|E0p|=%.6e ledger=%.6g | n_int=%d KE=%.6g PE=%.6g PEp=%.6g\n",
                       All.Time, E, (E - E0)/fabs(E0), (E_prop - E0_prop)/fabs(E0_prop),
                       (E_prop - E0_prop - ledger)/fabs(E0_prop), ledger, n_int, KE, PE, PE_prop);
                fflush(stdout);
            }
        }
        free(all); free(cnt); free(disp);
    }
    have0 = 1; if(do_print) last_t = All.Time;  /* keep print throttle in sync on all ranks */
}

#if defined(KETJU_REGULARIZATION) && defined(KETJU_HANDOFF_TRACE)
/* velocity watchpoint: print whenever the watched particle's velocity changes between checkpoints */
static void htrace_velwatch(const char *where)
{
    const unsigned long long WATCH_ID = 46;
    static double last[3] = {0,0,0}; static int have = 0;
    for(int i = 0; i < NumPart; i++) {
        if((unsigned long long)P[i].ID != WATCH_ID) continue;
        if(!have || P[i].Vel[0] != last[0] || P[i].Vel[1] != last[1] || P[i].Vel[2] != last[2]) {
            printf("VELWATCH t=%.10f id=%llu at[%s] vel=%.10g,%.10g,%.10g grav=%.6g,%.6g,%.6g tag=%d\n",
                   All.Time, WATCH_ID, where, P[i].Vel[0], P[i].Vel[1], P[i].Vel[2],
                   P[i].GravAccel[0], P[i].GravAccel[1], P[i].GravAccel[2], P[i].KetjuRegionTag);
            fflush(stdout);
            last[0]=P[i].Vel[0]; last[1]=P[i].Vel[1]; last[2]=P[i].Vel[2]; have = 1;
        }
        break;
    }
}
#define HTRACE_VELWATCH(w) htrace_velwatch(w)
#else
#define HTRACE_VELWATCH(w)
#endif


/*! This routine contains the main simulation loop that iterates over
 * single timesteps. The loop terminates when the cpu-time limit is
 * reached, when a `stop' file is found in the output directory, or
 * when the simulation ends because we arrived at TimeMax.
 */
void run(void)
{
    CPU_Step[CPU_MISC] += measure_time();

#ifdef GALSF_RESOLVEDISM_DUST_SELFTEST
    /* dump the resolvedism dust physics table (real table yields -> condensation/
     * destruction/rates) and exit cleanly; a python checker validates it. */
    resolvedism_dust_selftest();
    MPI_Barrier(MPI_COMM_WORLD);
    endrun(0);
#endif

    if(RestartFlag != 1)		/* need to compute forces at initial synchronization time, unless we restarted from restart files */
    {
        output_log_messages();

        domain_Decomposition(0, 0, 0);

        set_non_standard_physics_for_current_time();

#ifdef KETJU_REGULARIZATION
        ketju_tag_regions();  /* tag chain members BEFORE this first gravity so member<->member forces are
                               * excluded from the very first GravAccel that seeds MSTAR (else the full
                               * internal force is injected once as a spurious external kick) */
#endif
        compute_grav_accelerations();	/* compute gravitational accelerations for synchronous particles */

        compute_hydro_densities_and_forces();	/* densities, gradients, & hydro-accels for synchronous particles */

        calculate_non_standard_physics();	/* source terms are here treated in a strang-split fashion */
    }

    while(1)			/* main timestep iteration loop */
    {
        incode_total_energy();	/* DIAGNOSTIC: true KETJU-aware energy at sync point (prints on full syncs) */
        HTRACE_VELWATCH("loop_top");
        compute_statistics();	/* regular statistics outputs (like total energy) */

        write_cpu_log();		/* output some CPU usage log-info (accounts for everything needed up to the current sync-point) */

        if((All.Ti_Current >= TIMEBASE) || (All.Time > All.TimeMax)) /* check whether we reached the final time */
        {
            if(ThisTask == 0) {printf("\nFinal time=%g reached. Simulation ends.\n", All.TimeMax);}
            restart(0); /* write a restart file to allow continuation of the run for a larger value of TimeMax */
            if(All.Ti_lastoutput != All.Ti_Current) {savepositions(All.SnapshotFileCount++);} /* make a snapshot at the final time in case none has produced at this time; this will be overwritten if All.TimeMax is increased and the run is continued */
            break;
        }

        find_timesteps();		/* find-timesteps */
#ifdef KETJU_REGULARIZATION
        ketju_limit_timesteps();    /* force chain particles to shared timebin */
#endif
        int TreeReconstructFlag_local = TreeReconstructFlag;
#ifdef HERMITE_INTEGRATION
        HermiteOnlyFlag = 1;
        gravity_tree();	/* re-compute gravitational accelerations for synchronous particles */
        HermiteOnlyFlag = 0;
#endif
        do_first_halfstep_kick();	/* half-step kick at beginning of timestep for synchronous particles */


#ifdef KETJU_REGULARIZATION
        ketju_find_regions();       /* detect chain regions around massive stars/BHs */
        ketju_run_integration();    /* subtract tree force, run MSTAR, apply velocity trick */
        ketju_write_output();       /* write KETJU diagnostics to HDF5 */
#endif

        find_next_sync_point_and_drift();	/* find next synchronization point and drift particles to this time.
                                             * If needed, this function will also write an output file
                                             * at the desired time.
                                             */


        output_log_messages();	/* write some info to log-files */

        set_non_standard_physics_for_current_time();	/* update auxiliary physics for current time */

        int reconstructed_tree = 0;
#if defined(SINGLE_STAR_SINK_DYNAMICS)
        if(All.NumForcesSinceLastDomainDecomp > All.TreeDomainUpdateFrequency * All.TotNumPart) {TreeReconstructFlag_local = 1;}
#endif
        MPI_Allreduce(&TreeReconstructFlag_local, &TreeReconstructFlag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD); // if one process reconstructs the tree then everbody has to
        if(GlobNumForceUpdate > All.TreeDomainUpdateFrequency * All.TotNumPart)	/* check whether we have a big step */
        {
            domain_Decomposition(0, 0, 1);      /* do domain decomposition if step is big enough, and set new list of active particles  */
            reconstructed_tree = 1;
#ifdef KETJU_REGULARIZATION
            ketju_mark_regions_stale();  /* particle indices changed — invalidate cached KETJU communicators */
#endif
        }
        else if(TreeReconstructFlag) {
            domain_Decomposition(0, 0, 1); reconstructed_tree = 1;
#ifdef KETJU_REGULARIZATION
            ketju_mark_regions_stale();
#endif
        }
        else
        {
            force_update_tree();	/* update tree dynamically with kicks of last step so that it can be reused */
            make_list_of_active_particles();	/* now we can set the new chain list of active particles */
        }


#ifdef GALSF_RESOLVEDISM
        /* Log resolved-ISM SFR at domain decomposition steps */
        if(reconstructed_tree) {
            double windows_myr[] = {50.0, 40.0, 30.0, 20.0, 5.0, 1.0};
            int nwin = 6;
            double local_mass[6] = {0,0,0,0,0,0};
            int local_nstars = 0;
            for(int ii = 0; ii < NumPart; ii++) {
                if(P[ii].Type != 4 || P[ii].Mass <= 0) continue;
                local_nstars++;
                double age_gyr = evaluate_stellar_age_Gyr(ii);
                double age_myr = age_gyr * 1e3;
                double Mstar_solar = 0;
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
                Mstar_solar = P[ii].MstarSampleIMF[0];
                if(Mstar_solar <= 0) Mstar_solar = P[ii].Mass * UNIT_MASS_IN_SOLAR;
#else
                Mstar_solar = P[ii].Mass * UNIT_MASS_IN_SOLAR;
#endif
                for(int w = 0; w < nwin; w++) {
                    if(age_myr <= windows_myr[w]) local_mass[w] += Mstar_solar;
                }
            }
            double glob_mass[6]; int glob_nstars;
            MPI_Reduce(local_mass, glob_mass, 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_nstars, &glob_nstars, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            if(ThisTask == 0) {
                fprintf(FdSFRrism, "%12.6f  %8d", All.Time, glob_nstars);
                for(int w = 0; w < nwin; w++) {
                    double sfr_w = (windows_myr[w] > 0) ? glob_mass[w] / (windows_myr[w] * 1e6) : 0;
                    fprintf(FdSFRrism, "  %12.6e", sfr_w);
                }
                fprintf(FdSFRrism, "\n"); fflush(FdSFRrism);
            }
        }
#endif


        compute_grav_accelerations();	/* compute gravitational accelerations for synchronous particles */


#ifdef GALSF_SUBGRID_WINDS
#if (GALSF_SUBGRID_WIND_SCALING==2)
/*
#ifdef PMGRID
        //if(All.Ti_Current == All.PM_Ti_endstep && get_random_number(1+All.Ti_Current) < 0.05) // compute the DM velocity dispersion around gas particles every 20 PM steps, should be sufficient ? not ideal for many applications, in fact, now only acts on active //
#else
        //if(All.HighestActiveTimeBin == All.HighestOccupiedTimeBin) // only acts on top-level timebin -- only enable this if you are trying to radically reduce the number of operations of this mode //
#endif
*/
        {
            disp_density();
        }
#endif
#endif

        /* flag particles which will be feedback centers, so kernel lengths can be computed for them */
#ifdef GALSF_FB_MECHANICAL
        determine_where_SNe_occur(); // for mechanical FB models
#endif
#ifdef GALSF_RESOLVEDISM_FB
        resolvedism_determine_SNe(); // resolved ISM SN event flagging
        /* Injection happens later in compute_stellar_feedback() (called from
           compute_hydro_densities_and_forces()), AFTER density() refreshes
           DensityAroundParticle. Mirrors FIRE's pattern; avoids stale wt_sum. */
#endif
#ifdef GALSF_FB_THERMAL
        determine_where_addthermalFB_events_occur(); // (same, but for simple thermal feedback models)
#endif

        compute_hydro_densities_and_forces();	/* densities, gradients, & hydro-accels for synchronous particles */

        
#ifdef PARTICLE_MERGE_SPLIT_EVERY_TIMESTEP // do merge/split routines every single timestep - need to do it here if we didn't do it during domain decomp on a coarse timestep
        if(!reconstructed_tree)
        {
            MBARY_STEP("pre_merge_split");
            merge_and_split_particles();
            rearrange_particle_sequence();
            MBARY_STEP("post_merge_split");
        }
#endif
        
        do_second_halfstep_kick();	/* this does the half-step kick at the end of the timestep */


        calculate_non_standard_physics();	/* source terms are here treated in a strang-split fashion */


#ifdef HERMITE_INTEGRATION // we do a prediction step using the saved "old" pos, accel and jerk from the beginning of the timestep. Then we recompute accel and jerk and do the correction
        do_hermite_prediction();
        HermiteOnlyFlag = 2;
        gravity_tree();	/* re-compute gravitational accelerations for synchronous particles */
        HermiteOnlyFlag = 0;
        do_hermite_correction();
#endif
#ifdef KETJU_REGULARIZATION
        ketju_finish_step();        /* clean up KETJU region data (flags persist for next step's guards) */
#endif
        /* Check whether we need to interrupt the run */
        int stopflag = 0;
        if(ThisTask == 0)
        {
            FILE *fd;
            char stopfname[DEFAULT_PATH_BUFFERSIZE_TOUSE];
            snprintf(stopfname, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%sstop", All.OutputDir);
            if((fd = fopen(stopfname, "r")))	/* Is the stop-file present? If yes, interrupt the run. */
            {
                fclose(fd);
                stopflag = 1;
                unlink(stopfname);
            }

            if(CPUThisRun > 0.85 * All.TimeLimitCPU)	/* are we running out of CPU-time ? If yes, interrupt run. */
            {
                printf("reaching time-limit. stopping.\n");
                stopflag = 2;
            }
        }

        MPI_Bcast(&stopflag, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if(stopflag)
        {
            restart(0);		/* write restart file */
            MPI_Barrier(MPI_COMM_WORLD);

            if(stopflag == 2 && ThisTask == 0)
            {
                FILE *fd;
                char contfname[DEFAULT_PATH_BUFFERSIZE_TOUSE];
                snprintf(contfname, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%scont", All.OutputDir);
                if((fd = fopen(contfname, "w")))
                    fclose(fd);

                if(All.ResubmitOn)
                    execute_resubmit_command();
            }
            return;
        }

        if(ThisTask == 0)
        {
            /* is it time to write one of the regularly space restart-files? */
            if((CPUThisRun - All.TimeLastRestartFile) >= All.CpuTimeBetRestartFile)
            {
                All.TimeLastRestartFile = CPUThisRun;
                stopflag = 3;
            }
            else
                stopflag = 0;
        }

        MPI_Bcast(&stopflag, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if(stopflag == 3)
        {
            restart(0);		/* write an occasional restart file */
            stopflag = 0;
            All.TimeLastRestartFile += report_time();
        }

        report_memory_usage(&HighMark_run, "RUN");
    }

}



void set_non_standard_physics_for_current_time(void)
{
#if defined(COOLING) && !defined(CHIMES)
    /* set UV background for the current time */
    IonizeParams();
#endif

#if defined(COOL_METAL_LINES_BY_SPECIES) && !defined(CHIMES)
    /* load the metal-line cooling tables appropriate for the UV background */
    if(All.ComovingIntegrationOn) {LoadMultiSpeciesTables();}
#endif

#if defined(GALSF_SFR_IMF_SAMPLING_DISTRIBUTE_SF)
    update_stellarnumber_and_timedistribofstarformation();
#endif
}



void calculate_non_standard_physics(void)
{
#ifdef PARTICLE_EXCISION
    apply_excision();
#endif


#if defined(TURB_DRIVING) && defined(TURB_DRIVING_SPECTRUMGRID)
    if(All.Time >= All.TimeNextTurbSpectrum) {powerspec_turb(All.FileNumberTurbSpectrum++); All.TimeNextTurbSpectrum += All.TimeBetTurbSpectrum;}
#endif


#ifdef SINK_PARTICLES /***** sink accretion and feedback *****/
    CPU_Step[CPU_MISC] += measure_time();
#ifdef GALSF_LIMIT_FBTIMESTEPS_FROM_BELOW
    if(All.Dt_Since_LastFBCalc_Gyr >= All.Dt_Min_Between_FBCalc_Gyr)
#endif
    {
        sink_accretion();
#ifdef SINK_WIND_SPAWN
        double Max_Unspawned_MassUnits_fromSink_global;
        MPI_Allreduce(&Max_Unspawned_MassUnits_fromSink, &Max_Unspawned_MassUnits_fromSink_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if(Max_Unspawned_MassUnits_fromSink_global > 1)
        {
            spawn_sink_wind_feedback();
            rearrange_particle_sequence();
            Max_Unspawned_MassUnits_fromSink=Max_Unspawned_MassUnits_fromSink_global=0.;
        }
#if defined(SNE_NONSINK_SPAWN)
        {int i; for(i=0;i<NumPart;i++) {if(P[i].Type != 4) {continue;}
            double n_unspawned = P[i].unspawned_wind_mass / ((SINK_WIND_SPAWN)*target_mass_for_wind_spawning(i)); // number of spawned gas cells that can be made from the mass in the reservoir
            if(n_unspawned> Max_Unspawned_MassUnits_fromSink) {Max_Unspawned_MassUnits_fromSink = n_unspawned;} // track the maximum integer number of elements this sink could spawn
        }}
#endif
#endif
        MPI_Barrier(MPI_COMM_WORLD); CPU_Step[CPU_SINKS] += measure_time();
    }
#endif


#if (defined(SINK_PARTICLES) || defined(GALSF_SUBGRID_WINDS)) && defined(FOF)
    if(All.Time >= All.TimeNextOnTheFlyFoF) {fof_fof(-1); /* this will find new sink seed halos and/or assign host halo masses for the variable wind model */
        if(All.ComovingIntegrationOn) {All.TimeNextOnTheFlyFoF *= All.TimeBetOnTheFlyFoF;} else {All.TimeNextOnTheFlyFoF += All.TimeBetOnTheFlyFoF;}}
#endif

#ifdef RADTRANSFER
    CPU_Step[CPU_MISC] += measure_time();
#if defined(RT_SOURCE_INJECTION)
    int flag; flag=1;
#if !defined(RT_INJECT_PHOTONS_DISCRETELY)
    flag = Flag_FullStep; /* for continous injection, requires all sources and gas be active synchronously or else 2x-counts */
#endif
#if !defined(GRAIN_RDI_TESTPROBLEM_LIVE_RADIATION_INJECTION)
    if(flag) {rt_source_injection();} /* source injection into neighbor gas particles (only on full timesteps, if using non-discrete scheme) */
#endif
#endif
#if defined(RT_DIFFUSION_CG) /* use the CG method to solve the RT diffusion equation implicitly for all particles; do only on full timesteps, requires synchronous timestepping right now */
    if(Flag_FullStep) {All.Radiation_Ti_endstep = All.Ti_Current; rt_diffusion_cg_solve(); All.Radiation_Ti_begstep = All.Radiation_Ti_endstep;}
#endif
#if defined(RT_CHEM_PHOTOION) && !defined(COOLING)
    rt_update_chemistry(); /* chemistry updated at sub-stepping as well */
#ifdef OUTPUT_ADDITIONAL_RUNINFO
    if(Flag_FullStep) {rt_write_chemistry_stats();}
#endif
#endif
    MPI_Barrier(MPI_COMM_WORLD); CPU_Step[CPU_RTNONFLUXOPS] += measure_time();
#endif // RADTRANSFER block

    MCBAL_LOG("pre_photoion");
#ifdef GALSF_RESOLVEDISM_PHOTOION
    resolvedism_photoionize(); // resolved ISM Stromgren sphere photo-ionization
#endif

    MCBAL_LOG("pre_cooling");
#ifdef COOLING	/* radiative cooling and chemistry  */
    cooling_parent_routine(); // top-level cooling and chemistry subroutine //
    MPI_Barrier(MPI_COMM_WORLD); CPU_Step[CPU_COOLINGSFR] += measure_time(); // finish time calc for SFR+cooling
#endif
    MCBAL_LOG("post_cooling");
#ifdef GALSF_RESOLVEDISM_DUST
    resolvedism_dust_evolve(); // ISM dust sputtering (operator-split after chemistry)
#endif


#ifdef GALSF /* star/sink particle formation */
    MCBAL_LOG("pre_starform_FB");
    MBARY_STEP("pre_sf_spawn");
    star_formation_parent_routine(); // top-level star formation routine (because this involves common particle conversions, want to keep this at end of this subroutine) //
    MBARY_STEP("post_sf_spawn");
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
    MBARY_STEP("pre_imf_accrete");
    assign_stellar_masses(); // sample individual stellar masses from Kroupa IMF for newly formed star particles
    MBARY_STEP("post_imf_accrete");
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
    recompute_resolvedism_fuv_luminosities(); // BUGFIX: re-evaluate stellar FUV/LW at current age each step (else stuck at PMS-zero -> G0 floor)
#endif
#endif
    MPI_Barrier(MPI_COMM_WORLD); CPU_Step[CPU_COOLINGSFR] += measure_time(); // finish time calc for SFR+cooling
    MCBAL_LOG("post_starform_FB");
#endif

#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
    int i; for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i]){if(P[i].Type == 5 && P[i].do_gas_search_this_timestep){P[i].dt_since_last_gas_search = 0;}}
#endif

}



void compute_statistics(void)
{
    if((All.Time - All.TimeLastStatistics) >= All.TimeBetStatistics)
    {
#if !defined(EVALPOTENTIAL)          // compute_potential is not defined if EVALPOTENTIAL is on //
#ifdef COMPUTE_POTENTIAL_ENERGY
        compute_potential();
#endif
#endif
        energy_statistics();	/* compute and output energy statistics */
        All.TimeLastStatistics += All.TimeBetStatistics;
    }
}



void execute_resubmit_command(void)
{
    char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE];
    snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s", All.ResubmitCommand);
    system(buf);
}



/*! This function finds the next synchronization point of the system
 * (i.e. the earliest point of time any of the particles needs a force
 * computation), and drifts the system to this point of time.  If the
 * system drifts over the desired time of a snapshot file, the
 * function will drift to this moment, generate an output, and then
 * resume the drift.
 */
void find_next_sync_point_and_drift(void)
{
  int n, i, prev;
  integertime dt_bin, ti_next_for_bin, ti_next_kick, ti_next_kick_global;
  int highest_active_bin, highest_occupied_bin;
  double timeold;

  timeold = All.Time;

  All.NumCurrentTiStep++;	/* we are now moving to the next sync point */

  /* find the next kick time */
  for(n = 0, ti_next_kick = TIMEBASE, highest_occupied_bin = 0; n < TIMEBINS; n++)
    {
      if(TimeBinCount[n])
	{
	  if(n > 0)
	    {
	      highest_occupied_bin = n;
	      dt_bin = GET_INTEGERTIME_FROM_TIMEBIN(n);
	      ti_next_for_bin = (All.Ti_Current / dt_bin) * dt_bin + dt_bin;	/* next kick time for this timebin */
	    }
	  else
	    {
	      dt_bin = 0;
	      ti_next_for_bin = All.Ti_Current;
	    }

	  if(ti_next_for_bin < ti_next_kick)
	    ti_next_kick = ti_next_for_bin;
	}
    }

  MPI_Allreduce(&ti_next_kick, &ti_next_kick_global, 1, MPI_TYPE_TIME, MPI_MIN, MPI_COMM_WORLD);

  while(ti_next_kick_global >= All.Ti_nextoutput && All.Ti_nextoutput >= 0)
    {
        All.Ti_Current = All.Ti_nextoutput;

        if(All.ComovingIntegrationOn) {All.Time = All.TimeBegin * exp(All.Ti_Current * All.Timebase_interval);}
            else {All.Time = All.TimeBegin + All.Ti_Current * All.Timebase_interval;}

        set_cosmo_factors_for_current_time();

        move_particles(All.Ti_nextoutput);
        MPI_Barrier(MPI_COMM_WORLD); CPU_Step[CPU_DRIFT] += measure_time();

#ifdef OUTPUT_POTENTIAL
#if !defined(EVALPOTENTIAL) || (defined(EVALPOTENTIAL) && defined(OUTPUT_RECOMPUTE_POTENTIAL))
        domain_Decomposition(0, 0, 0);
        compute_potential();
#endif
#endif

        savepositions(All.SnapshotFileCount++);	/* write snapshot file */
        All.Ti_nextoutput = find_next_outputtime(All.Ti_nextoutput + 1);
    }


  All.Previous_Ti_Current = All.Ti_Current;
  All.Ti_Current = ti_next_kick_global;

  if(All.ComovingIntegrationOn) {All.Time = All.TimeBegin * exp(All.Ti_Current * All.Timebase_interval);}
    else {All.Time = All.TimeBegin + All.Ti_Current * All.Timebase_interval;}

  set_cosmo_factors_for_current_time();
#ifdef BOX_SHEARING
    calc_shearing_box_pos_offset();
#endif

#ifdef GR_TABULATED_COSMOLOGY_G
  All.G = All.Gini * dGfak(All.Time);
#endif

  All.TimeStep = All.Time - timeold;

  /* mark the bins that will be active */
  for(n = 1, TimeBinActive[0] = 1, NumForceUpdate = TimeBinCount[0], highest_active_bin = 0; n < TIMEBINS; n++)
    {
      dt_bin = GET_INTEGERTIME_FROM_TIMEBIN(n);
      if((ti_next_kick_global % dt_bin) == 0)
	{
	  TimeBinActive[n] = 1;
	  NumForceUpdate += TimeBinCount[n];
	  if(TimeBinCount[n])
	    highest_active_bin = n;
	}
      else
	TimeBinActive[n] = 0;
    }

  sumup_large_ints(1, &NumForceUpdate, &GlobNumForceUpdate);
  All.NumForcesSinceLastDomainDecomp += GlobNumForceUpdate;
  MPI_Allreduce(&highest_active_bin, &All.HighestActiveTimeBin, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&highest_occupied_bin, &All.HighestOccupiedTimeBin, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  if(GlobNumForceUpdate == All.TotNumPart)
    {
      Flag_FullStep = 1;
      if(All.HighestActiveTimeBin != All.HighestOccupiedTimeBin)
	terminate("Something is wrong with the time bins.\n");
    }
  else
    Flag_FullStep = 0;




  /* move the new set of active/synchronized particles. Note: We do not yet call make_list_of_active_particles(), since we
   * may still need to old list in the dynamic tree update */
  for(n = 0, prev = -1; n < TIMEBINS; n++)
    {if(TimeBinActive[n]) {for(i = FirstInTimeBin[n]; i >= 0; i = NextInTimeBin[i]) {drift_particle(i, All.Ti_Current);}}}

#ifdef KETJU_REGULARIZATION
  ketju_set_final_velocities(); /* swap in true physical velocities after drift (velocity trick) */
#endif

}


void make_list_of_active_particles(void)
{
    int i, n, prev;
    /* make a link list with the particles in the active time bins */
    FirstActiveParticle = -1;
    ActiveParticleNumber = 0;

    for(n = 0, prev = -1; n < TIMEBINS; n++)
    {
        if(TimeBinActive[n])
        {
            for(i = FirstInTimeBin[n]; i >= 0; i = NextInTimeBin[i])
            {
                if(P[i].Mass <= 0) {continue;}
                if(prev == -1) {FirstActiveParticle = i;}
                if(prev >= 0) {NextActiveParticle[prev] = i;}
                prev = i;
                ActiveParticleList[ActiveParticleNumber] = i;
                ActiveParticleNumber++;
            }
        }
    }

    if(prev >= 0) {NextActiveParticle[prev] = -1;}
}





/*! this function returns the next output time that is equal or larger to
 *  ti_curr
 */
integertime find_next_outputtime(integertime ti_curr)
{
  long long i, iter = 0;
  integertime ti, ti_next;
  double next, time;

  DumpFlag = 1;
  ti_next = -1;


  if(All.OutputListOn)
    {
      for(i = 0; i < All.OutputListLength; i++)
	{
	  time = All.OutputListTimes[i];

	  if(time >= All.TimeBegin && time <= All.TimeMax)
	    {
	      if(All.ComovingIntegrationOn) {ti = (integertime) (log(time / All.TimeBegin) / All.Timebase_interval);}
          else {ti = (integertime) ((time - All.TimeBegin) / All.Timebase_interval);}

	      if(ti >= ti_curr)
		{
		  if(ti_next == -1)
		    {
		      ti_next = ti;
		      DumpFlag = All.OutputListFlag[i];
		      if(i > All.SnapshotFileCount) {All.SnapshotFileCount = i;}
		    }

		  if(ti_next > ti)
		    {
		      ti_next = ti;
		      DumpFlag = All.OutputListFlag[i];
		      if(i > All.SnapshotFileCount) {All.SnapshotFileCount = i;}
		    }
		}
	    }
	}
    }
  else
    {
      if(All.ComovingIntegrationOn)
	{
	  if(All.TimeBetSnapshot <= 1.0)
	    {
	      printf("TimeBetSnapshot > 1.0 required for your simulation.\n");
	      endrun(13123);
	    }
	}
      else
	{
	  if(All.TimeBetSnapshot <= 0.0)
	    {
	      printf("TimeBetSnapshot > 0.0 required for your simulation.\n");
	      endrun(13123);
	    }
	}
      time = All.TimeOfFirstSnapshot;

      iter = 0;

      while(time < All.TimeBegin)
	{
	  if(All.ComovingIntegrationOn)
	    time *= All.TimeBetSnapshot;
	  else
	    time += All.TimeBetSnapshot;

	  iter++;

	  if(iter > 10000000000)
	    {
          printf("Can't determine next output time. iter=%lld time=%g All.TimeBegin=%g All.TimeBetSnapshot=%g All.TimeOfFirstSnapshot=%g \n",iter,time,All.TimeBegin,All.TimeBetSnapshot,All.TimeOfFirstSnapshot);
	      endrun(110);
	    }
	}
      while(time <= All.TimeMax)
	{
	  if(All.ComovingIntegrationOn) {ti = (integertime) (log(time / All.TimeBegin) / All.Timebase_interval);}
        else {ti = (integertime) ((time - All.TimeBegin) / All.Timebase_interval);}

	  if(ti >= ti_curr)
	    {
	      ti_next = ti;
	      break;
	    }

	  if(All.ComovingIntegrationOn)
	    time *= All.TimeBetSnapshot;
	  else
	    time += All.TimeBetSnapshot;

	  iter++;

	  if(iter > 10000000000)
	    {
          printf("Can't determine next output time. iter=%lld time=%g All.TimeBegin=%g All.TimeMax=%g All.TimeBetSnapshot=%g All.TimeOfFirstSnapshot=%g All.Timebase_interval=%g \n",iter,time,All.TimeBegin,All.TimeMax,All.TimeBetSnapshot,All.TimeOfFirstSnapshot,All.Timebase_interval);
	      endrun(111);
	    }
	}
    }


  if(ti_next == -1)
    {
      ti_next = 2 * TIMEBASE;	/* this will prevent any further output */
      if(ThisTask == 0) {printf("\nThere is no valid time for a further snapshot file.\n");}
    }
  else
    {
      if(All.ComovingIntegrationOn) {next = All.TimeBegin * exp(ti_next * All.Timebase_interval);}
      else {next = All.TimeBegin + ti_next * All.Timebase_interval;}

      if(ThisTask == 0) {printf("\nSetting next time for snapshot file to Time_next= %.16g  (DumpFlag=%d)\n", next, DumpFlag);}

    }

  return ti_next;
}




/*! This routine writes for every synchronisation point in the timeline information to two log-files:
 * In FdInfo, we just list the timesteps that have been done, while in
 * FdTimebins we inform about the distribution of particles over the timebins, and which timebins are active on this step.
 * code is stored.
 */
void output_log_messages(void)
{
  double z;
  int i, j;
  long long tot, tot_gas;
  long long tot_count[TIMEBINS];
  long long tot_count_gas[TIMEBINS];
  long long tot_cumulative[TIMEBINS];
  int weight, corr_weight;
  double sum, avg_CPU_TimeBin[TIMEBINS], frac_CPU_TimeBin[TIMEBINS];

  sumup_large_ints(TIMEBINS, TimeBinCount, tot_count);
  sumup_large_ints(TIMEBINS, TimeBinCountGas, tot_count_gas);

#if defined(IO_SUPPRESS_TIMEBIN_STDOUT)
    if((ThisTask == 0) && (All.HighestActiveTimeBin>=(TIMEBINS-IO_SUPPRESS_TIMEBIN_STDOUT)))
#else
    if(ThisTask == 0)
#endif
    {
        if(All.ComovingIntegrationOn)
        {
            z = 1.0 / (All.Time) - 1;
#ifdef OUTPUT_ADDITIONAL_RUNINFO
            fprintf(FdInfo, "Sync-Point %lld, Time: %.16g, Redshift: %g, Nf = %d%09d, Systemstep: %g, Dloga: %g\n",
                    (long long) All.NumCurrentTiStep, All.Time, z, (int) (GlobNumForceUpdate / 1000000000), (int) (GlobNumForceUpdate % 1000000000), All.TimeStep, log(All.Time) - log(All.Time - All.TimeStep));
            fflush(FdInfo);
            fprintf(FdTimebin, "Sync-Point %lld, Time: %.16g, Redshift: %g, Systemstep: %g, Dloga: %g\n", (long long) All.NumCurrentTiStep, All.Time, z, All.TimeStep, log(All.Time) - log(All.Time - All.TimeStep));
#endif
            printf("\nSync-Point %lld, Time: %.16g, Redshift: %g, Systemstep: %g, Dloga: %g\n", (long long) All.NumCurrentTiStep, All.Time, z, All.TimeStep, log(All.Time) - log(All.Time - All.TimeStep));
        }
        else
        {
#ifdef OUTPUT_ADDITIONAL_RUNINFO
            fprintf(FdInfo, "Sync-Point %lld, Time: %.16g, Nf = %d%09d, Systemstep: %g\n", (long long) All.NumCurrentTiStep,
                    All.Time, (int) (GlobNumForceUpdate / 1000000000), (int) (GlobNumForceUpdate % 1000000000), All.TimeStep);
            fflush(FdInfo);
            fprintf(FdTimebin, "Sync-Point %lld, Time: %.16g, Systemstep: %g\n", (long long) All.NumCurrentTiStep, All.Time, All.TimeStep);
#endif
            printf("\nSync-Point %lld, Time: %.16g, Systemstep: %g\n", (long long) All.NumCurrentTiStep, All.Time, All.TimeStep);
        }

        for(i = 1, tot_cumulative[0] = tot_count[0]; i < TIMEBINS; i++) {tot_cumulative[i] = tot_count[i] + tot_cumulative[i - 1];}


      for(i = 0; i < TIMEBINS; i++)
	{
	  for(j = 0, sum = 0; j < All.CPU_TimeBinCountMeasurements[i]; j++) {sum += All.CPU_TimeBinMeasurements[i][j];}
	  if(All.CPU_TimeBinCountMeasurements[i]) {avg_CPU_TimeBin[i] = sum / All.CPU_TimeBinCountMeasurements[i];} else {avg_CPU_TimeBin[i] = 0;}
	}

      for(i = All.HighestOccupiedTimeBin, weight = 1, sum = 0; i >= 0 && tot_count[i] > 0; i--, weight *= 2)
	{
	  if(weight > 1) {corr_weight = weight / 2;} else {corr_weight = weight;}
	  frac_CPU_TimeBin[i] = corr_weight * avg_CPU_TimeBin[i];
	  sum += frac_CPU_TimeBin[i];
	}

      for(i = All.HighestOccupiedTimeBin; i >= 0 && tot_count[i] > 0; i--) {if(sum) {frac_CPU_TimeBin[i] /= sum;}}


        printf("Occupied timebins: non-cells     cells       dt                 cumulative A D    avg-time  cpu-frac\n");
#ifdef OUTPUT_ADDITIONAL_RUNINFO
        fprintf(FdTimebin,"Occupied timebins: non-cells     cells       dt                 cumulative A D    avg-time  cpu-frac\n");
#endif
        for(i = TIMEBINS - 1, tot = tot_gas = 0; i >= 0; i--)
            if(tot_count_gas[i] > 0 || tot_count[i] > 0)
            {
                printf(" %c  bin=%2d      %10llu  %10llu   %16.12f       %10llu %c %c  %10.2f    %5.1f%%\n", TimeBinActive[i] ? 'X' : ' ', i, tot_count[i] - tot_count_gas[i], tot_count_gas[i],
                       GET_INTEGERTIME_FROM_TIMEBIN(i) * All.Timebase_interval, tot_cumulative[i], (i == All.HighestActiveTimeBin) ? '<' : ' ',
                       (tot_cumulative[i] > All.TreeDomainUpdateFrequency * All.TotNumPart) ? '*' : ' ', avg_CPU_TimeBin[i], 100.0 * frac_CPU_TimeBin[i]);
#ifdef OUTPUT_ADDITIONAL_RUNINFO
                fprintf(FdTimebin," %c  bin=%2d      %10llu  %10llu   %16.12f       %10llu %c %c  %10.2f    %5.1f%%\n", TimeBinActive[i] ? 'X' : ' ', i, tot_count[i] - tot_count_gas[i], tot_count_gas[i],
                        GET_INTEGERTIME_FROM_TIMEBIN(i) * All.Timebase_interval, tot_cumulative[i], (i == All.HighestActiveTimeBin) ? '<' : ' ',
                        (tot_cumulative[i] > All.TreeDomainUpdateFrequency * All.TotNumPart) ? '*' : ' ', avg_CPU_TimeBin[i], 100.0 * frac_CPU_TimeBin[i]);
#endif
                if(TimeBinActive[i])
                {
                    tot += tot_count[i];
                    tot_gas += tot_count_gas[i];
                }
            }
        printf("               ------------------------\n");
#ifdef OUTPUT_ADDITIONAL_RUNINFO
        fprintf(FdTimebin, "               ------------------------\n");
#endif
#ifdef PMGRID
        if(All.PM_Ti_endstep == All.Ti_Current)
        {
            printf("PM-Step. Total: %10llu  %10llu    Sum: %10llu\n\n", tot - tot_gas, tot_gas, tot);
#ifdef OUTPUT_ADDITIONAL_RUNINFO
            fprintf(FdTimebin, "PM-Step. Total: %10llu  %10llu    Sum: %10llu\n", tot - tot_gas, tot_gas, tot);
#endif
        }
        else
#endif
        {
            printf("Total active:   %10llu  %10llu    Sum: %10llu\n\n", tot - tot_gas, tot_gas, tot);
#ifdef OUTPUT_ADDITIONAL_RUNINFO
            fprintf(FdTimebin, "Total active:   %10llu  %10llu    Sum: %10llu\n", tot - tot_gas, tot_gas, tot);
#endif
        }
#ifdef OUTPUT_ADDITIONAL_RUNINFO
        fprintf(FdTimebin, "\n");
        fflush(FdTimebin);
#endif
    }

  output_extra_log_messages();
}




void write_cpu_log(void)
{
  double max_CPU_Step[CPU_PARTS], avg_CPU_Step[CPU_PARTS], t0, t1, tsum; int i; t0=0; t1=0; tsum=0;
  CPU_Step[CPU_MISC] += measure_time();

  for(i = 1, CPU_Step[0] = 0; i < CPU_PARTS; i++) {CPU_Step[0] += CPU_Step[i];}

  MPI_Reduce(CPU_Step, max_CPU_Step, CPU_PARTS, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(CPU_Step, avg_CPU_Step, CPU_PARTS, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if(ThisTask == 0)
    {
      for(i = 0; i < CPU_PARTS; i++) {avg_CPU_Step[i] /= NTask;}

#ifdef OUTPUT_ADDITIONAL_RUNINFO
      put_symbol(0.0, 1.0, '#');
      for(i = 1, tsum = 0.0; i < CPU_PARTS; i++)
      {
            if(max_CPU_Step[i] > 0)
            {
              t0 = tsum; t1 = t0 + avg_CPU_Step[i] * (avg_CPU_Step[i] / max_CPU_Step[i]);
              put_symbol(t0 / avg_CPU_Step[0], t1 / avg_CPU_Step[0], CPU_Symbol[i]);
              tsum += t1 - t0;

              t0 = tsum; t1 = t0 + avg_CPU_Step[i] * ((max_CPU_Step[i] - avg_CPU_Step[i]) / max_CPU_Step[i]);
              put_symbol(t0 / avg_CPU_Step[0], t1 / avg_CPU_Step[0], CPU_SymbolImbalance[i]);
              tsum += t1 - t0;
            }
      }
      put_symbol(tsum / max_CPU_Step[0], 1.0, '-');
      fprintf(FdBalance, "Step=%7lld  sec=%10.3f  Nf=%2d%09d  %s\n", (long long) All.NumCurrentTiStep, max_CPU_Step[0], (int) (GlobNumForceUpdate / 1000000000), (int) (GlobNumForceUpdate % 1000000000), CPU_String); fflush(FdBalance);
#endif

      if(All.CPU_TimeBinCountMeasurements[All.HighestActiveTimeBin] == NUMBER_OF_MEASUREMENTS_TO_RECORD)
	{
	  All.CPU_TimeBinCountMeasurements[All.HighestActiveTimeBin]--;
	  memmove(&All.CPU_TimeBinMeasurements[All.HighestActiveTimeBin][0], &All.CPU_TimeBinMeasurements[All.HighestActiveTimeBin][1], (NUMBER_OF_MEASUREMENTS_TO_RECORD - 1) * sizeof(double));
	}

      All.CPU_TimeBinMeasurements[All.HighestActiveTimeBin][All.CPU_TimeBinCountMeasurements[All.HighestActiveTimeBin]++] = max_CPU_Step[0];
    }

    CPUThisRun += CPU_Step[0];

    for(i = 0; i < CPU_PARTS; i++) {CPU_Step[i] = 0;}
    if(ThisTask == 0)
    {
        for(i = 0; i < CPU_PARTS; i++) {All.CPU_Sum[i] += avg_CPU_Step[i];}
    }

#ifndef OUTPUT_ADDITIONAL_RUNINFO
    if(All.HighestActiveTimeBin == All.HighestOccupiedTimeBin) // only do the actual -print- operation on global timesteps
#endif
  if(ThisTask == 0)
    {
      fprintf(FdCPU, "Step %lld, Time: %.16g, CPUs: %d\n",(long long) All.NumCurrentTiStep, All.Time, NTask);
      fprintf(FdCPU, "Nactive=%lld, Imbal(Max/Mean)=%g \n", (long long) GlobNumForceUpdate, (max_CPU_Step[0]/(MIN_REAL_NUMBER + avg_CPU_Step[0])-1.)*NTask+1.);
      fprintf(FdCPU,
	      "total         %10.2f  %5.1f%%\n"
	      "tree+gravity  %10.2f  %5.1f%%\n"
	      "   treebuild  %10.2f  %5.1f%%\n"
	      "   treewalk   %10.2f  %5.1f%%\n"
	      "   treecomm   %10.2f  %5.1f%%\n"
	      "   treeimbal  %10.2f  %5.1f%%\n"
#ifdef PMGRID
          "pm-gravity    %10.2f  %5.1f%%\n"
#endif
#if !defined(EVALPOTENTIAL) && (defined(COMPUTE_POTENTIAL_ENERGY) || defined(OUTPUT_POTENTIAL))
          "potentialeval %10.2f  %5.1f%%\n"
#endif
#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
	      "ags-nongas    %10.2f  %5.1f%%\n"
	      "   agsdensity %10.2f  %5.1f%%\n"
	      "   agscomm    %10.2f  %5.1f%%\n"
	      "   agsimbal   %10.2f  %5.1f%%\n"
          "   agsmisc    %10.2f  %5.1f%%\n"
#endif
#ifdef TURB_DIFF_DYNAMIC
          "dyndiff       %10.2f  %5.1f%%\n"
          "   compute    %10.2f  %5.1f%%\n"
          "   comm       %10.2f  %5.1f%%\n"
          "   wait       %10.2f  %5.1f%%\n"
          "   misc       %10.2f  %5.1f%%\n"
          "velsmooth     %10.2f  %5.1f%%\n"
          "   compute    %10.2f  %5.1f%%\n"
          "   comm       %10.2f  %5.1f%%\n"
          "   wait       %10.2f  %5.1f%%\n"
          "   misc       %10.2f  %5.1f%%\n"
#endif
	      "hydro/fluids  %10.2f  %5.1f%%\n"
	      "   dens+grad  %10.2f  %5.1f%%\n"
	      "   denscomm   %10.2f  %5.1f%%\n"
	      "   densimbal  %10.2f  %5.1f%%\n"
	      "   hydrofrc   %10.2f  %5.1f%%\n"
	      "   hydcomm    %10.2f  %5.1f%%\n"
	      "   hydimbal   %10.2f  %5.1f%%\n"
	      "   hmaxupdate %10.2f  %5.1f%%\n"
          "   hydmisc    %10.2f  %5.1f%%\n"
	      "domain        %10.2f  %5.1f%%\n"
          "peano         %10.2f  %5.1f%%\n"
#ifdef FOF
          "fof/subfind   %10.2f  %5.1f%%\n"
#endif
          "drift/splitmg %10.2f  %5.1f%%\n"
	      "kicks         %10.2f  %5.1f%%\n"
	      "io/snapshots  %10.2f  %5.1f%%\n"
#ifdef COOLING
	      "cooling+chem  %10.2f  %5.1f%%\n"
#endif
#ifdef CHIMES
	      " coolchmimbal %10.2f  %5.1f%%\n"
#endif
#ifdef SINK_PARTICLES
	      "sinks         %10.2f  %5.1f%%\n"
#endif
#ifdef GRAIN_FLUID
          "grains        %10.2f  %5.1f%%\n"
#endif
#if defined(GALSF_FB_MECHANICAL) || defined(GALSF_FB_THERMAL)
          "mech_fb_loop  %10.2f  %5.1f%%\n"
#endif
#if defined(GALSF_FB_FIRE_RT_HIIHEATING)
          "hII_fb_loop   %10.2f  %5.1f%%\n"
#endif
#if defined(GALSF_FB_FIRE_RT_LOCALRP)
          "localwindkik  %10.2f  %5.1f%%\n"
#endif
#if defined(RADTRANSFER)
          "rt_nonfluxops %10.2f  %5.1f%%\n"
#endif
          "misc          %10.2f  %5.1f%%\n",

    All.CPU_Sum[CPU_ALL], 100.0,
    All.CPU_Sum[CPU_TREEWALK1] + All.CPU_Sum[CPU_TREEWALK2] + All.CPU_Sum[CPU_TREESEND] + All.CPU_Sum[CPU_TREERECV]
              + All.CPU_Sum[CPU_TREEWAIT1] + All.CPU_Sum[CPU_TREEWAIT2] + All.CPU_Sum[CPU_TREEBUILD] + All.CPU_Sum[CPU_TREEMISC],
    (All.CPU_Sum[CPU_TREEWALK1] + All.CPU_Sum[CPU_TREEWALK2] + All.CPU_Sum[CPU_TREESEND] + All.CPU_Sum[CPU_TREERECV]
              + All.CPU_Sum[CPU_TREEWAIT1] + All.CPU_Sum[CPU_TREEWAIT2] + All.CPU_Sum[CPU_TREEBUILD] + All.CPU_Sum[CPU_TREEMISC]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_TREEBUILD], (All.CPU_Sum[CPU_TREEBUILD]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_TREEWALK1] + All.CPU_Sum[CPU_TREEWALK2], (All.CPU_Sum[CPU_TREEWALK1] + All.CPU_Sum[CPU_TREEWALK2]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_TREESEND] + All.CPU_Sum[CPU_TREERECV], (All.CPU_Sum[CPU_TREESEND] + All.CPU_Sum[CPU_TREERECV]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_TREEWAIT1] + All.CPU_Sum[CPU_TREEWAIT2], (All.CPU_Sum[CPU_TREEWAIT1] + All.CPU_Sum[CPU_TREEWAIT2]) / All.CPU_Sum[CPU_ALL] * 100,
#ifdef PMGRID
    All.CPU_Sum[CPU_MESH], (All.CPU_Sum[CPU_MESH]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
#if !defined(EVALPOTENTIAL) && (defined(COMPUTE_POTENTIAL_ENERGY) || defined(OUTPUT_POTENTIAL))
    All.CPU_Sum[CPU_POTENTIAL], (All.CPU_Sum[CPU_POTENTIAL]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
    All.CPU_Sum[CPU_AGSDENSCOMPUTE] + All.CPU_Sum[CPU_AGSDENSWAIT] + All.CPU_Sum[CPU_AGSDENSCOMM] + All.CPU_Sum[CPU_AGSDENSMISC],
              (All.CPU_Sum[CPU_AGSDENSCOMPUTE] + All.CPU_Sum[CPU_AGSDENSWAIT] + All.CPU_Sum[CPU_AGSDENSCOMM] + All.CPU_Sum[CPU_AGSDENSMISC]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_AGSDENSCOMPUTE], (All.CPU_Sum[CPU_AGSDENSCOMPUTE]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_AGSDENSCOMM], (All.CPU_Sum[CPU_AGSDENSCOMM]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_AGSDENSWAIT], (All.CPU_Sum[CPU_AGSDENSWAIT]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_AGSDENSMISC], (All.CPU_Sum[CPU_AGSDENSMISC]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
#ifdef TURB_DIFF_DYNAMIC
    (All.CPU_Sum[CPU_DYNDIFFCOMPUTE] + All.CPU_Sum[CPU_DYNDIFFWAIT] + All.CPU_Sum[CPU_DYNDIFFCOMM] + All.CPU_Sum[CPU_DYNDIFFMISC]), (All.CPU_Sum[CPU_DYNDIFFCOMPUTE] + All.CPU_Sum[CPU_DYNDIFFWAIT] + All.CPU_Sum[CPU_DYNDIFFCOMM] + All.CPU_Sum[CPU_DYNDIFFMISC]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_DYNDIFFCOMPUTE], (All.CPU_Sum[CPU_DYNDIFFCOMPUTE]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_DYNDIFFWAIT], (All.CPU_Sum[CPU_DYNDIFFWAIT]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_DYNDIFFCOMM], (All.CPU_Sum[CPU_DYNDIFFCOMM]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_DYNDIFFMISC], (All.CPU_Sum[CPU_DYNDIFFMISC]) / All.CPU_Sum[CPU_ALL] * 100,
    (All.CPU_Sum[CPU_IMPROVDIFFCOMPUTE] + All.CPU_Sum[CPU_IMPROVDIFFWAIT] + All.CPU_Sum[CPU_IMPROVDIFFCOMM] + All.CPU_Sum[CPU_IMPROVDIFFMISC]), (All.CPU_Sum[CPU_IMPROVDIFFCOMPUTE] + All.CPU_Sum[CPU_IMPROVDIFFWAIT] + All.CPU_Sum[CPU_IMPROVDIFFCOMM] + All.CPU_Sum[CPU_IMPROVDIFFMISC]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_IMPROVDIFFCOMPUTE], (All.CPU_Sum[CPU_IMPROVDIFFCOMPUTE]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_IMPROVDIFFWAIT], (All.CPU_Sum[CPU_IMPROVDIFFWAIT]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_IMPROVDIFFCOMM], (All.CPU_Sum[CPU_IMPROVDIFFCOMM]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_IMPROVDIFFMISC], (All.CPU_Sum[CPU_IMPROVDIFFMISC]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
    All.CPU_Sum[CPU_DENSCOMPUTE] + All.CPU_Sum[CPU_DENSCOMM] + All.CPU_Sum[CPU_DENSWAIT] + All.CPU_Sum[CPU_DENSMISC]
              + All.CPU_Sum[CPU_HYDCOMPUTE] + All.CPU_Sum[CPU_HYDCOMM] + All.CPU_Sum[CPU_HYDMISC]
              + All.CPU_Sum[CPU_HYDWAIT] + All.CPU_Sum[CPU_TREEHMAXUPDATE],
    (All.CPU_Sum[CPU_DENSCOMPUTE] + All.CPU_Sum[CPU_DENSCOMM] + All.CPU_Sum[CPU_DENSWAIT] + All.CPU_Sum[CPU_DENSMISC]
              + All.CPU_Sum[CPU_HYDCOMPUTE] + All.CPU_Sum[CPU_HYDCOMM] + All.CPU_Sum[CPU_HYDMISC]
              + All.CPU_Sum[CPU_HYDWAIT] + All.CPU_Sum[CPU_TREEHMAXUPDATE]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_DENSCOMPUTE], (All.CPU_Sum[CPU_DENSCOMPUTE]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_DENSCOMM], (All.CPU_Sum[CPU_DENSCOMM]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_DENSWAIT], (All.CPU_Sum[CPU_DENSWAIT]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_HYDCOMPUTE], (All.CPU_Sum[CPU_HYDCOMPUTE]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_HYDCOMM], (All.CPU_Sum[CPU_HYDCOMM]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_HYDWAIT], (All.CPU_Sum[CPU_HYDWAIT]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_TREEHMAXUPDATE], (All.CPU_Sum[CPU_TREEHMAXUPDATE]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_HYDMISC] + All.CPU_Sum[CPU_DENSMISC], (All.CPU_Sum[CPU_HYDMISC] + All.CPU_Sum[CPU_DENSMISC]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_DOMAIN], (All.CPU_Sum[CPU_DOMAIN]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_PEANO], (All.CPU_Sum[CPU_PEANO]) / All.CPU_Sum[CPU_ALL] * 100,
#ifdef FOF
    All.CPU_Sum[CPU_FOF], (All.CPU_Sum[CPU_FOF]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
    All.CPU_Sum[CPU_DRIFT], (All.CPU_Sum[CPU_DRIFT]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_TIMELINE], (All.CPU_Sum[CPU_TIMELINE]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_SNAPSHOT], (All.CPU_Sum[CPU_SNAPSHOT]) / All.CPU_Sum[CPU_ALL] * 100,
#ifdef COOLING
    All.CPU_Sum[CPU_COOLINGSFR], (All.CPU_Sum[CPU_COOLINGSFR]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
#ifdef CHIMES
    All.CPU_Sum[CPU_COOLSFRIMBAL], (All.CPU_Sum[CPU_COOLSFRIMBAL]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
#ifdef SINK_PARTICLES
    All.CPU_Sum[CPU_SINKS], (All.CPU_Sum[CPU_SINKS]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
#ifdef GRAIN_FLUID
    All.CPU_Sum[CPU_DRAGFORCE], (All.CPU_Sum[CPU_DRAGFORCE]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
#if defined(GALSF_FB_MECHANICAL) || defined(GALSF_FB_THERMAL)
    All.CPU_Sum[CPU_SNIIHEATING], (All.CPU_Sum[CPU_SNIIHEATING]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
#if defined(GALSF_FB_FIRE_RT_HIIHEATING)
    All.CPU_Sum[CPU_HIIHEATING], (All.CPU_Sum[CPU_HIIHEATING]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
#if defined(GALSF_FB_FIRE_RT_LOCALRP)
    All.CPU_Sum[CPU_LOCALWIND], (All.CPU_Sum[CPU_LOCALWIND]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
#if defined(RADTRANSFER)
    All.CPU_Sum[CPU_RTNONFLUXOPS], (All.CPU_Sum[CPU_RTNONFLUXOPS]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
    All.CPU_Sum[CPU_MISC], (All.CPU_Sum[CPU_MISC]) / All.CPU_Sum[CPU_ALL] * 100);

    fprintf(FdCPU, "\n");
    fflush(FdCPU);
    }
}


void put_symbol(double t0, double t1, char c)
{
    int i, j;
    i = (int) (t0 * CPU_STRING_LEN + 0.5);
    j = (int) (t1 * CPU_STRING_LEN);
    if(i < 0) {i = 0;}
    if(j < 0) {j = 0;}
    if(i >= CPU_STRING_LEN) {i = CPU_STRING_LEN;}
    if(j >= CPU_STRING_LEN) {j = CPU_STRING_LEN;}
    while(i <= j) {CPU_String[i++] = c;}
    CPU_String[CPU_STRING_LEN] = 0;
}




/*! This routine first calls a computation of various global
 * quantities of the particle distribution, and then writes some
 * statistics about the energies in the various particle components to
 * the file FdEnergy.
 */
void energy_statistics(void)
{
#ifdef OUTPUT_ADDITIONAL_RUNINFO
  compute_global_quantities_of_system();

  if(ThisTask == 0)
    {
      fprintf(FdEnergy,
	      "%.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g",
	      All.Time, SysState.EnergyInt, SysState.EnergyPot, SysState.EnergyKin, SysState.EnergyIntComp[0],
	      SysState.EnergyPotComp[0], SysState.EnergyKinComp[0], SysState.EnergyIntComp[1],
	      SysState.EnergyPotComp[1], SysState.EnergyKinComp[1], SysState.EnergyIntComp[2],
	      SysState.EnergyPotComp[2], SysState.EnergyKinComp[2], SysState.EnergyIntComp[3],
	      SysState.EnergyPotComp[3], SysState.EnergyKinComp[3], SysState.EnergyIntComp[4],
	      SysState.EnergyPotComp[4], SysState.EnergyKinComp[4], SysState.EnergyIntComp[5],
	      SysState.EnergyPotComp[5], SysState.EnergyKinComp[5], SysState.MassComp[0],
	      SysState.MassComp[1], SysState.MassComp[2], SysState.MassComp[3], SysState.MassComp[4],
	      SysState.MassComp[5]);

      fprintf(FdEnergy," \n");
      fflush(FdEnergy);
    }
#endif

#if defined(CHEMCOOL) && defined(GALSF_RESOLVEDISM_FB)
  /* Conservation budget: mass and energy tracking.
   * BUGFIX 2026-07-03: this used to ALSO require HighestActiveTimeBin==HighestOccupiedTimeBin,
   * but energy_statistics() is triggered by the TimeBetStatistics threshold, which is almost
   * always crossed on a small-timebin substep — the full-step condition then skipped the write
   * while the cadence marker advanced anyway, so ENERGYinfo.txt got ONE row per run (t=0 only).
   * The cadence itself already bounds the cost; write whenever we're called. */
  {
    compute_global_quantities_of_system(); /* ensure SysState is populated for conservation budget */
    double cool_glob = 0;
    MPI_Reduce(&CumulCoolingEnergyLoss, &cool_glob, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* min/max metallicity for gas (Type 0) and stars (Type 4) */
    double Zmin_gas_loc = 1e30, Zmax_gas_loc = -1e30, Zmin_star_loc = 1e30, Zmax_star_loc = -1e30;
    {int ii; for(ii = 0; ii < NumPart; ii++) {
        if(P[ii].Mass <= 0) continue;
        double Z = P[ii].Metallicity[0];  /* total Z (slot 0 in both layouts) */
        if(P[ii].Type == 0) { if(Z < Zmin_gas_loc) Zmin_gas_loc = Z; if(Z > Zmax_gas_loc) Zmax_gas_loc = Z; }
        if(P[ii].Type == 4) { if(Z < Zmin_star_loc) Zmin_star_loc = Z; if(Z > Zmax_star_loc) Zmax_star_loc = Z; }
    }}
    double Zmin_gas, Zmax_gas, Zmin_star, Zmax_star;
    MPI_Reduce(&Zmin_gas_loc, &Zmin_gas, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Zmax_gas_loc, &Zmax_gas, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Zmin_star_loc, &Zmin_star, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Zmax_star_loc, &Zmax_star, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(ThisTask == 0) {
      double M_gas = SysState.MassComp[0], M_star = SysState.MassComp[4];
      double M_total = M_gas + M_star + SysState.MassComp[1] + SysState.MassComp[2] + SysState.MassComp[3] + SysState.MassComp[5];
      double E_therm = SysState.EnergyInt;
      double E_kin = SysState.EnergyKin;
      double E_pot = SysState.EnergyPot;
      double E_total = E_therm + E_kin + E_pot;
      double cool_erg = cool_glob * UNIT_ENERGY_IN_CGS;
      double fb_erg = CumulFeedbackEnergy;
      printf("BUDGET t=%.16g  M_gas=%.16g M_star=%.16g M_tot=%.16g  E_therm=%.16g E_kin=%.16g E_pot=%.16g E_tot=%.16g  E_cool_cum=%.16e[erg]  E_fb_cum=%.16e[erg]  M_fb_ret=%.16g[Msun]\n",
        All.Time, M_gas, M_star, M_total, E_therm, E_kin, E_pot, E_total, cool_erg, fb_erg, CumulFeedbackMass);
      fprintf(FdENERGYinfo, "%12.6f  %14.6e %14.6e %14.6e  %14.6e %14.6e %14.6e %14.6e  %14.6e  %14.6e  %14.6e\n",
        All.Time, M_gas*UNIT_MASS_IN_SOLAR, M_star*UNIT_MASS_IN_SOLAR, M_total*UNIT_MASS_IN_SOLAR,
        E_therm*UNIT_ENERGY_IN_CGS, E_kin*UNIT_ENERGY_IN_CGS, E_pot*UNIT_ENERGY_IN_CGS, E_total*UNIT_ENERGY_IN_CGS,
        cool_erg, fb_erg, CumulFeedbackMass);
      fflush(FdENERGYinfo);
      printf("BUDGET_Z t=%.16g  Z_gas=[%.16e,%.16e]", All.Time, Zmin_gas, Zmax_gas);
      if(M_star > 0) printf("  Z_star=[%.16e,%.16e]", Zmin_star, Zmax_star);
      else printf("  Z_star=n/a");
      printf("\n");
    }
  }
#endif
}



void output_extra_log_messages(void)
{
#if defined(TURB_DRIVING) && defined(OUTPUT_ADDITIONAL_RUNINFO)
    log_turb_temp();
#endif

#if defined(GR_TABULATED_COSMOLOGY) && defined(OUTPUT_ADDITIONAL_RUNINFO)
    if((ThisTask == 0) && (All.ComovingIntegrationOn == 1)
    {
        double hubble_a;

        hubble_a = hubble_function(All.Time);
        fprintf(FdDE, "%lld %.16g %e ", (long long) All.NumCurrentTiStep, All.Time, hubble_a);
#ifndef GR_TABULATED_COSMOLOGY_W
        fprintf(FdDE, "%e ", All.DarkEnergyConstantW);
#else
        fprintf(FdDE, "%e %e ", get_wa(All.Time), DarkEnergy_a(All.Time));
#endif
#ifdef GR_TABULATED_COSMOLOGY_G
        fprintf(FdDE, "%e %e", dHfak(All.Time), dGfak(All.Time));
#endif
        fprintf(FdDE, "\n");
        fflush(FdDE);
    }
#endif
}
