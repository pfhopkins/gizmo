/* ETOT_INCODE: direct-sum total energy of the star particles (as stella_dev core/run.cc incode_total_energy) */
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../../declarations/allvars.h"
#include "../../core/proto.h"
#include "../../mesh/kernel.h"

#ifdef KETJU_REGULARIZATION
/* Stella: KETJU stars are PartType 4 (dead remnants included); Type 5 only with sinks enabled */
#ifdef SINK_PARTICLES
#define KETJU_ETOT_PTYPE(i) (P[(i)].Type == 4 || P[(i)].Type == 5)
#else
#define KETJU_ETOT_PTYPE(i) (P[(i)].Type == 4)
#endif
void ketju_energy_diagnostic(void)
{
    ketju_energy_diag_prepare(); /* stars at the current time (lazy host drift) */
    /* MPI-aware: gather all Type-5 particles to rank 0 and do the exact double-precision
       KE + unsoftened PE there. Works for any NTask. */
    static double E0 = 0; static int have0 = 0; static double last_t = -1e30;
    /* NOTE: this routine now runs its gather + pair loop EVERY step (no early throttle): the
     * Hamiltonian-SWITCH LEDGER below must catch region-membership transitions at the step they occur
     * (the potential-convention switch of an inside-kernel pair must be evaluated at the separation
     * where it actually happened). Only the PRINT remains throttled to snapshot cadence. Cost is
     * O(N_type5^2) per step — the diagnostic already bails for N>4096. */
    /* print only where every counted particle ends its step here: elsewhere inactive particles carry half-kicked
     * velocities and KETJU members part-way through their step sit on the host's chord, so E is not the conserved one */
    int nsync_loc = 0, nsync = 0;
    for(int q = 0; q < NumPart; q++) if(KETJU_ETOT_PTYPE(q) && P[q].Ti_begstep + P[q].dt_step != All.Ti_Current) nsync_loc++;
    MPI_Allreduce(&nsync_loc, &nsync, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    int do_print = (nsync == 0) && !(have0 && (All.Time - last_t < All.TimeBetSnapshot - 1e-9));

    /* per particle, 11 doubles: pos[3], Vel[3], (unused), mass, integrated-flag.
     * At a print every member ends its step here, so P.Pos/P.Vel are MSTAR's end-of-step state. */
    const int W = 12;  /* [11] = particle ID (for the switch-ledger pair tracking across steps) */
    int nloc = 0, i, j;
    for(i = 0; i < NumPart; i++) if(KETJU_ETOT_PTYPE(i)) nloc++;
    double *sloc = (double*)malloc((nloc > 0 ? nloc : 1) * W * sizeof(double));
    int k = 0;
    for(i = 0; i < NumPart; i++) if(KETJU_ETOT_PTYPE(i)) {
        /* P.Pos/P.Vel: exact for members only when their step ends here (mid-step they are the chord), see do_print */
        double px=P[i].Pos[0], py=P[i].Pos[1], pz=P[i].Pos[2];
        double vx=P[i].Vel[0], vy=P[i].Vel[1], vz=P[i].Vel[2];
        sloc[k*W+10]=0.0;
#ifdef KETJU_REGULARIZATION
        sloc[k*W+10]=(P[i].KetjuIntegrated ? 1.0 : 0.0);
#endif
        sloc[k*W+0]=px; sloc[k*W+1]=py; sloc[k*W+2]=pz;
        sloc[k*W+3]=vx; sloc[k*W+4]=vy; sloc[k*W+5]=vz;
        /* [6] = kernel radius (compact-support h) for the softened-potential metric;
         * [7] = KETJU region tag (0 = not a chain member; same nonzero tag = same chain);
         * [8] = 1 => chain-internal pairs are UNSOFTENED (ALWAYS, by design: gravity
         *       inside a KETJU region is never softened -- see ketju_coupling.cc). */
        sloc[k*W+6]=ForceSoftening_KernelRadius(i);
        sloc[k*W+7]=0.0; sloc[k*W+8]=0.0;
#ifdef KETJU_REGULARIZATION
        sloc[k*W+7]=(double)P[i].KetjuChainID; /* same nonzero id = same chain */
        sloc[k*W+8]=1.0;
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
                 * for every pair (as the tree applies), EXCEPT unsoftened for chain-internal pairs
                 * (chains are ALWAYS integrated unsoftened inside a KETJU region, by design). */
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
            static double E0_prop = 0; if(!have0) { E0 = E; E0_prop = E_prop; have0 = 1; }

            if(do_print) {
#ifdef KETJU_REGULARIZATION
                ketju_note_total_energy(E, E0); /* printed evaluations only: all counted particles synchronised */
#endif
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
#endif /* KETJU_REGULARIZATION */
