#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "resolvedism_star_age.h"
#include "../mesh/kernel.h"
#include "../gravity/healpix_utils.h"

/* Resolved-ISM photo-ionization for individually sampled single stars.
 * Each star particle represents ONE star drawn from the Kroupa IMF.
 * If the single star is massive enough (>=8 Msun) and still alive, it
 * ionizes surrounding gas using direction-dependent Stromgren radii via
 * HEALPix angular decomposition (NSIDE=1, 12 pixels). Each pixel gets
 * 1/12 of the ionizing photon budget.
 *
 * ALGORITHM (v2, 2026-07-28): per-pixel CUMULATIVE recombination integral.
 * For each pixel the discrete Stromgren condition
 *     Sum_{cells, sorted by radius} n_e n_H alpha_B V_cell = Q / NPIX
 * is solved by accumulating each candidate cell's recombination cost
 * (alpha_B X^2 rho M / m_p^2, i.e. its actual density — no mean-density
 * bias) into a radial histogram of PI_NBIN bins per pixel. The histograms
 * are summed across MPI ranks through the standard xchange output-struct
 * reduction, which makes the result INDEPENDENT of the local/remote
 * evaluation order: the v1 walk spent the running photon budget on local
 * cells first and shipped only the remainder to remote ranks, so whenever
 * a domain boundary cut the HII region the budget was spent on far local
 * cells before near remote cells — the front overshot on one side,
 * undershot on the other, and "breathed" as the domain decomposition
 * shifted. The histogram is order-free by construction. The driver then
 * finds, per pixel, the radius where the cumulative cost crosses the pixel
 * budget (interpolated inside the crossing bin), and a second tree sweep
 * marks every cell inside that per-pixel front radius as ionized.
 *
 * v3 (2026-07-29): DETERMINISTIC PARTIAL-CELL CHARGING AT THE CROSSING BIN.
 * v2 placed the *reported* front fractionally (partial charge inside the
 * crossing bin) but FLAGGED cells binarily at r <= front. During D-type
 * expansion the crossing bin is dominated by ONE dense swept-shell cell:
 * whether it was heated depended on its random sub-bin position, and since
 * its bill c is large compared to the leftover budget (frac = leftover/c
 * << 1) the interpolated front pinned just inside the shell cell's radius
 * — the shell cell was (almost) never heated, the front sat at the shell's
 * INNER edge, and the interior ran ~16% under-pressured relative to
 * ionization balance at the actual front radius (-19% vs Spitzer at n=10).
 * v3 reconciles heating with the fractional front by a deterministic
 * MAJORITY rule at the crossing bin: let frac = (budget remaining at the
 * crossing bin's inner edge) / (the bin's total recombination bill). If
 * frac >= 0.5 the whole crossing bin is charged and flagged (marking front
 * = bin OUTER edge); if frac < 0.5 it stays neutral (marking front = bin
 * INNER edge). Heating stays all-or-nothing per cell (chemistry sanity);
 * the heated mass now matches the fractional front to within half of one
 * crossing-cell bill, and the rounding sign alternates as the interior
 * expands and frees budget (frac sweeps through 0.5), so the front is
 * unbiased on average instead of systematically pinned inward. The
 * fractional front is kept separately (PIFrontRadius) for diagnostics and
 * shrink-refinement; the quantized marking front lives in PIFrontMark.
 * NOTE on a residual credit: the per-step rounding residual ((1-frac) or
 * -frac of one bin bill) could in principle be banked per star+pixel and
 * credited to the next STEP's budget for strict time-averaged
 * unbiasedness; that needs persistent per-pixel state on the star particle
 * (migration + restart plumbing) and is deferred — the stateless majority
 * rule already bounds the error to half a cell bill with alternating sign.
 *
 * NOTE on the shell recombination bill (second D-type candidate, examined
 * 2026-07-29): the bill uses n_e n_p = (X rho/m_p)^2 at the cell's CURRENT
 * density, including freshly-flagged swept-shell cells at their compressed
 * density. This is CORRECT: an ionized cell recombines at alpha_B n^2 of
 * its actual density, so under-billing it (e.g. at an anticipated
 * post-expansion "ionized equilibrium" density) would over-ionize and
 * break photon conservation against the chemistry the flag actually
 * drives. The transient over-bill of a just-ionized compressed cell
 * relaxes on its re-expansion (sound-crossing) time — that is real
 * photo-evaporation of the shell's inner surface, not a bug. The v2
 * self-limiting came from the binary-flagging pin above, not from the
 * bill's density.
 *
 * v4 (2026-07-29): DETERMINISTIC MULTI-SOURCE SERIALIZATION (overlap-suite
 * fix). v3 finalized ALL stars converging in the same driver iteration
 * TOGETHER: the Ionized==2 zero-cost skip in the histogram sweep only
 * shields cells claimed by stars finalized in EARLIER iterations, so two
 * identical overlapping sources (which always converge in the same
 * iteration) each billed the shared overlap cells at full cost and carved
 * two INDEPENDENT Stromgren spheres. The union ionized mass was the
 * geometric union of two spheres — 1.37x single at d=0.5 R_S (the measured
 * hii_overlap deficit, 1.35 at 50 kyr) instead of the photon-conserving
 * 2.0x demanded by the equilibrium theorem. v4 gates finalization on a
 * deterministic global order: each iteration, the (ID,pos,SearchRadius) of
 * every still-unfinalized ionizing star is allgathered, and a star may
 * finalize only if NO unfinalized star with a smaller ID has an
 * overlapping search sphere. The smallest-ID star of each overlap cluster
 * finalizes and marks its cells; gated stars re-sweep next iteration, see
 * the claimed cells at zero cost, and the EXISTING budget-aware expansion
 * (>1% unspent budget in any pixel grows the sphere toward the
 * MaxPISearchRadius cap) pushes their unspent budget beyond the
 * neighbor's far edge. Deterministic (ID order + MPI-summed histograms,
 * independent of rank layout), bounded by max_iter as before: the final
 * pass finalizes everyone unconditionally (v3 fallback). Single-star
 * behavior is unchanged (the gate never triggers without an overlapping
 * unfinalized neighbor).
 *
 * Ionization state machine (following gizmo2017 photoionize.c):
 *   Ionized=0 : not ionized
 *   Ionized=1 : confirmed ionized (persists from previous step)
 *   Ionized=2 : newly ionized THIS step (set by the marking sweep)
 * Global reset each step: 1->0, then evaluate marks 2, then 2->1.
 * This ensures dead/inactive stars' HII regions recombine naturally.
 *
 * Convergence tolerance (relative, 2026-07-06 semantics preserved): a pixel
 * whose total cost inside the search sphere leaves <1% of its budget
 * unspent counts as converged; only pixels with >1% unspent budget drive
 * search-radius expansion (bounded by MaxPISearchRadius and max_iter).
 *
 * Uses the code_block_xchange framework for MPI-parallel tree walks. */

#ifdef GALSF_RESOLVEDISM_PHOTOION

#define HEALPIX_NSIDE 1
#define HEALPIX_NPIX  12
/* radial bins of the per-pixel cumulative recombination-cost histogram;
 * bin width = SearchRadius/PI_NBIN, and the driver shrink-refines the
 * search sphere onto the front, so the effective front resolution is
 * ~1% of the front radius (plus in-bin linear interpolation). */
#define PI_NBIN 64

/* case B recombination coefficient at T=10^4 K [cm^3/s] */
#define PI_ALPHA_B 2.6e-13

/* Stromgren radius: Rs = (3 S_ly / (4 pi alpha_B n_H^2))^(1/3) */
static double compute_stromgren_radius_cgs(double S_ly, double nH_cgs)
{
    if(nH_cgs <= 0 || S_ly <= 0) return 0;
    return pow(3.0 * S_ly / (4.0 * M_PI * PI_ALPHA_B * nH_cgs * nH_cgs), 1.0/3.0);
}

/* Per-star driver state, allocated in resolvedism_photoionize().
 * PISearchRadius : current search-sphere radius [code units], per star.
 *                  (The mean-density Stromgren estimate is now ONLY the
 *                  initial guess for this sphere; the front itself comes
 *                  from the cumulative integral.)
 * PIFrontRadius  : finalized per-pixel ionization-front radius [code units],
 *                  FRACTIONAL (partial-charge interpolation inside the
 *                  crossing bin) — diagnostics + shrink-refinement.
 * PIFrontMark    : finalized per-pixel MARKING radius [code units] — v6:
 *                  fractional front rounded UP through the crossing bin's
 *                  innermost cell (gizmo2017 straddling-cell convention);
 *                  consumed by the marking sweep.
 * PISlot         : star index -> compact histogram slot (-1 = not ionizing).
 * PIDoWalk       : per-star participation flag for the current tree sweep.
 * PIBinCost      : per-slot per-pixel radial recombination-cost histogram
 *                  [nslots * HEALPIX_NPIX * PI_NBIN], MPI-summed via
 *                  out2particle (order-independent).
 * PIBinRmin      : per-slot per-pixel per-bin minimum cell radius [code units]
 *                  (0 = empty bin), min-combined via out2particle; locates the
 *                  straddling cell of the crossing bin. */
static double *PISearchRadius = NULL;
static double *PIFrontRadius = NULL;
static double *PIFrontMark = NULL;
static int    *PISlot = NULL;
static char   *PIDoWalk = NULL;
static double *PIBinCost = NULL;
static double *PIBinRmin = NULL;
static int     PIWalkMode = 0; /* 0 = accumulate cost histogram, 1 = mark ionized cells */

/* Compute S_ly for a single star (used in particle2in and pre-processing) */
static double compute_single_star_S_ly(int i)
{
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
    /* StarBench fixed-Q test: honor the pinned rate. recompute_resolvedism_fuv_
     * luminosities pins P.Lyman_photons_per_sec/Ion_luminosity every step, but this
     * function recomputed from the table -> Q~0 for young stars -> the 2026-07-09
     * fixed-Q PI matrix ionized NOTHING in pisub/treeray. Early-out fixes the walk
     * and defuses the Lyman clobber in the pre-pass below. */
    if(All.TestFixedQion > 0) {return All.TestFixedQion;}
#endif
    double S_ly = 0, Mstar = 0;
    double star_age_yr = resolvedism_star_age_yr(i);
    if(star_age_yr <= 0) return 0;

#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
    Mstar = P[i].Mstar;
#endif
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
    if(!P[i].sampled) return 0;
    Mstar = P[i].MstarSampleIMF[0]; /* single star */
#endif
    if(Mstar < 5.0 || Mstar <= 0) return 0; /* early B-type and above have significant ionizing flux */
    if(star_age_yr >= get_lifetime(Mstar, P[i].BirthMetallicity)) return 0; /* star already dead */

    /* PI feedback delay: stars younger than min(t_KH, t_ff) do not ionize */
#if defined(GALSF_RESOLVEDISM_SAMPLE_IMF) || defined(GALSF_RESOLVEDISM_STOCHASTIC_IMF)
    if(P[i].FormationDensity > 0) {
        double t_KH = 1.5e7 * pow(Mstar, -2.1); /* Kelvin-Helmholtz time [yr] */
        double rho_form_cgs = P[i].FormationDensity * UNIT_DENSITY_IN_CGS; /* already physical at formation */
        double t_ff_yr = sqrt(3.0 * M_PI / (32.0 * GRAVITY_G_CGS * rho_form_cgs)) / SECONDS_PER_YEAR;
        double delay = fmin(t_KH, t_ff_yr);
        if(delay > 1.0e6) delay = 1.0e6; /* cap at 1 Myr */
        if(star_age_yr < delay) return 0;
    }
#endif

#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
    {
        double logM = log10(Mstar);
        double logZ = log10(DMAX(P[i].BirthMetallicity, 1e-10));
        double table_age = star_age_yr - stellar_t_PMS(logM, logZ);
        if(table_age <= 0) return 0; /* PMS: no photoionization */
        double log_age = log10(DMAX(table_age, 100.0));
        S_ly = pow(10., stellar_log_Q_ion(logM, logZ, log_age));
    }
#else
    S_ly = pow(10., get_logS_ly(Mstar));
#endif
    return S_ly;
}


/* =========================================================================== */
/*  HEALPix photo-ionization via code_block_xchange framework                 */
/* =========================================================================== */

#define CORE_FUNCTION_NAME resolvedismPI_evaluate
#define INPUTFUNCTION_NAME particle2in_resolvedismPI
#define OUTPUTFUNCTION_NAME out2particle_resolvedismPI
#define CONDITIONFUNCTION_FOR_EVALUATION if(resolvedismPI_active_check(i))
#include "../system/code_block_xchange_initialize.h"

struct INPUT_STRUCT_NAME
{
    MyDouble Pos[3], SearchRadius, S_ly;
    MyDouble Front[HEALPIX_NPIX]; /* finalized per-pixel MARKING radii (v3: majority-quantized to crossing-bin edges) [code units] — used by the marking sweep (PIWalkMode=1) */
    int NodeList[NODELISTLENGTH];
}
*DATAIN_NAME, *DATAGET_NAME;

void particle2in_resolvedismPI(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration)
{
    int k; for(k=0;k<3;k++) {in->Pos[k]=P[i].Pos[k];}
    in->S_ly = compute_single_star_S_ly(i);
    /* SearchRadius is fully driver-managed (initialized + expanded/refined in
     * resolvedism_photoionize); no state is consumed inside the walk, so the
     * export packing order relative to the local evaluation does not matter. */
    in->SearchRadius = (PISearchRadius != NULL) ? PISearchRadius[i] : 0;
    for(k = 0; k < HEALPIX_NPIX; k++)
        in->Front[k] = (PIFrontMark != NULL) ? PIFrontMark[i * HEALPIX_NPIX + k] : 0; /* v3: marking sweep uses the majority-quantized front, not the fractional one */
}

struct OUTPUT_STRUCT_NAME
{
    double BinCost[HEALPIX_NPIX * PI_NBIN]; /* per-pixel radial recombination-cost histogram [photons/s] */
    double BinRmin[HEALPIX_NPIX * PI_NBIN]; /* per-pixel per-bin minimum cell radius [code units], 0 = empty (v6 straddling-cell round-up) */
    double NMarked;                         /* cells newly marked ionized (marking sweep diagnostic) */
}
*DATARESULT_NAME, *DATAOUT_NAME;

void out2particle_resolvedismPI(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration)
{
    if(PIWalkMode != 0) return; /* marking sweep writes gas cells directly; nothing to reduce onto the star */
    if(PIBinCost == NULL || PISlot == NULL) return;
    if(PISlot[i] < 0) return;
    int k;
    double *bins = &PIBinCost[(size_t)PISlot[i] * HEALPIX_NPIX * PI_NBIN];
    for(k = 0; k < HEALPIX_NPIX * PI_NBIN; k++) {bins[k] += out->BinCost[k];}
    /* '+=' over mode 0 and all mode-1 imports: a plain order-independent sum,
     * unlike the v1 running-budget subtraction. */
    double *rmins = &PIBinRmin[(size_t)PISlot[i] * HEALPIX_NPIX * PI_NBIN];
    for(k = 0; k < HEALPIX_NPIX * PI_NBIN; k++) {
        if(out->BinRmin[k] > 0 && (rmins[k] <= 0 || out->BinRmin[k] < rmins[k])) {rmins[k] = out->BinRmin[k];}
    } /* min-combine, order-independent like the sum above */
}

/* base eligibility: massive single star (the >=8 Msun activity gate) */
static int resolvedismPI_base_check(int i)
{
    if(P[i].Type != 4) {return 0;}
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
    if(!P[i].sampled) {return 0;}
    if(P[i].MstarSampleIMF[0] < 8.0) {return 0;} /* single star must be massive to ionize */
#endif
#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
    if(P[i].Mstar < 8.0) {return 0;}
#endif
    return 1;
}

int resolvedismPI_active_check(int i);
int resolvedismPI_active_check(int i)
{
    if(!resolvedismPI_base_check(i)) {return 0;}
    if(PIDoWalk == NULL || !PIDoWalk[i]) {return 0;} /* driver selects participants per sweep */
    return 1;
}


/*!   -- this subroutine writes to shared memory in the marking sweep [setting CellP.Ionized=2]:
 *       the store is an idempotent constant, so the unprotected write is benign under openmp */
int resolvedismPI_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)
{
    int startnode, numngb_inbox, listindex = 0, j, k, n;
    struct INPUT_STRUCT_NAME local;
    struct OUTPUT_STRUCT_NAME out;
    memset(&out, 0, sizeof(struct OUTPUT_STRUCT_NAME));

    if(mode == 0) {particle2in_resolvedismPI(&local, target, loop_iteration);} else {local = DATAGET_NAME[target];}
    if(local.S_ly <= 0) return 0;
    if(local.SearchRadius <= 0) return 0;

    double h2 = local.SearchRadius * local.SearchRadius;
    double inv_dr = (double)PI_NBIN / local.SearchRadius;

    if(mode == 0) {startnode = All.MaxPart;}
    else {startnode = DATAGET_NAME[target].NodeList[0]; startnode = Nodes[startnode].u.d.nextnode;}

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb_inbox = ngb_treefind_variable_threads(local.Pos, local.SearchRadius, target, &startnode, mode, exportflag, exportnodecount, exportindex, ngblist);
            if(numngb_inbox < 0) {return -2;}
            for(n = 0; n < numngb_inbox; n++)
            {
                j = ngblist[n];
                if(P[j].Type != 0) continue;
                if(P[j].Mass <= 0) continue;

                double dp[3];
                for(k=0;k<3;k++) {dp[k] = local.Pos[k] - P[j].Pos[k];}
                NEAREST_XYZ(dp[0],dp[1],dp[2],1);
                double r2 = dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2];
                if(r2 <= 0 || r2 >= h2) continue;

                /* Direction vector star -> cell, assign to HEALPix pixel */
                double vec[3] = {-dp[0], -dp[1], -dp[2]};
                long ipix;
                vec2pix_ring(HEALPIX_NSIDE, vec, &ipix);
                if(ipix < 0 || ipix >= HEALPIX_NPIX) ipix = 0;
                double r = sqrt(r2);

                if(PIWalkMode == 0)
                {
                    /* histogram sweep: accumulate the cell's recombination cost
                     * (alpha_B n_e n_H V, with n_e = n_H = X rho/m_p) at its radius.
                     * Deliberately billed at the cell's CURRENT density — including
                     * compressed swept-shell cells — because an ionized cell really
                     * recombines at alpha_B n^2 of its actual density; see the v3
                     * header note (under-billing would break photon conservation
                     * against the chemistry the Ionized flag drives). */
                    if(CellP[j].Ionized == 2) continue; /* already claimed this step by a previously-finalized star: no cost (multi-star skip semantics preserved) */
                    int b = (int)(r * inv_dr); if(b >= PI_NBIN) {b = PI_NBIN - 1;}
                    double rho_cgs = CellP[j].Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS;
                    double M_cgs   = P[j].Mass * UNIT_MASS_IN_CGS;
                    double recomb  = PI_ALPHA_B * HYDROGEN_MASSFRAC * HYDROGEN_MASSFRAC
                                     * rho_cgs * M_cgs / (PROTONMASS_CGS * PROTONMASS_CGS);
                    out.BinCost[ipix * PI_NBIN + b] += recomb;
                    if(out.BinRmin[ipix * PI_NBIN + b] <= 0 || r < out.BinRmin[ipix * PI_NBIN + b]) {out.BinRmin[ipix * PI_NBIN + b] = r;} /* v6: innermost cell of each bin */
                }
                else
                {
                    /* marking sweep: this star's front is finalized — flag every
                     * cell inside its pixel's MARKING radius as newly ionized.
                     * v3: Front is quantized to the crossing bin's inner/outer
                     * edge by the majority rule, so the binary r-test below is
                     * exactly consistent with the (partial) budget charged:
                     * crossing-bin cells are all-in (frac>=0.5, whole bin
                     * charged) or all-out (frac<0.5, bin uncharged) — heating
                     * stays all-or-nothing per cell. */
                    if(r <= local.Front[ipix] && CellP[j].Ionized != 2) {
                        CellP[j].Ionized = 2;
                        out.NMarked += 1;
                    }
                }
            }
        }
        if(mode == 1)
        {
            listindex++;
            if(listindex < NODELISTLENGTH)
            {
                startnode = DATAGET_NAME[target].NodeList[listindex];
                if(startnode >= 0) {startnode = Nodes[startnode].u.d.nextnode;}
            }
        }
    }

    if(mode == 0) {out2particle_resolvedismPI(&out, target, 0, loop_iteration);} else {DATARESULT_NAME[target] = out;}
    return 0;
}


void resolvedism_photoionize(void)
{
    int i, j, k;

    /* === Global reset (gizmo2017 pattern) ===
     * Clear all previously-ionized cells. The marking sweep will re-mark cells
     * inside active HII regions as Ionized=2. After the sweeps, we
     * transition 2->1. Cells not re-ionized (dead/inactive stars) revert
     * to Ionized=0 and recombine naturally via CHEMCOOL. */
    for(j = 0; j < N_gas; j++) {
        if(P[j].Type != 0) continue;
        if(CellP[j].Ionized == 1) CellP[j].Ionized = 0;
    }

    /* Pre-compute Lyman photon rates for ALL stars.
     * BUGFIX 2026-07-31: this and the walk loop below iterated over ACTIVE
     * particles only, while the global reset above unflags EVERY cell EVERY
     * call — so on each fine step where an ionizing star's timebin left it
     * inactive, its whole HII region was silently unflagged and CHEMCOOL
     * cooled/recombined the dense front cells (t_cool ~ 1 kyr at n=100 <
     * the star's timestep) before the next re-mark. The bubble was cold-
     * started over and over: no photo-evaporation refill, interior 7x
     * under-dense, delivered photon budget collapsing to 16% of Q, front
     * stalled at -40..-55%% vs StarBench (density-dependent exactly as
     * t_cool scales; few-%% at n=10). Smith+21 spec: the full algorithm runs
     * "every fine time-step for ALL sources" — this restores that. */
    for(i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 4) continue;
#if defined(GALSF_RESOLVEDISM_G0_VARIABLE) && defined(GALSF_RESOLVEDISM_PHOTOION)
        P[i].Lyman_photons_per_sec = compute_single_star_S_ly(i);
#endif
    }

    /* Allocate per-star driver state */
    PISearchRadius = (double *) mymalloc("PISearchRadius", NumPart * sizeof(double));
    PIFrontRadius = (double *) mymalloc("PIFrontRadius", (size_t)NumPart * HEALPIX_NPIX * sizeof(double));
    PIFrontMark = (double *) mymalloc("PIFrontMark", (size_t)NumPart * HEALPIX_NPIX * sizeof(double));
    PISlot = (int *) mymalloc("PISlot", NumPart * sizeof(int));
    PIDoWalk = (char *) mymalloc("PIDoWalk", NumPart * sizeof(char));
    char *pi_final = (char *) mymalloc("PIFinal", NumPart * sizeof(char));   /* 0=iterating, 1=finalized this iter (to mark), 2=marked */
    char *pi_refine = (char *) mymalloc("PIRefine", NumPart * sizeof(char)); /* shrink-refinement counter */
    memset(PISearchRadius, 0, NumPart * sizeof(double));
    memset(PIFrontRadius, 0, (size_t)NumPart * HEALPIX_NPIX * sizeof(double));
    memset(PIFrontMark, 0, (size_t)NumPart * HEALPIX_NPIX * sizeof(double));
    memset(PISlot, 0xFF, NumPart * sizeof(int)); /* all -1 */
    memset(PIDoWalk, 0, NumPart * sizeof(char));
    memset(pi_final, 0, NumPart * sizeof(char));
    memset(pi_refine, 0, NumPart * sizeof(char));

    double max_r = All.MaxPISearchRadius / (UNIT_LENGTH_IN_KPC * All.cf_atime); /* physical kpc -> comoving code (cap semantics unchanged) */

    /* Identify ALL ionizing stars (BUGFIX 2026-07-31: was active-only — see the
     * Lyman precompute loop above; the global reset requires every source to
     * re-mark every call or its region flickers off between its timebin steps).
     * The mean-density Stromgren estimate survives ONLY as this guess — it
     * biases nothing, because the front is solved from the per-cell integral. */
    int nslots = 0;
    for(i = 0; i < NumPart; i++) {
        if(P[i].Type != 4) continue;
        if(!resolvedismPI_base_check(i)) continue;
        double S_ly = compute_single_star_S_ly(i);
        if(S_ly <= 0) continue;
        PISlot[i] = nslots++;
        double h = P[i].KernelRadius;
        double nH_local = 1.0; /* fallback: 1 cm^-3 */
#ifdef DO_DENSITY_AROUND_NONGAS_PARTICLES
        if(P[i].DensityAroundParticle > 0)
            nH_local = HYDROGEN_MASSFRAC * P[i].DensityAroundParticle * All.cf_a3inv * UNIT_DENSITY_IN_CGS / PROTONMASS_CGS;
#endif
        double h_stromgren = compute_stromgren_radius_cgs(S_ly, nH_local) / (UNIT_LENGTH_IN_CGS * All.cf_atime);
        if(h_stromgren > h) h = h_stromgren;
        if(h <= 0) h = All.ForceSoftening[P[i].Type];
        if(max_r > 0 && h > max_r) h = max_r;
        PISearchRadius[i] = h;
    }
    PIBinCost = (double *) mymalloc("PIBinCost", (size_t)(nslots > 0 ? nslots : 1) * HEALPIX_NPIX * PI_NBIN * sizeof(double));
    PIBinRmin = (double *) mymalloc("PIBinRmin", (size_t)(nslots > 0 ? nslots : 1) * HEALPIX_NPIX * PI_NBIN * sizeof(double));

    /* the sweep template iterates the ACTIVE chain only: splice inactive
     * ionizing stars in for the driver (restored below), else their regions
     * are wiped by the global reset on every star-inactive fine step */
    size_t pi_saved_n_active = ActiveParticleList.size();
    for(i = 0; i < NumPart; i++) {
        if(PISlot[i] < 0) continue;
        if(TimeBinActive[P[i].TimeBin]) continue;
        ActiveParticleList.push_back(i);
    }

    /* Iterative driver. Per iteration:
     *   sweep A (PIWalkMode=0): accumulate each unfinished star's per-pixel radial
     *     cost histogram over ALL candidate cells (all ranks, order-free sum);
     *   analysis: per pixel, find the radius where the cumulative cost crosses the
     *     pixel budget Q/NPIX (linear interpolation inside the crossing bin).
     *     - any pixel with >1% budget unspent inside the sphere -> expand the
     *       sphere (bounded by MaxPISearchRadius cap and max_iter) and re-sweep;
     *     - all pixels crossed but deep inside the sphere -> shrink-refine once or
     *       twice for finer bins (this also defuses the inflated mean-density
     *       initial guess over an evacuated cavity);
     *     - otherwise finalize the star's per-pixel front radii;
     *   sweep B (PIWalkMode=1): mark cells inside newly-finalized fronts Ionized=2.
     * Stars finalized in earlier iterations shield their claimed cells from later
     * stars' histograms (Ionized==2 skip, at zero cost — same multi-star
     * bookkeeping as v1). */
    PRINT_STATUS(" ..photo-ionizing gas around single stars");
    int nslots_tot = 0; MPI_Allreduce(&nslots, &nslots_tot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    int iter = 0, max_iter = (int)DMIN(20 + nslots_tot, 1000); /* ID serialization finalizes >=1 star per overlap cluster per pass */
    int incomplete_total = 0;
    do {
        /* --- sweep A: cost histograms for all unfinished ionizing stars --- */
        int nwalk_local = 0;
        for(i = 0; i < NumPart; i++)  /* ALL sources every call (2026-07-31 fix; bodies gate on PISlot/Type) */ {
            PIDoWalk[i] = 0;
            if(PISlot[i] >= 0 && pi_final[i] == 0) {
                PIDoWalk[i] = 1; nwalk_local++;
                memset(&PIBinCost[(size_t)PISlot[i] * HEALPIX_NPIX * PI_NBIN], 0, HEALPIX_NPIX * PI_NBIN * sizeof(double));
                memset(&PIBinRmin[(size_t)PISlot[i] * HEALPIX_NPIX * PI_NBIN], 0, HEALPIX_NPIX * PI_NBIN * sizeof(double));
            }
        }
        int nwalk_total = 0;
        MPI_Allreduce(&nwalk_local, &nwalk_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        PIWalkMode = 0;
        if(nwalk_total > 0)
        {
            #include "../system/code_block_xchange_perform_ops_malloc.h"
            #include "../system/code_block_xchange_perform_ops.h"
            #include "../system/code_block_xchange_perform_ops_demalloc.h"
        }

        /* --- v4 serialization gather: (ID, pos, R) of every still-unfinalized
         * ionizing star, for the deterministic finalize-order gate below --- */
        int npi_unfin_local = 0;
        for(i = 0; i < NumPart; i++)  /* ALL sources every call (2026-07-31 fix; bodies gate on PISlot/Type) */
            if(PISlot[i] >= 0 && pi_final[i] == 0 && PISearchRadius[i] > 0) {npi_unfin_local++;}
        int *pi_gcnt = (int *) mymalloc("PIGathCnt", NTask * sizeof(int));
        int *pi_gdsp = (int *) mymalloc("PIGathDsp", NTask * sizeof(int));
        int *pi_gcnt4 = (int *) mymalloc("PIGathCnt4", NTask * sizeof(int));
        int *pi_gdsp4 = (int *) mymalloc("PIGathDsp4", NTask * sizeof(int));
        MPI_Allgather(&npi_unfin_local, 1, MPI_INT, pi_gcnt, 1, MPI_INT, MPI_COMM_WORLD);
        int npi_unfin_glob = 0;
        for(j = 0; j < NTask; j++) {
            pi_gdsp[j] = npi_unfin_glob; pi_gcnt4[j] = 4 * pi_gcnt[j]; pi_gdsp4[j] = 4 * npi_unfin_glob;
            npi_unfin_glob += pi_gcnt[j];
        }
        unsigned long long *pi_gid = (unsigned long long *) mymalloc("PIGathID", (size_t)(npi_unfin_glob > 0 ? npi_unfin_glob : 1) * sizeof(unsigned long long));
        double *pi_gdat = (double *) mymalloc("PIGathDat", (size_t)(npi_unfin_glob > 0 ? npi_unfin_glob : 1) * 4 * sizeof(double));
        {
            unsigned long long *sid = (unsigned long long *) mymalloc("PIGathSID", (size_t)(npi_unfin_local > 0 ? npi_unfin_local : 1) * sizeof(unsigned long long));
            double *sdat = (double *) mymalloc("PIGathSDat", (size_t)(npi_unfin_local > 0 ? npi_unfin_local : 1) * 4 * sizeof(double));
            int ns = 0;
            for(i = 0; i < NumPart; i++)  /* ALL sources every call (2026-07-31 fix; bodies gate on PISlot/Type) */ {
                if(!(PISlot[i] >= 0 && pi_final[i] == 0 && PISearchRadius[i] > 0)) continue;
                sid[ns] = (unsigned long long) P[i].ID;
                sdat[4*ns+0] = P[i].Pos[0]; sdat[4*ns+1] = P[i].Pos[1]; sdat[4*ns+2] = P[i].Pos[2];
                sdat[4*ns+3] = PISearchRadius[i];
                ns++;
            }
            MPI_Allgatherv(sid, npi_unfin_local, MPI_UNSIGNED_LONG_LONG, pi_gid, pi_gcnt, pi_gdsp, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
            MPI_Allgatherv(sdat, 4 * npi_unfin_local, MPI_DOUBLE, pi_gdat, pi_gcnt4, pi_gdsp4, MPI_DOUBLE, MPI_COMM_WORLD);
            myfree(sdat); myfree(sid);
        }

        /* --- analysis: solve the discrete Stromgren condition per pixel --- */
        int incomplete_local = 0, newfinal_local = 0;
        int allow_iter = (iter < max_iter - 1); /* last pass must finalize everyone */
        for(i = 0; i < NumPart; i++)  /* ALL sources every call (2026-07-31 fix; bodies gate on PISlot/Type) */ {
            if(PISlot[i] < 0 || pi_final[i]) continue;
            double S_ly = compute_single_star_S_ly(i);
            if(S_ly <= 0) {pi_final[i] = 1; newfinal_local++; continue;}
            /* v4 finalize-order gate: defer this star if any still-unfinalized
             * star with a SMALLER ID has an overlapping search sphere — that
             * star finalizes and claims its cells first; we re-sweep next
             * iteration with those cells at zero cost (line ~315 skip) and the
             * budget-aware expansion below redeploys the freed budget outward.
             * Disabled on the last pass (allow_iter==0): everyone finalizes. */
            if(allow_iter && npi_unfin_glob > 1) {
                int m, gated = 0;
                for(m = 0; m < npi_unfin_glob; m++) {
                    if(pi_gid[m] >= (unsigned long long) P[i].ID) continue;
                    double gdx = pi_gdat[4*m+0] - P[i].Pos[0];
                    double gdy = pi_gdat[4*m+1] - P[i].Pos[1];
                    double gdz = pi_gdat[4*m+2] - P[i].Pos[2];
                    NEAREST_XYZ(gdx, gdy, gdz, 1);
                    double rsum = PISearchRadius[i] + pi_gdat[4*m+3];
                    if(gdx*gdx + gdy*gdy + gdz*gdz < rsum * rsum) {gated = 1; break;}
                }
                if(gated) {incomplete_local++; continue;}
            }
            double S_pix = S_ly / HEALPIX_NPIX;
            double *bins = &PIBinCost[(size_t)PISlot[i] * HEALPIX_NPIX * PI_NBIN];
            double *rmins = &PIBinRmin[(size_t)PISlot[i] * HEALPIX_NPIX * PI_NBIN];
            double R = PISearchRadius[i], dr = R / PI_NBIN;
            double front[HEALPIX_NPIX], fmark[HEALPIX_NPIX], front_max = 0, cum_min = MAX_REAL_NUMBER;
            int n_uncrossed = 0;
            for(k = 0; k < HEALPIX_NPIX; k++) {
                double cum = 0, f = -1, fm = -1; int b;
                for(b = 0; b < PI_NBIN; b++) {
                    double c = bins[k * PI_NBIN + b];
                    if(c <= 0) continue;
                    if(cum + c >= S_pix) {
                        /* marking front rounds UP through the straddling cell (gizmo2017) */
                        double frac = (S_pix - cum) / c;
                        f = ((double)b + frac) * dr;
                        double rmin_b = rmins[k * PI_NBIN + b];
                        if(rmin_b > 0 && rmin_b >= f) {fm = rmin_b * (1.0 + 1.0e-9);}
                        else {fm = f;}
                        break;
                    }
                    cum += c;
                }
                if(f < 0) { /* budget not exhausted within the search sphere */
                    if(cum <= 0) {f = 0; n_uncrossed++; cum_min = 0;} /* empty pixel: nothing marked; still drives expansion toward the cap */
                    else if(S_pix - cum <= 0.01 * S_pix) {f = R;}     /* <1% budget left: converged (relative tolerance) — everything in the pixel is ionized */
                    else {f = R; n_uncrossed++; if(cum < cum_min) cum_min = cum;}
                    fm = f; /* no crossing bin: marking front == fractional front */
                }
                front[k] = f; fmark[k] = fm; if(f > front_max) front_max = f;
            }
            if(n_uncrossed > 0 && allow_iter && !(max_r > 0 && R >= 0.999 * max_r)) {
                /* expand: volumetric extrapolation from the worst pixel, clipped [1.25, 2.5] */
                double grow = 2.0;
                if(cum_min > 0 && cum_min < MAX_REAL_NUMBER) {
                    grow = pow(S_pix / cum_min, 1.0/3.0);
                    if(grow < 1.25) grow = 1.25;
                    if(grow > 2.5) grow = 2.5;
                }
                R *= grow; if(max_r > 0 && R > max_r) R = max_r;
                PISearchRadius[i] = R; incomplete_local++; continue;
            }
            if(n_uncrossed == 0 && allow_iter && pi_refine[i] < 2 && front_max > 0 && front_max < 0.5 * R) {
                /* all fronts found but deep inside the sphere (e.g. inflated
                 * mean-density guess over an evacuated cavity): shrink for
                 * finer bins. Pixels converged by tolerance have front=R and
                 * block this, so no marked gas can be lost to the shrink. */
                PISearchRadius[i] = 1.3 * front_max; pi_refine[i]++; incomplete_local++; continue;
            }
            /* finalize this star's per-pixel fronts: fractional (diagnostics)
             * + majority-quantized marking radii (consumed by sweep B) */
            pi_final[i] = 1; newfinal_local++;
            for(k = 0; k < HEALPIX_NPIX; k++) {
                PIFrontRadius[i * HEALPIX_NPIX + k] = front[k];
                PIFrontMark[i * HEALPIX_NPIX + k] = fmark[k];
            }
        }

        /* v4: release the serialization-gather arrays (LIFO) */
        myfree(pi_gdat); myfree(pi_gid);
        myfree(pi_gdsp4); myfree(pi_gcnt4); myfree(pi_gdsp); myfree(pi_gcnt);

        /* --- sweep B: mark cells inside newly-finalized fronts --- */
        for(i = 0; i < NumPart; i++)  /* ALL sources every call (2026-07-31 fix; bodies gate on PISlot/Type) */ {
            PIDoWalk[i] = (PISlot[i] >= 0 && pi_final[i] == 1) ? 1 : 0;
        }
        int newfinal_total = 0;
        MPI_Allreduce(&newfinal_local, &newfinal_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        PIWalkMode = 1;
        if(newfinal_total > 0)
        {
            #include "../system/code_block_xchange_perform_ops_malloc.h"
            #include "../system/code_block_xchange_perform_ops.h"
            #include "../system/code_block_xchange_perform_ops_demalloc.h"
        }
        for(i = 0; i < NumPart; i++)  /* ALL sources every call (2026-07-31 fix; bodies gate on PISlot/Type) */ {
            if(pi_final[i] == 1) pi_final[i] = 2;
        }

        MPI_Allreduce(&incomplete_local, &incomplete_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        iter++;
    } while(incomplete_total > 0 && iter < max_iter);
    if(ThisTask == 0 && iter >= max_iter && max_iter >= 1000) {printf("WARNING: RESOLVEDISM PI hit max_iter=%d with %d sources: final pass finalized overlapping sources together (shared cells billed more than once)\n", max_iter, nslots_tot);}

    ActiveParticleList.resize(pi_saved_n_active); /* unsplice */

    if(iter > 1) {
        double local_max_r = 0, local_min_r = MAX_REAL_NUMBER;
        for(i = 0; i < NumPart; i++) {
            if(P[i].Type == 4 && PISearchRadius != NULL && PISearchRadius[i] > 0) {
                if(PISearchRadius[i] > local_max_r) local_max_r = PISearchRadius[i];
                if(PISearchRadius[i] < local_min_r) local_min_r = PISearchRadius[i];
            }
        }
        if(local_min_r >= MAX_REAL_NUMBER) local_min_r = 0;
        double glob_max_r = 0, glob_min_r = 0;
        MPI_Reduce(&local_max_r, &glob_max_r, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_min_r, &glob_min_r, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        if(ThisTask == 0)
            printf("RESOLVEDISM PI: adjusted search radius in %d iterations (R_min=%.1f pc, R_max=%.1f pc, cap=%.1f pc)\n",
                iter, glob_min_r * All.cf_atime * UNIT_LENGTH_IN_KPC * 1e3, glob_max_r * All.cf_atime * UNIT_LENGTH_IN_KPC * 1e3, All.MaxPISearchRadius * 1e3);
    }

    /* HII region size summary: per-star min/max pixel front radius (asymmetry measure) */
    {
        int npi_local = 0;
        double local_rfront_min = MAX_REAL_NUMBER, local_rfront_max = 0;
        double local_asym_max = 0;
        for(i = 0; i < NumPart; i++)  /* ALL sources every call (2026-07-31 fix; bodies gate on PISlot/Type) */ {
            if(P[i].Type != 4) continue;
            if(PISearchRadius[i] <= 0) continue;
            npi_local++;
            double star_rmin = MAX_REAL_NUMBER, star_rmax = 0;
            int npix_active = 0;
            for(k = 0; k < HEALPIX_NPIX; k++) {
                double rf = PIFrontRadius[i * HEALPIX_NPIX + k];
                if(rf > 0) {
                    npix_active++;
                    if(rf > star_rmax) star_rmax = rf;
                    if(rf < star_rmin && rf > 0) star_rmin = rf;
                }
            }
            if(npix_active > 0) {
                if(star_rmin < local_rfront_min) local_rfront_min = star_rmin;
                if(star_rmax > local_rfront_max) local_rfront_max = star_rmax;
                double asym = (star_rmin > 0) ? star_rmax / star_rmin : 0;
                if(asym > local_asym_max) local_asym_max = asym;
            }
        }
        if(local_rfront_min >= MAX_REAL_NUMBER) local_rfront_min = 0;

        int npi_total = 0;
        double glob_rfront_min = 0, glob_rfront_max = 0, glob_asym_max = 0;
        MPI_Allreduce(&npi_local, &npi_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Reduce(&local_rfront_min, &glob_rfront_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_rfront_max, &glob_rfront_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_asym_max, &glob_asym_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if(ThisTask == 0 && npi_total > 0) {
            printf("RESOLVEDISM PI: %d ionizing stars at t=%g (%d iter)\n", npi_total, All.Time, iter);
            printf("RESOLVEDISM PI: HII front R_min=%.1f pc, R_max=%.1f pc, max asymmetry=%.1f:1\n",
                glob_rfront_min * All.cf_atime * UNIT_LENGTH_IN_KPC * 1e3, glob_rfront_max * All.cf_atime * UNIT_LENGTH_IN_KPC * 1e3, glob_asym_max);
        }
    }

    /* Post-evaluate ionization state transitions (gizmo2017 pattern):
     * The global reset already cleared 1->0 at the top.
     * Now transition 2->1. No wakeup: gizmo2017 photoionize.c:718 wakes SphP[j] with a stale j, never the ionized cell.
     * Cells not re-ionized remain at 0 and recombine via CHEMCOOL. */
    int sum_ionized = 0;
    for(j = 0; j < N_gas; j++) {
        if(P[j].Type != 0) continue;
        if(CellP[j].Ionized == 2) {
            CellP[j].Ionized = 1;
        }
        if(CellP[j].Ionized == 1)
            sum_ionized++;
    }
    {
        int total_ionized = 0;
        MPI_Reduce(&sum_ionized, &total_ionized, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if(ThisTask == 0)
            printf("RESOLVEDISM PI: %d ionized\n", total_ionized);
    }

    /* Free in LIFO order */
    myfree(PIBinRmin); PIBinRmin = NULL;
    myfree(PIBinCost); PIBinCost = NULL;
    myfree(pi_refine);
    myfree(pi_final);
    myfree(PIDoWalk); PIDoWalk = NULL;
    myfree(PISlot); PISlot = NULL;
    myfree(PIFrontMark); PIFrontMark = NULL;
    myfree(PIFrontRadius); PIFrontRadius = NULL;
    myfree(PISearchRadius); PISearchRadius = NULL;
}
#include "../system/code_block_xchange_finalize.h"


#endif /* GALSF_RESOLVEDISM_PHOTOION */
