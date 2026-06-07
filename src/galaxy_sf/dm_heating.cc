/* galaxy_sf/dm_heating.cc — continuous DM annihilation + decay heating of gas.
 *
 * Consumes CellP[i].DM_Rho (populated by dm_dispersion_loop / disp_density)
 * and adds the analytic heating rate to CellP[i].DtInternalEnergy. Called
 * from run.cc after compute_hydro_densities_and_forces() and before the
 * transport/cooling block, so the heating accumulates in the same
 * DtInternalEnergy budget as hydro PdV and shock heating.
 *
 * Physics (self-conjugate-DM convention; see Template_Config.sh + docs):
 *   annihilation: dE/dt/m_gas = f_h^ann * (<sigma v>/m_chi) * rho_DM^2 * c^2 / rho_gas
 *   decay:        dE/dt/m_gas = f_h^dec * Gamma * rho_DM * c^2 / rho_gas
 *
 * No MC, no daughter velocity kicks, no DM mass-loss bookkeeping (Gamma*t << 1
 * assumed in any realistic run). v-dependence of <sigma v> is folded by the
 * user into the supplied scalar coefficient. Host-only: per-cell scalar add
 * is cheap; no GPU port needed.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO 2026.
 */

#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"

#ifdef DM_HEATING

void apply_dm_heating(void)
{
    /* Speed of light in code velocity units. UNIT_VEL_IN_CGS is the
     * cm/s value of one code velocity unit, so c_code = C_LIGHT_CGS / UNIT_VEL_IN_CGS
     * gives c in code-velocity units; squaring gives c^2 in code-(velocity^2). */
    const double c_code  = C_LIGHT_CGS / UNIT_VEL_IN_CGS;
    const double c2_code = c_code * c_code;

    const double sv_over_m = All.DM_AnnihilationSigmaV_over_mChi;  /* code units, set in set_units() */
    const double fh_ann    = All.DM_AnnihilationHeatingFraction;
    const double Gamma     = All.DM_DecayRate;                     /* code units */
    const double fh_dec    = All.DM_DecayHeatingFraction;

    const int ann_on = (sv_over_m > 0) && (fh_ann > 0);
    const int dec_on = (Gamma     > 0) && (fh_dec > 0);
    if (!ann_on && !dec_on) return;

    for (int i : ActiveParticleList) {
        if (P[i].Type != 0) continue;
        if (P[i].Mass <= 0) continue;

        const double rho_dm  = (double)CellP[i].DM_Rho;                 /* physical, finalized in dm_dispersion_finalize_post_runner */
        const double rho_gas = (double)CellP[i].Density * All.cf_a3inv; /* physical */

        /* Hard abort on non-finite input. Neither field can legitimately be
         * NaN/Inf at this point in the step: dispersion finalize zero-inits
         * when Ngb=0, and hydro just populated Density. A non-finite value
         * here indicates an upstream bug; let it propagate into DtInternalEnergy
         * → cooling → silent cell corruption. Loud failure matches the
         * never-silent-degradation rule. */
        if (!isfinite(rho_dm) || !isfinite(rho_gas)) {
            printf("apply_dm_heating: non-finite input at i=%d (rho_dm=%g, rho_gas=%g, ThisTask=%d)\n",
                   i, rho_dm, rho_gas, ThisTask);
            endrun(91301);
            continue; /* soft-stop + skip: avoid injecting NaN/Inf into CellP[i].DtInternalEnergy (the corruption this guard exists to prevent); local loop drains at next poll */
        }

        if (rho_gas <= 0 || rho_dm <= 0) continue;  /* no DM here, or unresolved gas */

        double dEdt_specific = 0;
        if (ann_on) dEdt_specific += fh_ann * sv_over_m * rho_dm * rho_dm * c2_code / rho_gas;
        if (dec_on) dEdt_specific += fh_dec * Gamma     * rho_dm           * c2_code / rho_gas;
        CellP[i].DtInternalEnergy += dEdt_specific;
    }
}

#endif /* DM_HEATING */
