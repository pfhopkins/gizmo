/* disk_betacool.h — minimal beta-cooling + constant irradiation floor.
 *
 * Activated by DISK_BETA_COOL. Mutually exclusive with COOLING.
 *
 * du/dt = -(u - u_irr) / t_cool, with t_cool = beta / Omega.
 * Integrated as exact exponential per cell per timestep:
 *     u_new = u_irr + (u_old - u_irr) * exp(-dt/t_cool)
 *
 * Omega comes from BOX_SHEARING_OMEGA_BOX_CENTER if BOX_SHEARING is set;
 * otherwise we assume a central potential at the origin and read
 * Omega from the local gravitational acceleration as
 *     Omega = sqrt(|a|/|r|)   (=> |a| = Omega^2 r for a Keplerian orbit).
 *
 * u_irr is a single global, computed at startup from BetaCool_Tirr.
 * Set BetaCool_Tirr = 0 to disable the floor (pure cooling to zero).
 *
 * This is intentionally the simplest possible implementation. A "real"
 * version would use a Stamatellos-Whitworth-style position- and
 * column-dependent irradiation floor with optical-depth attenuation,
 * tied to a stellar luminosity and dust opacity. None of that is here:
 * the proper home for that physics is real RHD with coupled dust-gas
 * thermal physics (RT_OPACITY_FROM_EXPLICIT_GRAINS), which is what
 * this module is a modest approximation for. Don't grow this file.
 */
#ifndef DISK_BETACOOL_H
#define DISK_BETACOOL_H

#ifdef DISK_BETA_COOL

/* Relies on includer (currently core/run.cc) having already pulled in
 * declarations/allvars.h and core/proto.h. Those headers are not double-include
 * safe, so we deliberately do not include them here. */

#include <math.h>

static inline void disk_betacool_parent_routine(void)
{
    const double u_irr  = All.BetaCool_u_irr;
    const double beta   = All.BetaCool_Beta;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 0 || P[i].Mass <= 0) continue;
        if(!TimeBinActive[P[i].TimeBin]) continue;
        double dt = get_particle_timestep_in_physical(i, P);
        if(dt <= 0) continue;

#ifdef BOX_SHEARING
        double Omega = (double)BOX_SHEARING_OMEGA_BOX_CENTER;
#else
        double r2 = P[i].Pos[0]*P[i].Pos[0] + P[i].Pos[1]*P[i].Pos[1] + P[i].Pos[2]*P[i].Pos[2];
        if(r2 <= 0) continue;
        double a2 = P[i].GravAccel[0]*P[i].GravAccel[0]
                  + P[i].GravAccel[1]*P[i].GravAccel[1]
                  + P[i].GravAccel[2]*P[i].GravAccel[2];
        double Omega = sqrt(sqrt(a2 / r2));
#endif
        if(Omega <= 0) continue;

        double tcool = beta / Omega;
        double uold  = DMAX(All.MinEgySpec, CellP[i].InternalEnergy);
        double unew  = u_irr + (uold - u_irr) * exp(-dt / tcool);
        unew = DMAX(All.MinEgySpec, unew);

        CellP[i].InternalEnergy     = unew;
        CellP[i].InternalEnergyPred = unew;
        CellP[i].DtInternalEnergy   = 0;
    }
}

#endif // DISK_BETA_COOL
#endif // DISK_BETACOOL_H
