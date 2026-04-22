#include "../../GIZMO_config.h"
c=======================================================================
c  Test the user's hypothesis that the metal-diffusion / chemistry
c  inconsistency for He drives an H2 overshoot.
c
c  Setup: feed the chemistry an initial state where abhep > ABHE
c  (the bound the chemistry's compile-time He budget enforces).
c  This is exactly what diffusion can deliver when X_He/(4*X_H) of a
c  cell exceeds the compile-time ABHE and the GALSF_CHEMCOOL_VARIABLE_
c  XH_AND_ABHE flag is OFF, so chemistry sees fixed ABHE.
c
c  Inside rate_eq_ulli, abheI = max(0, ABHE - abhep - abhepp) clamps
c  the neutral-He pool to zero -- killing both H2 formation channel
c  (2H + He -> H2) and H2 destruction channel (H2 + He). Net effect
c  on abh2 is unclear analytically; this test runs DVODE for many
c  steps to see if abh2 drifts past 0.5.
c
c  Sweep: abhep / ABHE ratio = {1.0, 1.1, 1.5, 2.0, 5.0}
c         T0 = {100, 1000, 10000} K
c=======================================================================
      program test_chemistry_abhep_overshoot
      implicit none
#include "cool.h"
#include "non_eq.h"
      REAL dl, rho, t_total, dt, yn
      integer nsteps
      REAL ratios(5), t_vals(3)
      integer i, j, nargs
      character(len=64) arg_t, arg_r
      REAL t_arg, r_arg

      REAL PROTONMASS_LOC
      parameter(PROTONMASS_LOC = 1.6726231d-24)

      rho     = 100d0 * PROTONMASS_LOC / 0.76d0   ! yn=100 at X_H=0.76
      yn      = rho * 0.76d0 / PROTONMASS_LOC
      t_total = 1d6 * 3.1557d7                    ! 1e6 yr (long)
      dt      = 1d2 * 3.1557d7                    ! 100 yr per call
      nsteps  = int(t_total / dt)
      dl      = 3.086d18

      ratios(1) = 1.0d0
      ratios(2) = 1.1d0
      ratios(3) = 1.5d0
      ratios(4) = 2.0d0
      ratios(5) = 5.0d0

      t_vals(1) = 1d2
      t_vals(2) = 1d3
      t_vals(3) = 1d4

c --- chemcool common-block defaults ---
      deff = 1d0
      G0 = 1.7d0
      G0_dust = 1.7d0
      G0_LW = 1.7d0
      phi_pah = 0.5d0
      abundc = 1.41d-5    ! 0.1 solar
      abundo = 3.16d-5
      abundsi = 0d0
      abundD = 2.6d-5
      abundN = 0d0
      abundM = 0d0
      iflag_ad = 1
      iflag_mn = 1
      iflag_atom = 2
      iflag_h3pra = 1
      iflag_3bh2a = 1
      iflag_3bh2b = 1
      iphoto = 1
      isrf_option = 1
      f_rsc = 1d0
      AV_conversion_factor = 5.348d-22
      AV_ext = 0d0
      h2_form_ex = 0d0
      h2_form_kin = 0d0
      redshift = 0d0
      tdust = 15d0
      no_chem = 0
      dm_density = 0d0
      cr_energy_density = 0d0
      pdv_term = 0d0
      cosmic_ray_ion_rate = 2d-16
      dust_to_gas_ratio = 1d-2

      call coolinmo
      call cheminmo
      call init_tolerances
      call load_h2_table
      call init_temperature_lookup

      nargs = command_argument_count()
      if (nargs .ge. 2) then
        call get_command_argument(1, arg_t)
        call get_command_argument(2, arg_r)
        read(arg_t, *) t_arg
        read(arg_r, *) r_arg
        print*, '# T0=', t_arg, ' ratio=', r_arg, ' ABHE=', abhe
        call run_one(rho, t_arg, r_arg, dt, nsteps, dl)
      else
        print*, '=================================================='
        print*, ' Chemistry: does abhep > ABHE drive abh2 overshoot?'
        print*, ' ABHE (compile-time) =', abhe
        print*, ' nsteps =', nsteps, ' dt =', dt/3.1557d7, ' yr each'
        print*, '=================================================='
        print*, '  T0      abhep0/ABHE   abh2_init   abh2_max',
     $          '   abh2_final   T_final  status'
        do j = 1, 3
          do i = 1, 5
            call run_one(rho, t_vals(j), ratios(i), dt, nsteps, dl)
          enddo
        enddo
      endif

      end

c=======================================================================
c  One run: initial abhep = ratio * ABHE (set ABOVE ABHE if ratio>1).
c  Track max abh2 over the integration. Report whether DVODE crashed.
c=======================================================================
      subroutine run_one(rho, temp_init, ratio, dt, nsteps, dl)
      implicit none
#include "cool.h"
#include "non_eq.h"
      REAL rho, temp_init, ratio, dt, dl
      integer nsteps

      REAL y(nspec), rpar(nrpar)
      REAL energy, column_est, divv_loc, yn, t_final, abe_loc
      REAL max_abh2, abh2_init
      integer istep, did_crash

      REAL PROTONMASS_LOC
      parameter(PROTONMASS_LOC = 1.6726231d-24)

      yn = rho * 0.76d0 / PROTONMASS_LOC
      divv_loc = 0d0

c --- initial conditions: cold molecular gas ---
      y(ih2) = 0.20d0           ! moderate H2
      y(ihp) = 1d-6
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6 || CHEMISTRYNETWORK == 17
      y(ico) = 1d-10
#endif
#if CHEMISTRYNETWORK == 17
      y(ihep)  = ratio * abhe   ! THE TEST: above-ABHE He+ population
      y(ihepp) = 0d0
      y(idp)   = 1d-7
      y(ihd)   = 1d-7
#endif
      abh2_init = y(ih2)
      max_abh2  = y(ih2)

      energy = 1.5d0 * BOLTZMANN * temp_init
     $       * (1d0 + abhe - y(ih2) + y(ihp) + abundc + abundsi)
     $       * yn
      column_est = yn * dl
      rpar(1) = yn
      rpar(2) = dl
      rpar(3) = 0d0
      rpar(5) = column_est
      call calc_photo_wrapper(temp_init, rpar, y(ih2), y(ihp),
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6 || CHEMISTRYNETWORK == 17
     $                        y(ico))
#else
     $                        0d0)
#endif
      y(itmp) = energy

      did_crash = 0
      do istep = 1, nsteps
        energy = y(itmp)
        call evolve_abundances(dt, dl, yn, divv_loc,
     $                         energy, y, column_est)
        if (y(ih2) .ne. y(ih2)) then
          did_crash = 1
          goto 99
        endif
        if (y(ih2) .gt. max_abh2) max_abh2 = y(ih2)
        if (y(ih2) .gt. 0.5d0) then
          did_crash = 2
          goto 99
        endif
      enddo
 99   continue

c --- recover T from energy ---
#if CHEMISTRYNETWORK == 17
      abe_loc = y(ihp) + y(ihep) + 2d0*y(ihepp) + y(idp)
     $          + abundc + abundsi
#else
      abe_loc = y(ihp) + abundc + abundsi
#endif
      t_final = y(itmp) / (1.5d0 * BOLTZMANN * yn
     $        * (1d0 + abhe - y(ih2) + abe_loc))

      if (did_crash .eq. 1) then
        write(*,'(F8.1,2X,F8.2,4X,ES10.3,2X,ES10.3,2X,
     $            ES10.3,2X,ES10.3,2X,A)')
     $    temp_init, ratio, abh2_init, max_abh2, y(ih2),
     $    t_final, 'NaN'
      elseif (did_crash .eq. 2) then
        write(*,'(F8.1,2X,F8.2,4X,ES10.3,2X,ES10.3,2X,
     $            ES10.3,2X,ES10.3,2X,A)')
     $    temp_init, ratio, abh2_init, max_abh2, y(ih2),
     $    t_final, 'OVER_H2'
      else
        write(*,'(F8.1,2X,F8.2,4X,ES10.3,2X,ES10.3,2X,
     $            ES10.3,2X,ES10.3,2X,A)')
     $    temp_init, ratio, abh2_init, max_abh2, y(ih2),
     $    t_final, 'OK'
      endif
      return
      end
