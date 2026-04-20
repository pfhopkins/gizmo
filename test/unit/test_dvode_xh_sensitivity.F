#include "../../GIZMO_config.h"
c=======================================================================
c  Sweep X_H at fixed gas mass density rho. yn = rho*X_H/m_p is computed
c  per scenario and passed to DVODE. ABHE is compile-time (set via
c  -DABHE=... at build time); rebuild and rerun to sweep ABHE.
c
c  For each X_H value, run consecutive evolve_abundances calls until a
c  total integration time of t_total, then report final T, H+, H2, CO.
c=======================================================================
      program test_dvode_xh_sensitivity
      implicit none
#include "cool.h"
#include "non_eq.h"
      REAL dl, rho, t_total, dt
      integer nsteps
      REAL xh_vals(3), t_vals(7), z_vals(5)
      REAL abundc_sol, abundo_sol, abundsi_sol
      integer i, j, k

      REAL PROTONMASS_LOC
      parameter(PROTONMASS_LOC = 1.6726231d-24)

c --- physical inputs ---
      rho       = 100d0 * PROTONMASS_LOC / 0.76d0  ! ~ yn=100 at X_H=0.76
      t_total   = 1d5 * 3.1557d7    ! 1e5 yr
      dt        = 1d2 * 3.1557d7    ! 100 yr per call
      nsteps    = int(t_total / dt)
      dl        = 3.086d18

      xh_vals(1) = 0.76d0
      xh_vals(2) = 0.45d0
      xh_vals(3) = 0.30d0

      t_vals(1) = 1d1
      t_vals(2) = 1d2
      t_vals(3) = 1d3
      t_vals(4) = 1d4
      t_vals(5) = 1d5
      t_vals(6) = 1d6
      t_vals(7) = 1d7

c --- Z grid ---
#if CHEMISTRYNETWORK == 17
      z_vals(1) = 0d0       ! pristine (He+ should provide e-)
      z_vals(2) = 0.01d0    ! 0.01 solar
      z_vals(3) = 0.1d0     ! 0.1 solar
      z_vals(4) = 1d0       ! solar
      z_vals(5) = 10d0      ! 10x solar
#else
c --- network 5 has no He/D/HD chemistry so Z=0 is ill-posed ---
      z_vals(1) = 0.01d0    ! 0.01 solar
      z_vals(2) = 0.1d0     ! 0.1 solar
      z_vals(3) = 1d0       ! solar
      z_vals(4) = 3d0       ! 3x solar
      z_vals(5) = 10d0      ! 10x solar (SN-enriched)
#endif

      abundc_sol  = 1.41d-4
      abundo_sol  = 3.16d-4
      abundsi_sol = 1.50d-5

c --- chemcool common-block parameters ---
      deff = 1d0
      G0 = 1.7d0
      G0_dust = 1.7d0
      G0_LW = 1.7d0
      phi_pah = 0.5d0
      abundc = 0d0       ! overwritten per Z below
      abundo = 0d0
      abundsi = 0d0
#if CHEMISTRYNETWORK == 17
      abundD = 2.6d-5    ! primordial D/H
#else
      abundD = 0d0
#endif
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

      print*, '=================================================='
      print*, ' DVODE sensitivity to (X_H, T, Z) at fixed ABHE'
      print*, ' rho   =', rho, ' g/cm^3'
      print*, ' t_tot =', t_total/3.1557d7, ' yr'
      print*, ' nsteps=', nsteps
      print*, ' ABHE  =', abhe, ' (compile-time)'
      print*, '=================================================='
      print*, '   T0      X_H     Z/Zsol     yn       ne       ',
     $        '   H+_final    H2_final   CO_final   T_final  status'

      do j = 1, 7
        do k = 1, 5
          abundc  = abundc_sol  * z_vals(k)
          abundo  = abundo_sol  * z_vals(k)
          abundsi = abundsi_sol * z_vals(k)
          do i = 1, 3
            call run_one(rho, xh_vals(i), t_vals(j), dt, nsteps,
     $                   dl, z_vals(k))
          enddo
        enddo
      enddo

      end

c=======================================================================
c  One run at (rho, X_H). yn = rho*X_H/m_p. ABHE comes from compile.
c=======================================================================
      subroutine run_one(rho, xh, temp_init, dt, nsteps, dl, zfrac)
      implicit none
#include "cool.h"
#include "non_eq.h"
      REAL rho, xh, temp_init, dt, dl, zfrac
      integer nsteps

      REAL y(nspec), rpar(nrpar)
      REAL energy, column_est, divv_loc, yn, t_final, abe, ne
      integer istep
      character*8 status

      REAL PROTONMASS_LOC
      parameter(PROTONMASS_LOC = 1.6726231d-24)

      yn = rho * xh / PROTONMASS_LOC
      divv_loc = 0d0

c --- initial abundances scaled by T0 ---
      if (temp_init .lt. 1d3) then
        y(ih2) = 0.1d0
        y(ihp) = 1d-6
      elseif (temp_init .lt. 5d4) then
        y(ih2) = 1d-10
        y(ihp) = 1d-2
      else
        y(ih2) = 1d-12
        y(ihp) = 0.99d0
      endif
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6
      y(ico) = 1d-10
#endif
#if CHEMISTRYNETWORK == 17
c --- minimal seeds, let chemistry relax to equilibrium ---
      y(ico)   = 0d0
      y(ihd)   = 0d0
      y(ihep)  = 0d0
      y(ihepp) = 0d0
      y(idp)   = 0d0
#endif

      energy = 1.5d0 * BOLTZMANN * temp_init
     $       * (1d0 + abhe - y(ih2) + y(ihp) + abundc + abundsi)
     $       * yn
      column_est = yn * dl
      rpar(1) = yn
      rpar(2) = dl
      rpar(3) = 0d0
      rpar(5) = column_est
      call calc_photo_wrapper(temp_init, rpar, y(ih2), y(ihp),
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6
     $                        y(ico))
#else
     $                        0d0)
#endif
      y(itmp) = energy

      status = 'OK'
      do istep = 1, nsteps
        energy = y(itmp)
        call evolve_abundances(dt, dl, yn, divv_loc,
     $                         energy, y, column_est)
        if (y(ihp) .ne. y(ihp)) then
          status = 'CRASH'
          goto 99
        endif
        if (y(ihp) .gt. 1d0 + 1d-4) then
          status = 'OVER_HP'
          goto 99
        endif
      enddo

 99   continue

c --- recover T from energy: e = 1.5*kT*yn*(1+abhe-H2+abe+Z) ---
#if CHEMISTRYNETWORK == 17
      abe = y(ihp) + y(ihep) + 2d0*y(ihepp) + y(idp)
     $      + abundc + abundsi
#else
      abe = y(ihp) + abundc + abundsi
#endif
      ne  = abe * yn
      t_final = y(itmp) / (1.5d0 * BOLTZMANN * yn
     $        * (1d0 + abhe - y(ih2) + abe))

#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6
      write(*,'(ES8.1,2X,F6.3,2X,ES8.1,2X,ES10.3,2X,ES10.3,2X,
     $          ES10.3,2X,ES10.3,2X,ES10.3,2X,ES10.3,2X,A)')
     $  temp_init, xh, zfrac, yn, ne, y(ihp), y(ih2),
     $  y(ico), t_final, status
#else
      write(*,'(ES8.1,2X,F6.3,2X,ES8.1,2X,ES10.3,2X,ES10.3,2X,
     $          ES10.3,2X,ES10.3,2X,ES10.3,2X,ES10.3,2X,A)')
     $  temp_init, xh, zfrac, yn, ne, y(ihp), y(ih2),
     $  0d0, t_final, status
#endif
      return
      end
