#include "../../GIZMO_config.h"
c=======================================================================
c  Test: does DVODE overshoot H+ past 1.0 + eps_max after SN heating?
c
c  Scenario: cold neutral gas (T ~ 10-100 K, H+ ~ 1e-6, H2 ~ 0.1)
c  suddenly heated to T ~ 1e6 - 1e8 K by SN thermal injection.
c  Hydrogen should fully ionize (H+ -> 1.0), but DVODE might overshoot.
c
c  The crash in job 150573 had:
c    y(ihp) = 1.00014 > 1 + eps_max (1e-4)
c    -> validate_input sets ipar(1)=1 -> ydot=0 -> crash
c
c  Sweep: density, post-SN temperature, metallicity (abundc, abundo),
c         initial H2, timestep
c=======================================================================
      program test_dvode_hp_overshoot
      implicit none
#include "cool.h"
#include "non_eq.h"
      REAL dl
      integer n_fail, n_total, n_crash, n_clamp

c --- grid dimensions ---
      integer nyn, ntemp, nZc, nZo, ndt, nh2
      parameter (nyn=4, ntemp=6, nZc=5, nZo=5, ndt=4, nh2=3)

      REAL yn_vals(nyn), temp_vals(ntemp)
      REAL Zc_vals(nZc), Zo_vals(nZo)
      REAL dt_vals(ndt), h2_vals(nh2)

      integer iyn, itemp, iZc, iZo, idt, ih2i

c --- density (cm^-3) ---
      yn_vals(1) = 1d0
      yn_vals(2) = 1d1
      yn_vals(3) = 1d2
      yn_vals(4) = 1d3

c --- post-SN temperature (K) ---
c  All above the critical T ~ 7e4 K where H+ eq > 1 - eps_max
      temp_vals(1) = 1d5
      temp_vals(2) = 3d5
      temp_vals(3) = 1d6
      temp_vals(4) = 1d7
      temp_vals(5) = 3d7
      temp_vals(6) = 1d8

c --- abundc (C/H number ratio) ---
c  From pristine to extreme SN enrichment
      Zc_vals(1) = 1.41d-5      ! 0.1 solar
      Zc_vals(2) = 1.41d-4      ! solar
      Zc_vals(3) = 1.41d-3      ! 10x solar
      Zc_vals(4) = 1.41d-2      ! 100x solar (SN ejecta)
      Zc_vals(5) = 7.05d-2      ! 500x solar (extreme)

c --- abundo (O/H number ratio) ---
      Zo_vals(1) = 3.16d-5      ! 0.1 solar
      Zo_vals(2) = 3.16d-4      ! solar
      Zo_vals(3) = 3.16d-3      ! 10x solar
      Zo_vals(4) = 3.16d-2      ! 100x solar (SN ejecta)
      Zo_vals(5) = 1.58d-1      ! 500x solar (extreme)

c --- timestep (seconds) ---
      dt_vals(1) = 1d1 * 3.1557d7     ! 10 yr
      dt_vals(2) = 1d2 * 3.1557d7     ! 100 yr
      dt_vals(3) = 1d3 * 3.1557d7     ! 1000 yr
      dt_vals(4) = 1d4 * 3.1557d7     ! 10000 yr

c --- initial H2 fraction ---
      h2_vals(1) = 1d-6        ! essentially atomic
      h2_vals(2) = 1d-2        ! slightly molecular
      h2_vals(3) = 0.1d0       ! molecular cloud

c --- fixed parameters ---
      dl = 3.086d18
      deff = 1d0
      G0 = 0d0
      G0_dust = 0d0
      G0_LW = 0d0
      G0_NUV = 0d0
      G0_OPT = 0d0
      chi_NUV = 0d0
      chi_OPT = 0d0
      phi_pah = 0.5d0
      abundsi = 0d0
      abundD = 0d0
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

c --- init tables ---
      call coolinmo
      call cheminmo
      call init_tolerances
      call load_h2_table
      call init_temperature_lookup

      n_fail  = 0
      n_crash = 0
      n_clamp = 0
      n_total = 0

      print*, '=================================================='
      print*, 'DVODE H+ overshoot test after SN heating'
      print*, 'Initial: cold neutral gas (H+~1e-6, H2 variable)'
      print*, 'Post-SN: T = 1e5 - 1e8 K'
      print*, '=================================================='
      print*, ' '

c =================================================================
c  BLOCK 1: density x temperature (solar metallicity)
c =================================================================
      print*, '--- Block 1: yn x T (solar Z) ---'
      do iyn = 1, nyn
       do itemp = 1, ntemp
        call run_hp_test(yn_vals(iyn), temp_vals(itemp),
     $    1.41d-4, 3.16d-4, 1d3*3.1557d7, 1d-2,
     $    dl, n_total, n_fail, n_crash, n_clamp)
       enddo
      enddo

c =================================================================
c  BLOCK 2: density x temperature x HIGH metallicity
c  This is the crash regime — SN-enriched gas
c =================================================================
      print*, '--- Block 2: yn x T x high Z ---'
      do iyn = 1, nyn
       do itemp = 1, ntemp
        do iZc = 3, nZc
c  O-rich (like the crash: abundo >> abundc)
         call run_hp_test(yn_vals(iyn), temp_vals(itemp),
     $     Zc_vals(iZc), 1.58d-1, 1d3*3.1557d7, 1d-2,
     $     dl, n_total, n_fail, n_crash, n_clamp)
c  C-rich
         call run_hp_test(yn_vals(iyn), temp_vals(itemp),
     $     7.05d-2, Zo_vals(3), 1d3*3.1557d7, 1d-2,
     $     dl, n_total, n_fail, n_crash, n_clamp)
        enddo
       enddo
      enddo

c =================================================================
c  BLOCK 3: exact crash conditions (particle 3095686)
c =================================================================
      print*, '--- Block 3: exact crash conditions ---'
c  n=17.17, T~1.9e7 (from 1e51/32 into 4 Msun), high O
c  abundc=0.0125, abundo=0.054
      call run_hp_test(17.17d0, 1.9d7,
     $  1.25d-2, 5.4d-2, 1d3*3.1557d7, 0.074d0,
     $  dl, n_total, n_fail, n_crash, n_clamp)
c  Same but with 100 yr timestep
      call run_hp_test(17.17d0, 1.9d7,
     $  1.25d-2, 5.4d-2, 1d2*3.1557d7, 0.074d0,
     $  dl, n_total, n_fail, n_crash, n_clamp)
c  Same but with 10 yr timestep
      call run_hp_test(17.17d0, 1.9d7,
     $  1.25d-2, 5.4d-2, 1d1*3.1557d7, 0.074d0,
     $  dl, n_total, n_fail, n_crash, n_clamp)

c =================================================================
c  BLOCK 4: timestep sweep at crash metallicity
c =================================================================
      print*, '--- Block 4: dt sweep at crash Z ---'
      do idt = 1, ndt
       call run_hp_test(17.17d0, 1.9d7,
     $   1.25d-2, 5.4d-2, dt_vals(idt), 0.074d0,
     $   dl, n_total, n_fail, n_crash, n_clamp)
      enddo

c =================================================================
c  BLOCK 5: H2 sweep at crash conditions
c =================================================================
      print*, '--- Block 5: H2 sweep at crash conditions ---'
      do ih2i = 1, nh2
       call run_hp_test(17.17d0, 1.9d7,
     $   1.25d-2, 5.4d-2, 1d3*3.1557d7, h2_vals(ih2i),
     $   dl, n_total, n_fail, n_crash, n_clamp)
      enddo

c =================================================================
c  BLOCK 6: metallicity sweep at fixed T=3e7
c =================================================================
      print*, '--- Block 6: metallicity sweep ---'
      do iZc = 1, nZc
       do iZo = 1, nZo
        call run_hp_test(17d0, 3d7,
     $    Zc_vals(iZc), Zo_vals(iZo), 1d3*3.1557d7, 0.05d0,
     $    dl, n_total, n_fail, n_crash, n_clamp)
       enddo
      enddo

c =================================================================
c  Summary
c =================================================================
      print*, ' '
      print*, '=================================================='
      print*, 'Results:', n_total, ' tests'
      print*, '  H+ overshoots (>1+eps_max):', n_fail
      print*, '  crashes (NaN/abort)        :', n_crash
      print*, '  H+ clamped (>1, <1+eps_max):', n_clamp
      if (n_fail .eq. 0 .and. n_crash .eq. 0) then
        print*, 'PASS: DVODE never overshoots H+ past 1+eps_max'
      else
        print*, 'FAIL'
      endif
      print*, '=================================================='

      end

c=======================================================================
      subroutine run_hp_test(yn_in, temp_in, abundc_in,
     $  abundo_in, dt_in, h2_in,
     $  dl_in, n_total, n_fail, n_crash, n_clamp)
c=======================================================================
      implicit none
#include "cool.h"
#include "non_eq.h"
      REAL yn_in, temp_in, abundc_in, abundo_in
      REAL dt_in, h2_in
      REAL dl_in
      integer n_total, n_fail, n_crash, n_clamp

      REAL y(nspec), rpar(nrpar)
      REAL energy, column_est, divv_loc, hp_final
      integer ipar(nipar)

      REAL eps_max_val
      parameter (eps_max_val = 1d-4)

      n_total = n_total + 1

c --- set common block parameters ---
      abundc = abundc_in
      abundo = abundo_in
      G0 = 0d0
      G0_dust = 0d0
      dust_to_gas_ratio = 1d-2

      divv_loc = 0d0

c --- initial abundances: cold neutral gas ---
      y(ih2) = h2_in
      y(ihp) = 1d-6            ! nearly neutral
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6
      y(ico) = 1d-8            ! trace CO
#endif

c --- energy: set to post-SN temperature ---
      energy = 1.5d0 * BOLTZMANN * temp_in
     $       * (1d0 + abhe - y(ih2) + y(ihp) + abundc + abundsi)
     $       * yn_in

      column_est = yn_in * dl_in

c --- photorates ---
      rpar(1) = yn_in
      rpar(2) = dl_in
      rpar(3) = 0d0
      rpar(5) = column_est
      call calc_photo_wrapper(temp_in, rpar, y(ih2), y(ihp),
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6
     $                        y(ico))
#else
     $                        0d0)
#endif

c --- energy into y ---
      y(itmp) = energy

c --- evolve ---
      call evolve_abundances(dt_in, dl_in, yn_in, divv_loc, energy,
     $                       y, column_est)

      hp_final = y(ihp)

c --- check ---
      if (hp_final .ne. hp_final) then
c NaN
        n_crash = n_crash + 1
        write(*,'(A,ES8.1,A,ES8.1,A,ES8.1,A,ES8.1,A,ES8.1)')
     $    ' CRASH yn=',yn_in,' T=',temp_in,
     $    ' Zc=',abundc_in,' Zo=',abundo_in,' dt=',dt_in
      elseif (hp_final .gt. 1d0 + eps_max_val) then
        n_fail = n_fail + 1
        write(*,'(A,ES8.1,A,ES8.1,A,ES8.1,A,ES8.1,A,F10.7)')
     $    ' OVER yn=',yn_in,' T=',temp_in,
     $    ' Zc=',abundc_in,' Zo=',abundo_in,
     $    ' H+=',hp_final
      elseif (hp_final .gt. 1d0) then
        n_clamp = n_clamp + 1
        write(*,'(A,ES8.1,A,ES8.1,A,ES8.1,A,ES8.1,A,F10.7,A)')
     $    ' CLAMP yn=',yn_in,' T=',temp_in,
     $    ' Zc=',abundc_in,' Zo=',abundo_in,
     $    ' H+=',hp_final,' (clamped to 1)'
      else
        write(*,'(A,ES8.1,A,ES8.1,A,ES8.1,A,ES8.1,A,F10.7)')
     $    '  OK  yn=',yn_in,' T=',temp_in,
     $    ' Zc=',abundc_in,' Zo=',abundo_in,
     $    ' H+=',hp_final
      endif

      return
      end
