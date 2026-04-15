#include "../../GIZMO_config.h"
c=======================================================================
c  Test: can DVODE overshoot CO past min(abundc, abundo)?
c
c  Sweep across: yn, T, CO/ceiling, G0, dust_to_gas, CR rate,
c                abundc, abundo, dt, divv, H2 fraction
c  Tests both C-limited (abundc < abundo) and O-limited regimes.
c=======================================================================
      program test_dvode_co
      implicit none
#include "cool.h"
#include "non_eq.h"
      REAL dl
      integer n_fail, n_total, n_crash

c --- grid dimensions ---
      integer nyn, ntemp, nfrac, nG0, ndgr, ncr, nZc, nZo
      integer ndt, ndivv, nh2
      parameter (nyn=5, ntemp=5, nfrac=6, nG0=4, ndgr=4, ncr=3)
      parameter (nZc=4, nZo=4, ndt=4, ndivv=3, nh2=3)

      REAL yn_vals(nyn), temp_vals(ntemp), frac_vals(nfrac)
      REAL G0_vals(nG0), dgr_vals(ndgr), cr_vals(ncr)
      REAL Zc_vals(nZc), Zo_vals(nZo)
      REAL dt_vals(ndt), divv_vals(ndivv), h2_vals(nh2)

      integer iyn, itemp, ifrac, iG0, idgr, icr, iZc, iZo
      integer idt, idivv, ih2i

c --- density ---
      yn_vals(1) = 1d1
      yn_vals(2) = 1d2
      yn_vals(3) = 1d3
      yn_vals(4) = 1d5
      yn_vals(5) = 1d7

c --- temperature ---
      temp_vals(1) = 10d0
      temp_vals(2) = 50d0
      temp_vals(3) = 300d0
      temp_vals(4) = 3000d0
      temp_vals(5) = 3d4

c --- CO fraction of min(abundc, abundo) ---
      frac_vals(1) = 0.0d0
      frac_vals(2) = 0.5d0
      frac_vals(3) = 0.9d0
      frac_vals(4) = 0.99d0
      frac_vals(5) = 0.999d0
      frac_vals(6) = 1.0d0

c --- radiation field G0 ---
      G0_vals(1) = 0d0
      G0_vals(2) = 1.7d0
      G0_vals(3) = 100d0
      G0_vals(4) = 1d4

c --- dust to gas ratio ---
      dgr_vals(1) = 1d-4
      dgr_vals(2) = 1d-3
      dgr_vals(3) = 1d-2
      dgr_vals(4) = 1d-1

c --- cosmic ray ionization rate ---
      cr_vals(1) = 2d-17
      cr_vals(2) = 2d-16
      cr_vals(3) = 2d-14

c --- abundc ---
      Zc_vals(1) = 1.41d-6
      Zc_vals(2) = 1.41d-5
      Zc_vals(3) = 1.41d-4
      Zc_vals(4) = 1.41d-3

c --- abundo ---
      Zo_vals(1) = 3.16d-6
      Zo_vals(2) = 3.16d-5
      Zo_vals(3) = 3.16d-4
      Zo_vals(4) = 3.16d-3

c --- timestep ---
      dt_vals(1) = 1d2 * 3.1557d7
      dt_vals(2) = 1d4 * 3.1557d7
      dt_vals(3) = 1d6 * 3.1557d7
      dt_vals(4) = 1d8 * 3.1557d7

c --- velocity divergence ---
      divv_vals(1) = 0d0
      divv_vals(2) = -1d-13
      divv_vals(3) = 1d-13

c --- H2 initial fraction ---
      h2_vals(1) = 1d-6
      h2_vals(2) = 1d-2
      h2_vals(3) = 0.4d0

c --- fixed parameters ---
      dl = 3.086d18
      deff = 1d0
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

c --- init tables ---
      call coolinmo
      call cheminmo
      call init_tolerances
      call load_h2_table
      call init_temperature_lookup

      n_fail  = 0
      n_crash = 0
      n_total = 0
      n_co_clips = 0
      n_co_rejects = 0

      print*, '=================================================='
      print*, 'DVODE CO overshoot test - full parameter sweep'
      print*, 'Varying: yn, T, CO/ceil, G0, DGR, CR, Zc, Zo,'
      print*, '         dt, divv, H2_init'
      print*, '=================================================='
      print*, ' '

c =================================================================
c  BLOCK 1: yn x T x frac  (defaults)
c =================================================================
      print*, '--- Block 1: yn x T x frac ---'
      do iyn = 1, nyn
       do itemp = 1, ntemp
        do ifrac = 1, nfrac
          call run_one_test(yn_vals(iyn), temp_vals(itemp),
     $      frac_vals(ifrac), 1.41d-4, 3.16d-4,
     $      1.7d0, 1d-2, 2d-16,
     $      1d4*3.1557d7, 0d0, 1d-3,
     $      dl, n_total, n_fail, n_crash)
        enddo
       enddo
      enddo

c =================================================================
c  BLOCK 2: G0 sweep
c =================================================================
      print*, '--- Block 2: G0 sweep ---'
      do iG0 = 1, nG0
       do itemp = 1, ntemp
        call run_one_test(1d4, temp_vals(itemp),
     $    0.99d0, 1.41d-4, 3.16d-4,
     $    G0_vals(iG0), 1d-2, 2d-16,
     $    1d4*3.1557d7, 0d0, 1d-3,
     $    dl, n_total, n_fail, n_crash)
       enddo
      enddo

c =================================================================
c  BLOCK 3: dust-to-gas sweep
c =================================================================
      print*, '--- Block 3: DGR sweep ---'
      do idgr = 1, ndgr
       do itemp = 1, ntemp
        call run_one_test(1d4, temp_vals(itemp),
     $    0.99d0, 1.41d-4, 3.16d-4,
     $    1.7d0, dgr_vals(idgr), 2d-16,
     $    1d4*3.1557d7, 0d0, 1d-3,
     $    dl, n_total, n_fail, n_crash)
       enddo
      enddo

c =================================================================
c  BLOCK 4: CR rate sweep
c =================================================================
      print*, '--- Block 4: CR rate sweep ---'
      do icr = 1, ncr
       do itemp = 1, ntemp
        call run_one_test(1d4, temp_vals(itemp),
     $    0.99d0, 1.41d-4, 3.16d-4,
     $    1.7d0, 1d-2, cr_vals(icr),
     $    1d4*3.1557d7, 0d0, 1d-3,
     $    dl, n_total, n_fail, n_crash)
       enddo
      enddo

c =================================================================
c  BLOCK 5: abundc sweep (C-limited regime when abundc < abundo)
c =================================================================
      print*, '--- Block 5: abundc sweep ---'
      do iZc = 1, nZc
       do itemp = 1, ntemp
        call run_one_test(1d4, temp_vals(itemp),
     $    0.99d0, Zc_vals(iZc), 3.16d-4,
     $    1.7d0, 1d-2, 2d-16,
     $    1d4*3.1557d7, 0d0, 1d-3,
     $    dl, n_total, n_fail, n_crash)
       enddo
      enddo

c =================================================================
c  BLOCK 5b: abundo sweep (O-limited regime when abundo < abundc)
c =================================================================
      print*, '--- Block 5b: abundo sweep ---'
      do iZo = 1, nZo
       do itemp = 1, ntemp
        call run_one_test(1d4, temp_vals(itemp),
     $    0.99d0, 1.41d-4, Zo_vals(iZo),
     $    1.7d0, 1d-2, 2d-16,
     $    1d4*3.1557d7, 0d0, 1d-3,
     $    dl, n_total, n_fail, n_crash)
       enddo
      enddo

c =================================================================
c  BLOCK 5c: C-rich / O-limited (abundc > abundo, the crash regime)
c =================================================================
      print*, '--- Block 5c: C-rich / O-limited ---'
      do ifrac = 1, nfrac
       do itemp = 1, ntemp
c abundc=1.41e-3 >> abundo=3.16e-4 => CO ceiling = abundo
        call run_one_test(1d4, temp_vals(itemp),
     $    frac_vals(ifrac), 1.41d-3, 3.16d-4,
     $    1.7d0, 1d-2, 2d-16,
     $    1d4*3.1557d7, 0d0, 1d-3,
     $    dl, n_total, n_fail, n_crash)
c abundc = abundo (equal ceiling)
        call run_one_test(1d4, temp_vals(itemp),
     $    frac_vals(ifrac), 1.41d-4, 1.41d-4,
     $    1.7d0, 1d-2, 2d-16,
     $    1d4*3.1557d7, 0d0, 1d-3,
     $    dl, n_total, n_fail, n_crash)
c abundo=1.41e-4 < abundc=3.16e-4 => O-limited
        call run_one_test(1d4, temp_vals(itemp),
     $    frac_vals(ifrac), 3.16d-4, 1.41d-4,
     $    1.7d0, 1d-2, 2d-16,
     $    1d4*3.1557d7, 0d0, 1d-3,
     $    dl, n_total, n_fail, n_crash)
       enddo
      enddo

c =================================================================
c  BLOCK 6: timestep sweep
c =================================================================
      print*, '--- Block 6: dt sweep ---'
      do idt = 1, ndt
       do itemp = 1, ntemp
        call run_one_test(1d4, temp_vals(itemp),
     $    0.99d0, 1.41d-4, 3.16d-4,
     $    1.7d0, 1d-2, 2d-16,
     $    dt_vals(idt), 0d0, 1d-3,
     $    dl, n_total, n_fail, n_crash)
       enddo
      enddo

c =================================================================
c  BLOCK 7: divv sweep
c =================================================================
      print*, '--- Block 7: divv sweep ---'
      do idivv = 1, ndivv
       do itemp = 1, ntemp
        call run_one_test(1d4, temp_vals(itemp),
     $    0.99d0, 1.41d-4, 3.16d-4,
     $    1.7d0, 1d-2, 2d-16,
     $    1d4*3.1557d7, divv_vals(idivv), 1d-3,
     $    dl, n_total, n_fail, n_crash)
       enddo
      enddo

c =================================================================
c  BLOCK 8: H2 fraction sweep
c =================================================================
      print*, '--- Block 8: H2 sweep ---'
      do ih2i = 1, nh2
       do itemp = 1, ntemp
        call run_one_test(1d4, temp_vals(itemp),
     $    0.99d0, 1.41d-4, 3.16d-4,
     $    1.7d0, 1d-2, 2d-16,
     $    1d4*3.1557d7, 0d0, h2_vals(ih2i),
     $    dl, n_total, n_fail, n_crash)
       enddo
      enddo

c =================================================================
c  BLOCK 9: extreme corners
c =================================================================
      print*, '--- Block 9: extreme corners ---'
c Very high Z, dense, cold, no UV
      call run_one_test(1d7, 10d0,
     $  1.0d0, 1.41d-3, 3.16d-3,
     $  0d0, 1d-1, 2d-14,
     $  1d6*3.1557d7, -1d-13, 0.49d0,
     $  dl, n_total, n_fail, n_crash)
c Very low Z, diffuse, hot, strong UV
      call run_one_test(1d1, 3d4,
     $  0.999d0, 1.41d-6, 3.16d-6,
     $  1d4, 1d-4, 2d-17,
     $  1d8*3.1557d7, 1d-13, 1d-6,
     $  dl, n_total, n_fail, n_crash)
c Near-solar, moderate density, CO-saturated
      call run_one_test(1d4, 50d0,
     $  1.0d0, 1.41d-4, 3.16d-4,
     $  1.7d0, 1d-2, 2d-16,
     $  1d6*3.1557d7, 0d0, 0.4d0,
     $  dl, n_total, n_fail, n_crash)
c Dense core, very cold, maximal CO
      call run_one_test(1d6, 10d0,
     $  1.0d0, 1.41d-4, 3.16d-4,
     $  0d0, 1d-2, 2d-16,
     $  1d4*3.1557d7, -1d-12, 0.49d0,
     $  dl, n_total, n_fail, n_crash)
c C-rich extreme: abundc >> abundo, dense, cold
      call run_one_test(1d6, 10d0,
     $  1.0d0, 1.41d-3, 1.41d-4,
     $  0d0, 1d-2, 2d-16,
     $  1d6*3.1557d7, 0d0, 0.49d0,
     $  dl, n_total, n_fail, n_crash)
c O-poor extreme: very low abundo
      call run_one_test(1d5, 50d0,
     $  1.0d0, 1.41d-4, 1.41d-5,
     $  1.7d0, 1d-2, 2d-16,
     $  1d4*3.1557d7, 0d0, 1d-2,
     $  dl, n_total, n_fail, n_crash)

c =================================================================
c  BLOCK 10: high metallicity — both C and O super-solar
c  SN enrichment can locally push to 10-100x solar.
c  Solar: abundc~1.41e-4, abundo~3.16e-4
c =================================================================
      print*, '--- Block 10: high metallicity ---'
c 10x solar C + 10x solar O, dense, cold, CO saturated
      do ifrac = 1, nfrac
       do iyn = 1, nyn
        call run_one_test(yn_vals(iyn), 10d0,
     $    frac_vals(ifrac), 1.41d-3, 3.16d-3,
     $    1.7d0, 1d-1, 2d-16,
     $    1d4*3.1557d7, 0d0, 0.4d0,
     $    dl, n_total, n_fail, n_crash)
       enddo
      enddo
c 50x solar C + 50x solar O
      do ifrac = 1, nfrac
       do itemp = 1, ntemp
        call run_one_test(1d5, temp_vals(itemp),
     $    frac_vals(ifrac), 7.05d-3, 1.58d-2,
     $    0d0, 5d-2, 2d-16,
     $    1d4*3.1557d7, 0d0, 0.4d0,
     $    dl, n_total, n_fail, n_crash)
       enddo
      enddo
c 100x solar C, 10x solar O (extreme C-rich from CCSN ejecta)
      do ifrac = 1, nfrac
       do itemp = 1, ntemp
        call run_one_test(1d5, temp_vals(itemp),
     $    frac_vals(ifrac), 1.41d-2, 3.16d-3,
     $    0d0, 1d-1, 2d-16,
     $    1d4*3.1557d7, 0d0, 0.3d0,
     $    dl, n_total, n_fail, n_crash)
       enddo
      enddo
c 10x solar C, 100x solar O (extreme O-rich from stripped-env SN)
      do ifrac = 1, nfrac
       do itemp = 1, ntemp
        call run_one_test(1d5, temp_vals(itemp),
     $    frac_vals(ifrac), 1.41d-3, 3.16d-2,
     $    0d0, 1d-1, 2d-16,
     $    1d4*3.1557d7, 0d0, 0.3d0,
     $    dl, n_total, n_fail, n_crash)
       enddo
      enddo
c 500x solar C, 500x solar O (post-SN pocket, extreme)
      do ifrac = 1, nfrac
       do itemp = 1, ntemp
        call run_one_test(1d4, temp_vals(itemp),
     $    frac_vals(ifrac), 7.05d-2, 1.58d-1,
     $    0d0, 1d-1, 2d-16,
     $    1d4*3.1557d7, 0d0, 0.3d0,
     $    dl, n_total, n_fail, n_crash)
       enddo
      enddo
c 1000x solar C, 100x solar O (extreme C-dominated pocket)
      do ifrac = 1, nfrac
       do itemp = 1, ntemp
        call run_one_test(1d4, temp_vals(itemp),
     $    frac_vals(ifrac), 1.41d-1, 3.16d-2,
     $    0d0, 1d-1, 2d-16,
     $    1d4*3.1557d7, 0d0, 0.3d0,
     $    dl, n_total, n_fail, n_crash)
       enddo
      enddo
c 100x solar C, 1000x solar O
      do ifrac = 1, nfrac
       do itemp = 1, ntemp
        call run_one_test(1d4, temp_vals(itemp),
     $    frac_vals(ifrac), 1.41d-2, 3.16d-1,
     $    0d0, 1d-1, 2d-16,
     $    1d4*3.1557d7, 0d0, 0.3d0,
     $    dl, n_total, n_fail, n_crash)
       enddo
      enddo
c Near-equal high Z (abundc ~ abundo), CO ceiling is tight
      do ifrac = 1, nfrac
        call run_one_test(1d6, 10d0,
     $    frac_vals(ifrac), 3.0d-3, 3.16d-3,
     $    0d0, 5d-2, 2d-16,
     $    1d6*3.1557d7, -1d-13, 0.49d0,
     $    dl, n_total, n_fail, n_crash)
        call run_one_test(1d4, 300d0,
     $    frac_vals(ifrac), 3.0d-3, 3.16d-3,
     $    1.7d0, 5d-2, 2d-16,
     $    1d4*3.1557d7, 0d0, 1d-2,
     $    dl, n_total, n_fail, n_crash)
      enddo

c =================================================================
c  BLOCK 11: very cold + zero/tiny radiation + high metallicity
c  This is the dangerous regime: gamma_co -> 0 so CO only forms,
c  never destructs.  CO must converge to min(abundc,abundo) without
c  overshooting.  Sweep density, metallicity, CO saturation, and
c  tiny nonzero G0 values that might trigger edge cases.
c =================================================================
      print*, '--- Block 11: cold + no/low radiation + high Z ---'
c T=10K, G0=0, high Z, all densities, all CO fractions
      do ifrac = 1, nfrac
       do iyn = 1, nyn
        do iZc = 1, nZc
c O-rich: ceiling = abundc
         call run_one_test(yn_vals(iyn), 10d0,
     $     frac_vals(ifrac), Zc_vals(iZc), 3.16d-3,
     $     0d0, 1d-2, 2d-16,
     $     1d6*3.1557d7, 0d0, 0.49d0,
     $     dl, n_total, n_fail, n_crash)
c C-rich: ceiling = abundo
         call run_one_test(yn_vals(iyn), 10d0,
     $     frac_vals(ifrac), 3.16d-3, Zc_vals(iZc),
     $     0d0, 1d-2, 2d-16,
     $     1d6*3.1557d7, 0d0, 0.49d0,
     $     dl, n_total, n_fail, n_crash)
        enddo
       enddo
      enddo
c T=10K, G0=0, extreme metallicity (100-1000x solar), long dt
      do ifrac = 1, nfrac
       do idt = 1, ndt
        call run_one_test(1d6, 10d0,
     $    frac_vals(ifrac), 1.41d-2, 3.16d-2,
     $    0d0, 1d-1, 2d-16,
     $    dt_vals(idt), -1d-13, 0.49d0,
     $    dl, n_total, n_fail, n_crash)
        call run_one_test(1d6, 10d0,
     $    frac_vals(ifrac), 1.41d-1, 3.16d-2,
     $    0d0, 1d-1, 2d-16,
     $    dt_vals(idt), -1d-13, 0.49d0,
     $    dl, n_total, n_fail, n_crash)
        call run_one_test(1d6, 10d0,
     $    frac_vals(ifrac), 1.41d-2, 3.16d-1,
     $    0d0, 1d-1, 2d-16,
     $    dt_vals(idt), -1d-13, 0.49d0,
     $    dl, n_total, n_fail, n_crash)
       enddo
      enddo
c Tiny nonzero G0 (edge cases in rate equations)
      do ifrac = 1, nfrac
       call run_one_test(1d5, 10d0,
     $   frac_vals(ifrac), 1.41d-3, 3.16d-3,
     $   1d-10, 1d-2, 2d-16,
     $   1d6*3.1557d7, 0d0, 0.49d0,
     $   dl, n_total, n_fail, n_crash)
       call run_one_test(1d5, 10d0,
     $   frac_vals(ifrac), 1.41d-3, 3.16d-3,
     $   1d-20, 1d-2, 2d-16,
     $   1d6*3.1557d7, 0d0, 0.49d0,
     $   dl, n_total, n_fail, n_crash)
       call run_one_test(1d5, 10d0,
     $   frac_vals(ifrac), 1.41d-3, 3.16d-3,
     $   1d-30, 1d-2, 2d-16,
     $   1d6*3.1557d7, 0d0, 0.49d0,
     $   dl, n_total, n_fail, n_crash)
c  High Z + tiny G0
       call run_one_test(1d6, 10d0,
     $   frac_vals(ifrac), 1.41d-2, 3.16d-2,
     $   1d-10, 1d-1, 2d-16,
     $   1d6*3.1557d7, -1d-13, 0.49d0,
     $   dl, n_total, n_fail, n_crash)
       call run_one_test(1d6, 10d0,
     $   frac_vals(ifrac), 1.41d-1, 3.16d-2,
     $   1d-20, 1d-1, 2d-16,
     $   1d6*3.1557d7, -1d-13, 0.49d0,
     $   dl, n_total, n_fail, n_crash)
      enddo
c Very cold (T=5K — below our usual floor), dense, no UV
      do ifrac = 1, nfrac
       call run_one_test(1d7, 5d0,
     $   frac_vals(ifrac), 1.41d-4, 3.16d-4,
     $   0d0, 1d-2, 2d-17,
     $   1d6*3.1557d7, -1d-13, 0.499d0,
     $   dl, n_total, n_fail, n_crash)
       call run_one_test(1d7, 5d0,
     $   frac_vals(ifrac), 1.41d-3, 3.16d-3,
     $   0d0, 1d-1, 2d-16,
     $   1d6*3.1557d7, -1d-13, 0.499d0,
     $   dl, n_total, n_fail, n_crash)
       call run_one_test(1d7, 5d0,
     $   frac_vals(ifrac), 1.41d-2, 3.16d-2,
     $   0d0, 1d-1, 2d-16,
     $   1d6*3.1557d7, -1d-13, 0.499d0,
     $   dl, n_total, n_fail, n_crash)
      enddo
c Extreme: yn=1e8, T=3K, G0=0, 1000x solar, CO saturated, long dt
      call run_one_test(1d8, 3d0,
     $  1.0d0, 1.41d-1, 3.16d-1,
     $  0d0, 1d-1, 2d-14,
     $  1d8*3.1557d7, -1d-12, 0.499d0,
     $  dl, n_total, n_fail, n_crash)
      call run_one_test(1d8, 3d0,
     $  1.0d0, 1.41d-1, 3.16d-2,
     $  0d0, 1d-1, 2d-14,
     $  1d8*3.1557d7, -1d-12, 0.499d0,
     $  dl, n_total, n_fail, n_crash)
      call run_one_test(1d8, 3d0,
     $  1.0d0, 1.41d-2, 3.16d-1,
     $  0d0, 1d-1, 2d-14,
     $  1d8*3.1557d7, -1d-12, 0.499d0,
     $  dl, n_total, n_fail, n_crash)

c =================================================================
c  Summary
c =================================================================
      print*, ' '
      print*, '=================================================='
      print*, 'Results:', n_total, ' tests'
      print*, '  overshoots:', n_fail
      print*, '  crashes   :', n_crash
      print*, '  CO clips  :', n_co_clips, ' (overshoot < RTOL)'
      print*, '  CO rejects:', n_co_rejects, ' (overshoot > RTOL)'
      if (n_fail .eq. 0 .and. n_crash .eq. 0) then
        print*, 'PASS: DVODE never overshoots CO past min(ac,ao)'
      else
        print*, 'FAIL'
      endif
      print*, '=================================================='

      end

c=======================================================================
      subroutine run_one_test(yn_in, temp_in, frac_in, abundc_in,
     $  abundo_in, G0_in, dgr_in, cr_in, dt_in, divv_in, h2_in,
     $  dl_in, n_total, n_fail, n_crash)
c=======================================================================
      implicit none
#include "cool.h"
#include "non_eq.h"
      REAL yn_in, temp_in, frac_in, abundc_in, abundo_in
      REAL G0_in, dgr_in, cr_in, dt_in, divv_in, h2_in
      REAL dl_in
      integer n_total, n_fail, n_crash

      REAL y(nspec), rpar(nrpar)
      REAL energy, column_est, co_final, co_ceil, divv_loc
      integer ipar(nipar)

      n_total = n_total + 1

c --- set common block parameters ---
      abundc = abundc_in
      abundo = abundo_in
      G0 = G0_in
      G0_dust = G0_in
      dust_to_gas_ratio = dgr_in
      cosmic_ray_ion_rate = cr_in

c --- reinit rates with new CR ---
      call cheminmo

      divv_loc = divv_in

c --- initial abundances (CO ceiling = min(abundc, abundo)) ---
      y(ih2) = h2_in
      y(ihp) = 1d-4
      y(ico) = frac_in * min(abundc_in, abundo_in)

c --- energy ---
      energy = 1.5d0 * BOLTZMANN * temp_in
     $       * (1d0 + abhe - y(ih2) + y(ihp) + abundc + abundsi)
     $       * yn_in

      column_est = yn_in * dl_in

c --- photorates ---
      rpar(1) = yn_in
      rpar(2) = dl_in
      rpar(3) = 0d0
      rpar(5) = column_est
      call calc_photo_wrapper(temp_in, rpar, y(ih2), y(ihp), y(ico))

c --- energy into y ---
      y(itmp) = energy

c --- evolve ---
      call evolve_abundances(dt_in, dl_in, yn_in, divv_loc, energy,
     $                       y, column_est)

      co_final = y(ico)

c --- check ---
      co_ceil = min(abundc, abundo)
      if (co_final .ne. co_final) then
c NaN
        n_crash = n_crash + 1
        write(*,'(A,ES8.1,A,F7.0,A,F5.3,A,ES8.1,A,ES8.1,A,ES8.1)')
     $    ' CRASH yn=',yn_in,' T=',temp_in,' f=',frac_in,
     $    ' G0=',G0_in,' Zc=',abundc_in,' dt=',dt_in
      elseif (co_final .gt. co_ceil * 1.001d0) then
        n_fail = n_fail + 1
        write(*,'(A,ES8.1,A,F7.0,A,F5.3,A,ES10.3,A,ES10.3,'//
     $    'A,ES10.3)')
     $    ' OVER yn=',yn_in,' T=',temp_in,' f=',frac_in,
     $    ' CO=',co_final,' ac=',abundc,' ao=',abundo
      else
        write(*,'(A,ES8.1,A,F7.0,A,F5.3,A,ES8.1,A,ES8.1,'//
     $    'A,ES8.1,A,ES9.2,A,ES9.2,A,ES9.2)')
     $    '  OK yn=',yn_in,' T=',temp_in,' f=',frac_in,
     $    ' G0=',G0_in,' dgr=',dgr_in,' cr=',cr_in,
     $    ' CO=',co_final,' ac=',abundc,' ao=',abundo
      endif

      return
      end
