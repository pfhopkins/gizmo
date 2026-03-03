      program test
      implicit none
#include "cool.h"
#include "non_eq.h"
      REAL dl, yn, divv, rpar(nrpar)
      REAL y(nspec), ydot(nspec)
      REAL abh2, abhp, abco, abe, energy, ekn, temp
      REAL dt, column_est
      integer ipar(nipar), I, J

      dl = 0.0
      divv= 0.0
      deff = 1e0
      G0 = 1.7
      dust_to_gas_ratio = 1.0
      phi_pah = 1.0
      abundc = 1.41d-4
      abundo = 3.16d-4
      abundsi = 0d0
      abundD = 0d0
      iflag_ad = 1
      iflag_mn = 1
      iflag_atom = 2
      iflag_h3pra = 1
      iflag_3bh2a = 1
      iflag_3bh2b = 1
      iphoto = 1
      isrf_option = 1
      f_rsc   = 1e0
      AV_conversion_factor = 5.348d-22
      h2_form_ex  = 0e0
      h2_form_kin = 0e0
      redshift = 0e0
      tdust = 10.0

      call coolinmo
      call cheminmo
      call init_tolerances
      call load_h2_table
      call init_temperature_lookup

      yn = 1d4
      rpar(1) = yn
      rpar(2) = 3.086d18
      rpar(3) = 0d0
       
      do I = 1, nspec
        y(I) = 0
      enddo
      print*, "nspec=", nspec

      abe = y(ihp)
      abh2 = y(ih2)
      abhp = y(ihp)
      abco = y(ico)
      energy = 1.5d-10
      ekn  = energy / (BOLTZMANN * (1.0 + ABHE - abh2 + abe) * yn)
      print*, energy, ekn
      column_est = 0.0

      call calc_temp(abh2, ekn, temp)
      print*, "temp=", temp
c      y(itmp) = temp
      call calc_photo_wrapper(temp, rpar, abh2, abhp, abco)

      print*, "y=", y(ihp), y(ih2), y(ico), y(itmp)
      dt = 1e4 * 3.1557d7
c      dt = 0.0

      print*, "start evolve_abundances..."
      call evolve_abundances(dt, dl, yn, divv, energy,
     $                         y,
     $                         column_est)
      print*, "evolve_abundances done"
      print*, "y=", y(ihp), y(ih2), y(ico), y(itmp)

      ekn  = energy / (BOLTZMANN * (1.0 + ABHE - abh2 + abe) * yn)
      call calc_temp(abh2, ekn, temp)
      print*, energy, temp
c      do I = 1, 20
c        print*, I
c        y(ih2) = 0
c        y(ihp) = 0
c        y(ico) = 0
c        y(itmp) = 2.1e-10 * I
c        call rate_eq(nspec, 0.0, y, ydot, rpar, ipar)
c        do J = 1, nspec
c          print*, ydot(j)
c        enddo
c        print*, ' '
c      enddo

      end
