#include "../../GIZMO_config.h"
      subroutine read_wss_collis_table
      implicit none
#ifdef WSS_CIE_COOL
#include "cool.h"

      REAL Temp, metal_only
      REAL C_only, N_only, O_only, Ne_only
      REAL Mg_only, Si_only, S_only, Ca_only, Fe_only
      REAL l1, l2, l3, l4, l5, l6, l7
      REAL n1, n2, n3, n4, n5, n6, n7
      REAL fac
      
      REAL logTemp_tbl(352), Temp_tbl(352)
c      REAL C_tbl(352), N_tbl(352), O_tbl(352)
c      REAL Ne_tbl(352), Mg_tbl(352), Si_tbl(352)
c      REAL S_tbl(352), Ca_tbl(352), Fe_tbl(352)
      
c      common /cie_cool_data/ 
c     $     C_tbl, N_tbl, O_tbl,
c     $     Ne_tbl, Mg_tbl, Si_tbl, S_tbl,
c     $     Ca_tbl, Fe_tbl

      integer I

      open(1, file='gizmo/cooling/chemcool/wss_z_collis.txt',
     $     status='old')

c     wss_z_collis.txt has 352 rows and 25 columns                                

      do I = 1, 352
         read(1, *)
     $        Temp_tbl(I), l1, n1, l2, n2, l3, n3, l4, n4,
     $        l5, n5, l6, n6, l7, n7, metal_only,
     $        C_tbl(I), N_tbl(I), O_tbl(I),
     $        Ne_tbl(I), Mg_tbl(I), Si_tbl(I), S_tbl(I),
     $        Ca_tbl(I), Fe_tbl(I)
      enddo
      close(1, status='keep')


      do I = 1, 352         
         logTemp_tbl(I) = log10( Temp_tbl(I) )
         
c     fac: to suppress the values at T<1e4 K to prevent from double-counting
         fac = exp( -(1e4 / Temp_tbl(I))**2 )
         if (fac < 1e-20) then
            fac = 1e-20
         endif

         C_tbl(I) = log10( abs( C_tbl(I) * fac ) )
         N_tbl(I) = log10( abs( N_tbl(I) * fac ) )
         O_tbl(I) = log10( abs( O_tbl(I) * fac ) )
         Ne_tbl(I) = log10( abs( Ne_tbl(I) * fac ) )
         Mg_tbl(I) = log10( abs( Mg_tbl(I) * fac ) )
         Si_tbl(I) = log10( abs( Si_tbl(I) * fac ) )
         S_tbl(I) = log10( abs( S_tbl(I) * fac ) )
         Ca_tbl(I) = log10( abs( Ca_tbl(I) * fac ) )
         Fe_tbl(I) = log10( abs( Fe_tbl(I) * fac ) )
      enddo
      
      end
c
c
c


      subroutine get_coolrates_wssCIE(temp, Zm, rate_tot)
      implicit none
#include "cool.h"

      INTEGER idx, NlT
      REAL temp, lT_range, idx_real, x
      REAL Zm(12)
      REAL rate_tot
      REAL rate_C, rate_N, rate_O
      REAL rate_Ne, rate_Mg, rate_Si
      REAL rate_S, rate_Ca, rate_Fe
      REAL rC, rN, rO
      REAL rNe, rMg, rSi
      REAL rS, rCa, rFe

c      REAL C_tbl(352), N_tbl(352), O_tbl(352)
c      REAL Ne_tbl(352), Mg_tbl(352), Si_tbl(352)
c      REAL S_tbl(352), Ca_tbl(352), Fe_tbl(352)
c      common /cie_cool_data/ 
c     $     C_tbl, N_tbl, O_tbl,
c     $     Ne_tbl, Mg_tbl, Si_tbl, S_tbl,
c     $     Ca_tbl, Fe_tbl


c     logT from 2.0 to 8.98185
      lT_range = 6.98185

c     We have 352 data points -> 351 intervals
      NlT = 351
      
c     FORTRAN index start with 1, so +1 in the end
c      print*, 'temp = ', temp

      idx_real =  NlT * (log10(temp) - 2.0) / lT_range + 1 
      idx = INT(idx_real)
      


c     The weight for linear interpolation
      x = idx_real - idx 

      if (idx < 1) then
         idx = 1
         x = 0.0
      elseif (idx > NlT-1) then
         idx = NlT-1
         x = 1.0         
      endif

c      print*, 'idx = ', idx
c      print*, 'x = ', x



c     Recall that FORTRAN index start with 1
      rC  = Zm(2)  / 2.07e-3
      rN  = Zm(8)  / 8.36e-4
      rO  = Zm(4)  / 5.49e-3
      rNe = Zm(9)  / 1.41e-3
      rMg = Zm(3)  / 5.91e-4
      rSi = Zm(6)  / 6.83e-4
      rS  = Zm(10) / 4.09e-4
      rCa = Zm(11) / 6.44e-5
      rFe = Zm(5)  / 1.10e-3

c      print*, 'ratio=', rC, rN, rO, rNe, rMg, rSi, rS, rCa, rFe

      rate_C  = 10**( (1.0 - x) *  C_tbl(idx) + x *  C_tbl(idx+1) )
      rate_N  = 10**( (1.0 - x) *  N_tbl(idx) + x *  N_tbl(idx+1) ) 
      rate_O  = 10**( (1.0 - x) *  O_tbl(idx) + x *  O_tbl(idx+1) )
      rate_Ne = 10**( (1.0 - x) * Ne_tbl(idx) + x * Ne_tbl(idx+1) )
      rate_Mg = 10**( (1.0 - x) * Mg_tbl(idx) + x * Mg_tbl(idx+1) )
      rate_Si = 10**( (1.0 - x) * Si_tbl(idx) + x * Si_tbl(idx+1) )
      rate_S  = 10**( (1.0 - x) *  S_tbl(idx) + x *  S_tbl(idx+1) )
      rate_Ca = 10**( (1.0 - x) * Ca_tbl(idx) + x * Ca_tbl(idx+1) )
      rate_Fe = 10**( (1.0 - x) * Fe_tbl(idx) + x * Fe_tbl(idx+1) )


      rate_C  = rate_C  * rC
      rate_N  = rate_N  * rN
      rate_O  = rate_O  * rO
      rate_Ne = rate_Ne * rNe
      rate_Mg = rate_Mg * rMg
      rate_Si = rate_Si * rSi
      rate_S  = rate_S  * rS
      rate_Ca = rate_Ca * rCa
      rate_Fe = rate_Fe * rFe


c      print*, 'rates = ',
c     $     rate_C, rate_N, rate_O,
c     $     rate_Ne, rate_Mg, rate_Si,
c     $     rate_S, rate_Ca, rate_Fe


      rate_tot = 
     $     rate_C  + rate_N  + rate_O +
     $     rate_Ne + rate_Mg + rate_Si + 
     $     rate_S  + rate_Ca + rate_Fe

c      print*, 'rate_tot=', rate_tot
#endif
      end



#if 0
      PROGRAM read_table
      implicit none

      REAL T, R
      REAL Zm(12)

      T = 10**5.5


      Zm(1) = 0.7065
      Zm(7) = 0.2806
      Zm(2)  = 2.07e-3
      Zm(8)  = 8.36e-4
      Zm(4)  = 5.49e-3
      Zm(9)  = 1.41e-3
      Zm(3)  = 5.91e-4
      Zm(6)  = 6.83e-4
      Zm(10) = 4.09e-4
      Zm(11) = 6.44e-5
      Zm(5)  = 1.10e-3

      call read_wss_collis_table()
      call get_coolrates_wssCIE(T, Zm, R)
      print*, R

      END PROGRAM read_table
#endif
