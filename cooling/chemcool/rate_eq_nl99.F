#include "gadgetconfig.h"

#if CHEMISTRYNETWORK == 15
#ifndef KOYAMA
c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////             R A T E _ E Q                 \\\\\\\\\\
c
c=======================================================================
c
      subroutine rate_eq(neq, time, y, ydot, rpar, ipar)
#ifdef CHEMCOOL
      implicit NONE
#include "cool.h"
#include "non_eq.h"
#include "current_photo.h"
      integer neq
c
      REAL time, y(nspec), ydot(nspec), rpar(nrpar)
      integer ipar(nipar)
c
      integer itemp, i
      REAL abundances(nabn), rates(nrates), 
     $     rates_chem(nrates_chem)
c
      REAL abe     , abhp    , abHI    , abh2    ,
     $     abcp    , abco    , abheI   , abh3p   ,
     $     abcI    , aboI    , abchx   , abohx   ,
     $     abmp    , abMI    , abhep   , abhcop
c
      REAL ylam    , temp    , NH      ,
     $     yn      , dl      , divv    , yntot   , gamma , 
     $     energy  , dtemp   , ncrinv  , h2var0  , h2var1, 
     $     h2var2  , ykdh    , ykdh2   , ykdis   ,
     $     ylam_chem, ekn    , f_ex    , f_tot
c
      REAL ph2, ph10, ph47, ph48, ph49, ph50, ph51
c
      REAL fh3p
c
      REAL num, denom
c
      REAL k0, k1, gamma_chx, gamma_co, beta, abo
      REAL f_kc1, f_ck2, Rdust, RH2
c
      REAL fshield
c
      REAL phi     , h_gr    , hgrvar1 , hgrvar2, dii_gr
c
      REAL cr1     , cr2     , cr6
c
      REAL ch1     , ch2     , ch3     , ch4     , ch5  ,
     $     ch6     , ch9     , ch11    , ch13 , 
     $     ch17    , ch25    , ch34    , ch35    , ch40 ,
     $     ch56    , ch92    , ch105   , ch106   , ch146,
     $     ch147
c
      REAL dtch1   , dtch2   , dtch3   , dtch4   , dtch5  ,
     $     dtch6   , dtch9   , dtch11  , dtch13 , 
     $     dtch17  , dtch25  , dtch34  , dtch35  , dtch40 ,
     $     dtch56  , dtch92  , dtch105 , dtch106 , dtch146,
     $     dtch147
c
      REAL ch106a
c
      REAL cdex, rdex, fpheat
c
c Unpack parameters
c
      yn   = rpar(1)
      dl   = rpar(2)
      divv = rpar(3)
c
      NH   = 0.5d0 * yn * dl
c
c Chemical species:
c
      call validate_input(y, ipar)
c
c Tracked, non-equilibrium species
c
      abh2   = max(0d0, y(ih2))
      abhp   = max(0d0, y(ihp))
      abcp   = max(0d0, y(ic))
      abchx  = max(0d0, y(ichx))
      abohx  = max(0d0, y(iohx))
      abco   = max(0d0, y(ico))
      abhcop = max(0d0, y(ihcop))
      abhep  = max(0d0, y(ihep))
      abmp   = max(0d0, y(imp))
c
c Non-equilibrium species computed using conservation laws
c
      abe   = max(abhp + abcp + abhcop + abhep + abmp, 0d0)
      abHI  = max(1d0 - 2d0 * abh2 - abhp - abchx - abohx, 0d0)
      abheI = max(abhe - abhep, 0d0)
      abcI  = max(abundc - abcp - abchx - abohx - abco - abhcop, 0d0)
      aboI  = max(abundo - abohx - abco, 0d0)
      abmI  = max(abundM - abmp, 0d0)
c
c Internal energy (erg cm^-3)
c
      energy = y(itmp)
c
c Compute temperature
c
c [NB We ignore corrections to yntot & gamma arising from low abundance 
c     molecular species (e.g. H-, H2+) and metals]
c
      yntot = (1d0 + abhe - abh2 + abe) * yn
      ekn   = energy / (yntot * kboltz)
      call lookup_temp_gamma(abh2, ekn, temp, gamma)
c
c Bounds checking
c
c Temp:
c
      if (temp .le. 0d0) then
        itemp   = 1
        dtemp   = 0d0
        temp    = tmin
        ipar(1) = 1
      elseif (temp .le. tmin) then
        itemp = 1
        dtemp = 0d0
      elseif (temp .ge. tmax) then
        itemp = nmd
        dtemp = 0d0
      else
        itemp = int(dlog10(temp) / dtlog) + 1
        if (itemp .le. 0 .or. itemp .gt. nmd) then
          print*, 'Fatal error in rate_eq.F', itemp, temp, energy
          ABORT(1)
        endif
        dtemp = temp - temptab(itemp)
      endif
c
c Read in coefficients
c
      dtch1  = dtchtab(1, itemp)
      dtch2  = dtchtab(2, itemp)
      dtch3  = dtchtab(3, itemp)
      dtch4  = dtchtab(4, itemp)
      dtch5  = dtchtab(5, itemp)
      dtch6  = dtchtab(6, itemp)
      dtch9  = dtchtab(9, itemp)
      dtch11 = dtchtab(11, itemp)
      dtch13 = dtchtab(13, itemp)
      dtch17 = dtchtab(17, itemp)
      dtch25 = dtchtab(25, itemp)
      dtch34 = dtchtab(34, itemp)
      dtch35 = dtchtab(35, itemp)
      dtch40 = dtchtab(40, itemp)
      dtch56 = dtchtab(56, itemp)
      dtch92 = dtchtab(92, itemp)
      dtch105 = dtchtab(105, itemp)
      dtch106 = dtchtab(106, itemp)
      dtch146 = dtchtab(146, itemp)
      dtch147 = dtchtab(147, itemp)
c      
      ch1  = chtab(1, itemp) + dtemp * dtch1
      ch2  = chtab(2, itemp) + dtemp * dtch2
      ch3  = chtab(3, itemp) + dtemp * dtch3
      ch4  = chtab(4, itemp) + dtemp * dtch4
      ch5  = chtab(5, itemp) + dtemp * dtch5
      ch6  = chtab(6, itemp) + dtemp * dtch6
      ch9  = chtab(9, itemp) + dtemp * dtch9
      ch11 = chtab(11, itemp) + dtemp * dtch11
      ch13 = chtab(13, itemp) + dtemp * dtch13
      ch17 = chtab(17, itemp) + dtemp * dtch17
      ch25 = chtab(25, itemp) + dtemp * dtch25
      ch34 = chtab(34, itemp) + dtemp * dtch34
      ch35 = chtab(35, itemp) + dtemp * dtch35
      ch40 = chtab(40, itemp) + dtemp * dtch40
      ch56 = chtab(56, itemp) + dtemp * dtch56
      ch92 = chtab(92, itemp) + dtemp * dtch92
      ch105 = chtab(105, itemp) + dtemp * dtch105
      ch106 = chtab(106, itemp) + dtemp * dtch106
      ch146 = chtab(146, itemp) + dtemp * dtch146
      ch147 = chtab(147, itemp) + dtemp * dtch147
c
c H3+ dissociative recombination: branching fractions from UMIST06,
c based on MAC04
c
      ch106a = (2.34d0 / 6.7d0) * ch106   ! H2 + H
c
c H2 collisional dissociation rates -- need special handling because of
c density dependence
c
      ncrinv   = (2d0 * abh2 * (ch6 - ch5) + ch5)
      h2var0   = 1d0 / ( 1d0 + yn * ncrinv)
      h2var1   = ch3**h2var0
      h2var2   = ch4**h2var0
      ykdh     = ch1 * h2var1
      ykdh2    = ch2 * h2var2
c
c Photodissociation:
c
      call calc_H2_shielding_factor(fshield, temp, abh2, NH, rpar)
      ykdis = phrates(1) * fshield
      ph2   = phrates(2)
      ph10  = phrates(10)
      ph48  = phrates(48)
      ph49  = phrates(49)
      ph50  = phrates(50)
      ph51  = phrates(51)
c
c Rates for recombination on grain surfaces. These rates must be 
c computed here because of their dependence on the electron number
c density. 
c
      if (abe .eq. 0d0) then
c If the fractional ionization is zero, then there won't be any recombination,
c so the value we use for phi doesn't matter too much -- 1d20 is simply an 
c arbitrary large number
c 
        phi = 1d20
      else
        phi = G_dust * sqrt(temp) / (yn * abe)
      endif
c
c H+:
c
      if (phi .lt. 1d0) then
        h_gr = 1.225d-13 * dust_to_gas_ratio
      else
        hgrvar1  = 8.074d-6 * phi**1.378d0
        hgrvar2  = (1d0 + ch34 * phi**ch35)
        h_gr     = 1.225d-13 * dust_to_gas_ratio /
     $             (1d0 + hgrvar1 * hgrvar2)
      endif
c
c H2 formation rate on dust -- we calculate this here, as this allows us to
c vary tdust without needing to call cheminmo again
c                  
      if (yn .gt. 1d9 / dust_to_gas_ratio**1.5d0) then
        tdust = temp
        call calc_dust_temp(yn, chi_isrf, temp, abHI, RH2, Rdust, 1)
      else
        call calc_dust_temp(yn, chi_isrf, temp, abHI, RH2, Rdust, 0)
      endif
c
c Cosmic ray ionization rates: precomputed in cheminmo.
c
c HI:
      cr1 = crtab(1)
      cr2 = crtab(2)
c HeI:
      cr6 = crtab(6)
c
c H3+ equilibrium abundance
c
      fh3p = ch105 * abh2 / (ch105 * abh2 + cst(1) * abHI)
c
      num   = fh3p * ((cr2 / yn) * abh2 + cst(75) * abhep * abh2)
      denom = cst(84) * abcI + cst(86) * aboI + cst(59) * abco 
     $      + ch106 * abe + cst(81) * abmI
c
      abh3p = num / denom
      abe   = abe + abh3p
c
c Protons
c 
      ydot(ihp) = (ch11 * abe   * abHI          ! Collisional ion: H  + e-
     $          -  ch13 * abhp  * abe           ! Gas-phase H recomb. (case B)
     $          + (1d0 - fh3p)  * cst(75) * abhep * abh2
     $          +  ch9  * abhep * abh2) * yn    ! DCT: He+ + H2 
     $          +  cr1  * abHI                  ! Cosmic ray ion.:  H + CR
     $          + (1d0 - fh3p) * cr2 * abh2     ! H2+ + H -> H2 + H+
     $          +  ph50 * abhcop                ! Photodissoc. of HCO+
c
c Molecular hydrogen
c
      ydot(ih2) = (-ch25  * abe  * abh2          ! Collisional dissoc.: H2 + e- 
     $          -   ykdh  * abHI * abh2          ! Collisional dissoc.: H2 + H 
     $          -   ykdh2 * abh2 * abh2          ! Collisional dissoc.: H2 + H2
     $          +   Rdust * abHI * 1d0 
     $          - cst(82) * abh2 * abcp
     $          - cst(54) * abcI * abh2
     $          + cst(84) * abh3p * abcI
     $          + cst(86) * abh3p * aboI
     $          + cst(59) * abh3p * abco
     $          -     ch9 * abhep * abh2
     $          - cst(75) * abhep * abh2 * (1d0 - fh3p)
     $          +  ch106a * abh3p * abe
     $          + cst(81) * abh3p * abmI) * yn
     $          - 2d0 * fh3p * cr2 * abh2        ! Cosmic rays
     $          -   ykdis * abh2                 ! Photodissoc:         H2 + PH
c
c C+
c
      ydot(ic)  = ( ch92 * abhep * abco
     $          -   ch40 * abcp * abe
     $          -cst(82) * abcp * abh2
     $          -cst(83) * abcp * abohx) * yn
     $          +    ph2 * abcI
c
c CHx
c
      ydot(ichx) = (cst(84) * abcI * abh3p
     $           +  cst(82) * abcp * abh2
     $           +  cst(54) * abcI * abh2
     $           -  cst(85) * aboI * abchx) * yn
     $           -     ph48 * abchx
c
c OHx
c
      ydot(iohx) = (cst(86) * aboI * abh3p
     $           -    ch147 * abcI * abohx
     $           -  cst(83) * abcp * abohx) * yn
     $           -     ph49 * abohx
c
c HCO+
c
      ydot(ihcop) = (cst(59) * abh3p  * abco
     $            +  cst(83) * abcp   * abohx
     $            -     ch56 * abhcop * abe) * yn
     $            -     ph50 * abhcop
c
c CO
c
      ydot(ico) = (cst(85) * abchx  * aboI
     $          +    ch147 * abohx  * abcI
     $          +     ch56 * abhcop * abe
     $          -  cst(59) * abh3p  * abco
     $          -     ch92 * abhep  * abco) * yn
     $          -     ph10 * abco
     $          +     ph50 * abhcop      ! produces H+       
c
c M+
c
      ydot(imp) = (cst(81) * abh3p * abmI
     $          -    ch146 * abmp  * abe) * yn
     $          +     ph51 * abmI
c
c He+
c
      ydot(ihep) =  cr6  * abheI
     $           - ( ch9 * abhep * abh2
     $           +  ch92 * abhep * abco
     $           +  ch17 * abhep * abe
     $         + cst(75) * abhep * abh2) * yn
c
c H2 collisional dissociation cooling:
c
      rates_chem(1) = 4.48d0 * eV * (ykdh  * abHI * abh2 + 
     $                               ykdh2 * abh2 * abh2 +
     $                               ch25  * abe  * abh2 ) * yn**2
c
c H2 photodissociation heating -- 0.4eV per photodissociation:
c
      rates_chem(2) = - 4d-1 * eV * abH2 * yn * ykdis
c
c Heating by UV pumping of H2 -- effective for n > n_crit
c
c We assume 2 eV per UV photon (Burton, Hollenbach & Tielens 1990),
c and adopt a pumping rate 8.5 times larger than the dissociation rate
c (Draine & Bertoldi 1996, table 2). 
c
      cdex = (1.4 * dexp(-1.81d4 / (temp + 1200)) * abH2
     $     +  1.0 * dexp(-1d3 / temp) * abHI) * 1d-12 * dsqrt(temp) * yn
      rdex = 2d-7
      fpheat = cdex / (cdex + rdex)

      rates_chem(3) = - 2d0 * eV * abH2 * yn * ykdis * 8.5d0 * fpheat
c
c H2 formation heating -- we assume a similar depedence on n / n_crit to 
c the collisional dissociation rates, although ideally one would like to
c do a more careful, level-dependent treatment.
c
c Note that energy input depends on formation mechanism:
c
c Dust -- 4.48eV 
c
      if (RH2 .gt. 0d0) then
        call compute_h2_heating_fraction(temp, yn, abh2, f_ex)
        f_ex = f_ex * h2_form_ex
      else
        f_ex = 0d0
      endif
      f_tot = h2_form_kin + f_ex
c
      rates_chem(4) = - 4.48d0 * eV * f_tot * RH2
c
c HI collisional ionization cooling -- 13.6eV per ionization:
c
      rates_chem(5) = 13.6d0 * eV * ch11 * abe * abHI * yn**2
c
c HII recombination cooling -- we assume that kT ergs is radiated 
c per recombination and that this is independent of whether the
c recombination occurs in the gas phase or on a grain surface
c
      rates_chem(6) = kboltz * temp * abhp * (ch13 * abe + h_gr) * yn**2
c
      ylam_chem = rates_chem(1)  + rates_chem(2)  + rates_chem(3)  +
     $            rates_chem(4)  + rates_chem(5)  + rates_chem(6)
c
c Initialize all abundances to zero, then just set the non-zero ones
c
      do i = 1, nabn
        abundances(i) = 0d0
      enddo
c
      abundances(1)  = abh2
      abundances(2)  = aboI
      abundances(5)  = abco
      abundances(6)  = abcI
      abundances(7)  = abcp
      abundances(9)  = abundsi
      abundances(10) = abe
      abundances(11) = abhp
      abundances(12) = abHI
c
      call cool_func(temp, yn, dl, divv, G_dust, abundances, ylam,
     $               rates, 0)
c
      ydot(itmp) = -(ylam + ylam_chem)

#ifdef THERMAL_INFO_DUMP
c
cc We'll leave the accretion heating holder here for the future.
c
      heat_accr   = 0.
      heat_cray    = rates(11)
      heat_photo_elec = rates(12)
      heat_h2_form = rates_chem(4)
      heat_h2_pdiss = rates_chem(2)
      cool_h2_line = rates(2)
      cool_cplus   = rates(16)
      cool_oxy     = rates(13)
      cool_co      = rates(7) + rates(21) + rates(22) +
     $               rates(8) + rates(9) + rates(24)
      cool_gas_dust = rates(1)
      cool_c        = rates(14)
#endif

c
c Now add the pdV term:  Actually, let's not! Gadget evolves entropy... :)
c 
c      ydot(itmp) = ydot(itmp) - (gamma - 1d0) * energy * divv
c
#endif /* CHEMCOOL */
      return      
      end
c=======================================================================
c
c    \\\\\\\\\\        E N D   S U B R O U T I N E        //////////
c    //////////              R A T E _ E Q                \\\\\\\\\\
c
c=======================================================================
#endif /* not KOYAMA */
#endif /* CHEMISTRYNETWORK == 15 */
