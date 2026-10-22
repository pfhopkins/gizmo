#include "../../GIZMO_config.h"

#if CHEMISTRYNETWORK == 16
c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////             R A T E _ E Q                 \\\\\\\\\\
c
c=======================================================================
c
c Gong, Ostriker & Wolfire (2017) network, ported from AREPO SGChem
c rate_eq_gong17.F (13CO variant dropped). Photorates, G_dust and
c chi_isrf come from calc_photo; M is silicon.
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
      REAL CMB_temp, ylam_floor, rates_floor(nrates)
c
      REAL abe     , abhp    , abHI    , abh2    ,
     $     abcp    , abco    , abheI   , abh3p   ,
     $     abcI    , aboI    , abchx   , abohx   ,
     $     abmp    , abMI    , abhep   , abhcop  ,
     $     abop    , abotot
c
      REAL ylam    , temp    , NH      ,
     $     yn      , dl      , divv    , yntot   , gamma ,
     $     energy  , dtemp   , ncrinv  , h2var0  , h2var1,
     $     h2var2  , ykdh    , ykdh2   , ykdis   ,
     $     ylam_chem, ekn    , f_ex    , f_tot   , ynsq
c
      REAL ph2, ph3, ph10, ph48, ph49
c
      REAL fh3p
c
      REAL num, denom
c
      REAL Rdust, RH2
c
      REAL fshield
c
      REAL phi     , h_gr    , hgrvar1 ,    hgrvar2, cii_gr,
     $     c2grvar1, c2grvar2, si2_gr  , si2grvar1, si2grvar2,
     $     heii_gr, hegrvar1, hegrvar2
c
      REAL cr1     , cr2     , cr3     , cr6     , cr9  ,
     $     cr10    , cr11    , cp1     , cp11    , cp13
c
      REAL ch1     , ch2     , ch3     , ch4     , ch5  ,
     $     ch6     , ch9     , ch11    , ch13    , ch14 ,
     $     ch17    , ch25    , ch34    , ch35    , ch36 ,
     $     ch37    , ch38    , ch39    , ch40    , ch41 ,
     $     ch42    , ch43    , ch44    , ch46    , ch47 ,
     $     ch53    , ch87    , ch89    , ch92    , ch105,
     $     ch106   , ch135   , ch136   , ch147   ,
     $     ch174   , ch175   , ch176   , ch177   ,
     $     ch178   , ch179   , ch180   , ch181
c
      REAL dtch1   , dtch2   , dtch3   , dtch4   , dtch5  ,
     $     dtch6   , dtch9   , dtch11  , dtch13  , dtch14 ,
     $     dtch17  , dtch25  , dtch34  , dtch35  , dtch36 ,
     $     dtch37  , dtch38  , dtch39  , dtch40  , dtch41 ,
     $     dtch42  , dtch43  , dtch44  , dtch46  , dtch47 ,
     $     dtch53  , dtch87  , dtch89  , dtch92  , dtch105,
     $     dtch106 , dtch135 , dtch136 , dtch147 ,
     $     dtch174 , dtch175 , dtch176 , dtch177 ,
     $     dtch178 , dtch179 , dtch180 , dtch181
c
      REAL ch87a, ch87b, ch106a, ch174a, ch176a, cst93a
c
      REAL h2op_branch
c
      REAL cdex, rdex, fpheat
c
      REAL rho_lj, cs2_lj, L_jeans, energy_dummy
c
c Unpack parameters
c
      yn   = rpar(1)
      dl   = rpar(2)
      divv = rpar(3)
c
      NH   = 0.5d0 * yn * dl
      ynsq = yn * yn
c
c Chemical species:
c
      call validate_input(y, ipar)
      if (ipar(1) .eq. 1) then
c Return with zeroed ydot array if input abundances invalid
        do I = 1, nspec
          ydot(I) = 0d0
        enddo
        return
      endif
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
      abHI  = max(1d0 - 2d0 * abh2 - abhp - abchx - abohx - abhcop,
     $            0d0)
      abheI = max(abhe - abhep, 0d0)
      abcI  = max(abundc - abcp - abchx - abco - abhcop, 0d0)
      abmI  = max(abundM - abmp, 0d0)
c
c O + O+ total; the O+ fraction is taken to be in equilibrium (below)
c
      abotot = max(abundo - abohx - abco - abhcop, 0d0)
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
      dtch14 = dtchtab(14, itemp)
      dtch17 = dtchtab(17, itemp)
      dtch25 = dtchtab(25, itemp)
      dtch34 = dtchtab(34, itemp)
      dtch35 = dtchtab(35, itemp)
      dtch36 = dtchtab(36, itemp)
      dtch37 = dtchtab(37, itemp)
      dtch38 = dtchtab(38, itemp)
      dtch39 = dtchtab(39, itemp)
      dtch40 = dtchtab(40, itemp)
      dtch41 = dtchtab(41, itemp)
      dtch42 = dtchtab(42, itemp)
      dtch43 = dtchtab(43, itemp)
      dtch44 = dtchtab(44, itemp)
      dtch46 = dtchtab(46, itemp)
      dtch47 = dtchtab(47, itemp)
      dtch53 = dtchtab(53, itemp)
      dtch87 = dtchtab(87, itemp)
      dtch89 = dtchtab(89, itemp)
      dtch92 = dtchtab(92, itemp)
      dtch105 = dtchtab(105, itemp)
      dtch106 = dtchtab(106, itemp)
      dtch135 = dtchtab(135, itemp)
      dtch136 = dtchtab(136, itemp)
      dtch147 = dtchtab(147, itemp)
      dtch174 = dtchtab(174, itemp)
      dtch175 = dtchtab(175, itemp)
      dtch176 = dtchtab(176, itemp)
      dtch177 = dtchtab(177, itemp)
      dtch178 = dtchtab(178, itemp)
      dtch179 = dtchtab(179, itemp)
      dtch180 = dtchtab(180, itemp)
      dtch181 = dtchtab(181, itemp)
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
      ch14 = chtab(14, itemp) + dtemp * dtch14
      ch17 = chtab(17, itemp) + dtemp * dtch17
      ch25 = chtab(25, itemp) + dtemp * dtch25
      ch34 = chtab(34, itemp) + dtemp * dtch34
      ch35 = chtab(35, itemp) + dtemp * dtch35
      ch36 = chtab(36, itemp) + dtemp * dtch36
      ch37 = chtab(37, itemp) + dtemp * dtch37
      ch38 = chtab(38, itemp) + dtemp * dtch38
      ch39 = chtab(39, itemp) + dtemp * dtch39
      ch40 = chtab(40, itemp) + dtemp * dtch40
      ch41 = chtab(41, itemp) + dtemp * dtch41
      ch42 = chtab(42, itemp) + dtemp * dtch42
      ch43 = chtab(43, itemp) + dtemp * dtch43
      ch44 = chtab(44, itemp) + dtemp * dtch44
      ch46 = chtab(46, itemp) + dtemp * dtch46
      ch47 = chtab(47, itemp) + dtemp * dtch47
      ch53 = chtab(53, itemp) + dtemp * dtch53
      ch87 = chtab(87, itemp) + dtemp * dtch87
      ch89 = chtab(89, itemp) + dtemp * dtch89
      ch92 = chtab(92, itemp) + dtemp * dtch92
      ch105 = chtab(105, itemp) + dtemp * dtch105
      ch106 = chtab(106, itemp) + dtemp * dtch106
      ch135 = chtab(135, itemp) + dtemp * dtch135
      ch136 = chtab(136, itemp) + dtemp * dtch136
      ch147 = chtab(147, itemp) + dtemp * dtch147
      ch174 = chtab(174, itemp) + dtemp * dtch174
      ch175 = chtab(175, itemp) + dtemp * dtch175
      ch176 = chtab(176, itemp) + dtemp * dtch176
      ch177 = chtab(177, itemp) + dtemp * dtch177
      ch178 = chtab(178, itemp) + dtemp * dtch178
      ch179 = chtab(179, itemp) + dtemp * dtch179
      ch180 = chtab(180, itemp) + dtemp * dtch180
      ch181 = chtab(181, itemp) + dtemp * dtch181
c
c H3+ dissociative recombination: branching fractions from UMIST06,
c based on MAC04
c
      ch106a = (2.34d0 / 6.7d0) * ch106   ! H2 + H
c
c HCO+ dissociative recombination: rate and branching ratios from
c Geppert et al (2005)
c
      ch87a = 0.92d0 * ch87       ! CO + H
      ch87b = 0.08d0 * ch87       ! OH + C
c
c O + H3+ -> products; rate and branching ratio from GONG17, rate based
c on de Ruette et al 2015. Tiny term in denominator is to avoid division
c by zero in the case where abh2 and abe are both zero
c
      h2op_branch = cst(92) * abh2 / (cst(92) * abh2 + ch175 * abe
     $            + 1d-30)

      ch174a = h2op_branch * ch174    ! OHx + H2
c
c O+ + H2 -> OHx: rate from GONG17, based on Wakelam et al (2010)
c
      cst93a = h2op_branch * cst(93)
c
c C+ + H2 -> products: rate and branching ratio from GONG17, based on
c Wakelam et al 2010
c
      ch176a = (2.31d0 / 3.3d0) * ch176
c
c H2 collisional dissociation rates -- need special handling because of
c density dependence
c
      if (temp .lt. 5d2) then
c At this temperature, H2 collisional dissociation should be negligible
        ykdh  = 0d0
        ykdh2 = 0d0
      else
        ncrinv   = (2d0 * abh2 * (ch6 - ch5) + ch5)
        h2var0   = 1d0 / ( 1d0 + yn * ncrinv)
        h2var1   = ch3**h2var0
        h2var2   = ch4**h2var0
        ykdh     = ch1 * h2var1
        ykdh2    = ch2 * h2var2
      endif
c
c Photodissociation and photoionization (dust factors from calc_photo)
c
      call calc_H2_shielding_factor(fshield, temp, abh2, NH, rpar)
      ykdis = phrates(1) * fshield
      ph2   = phrates(2)
      ph3   = phrates(3)    ! Si photoion.
      ph10  = phrates(10)
      ph48  = phrates(48)
      ph49  = phrates(49)
c
c Rates for recombination on grain surfaces, from Weingartner & Draine (2001).
c These rates must be computed here because of their dependence on the
c electron number density.
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
c Grain recombination fits are only valid for phi > 100; the rates are
c insensitive to phi below this, so phi is floored at 100
      phi = max(phi, 1d2)
c
c H+:
c
      hgrvar1  = 8.074d-6 * phi**1.378d0
      hgrvar2  = (1d0 + ch34 * phi**ch35)
      h_gr     = 1.225d-13 * dust_to_gas_ratio /
     $           (1d0 + hgrvar1 * hgrvar2)
c
c He+
c
      hegrvar1  = 3.185d-7 * phi**1.512d0
      hegrvar2  = (1d0 + ch135 * phi**ch136)
      heii_gr   = 5.572d-14 * dust_to_gas_ratio /
     $            (1d0 + hegrvar1 * hegrvar2)
c
c C+:
c
      c2grvar1  = 6.089d-3 * phi**1.128d0
      c2grvar2  = (1d0 + ch36 * phi**ch37)
      cii_gr    = 4.558d-13 * dust_to_gas_ratio /
     $            (1d0 + c2grvar1 * c2grvar2)
c
c Si+:
c
      si2grvar1 = 5.678d-8 * phi**1.874d0
      si2grvar2 = (1d0 + ch38 * phi**ch39)
      si2_gr    = 2.166d-14 * dust_to_gas_ratio /
     $            (1d0 + si2grvar1 * si2grvar2)
c
c Exponentially reduce rates at high temperature to account for fact that we
c expect hot gas to be largely dust free. This also prevents us from getting
c an anomalously large grain recombination cooling rate when T is large
c and G_dust is very small or zero
c
      h_gr    = h_gr    * dexp(-temp / 3d4)
      heii_gr = heii_gr * dexp(-temp / 3d4)
      cii_gr  = cii_gr  * dexp(-temp / 3d4)
      si2_gr  = si2_gr  * dexp(-temp / 3d4)
c
c Gong et al (2017) adopt grain recombination rates that are a factor of
c 0.6 times the Weingartner & Draine (2001) values, to match the values
c used in Wolfire et al (2008)
c
      h_gr    = 0.6d0 * h_gr
      heii_gr = 0.6d0 * heii_gr
      cii_gr  = 0.6d0 * cii_gr
      si2_gr  = 0.6d0 * si2_gr
c
c H2 formation rate on dust -- we calculate this here, as this allows us to
c vary tdust without needing to call cheminmo again
c
c     (no dust, e.g. hot gas: skip the coupling test, which divides by it)
      if (dust_to_gas_ratio .gt. 0d0) then
        if (yn .gt. 1d8 / dust_to_gas_ratio**1.5d0) then
          tdust = temp
          call calc_dust_temp(yn, chi_isrf, temp, abHI, RH2, Rdust, 1)
        else
          call calc_dust_temp(yn, chi_isrf, temp, abHI, RH2, Rdust, 0)
        endif
      else
        call calc_dust_temp(yn, chi_isrf, temp, abHI, RH2, Rdust, 0)
      endif
c
c Cosmic ray ionization rates: precomputed in cheminmo per unit rate.
c
c HI, H2 (with secondary-ionization weighting as in GONG17):
      cr1 = crtab(1) * cosmic_ray_ion_rate * (2.3d0 * abh2 + 1.5d0*abHI)
      cr2 = crtab(2) * cosmic_ray_ion_rate * (2.3d0 * abh2 + 1.5d0*abHI)
      cr10 = crtab(10) * cosmic_ray_ion_rate ! H2 -> H + H+ + e
      cr11 = crtab(11) * cosmic_ray_ion_rate ! H2 -> H + H
c HeI:
      cr6 = crtab(6) * cosmic_ray_ion_rate
c
      cr3 = crtab(3) * cosmic_ray_ion_rate   ! direct CR ionization of C
c CO + CR -> CO+ + e; CO+ + H -> HCO+
      cr9 = crtab(9) * cosmic_ray_ion_rate
c
c CR-induced photoionization of C and Si, photodissociation of CO
      cp1  = 2d0 * abh2 * crphot(1)  * cosmic_ray_ion_rate
      cp11 = 2d0 * abh2 * crphot(11) * cosmic_ray_ion_rate
      cp13 = 2d0 * abh2 * crphot(13) * cosmic_ray_ion_rate
c
c O+ equilibrium abundance
c
c Formation rate due to charge transfer with H+ depends on O abundance,
c which is not yet known. However, if we add ch47 * abhp * abop to both
c sides of balance equation, can remove this dependence, as l.h.s. now
c depends only on abotot, which is known. The other advantage this has
c is that it allows us to avoid a division-by-zero error if abHI and abh2
c are both zero. Instead, we simply find that abop=abotot and aboI=0, which
c is correct for fully ionized gas
c
      num   = ch47 * abotot * abhp + ch179 * abhep * abohx
      denom = cst(93) * abH2 + ch46 * abHI + ch47 * abhp
c
c     (no H, H2 or H+: O+ cannot form or be destroyed here; keep it neutral)
      if (denom .gt. 0d0) then
        abop = min(num / denom, abotot)
      else
        abop = 0d0
      endif
      aboI = abotot - abop
      abe  = abe + abop
c
c H3+ equilibrium abundance
c
c If the H2 abundance is very small, we can assume that the H3+ abundance
c is zero. (This isn't quite true in reality, but it will be negligibly small)
c
      if (abh2 .lt. 1d-10) then
        fh3p = 0d0
      else
c If the temperature is very large, the H3+ formation rate coefficient can
c be zero (to machine precision). If there's also no HI (i.e. we're in a
c hot, fully ionized bubble), then we need to set fh3p to zero, as
c otherwise we'll end up dividing 0/0 which will cause a crash. For safety,
c we set fh3p to zero whenever ch105 is very small
        if (ch105 .lt. 1d-40) then
          fh3p = 0d0
        else
          fh3p = ch105 * abh2 / (ch105 * abh2 + cst(1) * abHI)
        endif
      endif
c
      num   = fh3p * ((cr2 / yn) * abh2 + cst(75) * abhep * abh2)
      denom = ch180 * abcI + ch174 * aboI + cst(59) * abco
     $      + ch106 * abe
c
c H3+ destruction rate depends on electron number density, which depends
c (in part) on H3+ abundance. We therefore wind up with a quadratic
c equation to solve for the H3+ equilibrium abundance.
c
      if (num .eq. 0d0) then
        abh3p = 0d0
      else
        abh3p = (-denom + dsqrt(denom**2 + 4d0 * ch106 * num))
     $        / (2d0 * ch106)
      endif
c H3+ holds three H atoms of the HI pool (not capped: keeps H3+ chemistry in fully molecular gas)
      abHI = max(abHI - 3d0 * abh3p, 0d0)
      abe = abe + abh3p
c
c Protons
c
      ydot(ihp) = (ch11 * abe   * abHI          ! Collisional ion: H  + e-
     $          -  ch13 * abhp  * abe           ! Gas-phase H recomb.(case B)
     $          -  h_gr * abhp  * 1d0           ! Grain H recomb.
     $          + (1d0 - fh3p)  * cst(75) * abhep * abh2
     $          +  ch46 * abop  * abhI
     $          -  ch47 * aboI  * abhp
     $          +  ch9  * abhep * abh2) * yn    ! DCT: He+ + H2
     $          +  cr1  * abHI                  ! Cosmic ray ion.:  H + CR
     $          +  cr10 * abh2                  ! H2 + CR -> H + H+ + e
     $          + (1d0 - fh3p) * cr2 * abh2     ! H2+ + H -> H2 + H+
c
c Molecular hydrogen
c
      ydot(ih2) = (-ch25  * abe  * abh2          ! Collisional dissoc.: H2 + e-
     $          -   ykdh  * abHI * abh2          ! Collisional dissoc.: H2 + H
     $          -   ykdh2 * abh2 * abh2          ! Collisional dissoc.: H2 + H2
     $          +   Rdust * abHI * 1d0
     $          -   ch176 * abh2 * abcp
     $          - cst(54) * abcI * abh2
     $          +   ch180 * abh3p * abcI
     $          + ch174   * abh3p * aboI
     $          + cst(59) * abh3p * abco
     $          -     ch9 * abhep * abh2
     $          - 2d0 * fh3p * cst(75) * abhep * abh2  ! via H2+ + H2
     $          - cst(93) * abop  * abh2
     $          +  ch106a * abh3p * abe
     $          +   ch53  * abhI  * abohx
     $          +   ch178 * abHI  * abchx) * yn
     $          - 2d0 * fh3p * cr2 * abh2        ! Cosmic rays
     $          -   ykdis * abh2                 ! Photodissoc:         H2 + PH
     $          -    cr10 * abh2
     $          -    cr11 * abh2
c
c C+
c
      ydot(ic)  = ( ch92 * abhep  * abco
     $          -   ch40 * abcp   * abe
     $          +   ch43 * abcI   * abe
     $          - cii_gr * abcp   * 1d0
     $          -  ch176 * abcp   * abh2
     $          -  ch177 * abcp   * abohx
     $          +  ch181 * abchx  * abhep) * yn
     $          +    ph2 * abcI
     $          +    cr3 * abcI
     $          +    cp1 * abcI
c
c CHx
c
      ydot(ichx) = (  ch180 * abcI   * abh3p
     $           +   ch176a * abcp   * abh2
     $           +  cst(54) * abcI   * abh2
     $           -  cst(85) * aboI   * abchx
     $           -    ch181 * abchx  * abhep
     $           -    ch178 * abHI   * abchx) * yn
     $           -     ph48 * abchx
c
c OHx
c
      ydot(iohx) = ( ch174a * aboI * abh3p
     $           +    ch87b * abhcop * abe
     $           +   cst93a * abop * abh2
     $           +    ch89  * abco * abhI
     $           -    ch147 * abcI * abohx
     $           -    ch177 * abcp   * abohx
     $           -    ch53  * abhI   * abohx
     $           -  cst(94) * aboI   * abohx
     $           -    ch179 * abhep  * abohx) * yn
     $           -     ph49 * abohx
c
c HCO+
c
      ydot(ihcop) = (cst(59) * abh3p  * abco
     $            +    ch177 * abcp   * abohx
     $            -     ch87 * abhcop * abe) * yn
     $            +      cr9 * abco
c
c CO
c
      ydot(ico) = (cst(85) * abchx  * aboI
     $          +    ch147 * abohx  * abcI
     $          +    ch87a * abhcop * abe
     $          -  cst(59) * abh3p  * abco
     $          -     ch92 * abhep  * abco
     $          -     ch89 * abhI   * abco) * yn
     $          -     ph10 * abco
     $          -      cr9 * abco
     $          -     cp11 * abco
c
c M+ [silicon]
c
      ydot(imp) = (0d0
     $          -     ch41 * abmp  * abe
     $          -   si2_gr * abmp  * 1d0
     $          +     ch44 * abmI  * abe) * yn
     $          +      ph3 * abmI
     $          +     cp13 * abmI
c
c He+
c
      ydot(ihep) =  cr6  * abheI
     $           +  ch14 * abheI * abe * yn
     $           - ( ch9 * abhep * abh2
     $           +  ch92 * abhep * abco
     $           +  ch17 * abhep * abe
     $         + cst(75) * abhep * abh2
     $         + heii_gr * abhep * 1d0
     $           + ch179 * abhep * abohx
     $           + ch181 * abhep * abchx) * yn
c
c H2 collisional dissociation cooling:
c
      rates_chem(1) = 4.48d0 * eV * (ykdh  * abHI * abh2 +
     $                               ykdh2 * abh2 * abh2 +
     $                               ch25  * abe  * abh2 ) * ynsq
c
c H2 photodissociation heating -- 0.4eV per photodissociation:
c
      rates_chem(2) = - 4d-1 * eV * abH2 * yn * ykdis
c
c Heating by UV pumping of H2 -- effective for n > n_crit
c
c We assume 2 eV per UV photon (Burton, Hollenbach & Tielens 1990),
c and adopt a pumping rate 6.94 times larger than the dissociation rate
c (Draine & Bertoldi 1996, table 2), as in our other networks.
c
      cdex = (1.4 * dexp(-1.81d4 / (temp + 1200)) * abH2
     $     +  1.0 * dexp(-1d3 / temp) * abHI) * 1d-12 * dsqrt(temp) * yn
      rdex = 2d-7
      fpheat = cdex / (cdex + rdex)

      rates_chem(3) = - 2d0 * eV * abH2 * yn * ykdis * 6.94d0 * fpheat
c
c H2 formation heating on dust: kinetic plus de-excited share of 4.48 eV
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
      rates_chem(5) = 13.6d0 * eV * ch11 * abe * abHI * ynsq
c
c HII recombination cooling -- we assume that kT ergs is radiated
c per recombination and that this is independent of whether the
c recombination occurs in the gas phase or on a grain surface
c
      rates_chem(6) = kboltz * temp * abhp * (ch13 * abe + h_gr) * ynsq
c
      ylam_chem = rates_chem(1)  + rates_chem(2)  + rates_chem(3)  +
     $            rates_chem(4)  + rates_chem(5)  + rates_chem(6)
      do i = 1, 6
        lambda_chem(i) = rates_chem(i)
      enddo
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
      abundances(8)  = abmI   ! M is silicon
      abundances(9)  = abmp
      abundances(10) = abe
      abundances(11) = abhp
      abundances(12) = abHI
      abundances(14) = abheI
      abundances(15) = abhep
c
c Local Jeans length and energy for cool_func, as in rate_eq_simple.F
      rho_lj  = (1d0 + 4d0 * abhe) * PROTONMASS * yn
      cs2_lj  = gamma * (gamma - 1d0) * energy / rho_lj
      L_jeans = dsqrt(PI * cs2_lj /
     $          (GRAVITY * rho_lj))
      energy_dummy = energy
      call cool_func(temp, yn, L_jeans, divv, G_dust, abundances, ylam,
     $               rates, 0, energy_dummy)
c
c T_CMB floor
      CMB_temp = 2.726d0 * (1d0 + redshift)
      call cool_func(CMB_temp, yn, L_jeans, divv, G_dust, abundances,
     $               ylam_floor, rates_floor, 1, energy_dummy)
      ydot(itmp) = -(ylam + ylam_chem - ylam_floor)
c
#ifdef THERMAL_INFO_DUMP
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
#endif /* CHEMCOOL */
      return
      end
c=======================================================================
c
c    \\\\\\\\\\        E N D   S U B R O U T I N E        //////////
c    //////////              R A T E _ E Q                \\\\\\\\\\
c
c=======================================================================
#endif /* CHEMISTRYNETWORK == 16 */
