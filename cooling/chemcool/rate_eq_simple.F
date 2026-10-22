#include "../../GIZMO_config.h"
#if CHEMISTRYNETWORK == 4 || CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6
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
     $     rates_chem(nrates_chem),
     $     y_in(nspec)
      REAL CMB_temp, ylam_floor, rates_floor(nrates)
c
      REAL abe     , abhp    , abHI    , abh2    ,
     $     abcp    , abco    , abo     , abcos
c
      REAL ylam    , temp    ,
     $     yn      , dl      , divv    , yntot   , gamma , 
     $     energy  , dtemp   , ncrinv  , h2var0  , 
     $     h2var2  , cdh2_h  , cdh2_h_dt, cdh2_h2, cdh2_he,
     $     ykdis   , NH    ,
     $     ylam_chem, f_ex   , rho     , cs2     , L_jeans,
     $     f_tot   , RH2     , Rdust   , ekn, energy_dummy
c
      REAL fshield
c
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6
      REAL k0, k1, gamma_chx, gamma_co, beta
#if CHEMISTRYNETWORK == 6
      REAL kd1, kd2, kd3
#endif
#endif
c
      REAL phi     , h_gr    , hgrvar1 , hgrvar2
c
      REAL cr1, cr2, cr10, cr11
c
      REAL ch2     , ch4     , ch5     ,
     $     ch11    , ch13    , ch25    ,
     $     ch34    , ch35    , ch137   , ch138   , ch139   ,
     $     ch161   , ch162   , ch163   , ch164   , ch165   ,
     $     ch166   , ch167   , ch168   , ch169   , ch170   ,
     $     ch171   , ch172
c
      REAL dtch2   , dtch4   , dtch5   ,
     $     dtch11  , dtch13  , dtch25  ,
     $     dtch34  , dtch35  , dtch137 , dtch138 , dtch139 ,
     $     dtch161 , dtch162 , dtch163 , dtch164 , dtch165 ,
     $     dtch166 , dtch167 , dtch168 , dtch169 , dtch170 ,
     $     dtch171 , dtch172
c
      REAL cdex, rdex, fpheat
c M1 RT photoionization (ph38) + photoheating (ph42) of HI; mirrors net-17.
c Both are 0 unless RADTRANSFER is active (calc_photo adds the C-side rates
c into phrates(38)/phrates(42)), so subgrid-net-5 behaviour is unchanged.
      REAL ph38, ph42, abHI_ph
c
      REAL mh
      parameter (mh  = PROTONMASS)
c
c Unpack parameters
c
      yn   = rpar(1)
      dl   = rpar(2)
      divv = rpar(3)
c
      NH = yn * dl
c
c Chemical species:
c
      do I = 1, nspec
        y_in(I) = y(I)
      enddo
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
      abhp = max(0d0, y(ihp))
      abh2 = max(0d0, y(ih2))
      abco  = 0d0
      abcos = 0d0
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6
      abco = max(0d0, y(ico))
#endif
#if CHEMISTRYNETWORK == 6
      abcos = max(0d0, y(icos))
#endif

c
c Non-equilibrium species computed using conservation laws
c
      abHI = max(1d0 - 2d0 * abh2 - abhp, 0d0)
c H+ above 1 (clipped by validate_input) stays a restoring PI sink
      abHI_ph = 1d0 - 2d0 * abh2 - max(y_in(ihp), abhp)
      if (abHI_ph .ge. 0d0) abHI_ph = abHI
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6
      abcp = max(0d0, abundc - abco - abcos)
      abe  = abhp + abcp + abundsi
#else
      abcp = abundc
      abe  = abhp + abundc + abundsi
#endif
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



#ifdef BS19_XH2
      abh2 = 1d-7 * (temp*1d-3)**1.01 * (1.0) 
     $     * sqrt(cosmic_ray_ion_rate*1d16 * yn)
     $     / (G0 / 1.7)

      if (abh2 .gt. 0.5) then
         abh2 = 0.5
      endif
#endif


c
c      if (id_current .eq. 1258) then
c        print*, "Particle ", id_current, yntot, ekn, temp
c      endif
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
          print*, 'Fatal error in rate_eq.F', itemp, temp, id_current 
          ABORT(1)
        endif
        dtemp = temp - temptab(itemp)
      endif
c
c Read in coefficients
c
      dtch2  = dtchtab(2, itemp)
      dtch4  = dtchtab(4, itemp)
      dtch5  = dtchtab(5, itemp)
      dtch11 = dtchtab(11, itemp)
      dtch13 = dtchtab(13, itemp)
      dtch25 = dtchtab(25, itemp)
      dtch34 = dtchtab(34, itemp)
      dtch35 = dtchtab(35, itemp)
      dtch137 = dtchtab(137, itemp)
      dtch138 = dtchtab(138, itemp)
      dtch139 = dtchtab(139, itemp)
      dtch161 = dtchtab(161, itemp)
      dtch162 = dtchtab(162, itemp)
      dtch163 = dtchtab(163, itemp)
      dtch164 = dtchtab(164, itemp)
      dtch165 = dtchtab(165, itemp)
      dtch166 = dtchtab(166, itemp)
      dtch167 = dtchtab(167, itemp)
      dtch168 = dtchtab(168, itemp)
      dtch169 = dtchtab(169, itemp)
      dtch170 = dtchtab(170, itemp)
      dtch171 = dtchtab(171, itemp)
      dtch172 = dtchtab(172, itemp)
c
      ch2  = chtab(2, itemp) + dtemp * dtch2
      ch4  = chtab(4, itemp) + dtemp * dtch4
      ch5  = chtab(5, itemp) + dtemp * dtch5
      ch11 = chtab(11, itemp) + dtemp * dtch11
      ch13 = chtab(13, itemp) + dtemp * dtch13
      ch25 = chtab(25, itemp) + dtemp * dtch25
      ch34 = chtab(34, itemp) + dtemp * dtch34
      ch35 = chtab(35, itemp) + dtemp * dtch35
      ch137 = chtab(137, itemp) + dtemp * dtch137
      ch138 = chtab(138, itemp) + dtemp * dtch138
      ch139 = chtab(139, itemp) + dtemp * dtch139
      ch161 = chtab(161, itemp) + dtemp * dtch161
      ch162 = chtab(162, itemp) + dtemp * dtch162
      ch163 = chtab(163, itemp) + dtemp * dtch163
      ch164 = chtab(164, itemp) + dtemp * dtch164
      ch165 = chtab(165, itemp) + dtemp * dtch165
      ch166 = chtab(166, itemp) + dtemp * dtch166
      ch167 = chtab(167, itemp) + dtemp * dtch167
      ch168 = chtab(168, itemp) + dtemp * dtch168
      ch169 = chtab(169, itemp) + dtemp * dtch169
      ch170 = chtab(170, itemp) + dtemp * dtch170
      ch171 = chtab(171, itemp) + dtemp * dtch171
      ch172 = chtab(172, itemp) + dtemp * dtch172
c
c H2 collisional dissociation, as rate_eq_ulli.F: MSM96 CID + DT by H; He neutral
c
      cdh2_h = (ch161 - (ch161 - ch163)
     $       / (1d0 + (yn * abHI / ch165)**ch166))
     $       + (ch162 - (ch162 - ch164)
     $       / (1d0 + (yn * abHI / (1d1**(-1.164408e0)
     $       * ch165))**ch166))
      cdh2_h = 1d1**cdh2_h
c
      cdh2_h_dt = (ch167 - (ch167 - ch169)
     $          / (1d0 + (yn * abHI / ch171)**ch172))
     $          + (ch168 - (ch168 - ch170)
     $          / (1d0 + (yn * abHI / (1d1**(-1.174242e0)
     $          * ch171))**ch172))
      cdh2_h_dt = 1d1**cdh2_h_dt
c
      cdh2_h = cdh2_h + cdh2_h_dt
c
      ncrinv   = ch5 + abhe * ch139
      h2var0   = 1d0 / ( 1d0 + yn * ncrinv)
      h2var2   = ch4**h2var0
      cdh2_h2  = ch2 * h2var2
      if (ch138 .lt. 1d-100) then
        cdh2_he = 1d-100
      else
        cdh2_he = ch138 * (ch137 / ch138)**h2var0
      endif
c
c Photodissociation:
c
      if (iphoto .eq. 0) then
         fshield = 1d0
      else
         call calc_H2_shielding_factor(fshield, temp, abh2, NH, rpar)
      endif

c      call calc_H2_shielding_factor(fshield, temp, abh2, NH, rpar)
      ykdis = phrates(1) * fshield
c      if (id_current .eq. 10) then
c         print*, "phrates(1)=", phrates(1), "  fshield=", fshield
c      endif

#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6
      gamma_co = phrates(10)
      gamma_chx = 2.94d-10 * G_dust
c M1 RT ionizing-band rates (HI photoionization + heating), 0 without RADTRANSFER
      ph38 = phrates(38)
      ph42 = phrates(42)
#endif
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
c Grain recombination fits are valid for phi > 100 only
      phi = max(phi, 1d2)
c
c HII:
c
      hgrvar1  = 8.074d-6 * phi**1.378d0
      hgrvar2  = (1d0 + ch34 * phi**ch35)
      h_gr     = 1.225d-13 * dust_to_gas_ratio /
     $           (1d0 + hgrvar1 * hgrvar2)

      h_gr = h_gr * dexp(-temp / 3d4)
c
#if defined(RT_INFRARED)
c     M1 IR solver owns the dust temperature (chemcool.cc set tdust from
c     CellP.Dust_Temperature on entry); skip the optically-thin solve but
c     still compute RH2 and Rdust for H2 formation on dust. NOTE: gate is
c     RT_INFRARED (not RADTRANSFER) -- without the IR band nobody updates
c     Dust_Temperature and skipping here would freeze tdust.
      call calc_dust_temp(yn, chi_isrf, temp, abHI, RH2, Rdust, 1)
#else
      call calc_dust_temp(yn, chi_isrf, temp, abHI, RH2, Rdust, 0)
#endif
c
c Cosmic ray ionization rates: per unit rate from cheminmo, times the cell rate
c
c HI:
      cr1  = crtab(1) * cosmic_ray_ion_rate  
      cr2  = crtab(2) * cosmic_ray_ion_rate   ! H2 -> H2+ + e-
      cr10 = crtab(10) * cosmic_ray_ion_rate  ! H2 -> H + H+ + e
      cr11 = crtab(11) * cosmic_ray_ion_rate  ! H2 -> H + H
c
c Protons
c 
      ydot(ihp) = (ch11 * abe   * abHI          ! Collisional ion: H  + e-
     $          -  ch13 * abhp  * abe           ! Gas-phase H recomb.(case B)
     $          -  h_gr * abhp  * 1d0) * yn     ! Grain H recomb.
     $          +  cr1  * abHI                  ! Cosmic ray ion.:  H + CR
     $          + cr10  * abh2
     $          + ph38  * abHI_ph               ! M1 RT photoionization: H + gamma(>13.6eV)
c
c Molecular hydrogen
c
      ydot(ih2) = (-ch25  * abe  * abh2          ! Collisional dissoc.: H2 + e- 
     $          -   cdh2_h  * abHI * abh2        ! Collisional dissoc.: H2 + H 
     $          -   cdh2_h2 * abh2 * abh2        ! Collisional dissoc.: H2 + H2
     $          -   cdh2_he * abhe * abh2        ! Collisional dissoc.: H2 + He
     $          +   Rdust * abHI * 1d0 ) * yn    ! H2 formation on dust
     $          -   ykdis * abh2                 ! Photodissoc:         H2 + PH
     $      - 0.5d0 * cr2 * abh2   ! Fudge factor of two gives correct HI production
     $          -   cr10  * abh2
     $          -   cr11  * abh2



c      if (id_current .eq. 10) then
c         print*, "Rdust=", Rdust, "  temp=", temp, 
c     $        "ykdis", ykdis, "  abh2=", abh2
c      endif

c
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6
c
c CO -- from NL97
c
      k0 = 5d-16
      k1 = 5d-10
#if CHEMISTRYNETWORK == 5
      abo = max(0.0, abundo - abco)
#endif
#if CHEMISTRYNETWORK == 6
      abo = max(0.0, abundo - abco - abcos)
#endif
      if (abo .eq. 0.0) then
        beta = 0.0
      elseif (gamma_chx .lt. 1d-30) then
        beta = 1.0
      else
        beta = k1 * abo / (k1 * abo + 
     $       gamma_chx / yn)
      endif
c
      ydot(ico) = k0 * abcp * abh2 * beta * yn 
     $          - gamma_co * abco
c
#if CHEMISTRYNETWORK == 6
c
      kd1 = 1.1d-17 * dsqrt(temp)  ! Accretion
      kd2 = 1.04d12 * dexp(-1.21d3 / tdust) ! Thermal evaporation
      kd3 = 9.8d-15 ! Cosmic ray desorption
c
      ydot(icos) = kd1 * abco * yn - (kd2 + kd3) * abcos
      ydot(ico) = ydot(ico) - ydot(icos)
c
#endif
#endif

#ifdef NO_CO_FORM
c      y(ico) = 0.0
      ydot(ico) = 0.0
#endif


#ifdef CHEM_EQ_H2
      ydot(ih2) = 0.0
#endif

#ifdef BS19_XH2
      ydot(ih2) = 0.0
c      y(ih2) = abh2
#endif

c
c H2 collisional dissociation cooling:
c
      rates_chem(1) = 4.48d0 * eV * (cdh2_h  * abHI * abh2 + 
     $                               cdh2_h2 * abh2 * abh2 +
     $                               cdh2_he * abhe * abh2 +
     $                               ch25  * abe  * abh2 ) * yn**2
c
c H2 photodissociation heating -- 0.4eV per photodissociation:
c
      rates_chem(2) = - 4d-1 * eV * abH2 * yn * ykdis
c
c Heating by UV pumping of H2 -- effective for n > n_crit
c
c We assume 2 eV per UV photon (Burton, Hollenbach & Tielens 1990),
c and a pumping rate 6.94 times the dissociation rate (as rate_eq_ulli.F).
c
      cdex = (1.4 * dexp(-1.81d4 / (temp + 1200)) * abH2
     $     +  1.0 * dexp(-1d3 / temp) * abHI) * 1d-12 * dsqrt(temp) * yn
      rdex = 2d-7
      fpheat = cdex / (cdex + rdex)
#ifdef CONST_NCRIT_UVPUMP
      fpheat = 1.0 / (1.0 + 1e4 / yn)
#endif

      rates_chem(3) = - 2d0 * eV * abH2 * yn * ykdis * 6.94d0 * fpheat
c
c H2 formation heating -- we assume a similar depedence on n / n_crit to 
c the collisional dissociation rates, although ideally one would like to
c do a more careful, level-dependent treatment.
c
c We assume that a fraction f_ex of the energy goes into rotational &
c vibrational excitation of the H2, and a further fraction f_kin goes
c directly into kinetic energy (i.e. heat). The remaining energy goes
c directly into the dust 
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

      if (temp .lt. 100.) then
         rates_chem(6) = 0.
      endif
c
      ylam_chem = rates_chem(1)  + rates_chem(2)  + rates_chem(3)  +
     $            rates_chem(4)  + rates_chem(5)  + rates_chem(6)
     $          - (ph42 * abHI) * yn        ! M1 RT photoheating (neg = heating)


c#ifdef OUTPUT_INDIVIDUAL_COOLRATES
      lambda_chem(1) = rates_chem(1)
      lambda_chem(2) = rates_chem(2)
      lambda_chem(3) = rates_chem(3)
      lambda_chem(4) = rates_chem(4)
      lambda_chem(5) = rates_chem(5)
      lambda_chem(6) = rates_chem(6)
c#endif



c      if (temp.gt.1d7) then
c         print *, "temp =  ", temp
c         print *, "rates_chem(1) =  ", rates_chem(1)
c         print *, "rates_chem(2) =  ", rates_chem(2)
c         print *, "rates_chem(3) =  ", rates_chem(3)
c         print *, "rates_chem(4) =  ", rates_chem(4)
c         print *, "rates_chem(5) =  ", rates_chem(5)
c         print *, "rates_chem(6) =  ", rates_chem(6)
c      end if


c     
c Initialize all abundances to zero, then just set the non-zero ones
c
      do i = 1, nabn
        abundances(i) = 0d0
      enddo
c
      abundances(1)  = abh2
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6
      abundances(2)  = max(0d0, abundo - abco - abcos) * (1d0 - abhp)  ! Assume O+/O = H+/H
      abundances(5)  = abco
      abundances(7)  = abcp
#else
      abundances(2)  = abundo
      abundances(7)  = abundc
#endif
      abundances(9)  = abundsi
      abundances(10) = abe
      abundances(11) = abhp
      abundances(12) = abHI
c
#ifdef ADIABATIC_DENSITY_THRESHOLD
      if (yn .ge. yn_adiabatic) then
        ydot(itmp) = 0d0
        return
      endif
#endif
c
c Compute local Jeans length (for use in cool_func)
c
      rho = (1d0 + 4d0 * abhe) * mh * yn
      cs2 = (gamma * (gamma - 1d0) * energy / rho)
      if (cs2 .lt. 0d0) cs2 = 0d0
      L_jeans  = dsqrt(PI * cs2 /
     $           (GRAVITY * rho))
c
      energy_dummy = energy
      call cool_func(temp, yn, L_jeans, divv, G_dust, abundances, ylam,
     $               rates, 0, energy_dummy)
      
c
c T_CMB floor
      CMB_temp = 2.726d0 * (1d0 + redshift)
      call cool_func(CMB_temp, yn, L_jeans, divv, G_dust, abundances,
     $               ylam_floor, rates_floor, 1, energy_dummy)
      ydot(itmp) = -(ylam + ylam_chem - ylam_floor) + pdv_term
c
#if 0
c Useful for debugging if we can isolate problems to a single
c particle. 
      if (id_current .eq. 1345975) then
        print*, "Particle ID", id_current
        print*, 'yn, temp = ', yn, temp
        print*, 'y = ', y
        print*, 'ydot = ', ydot
        print*, 'ylam, ylam_chem, pdv_term = ', ylam,ylam_chem,pdv_term
        print*, 'G_dust = ', G_dust
        print*, 'rates_chem = ', rates_chem
        print*, 'rates = ', rates
        print*, 'tdust = ', tdust
        print*, 'divv = ', divv
      endif
#endif

      call validate_output(y,ydot,rpar)
c
c#ifdef DEBUG_RS
      if (temp .lt. 0.) then     
        print*, 'RES', y, ydot, ylam, ylam_chem
        print*, 'RES', yn, dl, divv, temp, pdv_term
        print*, 'RES', abundances, G_dust
        print*, 'RES', rates
        stop
      endif
c#endif
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
#endif /* CHEMISTRYNETWORK == 4 || 5 || 6 */
#if CHEMISTRYNETWORK == 20 || CHEMISTRYNETWORK == 18
c Networks 18 and 20 exist only in the C++ backend (CHEMCOOL_CXX); this stub satisfies the link.
      subroutine rate_eq(neq, time, y, ydot, rpar, ipar)
      implicit NONE
      integer neq, ipar(*), code
      real*8 time, y(*), ydot(*), rpar(*)
      code = CHEMISTRYNETWORK
      call chemcool_fatal(code)
      return
      end
#endif /* CHEMISTRYNETWORK == 20 || 18 */
