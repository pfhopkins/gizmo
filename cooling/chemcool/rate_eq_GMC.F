#include "gadgetconfig.h"
#if CHEMISTRYNETWORK == 7 || CHEMISTRYNETWORK == 11
c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////             R A T E _ E Q                 \\\\\\\\\\
c
c=======================================================================
c
c    Written by S. Glover (2004-2007)
c
c  REFERENCES
c
c   FJ84   -- Frank & Just, 1984, Proc. Int. Symp. Shock Tubes Waves, 14, 706
c   F86    -- Frank, 1986, Proc. Int. Symp. Rarefied Gas Dyn., 2, 422
c   FBJ88  -- Frank, Bhaskaran, & Just, 1988, Symp. Int. Combust. Proc., 21, 885
c   JBS00  -- Jensen, Bilodeau, Safvan et al, 2000, ApJ, 543, 764
c   RDS00  -- Rosen, Derkatch, Semaniak et al, 2000, Far. Disc., 115, 295
c   BHRB04 -- Bergin et al, 2004, ApJ, 612, 921
c   MAC04  -- McCall et al, 2004, Phys. Rev. A, 70, 052716
c
      subroutine rate_eq(neq, time, y, ydot, rpar, ipar)
#ifdef CHEMCOOL
      implicit NONE
#include "cool.h"
#include "non_eq.h"
#include "shield_data.h"
#include "current_photo.h"
      integer neq
c
      REAL time, y(nspec), ydot(nspec), rpar(nrpar)
      integer ipar(nipar)
c
      REAL y_save(nspec)
      integer niter
      parameter (niter = 50)
      REAL eqtol
      parameter (eqtol = 1d-3)
c
      REAL    dtemp
      integer itemp, I, K
c
      integer ndust_iter
      parameter (ndust_iter = 20)
c
      REAL abundances(nabn), rates(nrates),
     $     rates_chem(nrates_chem), ylam, ylam_chem,
     $     background_temp, ylam_back, rates_back(nrates)
c
      REAL temp, yn, dl, divv, energy, gamma, yntot,
     $     ynsq, yncb  , f_ex, f_tot , RH2  , ekn
c
      REAL abhp, abh2, abhep, abcp,
     $     abop, aboh, abh2o, abco, abc2, abo2,
     $     abhcop, abch, abch2, abch3p, ab3M
c
      REAL abhm, abhm_old, abh2p, abh2p_old, abh3p, abh3p_old,
     $     abchp, abchp_old, abch2p, abch2p_old,
     $     abh3op, abcop_old, abcop, 
     $     abohp_old, abohp, abh2op_old, abh2op, abh3op_old,
     $     abhocp_old, abhocp, abom_old, abom, abcm_old,
     $     abcm, abo2p_old, abo2p, abch4
c
      REAL abcos, abh2os, abo2s, abch4s
c
#if CHEMISTRYNETWORK == 11
      REAL ice_fraction, nsurf, surface_fraction,
     $     ice_frac_co, ice_frac_h2o, ice_frac_o2, ice_frac_ch4,
     $     df18, kd_pd, kd_acc_c, kd_acc_H, kd_acc_CH4, kd_des_c, 
     $     kd_des_ch, kd_acc_H_single, c_react, ch_react, frac_ch4, 
     $     frac_ch, kd_ch4_cr, kd_ch4_th, kd_ch4_ph, kd_acc_co, 
     $     kd_co_th, kd_co_cr, kd_co_ph, kd_acc_o, kd_acc_oh, 
     $     kd_acc_h2o, kd_o_th, frac_o, kd_h2o_th, kd_h2o_cr, kd_h2o_ph, 
     $     kd_acc_o2, kd_o2_th, kd_o2_cr, kd_o2_ph

      integer nlayer
#endif
c
#ifdef FF_COLLAPSE_TEST
      REAL n0, tffc, ndot, compress, tff
#endif
      REAL num, denom
      REAL comp_equilib
      external comp_equilib

      REAL time_to_equilibrium(num_eqb_species)

      REAL abe, abHI, abheI, abcI, aboI

      REAL ph2,  ph4,  ph5, ph6, ph8, ph9,
     $     ph10, ph11, ph12, 
     $     ph14, ph15, ph16, ph18, ph19, ph20, ph21,
     $     ph22, ph23, ph24, ph25, ph27, ph28, ph29,
     $     ph30, ph33, ph34, ph36, ph37, ph38, ph39,
     $     ph41, ph42, ph43, ph45, ph46

      REAL ph16a, ph16b, ph33a, ph33b, ph33c, ph33d, ph34a, 
     $     ph34b, ph34c, ph34d

      REAL cr1, cr2, cr3, cr4, cr6, cr9, cr10, cr11, cr12
      REAL cp1, cp2, cp3, cp4, cp5, cp6, cp7, cp8, cp9, cp10, cp11

      REAL ykdh, ykdh2, h_gr, ykdis, phi, ncrinv, 
     $     hgrvar1, hgrvar2, cii_gr, c2grvar1, c2grvar2, 
     $     oii_gr, h2var0, 
     $     h2var1, h2var2, NH, heii_gr, hegrvar1, hegrvar2,
     $     Rdust, energy_dummy

      REAL fshield

      REAL  ch1,  ch2,  ch3,  ch4,  ch5,  ch6,  ch7,  ch8,
     $      ch9, ch10, ch11, ch13, ch14, ch17, ch18, ch21, 
     $     ch22, ch23, ch24, ch25, ch26, ch27, ch28, ch29,
     $     ch30, ch32, ch33, ch34, ch35, ch36, ch37,
     $     ch40, ch42, ch43,
     $     ch45, ch46, ch47, ch48, ch50, ch51, ch52,
     $     ch53, ch54, ch55, ch56, ch57, ch58, ch59, ch60,
     $     ch61, ch62, ch63, ch64, ch65, ch66, ch67, ch68,
     $     ch69, ch70, ch71, ch72, ch71a, ch71b, ch71c, ch71d,
     $     ch73, ch73a, ch73b, ch73c,
     $     ch77, ch78, ch79, ch80, ch81, 
     $     ch82, ch83, ch84, ch85, ch86, ch87, ch88, ch89,
     $     ch90, ch92, ch93, ch94, ch97, ch104, ch105, 
     $     ch106, ch107, ch108, ch109, ch110, ch111, ch112,
     $     ch113, ch114, ch115, ch116, ch117, ch118, ch119,
     $     ch124, ch131, ch132, ch133, ch134, ch135, ch136,
     $     ch145

      REAL ch87a, ch87b, ch106a, cst64a, 
     $     cst64b

      REAL  dtch1,  dtch2,  dtch3,  dtch4,  dtch5,  dtch6,
     $      dtch7,  dtch8,  dtch9, dtch10, dtch11, dtch13, 
     $     dtch14, dtch17, dtch18, 
     $     dtch21, dtch22, dtch23, dtch24, dtch25, dtch26,
     $     dtch27, dtch28, dtch29, dtch30,
     $     dtch32, dtch33, dtch34, dtch35, dtch36, dtch37,
     $     dtch40, dtch42, dtch43, 
     $     dtch45, dtch46, dtch47, dtch48, 
     $     dtch50, dtch51, dtch52, dtch53, dtch54, dtch55, 
     $     dtch56, dtch57, dtch58, dtch59, dtch60, dtch61,
     $     dtch62, dtch63, dtch64, dtch65, dtch66, dtch67, 
     $     dtch68, dtch69, dtch70, dtch71, dtch72, dtch73, 
     $     dtch77, dtch78, dtch79,
     $     dtch80, dtch81, dtch82, dtch83, dtch84, dtch85, 
     $     dtch86, dtch87, dtch88, dtch89, dtch90, dtch92, 
     $     dtch93, dtch94, dtch97,
     $     dtch104, dtch105, dtch106, dtch107, dtch108,
     $     dtch109, dtch110, dtch111, dtch112, dtch113,
     $     dtch114, dtch115, dtch116, dtch117, dtch118,
     $     dtch119, dtch124, dtch131, dtch132, dtch133, 
     $     dtch134, dtch135, dtch136, dtch145
c
      REAL mh
      parameter (mh  = PROTONMASS)
#ifdef FF_COLLAPSE_TEST
      REAL G_n
      parameter (G_n = GRAVITY)
#endif
c
c Unpack parameters
c
      yn    = rpar(1)
      dl    = rpar(2)
      divv  = rpar(3)
c
c Total column density of HI nuclei within characteristic scale-length dl
c
      NH = dl * yn
c
#ifdef FF_COLLAPSE_TEST
      n0  = rpar(4)
      tffc =  dsqrt(32d0 * G_n * 1.4d0 * mh / (3d0 * PI))
      yn = (1d0 - 0.5d0 * tffc * dsqrt(n0) * time)**(-2d0) * n0 
#endif
c
      do I = 1, nspec
        y_save(I) = y(I)
      enddo
c
      call validate_input(y, ipar)
      if (ipar(1) .eq. 1) then
        do I = 1, nspec
          ydot(I) = 0d0
          y(I) = y_save(I)
        enddo
c We simply return here because we'll be redoing the step anyway
c (making any further work pointless) and because we may be doing 
c this on account of having negative chemical abundances, which 
c would otherwise cause serious problems later on.
        return
      endif
c
c From this point on, we know that our tracked abundances cannot be negative
c
      abhp   = y(ihp)
      abh2   = y(ih2)
      abhep  = y(ihep)
      abcp   = y(ic)
      abop   = y(io)
      aboh   = y(ioh)
      abh2o  = y(ih2o)
      abco   = y(ico)
      abc2   = y(ic2)
      abo2   = y(io2)
      abhcop = y(ihcop)
      abch   = y(ich)
      abch2  = y(ich2)
      abch3p = y(ich3p)
      abch4  = 0d0
      abch4s = 0d0
      abcos  = 0d0
      abh2os = 0d0
      abo2s  = 0d0
#if CHEMISTRYNETWORK == 11
      abch4  = max(abch4,  y(ich4))
      abch4s = max(abch4s, y(ich4s))
      abcos  = max(abcos,  y(icos))
      abh2os = max(abh2os, y(ih2os))
      abo2s  = max(abo2s,  y(io2s))
#endif
c
      abe  = abhp + abhep + abcp + abop + abhcop + abch3p
c
      abHI = 1d0 - 2d0 * abH2 - abhp - aboh 
     $     - 2d0 * abh2o - abhcop - abch - 2d0 * abch2 - 3d0 * abch3p
     $     - 2d0 * abh2os - 4d0 * abch4 - 4d0 * abch4s
      abHI = max(0d0, abHI)
c
      abheI = max(0d0, abhe - abhep)
c
      abcI = abundc - abcp - abco - 2d0 * abc2 - abhcop - abch - abch2 -
     $       abch3p - abch4 - abcos - abch4s
      abcI = max(0d0, abcI)
c
      aboI = abundo - abop - aboh - abh2o - abco - 2d0 * abo2 
     $     - abhcop - abcos - abh2os - 2d0 * abo2s
      aboI = max(0d0, aboI)
c
c For some three-body reactions, we only have a total rate, not individual
c rates for collisions with H, H2, He etc. To simplify our handling of 
c these reactions, we define an abundance ab3M that is simply the sum of
c the abundances of the three main colliders:
c
      ab3M = abHI + abh2 + abheI
c
c Internal energy (erg cm^-3)
      energy = y(itmp)
c
c Compute current adiabatic index and total number density
      yntot = (1d0 + abhe - abh2 + abe) * yn
      ekn = energy / (yntot * kboltz)
      call lookup_temp_gamma(abh2, ekn, temp, gamma)
c
c Useful auxiliary functions
c
      ynsq = yn**2
      yncb = yn**3
c
c Bounds checking
c
      if (temp .le. 0d0) then
        itemp   = 1
        dtemp   = 0d0
        temp    = tmin
        ipar(1) = 1
      elseif (temp .le. tmin) then
        itemp = 1
        temp  = tmin
        dtemp = 0d0
      elseif (temp .ge. tmax) then
        itemp = nmd
        dtemp = 0d0
      else
        itemp = int(dlog10(temp) / dtlog) + 1
        if (itemp .le. 0 .or. itemp .gt. nmd) then
          print*, 'Fatal error in rate_eq.F', itemp, temp
          ABORT(1)
        endif
        dtemp = temp - temptab(itemp)
      endif
c
c Rates:
c
       dtch1  = dtchtab(1, itemp)
       dtch2  = dtchtab(2, itemp)
       dtch3  = dtchtab(3, itemp)
       dtch4  = dtchtab(4, itemp)
       dtch5  = dtchtab(5, itemp)
       dtch6  = dtchtab(6, itemp)
c
       dtch8  = dtchtab(8, itemp)
       dtch9  = dtchtab(9, itemp) 
       dtch10 = dtchtab(10,itemp)
       dtch11 = dtchtab(11,itemp) 
       dtch13 = dtchtab(13,itemp)
       dtch14 = dtchtab(14,itemp)
       dtch17 = dtchtab(17,itemp)
       dtch18 = dtchtab(18,itemp)
       dtch21 = dtchtab(21,itemp)
       dtch22 = dtchtab(22,itemp)
       dtch23 = dtchtab(23,itemp)
       dtch24 = dtchtab(24,itemp)
       dtch25 = dtchtab(25,itemp)
       dtch26 = dtchtab(26,itemp)
       dtch27 = dtchtab(27,itemp)
       dtch28 = dtchtab(28,itemp)
       dtch29 = dtchtab(29,itemp)
       dtch30 = dtchtab(30,itemp)
       dtch32 = dtchtab(32,itemp)
       dtch33 = dtchtab(33,itemp)
       dtch34 = dtchtab(34,itemp)
       dtch35 = dtchtab(35,itemp)
       dtch36 = dtchtab(36,itemp)
       dtch37 = dtchtab(37,itemp)
c
       dtch40 = dtchtab(40,itemp)
c
       dtch42 = dtchtab(42,itemp)
       dtch43 = dtchtab(43,itemp)
c
       dtch45 = dtchtab(45,itemp)
       dtch46 = dtchtab(46,itemp)
       dtch47 = dtchtab(47,itemp)
       dtch48 = dtchtab(48,itemp)
c
       dtch50 = dtchtab(50,itemp)
       dtch51 = dtchtab(51,itemp) 
       dtch52 = dtchtab(52,itemp)
       dtch53 = dtchtab(53,itemp)
       dtch54 = dtchtab(54,itemp)
       dtch55 = dtchtab(55,itemp)
       dtch56 = dtchtab(56,itemp)
       dtch57 = dtchtab(57,itemp)
       dtch58 = dtchtab(58,itemp)
       dtch59 = dtchtab(59,itemp)
       dtch60 = dtchtab(60,itemp)
       dtch61 = dtchtab(61,itemp) 
       dtch62 = dtchtab(62,itemp)
       dtch63 = dtchtab(63,itemp)
       dtch64 = dtchtab(64,itemp)
       dtch65 = dtchtab(65,itemp)
       dtch66 = dtchtab(66,itemp)
       dtch67 = dtchtab(67,itemp)
       dtch68 = dtchtab(68,itemp)
       dtch69 = dtchtab(69,itemp)
       dtch70 = dtchtab(70,itemp)
       dtch71 = dtchtab(71,itemp) 
       dtch72 = dtchtab(72,itemp) 
       dtch73 = dtchtab(73,itemp)
c
       dtch77 = dtchtab(77,itemp)
       dtch78 = dtchtab(78,itemp)
       dtch79 = dtchtab(79,itemp)
       dtch80 = dtchtab(80,itemp)
       dtch81 = dtchtab(81,itemp) 
       dtch82 = dtchtab(82,itemp)
       dtch83 = dtchtab(83,itemp)
       dtch84 = dtchtab(84,itemp)
       dtch85 = dtchtab(85,itemp)
       dtch86 = dtchtab(86,itemp)
       dtch87 = dtchtab(87,itemp)
       dtch88 = dtchtab(88,itemp)
       dtch89 = dtchtab(89,itemp)
       dtch90 = dtchtab(90,itemp)
       dtch92 = dtchtab(92,itemp)
       dtch93 = dtchtab(93,itemp)
       dtch94 = dtchtab(94,itemp)
c
       dtch97 = dtchtab(97,itemp)
c
       dtch104 = dtchtab(104,itemp)
       dtch105 = dtchtab(105,itemp)
       dtch106 = dtchtab(106,itemp)
       dtch107 = dtchtab(107,itemp)
       dtch108 = dtchtab(108,itemp)
       dtch109 = dtchtab(109,itemp)
       dtch110 = dtchtab(110,itemp)
       dtch111 = dtchtab(111,itemp)
       dtch112 = dtchtab(112,itemp)
       dtch113 = dtchtab(113,itemp)
       dtch114 = dtchtab(114,itemp)
       dtch115 = dtchtab(115,itemp)
       dtch116 = dtchtab(116,itemp)
       dtch117 = dtchtab(117,itemp)
       dtch118 = dtchtab(118,itemp)
       dtch119 = dtchtab(119,itemp)
c
       dtch124 = dtchtab(124,itemp)
c
       dtch131 = dtchtab(131,itemp)
       dtch132 = dtchtab(132,itemp)
       dtch133 = dtchtab(133,itemp)
       dtch134 = dtchtab(134,itemp)
       dtch135 = dtchtab(135,itemp)
       dtch136 = dtchtab(136,itemp)
c
       dtch145 = dtchtab(145,itemp)
c
       ch1  = chtab(1, itemp) + dtemp * dtch1
       ch2  = chtab(2, itemp) + dtemp * dtch2
       ch3  = chtab(3, itemp) + dtemp * dtch3
       ch4  = chtab(4, itemp) + dtemp * dtch4
       ch5  = chtab(5, itemp) + dtemp * dtch5
       ch6  = chtab(6, itemp) + dtemp * dtch6
c
       ch8  = chtab(8, itemp) + dtemp * dtch8
       ch9  = chtab(9, itemp) + dtemp * dtch9
       ch10 = chtab(10,itemp) + dtemp * dtch10 
       ch11 = chtab(11,itemp) + dtemp * dtch11
       ch13 = chtab(13,itemp) + dtemp * dtch13
       ch14 = chtab(14,itemp) + dtemp * dtch14
       ch17 = chtab(17,itemp) + dtemp * dtch17
       ch18 = chtab(18,itemp) + dtemp * dtch18
       ch21 = chtab(21,itemp) + dtemp * dtch21
       ch22 = chtab(22,itemp) + dtemp * dtch22
       ch23 = chtab(23,itemp) + dtemp * dtch23
       ch24 = chtab(24,itemp) + dtemp * dtch24
       ch25 = chtab(25,itemp) + dtemp * dtch25
       ch26 = chtab(26,itemp) + dtemp * dtch26
       ch27 = chtab(27,itemp) + dtemp * dtch27
       ch28 = chtab(28,itemp) + dtemp * dtch28
       ch29 = chtab(29,itemp) + dtemp * dtch29
       ch30 = chtab(30,itemp) + dtemp * dtch30
       ch32 = chtab(32,itemp) + dtemp * dtch32
       ch33 = chtab(33,itemp) + dtemp * dtch33
       ch34 = chtab(34,itemp) + dtemp * dtch34
       ch35 = chtab(35,itemp) + dtemp * dtch35
       ch36 = chtab(36,itemp) + dtemp * dtch36
       ch37 = chtab(37,itemp) + dtemp * dtch37
c
       ch40 = chtab(40,itemp) + dtemp * dtch40
c
       ch42 = chtab(42,itemp) + dtemp * dtch42
       ch43 = chtab(43,itemp) + dtemp * dtch43
c
       ch45 = chtab(45,itemp) + dtemp * dtch45
       ch46 = chtab(46,itemp) + dtemp * dtch46
       ch47 = chtab(47,itemp) + dtemp * dtch47
       ch48 = chtab(48,itemp) + dtemp * dtch48
c
       ch50 = chtab(50,itemp) + dtemp * dtch50
       ch51 = chtab(51,itemp) + dtemp * dtch51
       ch52 = chtab(52,itemp) + dtemp * dtch52
       ch53 = chtab(53,itemp) + dtemp * dtch53
       ch54 = chtab(54,itemp) + dtemp * dtch54
       ch55 = chtab(55,itemp) + dtemp * dtch55
       ch56 = chtab(56,itemp) + dtemp * dtch56
       ch57 = chtab(57,itemp) + dtemp * dtch57
       ch58 = chtab(58,itemp) + dtemp * dtch58
       ch59 = chtab(59,itemp) + dtemp * dtch59
       ch60 = chtab(60,itemp) + dtemp * dtch60
       ch61 = chtab(61,itemp) + dtemp * dtch61
       ch62 = chtab(62,itemp) + dtemp * dtch62
       ch63 = chtab(63,itemp) + dtemp * dtch63
       ch64 = chtab(64,itemp) + dtemp * dtch64
       ch65 = chtab(65,itemp) + dtemp * dtch65
       ch66 = chtab(66,itemp) + dtemp * dtch66
       ch67 = chtab(67,itemp) + dtemp * dtch67
       ch68 = chtab(68,itemp) + dtemp * dtch68
       ch69 = chtab(69,itemp) + dtemp * dtch69
       ch70 = chtab(70,itemp) + dtemp * dtch70
       ch71 = chtab(71,itemp) + dtemp * dtch71
       ch72 = chtab(72,itemp) + dtemp * dtch72
       ch73 = chtab(73,itemp) + dtemp * dtch73
c
       ch77 = chtab(77,itemp) + dtemp * dtch77
       ch78 = chtab(78,itemp) + dtemp * dtch78
       ch79 = chtab(79,itemp) + dtemp * dtch79
       ch80 = chtab(80,itemp) + dtemp * dtch80
       ch81 = chtab(81,itemp) + dtemp * dtch81
       ch82 = chtab(82,itemp) + dtemp * dtch82
       ch83 = chtab(83,itemp) + dtemp * dtch83
       ch84 = chtab(84,itemp) + dtemp * dtch84
       ch85 = chtab(85,itemp) + dtemp * dtch85
       ch86 = chtab(86,itemp) + dtemp * dtch86
       ch87 = chtab(87,itemp) + dtemp * dtch87
       ch88 = chtab(88,itemp) + dtemp * dtch88
       ch89 = chtab(89,itemp) + dtemp * dtch89
       ch90 = chtab(90,itemp) + dtemp * dtch90
       ch92 = chtab(92,itemp) + dtemp * dtch92
       ch93 = chtab(93,itemp) + dtemp * dtch93
       ch94 = chtab(94,itemp) + dtemp * dtch94
c
       ch97 = chtab(97,itemp) + dtemp * dtch97
c
       ch104 = chtab(104,itemp) + dtemp * dtch104
       ch105 = chtab(105,itemp) + dtemp * dtch105
       ch106 = chtab(106,itemp) + dtemp * dtch106
       ch107 = chtab(107,itemp) + dtemp * dtch107
       ch108 = chtab(108,itemp) + dtemp * dtch108
       ch109 = chtab(109,itemp) + dtemp * dtch109
       ch110 = chtab(110,itemp) + dtemp * dtch110
       ch111 = chtab(111,itemp) + dtemp * dtch111
       ch112 = chtab(112,itemp) + dtemp * dtch112
       ch113 = chtab(113,itemp) + dtemp * dtch113
       ch114 = chtab(114,itemp) + dtemp * dtch114
       ch115 = chtab(115,itemp) + dtemp * dtch115
       ch116 = chtab(116,itemp) + dtemp * dtch116
       ch117 = chtab(117,itemp) + dtemp * dtch117
       ch118 = chtab(118,itemp) + dtemp * dtch118
       ch119 = chtab(119,itemp) + dtemp * dtch119
c
       ch124 = chtab(124,itemp) + dtemp * dtch124
c
       ch131 = chtab(131,itemp) + dtemp * dtch131
       ch132 = chtab(132,itemp) + dtemp * dtch132
       ch133 = chtab(133,itemp) + dtemp * dtch133
       ch134 = chtab(134,itemp) + dtemp * dtch134
       ch135 = chtab(135,itemp) + dtemp * dtch135
       ch136 = chtab(136,itemp) + dtemp * dtch136
c
       ch145 = chtab(145,itemp) + dtemp * dtch145
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
c Photochemistry
c
      call calc_H2_shielding_factor(fshield, temp, abh2, NH, rpar)
      ykdis = phrates(1) * fshield
      ph2   = phrates(2)
c
      ph4   = phrates(4)
      ph5   = phrates(5)
      ph6   = phrates(6)
c
      ph8   = phrates(8)
      ph9   = phrates(9)
      ph10  = phrates(10)
      ph11  = phrates(11)
      ph12  = phrates(12)
c
      ph14  = phrates(14)
      ph15  = phrates(15)
      ph16  = phrates(16)
      ph16a = 0.5d0 * ph16
      ph16b = 0.5d0 * ph16
c
      ph18  = phrates(18)
      ph19  = phrates(19)
      ph20  = phrates(20)
      ph21  = phrates(21)
      ph22  = phrates(22)
      ph23  = phrates(23)
      ph24  = phrates(24)
      ph25  = phrates(25)
c
      ph27  = phrates(27)
      ph28  = phrates(28)
      ph29  = phrates(29)
      ph30  = phrates(30)
c
      ph33  = phrates(33)
      ph34  = phrates(34)
c
      ph36  = phrates(36)
      ph37  = phrates(37)
      ph38  = phrates(38)
      ph39  = phrates(39)
c
      ph41  = phrates(41)
      ph42  = phrates(42)
      ph43  = phrates(43)
c
      ph45  = phrates(45)
      ph46  = phrates(46)
c
c Branching ratios for H2O+ photodissociation
c
      ph33a = (1d0/6d0) * ph33    ! H2+ + O
      ph33b = (1d0/6d0) * ph33    ! H+  + OH 
      ph33c = (1d0/6d0) * ph33    ! O+  + H2
      ph33d = 0.5d0     * ph33    ! OH+ + H
c
c Branching ratios for H3O+ photodissociation
c
      ph34a = (1d0 / 3.3d0) * ph34    ! H+   + H2O
      ph34b = (1d0 / 3.3d0) * ph34    ! H2+  + OH
      ph34c = (1d0 / 11d0)  * ph34    ! H2O+ + H
      ph34d = (1d0 / 3.3d0) * ph34    ! OH+  + H2
c
c Cosmic ray ionization
c
      cr1  = crtab(1)  ! HI
      cr2  = crtab(2)  ! H2 -> H2+ + e-
      cr10 = crtab(10) ! H2 -> H + H+ + e
      cr11 = crtab(11) ! H2 -> H + H
      cr12 = crtab(12) ! H2 -> H+ + H-
      cr3  = crtab(3)  ! CI
      cr4  = crtab(4)  ! OI
      cr6  = crtab(6)  ! HeI
      cr9  = crtab(9)  ! CO
c
c Cosmic ray photoprocesses
c
      cp1  = crphot(1)  ! C
      cp2  = crphot(2)  ! CH
      cp3  = crphot(3)  ! CH+
      cp4  = crphot(4)  ! CH2 -> CH2+
      cp5  = crphot(5)  ! CH2 -> CH
      cp6  = crphot(6)  ! C2
      cp7  = crphot(7)  ! OH
      cp8  = crphot(8)  ! H2O
      cp9  = crphot(9)  ! O2 -> O
      cp10 = crphot(10) ! O2 -> O2+
c NB cp11 rate per molecule scales as abco^(-1/2), so we fold in
c the other factor of abco here to avoid numerical difficulties
c when abco is small
      cp11 = crphot(11) * dsqrt(temp) * abH2 * dsqrt(abco) ! CO
c
      if (dust_to_gas_ratio .gt. 0d0) then
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
          phi = G_dust * dsqrt(temp) / (yn * abe)
        endif
#ifdef NO_GRAIN_RECOMB
        h_gr   = 0d0
        cii_gr = 0d0
        oii_gr = 0d0
        heii_gr = 0d0
#else
c
c H+:
c
        if (phi .lt. 1d-6) then
          h_gr = 1.225d-13 * dust_to_gas_ratio
        else
          hgrvar1  = 8.074d-6 * phi**1.378d0
          hgrvar2  = (1d0 + ch34 * phi**ch35)
          h_gr     = 1.225d-13 * dust_to_gas_ratio /
     $               (1d0 + hgrvar1 * hgrvar2)
        endif
c
c He+
c
        if (phi .lt. 1d-6) then
          heii_gr = 5.572d-14 * dust_to_gas_ratio
        else
          hegrvar1  = 3.185d-7 * phi**1.512d0
          hegrvar2  = (1d0 + ch135 * phi**ch136)
          heii_gr   = 5.572d-14 * dust_to_gas_ratio /
     $               (1d0 + hegrvar1 * hegrvar2)
        endif
c
c O+
c
        oii_gr = h_gr / 4d0
c
c C+:
c
        if (phi .lt. 1d-6) then
          cii_gr = 4.558d-13 * dust_to_gas_ratio
        else
          c2grvar1  = 6.089d-3 * phi**1.128d0
          c2grvar2  = (1d0 + ch36 * phi**ch37)
          cii_gr    = 4.558d-13 * dust_to_gas_ratio /
     $                (1d0 + c2grvar1 * c2grvar2)
        endif
#endif
c
        call calc_dust_temp(yn, chi_isrf, temp, abHI, RH2, Rdust, 0)
c
      else
        RH2   = 0d0
        Rdust = 0d0
        tdust = min(temp, 1.5d3)  ! sublimation cap, consistent w/ cool_util.F (2026-07-05)
        h_gr   = 0d0
        heii_gr = 0d0
        cii_gr = 0d0
        oii_gr = 0d0
      endif
c     
 101  continue
c
c Estimate whether we're optically thick to ionizing radiation;
c if we're not, then we need to set ch18 to zero.
c
      if (2d0 * NH * abhI .lt. 1.6d17) then
        ch18 = 0d0
      endif
c
c Branching ratios for H3O+ DR:
c
c From UMIST06, based on JBS00
c
      ch71a = 0.013d0 * ch71       ! O   + H2 + H
      ch71b = 0.597d0 * ch71       ! OH  + H  + H
      ch71c = 0.14d0 * ch71       ! OH  + H2
      ch71d = 0.25d0 * ch71       ! H2O + H
c
c Branching ratios for H2O+ DR:
c
c From UMIST06, based on RDS00
c
      ch73a = 0.71d0 * ch73       ! O  + H + H
      ch73b = 0.09d0 * ch73       ! O  + H2
      ch73c = 0.20d0 * ch73       ! OH + H
c
      UNUSED_PARAM(ch73a)
c
c Branching ratios for HCO+ dissociative recombination
c
c From GEP05. We have adjusted their quoted value for C + OH
c slightly to ensure that the values sum to 1.0, but this 
c adjustment is within the errors of their measured value
c
      ch87a = 0.92d0 * ch87       ! CO + H
      ch87b = 0.08d0 * ch87       ! OH + C
c
c Branching ratios for CO formation from CH2
c
c cst64a is from UMIST99 and is a fit to data from FJ84 and
c FBJ88. cst64b is from F86.
c
      cst64a = 0.376d0 * cst(64)  ! CO + H2
      cst64b = 0.624d0 * cst(64)  ! CO + H + H
      UNUSED_PARAM(cst64b)
c
c Equilibrium species
c
c Since the equilibrium abundances are coupled, we have to iterate to find 
c accpetable solutions. We iterate niter times (or until all abundances have
c converged).
c
      abhm_old  = 0d0
      abh2p_old = 0d0
      abh3p_old = 0d0
      abchp_old = 0d0
      abch2p_old = 0d0
      abohp_old  = 0d0
      abh2op_old = 0d0
      abh3op_old = 0d0
      abcop_old  = 0d0
      abhocp_old  = 0d0
      abom_old    = 0d0
      abcm_old    = 0d0
      abo2p_old   = 0d0
c
      abhm  = 0d0
      abh2p = 0d0
      abh3p = 0d0
      abchp = 0d0
      abch2p = 0d0
      abohp  = 0d0
      abh2op = 0d0
      abh3op = 0d0
      abcop  = 0d0
      abhocp  = 0d0
      abom    = 0d0
      abcm    = 0d0
      abo2p   = 0d0
#ifdef COLLECT_EQB_DATA
      do I = 1, num_eqb_species
        time_to_equilibrium(I) = 0d0
      enddo
#endif
c
      do I = 1, niter
c
c H- :
c
        num   = ch21 * abHI * abe + (cr12 * abh2 / yn)
        denom = ch22 * abHI + ch26 * abe + ch27 * abHI + ch28 * abhp
     $        + ch29 * abhp + (ph4 / yn) + ch8 * abhep + cst(74) * aboh 
     $        + cst(76) * abcI
c
        abhm = comp_equilib(num, denom, 1d0, 1, yn, time_to_equilibrium)
c
        abe  = max(0d0, abe  - (abhm - abhm_old))
        abHI = max(0d0, abHI - (abhm - abhm_old))
c
c H2+ :
c
        num   = ch23 * abHI * abhp + ch24 * abh2 * abhp
     $        + ch29 * abhp * abhm + (cr2 * abh2 / yn)
     $        + ch107 * abHI * abh3p + (ph37 * abh3p / yn)
     $        + (ph33a * abh2op / yn) + (ph34b * abh3op / yn) 
     $        + cst(75) * abhep * abh2 + (ph41 * abh2 / yn)

        denom = cst(1) * abHI  + ch30 * abe  + (ph5 / yn)
     $        + cst(13) * aboI + cst(77) * abcI
     $        + ch105  * abh2
c
        abh2p = comp_equilib(num, denom, 0.5d0, 2, yn, 
     $                       time_to_equilibrium)
c
        abe  = max(0d0, abe  + (abh2p - abh2p_old))
        abHI = max(0d0, abHI - 2d0*(abh2p - abh2p_old))
c
c H3+:
c
        num   = ch105 * abh2 * abh2p + cst(51) * abh2 * abhp
        denom = ch106 * abe + ch107 * abHI + ((ph36 + ph37) / yn)
     $        + cst(14) * aboI + cst(55) * abcI
     $        + cst(56) * aboh + cst(57) * abh2o
     $        + cst(58) * abco + cst(59) * abco
c
        abh3p = comp_equilib(num, denom, (1d0/3d0), 3, yn,
     $                       time_to_equilibrium)
c
        abe  = max(0d0, abe  + (abh3p - abh3p_old))
        abHI = max(0d0, abHI - 3d0*(abh3p - abh3p_old))
c
        if (abundc .eq. 0d0) then
          abchp  = 0d0
          abch2p = 0d0
          abcm   = 0d0
        else
c
c CH+
c
          num = ch104 * abcp   * abHI + ch60 * abcp * abH2
     $        +  ch62 * abch2p * abHI + cst(42) * abhcop * abcI
     $        + (ph15 / yn) * abch2p + (ph16b / yn) * abch3p
     $        + (ph19 / yn) * abch + cst(55) * abcI * abh3p
     $        + cst(62) * abhp * abch2 + cst(65) * abch * abhp
     $        + cst(77) * abh2p * abcI
          denom = cst(7)  * abHI + cst(8) * abH2 + ch61 * abe
     $          + cst(41) * aboI + (ph14 / yn) + (cp3 / yn)
c
          abchp = comp_equilib(num, denom, abundc, 4, yn,
     $                         time_to_equilibrium)
c
          abe  = max(0d0, abe  + (abchp - abchp_old))
          abHI = max(0d0, abHI - (abchp - abchp_old))
          abcI = max(0d0, abcI - (abchp - abchp_old))
c
c CH2+
c
          num = ch59 * abcp * abH2 + cst(8) * abchp * abH2
     $        + ch66 * abch3p * abHI + (ph16a / yn) * abch3p
     $        + (ph21 / yn) * abch2 + cst(66) * abch2 * abhp
     $        + (cp4 / yn) * abch2
          denom = cst(9) * abH2 + ch62 * abHI + ch63 * abe
     $          + ch64 * abe + ch65 * abe + (ph15 / yn)
     $          + cst(53) * aboI + cst(67) * abo2
c
          abch2p = comp_equilib(num, denom, abundc, 5, yn, 
     $                          time_to_equilibrium)
c
          abe  = max(0d0, abe  + (abch2p - abch2p_old))
          abHI = max(0d0, abHI - 2d0 * (abch2p - abch2p_old))
          abcI = max(0d0, abcI - (abch2p - abch2p_old))
c
c C-
c
          num = cst(47) * abcI * abe
          denom = cst(48) * aboI + ch88 * abhp + cst(49) * abHI
     $          + (ph29 / yn) + cst(61) * abH2
c
          abcm = comp_equilib(num, denom, abundc, 6, yn,
     $                        time_to_equilibrium)
c
          abe  = max(0d0, abe - (abcm - abcm_old))
          abcI = max(0d0, abcI - (abcm - abcm_old))
        endif
c
        if (abundo .eq. 0d0) then
          abohp  = 0d0
          abh2op = 0d0
          abh3op = 0d0
          abom   = 0d0
          abo2p  = 0d0
        else
c
c OH+
c
          num = cst(13) * aboI * abh2p + cst(14) * aboI * abh3p
     $        + cst(15) * abop  * abh2 + cst(25)  * aboh  * abhp
     $        + cst(29) * abh2o * abhep + (ph23 / yn) * aboh
     $        + (ph33d / yn) * abh2op + (ph34d / yn) * abh3op
          denom = cst(16) * abh2 + ch72 * abe + (ph22 / yn)
c
          abohp = comp_equilib(num, denom, abundo, 7, yn,
     $                         time_to_equilibrium)
c
          abe  = max(0d0, abe  + (abohp - abohp_old))
          abHI = max(0d0, abHI - (abohp - abohp_old))
          aboI = max(0d0, aboI - (abohp - abohp_old))
c
c H2O+
c
          num = cst(16) * abohp * abh2 + cst(27) * abh2o * abhp
     $        + cst(30) * abh2o * abhep + (ph24 / yn) * abh2o
     $        + (ph34c / yn) * abh3op + cst(56) * aboh * abh3p
          denom = cst(17) * abh2 + ch73 * abe +  (ph33 / yn)
c
          abh2op = comp_equilib(num, denom, abundo, 8, yn,
     $                          time_to_equilibrium)
c
          abe  = max(0d0, abe  + (abh2op - abh2op_old))
          abHI = max(0d0, abHI - 2d0 * (abh2op - abh2op_old))
          aboI = max(0d0, aboI - (abh2op - abh2op_old))
c
c H3O+
c
          num = cst(17) * abh2op * abh2 + cst(44) * abhcop * abh2o
     $        + cst(57) * abh3p * abh2o
          denom = ch71 * abe + (ph34 / yn)
     $          + cst(78) * abcI
c
          abh3op = comp_equilib(num, denom, abundo, 9, yn,
     $                          time_to_equilibrium)
c
          abe  = max(0d0, abe  + (abh3op - abh3op_old))
          abHI = max(0d0, abHI - 3d0 * (abh3op - abh3op_old))
          aboI = max(0d0, aboI - (abh3op - abh3op_old))
c
c O-
c
          num = cst(23) * aboI * abe
          denom = cst(24) * abhI + cst(50) * abcI + (ph30 / yn) +
     $            cst(63) * abh2 + ch88 * abhp
c
          abom = comp_equilib(num, denom, abundo, 10, yn, 
     $                        time_to_equilibrium)
c
          abe  = max(0d0, abe - (abom - abom_old))
          aboI = max(0d0, aboI - (abom - abom_old))
c
c O2+
c
          num = cst(35) * abo2 * abhp + cst(36) * abo2 * abhep
     $        + (ph28 / yn) * abo2 + (cp10 / yn) * abo2
          denom = cst(38)  * abcI  + cst(39)  * abcI + ch84 * abe
c
          abo2p = comp_equilib(num, denom, 0.5d0*abundo, 11, yn,
     $                         time_to_equilibrium)
c
          abe  = max(0d0, abe + (abo2p - abo2p_old))
          aboI = max(0d0, aboI - 2d0 * (abo2p - abo2p_old))
        endif
c
        if (abundc .eq. 0d0 .or. abundo .eq. 0d0) then
          abcop  = 0d0
          abhocp = 0d0
        else
c
c CO+
c
          num = cst(3) * abcp * aboh + cst(31) * abc2 * abop
     $        + cst(33) * abo2 * abcp + cst(38) * abo2p * abcI
     $        + cst(41) * abchp * aboI + ch112 * abcI * abop
     $        * ab3M * yn + ch112 * abcp * aboI * ab3M * yn 
     $        + ch124 * abcp * aboI + (cr9 / yn) * abco
          denom = cst(40) * abhI + ch119 * abe + cst(80) * abh2
c
          abcop = comp_equilib(num, denom, min(abundc, abundo), 12, yn,
     $                         time_to_equilibrium)
c
          abe  = max(0d0, abe  + (abcop - abcop_old))
          abcI = max(0d0, abcI - (abcop - abcop_old))
          aboI = max(0d0, aboI - (abcop - abcop_old))
c
c HOC+
c
          num   = cst(5) * abcp * abh2o + cst(58) * abh3p * abco
          denom = cst(45) * abco + cst(46) * abh2 + ch56 * abe
c
          abhocp = comp_equilib(num, denom, min(abundc, abundo), 13, yn,
     $                          time_to_equilibrium)
c
          abe  = max(0d0, abe  + (abhocp - abhocp_old))
          abHI = max(0d0, abHI - (abhocp - abhocp_old))
          abcI = max(0d0, abcI - (abhocp - abhocp_old))
          aboI = max(0d0, aboI - (abhocp - abhocp_old))
        endif
c
c Check for convergence:
c
        if (I .ge. 2) then
          if ((abs(abhm_old - abhm)   .lt. eqtol * abhm)  .and.
     $        (abs(abh2p_old - abh2p) .lt. eqtol * abh2p) .and.
     $        (abs(abh3p_old - abh3p) .lt. eqtol * abh3p) .and.
     $        (abs(abchp_old - abchp) .lt. eqtol * abchp) .and.
     $        (abs(abch2p_old - abch2p) .lt. eqtol * abch2p) .and.
     $        (abs(abohp_old - abohp)   .lt. eqtol * abohp)  .and.
     $        (abs(abh2op_old - abh2op) .lt. eqtol * abh2op) .and.
     $        (abs(abh3op_old - abh3op) .lt. eqtol * abh3op) .and.
     $        (abs(abcop_old  - abcop)  .lt. eqtol * abcop)  .and.
     $        (abs(abhocp_old - abhocp) .lt. eqtol * abhocp) .and.
     $        (abs(abom_old - abom)  .lt. eqtol * abom) .and.
     $        (abs(abcm_old - abcm)  .lt. eqtol * abcm) .and.
     $        (abs(abo2p_old - abo2p) .lt. eqtol * abo2p)) then
            goto 100
          endif
        endif
        abhm_old  = abhm
        abh2p_old = abh2p
        abh3p_old = abh3p
        abchp_old = abchp
        abch2p_old = abch2p
        abohp_old  = abohp
        abh2op_old = abh2op
        abh3op_old = abh3op
        abcop_old  = abcop
        abhocp_old  = abhocp
        abom_old    = abom
        abcm_old    = abcm
        abo2p_old   = abo2p
      enddo
c
100   continue
c
c Protons
c 
      ydot(ihp) = (ch11 * abe   * abHI          ! Collisional ion: H  + e-
     $         + cst(1) * abh2p * abHI          ! Charge transfer: H  + H2+
     $           - ch13 * abhp  * abe           ! Gas-phase H recomb.(case B)
     $           - ch23 * abHI  * abhp          ! H2+ formation:   H  + H+
     $           - ch24 * abh2  * abhp          ! Charge transfer: H2 + H+
     $           - ch28 * abhp  * abhm          ! Neutralization:  H- + H+
     $           - ch29 * abhp  * abhm          ! H2+ formation:   H- + H+
     $           - h_gr * abhp  * 1d0           ! Grain H recomb.
     $           + ch46 * abop * abHI           ! Charge transfer: H  + O+
     $           - ch47 * aboI  * abhp          ! Charge transfer: H+ + O
     $           - ch48 * abcI  * abhp          ! Charge transfer: H+ + C
     $        - cst(25) * aboh  * abhp          ! Charge transfer: OH + H+
     $        - cst(27) * abh2o * abhp          ! Charge transfer: H2O + H+
     $        + cst(28) * abh2o * abhep         ! Dissoc. CT: H2O + He+
     $        - cst(35) * abo2  * abhp          ! Charge transfer: O2  + H+
     $        + cst(40) * abcop * abhI          ! Charge transfer: CO+ + H
     $        - cst(51) * abh2  * abhp          ! RA: H2 + H+ -> H3+
     $        - cst(62) * abch2 * abhp          ! Exchge: CH2 + H+ -> CH+ + H2
     $        - cst(65) * abch  * abhp          ! CT with CH:  CH  + H+ -> CH+  + H
     $        - cst(66) * abch2 * abhp          ! CT with CH2: CH2 + H+ -> CH2+ + H
     $           - ch88 * abcm  * abhp          ! Mutual neut.:     C- + H+
     $           + ch58 * abcp  * abHI          ! Charge transfer:  C+ + H
     $           + ch18 * abhep * abe           ! Ion. from He+ rec: He+ + e
     $           + ch32 * abhep * abHI          ! Charge transfer: He+ + H
     $           - ch33 * abheI * abhp          ! Charge transfer: He  + H+
     $           - ch88 * abom  * abhp          ! Neutralization of O-
     $           + ch9  * abhep * abh2) * yn    ! Dissoc. CT of H2 with He+
     $           + ph5  * abh2p                 ! Photodissoc:    H2+ + PH
     $           + cr1  * abHI                  ! Cosmic ray ion.:  H + CR
     $           + cr10 * abh2                  ! Cosmic ray ion.:  H2 + CR
     $           + cr12 * abh2                  ! Cosmic ray ion.:  H2 + CR
     $           + ph14 * abchp                 ! Photodissoc:    CH+ + PH 
     $           + ph22 * abohp                 ! Photodissoc:    OH+ + PH
     $          + ph33b * abh2op                ! Photodissoc:   H2O+ + PH
     $          + ph34a * abh3op                ! Photodissoc:   H3O+ + PH
     $          + ph36  * abh3p                 ! Photodissoc:   H3+ + PH
     $          + ph38  * abHI                  ! Photoionization
c
c Molecular hydrogen
c
      ydot(ih2) = (ch108  * abHI**3           ! 3-body H2 form: 3H -> H2 + H 
     $           + ch109  * abHI**2 * abh2    ! "  : 2H + H2 -> H2 + H2
     $           + ch110  * abHI**2 * abheI) * ynsq ! "  : 2H + He -> H2 + He 
     $          + (ch22   * abHI    * abhm    ! H2 formation:        H- + H
     $          + cst(1)  * abHI    * abh2p   ! Charge transfer:     H  + H2+
     $           -  ch24  * abhp    * abh2    ! Charge transfer:     H2 + H+
     $           -  ch25  * abe     * abh2    ! Collisional dissoc.: H2 + e- 
     $           -  ykdh  * abHI    * abh2    ! Collisional dissoc.: H2 + H 
     $           -  ykdh2 * abh2    * abh2    ! Collisional dissoc.: H2 + H2
     $           +  Rdust * abHI    * 1d0     ! H2 formation on dust
     $           - ch52   * aboI    * abh2    ! OH formation:         O + H2
     $           + ch53   * aboh    * abhI    ! OH destruction:       H + OH 
     $           - ch54   * aboh    * abh2    ! H2O formation:       H2 + OH 
     $           + ch55   * abh2o   * abhI    ! H2O destruction:      H + H2O
     $           - ch59   * abcp    * abh2    ! Rad. association:    C+ + H2
     $           - ch60   * abcp    * abh2    ! CH+ formation:       C+ + H2
     $           + ch62   * abch2p  * abhI    ! CH2+ destruction:  CH2+ + H
     $           + ch65   * abch2p  * abe     ! Dissoc. rec:       CH2+ + e
     $           + ch66   * abch3p  * abHI    ! CH3+ destruction:  CH3+ + H
     $           + ch68   * abch3p  * abe     ! Dissoc. rec:       CH3+ + e
     $           - ch70   * abch    * abh2    ! CH2 formation:       CH + H2 
     $           + ch71a  * abh3op  * abe     ! Dissoc. rec:       H3O+ + e
     $           + ch71c  * abh3op  * abe     ! Dissoc. rec:       H3O+ + e
     $           + ch73b  * abh2op  * abe     ! Dissoc. rec:       H2O+ + e
     $           - ch78   * abcI    * abh2    ! CH formation:         C + H2
     $           - ch86   * abo2    * abh2    ! O2 destruction:      O2 + H2
     $           - ch105  * abh2p   * abh2    ! H3+ formation:      H2+ + H2
     $           + ch106a * abh3p   * abe     ! H3+ DR:             H3+ + e
     $           + ch107  * abh3p   * abHI    ! H3+ destruction:    H3+ + H
     $           + cst(7) * abchp   * abHI    ! CH+  destrcution:   CH+ + H
     $           - cst(8) * abchp   * abH2    ! CH2+ formation:     CH+ + H2
     $           - cst(9) * abch2p  * abH2    ! CH3+ formation:    CH2+ + H2
     $           + ch80   * abch    * abHI    ! CH  destruction:     CH + H 
     $           + cst(11) * abch2  * abHI    ! CH2 destruction:    CH2 + H 
     $           + cst(14) * aboI   * abh3p   ! OH+ formation:       O  + H3+
     $           - cst(15) * abop   * abh2    ! OH+  formation:      O+ + H2
     $           - cst(16) * abohp  * abh2    ! H2O+ formation:     OH+ + H2
     $           - cst(17) * abh2op * abh2    ! H3O+ formation:    H2O+ + H2
     $           - cst(51) * abh2   * abhp    ! RA: H2 + H+ -> H3+
     $           - cst(54) * abcI   * abh2    ! CH2 form:  C + H2 -> CH2 + gamma
     $           + cst(55) * abcI   * abh3p   ! CH+ form: C + H3+ -> CH+ + H2
     $           + cst(56) * aboh   * abh3p   ! H2O+ form: OH + H3+ -> H2O+ + H2
     $           + cst(57) * abh2o  * abh3p   ! H3O+ form: H2O + H3+ -> H3O+ + H2
     $           + cst(58) * abco   * abh3p   ! HOC+ form: CO + H3+ -> HOC+ + H2
     $           + cst(59) * abco   * abh3p   ! HCO+ form: CO + H3+ -> HCO+ + H2   
     $           - cst(61) * abcm   * abH2    ! CH2  form: C- + H2  -> CH2  + e
     $           - cst(63) * abom   * abH2    ! H2O  form: O- + H2  -> H2O + e
     $           + cst64a  * abch2  * aboI    ! CO form:   CH2 + O -> CO + H2
     $           + cst(68) * abhep  * abch2   ! CH2 dest.: CH2 + He+ -> C+ + He + H2
     $           + cst(69) * abch3p * aboI    ! CH3+ dest: CH3+ + O -> HCO+ + H2
     $           + cst(62) * abhp   * abch2   ! CH2 dest.: CH2 + H+ -> CH+ + H2
     $           + cst(78) * abh3op * abcI    ! H3O+ + C -> HCO+ + H2
     $           - cst(75) * abhep  * abh2    ! CT with He+
     $           - cst(80) * abcop  * abh2    ! CO+ + H2 -> HCO+ + H
     $           - ch9     * abhep  * abh2) * yn ! Dissoc. CT with He+
     $           -  ykdis  * abh2             ! Photodissoc:         H2 + PH
     $           -  cr2    * abh2             ! Cosmic ray ion.:     H2 + CR
     $           -  cr10   * abh2             ! Cosmic ray ion.:     H2 + CR
     $           -  cr11   * abh2             ! Cosmic ray ion.:     H2 + CR
     $           -  cr12   * abh2             ! Cosmic ray ion.:     H2 + CR
     $           +  ph33c  * abh2op           ! Photodissoc:       H2O+ + PH
     $           +  ph34d  * abh3op           ! Photodissoc:       H3O+ + PH
     $           +  ph16b  * abch3p           ! Photodissoc:       CH3+ + PH 
     $           +  ph36   * abh3p            ! Photodissoc:        H3+ + PH
     $           -  ph41   * abh2             ! Photoionization: H2 -> H2+
c
c Singly-ionized helium (He+)
c
      if (abhe .eq. 0d0) then
        ydot(ihep) = 0d0
      else
        ydot(ihep) = (-ch8    * abhm  * abhep
     $              - ch9     * abh2  * abhep
     $              + ch14    * abheI * abe
     $              - heii_gr * abhep * 1d0
     $              - ch17    * abhep * abe
     $              - ch32    * abhep * abHI
     $              + ch33    * abheI * abhp
     $              - ch92    * abco  * abhep
     $              - ch93    * abco  * abhep
     $              - ch94    * abcI  * abhep
     $              - ch97    * aboI  * abhep
     $              - cst(26) * aboh  * abhep
     $              - cst(28) * abh2o * abhep
     $              - cst(29) * abh2o * abhep
     $              - cst(30) * abh2o * abhep
     $              - cst(32) * abc2  * abhep
     $              - cst(36) * abo2  * abhep
     $              - cst(37) * abo2  * abhep
     $              - cst(75) * abh2  * abhep
     $              - cst(68) * abhep * abch2) * yn
     $              + cr6     * abheI
     $              + ph39    * abheI
      endif
c
c Ionized carbon (C+)
c
      ydot(ic) =(-ch104    * abcp  * abHI  
     $           - ch59    * abcp  * abH2
     $           - ch60    * abcp  * abH2
     $           + cst(7)  * abchp * abHI
     $           - ch58    * abcp  * abHI
     $           - ch40    * abcp  * abe
     $           + ch43    * abcI  * abe
     $           + ch48    * abcI  * abhp
     $           - cii_gr  * abcp  * 1d0
     $           - cst(3)  * abcp  * aboh
     $           - cst(4)  * abcp  * abh2o
     $           - cst(5)  * abcp  * abh2o
     $           + cst(32) * abc2  * abhep
     $           - cst(33) * abo2  * abcp
     $           - cst(34) * abo2  * abcp
     $           + cst(39) * abo2p * abcI
     $           + cst(68) * abhep * abch2
     $           + ch92    * abco  * abhep
     $           + ch94    * abcI  * abhep
     $           - ch124   * abcp  * aboI) * yn
     $           - ch112   * abcp  * aboI   * ab3M * ynsq
     $           + ph2     * abcI
     $           + ph14    * abchp
     $           + cr3     * abcI
     $           + cp1     * abcI
     $           + cp3     * abchp
c
c O+
c
      ydot(io) = (ch45    * aboI * abe
     $          + ch47    * aboI * abhp
     $          - ch46    * abop * abHI
     $          - ch42    * abop * abe
     $          - oii_gr  * abop * 1d0
     $          - cst(15) * abop * abh2
     $          + cst(26) * aboh * abhep
     $          - cst(31) * abc2 * abop
     $          + cst(34) * abo2 * abcp
     $          + cst(37) * abo2 * abhep
     $          + ch93    * abco * abhep
     $          + ch97    * aboI * abhep) * yn
     $          - ch112   * abop * abcI   * ab3M * ynsq
     $          + cr4     * aboI
     $          + ph33c   * abh2op
     $          + ph46    * aboI
c
c OH
c
      ydot(ioh) = (ch71b   * abh3op * abe
     $           + ch71c   * abh3op * abe
     $           + ch73c   * abh2op * abe
     $           + cst(22) * aboI   * abhm
     $           + cst(24) * abom   * abhI
     $           + ch50    * aboI   * abhI
     $           - ch51    * aboh   * abhI
     $           + ch52    * aboI   * abh2
     $           - ch53    * aboh   * abhI
     $           - ch54    * aboh   * abh2
     $           + ch55    * abh2o  * abhI
     $           - cst(71) * abcI   * aboh
     $           - cst(3)  * abcp   * aboh
     $           - cst(25) * aboh   * abhp
     $           - cst(26) * aboh   * abhep
     $           + cst(28) * abh2o  * abhep
     $           - ch81    * aboI   * aboh
     $           + ch85    * abo2   * abhI
     $           + 2d0*ch86 * abo2  * abh2
     $           + ch87b   * abhcop * abe
     $           + ch89    * abco   * abhI
     $           - cst(56) * aboh   * abh3p
     $           + cst(67) * abch2p * abo2
     $           - cst(74) * abhm   * aboh
     $           - 2d0*ch118 * aboh * aboh
     $           - ch131   * aboh   * abHI
     $           - ch132   * aboh   * abH2
     $           + ch133   * abh2o  * abHI
     $           + ch134   * abh2o  * abH2) * yn
     $           + ch113   * aboI   * abhI  * ab3M * ynsq
     $           - ch114   * aboh   * abhI  * ab3M * ynsq
     $           - ph11    * aboh
     $           - ph23    * aboh
     $           + ph12    * abh2o
     $           + ph33b   * abh2op
     $           + ph34b   * abh3op
     $           - cp7     * aboh
     $           + cp8     * abh2o
c
c H2O
c
      ydot(ih2o) = (ch71d  * abh3op * abe
     $           - cst(4)  * abcp   * abh2o
     $           - cst(5)  * abcp   * abh2o
     $           + ch51    * aboh   * abhI
     $           + ch54    * aboh   * abh2
     $           - ch55    * abh2o  * abhI
     $           - cst(27) * abh2o  * abhp
     $           - cst(28) * abh2o  * abhep
     $           - cst(29) * abh2o  * abhep
     $           - cst(30) * abh2o  * abhep
     $           - cst(44) * abhcop * abh2o
     $           - cst(57) * abh3p  * abh2o
     $           + cst(63) * abom   * abH2
     $           + cst(74) * abhm   * aboh
     $           + ch118   * aboh   * aboh
     $           - ch133   * abh2o  * abHI
     $           - ch134   * abh2o  * abH2) * yn
     $           + ch114   * aboh   * abhI   * ab3M * ynsq
     $           - ph12    * abh2o
     $           - ph24    * abh2o
     $           + ph34a   * abh3op
     $           - cp8     * abh2o
c
c CO
c
      ydot(ico) = ( ch57   * abcI   * aboI
     $           + cst(71) * abcI   * aboh
     $           + ch77    * abch   * aboI
     $           + cst(34) * abo2   * abcp
     $           + ch83    * abo2   * abcI
     $           + cst(40) * abcop  * abhI
     $           + cst(42) * abhcop * abcI
     $           + cst(44) * abhcop * abh2o
     $           + ch87a   * abhcop * abe
     $           + ch56    * abhocp * abe
     $           + cst(48) * abcm   * aboI
     $           + cst(50) * abom   * abcI
     $           - cst(58) * abh3p  * abco
     $           - cst(59) * abh3p  * abco
     $           + cst(64) * abch2  * aboI
     $           - ch89    * abco   * abhI
     $           + ch90    * abc2   * aboI
     $           - ch92    * abco   * abhep
     $           - ch93    * abco   * abhep) * yn
     $           + ch111   * abcI   * aboI   * ab3M * ynsq 
     $           - ph10    * abco
     $           - cr9     * abco
     $           - cp11  ! NB No factor of abco here -- see above
c
c C2
c
      ydot(ic2) = (ch79    * abcI * abcI
     $           + cst(10) * abcI * abch
     $           - cst(31) * abc2 * abop
     $           - cst(32) * abc2 * abhep
     $           - ch90    * abc2 * aboI) * yn
     $           + ch115   * abcI * abcI  * ab3M * ynsq
     $           - ph25    * abc2
     $           -  cp6    * abc2
c
c O2
c
      ydot(io2) = (ch81    * aboI   * aboh
     $           + ch82    * aboI   * aboI
     $           - cst(33) * abo2   * abcp
     $           - cst(34) * abo2   * abcp
     $           - ch83    * abo2   * abcI
     $           - cst(35) * abo2   * abhp
     $           - cst(36) * abo2   * abhep
     $           - cst(37) * abo2   * abhep
     $           + cst(39) * abo2p  * abcI
     $           - cst(67) * abo2   * abch2p
     $           - ch85    * abo2   * abhI
     $           - ch86    * abo2   * abh2) * yn
     $           + ch116   * aboI  * aboI  * ab3M * ynsq
     $           - ph27    * abo2
     $           - ph28    * abo2
     $           - cp9     * abo2
     $           - cp10    * abo2
c
c HCO+
c
      ydot(ihcop) = (cst(4)  * abcp   * abh2o 
     $            +  cst(45) * abhocp * abco 
     $            +  cst(46) * abhocp * abh2 
     $            +  ch117   * abch   * aboI
     $            +  cst(53) * abch2p * aboI 
     $            +  cst(59) * abh3p  * abco
     $            -  cst(42) * abcI   * abhcop
     $            -  cst(44) * abh2o  * abhcop
     $            +  cst(67) * abch2p * abo2
     $            +  cst(69) * abch3p * aboI
     $            +  cst(78) * abh3op * abcI
     $            +  cst(80) * abcop  * abh2
     $            -  ch87    * abe    * abhcop) * yn
c
c CH
c
      ydot(ich) = (ch63    * abch2p * abe
     $          +  ch68    * abch3p * abe
     $          +  ch69    * abch3p * abe
     $          +  cst(11) * abch2  * abHI
     $          +  cst(12) * abcI   * abHI 
     $          +  ch78    * abcI   * abH2
     $          +  cst(49) * abcm   * abhI 
     $          -  ch70    * abH2   * abch
     $          -  ch80    * abHI   * abch 
     $          -  ch77    * aboI   * abch 
     $          -  cst(10) * abcI   * abch 
     $          -  ch117   * aboI   * abch
     $          -  cst(65) * abhp   * abch
     $          +  cst(76) * abhm   * abcI) * yn
     $          - ph18     * abch 
     $          - ph19     * abch
     $          + ph20     * abch2
     $          -  cp2     * abch
     $          +  cp5     * abch2
c
c CH2
c
      ydot(ich2) = (ch67    * abch3p * abe 
     $           +  ch70    * abch   * abH2
     $           +  cst(54) * abcI   * abH2 
     $           +  cst(61) * abcm   * abH2
     $           -  cst(11) * abHI   * abch2
     $           -  cst(62) * abhp   * abch2 
     $           -  cst(64) * aboI   * abch2
     $           -  cst(66) * abhp   * abch2
     $           -  cst(68) * abhep  * abch2) * yn
     $           -  ph20    * abch2 
     $           -  ph21    * abch2
     $           -   cp4    * abch2
     $           -   cp5    * abch2
c
c CH3+
c
      ydot(ich3p) = (cst(9)  * abch2p * abH2
     $            -  ch66    * abHI   * abch3p
     $            -  ch67    * abe    * abch3p
     $            -  ch68    * abe    * abch3p
     $            -  ch69    * abe    * abch3p
     $            -  cst(69) * aboI   * abch3p) * yn
     $            -  ph16 * abch3p 
c
#if CHEMISTRYNETWORK == 11
c
c CH4 -- gas phase processes only here, surface chemistry handled below
c
      ydot(ich4) = 0d0 ! ***Temporary kludge***
c
c Total fractional abundance of ice 
c
      ice_fraction = y(icos) + y(ih2os) + y(io2s) + y(ich4s)
c
c Number of surface sites per cm^-3
c 
      nsurf = 8d-6
c
c Hence, number of monolayers of ice (given assumed surface area
c of dust per unit H atom == 8e-21 cm^2, and assumed number density
c of surface sites of 1e15 cm^-2)
c
      nlayer = int(ice_fraction / nsurf)
c
c Fraction of surface layer that is full
c
      surface_fraction = ice_fraction / nsurf - nlayer
c
c Fraction of surface sites occupied by each ice species
c
      ice_frac_co  = (y(icos)  / ice_fraction) * surface_fraction
      ice_frac_h2o = (y(ih2os) / ice_fraction) * surface_fraction
      ice_frac_o2  = (y(io2s)  / ice_fraction) * surface_fraction
      ice_frac_ch4 = (y(ich4s) / ice_fraction) * surface_fraction
c
c Basic photodesorption rate - modified for CO, H2O, and
c also weighted by surface fractions of species.
c
c Make use of the fact that photochem reaction 22 has same
c dependence on AV as the basic photodesorption rate
c (see Hollenbach et al, 2009 for the latter)

      df18 = phrates(22) / phtab(22)
      kd_pd = 2d-16 * df18
c
c CH4 surface chem
c
c Carbon accretion rate
c
      kd_acc_c = 1.53d0 * ch145 * abcI * yn
c 
c Hydrogen accretion rate
c
      kd_acc_H = 5.29d0 * ch145 * abHI * yn
c
c Methane accretion rate
c
      kd_acc_CH4 = 1.32d0 * ch145 * abch4 * yn
c
c Carbon desorption rate (per surface C atom)
c
      kd_des_c = 1.31d12 * exp(-8d2 / tdust)
c
c CH desorption rate (per surface CH radical)
c
      kd_des_ch = 1.13d12 * exp(-6.5d2 / tdust)
c
c NB. For simplicity, we assume that desorption of CH2, CH3 can be
c     ignored. In conditions where significant carbon is converted
c     to CH4, then this should be a good approximation -- efficient
c     conversion implies that we don't lose too much CH from the 
c     grain, and the binding energy of this is much smaller than that
c     of CH2 or CH3; hence if the CH desorption rate is negligible,
c     the same must be true of the CH2 and CH3 rates. OTOH, if the
c     rate isn't negligible, then there won't be much CH surviving 
c     to be converted to CH2, CH3 or CH4
c
c To determine whether accreted carbon reacts or desorbs, we compare
c against accretion rate of H atoms for an example grain with radius 
c a = 0.1 micron. If accretion rate > desorption rate, then the atom
c reacts; otherwise, it desorbs. [NB. This is a little crude, and the
c choice of grain size is somewhat arbitrary, but the strong dependence
c of the desorption rate on tdust means that in practice we won't be
c too sensitive to a]
c
      kd_acc_H_single =  1.1d-5 * sqrt(temp)
c 
      if (kd_acc_H_single .gt. kd_des_c) then
        c_react = 1d0
      else
        c_react = 0d0
      endif
c
c Now do the same thing for CH
c
      if (kd_acc_H_single .gt. kd_des_ch) then
        ch_react = 1d0
      else
        ch_react = 0d0
      endif
c
c Fraction of carbon converted to CH4 ice
c
      frac_ch4 = c_react * ch_react
c
c Fraction of carbon converted to CH and then desorbed
c
      frac_ch = c_react - frac_ch4
c
c Cosmic ray desorption
c
      kd_ch4_cr = 2.7d-15 * ice_frac_ch4 * nsurf
c
c Thermal desoprtion
c
      kd_ch4_th = 1.33d12 * exp(-1.1d3 / tdust) * ice_frac_ch4 * nsurf
c
c Photodesorption
c
      kd_ch4_ph = kd_pd * ice_frac_ch4
c
      ydot(ich4s) = kd_acc_c * frac_ch4 
     $            + kd_acc_ch4
     $            - kd_ch4_ph
     $            - kd_ch4_cr 
     $            - kd_ch4_th
c
c Update CH, CH4 rates of change to account for C surface chemistry
c 
      ydot(ich) = ydot(ich)
     $          + kd_acc_c * frac_ch
c
      ydot(ich4) = ydot(ich4)
     $           - kd_acc_ch4
     $           + kd_ch4_ph
     $           + kd_ch4_cr 
     $           + kd_ch4_th
c
c CO surface chem:
      kd_acc_co = ch145 * abco * yn ! Accretion
      kd_co_th  = 1.04d12 * dexp(-9.6d2 / tdust) * ice_frac_co * nsurf ! Thermal evaporation
      kd_co_cr  = 6d-13 * ice_frac_co * nsurf ! Cosmic ray desorption - from Hollenbach et al 2009
      kd_co_ph  = kd_pd * ice_frac_co
c
      ydot(icos) = kd_acc_co - kd_co_th - kd_co_cr - kd_co_ph
c
      ydot(ico) = ydot(ico) - ydot(icos)
c
c The procedure for OH and H2O is similar to that for CH4 above
c
c Oxygen accretion rate
c
      kd_acc_o = 1.32d0 * ch145 * aboI * yn
c
c OH accretion rate
c
      kd_acc_oh = 1.28d0 * ch145 * aboh * yn
c
c H2O accretion rate
c
      kd_acc_h2o = 1.25d0 * ch145 * abh2o * yn
c
c Oxygen thermal desorption rate
c
      kd_o_th = 1.1d13 * dexp(-8d2 / tdust)
c
c Fraction of oxygen converted to water ice
c
      if (kd_acc_H_single .gt. kd_o_th) then
        frac_o = 1d0
      else
        frac_o = 0d0
      endif
c
c H2O thermal desorption rate
c
      kd_h2o_th = 2.6d13 * dexp(-4.8d3 / tdust) * ice_frac_h2o * nsurf
c
c H2O cosmic ray desorption rate
c
      kd_h2o_cr = 4.4d-17 * ice_frac_h2o * nsurf
c
c H2O photodesorption rate
c
      kd_h2o_ph = 3d0 * kd_pd * ice_frac_h2o
c
c Water ice:
c
      ydot(ih2os) = kd_acc_o * frac_o 
     $            + kd_acc_h2o
     $            - kd_h2o_ph
     $            - kd_h2o_cr
     $            - kd_h2o_th
c
      ydot(ih2o) = ydot(ih2o)
     $           - kd_acc_h2o
     $           + kd_h2o_cr
     $           + kd_h2o_th
     $           + (1d0 / 3d0) * kd_h2o_ph
c
      y(ioh) = y(ioh) 
     $       + (2d0 / 3d0) * kd_h2o_ph
c
c O2 surface chem:
      kd_acc_o2 = 0.935d0 * ch145  ! Accretion
      kd_o2_th = 9.8d11 * dexp(-1.2d3 / tdust) * ice_frac_o2 * nsurf
      kd_o2_cr = 9.1d-15 * ice_frac_o2 * nsurf
      kd_o2_ph = kd_pd   * ice_frac_o2

c
      ydot(io2s) = kd_acc_o2 - kd_o2_th - kd_o2_cr - kd_o2_ph
      ydot(io2)  = ydot(io2) - ydot(io2s)
c
#endif /* CHEMISTRYNETWORK == 11 */
c
      if (abundc .eq. 0d0) then
        ydot(ic)    = 0d0 
        ydot(ico)   = 0d0 
        ydot(ic2)   = 0d0 
        ydot(ihcop) = 0d0
        ydot(ich)   = 0d0
        ydot(ich2)  = 0d0
        ydot(ich3p) = 0d0
#if CHEMISTRYNETWORK == 11
        ydot(icos)  = 0d0
        ydot(ich4s) = 0d0
#endif
      endif
      if (abundo .eq. 0d0) then
        ydot(io)    = 0d0
        ydot(ioh)   = 0d0
        ydot(ih2o)  = 0d0
        ydot(ico)   = 0d0
        ydot(ihcop) = 0d0
        ydot(io2)   = 0d0
#if CHEMISTRYNETWORK == 11
        ydot(icos)  = 0d0
        ydot(ih2os) = 0d0
        ydot(io2s)  = 0d0
#endif
      endif
c
c Chemical heating & cooling
c
c
c H2 collisional dissociation cooling:
c
      rates_chem(1) = 4.48d0 * eV * (ykdh  * abHI * abh2 + 
     $                               ykdh2 * abh2 * abh2 +
     $                               ch25  * abe  * abh2 ) * ynsq
c
c H2 destruction by charge transfer with HII -- endothermic by 1.83eV 
c
      rates_chem(2) = 1.83d0 * eV * ch24 * abh2 * abhp * ynsq
c
c H2 photodissociation heating -- 0.4eV per photodissociation:
c
      rates_chem(3) = - 4d-1 * eV * abH2 * yn * ykdis
c
c Heating by UV pumping of H2 -- effective for n > n_crit
c
c We assume 2 eV per UV photon (Burton, Hollenbach & Tielens 1990),
c and adopt a pumping rate 8.5 times larger than the dissociation rate
c (Draine & Bertoldi 1996, table 2). 
c
      rates_chem(4) = - 2d0 * eV * abH2 * yn * ykdis * 8.5d0 * 
     $                  (1d0 - h2var0)
c
c H2 formation heating -- we assume a similar depedence on n / n_crit to 
c the collisional dissociation rates, although ideally one would like to
c do a more careful, level-dependent treatment.
c
c Note that energy input depends on formation mechanism:
c
c H-   -- 3.53eV
c H2+  -- 1.83eV
c 3-body -- 4.48eV
c
      if (RH2 .gt. 0d0) then
        call compute_h2_heating_fraction(temp, yn, abh2, f_ex)
        f_ex = f_ex * h2_form_ex
      else
        f_ex = 0d0
      endif
      f_tot = h2_form_kin + f_ex
c
      rates_chem(5) = - 4.48d0 * eV * f_tot * RH2
c
      rates_chem(6) = - 3.53d0 * eV * ch22 * abHI * abhm * ynsq
     $                * (1d0 - h2var0)
c
      rates_chem(7) = - 1.83d0 * eV * cst(1) * abHI * abh2p
     $                * ynsq * (1d0 - h2var0)
c
      rates_chem(8) = - 4.48d0 * eV * abHI * abHI * (ch108 * abHI  
     $                + ch109 * abh2 + ch110 * abheI) * yncb
c
c HI collisional ionization cooling -- 13.6eV per ionization:
c
      rates_chem(9) = 13.6d0 * eV * ch11 * abe * abHI * ynsq
c
c HeI collisional ionization cooling -- 24.6 eV per ionization
c
      rates_chem(10) = 24.6d0 * eV * ch14 * abe * abheI * ynsq
c
c HeII collisional ionization cooling -- 54.4 eV per ionization
c Assumed zero in current model.
c
      rates_chem(11) = 0d0
c
c HII recombination cooling -- we assume that kT ergs is radiated 
c per recombination and that this is independent of whether the
c recombination occurs in the gas phase or on a grain surface
c
      rates_chem(12) = kboltz * temp * abhp * (ch13 * abe + h_gr) * 
     $                 ynsq
c
c HeII recombination cooling. Dielectronic recombination results
c in the loss of ~ 40.8eV per recombination, while for standard 
c radiative recombination, we make the same assumption as for HII
c
c Dielectronic:
      rates_chem(13) = 40.8d0 * eV * ch10 * abhep * abe * ynsq
c
c Radiative:
      rates_chem(14) = kboltz * temp * ((ch17 - ch10) * abe + heii_gr) *
     $                 abhep * ynsq
c
c HeIII recombination cooling -- assumed zero currently
c
      rates_chem(15) = 0d0
c
c CI photoionization heating
c
      rates_chem(16) = - ph6 * abcI * yn
c
c SiI photoionization heating
c
      rates_chem(17) = 0d0
c
c H- photodissociation heating
c
      rates_chem(18) = - ph8 * abhm * yn
c
c H2+ photodissociation heating
c
      rates_chem(19) = - ph9 * abh2p * yn
c
c HI photoionization heating
c 
      rates_chem(20) = - ph42 * abHI * yn
c
c HeI photoionization heating
c 
      rates_chem(21) = - ph43 * abheI * yn
c
c H2 photoionization heating
c
      rates_chem(22) = - ph45 * abh2 * yn
c
      ylam_chem = 0d0
      do I = 1, nrates_chem
        ylam_chem = ylam_chem + rates_chem(I)
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
      abundances(3)  = aboh
      abundances(4)  = abh2o
      abundances(5)  = abco
      abundances(6)  = abcI
      abundances(7)  = abcp
      abundances(8)  = 0d0
      abundances(9)  = 0d0
      abundances(10) = abe
      abundances(11) = abhp
      abundances(12) = abHI
      abundances(13) = 0d0
      abundances(14) = abheI
      abundances(15) = abhep
c No HeIII in current model
      abundances(16) = 0d0
c
#ifdef ADIABATIC_DENSITY_THRESHOLD
      if (yn .ge. yn_adiabatic) then
        ydot(itmp) = 0d0
#ifdef DEBUG_RATE_EQ
        call validate_output(y,ydot,rpar)
#endif
        return
      endif
#endif
c
      energy_dummy = energy
      call cool_func(temp, yn, dl, divv, G_dust, abundances, ylam, 
     $               rates, 0, energy_dummy)
c
c At high z, correct for effects of CMB. (In the local ISM, T_cmb
c is too low for this to be important: other sources of heating
c will dominate at v. low T)
c
      background_temp = 0.0
      if (background_temp .gt. 0d0) then
        call cool_func(background_temp, yn, dl, divv, G_dust, 
     $                 abundances, ylam_back, rates_back, 1, 
     $                 energy_dummy)
      else
        ylam_back = 0d0
      endif
c
      ydot(itmp) = -(ylam - ylam_back + ylam_chem)
c
#ifdef FF_COLLAPSE_TEST
      ndot = yn**1.5d0 * tffc
c
      compress = ((1.1d0 - abh2) * BOLTZMANN * temp) * ndot
      ydot(itmp) = ydot(itmp) + compress + 
     $             (ndot / yn) * y(itmp)
#endif

#ifdef DEBUG_RATE_EQ
      call validate_output(y,ydot,rpar)
#endif

#ifdef FF_COLLAPSE_TEST
      if (itest .gt. 0) then
        tff = 1d0 / (tffc * dsqrt(yn)) 
        do I = 1, num_eqb_species
          print*, I, time_to_equilibrium(I) / tff
        enddo
        itest = 0
      endif
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
c=======================================================================
c
c    \\\\\\\\\\        B E G I N   F U N C T I O N        //////////
c    //////////          C O M P _ E Q U I L I B          \\\\\\\\\\
c
c=======================================================================
c
#ifdef CHEMCOOL
      REAL function comp_equilib(num, denom, max_abundance, id, yn, 
     $                           time_to_equilibrium)
      implicit NONE
#include "cool.h"
#include "non_eq.h"
      REAL num, denom, max_abundance, yn
      REAL time_to_equilibrium(num_eqb_species)
      integer id
c
      if (num .eq. 0d0) then
        comp_equilib = 0d0
      elseif (denom .eq. 0d0) then
c If this happens, then our assumption that this species is in
c equilibrium has fatally broken down...
        print*, "Destruction without creation in comp_equilib"
        print*, "Species = ", id
        print*, num, denom
        ABORT(9751)
      else
#ifdef COLLECT_EQB_DATA
        time_to_equilibrium(id) = 1d0 / (denom * yn)
#endif
        comp_equilib = num / denom
        if (comp_equilib .gt. max_abundance) then
          print*, "Error: impossible abundance!"
          print*, "Species ", id
          print*, num, denom, num/denom
          ABORT(9752)
        endif
      endif
c
      return
      end
#endif /* CHEMCOOL */
c=======================================================================
c
c    \\\\\\\\\\          E N D     F U N C T I O N        //////////
c    //////////          C O M P _ E Q U I L I B          \\\\\\\\\\
c
c=======================================================================
#endif /* CHEMISTRYNETWORK == 7 || CHEMISTRYNETWORK == 11 */
