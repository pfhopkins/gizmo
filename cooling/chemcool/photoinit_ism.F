#include "../../GIZMO_config.h"
c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////             P H O T O I N I T             \\\\\\\\\\
c
c=======================================================================
c
       subroutine photoinit
#ifdef CHEMCOOL
c
c    written by: S. Glover, AMNH, June 2004
c
c    PURPOSE: Precompute and tabulate photochemical rate coefficients.
c
c    The shape of the incident spectrum is assumed to be the same as that
c    of the Draine (1978) field, but the scaling is in terms of the Habing
c    field (so the actual Draine field is obtained for G0 = 1.7). Note 
c    that in many of the referenced papers, rates are given assuming that
c    G0=1.7, and so have been rescaled here.
c
c    REFERENCES:
c
c      D68   -- Dunn, 1968, Phys. Rev., 172, 1
c      DJ72  -- de Jong, 1972, A&A, 20, 263
c      D78   -- Draine, 1978, ApJS, 36, 595
c      MMP83 -- Mathis, Mezger, & Panagia, 1983, A&A, 128, 212
c      L84   -- Lee, 1984, ApJ, 282, 172
c      NL85  -- Nee & Lee, 1985, ApJ, 291, 202 
c      vD87  -- van Dishoeck, 1987, in 'Astrochemistry', IAU Symp. 120, 51
c      vD88  -- van Dishoeck, 1988,in 'Rate Coefficients in Astrochemistry', 49
c      vDB88 -- van Dishoeck & Black, 1988, ApJ, 334, 771
c      RJLD91 -- Roberge, Jones, Lepp & Dalgarno, 1991, ApJS, 77, 287 
c      SD95  -- Sternberg & Dalgarno, 1995, ApJS, 99, 565
c      DB96  -- Draine & Bertoldi, 1996, ApJ, 468, 269
c      S97   -- Stancil et al, 1997, ApJ, 486, 574
c      G03   -- Glover, 2003, ApJ, 584, 331
c
c
      implicit NONE
#include "cool.h"
c
c (ph1) -- H2 photodissociation
c
c We assume an ortho-para ratio of 3:1, with no significant population
c of higher rotational or vibrational states. In this case, the optically
c thin photodissociation rate is simply (Draine & Bertoldi 1996, table 2):
c
      phtab(1) = 3.3d-11
c
c (ph2) -- CI ionization
c 
c  Rate from UMIST99, based on vD88.
c
      phtab(2) = 1.765d-10
c
c (ph3) -- SiI ionization
c
c  Rate from UMIST99, based on vD88.
c
      phtab(3) = 1.824d-9
c
c (ph4) -- H- photodissociation: 
c
c Rate taken from G03; original calculation assumed MMP83 spectrum
c (for 8 kpc Galactrocentric distance), DJ72 cross-section. 
c 
c N.B. 75% larger than the rate quoted in UMIST99, but the UMIST rate is
c uncertain to within a factor of two, so this is probably OK
c
      phtab(4) = 4.2d-7
c
c (ph5) -- H2+ photodissociation
c
c Rate assumes MMP83 spectrum at 8 kpc, cross-section of D68. 
c This gives a value approx. 15% larger than the UMIST value. 
c
      phtab(5) = 6.5d-10
c
c ph6 -- ph9 --> Heating rates:  XXX - not yet implemented
c
c (ph6) -- CI ionization
c
      phtab(6) = 0d0
c
c (ph7) -- SiI ionization
c
      phtab(7) = 0d0
c
c (ph8) -- H- photodissociation: 
c
      phtab(8) = 0d0
c
c (ph9) -- H2+ photodissociation: 
c
      phtab(9) = 0d0
c
c (ph10) -- CO photodissociation -- from vDB88
c
      phtab(10) = 1.235d-10
c
c (ph11) -- OH photodissociation: OH + gamma -> O + H
c
c From vDD84
c
      phtab(11) = 2.235d-10
c
c (ph12) -- H2O photodissociation: H2O + gamma -> OH + H
c
c From L84
c
      phtab(12) = 3.47d-10

c (phXX) -- H2O photodissociation: H2O + gamma -> O + H2

c
c (ph13) -- currently unused: formerly rate for CH+ + gamma -> C+ + H,
c           (based on UMIST99), but this has now been removed from the
c           network as it appears to be an error in UMIST99; the real
c           products are C + H+ (see below)
c
c (ph14) -- CH+ photodissociation
c  
c UMIST99 gives the products as C+ + H and cites RJLD91 as the source.
c However, RJLD91 actually list products as C + H+ (as does Kirby 1980)
c Moreover, UMIST05 now also lists products as C + H+, implying that
c UMIST99 is simply in error. 
c
      phtab(14) = 1.47d-10
c
c (ph15) -- CH2+ photodissociation: CH+ + H
c
c UMIST06, quoting van Dishoeck et al, 2006, in prep.
c
      phtab(15) = 2.71d-11
c
c (ph16) -- CH3+ photodissociation
c
c Rate and branching ratio from UMIST99:
c 50% of photodissociations -> CH2+ + H, 50% -> CH+ + H2
c
      phtab(16) = 5.88d-10    ! CH2+ + H
c
c (ph17) -- unused
c
c (ph18) -- CH photodissociation: CH + gamma -> C + H
c
c From RJLD91
c
      phtab(18) = 5.06d-10
c
c (ph19) -- CH photoionization: CH + gamma -> CH+ + e-
c
c From vD88
c
      phtab(19) = 4.47d-10
c
c (ph20) -- CH2 photodissociation: CH2 + gamma -> CH + H
c
c From vD88
c
      phtab(20) = 4.235d-10
c
c (ph21) -- CH2 photoionization: CH2 + gamma -> CH2+ + e-
c
c From UMIST99
c
      phtab(21) = 5.88d-10
c
c (ph22) -- OH+ photodissociation:  OH+ + gamma -> O + H+
c
c From vD87
c
      phtab(22) = 5.88d-13
c
c (ph23) -- OH photoionization: OH + gamma -> OH+ + e-
c
c From UMIST99
c
      phtab(23) = 9.41d-13
c
c (ph24) -- H2O photoionization: H2O + gamma -> H2O+ + e-
c
c From vD88
c
      phtab(24) = 1.94d-11
c
c (ph25) -- C2 photodissociation: C2 + gamma -> C + C
c
c From RJLD91
c
      phtab(25) = 8.82d-11
c
c (ph26) -- unused
c
c (ph27) -- O2 photodissociation: O2 + gamma -> O + O
c
c From RJLD91
c
      phtab(27) = 4.06d-10
c
c (ph28) -- O2 photoionization: O2 + gamma -> O2+ + e-
c
c From RJLD91
c
      phtab(28) = 3.29d-11
c
c (ph29) -- C- photodetachment:  C- + gamma -> C + e-
c
c Rate from UMIST99
c
      phtab(29) = 1.41d-7
c
c (ph30) -- O- photodetachment:  O- + gamma -> O + e-
c
c Rate from UMIST99
c
      phtab(30) = 1.41d-7
c
c (ph31-32) -- SiH+ photodissociation, rates from S97.
c
      phtab(31) = 1.21d-9      ! SiH+ + gamma -> Si+ + H
      phtab(32) = 1.279d-10    ! SiH+ + gamma -> Si  + H+
c
c (ph33) -- H2O+ photodissociation; branching ratios in rate_eq.F
c           Estimate from SD95.
c
      phtab(33) = 1.765d-10
c
c (ph34) -- H3O+ photodissociation; branching ratios in rate_eq.F
c           Estimate from SD95.
c
      phtab(34) = 4.85d-11
c
c (ph35) -- HD photodissociation
c XXX: recompute based on Abgrall & Roueff 2006
c
      phtab(35) = 3.3d-11
c
c (ph36) -- H3+ photodissociation, total rate
c 
c vD87 quotes an _upper limit_ of 1e-12 s^-1 for the H3+ 
c photodissociation rate in the local ISM, which we take
c as the actual rate in the absence of better data. UMIST99 
c assumes that two outcomes (H2 + H+ or H2+ + H) are equally 
c likely. We adopt the same assumption here (but rescale to
c G0=1)
c
      phtab(36) = 5.88d-13
c
c (ph37): unused
c
c (ph38 - ph45): Photoionization of HI, HeI, HeII and H2,
c                together with associated heating. 
c
c These are not currently set in the chemistry code, but can be
c set by external code
c
      phtab(38) = 0d0  ! HI   ion
      phtab(39) = 0d0  ! HeI  ion
      phtab(40) = 0d0  ! HeII ion
      phtab(41) = 0d0  ! H2   ion
      phtab(42) = 0d0  ! HI   heat
      phtab(43) = 0d0  ! HeI  heat
      phtab(44) = 0d0  ! HeII heat
      phtab(45) = 0d0  ! H2   heat
c
c (ph46): Photoionization of OI - currently set to zero
c
      phtab(46) = 0d0
c
c (ph47): Photoionization of SiI - currently set to zero
c
      phtab(47) = 0d0
c
c (ph48): Photodissociation of CHx [NL99]
c
      phtab(48) = 1d-9
c
c (ph49): Photodissociation of OHx [NL99]
c
      phtab(49) = 5d-10
c
c (ph50): Photodissociation of HCO+ [NL99]
c
      phtab(50) = 1.5d-10
c
c (ph51): Photoionization of M [NL99]
c
      phtab(51) = 2.0d-10
c
c (ph52): Photodissociation of N2  [vD88]
c
      phtab(52) = 1.353d-10
c
c (ph53): Photodissociation of NO
c
c Rate from Roberge et al (1991)
c
      phtab(53) = 2.53d-10
c
c (ph54): Photoionization of NO
c
      phtab(54) = 1.53d-10
c
c Scaling factor used to express radiation field strength in units of
c the Habing field.
c
      f_rsc = 1d0
c
#endif /* CHEMCOOL */
      return
      end
c=======================================================================
c
c    \\\\\\\\\\        E N D   S U B R O U T I N E        //////////
c    //////////             P H O T O I N I T             \\\\\\\\\\
c
c=======================================================================
