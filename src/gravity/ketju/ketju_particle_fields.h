/* KETJU per-particle fields; included inside struct particle_data. Member state sits in the star record (ketju_member_state). */
    short int KetjuIntegrated;   /*!< 1 while the particle's current step is integrated by MSTAR */
    MyIDType KetjuChainID;       /*!< hash of its chain's smallest member ID; 0 = not in a chain */
    float KetjuOpenRadius;       /*!< the walk opens nodes within this distance of a chain member (comoving); < 0: chain on one rank; 0 = none */
#ifdef HERMITE_INTEGRATION
    short int HermiteHistoryStale; /*!< 1 for the step after leaving a chain: KDK, Hermite history stale */
#endif
#ifdef KETJU_SINKS
    MyDouble KetjuSpin[3];       /*!< BH spin angular momentum S */
#if !defined(GALSF_SINGLE_STARS_BH_PROMOTION)
    short int SinkSubType;       /*!< 0 = SMBH (IC/seeded), 1 = stellar-mass BH (dead star) */
#endif
#endif
