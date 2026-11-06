/* KETJU run parameters; included inside struct global_data_all_processes */
  double KetjuRegionRadius;              /*!< physical radius of KETJU regions [code units] */
  double KetjuMinBHMass;                 /*!< minimum Type 5 stage-7 (BH/relic) mass to act as always-center [code units], 0=off */
  double KetjuMinStarMass;               /*!< minimum star mass to be a chain candidate [code units], 0=off */
  double KetjuIntegrationTolerance;      /*!< GBS relative tolerance */
  char   KetjuPNTerms[20];               /*!< PN term flags: "all","none","no_spin", or bitstring */
  int    KetjuEnableBHMergerKicks;       /*!< GW recoil kicks for BH mergers */
  int    KetjuUseStarStarSoftening;      /*!< keep softening between stars inside chain */
  double KetjuTimestepLimitingFactor;    /*!< radius factor for timestep limiting */
  int    KetjuMaxStepCount;              /*!< max MSTAR steps before bail-out (0=off) */
  double KetjuExpandBinariesFactor;      /*!< expand tight binaries if period < factor*dt (0=off) */
  double KetjuMinParticlesPerTask;       /*!< minimum particles per compute task */
  double KetjuRegionMotionCapFactor;     /*!< scales the region-step motion caps; 0 = off */
  double KetjuTidalStepRes;              /*!< region-step floor res*P(1-e)^1.5 for tidally kicked binaries (0 = off) */
  double KetjuTidalErrorBudget;          /*!< per-step tidal kick energy allowed, in units of the binary's |E| */
  int    KetjuSmoothTide;                /*!< 1: members feel the external tide inside MSTAR, the host kicks only the CoM */
  int    KetjuKeplerPairs;               /*!< 1: two-body regions without PN advance by an exact Kepler drift */
  double KetjuFlybyStepEta;              /*!< Type-4 stars outside regions: fly-by pair step factor (0 = off) */
