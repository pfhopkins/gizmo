/* KETJU run parameters; included inside struct global_data_all_processes */
  double KetjuRegionRadius;              /*!< physical radius of KETJU regions [code units] */
  double KetjuMinStarMass;               /*!< minimum star mass to be a chain candidate [code units], 0=off */
  double KetjuIntegrationTolerance;      /*!< GBS relative tolerance */
  int    KetjuUseStarStarSoftening;      /*!< keep softening between stars inside chain */
  double KetjuTimestepLimitingFactor;    /*!< radius factor for timestep limiting */
  int    KetjuMaxStepCount;              /*!< max MSTAR steps before bail-out (0=off) */
  double KetjuExpandBinariesFactor;      /*!< expand tight binaries if period < factor*dt (0=off) */
  double KetjuMinParticlesPerTask;       /*!< minimum particles per compute task */
  int    KetjuKeplerPairs;               /*!< 1: two-body regions without PN advance by an exact Kepler drift */
