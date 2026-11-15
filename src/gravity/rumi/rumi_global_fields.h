/* RUMI++ run-time parameters; included inside struct global_data_all_processes */
  char   RumiConfigFile[256];            /*!< RUMI++ config file (integrator keys) */
  double RumiStarMaxTimestep;            /*!< cap on the common star step [code units], 0 = none */
  double RumiLengthUnitPc, RumiMassUnitMsun, RumiVelUnitKms; /*!< RUMI++ units of the code units (0: the code units in pc, Msun, km/s) */
  int    RumiPrintInterval;              /*!< common star steps between rumi.txt rows and RUMI ENERGY lines, 0 = off */
  int    RumiNumRanks;                   /*!< tasks 0..RumiNumRanks-1 run RUMI++ (its distributed path) */
