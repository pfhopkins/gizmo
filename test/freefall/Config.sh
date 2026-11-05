########################################
# freefall -- pressureless collapse of a cold uniform gas sphere (MFM + self-gravity).
# Variants (test_freefall.py) add ADAPTIVE_GRAVSOFT_FORGAS, or
# TIDAL_TIMESTEP_CRITERION + ADAPTIVE_TREEFORCE_UPDATE.
########################################
HYDRO_MESHLESS_FINITE_MASS
EOS_GAMMA=(5.0/3.0)
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
STOP_WHEN_BELOW_MINTIMESTEP
OUTPUT_POTENTIAL            # gravitational term of the energy check
OUTPUT_ACCELERATION         # with OUTPUT_HYDROACCELERATION set, "Acceleration" is gravity only
OUTPUT_HYDROACCELERATION    # ... and "HydroAcceleration" the hydro part (file_io/io.cc IO_ACCEL)
OUTPUT_SOFTENING            # "Softening_KernelRadius": the kernel support each cell's gravity used
