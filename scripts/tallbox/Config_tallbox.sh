# Stratified shearing box of the solar neighbourhood with FIRE-3 physics (see README.md)
FIRE_PHYSICS_DEFAULTS=3
INIT_STELLAR_METALS_AGES_DEFINED      # initial gas metallicity from InitMetallicity (IC metals are not read on fresh starts)
BOX_PERIODIC
BOX_SHEARING=3                        # (x,y,z) = (r,phi,z)
BOX_SHEARING_Q=1                      # flat rotation curve
BOX_SHEARING_OMEGA_BOX_CENTER=0.028   # 28 km/s/kpc in code units (pc, km/s)
BOX_LONG_Z=8                          # z in [0, 8 L], mid-plane at 4 L
BOX_OUTFLOW_Z
GRAVITY_PERIODIC_XY                   # shear-periodic x/y, open z self-gravity
GRAVITY_ANALYTIC_STRATIFIED_DISK      # stellar disk + DM vertical gravity (StratDisk_* params)
MULTIPLEDOMAINS=16
