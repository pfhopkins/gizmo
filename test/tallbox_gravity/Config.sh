########################################
# tallbox_gravity -- self-gravity periodic in x/y by image sums, open in z (tall box).
# Collisionless particles only; the test compares the walk with a direct sum over the same images.
########################################
BOX_PERIODIC
BOX_LONG_Z=8
BOX_OUTFLOW_Z
GRAVITY_PERIODIC_XY_IMAGES
OUTPUT_ACCELERATION
OUTPUT_HYDROACCELERATION
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
