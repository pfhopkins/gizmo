# ags_face_matrix

Checks the face matrix that the adaptive-softening kernel loop stores in `P[].NV_T` for type-1 particles when `CBE_INTEGRATOR` or `DM_FUZZY` is on. Those modules build their effective faces and gradients from it, so it must be the inverse of the neighbour moment matrix `sum_j W_ij dx dx^T`.

A driver runs GIZMO's startup on a jittered lattice of 512 particles in a periodic box, then prints the stored matrix and the moment matrix recomputed by direct summation with the code's own kernel. The test asserts `max |NV_T M - 1| < 1e-10`. The reference is a matrix identity, so the tolerance only allows for round-off. The lattice is jittered because the off-diagonal moments vanish on a regular one, and an error in them would go unseen.

Run with `pytest test/ags_face_matrix`.
