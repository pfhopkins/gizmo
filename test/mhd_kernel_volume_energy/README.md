# mhd_kernel_volume_energy (broad: total-energy conservation)

`MHD_CONSERVE_ENERGY_ON_KERNEL_VOLUME_CHANGE` and `MHD_RESCALE_B_TO_BOOKED_MAGNETIC_ENERGY`, two alternative closures for the
MFM MHD energy gain: the cell magnetic energy follows the kernel volume, the internal-energy equation books the face volume change.

Problem: static periodic box, 20^3 cells with 20 % lattice jitter, 70 pc, 2400 Msun, uniform B_z = 6.61 uG, beta about 0.14,
adiabatic (the check is total-energy conservation), 2 ranks, 1.96 Myr. Energies E_kin + E_th + E_B from the five snapshots.

Checks (bars from stella_dev on the same box: +2.89 %, -0.20 %, -0.29 %): without either option the end drift is above +2 %;
with either option it is below 0.5 % in size.

Run on the port (2 ranks, 3 builds): 3 passed. End drift +2.75 % without either option, -0.33 % charging u, -0.42 % rescaling B.
