# readsnap_ids

`analysis/readsnap.py` returns particle IDs as stored, in the file's dtype. It used to allocate an int64 array and add 2^31 to
every ID above 1e9 or below 0, the bit-flip correction for the 32-bit IDs of new stars; with 64-bit IDs (the C++ code writes
`MyIDType` of 64 bits for large runs) that shifts valid IDs.

Problem: a one-file HDF5 snapshot of five stars with IDs 1, 7, 2^31 + 5, 2^40 + 3, 2^62 + 11, as uint64 and int64. No GIZMO
build.

Check: the IDs readsnap returns equal the stored ones.
