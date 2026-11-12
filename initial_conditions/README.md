# Initial conditions

Tools for building GIZMO initial-condition files.

- `make_IC.py` — general-purpose example for writing a GIZMO/GADGET-style HDF5
  IC file (particle positions, masses, velocities, internal energies, and the
  required header). A good starting template for hand-built ICs; see also the
  IC-format section of the user guide in `docs/`.

- `eos_tools/` — end-to-end builder for self-gravitating bodies in hydrostatic
  equilibrium against a tabulated equation of state (terrestrial/giant planets,
  asteroids, stars, white dwarfs, brown dwarfs, …). It carries the full recipe
  — HSE solve → particle placement → internal-energy correction → HDF5 output —
  behind one interface. Entry point: `eos_tools/build_layered_body.py`; a
  worked Earth-like example is in `eos_tools/test_hse_earth.py`. The user guide
  (`docs/`) documents the modules and usage in detail.

For reading and analyzing the snapshots a run produces, see `analysis/`.
