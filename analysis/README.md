# Analysis tools

Utilities for reading and analyzing GIZMO snapshots.

## Recommended: MESHOID

For most analysis we recommend [**MESHOID**](https://github.com/mikegrudic/meshoid),
a dedicated mesh-free analysis package (rendering, density estimation, gradients,
slices/projections, etc.). It understands GIZMO/GADGET-style HDF5 snapshots
directly. It is also listed as a dependency in `pyproject.toml`.

## Built-in readers (lightweight alternative)

If you just need to load fields out of a snapshot without extra dependencies:

- `load_from_snapshot.py` — the up-to-date reader (correct unit conversions and
  modern field names). Put `analysis/` on your Python path and use it directly:
  ```python
  from load_from_snapshot import load_from_snapshot
  pos = load_from_snapshot("Coordinates", 0, "output", 10)   # gas coords, snap 010
  ```
- `readsnap.py` — older general-purpose snapshot reader.
- `compress_gizmosnap.py` — utility to losslessly shrink snapshot files.
- `legacy_idl/readsnap.pro` — IDL reader, kept for legacy workflows.
- `visit/` — plugin for the [VisIt](https://visit-dav.github.io/visit-website/)
  visualization tool.

## Building initial conditions

IC-generation tools live in the top-level `initial_conditions/` directory, not here.
