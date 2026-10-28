# g0_consistency

Checks that the three Stella radiation paths hand the chemistry the same FUV/LW physics. The TREE_RAD path
(`GALSF_RESOLVEDISM_G0_VARIABLE`, the gizmo2017 convention) is the reference. A 40 Msun star (Stella tables, age 0.5 Myr)
sits at the centre of a 32^3 lattice, 20 pc across, and runs for 1.2 kyr through three media:

| case    | medium                                                          | what it probes                           |
|---------|-----------------------------------------------------------------|------------------------------------------|
| thin    | n_H = 0.3 cm^-3                                                 | unit conversion at the chemistry hand-off |
| uniform | n_H = 100 cm^-3                                                 | attenuation, with gas behind the star     |
| slab    | n_H = 0.3 plus an n_H = 200, 1.875 pc slab 2.5 pc from the star | attenuation by an absorber in between     |

The test reads the chemistry's own rates (`OUTPUT_CHEM_PHOTORATES`): H2 and CO photodissociation, the PE-heating field
`G_dust` and the dust-heating field. It compares them per cell with the TREE_RAD conventions applied to the exact field
L/(4 pi r^2 c) and the exact star-to-cell column, times the chemistry's own H2 and CO line self-shielding factors. It takes the median over cells at 3-7 pc (thin, uniform) or over the
shadowed cells behind the slab.

Measured (median rate / analytic):

| path     | thin      | uniform     | slab      | note                                                                   |
|----------|-----------|-------------|-----------|------------------------------------------------------------------------|
| TREE_RAD | 0.98-1.00 | 0.001-0.01  | 0.84-0.89 | TreeCol column in the star's pixel runs on past the star (over-shielding) |
| TREE_RAY | 1.00      | 0.91-0.94   | 0.98-0.99 | source-to-cell attenuation in the walk; attenuated field still drifts at 1.2 kyr (tree opening) |
| M1       | 1.21      | 1.4-1.9     | 2.0-2.7   | M1 closure near a point source; HLL attenuation per cell at tau_cell 0.5-0.8 |

The M1 run stops at 1.2 kyr, before light reflected at the closed box faces gets back to the sampled cells (GIZMO has no
radiation outflow boundary). The test builds GIZMO three times and runs nine short jobs, about 15 min on 4 ranks.
