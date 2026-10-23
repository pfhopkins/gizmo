# ketju_tide

A 1 + 1 Msun, 200 AU pair in a KETJU region, tidally perturbed by a 20 Msun star at 5000 AU, for 0.2 Myr; the pair's
semi-major axis is compared with a DOP853 three-body integration. The host reaches the region's members with the external
field only as two kicks per step, at random orbital phases when the step is a good fraction of the orbit (default:
|da/a| 7.4e-4). Two remedies are tested: a step floor res P (1-e)^1.5 when the per-step kick energy exceeds a budget
(`KetjuTidalStepRes`, `KetjuTidalErrorBudget`), and the linear tide applied inside MSTAR (`KetjuSmoothTide`).
