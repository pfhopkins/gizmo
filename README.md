Welcome!

This is GIZMO (beta version: likely to be Google-style and stay in beta for quite some time).

The simulation code GIZMO is a flexible, multi-method mhd+gravity code. The code includes a hydro solver using the Lagrangian meshless finite-mass method, a meshless finite volume method, the modern "pressure-sph" method, and "traditional" SPH. Self-gravity is solved fast, with a BH-Tree (optionally a hybrid PM-Tree for periodic boundaries), with adaptive gravitational softenings. Hydrodynamics and gravity are both optional. 

The code is descended from P-SPH/P-GADGET, itself descended from GADGET-3 (so a huge debt owes to the work of Volker Springel), and many of the naming conventions and routines remain (for the sake of compatibility with the large library of GADGET work and analysis software). You should see the source code for appropriate attribution of the code elements. Currently available modules include things like: hydrodynamics, MHD, cosmological integrations, galaxy/star/black hole formation with feedback from stars and black holes (both explicit, detailed models and sub-grid models), self-interacting dark matter, adaptive gravitational softening lengths for all particle types, anisotropic conduction and viscosity, sub-grid turbulent diffusion, the ability to insert arbitrary external gravitational fields, integration in non-standard cosmologies, sink particles, "dust fluids" (particulate-gas interactions), cosmic rays (with advection, diffusion, streaming, heating/cooling, and injection by SNe), nuclear+degenerate equations of state (being used in some code branches), and radiation hydrodynamics (in progress, partially implemented). Most of these are not in the public release of the code, but in the private (code development) branch; see the users guide for details.

No, the code title is not an acronym, I just liked it. It refers both to the code's multi-purpose applications and to its historical relationship to GADGET.

The BitBucket site is where I will post code updates, news, and documentation, so check back regularly. If you have code issues, feature requests, bugs, or just questions, please post to the GIZMO Google Group (also the GIZMO mailing list):

https://groups.google.com/d/forum/gizmo-code

If you have confirmed bug reports, please use the (public) Bitbucket issue tracker (after consulting the Google Group site)

The main reference for the numerical methods, setting up the code, code policies, branching etc, is the user's guide, available as the file "gizmo_documentation.html" in the "scripts" folder as part of this repository, or through download on the bitbucket site or at my website: 

http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html

Read it!

The code is written in standard ANSI C, and should run on all parallel platforms that support MPI. The portability of the code has been confirmed on a large number of systems -- if it can run GADGET (and it almost certainly can), it can run GIZMO.

The public version of the code is free software, distributed under the GNU General Public License (http://www.gnu.org/copyleft/gpl.html). This implies that you may freely distribute and copy the software. You may also modify it as you wish, and distribute these modified versions as long as you indicate prominently any changes you made in the original code, and as long as you leave the copyright notices, and the no-warranty notice intact. Please read the General Public License for more details. Note that the authors retain their copyright on the code. The public code is available at:

http://www.tapir.caltech.edu/~phopkins/public/gizmo_public.tgz

The private version of the code is closed and can only be used or distributed with explicit permission from the code authors. Please note that most of the non-public "modules" are proprietary and developed by active students/postdocs for their ongoing research - it is not acceptable to use or share these routines without first obtaining the explicit permission of both the lead code author and the author(s) of the relevant routines.

If you use any version of the code, please reference the code paper at: http://arxiv.org/abs/1409.7395 (Hopkins 2015); you should also reference Volker Springel's GADGET paper (Springel, 2005, MNRAS, 364, 1105) for the domain decomposition and N-body algorithms.
