# ============================================================================
# GIZMO top-level build delegator.
#
# The real Makefile and all C/C++ source now live in src/.  This thin wrapper
# lets you build from the repository root:
#
#     make                 # builds src/GIZMO
#     make clean           # forwarded to src/
#
# or you can work directly in the source tree:
#
#     cd src
#     cp Template_Config.sh Config.sh     # then edit Config.sh
#     make
#
# To change machine/compiler settings, edit src/Makefile.systype.
# ============================================================================
.PHONY: all
all:
	@$(MAKE) -C src

%:
	@$(MAKE) -C src $@
