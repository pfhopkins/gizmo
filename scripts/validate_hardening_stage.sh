#!/bin/bash
# Stage a validation sweep for one stage of the tree-hardening series and print the
# sbatch line. Submission is always manual. Usage:
#
#   ./scripts/validate_hardening_stage.sh stage1|stage2|stage3|stage4|stage5
#
# Test presets follow the hardening plan's validation matrix (each also carries the
# fewbody guards and compile_suite; hernquist rides along where step-count parity is
# the check). Stage 0 (comments/dead code) validates locally: compile_suite + one
# smoke run -- no node needed. Stage 4's DEVELOPER_MODE (deep-audit) arms build via
# each test's own Config plus DEVELOPER_MODE, which the audit gate keys off; run
# those locally or add DEVELOPER_MODE to the relevant test Configs in a scratch tree.
set -euo pipefail

STAGE="${1:?usage: validate_hardening_stage.sh stageN}"
case "$STAGE" in
    stage1) TESTS="binary triple hernquist fire gmc_cooling compile_suite";;
    stage2) TESTS="shu_jets wind_singlestar SN_singlestar binary triple compile_suite";;
    stage3) TESTS="SN_singlestar shu_jets wind_singlestar isodisk_mechfb_sinks HII_region gmc_cooling_rt fire fire_rtsources binary triple compile_suite";;
    stage4) TESTS="shu_jets SN_singlestar binary gmc_cooling compile_suite";;
    stage5) TESTS="shu_jets wind_singlestar SN_singlestar binary triple hernquist gmc_cooling fire isodisk_mechfb_sinks fire_gravtree_rt isodisk_thermalfb compile_suite";; # the last two are non-MAINTAIN GALSF: the only configs where the stage-5 deferred-rebuild ladder branch runs
    *) echo "unknown stage '$STAGE'"; exit 1;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="/mnt/ceph/users/mgrudic/starforge_dev_sweep/hardening_${STAGE}_$(date +%Y%m%d_%H%M%S)"
SWEEP_TESTS="$TESTS" "$REPO/scripts/sweep_prepare.sh" "$DEST"
