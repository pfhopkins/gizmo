#!/bin/bash
# The new-machine procedure of the Stella docs, end to end, with an empty HOME:
#   new_machine.sh SCRATCH
# Uses the caller's build environment (modules, LD_LIBRARY_PATH, SYSTYPE and its Kokkos). GIZMO_REPO / GIZMO_BRANCH: what
# is cloned (default: this checkout, its branch; committed state only). NP: MPI ranks and make jobs (default 4).
# Steps: clone; setup; build + cloud smoke run; KETJU config stops in make naming the setup; setup again (picks up KETJU
# from Config.sh); build + run with KETJU; rerun of the setup is a check only; a run without the cache stops at start-up.
set -euo pipefail
S="$(mkdir -p "$1" && cd "$1" && pwd)"; HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${GIZMO_REPO:-$(git -C "$HERE" rev-parse --show-toplevel)}"; BRANCH="${GIZMO_BRANCH:-$(git -C "$HERE" rev-parse --abbrev-ref HEAD)}"
NP="${NP:-4}"; G="$S/gizmo"
export HOME="$S/home"; mkdir -p "$HOME"; unset STELLA_DEPS_PREFIX KETJU_PATH
step() { echo "new_machine: $*"; }
fail() { echo "new_machine: FAILED: $*"; exit 1; }

step "clone $REPO ($BRANCH), cloud IC"
git clone -q -b "$BRANCH" "$REPO" "$G"
git clone -q --depth 1 -b ics https://github.com/Ulli1991/stella-tables.git "$S/stella-ics"
bash "$S/stella-ics/ICs/get_stella_ics.sh" --prefix "$S/ics" clouds/starforge_M2e4_m1.0_mol10K.hdf5 > /dev/null
IC="$S/ics/clouds/starforge_M2e4_m1.0_mol10K"

step "setup"
bash "$G/scripts/get_stella_deps.sh" > "$S/setup1.log" || fail "setup, see $S/setup1.log"
grep -q "all checks passed" "$S/setup1.log" || fail "setup summary"

build() { # name config
    cp "$2" "$G/src/Config.sh"
    (cd "$G" && make clean > /dev/null 2>&1; make -j "$NP" > "$S/build_$1.log" 2>&1)
}
run_cloud() { # name: a few steps of the shipped cloud params, tables from the cache
    local d="$S/run_$1"; mkdir -p "$d"
    sed -e "s|^InitCondFile .*|InitCondFile $IC|" -e "s|^OutputDir .*|OutputDir $d/output|" -e "s|^TimeMax .*|TimeMax 0.00002|" \
        -e "s|^TimeBetSnapshot .*|TimeBetSnapshot 0.00001|" "$G/scripts/stella/stella_cloud.params" > "$d/cloud.params"
    (cd "$d" && OMP_NUM_THREADS=1 mpirun -np "$NP" "$G/src/GIZMO" cloud.params > run.log 2>&1) || fail "run $1, see $d/run.log"
    grep -q "Final time=" "$d/run.log" || fail "run $1 did not reach TimeMax"
    grep -q "$HOME/.stella-deps/tables-.*/stellar_tables_unified.hdf5 ok" "$d/run.log" || fail "run $1: stellar tables not from the cache"
    grep -q "$HOME/.stella-deps/tables-.*/wss_z_collis.txt ok" "$d/run.log" || fail "run $1: chemistry tables not from the cache"
}

step "build + run cloud"
build cloud "$G/scripts/stella/Config_stella_cloud.sh" || fail "build, see $S/build_cloud.log"
run_cloud cloud

step "KETJU config without the install stops in make"
sed -e 's/^ADAPTIVE_GRAVSOFT_FORALL=1+16/ADAPTIVE_GRAVSOFT_FORALL=1/' "$G/scripts/stella/Config_stella_cloud.sh" > "$S/Config_ketju.sh"
echo KETJU_REGULARIZATION >> "$S/Config_ketju.sh"
build ketju_missing "$S/Config_ketju.sh" && fail "KETJU build without install succeeded"
grep -q "run scripts/get_stella_deps.sh --ketju" "$S/build_ketju_missing.log" || fail "make message, see $S/build_ketju_missing.log"

step "setup with KETJU (from Config.sh), build + run"
bash "$G/scripts/get_stella_deps.sh" > "$S/setup2.log" || fail "setup with KETJU, see $S/setup2.log"
grep -q "KETJU" "$S/setup2.log" && grep -q "all checks passed" "$S/setup2.log" || fail "setup with KETJU summary"
build ketju "$S/Config_ketju.sh" || fail "KETJU build, see $S/build_ketju.log"
run_cloud ketju

step "rerun of the setup"
t0=$(date +%s); bash "$G/scripts/get_stella_deps.sh" > "$S/setup3.log" || fail "rerun"
[ "$(grep -c "already installed" "$S/setup3.log")" = 2 ] && ! grep -q downloading "$S/setup3.log" || fail "rerun did work, see $S/setup3.log"
step "rerun took $(( $(date +%s) - t0 )) s"

step "run without the cache stops at start-up"
mkdir -p "$S/empty"
(cd "$S/run_ketju" && STELLA_DEPS_PREFIX="$S/empty" OMP_NUM_THREADS=1 mpirun -np 2 "$G/src/GIZMO" cloud.params > nocache.log 2>&1) && fail "run without tables succeeded"
grep -q "run scripts/get_stella_deps.sh" "$S/run_ketju/nocache.log" || fail "start-up message, see $S/run_ketju/nocache.log"

step "all steps passed"
