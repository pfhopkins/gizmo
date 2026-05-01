#!/usr/bin/env bash
# build_three_refs.sh — Build the three reference binaries for Step-15 benchmarks.
#
#   A.  GPU-current        : current branch (gpu_kokkos_bench), Kokkos build
#   B.  CPU-Kokkos-current : current branch, Kokkos OMP backend
#   C.  Legacy pre-GPU     : gizmo-cpp HEAD (production), no Kokkos
#
# Outputs three binaries into ${BENCH_BIN_DIR} with a manifest.json recording
# git SHA, SYSTYPE, Config.sh contents, build host, and timestamp for each.
#
# Usage:
#   ./build_three_refs.sh <problem> [--platform mac|vista] [--builds A,B,C]
#
# Example:
#   ./build_three_refs.sh poisson_box --platform mac --builds A,C
#   ./build_three_refs.sh gmc_cooling --platform vista
#
# Notes:
# - The script uses a git worktree at ${WORKTREE_DIR} for the legacy build.
#   First invocation creates the worktree; subsequent invocations reuse it.
# - On Mac, build B is identical to build A (Kokkos OMP backend on host),
#   so passing --builds A,B is collapsed to A on platform=mac.
# - The script never amends or pushes; worktree is local-only.
# - Legacy gizmo-cpp on Vista requires a Vista[_CPU] SYSTYPE block in its
#   Makefile. If missing, the script aborts with instructions.

set -euo pipefail

GIZMO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
BENCH_DIR="${GIZMO_ROOT}/test/benchmark/step15"
BENCH_BIN_DIR="${BENCH_DIR}/build"
WORKTREE_DIR="${GIZMO_ROOT}/../gizmo_legacy_worktree"
# Local-only branch forked off gizmo-cpp HEAD; carries Step15-specific build shims
# (e.g. corrected homebrew paths on Mac, Vista[_CPU] systype block). Never pushed.
LEGACY_BRANCH="gizmo-cpp-step15"
LEGACY_BASE="gizmo-cpp"
NJOBS="${NJOBS:-8}"

PROBLEM=""
PLATFORM=""
BUILDS="A,B,C"
LEGACY_DIR_OVERRIDE=""   # --legacy-dir: bypass git worktree, use this plain dir for build C

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)   PLATFORM="$2";          shift 2 ;;
        --builds)     BUILDS="$2";            shift 2 ;;
        --jobs)       NJOBS="$2";             shift 2 ;;
        --legacy-dir) LEGACY_DIR_OVERRIDE="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *)
            if [[ -z "$PROBLEM" ]]; then PROBLEM="$1"; shift
            else echo "ERROR: unknown arg $1" >&2; exit 2; fi ;;
    esac
done

[[ -n "$PROBLEM"  ]] || { echo "ERROR: missing <problem>" >&2; exit 2; }
[[ -n "$PLATFORM" ]] || { case "$(uname -s)" in
    Darwin) PLATFORM=mac ;;
    Linux)  PLATFORM=vista ;;
    *)      echo "ERROR: cannot autodetect platform" >&2; exit 2 ;;
esac; }

PROBLEM_DIR="${GIZMO_ROOT}/test/${PROBLEM}"

# Select Config.sh per build label.
# GPU/Kokkos builds (A, B) prefer *_gpu.sh variants; legacy build (C) prefers *_cpu.sh.
# Falls back to base Config.sh if no variant exists.
config_for_build() {
    local label="$1"
    # 1. Explicit named candidates (highest priority)
    local -a cands
    if [[ "$label" != "C" ]]; then
        cands=("Config_phase2e_gpu.sh" "Config_${PROBLEM}_gpu.sh")
    else
        cands=("Config_phase2e_cpu.sh" "Config_${PROBLEM}_cpu.sh")
    fi
    for cand in "${cands[@]}"; do
        if [[ -f "${PROBLEM_DIR}/${cand}" ]]; then echo "${PROBLEM_DIR}/${cand}"; return; fi
    done
    # 2. Any Config_*_gpu.sh / Config_*_cpu.sh glob (e.g. Config_phase5_gpu.sh)
    if [[ "$label" != "C" ]]; then
        for f in "${PROBLEM_DIR}"/Config_*_gpu.sh; do
            [[ -f "$f" ]] && { echo "$f"; return; }
        done
    else
        for f in "${PROBLEM_DIR}"/Config_*_cpu.sh; do
            [[ -f "$f" ]] && { echo "$f"; return; }
        done
    fi
    # 3. Generic named fallbacks
    for cand in "Config_${PROBLEM}.sh" "Config.sh"; do
        if [[ -f "${PROBLEM_DIR}/${cand}" ]]; then echo "${PROBLEM_DIR}/${cand}"; return; fi
    done
    echo ""
}

mkdir -p "${BENCH_BIN_DIR}"

# Map (build, platform) → SYSTYPE.
systype_for() {
    local build="$1" platform="$2"
    case "${build}-${platform}" in
        A-mac)   echo "MacBookCellar_Kokkos" ;;
        A-vista) echo "Vista" ;;
        B-mac)   echo "MacBookCellar_Kokkos" ;;   # collapses to A
        B-vista) echo "Vista_CPU" ;;
        C-mac)   echo "MacBookCellar" ;;
        C-vista) echo "Vista" ;;                  # legacy branch w/ Vista patch
        *) echo "ERROR: unsupported build/platform $1/$2" >&2; return 2 ;;
    esac
}

# Build helper. $1 = build label (A/B/C), $2 = source dir, $3 = output binary path
do_build() {
    local label="$1" srcdir="$2" outbin="$3"
    local systype; systype="$(systype_for "$label" "$PLATFORM")"
    local config_src; config_src="$(config_for_build "$label")"

    echo ""
    echo "============================================================"
    echo "BUILD ${label}  (SYSTYPE=${systype}, problem=${PROBLEM})"
    echo "Source: ${srcdir}"
    echo "Output: ${outbin}"
    echo "============================================================"

    pushd "${srcdir}" >/dev/null

    # Make sure this source tree's Makefile actually defines this SYSTYPE.
    if ! grep -q "ifeq (\$(SYSTYPE),\"${systype}\")" Makefile; then
        echo "ERROR: ${srcdir}/Makefile has no SYSTYPE=\"${systype}\" block." >&2
        echo "       For build ${label} on ${PLATFORM}, add a Vista[_CPU] block to" >&2
        echo "       the gizmo-cpp Makefile (see docs/step15_benchmark_plan.md §11)." >&2
        popd >/dev/null
        return 3
    fi

    # Snapshot existing Config.sh / Makefile.systype
    cp -f Config.sh         Config.sh.step15_backup       2>/dev/null || true
    cp -f Makefile.systype  Makefile.systype.step15_backup 2>/dev/null || true

    # Install per-problem Config.sh + per-build SYSTYPE
    cp -f "${config_src}" Config.sh
    printf 'SYSTYPE="%s"\n' "${systype}" > Makefile.systype

    make clean >/dev/null 2>&1 || true
    make -j"${NJOBS}" 2>&1 | tee "${outbin}.buildlog" | tail -n 5

    if [[ ! -f GIZMO ]]; then
        echo "ERROR: build ${label} produced no GIZMO binary" >&2
        cp -f Config.sh.step15_backup        Config.sh        2>/dev/null || true
        cp -f Makefile.systype.step15_backup Makefile.systype 2>/dev/null || true
        popd >/dev/null
        return 4
    fi

    cp -f GIZMO "${outbin}"
    cp -f Config.sh "${outbin}.Config.sh"

    # Restore originals
    cp -f Config.sh.step15_backup        Config.sh        2>/dev/null || true
    cp -f Makefile.systype.step15_backup Makefile.systype 2>/dev/null || true

    # Capture metadata (git may not be available on remote rsync-based setups)
    local sha; sha="$(git -C "${srcdir}" rev-parse HEAD 2>/dev/null || echo 'unknown')"
    local branch; branch="$(git -C "${srcdir}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
    cat > "${outbin}.meta.json" <<META
{
  "build_label":  "${label}",
  "systype":      "${systype}",
  "platform":     "${PLATFORM}",
  "problem":      "${PROBLEM}",
  "source_dir":   "${srcdir}",
  "git_sha":      "${sha}",
  "git_branch":   "${branch}",
  "build_host":   "$(hostname)",
  "build_time":   "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "njobs":        ${NJOBS}
}
META

    popd >/dev/null
    echo "  -> built ${outbin}"
}

# Ensure legacy worktree exists for build C
ensure_legacy_worktree() {
    if [[ -d "${WORKTREE_DIR}/.git" || -f "${WORKTREE_DIR}/.git" ]]; then
        echo ">>> Reusing legacy worktree: ${WORKTREE_DIR}"
        local cur_br
        cur_br="$(git -C "${WORKTREE_DIR}" rev-parse --abbrev-ref HEAD)"
        if [[ "${cur_br}" != "${LEGACY_BRANCH}" ]]; then
            echo "ERROR: legacy worktree is on '${cur_br}', expected '${LEGACY_BRANCH}'." >&2
            echo "       Switch it manually or remove the worktree to recreate." >&2
            return 5
        fi
    else
        echo ">>> Creating legacy worktree at ${WORKTREE_DIR}"
        # Create worktree on a local-only branch forked from origin/${LEGACY_BASE}
        if git -C "${GIZMO_ROOT}" show-ref --verify --quiet "refs/heads/${LEGACY_BRANCH}"; then
            git -C "${GIZMO_ROOT}" worktree add "${WORKTREE_DIR}" "${LEGACY_BRANCH}"
        else
            git -C "${GIZMO_ROOT}" worktree add -b "${LEGACY_BRANCH}" \
                "${WORKTREE_DIR}" "${LEGACY_BASE}"
        fi
    fi
}

# Run requested builds
IFS=',' read -r -a BUILD_ARR <<< "${BUILDS}"

# Mac collapses B → A
if [[ "${PLATFORM}" == "mac" ]]; then
    BUILD_ARR=( $(printf '%s\n' "${BUILD_ARR[@]}" | sed 's/^B$/A/' | awk '!seen[$0]++') )
    echo "[mac] B is identical to A on this platform; collapsed."
fi

# Pre-check: Config variant must exist for every requested build.
for label in "${BUILD_ARR[@]}"; do
    cfg="$(config_for_build "$label")"
    [[ -n "$cfg" ]] || {
        echo "ERROR: no Config.sh or known fallback in ${PROBLEM_DIR} for build ${label}" >&2; exit 2; }
    echo "[build ${label}] config: ${cfg}"
done

for label in "${BUILD_ARR[@]}"; do
    case "${label}" in
        A|B)
            outbin="${BENCH_BIN_DIR}/GIZMO_${PROBLEM}_${PLATFORM}_${label}"
            do_build "${label}" "${GIZMO_ROOT}" "${outbin}"
            ;;
        C)
            if [[ -n "${LEGACY_DIR_OVERRIDE}" ]]; then
                echo ">>> Using --legacy-dir override: ${LEGACY_DIR_OVERRIDE}"
                LEGACY_SRCDIR="${LEGACY_DIR_OVERRIDE}"
            else
                ensure_legacy_worktree
                LEGACY_SRCDIR="${WORKTREE_DIR}"
            fi
            outbin="${BENCH_BIN_DIR}/GIZMO_${PROBLEM}_${PLATFORM}_${label}"
            do_build "${label}" "${LEGACY_SRCDIR}" "${outbin}"
            ;;
        *) echo "ERROR: unknown build label '${label}'" >&2; exit 2 ;;
    esac
done

echo ""
echo "============================================================"
echo "DONE. Binaries:"
ls -lh "${BENCH_BIN_DIR}"/GIZMO_${PROBLEM}_${PLATFORM}_* 2>/dev/null || true
echo "============================================================"
