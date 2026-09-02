#!/usr/bin/env bash
# Render benchmark for blackhole. The process is deliberately run for a
# sustained interval rather than with --frame: --frame measures setup and
# simulation time, not steady-state animation rendering.
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BIN="$ROOT/target/release/blackhole"
SECONDS_TO_RUN=${BENCH_SECONDS:-10}
COLS=${BENCH_COLS:-80}
ROWS=${BENCH_ROWS:-24}
RAYS=${BENCH_RAYS:-200000}
MODE=${BENCH_MODE:-ascii}

cargo build --release >/dev/null || exit 1

common=(--mode "$MODE" --cols "$COLS" --rows "$ROWS" --rays "$RAYS" --fps 30)

run_case() {
    local name=$1
    shift
    printf '\n== %s ==\n' "$name"
    printf 'mode=%s cols=%s rows=%s rays=%s duration=%ss\n' \
        "$MODE" "$COLS" "$ROWS" "$RAYS" "$SECONDS_TO_RUN"
    if command -v perf >/dev/null 2>&1; then
        # timeout returns 124 when the requested measurement interval ends.
        perf stat -e task-clock,cycles,instructions,branches,branch-misses -- \
            timeout "$SECONDS_TO_RUN" "$BIN" "${common[@]}" "$@" >/dev/null
        local rc=$?
        [[ $rc -eq 0 || $rc -eq 124 ]] || return "$rc"
    else
        /usr/bin/time -f 'elapsed=%e sec cpu=%P' \
            timeout "$SECONDS_TO_RUN" "$BIN" "${common[@]}" "$@" >/dev/null
        local rc=$?
        [[ $rc -eq 0 || $rc -eq 124 ]] || return "$rc"
    fi
}

run_case plain
run_case super-current --super-star --funnel current
run_case super-tidal --super-star --funnel tidal
run_case super-spiral --super-star --funnel spiral
