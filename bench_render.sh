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
THREADS=${BENCH_THREADS:-0}
SPEED=${BENCH_SPEED:-1}

cargo build --release >/dev/null || exit 1

common=(--mode "$MODE" --cols "$COLS" --rows "$ROWS" --rays "$RAYS" --fps 30
        --speed "$SPEED" --threads "$THREADS")

run_case() {
    local name=$1
    shift
    printf '\n== %s ==\n' "$name"
    printf 'mode=%s cols=%s rows=%s rays=%s speed=%s threads=%s duration=%ss\n' \
        "$MODE" "$COLS" "$ROWS" "$RAYS" "$SPEED" "$THREADS" "$SECONDS_TO_RUN"
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

# Throughput companion: the paced runs above sleep to the fps budget for
# most of their wall time and pay the one-off geodesic re-trace, so their
# task-clock mostly measures idle and setup. This section renders flat out
# (a budget no frame can hit) and counts actually drawn frames: the status
# line carries " fps:" exactly once per rendered frame, so the achieved
# frame rate can be measured from the outside.
printf '\n== throughput (no pacing, warm caches) ==\n'
for name_f in 'super-current|--super-star --funnel current' \
              'super-tidal|--super-star --funnel tidal' \
              'super-spiral|--super-star --funnel spiral'; do
    name=${name_f%%|*}
    flags=${name_f#*|}
    # shellcheck disable=SC2086
    frames=$(timeout "$SECONDS_TO_RUN" "$BIN" --mode "$MODE" --cols "$COLS" \
        --rows "$ROWS" --rays "$RAYS" --fps 240 --speed "$SPEED" \
        --threads "$THREADS" $flags 2>/dev/null | \
        grep -a -o ' fps:' | wc -l)
    awk -v n="$frames" -v t="$SECONDS_TO_RUN" \
        'BEGIN { printf "%s: %d frames in %ss = %.1f fps (%.2f ms/frame)\n", \
                 "'"$name"'", n, t, n / t, t * 1000 / (n > 0 ? n : 1) }'
done
