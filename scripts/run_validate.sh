#!/usr/bin/env bash
# run_validate.sh -- correctness: validate plots, then search them with the CPU prover.
#
# Produces E1.1. For each plot it runs vaultx_validate (structural check: every
# stored pair is recomputed and its Table1 bucket, hash order, match distance and
# Table2 bucket are verified) and, if a CPU VaultX binary is given, a batch of
# random lookups to confirm the plot is searchable by the CPU prover.
#
# It can generate the plots first, or validate plots that already exist.
#
# Results: experiments/<host>/validate_<tag>.csv
#          experiments/<host>/validate_<tag>.logs/  (raw validator and prover output)
#
# Usage:
#   ./run_validate.sh -k 27-29 -backend cuda,sycl-gpu -drives /mnt/nvme
#   ./run_validate.sh -plots /mnt/nvme/plots                      # validate what is there
#   ./run_validate.sh -k 27 -drives /mnt/nvme -prover ~/vaultx/vaultx -challenges 1000
#   ./run_validate.sh -plots /mnt/nvme/plots -full                # check every bucket
#
# Options:
#   -k LIST         K values to generate, e.g. 27-29        (default: 27-29)
#   -backend LIST   cuda | sycl-gpu | sycl-cpu              (default: cuda)
#   -device N       GPU index                               (default: 0)
#   -drives LIST    directories to generate into            (default: /tmp)
#   -plots DIR      validate existing plots here; skips generation entirely
#   -sample N       validate N random buckets               (default: 5000)
#   -full           validate every bucket (slow, exhaustive)
#   -prover PATH    CPU VaultX binary for the lookup check  (default: none)
#   -challenges N   lookups per plot when -prover is given  (default: 1000)
#   -tag NAME       label used in the filename              (default: correctness)
#   -o PATH         output CSV path or directory
#   -keep           keep generated plots (implied by -plots)
#   -nodrop         do not drop page cache before lookups
#   -dry-run        print what would run, change nothing
#   -v              print each command
#   -h              this help
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

K_SPEC="27-29"; BACKENDS="cuda"; DEVICE="0"; DRIVES="/tmp"; PLOTS_DIR=""
SAMPLE=5000; FULL=false; PROVER=""; CHALLENGES=1000
TAG="correctness"; OUT=""; KEEP=false

usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; $d'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -k)          K_SPEC="$2"; shift 2 ;;
        -backend)    BACKENDS="$2"; shift 2 ;;
        -device)     DEVICE="$2"; shift 2 ;;
        -drives)     DRIVES="$2"; shift 2 ;;
        -plots)      PLOTS_DIR="$2"; KEEP=true; shift 2 ;;
        -sample)     SAMPLE="$2"; shift 2 ;;
        -full)       FULL=true; shift ;;
        -prover)     PROVER="$2"; shift 2 ;;
        -challenges) CHALLENGES="$2"; shift 2 ;;
        -tag)        TAG="$2"; shift 2 ;;
        -o)          OUT="$2"; shift 2 ;;
        -keep)       KEEP=true; shift ;;
        -nodrop)     DROP_CACHES=false; shift ;;
        -dry-run)    DRY_RUN=true; shift ;;
        -v)          VERBOSE=true; shift ;;
        -h|--help)   usage ;;
        *) die "unknown option '$1' (try -h)" ;;
    esac
done

VALIDATOR="${ROOT_DIR}/vaultx_validate"
[[ -x "$VALIDATOR" ]] || die "vaultx_validate not built -- run 'make validate'"

OUT="$(resolve_output "$OUT" "validate_${TAG}.csv")"
LOGDIR="${OUT%.csv}.logs"
mkdir -p "$LOGDIR"
record_machine_info

if [[ ! -f "$OUT" ]]; then
    echo "timestamp,host,backend,K,plot,plot_bytes,slots,records,empty,storage_efficiency,invalid,validate_s,validate_ok,prover_total_ms,prover_avg_ms,prover_ok" > "$OUT"
fi

# check_plot <backend> <K> <plot-path>
check_plot() {
    local backend="$1" k="$2" plot="$3"
    local base; base="$(basename "$plot")"
    local vlog="${LOGDIR}/${base}.validate.txt"

    local vargs=("$plot")
    $FULL || vargs+=(--sample "$SAMPLE")

    log "  validating $base"
    local t0 t1 vs rc=0
    t0="$(date +%s.%N)"
    if $DRY_RUN; then
        printf '  DRY: %s %s\n' "$VALIDATOR" "${vargs[*]}" >&2
    else
        "$VALIDATOR" "${vargs[@]}" > "$vlog" 2>&1 || rc=$?
    fi
    t1="$(date +%s.%N)"
    vs="$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.3f", b - a }')"
    $DRY_RUN && return 0

    local slots records empty se invalid vok
    slots=$(  awk '/slots checked/      { print $3 }' "$vlog")
    records=$(awk '/records present/    { print $3 }' "$vlog")
    empty=$(  awk '/empty slots/        { print $3 }' "$vlog")
    se=$(     awk '/storage efficiency/ { gsub(/%/, "", $3); print $3 }' "$vlog")
    invalid=$(awk '/invalid records/    { print $3 }' "$vlog")
    vok=$([[ $rc -eq 0 ]] && echo PASS || echo FAIL)

    # CPU prover: proves the plot is searchable by the unmodified CPU implementation.
    local ptotal="" pavg="" pok="skipped"
    if [[ -n "$PROVER" ]]; then
        local plog="${LOGDIR}/${base}.prover.txt"
        log "  searching $base with the CPU prover ($CHALLENGES lookups)"
        drop_caches
        local prc=0
        "$PROVER" -S "$CHALLENGES" -D 1 -f "$plot" > "$plog" 2>&1 || prc=$?
        local tline; tline="$(grep '^TIMING' "$plog" 2>/dev/null | head -1 || true)"
        if [[ -n "$tline" ]]; then
            ptotal="$(awk '{ print $6 }' <<< "$tline")"
            pavg="$(  awk '{ print $7 }' <<< "$tline")"
        fi
        pok=$([[ $prc -eq 0 ]] && echo PASS || echo "FAIL(exit $prc)")
    fi

    local bytes; bytes="$(stat -c %s "$plot" 2>/dev/null || echo 0)"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$HOSTNAME_SHORT" "$backend" "$k" \
        "$base" "$bytes" "${slots:-}" "${records:-}" "${empty:-}" "${se:-}" \
        "${invalid:-}" "$vs" "$vok" "$ptotal" "$pavg" "$pok" >> "$OUT"

    log "  -> $vok  SE=${se:-?}%  invalid=${invalid:-?}  prover=$pok"
}

# ── mode 1: validate plots that already exist ─────────────────────
if [[ -n "$PLOTS_DIR" ]]; then
    [[ -d "$PLOTS_DIR" ]] || die "$PLOTS_DIR is not a directory"
    shopt -s nullglob
    plots=("$PLOTS_DIR"/k*-*.plot)
    shopt -u nullglob
    (( ${#plots[@]} )) || die "no k*-*.plot files in $PLOTS_DIR"
    log "validating ${#plots[@]} existing plot(s) in $PLOTS_DIR"
    for plot in "${plots[@]}"; do
        k="$(basename "$plot" | sed -n 's/^k\([0-9]\+\)-.*/\1/p')"
        check_plot "existing" "${k:-?}" "$plot"
    done
    log "done -> $OUT"
    exit 0
fi

# ── mode 2: generate, then validate ───────────────────────────────
mapfile -t KS        < <(expand_list "$K_SPEC")
mapfile -t BACKEND_L < <(split_list  "$BACKENDS")
mapfile -t DRIVE_L   < <(split_list  "$DRIVES")

log "generating and validating: K=${KS[*]} backends=${BACKEND_L[*]}"

for backend in "${BACKEND_L[@]}"; do
    bin="$(backend_binary "$backend")"
    [[ -x "$bin" ]] || { warn "skipping $backend: $bin not built"; continue; }
    bflags="$(backend_flags "$backend")"
    benv="$(backend_env "$backend")"

    for drive in "${DRIVE_L[@]}"; do
        [[ -d "$drive" ]] || { warn "skipping drive $drive: not a directory"; continue; }
        for k in "${KS[@]}"; do
            log "generating $backend K=$k in $drive"
            args=(-k "$k" -f "$drive" -d "$DEVICE" --stats)
            [[ -n "$bflags" ]] && args+=($bflags)
            rc=0
            run_cmd "$benv" "$bin" "${args[@]}" > "${LOGDIR}/gen_${backend}_k${k}.txt" 2>&1 || rc=$?
            (( rc != 0 )) && { warn "generation failed (exit $rc); see ${LOGDIR}/gen_${backend}_k${k}.txt"; continue; }

            $DRY_RUN && continue
            shopt -s nullglob
            for plot in "$drive"/k"$k"-*.plot; do
                check_plot "$backend" "$k" "$plot"
                $KEEP || rm -f "$plot"
            done
            shopt -u nullglob
        done
    done
done

log "done -> $OUT"
log "  raw output: $LOGDIR"
