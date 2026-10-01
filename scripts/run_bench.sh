#!/usr/bin/env bash
# run_bench.sh -- the main plot-generation sweep.
#
# Sweeps K, backend, sort variant, drive, chunk size, O_DIRECT, overlap and power
# cap, one plot per combination, appending a row to a CSV per run. This one script
# produces the data for E1.3, E2.1, E2.3, E2.4, E3.2, E4.1-E4.4, E5.1-E5.3, E7.1
# and E7.3 -- which experiment you get depends only on which flags you vary.
#
# Results: experiments/<host>/bench_<tag>.csv  (the plotter's own schema)
#          experiments/<host>/bench_<tag>.runs.csv  (tag, drive, power; join on `run`)
#          experiments/<host>/bench_<tag>.logs/     (each run's full output)
#
# Usage:
#   ./run_bench.sh -k 27-31 -drives /mnt/nvme                       # E2.1, E2.4
#   ./run_bench.sh -k 29-31 -sort 0,1,2 -tag sortstudy              # E3.2
#   ./run_bench.sh -k 31 -overlap off -odirect on -tag decompose    # E4.1
#   ./run_bench.sh -k 31 -drives /mnt/nvme,/mnt/ssd,/tmp -tag media # E4.2
#   ./run_bench.sh -k 31 -overlap on,off -tag overlap               # E4.3
#   ./run_bench.sh -k 31 -chunk 16,64,256,1024 -odirect both        # E4.4
#   ./run_bench.sh -k 27-31 -backend cuda,sycl-gpu -tag cudavssycl  # E5.1
#   ./run_bench.sh -k 27-29 -backend sycl-cpu -tag syclcpu          # E5.3
#   ./run_bench.sh -k 31 -power -runs 3 -tag energy                 # E7.1
#   ./run_bench.sh -k 31 -power -powercap 150,200,250,none          # E7.3
#
# Options:
#   -k LIST         K values, e.g. 27-31 or 27,29,31          (default: 27-31)
#   -backend LIST   cuda | sycl-gpu | sycl-cpu                (default: cuda)
#   -device LIST    GPU indices                               (default: 0)
#   -drives LIST    output directories, one per storage medium(default: /tmp)
#   -sort LIST      sort variants 0,1,2 (needs the _sortN binaries) (default: 2)
#   -chunk LIST     staging chunk size in MB                  (default: 256)
#   -odirect V      on | off | both                           (default: on)
#   -overlap V      on | off | both                           (default: on)
#   -runs N         repeats per combination                   (default: 3)
#   -powercap LIST  watt caps, or 'none'                      (default: none)
#   -stats          read bucket counters back (SE, overflow)  (default: off)
#   -power          sample power and record joules per run    (default: off)
#   -tag NAME       label for this sweep, used in the filename(default: sweep)
#   -o PATH         output CSV path or directory              (default: experiments/<host>/)
#   -keep           keep generated plots instead of deleting  (default: delete)
#   -nodrop         do not drop page cache between runs
#   -dry-run        print what would run, change nothing
#   -verbose        print each command (note: the plotter's -v means verify)
#   -h              this help
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

K_SPEC="27-31"; BACKENDS="cuda"; DEVICES="0"; DRIVES="/tmp"
SORTS="2"; CHUNKS="256"; ODIRECT="on"; OVERLAP="on"; RUNS=3
POWERCAPS="none"; TAG="sweep"; OUT=""; KEEP=false; STATS=false

usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; $d'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -k)        K_SPEC="$2"; shift 2 ;;
        -backend)  BACKENDS="$2"; shift 2 ;;
        -device)   DEVICES="$2"; shift 2 ;;
        -drives)   DRIVES="$2"; shift 2 ;;
        -sort)     SORTS="$2"; shift 2 ;;
        -chunk)    CHUNKS="$2"; shift 2 ;;
        -odirect)  ODIRECT="$2"; shift 2 ;;
        -overlap)  OVERLAP="$2"; shift 2 ;;
        -runs)     RUNS="$2"; shift 2 ;;
        -powercap) POWERCAPS="$2"; shift 2 ;;
        -tag)      TAG="$2"; shift 2 ;;
        -o)        OUT="$2"; shift 2 ;;
        -stats)    STATS=true; shift ;;
        -power)    POWER_ENABLED=true; shift ;;
        -keep)     KEEP=true; shift ;;
        -nodrop)   DROP_CACHES=false; shift ;;
        -dry-run)  DRY_RUN=true; shift ;;
        -verbose)  VERBOSE=true; shift ;;
        -h|--help) usage ;;
        *) die "unknown option '$1' (try -h)" ;;
    esac
done

mapfile -t KS        < <(expand_list "$K_SPEC")
mapfile -t BACKEND_L < <(split_list  "$BACKENDS")
mapfile -t DEVICE_L  < <(split_list  "$DEVICES")
mapfile -t DRIVE_L   < <(split_list  "$DRIVES")
mapfile -t SORT_L    < <(split_list  "$SORTS")
mapfile -t CHUNK_L   < <(split_list  "$CHUNKS")
mapfile -t ODIRECT_L < <(expand_toggle "$ODIRECT")
mapfile -t OVERLAP_L < <(expand_toggle "$OVERLAP")
mapfile -t CAP_L     < <(split_list  "$POWERCAPS")

OUT="$(resolve_output "$OUT" "bench_${TAG}.csv")"
SIDECAR="$(sidecar_path "$OUT")"
LOGDIR="${OUT%.csv}.logs"
mkdir -p "$LOGDIR"
init_sidecar "$OUT"
record_machine_info

total=$(( ${#KS[@]} * ${#BACKEND_L[@]} * ${#DEVICE_L[@]} * ${#DRIVE_L[@]} * \
          ${#SORT_L[@]} * ${#CHUNK_L[@]} * ${#ODIRECT_L[@]} * ${#OVERLAP_L[@]} * \
          ${#CAP_L[@]} * RUNS ))
log "sweep '$TAG': $total runs -> $OUT"
log "  K=${KS[*]} backends=${BACKEND_L[*]} sorts=${SORT_L[*]} drives=${DRIVE_L[*]}"

done_n=0; failed=0

for backend in "${BACKEND_L[@]}"; do
for sort_mode in "${SORT_L[@]}"; do
    bin="$(backend_binary "$backend" "$sort_mode")"
    if [[ ! -x "$bin" ]]; then
        warn "skipping backend=$backend sort=$sort_mode: $bin not built"
        continue
    fi
    bflags="$(backend_flags "$backend")"
    benv="$(backend_env "$backend")"

for device in "${DEVICE_L[@]}"; do
for drive in "${DRIVE_L[@]}"; do
    if [[ ! -d "$drive" ]]; then
        warn "skipping drive $drive: not a directory"
        continue
    fi
    dlabel="$(drive_label "$drive")"

for chunk in "${CHUNK_L[@]}"; do
for odirect in "${ODIRECT_L[@]}"; do
for overlap in "${OVERLAP_L[@]}"; do
for cap in "${CAP_L[@]}"; do
    set_power_cap "$device" "$cap"

for k in "${KS[@]}"; do
for ((repeat = 1; repeat <= RUNS; repeat++)); do
    run_id="$(next_run_id "$OUT")"
    done_n=$((done_n + 1))

    args=(-k "$k" -f "$drive" -d "$device" --csv "$OUT" --run "$run_id" --chunk-mb "$chunk")
    [[ "$odirect" == "on" ]] && args+=(--o-direct)
    [[ "$overlap" == "off" ]] && args+=(--no-overlap)
    $STATS && args+=(--stats)
    [[ -n "$bflags" ]] && args+=($bflags)

    log "[$done_n/$total] $backend sort=$sort_mode K=$k dev=$device drive=$dlabel chunk=${chunk}MB odirect=$odirect overlap=$overlap cap=$cap run=$repeat/$RUNS"

    logf="${LOGDIR}/run${run_id}_${backend}_s${sort_mode}_k${k}_${dlabel}.txt"

    drop_caches
    power_start "$device"
    t0="$(date +%s.%N)"
    rc=0
    run_logged "$logf" "$benv" "$bin" "${args[@]}" || rc=$?
    t1="$(date +%s.%N)"
    wall="$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.3f", b - a }')"
    read -r avg_w max_w energy_j <<< "$(power_stop "$wall")"

    (( rc != 0 )) && { warn "run failed (exit $rc); see $logf"; failed=$((failed + 1)); }

    $DRY_RUN || printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$run_id" "$repeat" "$TAG" "$backend" "$device" "$k" "$drive" "$dlabel" \
        "$sort_mode" "$chunk" "$odirect" "$overlap" "$cap" \
        "$wall" "$avg_w" "$max_w" "$energy_j" "$rc" >> "$SIDECAR"

    $KEEP || $DRY_RUN || rm -f "$drive"/k"$k"-*.plot
done
done
    [[ "$cap" != "none" ]] && set_power_cap "$device" "$(nvidia-smi -i "$device" --query-gpu=power.default_limit --format=csv,noheader,nounits 2>/dev/null || echo none)"
done; done; done; done; done; done; done; done

log "done: $done_n runs, $failed failed"
log "  results: $OUT"
log "  run log: $SIDECAR"
log "  per-run output: $LOGDIR"
