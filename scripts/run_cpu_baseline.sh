#!/usr/bin/env bash
# run_cpu_baseline.sh -- CPU VaultX on the same host, swept over thread count.
#
# Produces E2.2 (and the CPU side of E7.1 via RAPL). The point is fairness: the
# GPU speedup must be quoted against the best CPU configuration on the same
# machine, not against one thread. The sweep also gives CPU parallel efficiency
# and the thread count at which the CPU stops scaling.
#
# Needs a built CPU VaultX binary from the sibling repo; nothing here builds it.
#
# Results: experiments/<host>/cpu_<tag>.csv
#          experiments/<host>/cpu_<tag>.logs/   (raw vaultx output per run)
#
# Usage:
#   ./run_cpu_baseline.sh -bin ~/vaultx/vaultx -k 27-31 -drives /mnt/nvme
#   ./run_cpu_baseline.sh -bin ~/vaultx/vaultx -k 31 -threads 1,8,32,64 -power
#
# Options:
#   -bin PATH      CPU VaultX binary                     (default: ../../vaultx/vaultx)
#   -k LIST        K values, e.g. 27-31                  (default: 27-31)
#   -threads LIST  thread counts; 'auto' = 1,2,4,...,nproc (default: auto)
#   -io-threads N  value for --threads_io                (default: 1)
#   -drives LIST   output directories                    (default: /tmp)
#   -runs N        repeats per combination               (default: 1)
#   -power         record CPU package energy via RAPL    (default: off)
#   -tag NAME      label used in the filename            (default: baseline)
#   -o PATH        output CSV path or directory
#   -keep          keep generated plots
#   -nodrop        do not drop page cache between runs
#   -dry-run       print what would run, change nothing
#   -v             print each command
#   -h             this help
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

CPU_BIN="${ROOT_DIR}/../vaultx/vaultx"
K_SPEC="27-31"; THREAD_SPEC="auto"; IO_THREADS=1; DRIVES="/tmp"; RUNS=1
TAG="baseline"; OUT=""; KEEP=false

usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; $d'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -bin)        CPU_BIN="$2"; shift 2 ;;
        -k)          K_SPEC="$2"; shift 2 ;;
        -threads)    THREAD_SPEC="$2"; shift 2 ;;
        -io-threads) IO_THREADS="$2"; shift 2 ;;
        -drives)     DRIVES="$2"; shift 2 ;;
        -runs)       RUNS="$2"; shift 2 ;;
        -power)      POWER_ENABLED=true; shift ;;
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

[[ -x "$CPU_BIN" ]] || die "CPU VaultX binary not found at '$CPU_BIN' (pass -bin PATH)"

# 1,2,4,8,... up to nproc, with nproc itself always included.
if [[ "$THREAD_SPEC" == "auto" ]]; then
    max="$(nproc)"; list=(); t=1
    while (( t < max )); do list+=("$t"); t=$((t * 2)); done
    list+=("$max")
    THREAD_L=("${list[@]}")
else
    mapfile -t THREAD_L < <(split_list "$THREAD_SPEC")
fi
mapfile -t KS      < <(expand_list "$K_SPEC")
mapfile -t DRIVE_L < <(split_list  "$DRIVES")

OUT="$(resolve_output "$OUT" "cpu_${TAG}.csv")"
LOGDIR="${OUT%.csv}.logs"; mkdir -p "$LOGDIR"
record_machine_info

[[ -f "$OUT" ]] || echo "timestamp,host,K,threads,io_threads,drive,drive_label,repeat,wall_s,reported_total_s,storage_efficiency,io_mb_s,peak_mem_mb,throughput_mh_s,avg_w,energy_j,exit_code" > "$OUT"

field() { sed -n "$1" "$2" | head -1; }

total=$(( ${#KS[@]} * ${#THREAD_L[@]} * ${#DRIVE_L[@]} * RUNS )); n=0
log "CPU baseline: $total runs, threads=${THREAD_L[*]}, K=${KS[*]} -> $OUT"

for drive in "${DRIVE_L[@]}"; do
    [[ -d "$drive" ]] || { warn "skipping drive $drive: not a directory"; continue; }
    dlabel="$(drive_label "$drive")"
    for k in "${KS[@]}"; do
    for threads in "${THREAD_L[@]}"; do
    for ((repeat = 1; repeat <= RUNS; repeat++)); do
        n=$((n + 1))
        log "[$n/$total] CPU K=$k threads=$threads drive=$dlabel run=$repeat/$RUNS"
        logf="${LOGDIR}/cpu_k${k}_t${threads}_${dlabel}_r${repeat}.txt"

        drop_caches
        power_start 0
        t0="$(date +%s.%N)"; rc=0
        run_cmd "" "$CPU_BIN" -k "$k" -f "$drive" -t "$threads" \
                --threads_io "$IO_THREADS" > "$logf" 2>&1 || rc=$?
        t1="$(date +%s.%N)"
        wall="$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.3f", b - a }')"
        read -r avg_w _max_w energy_j <<< "$(power_stop "$wall")"
        (( rc != 0 )) && warn "run failed (exit $rc); see $logf"

        if ! $DRY_RUN; then
            ttime=$(field 's/.*Total Time:[[:space:]]*\([0-9.]*\).*/\1/p' "$logf")
            se=$(   field 's/.*storage_efficiency_table2=\([0-9.]*\).*/\1/p' "$logf")
            iomb=$( field 's/.*Overall I\/O Throughput:[[:space:]]*\([0-9.]*\).*/\1/p' "$logf")
            pmem=$( field 's/.*Peak Memory Usage:[[:space:]]*\([0-9.]*\).*/\1/p' "$logf")
            thr=$(  field 's/.*Total Throughput:[[:space:]]*\([0-9.]*\).*/\1/p' "$logf")
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$HOSTNAME_SHORT" "$k" "$threads" \
                "$IO_THREADS" "$drive" "$dlabel" "$repeat" "$wall" \
                "${ttime:-NA}" "${se:-NA}" "${iomb:-NA}" "${pmem:-NA}" "${thr:-NA}" \
                "$avg_w" "$energy_j" "$rc" >> "$OUT"
            $KEEP || rm -f "$drive"/k"$k"-*.plot
        fi
    done; done; done
done

log "done -> $OUT"
