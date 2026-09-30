#!/usr/bin/env bash
# run_multigpu.sh -- concurrent single-GPU plotting, and the VRAM ceiling check.
#
# Produces E6.1 and E6.3.
#
#   default mode: launches N independent plotters at once, one per GPU, for each
#   N in the list. Aggregate throughput against N is the scale-out result; the
#   per-GPU slowdown says which host resource (PCIe, disk, host RAM) caps the
#   node first. Point several -drives at different disks to separate storage
#   contention from PCIe contention -- with one drive, expect the disk to cap it.
#
#   -vram mode: runs one plot per K while sampling nvidia-smi, and compares the
#   measured peak against the model 3*2^K*NONCE_SIZE + 2*2^24*4.
#
# Results: experiments/<host>/multigpu_<tag>.csv  or  vram_<tag>.csv
#
# Usage:
#   ./run_multigpu.sh -k 31 -n 1,2,4,8 -drives /mnt/nvme0,/mnt/nvme1
#   ./run_multigpu.sh -vram -k 27-32 -drives /mnt/nvme
#
# Options:
#   -k LIST        K values                                 (default: 31; -vram: 27-32)
#   -n LIST        GPU counts to test                       (default: 1,2,4,8)
#   -backend NAME  cuda | sycl-gpu                          (default: cuda)
#   -drives LIST   output dirs, cycled across GPUs          (default: /tmp)
#   -runs N        repeats per combination                  (default: 1)
#   -vram          VRAM ceiling mode (E6.3) instead of scaling
#   -tag NAME      label used in the filename               (default: scaleout / ceiling)
#   -o PATH        output CSV path or directory
#   -keep          keep generated plots
#   -nodrop        do not drop page cache between runs
#   -dry-run       print what would run, change nothing
#   -v             print each command
#   -h             this help
#
# NONCE_SIZE defaults to 4 for the VRAM model; override with NONCE_SIZE=5.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

K_SPEC=""; N_SPEC="1,2,4,8"; BACKEND="cuda"; DRIVES="/tmp"; RUNS=1
VRAM_MODE=false; TAG=""; OUT=""; KEEP=false
NONCE_SIZE="${NONCE_SIZE:-4}"

usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; $d'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -k)        K_SPEC="$2"; shift 2 ;;
        -n)        N_SPEC="$2"; shift 2 ;;
        -backend)  BACKEND="$2"; shift 2 ;;
        -drives)   DRIVES="$2"; shift 2 ;;
        -runs)     RUNS="$2"; shift 2 ;;
        -vram)     VRAM_MODE=true; shift ;;
        -tag)      TAG="$2"; shift 2 ;;
        -o)        OUT="$2"; shift 2 ;;
        -keep)     KEEP=true; shift ;;
        -nodrop)   DROP_CACHES=false; shift ;;
        -dry-run)  DRY_RUN=true; shift ;;
        -v)        VERBOSE=true; shift ;;
        -h|--help) usage ;;
        *) die "unknown option '$1' (try -h)" ;;
    esac
done

$VRAM_MODE && { K_SPEC="${K_SPEC:-27-32}"; TAG="${TAG:-ceiling}"; } \
           || { K_SPEC="${K_SPEC:-31}";    TAG="${TAG:-scaleout}"; }

BIN="$(backend_binary "$BACKEND")"
[[ -x "$BIN" ]] || die "$BIN not built"
BFLAGS="$(backend_flags "$BACKEND")"
BENV="$(backend_env "$BACKEND")"

mapfile -t KS      < <(expand_list "$K_SPEC")
mapfile -t DRIVE_L < <(split_list  "$DRIVES")
record_machine_info

# model_vram_mb <K> -- the budget the plotter itself checks against
model_vram_mb() {
    awk -v k="$1" -v ns="$NONCE_SIZE" \
        'BEGIN { printf "%.0f", (3 * 2^k * ns + 2 * 2^24 * 4) / 1048576 }'
}

# ── VRAM ceiling mode (E6.3) ──────────────────────────────────────
if $VRAM_MODE; then
    OUT="$(resolve_output "$OUT" "vram_${TAG}.csv")"
    [[ -f "$OUT" ]] || echo "timestamp,host,backend,device_idx,K,repeat,model_vram_mb,peak_vram_mb,total_vram_mb,ratio,wall_s,exit_code" > "$OUT"
    command -v nvidia-smi > /dev/null 2>&1 || die "-vram needs nvidia-smi"

    total_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 | head -1)"
    log "VRAM ceiling: K=${KS[*]} on a ${total_mb} MB device -> $OUT"

    for k in "${KS[@]}"; do
    for ((repeat = 1; repeat <= RUNS; repeat++)); do
        model="$(model_vram_mb "$k")"
        log "K=$k model=${model} MB (device has ${total_mb} MB)"
        samples="$(mktemp)"
        if ! $DRY_RUN; then
            nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
                       -i 0 -lms 200 > "$samples" 2>/dev/null &
            sampler=$!
        fi
        drop_caches
        t0="$(date +%s.%N)"; rc=0
        args=(-k "$k" -f "${DRIVE_L[0]}" -d 0)
        [[ -n "$BFLAGS" ]] && args+=($BFLAGS)
        run_cmd "$BENV" "$BIN" "${args[@]}" > /dev/null 2>&1 || rc=$?
        t1="$(date +%s.%N)"
        wall="$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.3f", b - a }')"

        if ! $DRY_RUN; then
            kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true
            peak="$(awk 'NF && $1+0 > m { m = $1+0 } END { print m+0 }' "$samples")"
            ratio="$(awk -v p="$peak" -v m="$model" 'BEGIN { if (m > 0) printf "%.3f", p/m; else print 0 }')"
            printf '%s,%s,%s,0,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$HOSTNAME_SHORT" "$BACKEND" \
                "$k" "$repeat" "$model" "$peak" "$total_mb" "$ratio" "$wall" "$rc" >> "$OUT"
            log "  peak=${peak} MB  model=${model} MB  ratio=${ratio}  $( ((rc)) && echo "EXIT $rc" || echo ok)"
            $KEEP || rm -f "${DRIVE_L[0]}"/k"$k"-*.plot
        fi
        rm -f "$samples"
    done; done
    log "done -> $OUT"
    exit 0
fi

# ── multi-GPU scaling (E6.1) ──────────────────────────────────────
mapfile -t N_L < <(split_list "$N_SPEC")
OUT="$(resolve_output "$OUT" "multigpu_${TAG}.csv")"
LOGDIR="${OUT%.csv}.logs"; mkdir -p "$LOGDIR"
[[ -f "$OUT" ]] || echo "timestamp,host,backend,K,n_gpus,repeat,wall_s,slowest_gpu_s,fastest_gpu_s,vaults_per_hour,scaling_efficiency,drives,failures" > "$OUT"

declare -A BASELINE   # vaults/hour at n=1, per K, for the efficiency column

log "multi-GPU scaling: K=${KS[*]} N=${N_L[*]} drives=${DRIVE_L[*]} -> $OUT"

for k in "${KS[@]}"; do
for n in "${N_L[@]}"; do
for ((repeat = 1; repeat <= RUNS; repeat++)); do
    log "K=$k with $n concurrent GPU(s), run $repeat/$RUNS"
    drop_caches

    pids=(); starts=(); logs=(); used_drives=()
    t0="$(date +%s.%N)"
    for ((g = 0; g < n; g++)); do
        drive="${DRIVE_L[$((g % ${#DRIVE_L[@]}))]}"
        used_drives+=("$(drive_label "$drive")")
        lf="${LOGDIR}/k${k}_n${n}_gpu${g}_r${repeat}.txt"
        logs+=("$lf")
        args=(-k "$k" -f "$drive" -d "$g")
        [[ -n "$BFLAGS" ]] && args+=($BFLAGS)
        if $DRY_RUN; then
            printf '  DRY: %s %s %s\n' "$BENV" "$BIN" "${args[*]}" >&2
        else
            starts+=("$(date +%s.%N)")
            ( if [[ -n "$BENV" ]]; then env $BENV "$BIN" "${args[@]}"; else "$BIN" "${args[@]}"; fi ) \
                > "$lf" 2>&1 &
            pids+=($!)
        fi
    done

    fails=0
    if ! $DRY_RUN; then
        for pid in "${pids[@]}"; do wait "$pid" || fails=$((fails + 1)); done
    fi
    t1="$(date +%s.%N)"
    wall="$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.3f", b - a }')"

    if ! $DRY_RUN; then
        # Per-GPU time comes from each process's own reported total.
        slowest=0; fastest=999999
        for lf in "${logs[@]}"; do
            s="$(sed -n 's/^Total:[[:space:]]*\([0-9.]*\).*/\1/p' "$lf" | head -1)"
            [[ -z "$s" ]] && continue
            awk -v a="$s" -v b="$slowest" 'BEGIN { exit !(a > b) }' && slowest="$s"
            awk -v a="$s" -v b="$fastest" 'BEGIN { exit !(a < b) }' && fastest="$s"
        done
        [[ "$fastest" == "999999" ]] && fastest=""

        vph="$(awk -v n="$n" -v w="$wall" 'BEGIN { if (w > 0) printf "%.2f", n * 3600 / w; else print 0 }')"
        [[ -z "${BASELINE[$k]:-}" ]] && BASELINE[$k]="$vph"
        eff="$(awk -v v="$vph" -v b="${BASELINE[$k]}" -v n="$n" \
               'BEGIN { if (b > 0 && n > 0) printf "%.3f", v / (b * n); else print "" }')"

        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"%s",%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$HOSTNAME_SHORT" "$BACKEND" "$k" "$n" \
            "$repeat" "$wall" "$slowest" "$fastest" "$vph" "$eff" \
            "$(IFS=';'; echo "${used_drives[*]}")" "$fails" >> "$OUT"
        log "  wall=${wall}s  slowest_gpu=${slowest}s  ${vph} vaults/h  efficiency=${eff}  failures=$fails"

        $KEEP || for d in "${DRIVE_L[@]}"; do rm -f "$d"/k"$k"-*.plot; done
    fi
done; done; done

log "done -> $OUT"
log "  per-process logs: $LOGDIR"
