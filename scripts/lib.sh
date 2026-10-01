#!/usr/bin/env bash
# lib.sh -- shared helpers for the VaultXGPU experiment scripts.
# Not runnable on its own; sourced by run_*.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
EXPERIMENTS_DIR="${VAULTX_EXPERIMENTS_DIR:-${ROOT_DIR}/experiments/${HOSTNAME_SHORT}}"

DRY_RUN=false
VERBOSE=false

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
warn() { printf '[%s] WARNING: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

# ── argument helpers ──────────────────────────────────────────────

# expand_list "27-31"  -> 27 28 29 30 31
# expand_list "27,29"  -> 27 29
# expand_list "27-29,32" -> 27 28 29 32
expand_list() {
    local spec="$1" out=() part lo hi
    IFS=',' read -ra _parts <<< "$spec"
    for part in "${_parts[@]}"; do
        if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            lo="${BASH_REMATCH[1]}"; hi="${BASH_REMATCH[2]}"
            for ((i = lo; i <= hi; i++)); do out+=("$i"); done
        else
            out+=("$part")
        fi
    done
    printf '%s\n' "${out[@]}"
}

# split_list "a,b,c" -> a b c
split_list() { local IFS=','; read -ra _s <<< "$1"; printf '%s\n' "${_s[@]}"; }

# on|off|both -> the values to iterate
expand_toggle() {
    case "$1" in
        on)   echo on ;;
        off)  echo off ;;
        both) printf 'on\noff\n' ;;
        *)    die "expected on|off|both, got '$1'" ;;
    esac
}

# ── backends ──────────────────────────────────────────────────────
# cuda      -> vaultx_cuda
# sycl-gpu  -> vaultx_sycl, refuses to run on a CPU device
# sycl-cpu  -> vaultx_sycl, forced onto the SYCL CPU device

backend_binary() {
    local backend="$1" sort_mode="${2:-2}" suffix=""
    [[ "$sort_mode" != "2" ]] && suffix="_sort${sort_mode}"
    case "$backend" in
        cuda)               echo "${ROOT_DIR}/vaultx_cuda${suffix}" ;;
        sycl-gpu|sycl-cpu)  echo "${ROOT_DIR}/vaultx_sycl${suffix}" ;;
        *) die "unknown backend '$backend' (want cuda, sycl-gpu or sycl-cpu)" ;;
    esac
}

backend_available() {
    local bin; bin="$(backend_binary "$1" "${2:-2}")"
    [[ -x "$bin" ]]
}

# Extra flags each backend needs. sycl-gpu must refuse the CPU fallback so a CPU
# number can never be recorded as a GPU result.
backend_flags() {
    case "$1" in
        sycl-gpu) echo "--require-gpu" ;;
        *)        echo "" ;;
    esac
}

# Environment for a backend. ONEAPI_DEVICE_SELECTOR is what forces the SYCL CPU
# device on a host that also has a GPU; override with VAULTX_SYCL_CPU_SELECTOR.
backend_env() {
    case "$1" in
        sycl-cpu) echo "ONEAPI_DEVICE_SELECTOR=${VAULTX_SYCL_CPU_SELECTOR:-*:cpu}" ;;
        *)        echo "" ;;
    esac
}

# ── output paths ──────────────────────────────────────────────────

# resolve_output <explicit-path-or-empty> <default-basename>
# Accepts a full path, or a directory (trailing / or existing dir), or nothing.
resolve_output() {
    local given="$1" default_name="$2" path
    if [[ -z "$given" ]]; then
        path="${EXPERIMENTS_DIR}/${default_name}"
    elif [[ -d "$given" || "$given" == */ ]]; then
        path="${given%/}/${default_name}"
    else
        path="$given"
    fi
    mkdir -p "$(dirname "$path")"
    echo "$path"
}

# Sidecar holding the per-run facts the plotter's own CSV does not carry
# (tag, drive, repeat index, power). Join to the main CSV on the `run` column,
# which this library keeps unique within an output file.
sidecar_path() { echo "${1%.csv}.runs.csv"; }

init_sidecar() {
    local f; f="$(sidecar_path "$1")"
    [[ -f "$f" ]] && return 0
    echo "run,repeat,tag,backend,device_idx,K,drive,drive_label,sort,chunk_mb,odirect,overlap,powercap_w,wall_s,avg_w,max_w,energy_j,exit_code" > "$f"
}

# Next unique run id for this output file.
next_run_id() {
    local f; f="$(sidecar_path "$1")"
    [[ -f "$f" ]] || { echo 1; return; }
    awk -F, 'NR>1 && $1+0 > m { m = $1+0 } END { print m+1 }' "$f"
}

# A short label for a drive path, safe in filenames: /mnt/nvme0 -> mnt_nvme0
drive_label() { echo "${1#/}" | tr '/ ' '__'; }

# ── system helpers ────────────────────────────────────────────────

DROP_CACHES=true

drop_caches() {
    $DROP_CACHES || return 0
    sync
    if [[ $EUID -eq 0 ]]; then
        echo 3 > /proc/sys/vm/drop_caches
    elif sudo -n true 2>/dev/null; then
        echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null
    else
        warn "cannot drop caches without passwordless sudo; write timings will include page-cache effects (use -odirect on)"
        DROP_CACHES=false
    fi
}

# ── power sampling ────────────────────────────────────────────────
# GPU: nvidia-smi polling at 100 ms. CPU: RAPL package counters.
# Both are best-effort; a run never fails because sampling is unavailable.

POWER_ENABLED=false
_power_pid=""
_power_file=""
_rapl_start=""

rapl_energy_uj() {
    local total=0 f v
    for f in /sys/class/powercap/intel-rapl:*/energy_uj; do
        [[ -r "$f" ]] || continue
        v="$(cat "$f" 2>/dev/null || echo 0)"
        total=$((total + v))
    done
    echo "$total"
}

power_start() {
    local device="${1:-0}"
    $POWER_ENABLED || return 0
    _power_file="$(mktemp)"
    _rapl_start="$(rapl_energy_uj)"
    if command -v nvidia-smi > /dev/null 2>&1; then
        nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits \
                   -i "$device" -lms 100 > "$_power_file" 2>/dev/null &
        _power_pid=$!
    else
        _power_pid=""
    fi
}

# power_stop <wall_seconds> -> "avg_w max_w energy_j"
power_stop() {
    local wall="$1" avg=0 max=0 energy=0
    $POWER_ENABLED || { echo "0 0 0"; return; }

    if [[ -n "$_power_pid" ]]; then
        kill "$_power_pid" 2>/dev/null || true
        wait "$_power_pid" 2>/dev/null || true
        read -r avg max <<< "$(awk 'NF && $1+0 > 0 { s += $1; n++; if ($1+0 > m) m = $1+0 }
                                   END { if (n) printf "%.2f %.2f", s/n, m; else print "0 0" }' \
                              "$_power_file")"
        energy="$(awk -v a="$avg" -v w="$wall" 'BEGIN { printf "%.1f", a * w }')"
    else
        # No NVIDIA device: fall back to the CPU package counters.
        local d=$(( $(rapl_energy_uj) - _rapl_start ))
        (( d < 0 )) && d=0
        energy="$(awk -v d="$d" 'BEGIN { printf "%.1f", d / 1000000 }')"
        avg="$(awk -v e="$energy" -v w="$wall" 'BEGIN { if (w > 0) printf "%.2f", e / w; else print 0 }')"
        max="$avg"
    fi
    rm -f "$_power_file"
    echo "$avg $max $energy"
}

set_power_cap() {
    local device="$1" watts="$2"
    [[ -z "$watts" || "$watts" == "none" ]] && return 0
    command -v nvidia-smi > /dev/null 2>&1 || { warn "no nvidia-smi; ignoring -powercap"; return 0; }
    if sudo -n nvidia-smi -i "$device" -pl "$watts" > /dev/null 2>&1; then
        log "power cap set to ${watts} W on device ${device}"
    else
        warn "could not set power cap (needs passwordless sudo); continuing uncapped"
    fi
}

# ── execution ─────────────────────────────────────────────────────

# run_cmd <env-assignments> <command...>
# Honours -dry-run. Returns the command's exit code.
run_cmd() {
    local envs="$1"; shift
    if $DRY_RUN; then
        printf '  DRY: %s %s\n' "$envs" "$*" >&2
        return 0
    fi
    $VERBOSE && printf '  RUN: %s %s\n' "$envs" "$*" >&2
    if [[ -n "$envs" ]]; then
        env $envs "$@"
    else
        "$@"
    fi
}

# run_logged <logfile> <env-assignments> <command...>
# Sends the command's output to <logfile> so a failure is diagnosable afterwards.
# Under -dry-run it prints the command and writes nothing. Prefer this over
# redirecting at the call site: a call site that redirects stderr would also
# swallow the -dry-run and -verbose lines.
run_logged() {
    local logf="$1"; shift
    local envs="$1"; shift
    if $DRY_RUN; then
        printf '  DRY: %s %s\n' "$envs" "$*" >&2
        return 0
    fi
    $VERBOSE && printf '  RUN: %s %s\n' "$envs" "$*" >&2
    mkdir -p "$(dirname "$logf")"
    if [[ -n "$envs" ]]; then
        env $envs "$@" > "$logf" 2>&1
    else
        "$@" > "$logf" 2>&1
    fi
}

# Record the machine's configuration once per experiments directory, so a CSV is
# never orphaned from the hardware that produced it.
record_machine_info() {
    local out="${EXPERIMENTS_DIR}/machine_info.txt"
    mkdir -p "$EXPERIMENTS_DIR"
    {
        echo "host:      $(hostname -f 2>/dev/null || hostname)"
        echo "date:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "kernel:    $(uname -sr)"
        echo "cpu:       $(awk -F: '/model name/ { print $2; exit }' /proc/cpuinfo | sed 's/^ *//')"
        echo "cores:     $(nproc) hardware threads"
        echo "memory:    $(awk '/MemTotal/ { printf "%.1f GB", $2/1048576 }' /proc/meminfo)"
        echo
        echo "--- GPUs ---"
        if command -v nvidia-smi > /dev/null 2>&1; then
            nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit \
                       --format=csv 2>/dev/null || echo "nvidia-smi failed"
        else
            echo "no nvidia-smi"
        fi
        command -v sycl-ls > /dev/null 2>&1 && { echo; echo "--- SYCL devices ---"; sycl-ls 2>/dev/null; }
        echo
        echo "--- mounts of interest ---"
        df -hT 2>/dev/null | awk 'NR==1 || /^\/dev|nfs/'
    } > "$out"
    log "machine info -> $out"
}
