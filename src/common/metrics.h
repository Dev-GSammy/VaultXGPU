#ifndef VAULTXGPU_METRICS_H
#define VAULTXGPU_METRICS_H

#include "globals.h"   // VAULTX_SORT_NAME

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <string>

// ──────────────────────────────────────────────
// Wall clock
// ──────────────────────────────────────────────

// Monotonic seconds. Use for anything that touches the filesystem; use GPU
// events for kernel time (a wall-clock timer around a kernel launch measures
// launch + sync, not execution).
double now_seconds();

// Simple scoped accumulator: start() ... stop() adds the interval to *sink.
struct StageTimer {
    double  start_time = 0.0;
    double* sink       = nullptr;

    void start(double* out) { sink = out; start_time = now_seconds(); }
    void stop()             { if (sink) *sink += now_seconds() - start_time; sink = nullptr; }
};

// ──────────────────────────────────────────────
// Storage I/O counters
//
// write() returns when data reaches the page cache, so a timer around write()
// can report RAM speed rather than device speed. These counters come from the
// kernel and are the independent cross-check: /proc/diskstats for local block
// devices, /proc/self/mountstats for NFS (which has no diskstats entry at all).
// ──────────────────────────────────────────────

enum StorageKind {
    STORAGE_UNKNOWN = 0,
    STORAGE_BLOCK   = 1,
    STORAGE_NFS     = 2
};

const char* storage_kind_name(StorageKind kind);

struct StorageSnapshot {
    StorageKind kind  = STORAGE_UNKNOWN;
    std::string label;             // block device name, NFS export, or "unknown"
    std::string fstype;            // as reported by /proc/self/mounts
    uint64_t    sectors_written = 0;  // block: 512-byte sectors
    uint64_t    write_ios      = 0;   // block: completed write requests
    uint64_t    io_ticks_ms    = 0;   // block: ms the device spent doing I/O
    uint64_t    nfs_write_bytes = 0;  // nfs: bytes sent to server
    bool        valid = false;        // false => counters unavailable, timings only
};

// Identify the device backing `path` and snapshot its counters. `path` may be a
// directory or a file that does not exist yet; the containing directory is used.
StorageSnapshot storage_snapshot(const char* path);

struct StorageDelta {
    StorageKind kind = STORAGE_UNKNOWN;
    std::string label;
    std::string fstype;
    uint64_t    bytes_written = 0;   // sectors*512, or NFS server write bytes
    uint64_t    write_ios     = 0;
    uint64_t    io_ticks_ms   = 0;
    bool        valid = false;
};

StorageDelta storage_delta(const StorageSnapshot& before, const StorageSnapshot& after);

// ──────────────────────────────────────────────
// Per-run measurements
// ──────────────────────────────────────────────

struct RunMetrics {
    // Stage wall-clock, seconds. d2h and write are measured separately; under
    // overlap they run concurrently, so d2h + write can exceed write_wall.
    double alloc      = 0.0;   // device allocation + memset, before timing starts
    double table1     = 0.0;   // host-observed Table1 stage (launch + sync)
    double sort       = 0.0;   // host-observed sort+match stage (launch + sync)
    double d2h        = 0.0;   // summed device-to-host copy time
    double write      = 0.0;   // summed write() time
    double fsync      = 0.0;   // fdatasync
    double write_wall = 0.0;   // wall time of the whole output stage
    double teardown   = 0.0;
    double total      = 0.0;

    // Kernel-only time from GPU events; -1 when the backend could not supply it.
    double table1_kernel = -1.0;
    double sort_kernel   = -1.0;

    uint64_t bytes_written = 0;

    // Populated only with --stats (one extra counter readback, no timing impact).
    bool     have_stats = false;
    uint64_t t1_hashed            = 0;  // records that hashed into a bucket
    uint64_t t1_stored            = 0;  // records that fit
    uint64_t t1_dropped           = 0;  // lost to bucket overflow
    uint64_t t1_buckets_overflow  = 0;
    uint32_t t1_max_occupancy     = 0;
    uint64_t t2_matched           = 0;  // pairs found
    uint64_t t2_stored            = 0;  // pairs that fit
    uint64_t t2_dropped           = 0;
    uint64_t t2_buckets_overflow  = 0;
    uint32_t t2_max_occupancy     = 0;
    double   storage_efficiency   = 0.0; // t2_stored / N
};

// Identity of a run: everything needed to tell two CSV rows apart.
struct RunContext {
    int         K            = 0;
    int         device       = 0;
    const char* backend      = "unknown";   // "cuda" | "sycl"
    const char* device_name  = "unknown";
    const char* driver       = "unknown";
    const char* plot_id_hex  = "unknown";
    const char* sort_name    = VAULTX_SORT_NAME;
    size_t      chunk_bytes  = 0;
    bool        o_direct     = false;
    bool        overlap      = false;
    int         run_index    = 0;
};

// Stable schema. Keep these two in sync; downstream plotting reads the header.
void print_csv_header(FILE* out);
void print_csv_row(FILE* out, const RunContext& rc, const RunMetrics& m,
                   const StorageDelta& io);

// Fold host-side copies of the bucket counters into m. Shared by both backends
// so the reported statistics cannot drift between them. Counters are unclamped
// occupancies: values above records_per_bucket mean that many records wanted the
// bucket and the excess was dropped. Pass t1_counters = nullptr to skip Table1.
void compute_bucket_stats(const uint32_t* t1_counters, const uint32_t* t2_counters,
                          uint32_t records_per_bucket, uint64_t N, RunMetrics& m);

// Human-readable summary of the --stats counters.
void print_stats_summary(FILE* out, int K, const RunMetrics& m);

#endif // VAULTXGPU_METRICS_H
