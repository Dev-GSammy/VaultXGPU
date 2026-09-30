#include "metrics.h"

#include <cstring>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/sysmacros.h>   // major(), minor()
#include <unistd.h>
#include <limits.h>

double now_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

const char* storage_kind_name(StorageKind kind) {
    switch (kind) {
        case STORAGE_BLOCK: return "block";
        case STORAGE_NFS:   return "nfs";
        default:            return "unknown";
    }
}

// ──────────────────────────────────────────────
// Mount lookup
// ──────────────────────────────────────────────

namespace {

// Directory component of `path`, or "." — works whether path names a directory
// or a file that does not exist yet.
std::string dir_of(const char* path) {
    struct stat st;
    if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) return std::string(path);
    std::string p(path ? path : ".");
    size_t slash = p.find_last_of('/');
    if (slash == std::string::npos) return std::string(".");
    if (slash == 0)                 return std::string("/");
    return p.substr(0, slash);
}

struct MountEntry {
    std::string source;
    std::string target;
    std::string fstype;
};

// The mount whose target is the longest prefix of `path`.
bool find_mount(const std::string& path, MountEntry& out) {
    FILE* f = fopen("/proc/self/mounts", "r");
    if (!f) return false;

    char resolved[PATH_MAX];
    std::string want = realpath(path.c_str(), resolved) ? std::string(resolved) : path;

    bool found = false;
    size_t best = 0;
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        char src[1024], tgt[1024], fst[256];
        if (sscanf(line, "%1023s %1023s %255s", src, tgt, fst) != 3) continue;
        std::string target(tgt);
        // Prefix match on whole path components only, so /var does not match /variable.
        if (want.compare(0, target.size(), target) != 0) continue;
        if (target != "/" && want.size() > target.size() && want[target.size()] != '/') continue;
        if (target.size() >= best) {
            best = target.size();
            out.source = src;
            out.target = target;
            out.fstype = fst;
            found = true;
        }
    }
    fclose(f);
    return found;
}

// /proc/diskstats fields after "major minor name":
//   1 reads  2 reads_merged  3 sectors_read  4 ms_reading
//   5 writes 6 writes_merged 7 sectors_written 8 ms_writing
//   9 in_flight 10 io_ticks_ms 11 weighted_ms
bool read_diskstats(unsigned major_want, unsigned minor_want, StorageSnapshot& snap) {
    FILE* f = fopen("/proc/diskstats", "r");
    if (!f) return false;

    bool found = false;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        unsigned maj = 0, min = 0;
        char name[256];
        unsigned long long v[11] = {0};
        int n = sscanf(line,
            "%u %u %255s %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu",
            &maj, &min, name,
            &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8], &v[9], &v[10]);
        if (n < 11) continue;
        if (maj != major_want || min != minor_want) continue;

        snap.label           = name;
        snap.sectors_written = v[6];
        snap.write_ios       = v[4];
        snap.io_ticks_ms     = (n >= 13) ? v[9] : 0;
        found = true;
        break;
    }
    fclose(f);
    return found;
}

// /proc/self/mountstats, per NFS mount:
//   device <src> mounted on <target> with fstype nfs...
//   ...
//   bytes: normalread normalwrite directread directwrite serverread serverwrite ...
bool read_mountstats_nfs(const std::string& target, StorageSnapshot& snap) {
    FILE* f = fopen("/proc/self/mountstats", "r");
    if (!f) return false;

    bool in_section = false, found = false;
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "device ", 7) == 0) {
            char src[1024], tgt[1024];
            in_section = false;
            if (sscanf(line, "device %1023s mounted on %1023s", src, tgt) == 2) {
                if (target == tgt) {
                    in_section = true;
                    snap.label = src;
                }
            }
            continue;
        }
        if (!in_section) continue;

        const char* p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (strncmp(p, "bytes:", 6) != 0) continue;

        unsigned long long b[8] = {0};
        if (sscanf(p + 6, "%llu %llu %llu %llu %llu %llu %llu %llu",
                   &b[0], &b[1], &b[2], &b[3], &b[4], &b[5], &b[6], &b[7]) >= 6) {
            // b[5] = serverwrite: bytes actually sent to the NFS server.
            snap.nfs_write_bytes = b[5];
            found = true;
        }
        break;
    }
    fclose(f);
    return found;
}

} // namespace

StorageSnapshot storage_snapshot(const char* path) {
    StorageSnapshot snap;
    std::string dir = dir_of(path);

    MountEntry mnt;
    if (find_mount(dir, mnt)) {
        snap.fstype = mnt.fstype;
    }

    // NFS has no /proc/diskstats entry on the client; its counters live in mountstats.
    if (snap.fstype.compare(0, 3, "nfs") == 0) {
        snap.kind = STORAGE_NFS;
        snap.valid = read_mountstats_nfs(mnt.target, snap);
        if (snap.label.empty()) snap.label = mnt.source;
        return snap;
    }

    struct stat st;
    if (stat(dir.c_str(), &st) != 0) {
        snap.label = "unknown";
        return snap;
    }
    if (read_diskstats(major(st.st_dev), minor(st.st_dev), snap)) {
        snap.kind  = STORAGE_BLOCK;
        snap.valid = true;
    } else {
        // tmpfs, overlayfs, a device-mapper target with no diskstats row, etc.
        snap.label = snap.fstype.empty() ? "unknown" : snap.fstype;
    }
    return snap;
}

StorageDelta storage_delta(const StorageSnapshot& before, const StorageSnapshot& after) {
    StorageDelta d;
    d.kind   = after.kind;
    d.label  = after.label;
    d.fstype = after.fstype;
    if (!before.valid || !after.valid || before.kind != after.kind) return d;

    if (after.kind == STORAGE_BLOCK) {
        d.bytes_written = (after.sectors_written - before.sectors_written) * 512ULL;
        d.write_ios     = after.write_ios   - before.write_ios;
        d.io_ticks_ms   = after.io_ticks_ms - before.io_ticks_ms;
    } else if (after.kind == STORAGE_NFS) {
        d.bytes_written = after.nfs_write_bytes - before.nfs_write_bytes;
    }
    d.valid = true;
    return d;
}

// ──────────────────────────────────────────────
// CSV
// ──────────────────────────────────────────────

void print_csv_header(FILE* out) {
    fprintf(out,
        "timestamp,host,backend,device_idx,device_name,driver,K,run,sort,"
        "chunk_bytes,o_direct,overlap,"
        "alloc_s,table1_s,table1_kernel_s,sort_s,sort_kernel_s,"
        "d2h_s,write_s,fsync_s,write_wall_s,teardown_s,total_s,"
        "bytes_written,storage_kind,storage_label,storage_fstype,"
        "dev_bytes_written,dev_write_ios,dev_io_ticks_ms,"
        "t1_hashed,t1_stored,t1_dropped,t1_buckets_overflow,t1_max_occ,"
        "t2_matched,t2_stored,t2_dropped,t2_buckets_overflow,t2_max_occ,"
        "storage_efficiency,plot_id\n");
}

void print_csv_row(FILE* out, const RunContext& rc, const RunMetrics& m,
                   const StorageDelta& io) {
    char host[256] = {0};
    if (gethostname(host, sizeof(host) - 1) != 0) strncpy(host, "unknown", sizeof(host) - 1);

    char stamp[64];
    time_t now = time(nullptr);
    struct tm tm_utc;
    gmtime_r(&now, &tm_utc);
    strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);

    fprintf(out,
        "%s,%s,%s,%d,\"%s\",%s,%d,%d,%s,"
        "%zu,%d,%d,"
        "%.6f,%.6f,%.6f,%.6f,%.6f,"
        "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
        "%llu,%s,\"%s\",%s,"
        "%llu,%llu,%llu,"
        "%llu,%llu,%llu,%llu,%u,"
        "%llu,%llu,%llu,%llu,%u,"
        "%.6f,%s\n",
        stamp, host, rc.backend, rc.device, rc.device_name, rc.driver,
        rc.K, rc.run_index, rc.sort_name,
        rc.chunk_bytes, rc.o_direct ? 1 : 0, rc.overlap ? 1 : 0,
        m.alloc, m.table1, m.table1_kernel, m.sort, m.sort_kernel,
        m.d2h, m.write, m.fsync, m.write_wall, m.teardown, m.total,
        (unsigned long long)m.bytes_written,
        storage_kind_name(io.kind),
        io.label.empty() ? "unknown" : io.label.c_str(),
        io.fstype.empty() ? "unknown" : io.fstype.c_str(),
        (unsigned long long)io.bytes_written,
        (unsigned long long)io.write_ios,
        (unsigned long long)io.io_ticks_ms,
        (unsigned long long)m.t1_hashed, (unsigned long long)m.t1_stored,
        (unsigned long long)m.t1_dropped, (unsigned long long)m.t1_buckets_overflow,
        m.t1_max_occupancy,
        (unsigned long long)m.t2_matched, (unsigned long long)m.t2_stored,
        (unsigned long long)m.t2_dropped, (unsigned long long)m.t2_buckets_overflow,
        m.t2_max_occupancy,
        m.storage_efficiency, rc.plot_id_hex);
}

void compute_bucket_stats(const uint32_t* t1_counters, const uint32_t* t2_counters,
                          uint32_t records_per_bucket, uint64_t N, RunMetrics& m) {
    const uint64_t buckets = TOTAL_BUCKETS;

    if (t1_counters) {
        uint64_t hashed = 0, stored = 0, over = 0;
        uint32_t maxocc = 0;
        for (uint64_t b = 0; b < buckets; b++) {
            uint32_t occ = t1_counters[b];
            hashed += occ;
            stored += (occ > records_per_bucket) ? records_per_bucket : occ;
            if (occ > records_per_bucket) over++;
            if (occ > maxocc) maxocc = occ;
        }
        m.t1_hashed           = hashed;
        m.t1_stored           = stored;
        m.t1_dropped          = hashed - stored;
        m.t1_buckets_overflow = over;
        m.t1_max_occupancy    = maxocc;
    }

    if (t2_counters) {
        uint64_t matched = 0, stored = 0, over = 0;
        uint32_t maxocc = 0;
        for (uint64_t b = 0; b < buckets; b++) {
            uint32_t occ = t2_counters[b];
            matched += occ;
            stored  += (occ > records_per_bucket) ? records_per_bucket : occ;
            if (occ > records_per_bucket) over++;
            if (occ > maxocc) maxocc = occ;
        }
        m.t2_matched          = matched;
        m.t2_stored           = stored;
        m.t2_dropped          = matched - stored;
        m.t2_buckets_overflow = over;
        m.t2_max_occupancy    = maxocc;
        m.storage_efficiency  = N ? (double)stored / (double)N : 0.0;
    }

    m.have_stats = true;
}

void print_stats_summary(FILE* out, int K, const RunMetrics& m) {
    if (!m.have_stats) return;

    const uint64_t N   = 1ULL << K;
    const uint32_t rpb = records_per_bucket_for(K);

    double t1_loss = m.t1_hashed ? 100.0 * (double)m.t1_dropped / (double)m.t1_hashed : 0.0;
    double t2_loss = m.t2_matched ? 100.0 * (double)m.t2_dropped / (double)m.t2_matched : 0.0;

    fprintf(out, "\n=== Stats (K=%d, RPB=%u, N=%llu) ===\n",
            K, rpb, (unsigned long long)N);
    fprintf(out, "Table1  hashed=%llu stored=%llu dropped=%llu (%.2f%%)\n",
            (unsigned long long)m.t1_hashed, (unsigned long long)m.t1_stored,
            (unsigned long long)m.t1_dropped, t1_loss);
    fprintf(out, "Table1  buckets overflowed=%llu of %llu (%.2f%%), max occupancy=%u\n",
            (unsigned long long)m.t1_buckets_overflow,
            (unsigned long long)TOTAL_BUCKETS,
            100.0 * (double)m.t1_buckets_overflow / (double)TOTAL_BUCKETS,
            m.t1_max_occupancy);
    fprintf(out, "Table2  matched=%llu stored=%llu dropped=%llu (%.2f%%)\n",
            (unsigned long long)m.t2_matched, (unsigned long long)m.t2_stored,
            (unsigned long long)m.t2_dropped, t2_loss);
    fprintf(out, "Table2  buckets overflowed=%llu of %llu (%.2f%%), max occupancy=%u\n",
            (unsigned long long)m.t2_buckets_overflow,
            (unsigned long long)TOTAL_BUCKETS,
            100.0 * (double)m.t2_buckets_overflow / (double)TOTAL_BUCKETS,
            m.t2_max_occupancy);
    fprintf(out, "Storage efficiency = %.4f%% of %llu slots filled\n",
            100.0 * m.storage_efficiency, (unsigned long long)N);
}
