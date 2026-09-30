#include "gpu_context_sycl.h"
#include "../common/plot_io.h"
#include "../common/plot_writer.h"
#include "../common/metrics.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>


// Get SYCL device by index
static sycl::device get_sycl_device(int device_id) {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) {
        // Print only once across all calls (gpu_print_device_info, gpu_query_device, gpu_init)
        static bool warned = false;
        if (!warned) {
            fprintf(stderr, "No SYCL GPU devices found.\n");
            warned = true;
        }
        return sycl::device{sycl::default_selector_v};
    }
    if (device_id >= static_cast<int>(devices.size())) {
        fprintf(stderr, "Device %d not found, using device 0.\n", device_id);
        device_id = 0;
    }
    return devices[device_id];
}

size_t gpu_query_device(int device_id) {
    auto dev   = get_sycl_device(device_id);
    size_t total = dev.get_info<sycl::info::device::global_mem_size>();
    bool is_gpu  = (dev.get_info<sycl::info::device::device_type>()
                    == sycl::info::device_type::gpu);
    if (!is_gpu) {
        // On a CPU SYCL device, all allocations share the same system RAM pool.
        // Reserve ~15% for OS/runtime overhead since global_mem_size returns total RAM,
        // not free RAM. The streaming write uses only a fixed 256 MB staging buffer
        // (no full host copy of d_table2), so no additional deduction is needed.
        return total * 85 / 100;
    }
    return total;
}

void gpu_print_device_info(int device_id) {
    auto dev = get_sycl_device(device_id);
    auto name = dev.get_info<sycl::info::device::name>();
    size_t total_mem  = dev.get_info<sycl::info::device::global_mem_size>();
    size_t max_alloc  = dev.get_info<sycl::info::device::max_mem_alloc_size>();
    bool   is_gpu     = (dev.get_info<sycl::info::device::device_type>()
                         == sycl::info::device_type::gpu);
    printf("GPU: %s (%zu MB total, %zu MB max single alloc%s)\n",
           name.c_str(),
           total_mem / (1024 * 1024),
           max_alloc / (1024 * 1024),
           is_gpu ? "" : " [CPU fallback]");
}

int gpu_init(SyclGPUContext& ctx, int K, const uint32_t* key_words, int device_id) {
    ctx.device_id = device_id;
    ctx.K = K;
    ctx.N = 1ULL << K;
    ctx.records_per_bucket = static_cast<uint32_t>(ctx.N / TOTAL_BUCKETS);

    auto dev = get_sycl_device(device_id);
    auto name = dev.get_info<sycl::info::device::name>();
    strncpy(ctx.device_name, name.c_str(), sizeof(ctx.device_name) - 1);
    ctx.device_name[sizeof(ctx.device_name) - 1] = '\0';

    ctx.total_mem = dev.get_info<sycl::info::device::global_mem_size>();
    ctx.free_mem  = ctx.total_mem; // SYCL has no direct free-mem query
    ctx.is_gpu    = (dev.get_info<sycl::info::device::device_type>()
                     == sycl::info::device_type::gpu);

    {
        auto drv = dev.get_info<sycl::info::device::driver_version>();
        snprintf(ctx.driver_version, sizeof(ctx.driver_version), "%s", drv.c_str());
    }

    // Check per-allocation limit before attempting large allocations.
    // Intel Arc and some other drivers cap each malloc_device call at
    // max_mem_alloc_size (often ~4 GB) regardless of total VRAM.
    size_t max_alloc   = dev.get_info<sycl::info::device::max_mem_alloc_size>();
    size_t table1_bytes = ctx.N * sizeof(MemoRecord);
    size_t table2_bytes = ctx.N * sizeof(MemoTable2Record);
    if (table1_bytes > max_alloc || table2_bytes > max_alloc) {
        fprintf(stderr,
            "SYCL device max_mem_alloc_size (%zu MB) is too small for K=%d.\n"
            "  Table1 needs %zu MB, Table2 needs %zu MB per allocation.\n"
            "  Try a smaller K value.\n",
            max_alloc / (1024*1024), K,
            table1_bytes / (1024*1024), table2_bytes / (1024*1024));
        return -1;
    }

    memcpy(ctx.key_words, key_words, 8 * sizeof(uint32_t));

    // Create queue
    // enable_profiling gives per-kernel device time; in_order keeps the stage
    // sequence explicit. Profiling is optional in SYCL, so the kernels tolerate
    // its absence and report -1.
    ctx.q = new sycl::queue(dev, sycl::property_list{
        sycl::property::queue::in_order{},
        sycl::property::queue::enable_profiling{}});

    // Allocate device memory (USM)
    ctx.d_table1          = sycl::malloc_device<MemoRecord>(ctx.N, *ctx.q);
    ctx.d_table1_counters = sycl::malloc_device<uint32_t>(TOTAL_BUCKETS, *ctx.q);
    ctx.d_table2          = sycl::malloc_device<MemoTable2Record>(ctx.N, *ctx.q);
    ctx.d_table2_counters = sycl::malloc_device<uint32_t>(TOTAL_BUCKETS, *ctx.q);
    ctx.d_key_words       = sycl::malloc_device<uint32_t>(8, *ctx.q);

    if (!ctx.d_table1 || !ctx.d_table1_counters || !ctx.d_table2 ||
        !ctx.d_table2_counters || !ctx.d_key_words) {
        fprintf(stderr, "SYCL device memory allocation failed.\n");
        return -1;
    }

    // Initialize
    ctx.q->memset(ctx.d_table1_counters, 0, TOTAL_BUCKETS * sizeof(uint32_t));
    ctx.q->memset(ctx.d_table2, 0, ctx.N * sizeof(MemoTable2Record));
    ctx.q->memset(ctx.d_table2_counters, 0, TOTAL_BUCKETS * sizeof(uint32_t));
    ctx.q->memcpy(ctx.d_key_words, key_words, 8 * sizeof(uint32_t));
    ctx.q->wait();

    return 0;
}

// Free Table1 after sort+match is done (reclaims N*4 bytes before the write phase)

void gpu_free_table1(SyclGPUContext& ctx) {
    if (ctx.d_table1)          { sycl::free(ctx.d_table1, *ctx.q);          ctx.d_table1 = nullptr; }
    if (ctx.d_table1_counters) { sycl::free(ctx.d_table1_counters, *ctx.q); ctx.d_table1_counters = nullptr; }
}

// ──────────────────────────────────────────────
// Output stage -- mirrors gpu_context_cuda.cu
//
// Transfer and write are separate operations with separate timers. Two staging
// buffers let the copy of chunk i overlap the write of chunk i-1.
//
// Divergence from CUDA worth noting for the portability table: there is no
// portable pinned-host-allocation call, so the staging buffers come from
// sycl::malloc_host (which the runtime may or may not pin) and fall back to
// aligned_alloc. Copy time is taken from queue event profiling rather than from
// a pair of recorded events.
// ──────────────────────────────────────────────

int gpu_write_table2(SyclGPUContext& ctx, int K, const uint8_t* plot_id,
                     const char* output_dir, const PlotWriterConfig& cfg,
                     RunMetrics& m) {
    char filepath[512];
    build_plot_path(filepath, sizeof(filepath), output_dir, K, plot_id);

    const size_t   record_bytes = sizeof(MemoTable2Record);
    const uint64_t total_bytes  = ctx.N * record_bytes;

    const size_t align = plot_writer_alignment();
    size_t chunk_bytes = cfg.chunk_bytes;
    if (chunk_bytes < align) chunk_bytes = align;
    chunk_bytes = (chunk_bytes / align) * align;
    if (static_cast<uint64_t>(chunk_bytes) > total_bytes)
        chunk_bytes = static_cast<size_t>(total_bytes);
    const size_t chunk_records = chunk_bytes / record_bytes;

    // aligned_alloc requires a size that is a multiple of the alignment, and
    // chunk_bytes was clamped to total_bytes, which need not be.
    const size_t staging_bytes = ((chunk_bytes + align - 1) / align) * align;

    PlotWriterConfig wcfg = cfg;
    wcfg.chunk_bytes = chunk_bytes;

    PlotWriter writer;
    if (writer.open(filepath, total_bytes, wcfg) != 0) return -1;
    printf("Writing plot file: %s (chunk=%zu MB, o_direct=%s, overlap=%s)\n",
           filepath, chunk_bytes / (1024 * 1024),
           writer.o_direct_effective() ? "yes" : "no",
           cfg.overlap ? "yes" : "no");

    const int NBUF = cfg.overlap ? 2 : 1;
    MemoTable2Record* staging[2] = {nullptr, nullptr};
    bool host_usm = true;
    for (int i = 0; i < NBUF; i++) {
        staging[i] = static_cast<MemoTable2Record*>(
            sycl::aligned_alloc_host(align, staging_bytes, *ctx.q));
        if (!staging[i]) { host_usm = false; break; }
    }
    if (!host_usm) {
        for (int i = 0; i < NBUF; i++) {
            if (staging[i]) { sycl::free(staging[i], *ctx.q); staging[i] = nullptr; }
        }
        for (int i = 0; i < NBUF; i++) {
            staging[i] = static_cast<MemoTable2Record*>(aligned_alloc(align, staging_bytes));
            if (!staging[i]) {
                fprintf(stderr, "Error: could not allocate %zu B staging buffer\n", staging_bytes);
                for (int j = 0; j < i; j++) free(staging[j]);
                writer.finish();
                return -1;
            }
        }
        fprintf(stderr, "Warning: host USM allocation failed; using heap staging.\n");
    }

    const double wall_start = now_seconds();
    size_t done = 0;
    int    rc   = 0;
    int    buf  = 0;

    while (done < ctx.N) {
        size_t batch = std::min(chunk_records, ctx.N - done);
        size_t bytes = batch * record_bytes;

        double copy_seconds = -1.0;
        try {
            sycl::event ev = ctx.q->memcpy(staging[buf], ctx.d_table2 + done, bytes);
            ev.wait_and_throw();
            try {
                uint64_t t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
                uint64_t t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
                copy_seconds = static_cast<double>(t1 - t0) * 1e-9;
            } catch (const sycl::exception&) {
                // Profiling unsupported for this copy; fall back below.
            }
        } catch (const sycl::exception& e) {
            fprintf(stderr, "Error copying Table2 chunk at record %zu: %s\n",
                    done, e.what());
            rc = -1;
            break;
        }
        if (copy_seconds < 0.0) {
            // No device timestamps: charge nothing rather than charge wall time,
            // which under overlap would double-count the write.
            copy_seconds = 0.0;
            static bool warned = false;
            if (!warned) {
                fprintf(stderr, "Warning: queue profiling unavailable; d2h_s will read 0.\n");
                warned = true;
            }
        }
        m.d2h += copy_seconds;

        if (writer.submit(staging[buf], bytes) != 0) { rc = -1; break; }

        done += batch;
        buf = (buf + 1) % NBUF;
    }

    if (writer.finish() != 0) rc = -1;

    m.write      += writer.write_seconds();
    m.fsync      += writer.fsync_seconds();
    m.bytes_written = writer.bytes_written();
    m.write_wall += now_seconds() - wall_start;

    for (int i = 0; i < NBUF; i++) {
        if (!staging[i]) continue;
        if (host_usm) sycl::free(staging[i], *ctx.q);
        else          free(staging[i]);
    }

    if (rc == 0)
        printf("Plot file written: %llu bytes\n", (unsigned long long)m.bytes_written);
    return rc;
}

// Counter readback for --stats


void gpu_read_stats(SyclGPUContext& ctx, RunMetrics& m) {
    const size_t bytes = TOTAL_BUCKETS * sizeof(uint32_t);

    uint32_t* h_t1 = nullptr;
    uint32_t* h_t2 = static_cast<uint32_t*>(malloc(bytes));
    if (ctx.d_table1_counters) h_t1 = static_cast<uint32_t*>(malloc(bytes));

    if (!h_t2 || (ctx.d_table1_counters && !h_t1)) {
        fprintf(stderr, "Warning: could not allocate %zu B for counter readback; "
                        "skipping --stats.\n", bytes);
        free(h_t1); free(h_t2);
        return;
    }

    bool ok = true;
    try {
        if (h_t1) ctx.q->memcpy(h_t1, ctx.d_table1_counters, bytes);
        ctx.q->memcpy(h_t2, ctx.d_table2_counters, bytes);
        ctx.q->wait_and_throw();
    } catch (const sycl::exception& e) {
        fprintf(stderr, "Warning: counter readback failed (%s); skipping --stats.\n", e.what());
        ok = false;
    }

    if (ok) compute_bucket_stats(h_t1, h_t2, ctx.records_per_bucket, ctx.N, m);

    free(h_t1);
    free(h_t2);
}

// Cleanup


void gpu_cleanup(SyclGPUContext& ctx) {
    if (ctx.d_table1)          sycl::free(ctx.d_table1, *ctx.q);
    if (ctx.d_table1_counters) sycl::free(ctx.d_table1_counters, *ctx.q);
    if (ctx.d_table2)          sycl::free(ctx.d_table2, *ctx.q);
    if (ctx.d_table2_counters) sycl::free(ctx.d_table2_counters, *ctx.q);
    if (ctx.d_key_words)       sycl::free(ctx.d_key_words, *ctx.q);

    delete ctx.q;

    ctx.d_table1 = nullptr;
    ctx.d_table1_counters = nullptr;
    ctx.d_table2 = nullptr;
    ctx.d_table2_counters = nullptr;
    ctx.d_key_words = nullptr;
    ctx.q = nullptr;
}
