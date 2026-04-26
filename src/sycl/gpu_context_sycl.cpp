#include "gpu_context_sycl.h"
#include "../common/plot_io.h"
#include <cstdio>
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
    ctx.q = new sycl::queue(dev, sycl::property::queue::in_order{});

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

// Stream d_table2 to disk in chunks — never allocates a full host copy.
// Uses a 256 MB staging buffer so peak host-RAM overhead is capped.

int gpu_write_table2(SyclGPUContext& ctx, int K, const uint8_t* plot_id, const char* output_dir) {
    char filepath[512];
    build_plot_path(filepath, sizeof(filepath), output_dir, K, plot_id);

    int fd = open(filepath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1) {
        fprintf(stderr, "Error opening file %s: %s\n", filepath, strerror(errno));
        return -1;
    }
    printf("Writing plot file: %s\n", filepath);

    constexpr size_t STAGING_BYTES = 256ULL * 1024 * 1024; // 256 MB staging buffer
    size_t staging_records = STAGING_BYTES / sizeof(MemoTable2Record);
    auto* staging = new MemoTable2Record[staging_records];

    size_t total_written_bytes = 0;
    size_t done = 0;
    int rc = 0;

    while (done < ctx.N) {
        size_t batch = std::min(staging_records, ctx.N - done);
        ctx.q->memcpy(staging, ctx.d_table2 + done, batch * sizeof(MemoTable2Record));
        ctx.q->wait();

        const char* ptr = reinterpret_cast<const char*>(staging);
        size_t to_write = batch * sizeof(MemoTable2Record);
        size_t written = 0;
        while (written < to_write) {
            ssize_t n = write(fd, ptr + written, to_write - written);
            if (n < 0) {
                fprintf(stderr, "Error writing at offset %zu: %s\n",
                        total_written_bytes + written, strerror(errno));
                rc = -1;
                goto done_write;
            }
            written += static_cast<size_t>(n);
        }
        total_written_bytes += written;
        done += batch;
    }

done_write:
    delete[] staging;
    close(fd);
    if (rc == 0)
        printf("Plot file written: %zu bytes\n", total_written_bytes);
    return rc;
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
