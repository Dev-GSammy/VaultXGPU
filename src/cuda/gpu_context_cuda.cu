#include "gpu_context_cuda.h"
#include "../common/plot_io.h"
#include "../common/plot_writer.h"
#include "../common/metrics.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>


// Key in constant memory (broadcast to all threads)
// Defined here; other .cu files use extern to reference it.

__constant__ uint32_t d_key_words[8];


#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            return -1;                                                   \
        }                                                                \
    } while (0)

#define CUDA_CHECK_VOID(call)                                            \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            return;                                                      \
        }                                                                \
    } while (0)


// Device query


size_t gpu_query_device(int device_id) {
    cudaSetDevice(device_id);
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

void gpu_print_device_info(int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    size_t free_mem = 0, total_mem = 0;
    cudaSetDevice(device_id);
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("GPU: %s (%zu MB total, %zu MB available)\n",
           prop.name, total_mem / (1024 * 1024), free_mem / (1024 * 1024));
}

// Initialize


int gpu_init(CudaGPUContext& ctx, int K, const uint32_t* key_words, int device_id) {
    ctx.device_id = device_id;
    ctx.K = K;
    ctx.N = 1ULL << K;
    ctx.records_per_bucket = static_cast<uint32_t>(ctx.N / TOTAL_BUCKETS);

    CUDA_CHECK(cudaSetDevice(device_id));

    // Get device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    strncpy(ctx.device_name, prop.name, sizeof(ctx.device_name) - 1);
    ctx.device_name[sizeof(ctx.device_name) - 1] = '\0';

    CUDA_CHECK(cudaMemGetInfo(&ctx.free_mem, &ctx.total_mem));

    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    snprintf(ctx.driver_version, sizeof(ctx.driver_version), "%d.%d/rt%d.%d",
             driver / 1000, (driver % 1000) / 10, runtime / 1000, (runtime % 1000) / 10);

    // Copy key words to host context and device constant memory
    memcpy(ctx.key_words, key_words, 8 * sizeof(uint32_t));
    CUDA_CHECK(cudaMemcpyToSymbol(d_key_words, key_words, 8 * sizeof(uint32_t)));

    // Allocate Table1
    CUDA_CHECK(cudaMalloc(&ctx.d_table1, ctx.N * sizeof(MemoRecord)));
    CUDA_CHECK(cudaMalloc(&ctx.d_table1_counters, TOTAL_BUCKETS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(ctx.d_table1_counters, 0, TOTAL_BUCKETS * sizeof(uint32_t)));

    // Allocate Table2
    CUDA_CHECK(cudaMalloc(&ctx.d_table2, ctx.N * sizeof(MemoTable2Record)));
    CUDA_CHECK(cudaMalloc(&ctx.d_table2_counters, TOTAL_BUCKETS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(ctx.d_table2, 0, ctx.N * sizeof(MemoTable2Record)));
    CUDA_CHECK(cudaMemset(ctx.d_table2_counters, 0, TOTAL_BUCKETS * sizeof(uint32_t)));

    return 0;
}

// Free Table1 after sort+match (reclaims VRAM before the write phase)

void gpu_free_table1(CudaGPUContext& ctx) {
    if (ctx.d_table1)          { cudaFree(ctx.d_table1);          ctx.d_table1 = nullptr; }
    if (ctx.d_table1_counters) { cudaFree(ctx.d_table1_counters); ctx.d_table1_counters = nullptr; }
}

// ──────────────────────────────────────────────
// Output stage
//
// Device-to-host transfer and disk write are separate operations with separate
// timers. Two pinned staging buffers let the D2H copy of chunk i run while
// PlotWriter's thread writes chunk i-1, so with cfg.overlap the stage costs
// max(PCIe, disk) rather than PCIe + disk. D2H is timed with CUDA events, which
// measures the copy itself and is unaffected by what the filesystem is doing.
// ──────────────────────────────────────────────

int gpu_write_table2(CudaGPUContext& ctx, int K, const uint8_t* plot_id,
                     const char* output_dir, const PlotWriterConfig& cfg,
                     RunMetrics& m) {
    char filepath[512];
    build_plot_path(filepath, sizeof(filepath), output_dir, K, plot_id);

    const size_t   record_bytes = sizeof(MemoTable2Record);
    const uint64_t total_bytes  = ctx.N * record_bytes;

    // Chunk size must hold whole records and, under O_DIRECT, be a multiple of
    // the alignment. Round down, never to zero.
    const size_t align = plot_writer_alignment();
    size_t chunk_bytes = cfg.chunk_bytes;
    if (chunk_bytes < align) chunk_bytes = align;
    chunk_bytes = (chunk_bytes / align) * align;
    if (static_cast<uint64_t>(chunk_bytes) > total_bytes)
        chunk_bytes = static_cast<size_t>(total_bytes);
    const size_t chunk_records = chunk_bytes / record_bytes;

    // Allocation size is rounded up to the alignment: aligned_alloc requires the
    // size to be a multiple of it, and chunk_bytes was clamped to total_bytes,
    // which need not be.
    const size_t staging_bytes = ((chunk_bytes + align - 1) / align) * align;

    PlotWriterConfig wcfg = cfg;
    wcfg.chunk_bytes = chunk_bytes;

    PlotWriter writer;
    if (writer.open(filepath, total_bytes, wcfg) != 0) return -1;
    printf("Writing plot file: %s (chunk=%zu MB, o_direct=%s, overlap=%s)\n",
           filepath, chunk_bytes / (1024 * 1024),
           writer.o_direct_effective() ? "yes" : "no",
           cfg.overlap ? "yes" : "no");

    // Two pinned buffers so the caller can refill one while the other is written.
    // cudaMallocHost is page-aligned, which is what O_DIRECT requires.
    const int NBUF = cfg.overlap ? 2 : 1;
    MemoTable2Record* staging[2] = {nullptr, nullptr};
    bool pinned = true;
    for (int i = 0; i < NBUF; i++) {
        if (cudaMallocHost(&staging[i], staging_bytes) != cudaSuccess || !staging[i]) {
            pinned = false;
            break;
        }
    }
    if (!pinned) {
        // Fall back to aligned host memory; slower D2H but still O_DIRECT-safe.
        for (int i = 0; i < NBUF; i++) {
            if (staging[i]) { cudaFreeHost(staging[i]); staging[i] = nullptr; }
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
        fprintf(stderr, "Warning: pinned staging allocation failed; using pageable memory.\n");
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t ev_a, ev_b;
    cudaEventCreate(&ev_a);
    cudaEventCreate(&ev_b);

    const double wall_start = now_seconds();
    size_t done = 0;
    int    rc   = 0;
    int    buf  = 0;

    while (done < ctx.N) {
        size_t batch = std::min(chunk_records, ctx.N - done);
        size_t bytes = batch * record_bytes;

        cudaEventRecord(ev_a, stream);
        cudaError_t cerr = cudaMemcpyAsync(staging[buf], ctx.d_table2 + done, bytes,
                                           cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(ev_b, stream);
        if (cerr == cudaSuccess) cerr = cudaStreamSynchronize(stream);
        if (cerr != cudaSuccess) {
            fprintf(stderr, "Error copying Table2 chunk at record %zu: %s\n",
                    done, cudaGetErrorString(cerr));
            rc = -1;
            break;
        }

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_a, ev_b);
        m.d2h += ms / 1000.0;

        // Returns once the previously submitted buffer has drained, so the other
        // buffer is free for the next copy.
        if (writer.submit(staging[buf], bytes) != 0) { rc = -1; break; }

        done += batch;
        buf = (buf + 1) % NBUF;
    }

    if (writer.finish() != 0) rc = -1;

    m.write      += writer.write_seconds();
    m.fsync      += writer.fsync_seconds();
    m.bytes_written = writer.bytes_written();
    m.write_wall += now_seconds() - wall_start;

    cudaEventDestroy(ev_a);
    cudaEventDestroy(ev_b);
    cudaStreamDestroy(stream);
    for (int i = 0; i < NBUF; i++) {
        if (!staging[i]) continue;
        if (pinned) cudaFreeHost(staging[i]);
        else        free(staging[i]);
    }

    if (rc == 0)
        printf("Plot file written: %llu bytes\n", (unsigned long long)m.bytes_written);
    return rc;
}

// Counter readback for --stats


void gpu_read_stats(CudaGPUContext& ctx, RunMetrics& m) {
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
    if (h_t1 && cudaMemcpy(h_t1, ctx.d_table1_counters, bytes,
                           cudaMemcpyDeviceToHost) != cudaSuccess) ok = false;
    if (cudaMemcpy(h_t2, ctx.d_table2_counters, bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess) ok = false;

    if (ok) {
        compute_bucket_stats(h_t1, h_t2, ctx.records_per_bucket, ctx.N, m);
    } else {
        fprintf(stderr, "Warning: counter readback failed; skipping --stats.\n");
    }

    free(h_t1);
    free(h_t2);
}

// Cleanup

void gpu_cleanup(CudaGPUContext& ctx) {
    if (ctx.d_table1)          cudaFree(ctx.d_table1);
    if (ctx.d_table1_counters) cudaFree(ctx.d_table1_counters);
    if (ctx.d_table2)          cudaFree(ctx.d_table2);
    if (ctx.d_table2_counters) cudaFree(ctx.d_table2_counters);

    ctx.d_table1 = nullptr;
    ctx.d_table1_counters = nullptr;
    ctx.d_table2 = nullptr;
    ctx.d_table2_counters = nullptr;
}
