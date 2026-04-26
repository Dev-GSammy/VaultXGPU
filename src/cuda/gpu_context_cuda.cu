#include "gpu_context_cuda.h"
#include "../common/plot_io.h"
#include <cuda_runtime.h>
#include <cstdio>
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

// Stream d_table2 to disk in 256 MB chunks using a pinned host staging buffer.
// This avoids allocating a full N*8-byte host copy — only 256 MB of host RAM is used
// regardless of K.

int gpu_write_table2(CudaGPUContext& ctx, int K, const uint8_t* plot_id, const char* output_dir) {
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

    MemoTable2Record* staging = nullptr;
    cudaError_t err = cudaMallocHost(&staging, staging_records * sizeof(MemoTable2Record));
    if (err != cudaSuccess || !staging) {
        // Fall back to regular malloc if pinned alloc fails
        staging = new MemoTable2Record[staging_records];
    }

    size_t total_written_bytes = 0;
    size_t done = 0;
    int rc = 0;

    while (done < ctx.N) {
        size_t batch = std::min(staging_records, ctx.N - done);
        cudaMemcpy(staging, ctx.d_table2 + done, batch * sizeof(MemoTable2Record),
                   cudaMemcpyDeviceToHost);

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
    // staging may be pinned or regular — try cudaFreeHost first
    if (cudaFreeHost(staging) != cudaSuccess)
        delete[] staging;
    close(fd);
    if (rc == 0)
        printf("Plot file written: %zu bytes\n", total_written_bytes);
    return rc;
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
