#include "gpu_context_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>


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

    // Allocate host-side table2 for D2H transfer
    ctx.h_table2 = new MemoTable2Record[ctx.N];
    memset(ctx.h_table2, 0, ctx.N * sizeof(MemoTable2Record));

    return 0;
}

// D2H Transfer


void gpu_get_table2(CudaGPUContext& ctx) {
    cudaMemcpy(ctx.h_table2, ctx.d_table2,
               ctx.N * sizeof(MemoTable2Record),
               cudaMemcpyDeviceToHost);
}

// Cleanup


void gpu_cleanup(CudaGPUContext& ctx) {
    if (ctx.d_table1)          cudaFree(ctx.d_table1);
    if (ctx.d_table1_counters) cudaFree(ctx.d_table1_counters);
    if (ctx.d_table2)          cudaFree(ctx.d_table2);
    if (ctx.d_table2_counters) cudaFree(ctx.d_table2_counters);
    if (ctx.h_table2)          delete[] ctx.h_table2;

    ctx.d_table1 = nullptr;
    ctx.d_table1_counters = nullptr;
    ctx.d_table2 = nullptr;
    ctx.d_table2_counters = nullptr;
    ctx.h_table2 = nullptr;
}
