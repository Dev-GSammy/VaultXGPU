#ifndef VAULTXGPU_GPU_CONTEXT_SYCL_H
#define VAULTXGPU_GPU_CONTEXT_SYCL_H

#include "../common/globals.h"
#include <cstdint>
#include <cstddef>
#include <sycl/sycl.hpp>

// GPU context for SYCL backend
struct SyclGPUContext {
    // Device info
    int    device_id;
    char   device_name[256];
    size_t total_mem;
    size_t free_mem;

    // Problem parameters
    int      K;
    uint64_t N;                  // 2^K
    uint32_t records_per_bucket; // N / TOTAL_BUCKETS

    // Key words (device-accessible)
    uint32_t key_words[8];

    // SYCL queue
    sycl::queue* q;

    // Device pointers (USM allocations)
    MemoRecord*       d_table1;
    uint32_t*         d_table1_counters;
    MemoTable2Record* d_table2;
    uint32_t*         d_table2_counters;
    uint32_t*         d_key_words;  // key on device

};

// ──────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────

size_t gpu_query_device(int device_id);
void   gpu_print_device_info(int device_id);
int    gpu_init(SyclGPUContext& ctx, int K, const uint32_t* key_words, int device_id);
void   gpu_generate_table1(SyclGPUContext& ctx);
void   gpu_free_table1(SyclGPUContext& ctx);
void   gpu_sort_and_match(SyclGPUContext& ctx);
int    gpu_write_table2(SyclGPUContext& ctx, int K, const uint8_t* plot_id, const char* output_dir);
void   gpu_cleanup(SyclGPUContext& ctx);

#endif // VAULTXGPU_GPU_CONTEXT_SYCL_H
