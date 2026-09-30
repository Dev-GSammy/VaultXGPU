#ifndef VAULTXGPU_GPU_CONTEXT_CUDA_H
#define VAULTXGPU_GPU_CONTEXT_CUDA_H

#include "../common/globals.h"
#include "../common/metrics.h"
#include "../common/plot_writer.h"
#include <cstdint>
#include <cstddef>

// GPU context holding all device allocations and state
struct CudaGPUContext {
    // Device info
    int    device_id;
    char   device_name[256];
    size_t total_mem;
    size_t free_mem;

    // Problem parameters
    int      K;
    uint64_t N;                  // 2^K
    uint32_t records_per_bucket; // N / TOTAL_BUCKETS

    // Key (host + device constant memory is set separately)
    uint32_t key_words[8];

    // Device pointers -- Table1
    MemoRecord* d_table1;          // bucket_storage: N entries
    uint32_t*   d_table1_counters; // TOTAL_BUCKETS entries

    // Device pointers -- Table2
    MemoTable2Record* d_table2;          // N entries (worst-case output)
    uint32_t*         d_table2_counters; // TOTAL_BUCKETS entries

    // Kernel-only time from CUDA events, seconds. -1 when unavailable.
    double table1_kernel_seconds = -1.0;
    double sort_kernel_seconds   = -1.0;

    // Driver version string for the CSV row.
    char driver_version[64] = {0};
};


// Public API (called from main.cpp via gpu_backend.h)


// Query GPU and print info. Returns available memory in bytes.
size_t gpu_query_device(int device_id);

// Print device info string
void gpu_print_device_info(int device_id);

// Initialize context: allocate GPU memory for the given K.
// Returns 0 on success.
int gpu_init(CudaGPUContext& ctx, int K, const uint32_t* key_words, int device_id);

// Generate Table1 on GPU: hash all nonces into buckets
void gpu_generate_table1(CudaGPUContext& ctx);

// Sort each bucket + find matches + generate Table2 on GPU
void gpu_sort_and_match(CudaGPUContext& ctx);

// Free Table1 device memory after sort+match (call before write phase)
void gpu_free_table1(CudaGPUContext& ctx);

// Stream Table2 from device to disk in chunks (no full host copy).
// Fills m.d2h, m.write, m.fsync, m.write_wall and m.bytes_written.
int gpu_write_table2(CudaGPUContext& ctx, int K, const uint8_t* plot_id,
                     const char* output_dir, const PlotWriterConfig& cfg,
                     RunMetrics& m);

// Read the Table1/Table2 bucket counters back to the host and fill in the
// occupancy, overflow and storage-efficiency fields of m. Must be called before
// gpu_free_table1(). Costs two TOTAL_BUCKETS*4 B transfers, so it is gated on
// --stats and runs outside the timed stages.
void gpu_read_stats(CudaGPUContext& ctx, RunMetrics& m);

// Free all GPU allocations
void gpu_cleanup(CudaGPUContext& ctx);

#endif // VAULTXGPU_GPU_CONTEXT_CUDA_H
