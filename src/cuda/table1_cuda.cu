#include "gpu_context_cuda.h"
#include "../blake3/blake3_common.h"
#include <cuda_runtime.h>
#include <cstdio>

// Key lives in constant memory
extern __constant__ uint32_t d_key_words[8];

// ──────────────────────────────────────────────
// Table1 generation kernel
//
// Each thread processes one nonce:
//   1. Convert thread global ID to NONCE_SIZE-byte nonce (little-endian)
//   2. Hash nonce with Blake3 keyed hash
//   3. Extract bucket index from hash prefix
//   4. Atomically insert nonce into bucket
// ──────────────────────────────────────────────

__global__ void generate_table1_kernel(
    MemoRecord* __restrict__ bucket_storage,
    uint32_t*   __restrict__ bucket_counters,
    uint64_t N,
    uint32_t records_per_bucket
) {
    uint64_t gid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    // 1. Convert nonce to bytes (little-endian, matching CPU VaultX)
    uint8_t nonce_bytes[NONCE_SIZE];
    uint64_t nonce_val = gid;
    for (int i = 0; i < NONCE_SIZE; i++) {
        nonce_bytes[i] = static_cast<uint8_t>(nonce_val & 0xFF);
        nonce_val >>= 8;
    }

    // 2. Blake3 keyed hash
    uint8_t hash[HASH_SIZE];
    blake3_keyed_hash(nonce_bytes, NONCE_SIZE, d_key_words, hash, HASH_SIZE);

    // 3. Bucket index from first PREFIX_SIZE bytes (big-endian)
    uint32_t bucket_idx = 0;
    for (int i = 0; i < PREFIX_SIZE && i < HASH_SIZE; i++) {
        bucket_idx = (bucket_idx << 8) | hash[i];
    }

    // 4. Atomic insert
    uint32_t slot = atomicAdd(&bucket_counters[bucket_idx], 1u);
    if (slot < records_per_bucket) {
        uint64_t storage_idx = static_cast<uint64_t>(bucket_idx) * records_per_bucket + slot;
        for (int i = 0; i < NONCE_SIZE; i++) {
            bucket_storage[storage_idx].nonce[i] = nonce_bytes[i];
        }
    } else {
        // Cap overflow: doesn't need to be exact, just prevent unbounded growth
        atomicMin(&bucket_counters[bucket_idx], records_per_bucket);
    }
}
// Launch wrapper


void gpu_generate_table1(CudaGPUContext& ctx) {
    constexpr int BLOCK_SIZE = 256;
    int grid_size = static_cast<int>((ctx.N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Generating Table1: %llu nonces, %u RPB, grid=%d blocks=%d\n",
           (unsigned long long)ctx.N, ctx.records_per_bucket, grid_size, BLOCK_SIZE);

    generate_table1_kernel<<<grid_size, BLOCK_SIZE>>>(
        ctx.d_table1,
        ctx.d_table1_counters,
        ctx.N,
        ctx.records_per_bucket
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Table1 kernel error: %s\n", cudaGetErrorString(err));
    }
}
