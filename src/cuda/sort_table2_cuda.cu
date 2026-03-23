#include "gpu_context_cuda.h"
#include "../blake3/blake3_common.h"
#include <cuda_runtime.h>
#include <cstdio>

// Key in constant memory (defined in gpu_context_cuda.cu)
extern __constant__ uint32_t d_key_words[8];


// Compute big-endian uint64 from first 8 bytes (or fewer)
__device__ inline uint64_t device_hash_to_uint64(const uint8_t* hash, int len) {
    uint64_t result = 0;
    int n = len < 8 ? len : 8;
    for (int i = 0; i < n; i++) {
        result = (result << 8) | hash[i];
    }
    return result;
}

// Compute bucket index from hash prefix (big-endian)
__device__ inline uint32_t device_getBucketIndex(const uint8_t* hash) {
    uint32_t idx = 0;
    for (int i = 0; i < PREFIX_SIZE && i < HASH_SIZE; i++) {
        idx = (idx << 8) | hash[i];
    }
    return idx;
}

// ──────────────────────────────────────────────
// Sort + Match kernel
//
// One block per bucket (2^24 blocks total).
// Each block:
//   1. Loads nonces from global memory into shared memory
//   2. Recomputes hashes
//   3. Insertion sort by hash value
//   4. Pairwise match finding
//   5. Writes Table2 records to global memory
// ──────────────────────────────────────────────

// Maximum records per bucket we support in shared memory.
// For k=27 RPB=8, k=28 RPB=16, k=29 RPB=32, k=30 RPB=64, k=31 RPB=128.
// We size shared mem dynamically.

__global__ void sort_and_match_kernel(
    const MemoRecord* __restrict__ bucket_storage,
    const uint32_t*   __restrict__ bucket_counters,
    MemoTable2Record* __restrict__ table2_output,
    uint32_t*         __restrict__ table2_counters,
    uint32_t records_per_bucket,
    uint64_t expected_distance,
    int K
) {
    uint32_t bucket_idx = blockIdx.x;

    // How many records are actually in this bucket
    uint32_t count = bucket_counters[bucket_idx];
    if (count <= 1) return;
    if (count > records_per_bucket) count = records_per_bucket;

    // Shared memory layout:
    //   nonces:  count * NONCE_SIZE bytes
    //   hashes:  count * HASH_SIZE bytes
    //   hash64:  count * 8 bytes (for sort comparison)
    extern __shared__ uint8_t smem[];
    uint8_t*  s_nonces = smem;
    uint8_t*  s_hashes = smem + records_per_bucket * NONCE_SIZE;
    uint64_t* s_hash64 = reinterpret_cast<uint64_t*>(
        smem + records_per_bucket * (NONCE_SIZE + HASH_SIZE));

    uint64_t base_offset = static_cast<uint64_t>(bucket_idx) * records_per_bucket;

    // 1. Load nonces from global memory
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        const uint8_t* src = bucket_storage[base_offset + i].nonce;
        uint8_t* dst = s_nonces + i * NONCE_SIZE;
        for (int b = 0; b < NONCE_SIZE; b++) {
            dst[b] = src[b];
        }
    }
    __syncthreads();

    // 2. Recompute hashes
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        blake3_keyed_hash(
            s_nonces + i * NONCE_SIZE,
            NONCE_SIZE,
            d_key_words,
            s_hashes + i * HASH_SIZE,
            HASH_SIZE
        );
        s_hash64[i] = device_hash_to_uint64(s_hashes + i * HASH_SIZE, HASH_SIZE);
    }
    __syncthreads();

    // 3. Insertion sort by hash64 (single-threaded per block -- count is small)
    //    Only thread 0 sorts.
    if (threadIdx.x == 0) {
        for (uint32_t i = 1; i < count; i++) {
            uint64_t key_val = s_hash64[i];
            // Save nonce and hash for element i
            uint8_t tmp_nonce[NONCE_SIZE];
            uint8_t tmp_hash[HASH_SIZE];
            for (int b = 0; b < NONCE_SIZE; b++)
                tmp_nonce[b] = s_nonces[i * NONCE_SIZE + b];
            for (int b = 0; b < HASH_SIZE; b++)
                tmp_hash[b] = s_hashes[i * HASH_SIZE + b];

            int j = static_cast<int>(i) - 1;
            while (j >= 0 && s_hash64[j] > key_val) {
                // Shift element j to j+1
                s_hash64[j + 1] = s_hash64[j];
                for (int b = 0; b < NONCE_SIZE; b++)
                    s_nonces[(j + 1) * NONCE_SIZE + b] = s_nonces[j * NONCE_SIZE + b];
                for (int b = 0; b < HASH_SIZE; b++)
                    s_hashes[(j + 1) * HASH_SIZE + b] = s_hashes[j * HASH_SIZE + b];
                j--;
            }
            // Insert element i at position j+1
            s_hash64[j + 1] = key_val;
            for (int b = 0; b < NONCE_SIZE; b++)
                s_nonces[(j + 1) * NONCE_SIZE + b] = tmp_nonce[b];
            for (int b = 0; b < HASH_SIZE; b++)
                s_hashes[(j + 1) * HASH_SIZE + b] = tmp_hash[b];
        }
    }
    __syncthreads();

    // 4. Pairwise match finding
    //    Each thread handles a subset of i values.
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        uint64_t hash_i = s_hash64[i];

        for (uint32_t j = i + 1; j < count; j++) {
            uint64_t hash_j = s_hash64[j];
            uint64_t distance = hash_j - hash_i; // sorted, so hash_j >= hash_i

            if (distance > expected_distance) break;

            // Compute Table2 hash: blake3_keyed_hash(nonce_i || nonce_j, key)
            uint8_t pair_input[NONCE_SIZE * 2];
            for (int b = 0; b < NONCE_SIZE; b++) {
                pair_input[b] = s_nonces[i * NONCE_SIZE + b];
                pair_input[NONCE_SIZE + b] = s_nonces[j * NONCE_SIZE + b];
            }

            uint8_t t2_hash[HASH_SIZE];
            blake3_keyed_hash(pair_input, NONCE_SIZE * 2, d_key_words, t2_hash, HASH_SIZE);

            uint32_t t2_bucket = device_getBucketIndex(t2_hash);
            uint32_t slot = atomicAdd(&table2_counters[t2_bucket], 1u);

            if (slot < records_per_bucket) {
                uint64_t t2_offset = static_cast<uint64_t>(t2_bucket) * records_per_bucket + slot;
                for (int b = 0; b < NONCE_SIZE; b++) {
                    table2_output[t2_offset].nonce1[b] = s_nonces[i * NONCE_SIZE + b];
                    table2_output[t2_offset].nonce2[b] = s_nonces[j * NONCE_SIZE + b];
                }
            } else {
                atomicMin(&table2_counters[t2_bucket], records_per_bucket);
            }
        }
    }
}

// Launch wrapper


void gpu_sort_and_match(CudaGPUContext& ctx) {
    double matching_factor = get_matching_factor(ctx.K);
    uint64_t expected_distance = static_cast<uint64_t>(
        static_cast<double>(1ULL << (64 - ctx.K)) * (1.0 / matching_factor));

    // Threads per block: 32 is enough for small buckets, scale up for larger
    int threads_per_block = 32;
    if (ctx.records_per_bucket > 32)  threads_per_block = 64;
    if (ctx.records_per_bucket > 64)  threads_per_block = 128;
    if (ctx.records_per_bucket > 128) threads_per_block = 256;

    // Shared memory per block:
    //   nonces: RPB * NONCE_SIZE
    //   hashes: RPB * HASH_SIZE
    //   hash64: RPB * 8
    size_t smem_size = static_cast<size_t>(ctx.records_per_bucket) *
                       (NONCE_SIZE + HASH_SIZE + sizeof(uint64_t));

    printf("Sort+Match: %u buckets, RPB=%u, threads/block=%d, smem=%zu bytes, "
           "expected_distance=%llu\n",
           (uint32_t)TOTAL_BUCKETS, ctx.records_per_bucket, threads_per_block,
           smem_size, (unsigned long long)expected_distance);

    sort_and_match_kernel<<<TOTAL_BUCKETS, threads_per_block, smem_size>>>(
        ctx.d_table1,
        ctx.d_table1_counters,
        ctx.d_table2,
        ctx.d_table2_counters,
        ctx.records_per_bucket,
        expected_distance,
        ctx.K
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Sort+Match kernel error: %s\n", cudaGetErrorString(err));
    }
}
