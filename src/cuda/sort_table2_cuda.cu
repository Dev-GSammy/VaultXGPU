#include "gpu_context_cuda.h"
#include "../blake3/blake3_common.h"
#include "../common/sort_net.h"
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
//   1. Loads its bucket's nonces from global memory into shared memory
//   2. Recomputes each nonce's hash (Table1 stores nonces only)
//   3. Sorts the bucket by hash -- see VAULTX_SORT in globals.h
//   4. Walks sorted neighbours for matches within expected_distance
//   5. Emits matched pairs into Table2 buckets
//
// Shared-memory layout depends on the sort mode:
//
//   VAULTX_SORT_INSERTION_RECORDS (baseline, as published)
//       [nonces: RPB*NONCE_SIZE][hashes: RPB*HASH_SIZE][keys: RPB*8]
//       The sort physically moves nonce and hash bytes on every shift.
//
//   VAULTX_SORT_INSERTION_INDEX / VAULTX_SORT_BITONIC
//       [keys: RPB*8][index: RPB*4][nonces: RPB*NONCE_SIZE]
//       Records stay put; only (key, index) pairs move. The full hash is never
//       stored -- it is consumed into the 64-bit key in registers -- which drops
//       RPB*HASH_SIZE bytes of shared memory per block and raises occupancy.
// ──────────────────────────────────────────────

#if VAULTX_SORT == VAULTX_SORT_INSERTION_RECORDS
#define VAULTX_SMEM_PER_RECORD (NONCE_SIZE + HASH_SIZE + (int)sizeof(uint64_t))
#else
#define VAULTX_SMEM_PER_RECORD ((int)sizeof(uint64_t) + (int)sizeof(uint32_t) + NONCE_SIZE)
#endif

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

    // Occupancy counters are not clamped by the producing kernel, so a bucket
    // that overflowed reports how many records wanted in. Only the first
    // records_per_bucket of them were actually stored.
    uint32_t count = bucket_counters[bucket_idx];
    if (count > records_per_bucket) count = records_per_bucket;
    if (count <= 1) return;

    extern __shared__ uint8_t smem[];
    const uint32_t rpb = records_per_bucket;

#if VAULTX_SORT == VAULTX_SORT_INSERTION_RECORDS
    uint8_t*  s_nonces = smem;
    uint8_t*  s_hashes = smem + rpb * NONCE_SIZE;
    uint64_t* s_hash64 = reinterpret_cast<uint64_t*>(smem + rpb * (NONCE_SIZE + HASH_SIZE));
#else
    uint64_t* s_hash64 = reinterpret_cast<uint64_t*>(smem);
    uint32_t* s_idx    = reinterpret_cast<uint32_t*>(smem + rpb * sizeof(uint64_t));
    uint8_t*  s_nonces = smem + rpb * (sizeof(uint64_t) + sizeof(uint32_t));
#endif

    uint64_t base_offset = static_cast<uint64_t>(bucket_idx) * rpb;

    // 1. Load nonces from global memory
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        const uint8_t* src = bucket_storage[base_offset + i].nonce;
        uint8_t* dst = s_nonces + i * NONCE_SIZE;
        for (int b = 0; b < NONCE_SIZE; b++) {
            dst[b] = src[b];
        }
    }
    __syncthreads();

    // 2. Recompute hashes (Table1 stores nonces only)
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
#if VAULTX_SORT == VAULTX_SORT_INSERTION_RECORDS
        blake3_keyed_hash(s_nonces + i * NONCE_SIZE, NONCE_SIZE, d_key_words,
                          s_hashes + i * HASH_SIZE, HASH_SIZE);
        s_hash64[i] = device_hash_to_uint64(s_hashes + i * HASH_SIZE, HASH_SIZE);
#else
        uint8_t hash[HASH_SIZE];
        blake3_keyed_hash(s_nonces + i * NONCE_SIZE, NONCE_SIZE, d_key_words,
                          hash, HASH_SIZE);
        s_hash64[i] = device_hash_to_uint64(hash, HASH_SIZE);
        s_idx[i]    = i;
#endif
    }

#if VAULTX_SORT == VAULTX_SORT_BITONIC
    // Pad the tail so the network can run over the full power-of-two RPB.
    for (uint32_t i = count + threadIdx.x; i < rpb; i += blockDim.x) {
        s_hash64[i] = VAULTX_PAD_KEY;
        s_idx[i]    = i;
    }
#endif
    __syncthreads();

    // 3. Sort the bucket by hash key.
#if VAULTX_SORT == VAULTX_SORT_BITONIC
    // Block-parallel bitonic network over (key, index); see sort_net.h.
    for (uint32_t k = 2; k <= rpb; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0; j >>= 1) {
            for (uint32_t i = threadIdx.x; i < rpb; i += blockDim.x) {
                uint32_t l = i ^ j;
                if (l > i) {
                    bool ascending = ((i & k) == 0);
                    if (bitonic_should_swap(s_hash64[i], s_idx[i],
                                            s_hash64[l], s_idx[l], ascending)) {
                        uint64_t tk = s_hash64[i]; s_hash64[i] = s_hash64[l]; s_hash64[l] = tk;
                        uint32_t ti = s_idx[i];    s_idx[i]    = s_idx[l];    s_idx[l]    = ti;
                    }
                }
            }
            __syncthreads();
        }
    }
#elif VAULTX_SORT == VAULTX_SORT_INSERTION_INDEX
    // Serial insertion sort, but over (key, index) only -- isolates the cost of
    // running on one thread from the cost of moving whole records.
    if (threadIdx.x == 0) {
        for (uint32_t i = 1; i < count; i++) {
            uint64_t key_val = s_hash64[i];
            uint32_t idx_val = s_idx[i];
            int j = static_cast<int>(i) - 1;
            while (j >= 0 && s_hash64[j] > key_val) {
                s_hash64[j + 1] = s_hash64[j];
                s_idx[j + 1]    = s_idx[j];
                j--;
            }
            s_hash64[j + 1] = key_val;
            s_idx[j + 1]    = idx_val;
        }
    }
    __syncthreads();
#else
    // Baseline: serial insertion sort moving nonce and hash bytes on every shift.
    if (threadIdx.x == 0) {
        for (uint32_t i = 1; i < count; i++) {
            uint64_t key_val = s_hash64[i];
            uint8_t tmp_nonce[NONCE_SIZE];
            uint8_t tmp_hash[HASH_SIZE];
            for (int b = 0; b < NONCE_SIZE; b++)
                tmp_nonce[b] = s_nonces[i * NONCE_SIZE + b];
            for (int b = 0; b < HASH_SIZE; b++)
                tmp_hash[b] = s_hashes[i * HASH_SIZE + b];

            int j = static_cast<int>(i) - 1;
            while (j >= 0 && s_hash64[j] > key_val) {
                s_hash64[j + 1] = s_hash64[j];
                for (int b = 0; b < NONCE_SIZE; b++)
                    s_nonces[(j + 1) * NONCE_SIZE + b] = s_nonces[j * NONCE_SIZE + b];
                for (int b = 0; b < HASH_SIZE; b++)
                    s_hashes[(j + 1) * HASH_SIZE + b] = s_hashes[j * HASH_SIZE + b];
                j--;
            }
            s_hash64[j + 1] = key_val;
            for (int b = 0; b < NONCE_SIZE; b++)
                s_nonces[(j + 1) * NONCE_SIZE + b] = tmp_nonce[b];
            for (int b = 0; b < HASH_SIZE; b++)
                s_hashes[(j + 1) * HASH_SIZE + b] = tmp_hash[b];
        }
    }
    __syncthreads();
#endif

    // 4. Pairwise match finding over the sorted keys.
    //    The keys are ascending, so once the gap exceeds expected_distance no
    //    later j can match either and the scan stops.
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        uint64_t hash_i = s_hash64[i];
#if VAULTX_SORT == VAULTX_SORT_INSERTION_RECORDS
        const uint8_t* nonce_i = s_nonces + i * NONCE_SIZE;
#else
        const uint8_t* nonce_i = s_nonces + s_idx[i] * NONCE_SIZE;
#endif

        for (uint32_t j = i + 1; j < count; j++) {
            uint64_t hash_j = s_hash64[j];
            uint64_t distance = hash_j - hash_i; // sorted, so hash_j >= hash_i

            if (distance > expected_distance) break;

#if VAULTX_SORT == VAULTX_SORT_INSERTION_RECORDS
            const uint8_t* nonce_j = s_nonces + j * NONCE_SIZE;
#else
            const uint8_t* nonce_j = s_nonces + s_idx[j] * NONCE_SIZE;
#endif

            // Table2 hash: blake3_keyed_hash(nonce_i || nonce_j, key)
            uint8_t pair_input[NONCE_SIZE * 2];
            for (int b = 0; b < NONCE_SIZE; b++) {
                pair_input[b]              = nonce_i[b];
                pair_input[NONCE_SIZE + b] = nonce_j[b];
            }

            uint8_t t2_hash[HASH_SIZE];
            blake3_keyed_hash(pair_input, NONCE_SIZE * 2, d_key_words, t2_hash, HASH_SIZE);

            uint32_t t2_bucket = device_getBucketIndex(t2_hash);
            uint32_t slot = atomicAdd(&table2_counters[t2_bucket], 1u);

            // The counter is deliberately left unclamped: its final value is the
            // number of pairs that wanted this bucket, which is what --stats
            // needs to report match count and overflow loss. Only the first
            // records_per_bucket pairs are stored.
            if (slot < records_per_bucket) {
                uint64_t t2_offset = static_cast<uint64_t>(t2_bucket) * records_per_bucket + slot;
                for (int b = 0; b < NONCE_SIZE; b++) {
                    table2_output[t2_offset].nonce1[b] = nonce_i[b];
                    table2_output[t2_offset].nonce2[b] = nonce_j[b];
                }
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

    size_t smem_size = static_cast<size_t>(ctx.records_per_bucket) * VAULTX_SMEM_PER_RECORD;

    printf("Sort+Match: %u buckets, RPB=%u, threads/block=%d, smem=%zu bytes, "
           "sort=%s, expected_distance=%llu\n",
           (uint32_t)TOTAL_BUCKETS, ctx.records_per_bucket, threads_per_block,
           smem_size, VAULTX_SORT_NAME, (unsigned long long)expected_distance);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);

    sort_and_match_kernel<<<TOTAL_BUCKETS, threads_per_block, smem_size>>>(
        ctx.d_table1,
        ctx.d_table1_counters,
        ctx.d_table2,
        ctx.d_table2_counters,
        ctx.records_per_bucket,
        expected_distance,
        ctx.K
    );

    cudaEventRecord(ev_stop);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Sort+Match kernel error: %s\n", cudaGetErrorString(err));
        ctx.sort_kernel_seconds = -1.0;
    } else {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        ctx.sort_kernel_seconds = ms / 1000.0;
    }
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
}
