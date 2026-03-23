#include "gpu_context_sycl.h"
#include "../blake3/blake3_common.h"
#include <cstdio>


// Same logic as the CUDA version

static inline uint64_t device_hash_to_uint64(const uint8_t* hash, int len) {
    uint64_t result = 0;
    int n = len < 8 ? len : 8;
    for (int i = 0; i < n; i++) {
        result = (result << 8) | hash[i];
    }
    return result;
}

static inline uint32_t device_getBucketIndex(const uint8_t* hash) {
    uint32_t idx = 0;
    for (int i = 0; i < PREFIX_SIZE && i < HASH_SIZE; i++) {
        idx = (idx << 8) | hash[i];
    }
    return idx;
}

// Sort + Match kernel (SYCL)
//
// One work-group per bucket.
// Uses local memory (shared memory) for sort.


void gpu_sort_and_match(SyclGPUContext& ctx) {
    double matching_factor = get_matching_factor(ctx.K);
    uint64_t expected_distance = static_cast<uint64_t>(
        static_cast<double>(1ULL << (64 - ctx.K)) * (1.0 / matching_factor));

    uint32_t rpb = ctx.records_per_bucket;
    int K = ctx.K;

    int threads_per_group = 32;
    if (rpb > 32)  threads_per_group = 64;
    if (rpb > 64)  threads_per_group = 128;
    if (rpb > 128) threads_per_group = 256;

    // Local memory size per work-group
    size_t local_nonces_size = rpb * NONCE_SIZE;
    size_t local_hashes_size = rpb * HASH_SIZE;
    size_t local_hash64_size = rpb * sizeof(uint64_t);
    size_t total_local_size = local_nonces_size + local_hashes_size + local_hash64_size;

    printf("Sort+Match: %llu buckets, RPB=%u, threads/group=%d, local_mem=%zu bytes, "
           "expected_distance=%llu\n",
           (unsigned long long)TOTAL_BUCKETS, rpb, threads_per_group,
           total_local_size, (unsigned long long)expected_distance);

    MemoRecord*       d_table1 = ctx.d_table1;
    uint32_t*         d_t1_counters = ctx.d_table1_counters;
    MemoTable2Record* d_table2 = ctx.d_table2;
    uint32_t*         d_t2_counters = ctx.d_table2_counters;
    uint32_t*         d_kw = ctx.d_key_words;

    sycl::range<1> global_range(static_cast<size_t>(TOTAL_BUCKETS) * threads_per_group);
    sycl::range<1> local_range(threads_per_group);

    ctx.q->submit([&](sycl::handler& cgh) {
        // Allocate local memory
        sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(total_local_size), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(global_range, local_range),
            [=](sycl::nd_item<1> item) {
                uint32_t bucket_idx = item.get_group(0);
                uint32_t lid = item.get_local_id(0);
                uint32_t local_size = item.get_local_range(0);

                uint32_t count = d_t1_counters[bucket_idx];
                if (count <= 1) return;
                if (count > rpb) count = rpb;

                // Local memory pointers
                uint8_t*  s_nonces = &local_mem[0];
                uint8_t*  s_hashes = s_nonces + rpb * NONCE_SIZE;
                uint64_t* s_hash64 = reinterpret_cast<uint64_t*>(
                    s_hashes + rpb * HASH_SIZE);

                uint64_t base_offset = static_cast<uint64_t>(bucket_idx) * rpb;

                // 1. Load nonces
                for (uint32_t i = lid; i < count; i += local_size) {
                    const uint8_t* src = d_table1[base_offset + i].nonce;
                    uint8_t* dst = s_nonces + i * NONCE_SIZE;
                    for (int b = 0; b < NONCE_SIZE; b++) dst[b] = src[b];
                }
                sycl::group_barrier(item.get_group());

                // 2. Recompute hashes
                for (uint32_t i = lid; i < count; i += local_size) {
                    blake3_keyed_hash(
                        s_nonces + i * NONCE_SIZE, NONCE_SIZE,
                        d_kw, s_hashes + i * HASH_SIZE, HASH_SIZE);
                    s_hash64[i] = device_hash_to_uint64(
                        s_hashes + i * HASH_SIZE, HASH_SIZE);
                }
                sycl::group_barrier(item.get_group());

                // 3. Insertion sort (thread 0 only)
                if (lid == 0) {
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
                sycl::group_barrier(item.get_group());

                // 4. Pairwise match finding
                for (uint32_t i = lid; i < count; i += local_size) {
                    uint64_t hash_i = s_hash64[i];

                    for (uint32_t j = i + 1; j < count; j++) {
                        uint64_t hash_j = s_hash64[j];
                        uint64_t distance = hash_j - hash_i;

                        if (distance > expected_distance) break;

                        uint8_t pair_input[NONCE_SIZE * 2];
                        for (int b = 0; b < NONCE_SIZE; b++) {
                            pair_input[b] = s_nonces[i * NONCE_SIZE + b];
                            pair_input[NONCE_SIZE + b] = s_nonces[j * NONCE_SIZE + b];
                        }

                        uint8_t t2_hash[HASH_SIZE];
                        blake3_keyed_hash(pair_input, NONCE_SIZE * 2,
                                          d_kw, t2_hash, HASH_SIZE);

                        uint32_t t2_bucket = device_getBucketIndex(t2_hash);

                        auto counter_ref = sycl::atomic_ref<uint32_t,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                d_t2_counters[t2_bucket]);

                        uint32_t slot = counter_ref.fetch_add(1u);
                        if (slot < rpb) {
                            uint64_t t2_offset = static_cast<uint64_t>(t2_bucket) * rpb + slot;
                            for (int b = 0; b < NONCE_SIZE; b++) {
                                d_table2[t2_offset].nonce1[b] = s_nonces[i * NONCE_SIZE + b];
                                d_table2[t2_offset].nonce2[b] = s_nonces[j * NONCE_SIZE + b];
                            }
                        } else {
                            counter_ref.fetch_min(rpb);
                        }
                    }
                }
            });
    });
    ctx.q->wait();
}
