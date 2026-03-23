#include "gpu_context_sycl.h"
#include "../blake3/blake3_common.h"
#include <cstdio>

// ──────────────────────────────────────────────
// Table1 generation kernel (SYCL)
//
// Each work-item processes one nonce:
//   1. Convert ID to NONCE_SIZE-byte nonce
//   2. Blake3 keyed hash
//   3. Extract bucket index
//   4. Atomic insert into bucket
// ──────────────────────────────────────────────

void gpu_generate_table1(SyclGPUContext& ctx) {
    constexpr int BLOCK_SIZE = 256;
    uint64_t N = ctx.N;
    uint32_t rpb = ctx.records_per_bucket;

    MemoRecord* d_table1 = ctx.d_table1;
    uint32_t*   d_counters = ctx.d_table1_counters;
    uint32_t*   d_kw = ctx.d_key_words;

    printf("Generating Table1: %llu nonces, %u RPB\n",
           (unsigned long long)N, rpb);

    size_t global_size = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    ctx.q->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(global_size), [=](sycl::id<1> id) {
            uint64_t gid = id[0];
            if (gid >= N) return;

            // 1. Nonce to bytes (little-endian)
            uint8_t nonce_bytes[NONCE_SIZE];
            uint64_t nonce_val = gid;
            for (int i = 0; i < NONCE_SIZE; i++) {
                nonce_bytes[i] = static_cast<uint8_t>(nonce_val & 0xFF);
                nonce_val >>= 8;
            }

            // 2. Blake3 keyed hash
            uint8_t hash[HASH_SIZE];
            blake3_keyed_hash(nonce_bytes, NONCE_SIZE, d_kw, hash, HASH_SIZE);

            // 3. Bucket index
            uint32_t bucket_idx = 0;
            for (int i = 0; i < PREFIX_SIZE && i < HASH_SIZE; i++) {
                bucket_idx = (bucket_idx << 8) | hash[i];
            }

            // 4. Atomic insert
            auto counter_ref = sycl::atomic_ref<uint32_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>(d_counters[bucket_idx]);

            uint32_t slot = counter_ref.fetch_add(1u);
            if (slot < rpb) {
                uint64_t storage_idx = static_cast<uint64_t>(bucket_idx) * rpb + slot;
                for (int b = 0; b < NONCE_SIZE; b++) {
                    d_table1[storage_idx].nonce[b] = nonce_bytes[b];
                }
            } else {
                counter_ref.fetch_min(rpb);
            }
        });
    });
    ctx.q->wait();
}
