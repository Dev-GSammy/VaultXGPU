#include "memory.h"
#include "globals.h"
#include <cstdio>

//
// Peak memory = Table1 nonces + Table2 output + bucket counters
//   = N * NONCE_SIZE + N * 2 * NONCE_SIZE + 2 * TOTAL_BUCKETS * 4
//   = N * 3 * NONCE_SIZE + 128 MB
//
// Where N = 2^K, TOTAL_BUCKETS = 2^24
//

size_t estimate_gpu_memory_required(int K) {
    size_t N = 1ULL << K;
    size_t table1_size  = N * NONCE_SIZE;                     // bucket nonces
    size_t table2_size  = N * 2 * NONCE_SIZE;                 // matched nonce pairs
    size_t counters     = 2ULL * TOTAL_BUCKETS * sizeof(uint32_t); // table1 + table2 counters
    return table1_size + table2_size + counters;
}

bool can_fit_in_memory(int K, size_t available_bytes) {
    return estimate_gpu_memory_required(K) <= available_bytes;
}

int largest_k_that_fits(size_t available_bytes, int max_k) {
    for (int k = max_k; k >= 1; k--) {
        if (can_fit_in_memory(k, available_bytes)) {
            return k;
        }
    }
    return 0;
}

void print_memory_budget(int K, size_t available_bytes) {
    size_t required = estimate_gpu_memory_required(K);
    double required_mb = static_cast<double>(required) / (1024.0 * 1024.0);
    double available_mb = static_cast<double>(available_bytes) / (1024.0 * 1024.0);

    printf("K=%d, RECORD_SIZE=%d, NONCE_SIZE=%d, HASH_SIZE=%d\n",
           K, RECORD_SIZE, NONCE_SIZE, HASH_SIZE);
    printf("Memory required: %.0f MB\n", required_mb);

    if (required <= available_bytes) {
        printf("Status: FITS (%.0f MB available)\n", available_mb);
    } else {
        printf("Status: DOES NOT FIT (%.0f MB available, %.0f MB required)\n",
               available_mb, required_mb);
        int best_k = largest_k_that_fits(available_bytes);
        if (best_k > 0) {
            double best_mb = static_cast<double>(
                estimate_gpu_memory_required(best_k)) / (1024.0 * 1024.0);
            printf("Largest K that fits: K=%d (%.0f MB)\n", best_k, best_mb);
        } else {
            printf("No K value fits in available GPU memory.\n");
        }
    }
}
