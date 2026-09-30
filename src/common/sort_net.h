#ifndef VAULTXGPU_SORT_NET_H
#define VAULTXGPU_SORT_NET_H

#include <cstdint>

#if defined(__CUDACC__)
#define VAULTX_HD __host__ __device__
#else
#define VAULTX_HD
#endif

// ──────────────────────────────────────────────
// Bucket sort network, shared by both backends
//
// Each Table2 block sorts one Table1 bucket by 64-bit hash key. Records are not
// moved: we sort (key, index) pairs and the match stage dereferences the index,
// so a swap costs 12 bytes instead of NONCE_SIZE + HASH_SIZE per element.
//
// Records-per-bucket is a power of two for every supported K (8 at K=27 through
// 256 at K=32), so the network runs over the full RPB and slots beyond the live
// `count` are padded with key = PAD_KEY and index = their own slot. Padding
// sorts to the end, leaving the live records in positions [0, count).
//
// The comparator breaks key ties by index. That is what makes the padding safe:
// a real record whose hash happens to equal PAD_KEY still sorts ahead of every
// pad entry, because pad indices are all >= count. Without the tie-break, such a
// record could be displaced past `count` and a pad entry — whose index points at
// uninitialized nonce storage — pulled into the live range.
//
// The loop itself lives in each backend because the barrier differs
// (__syncthreads vs sycl::group_barrier); it is:
//
//   for (k = 2; k <= n; k <<= 1)
//     for (j = k >> 1; j > 0; j >>= 1) {
//       for (i = tid; i < n; i += nthreads) {
//         l = i ^ j;
//         if (l > i && bitonic_should_swap(key[i], idx[i], key[l], idx[l], (i & k) == 0))
//            swap(key[i], key[l]); swap(idx[i], idx[l]);
//       }
//       barrier();
//     }
//
// Work is O(n log^2 n) spread over n/2 threads, i.e. log^2(n) dependent steps,
// against O(n^2) serial steps for the insertion sort it replaces.
// ──────────────────────────────────────────────

#define VAULTX_PAD_KEY 0xFFFFFFFFFFFFFFFFULL

// True when the pair at (i, l) is out of order for this stage's direction.
VAULTX_HD inline bool bitonic_should_swap(uint64_t key_i, uint32_t idx_i,
                                         uint64_t key_l, uint32_t idx_l,
                                         bool ascending) {
    bool greater = (key_i > key_l) || (key_i == key_l && idx_i > idx_l);
    return greater == ascending;
}

#endif // VAULTXGPU_SORT_NET_H
