#ifndef VAULTXGPU_GLOBALS_H
#define VAULTXGPU_GLOBALS_H

#include <cstdint>
#include <cstddef>

// ──────────────────────────────────────────────
// Compile-time constants (set via -D flags)
// ──────────────────────────────────────────────
#ifndef NONCE_SIZE
#define NONCE_SIZE 4
#endif

#ifndef RECORD_SIZE
#define RECORD_SIZE 12
#endif

#define HASH_SIZE    (RECORD_SIZE - NONCE_SIZE)

// Bucket count is 2^(8*PREFIX_SIZE). Overridable so host-side tests can run the
// whole pipeline at a tractable size; production builds leave it at 3.
#ifndef PREFIX_SIZE
#define PREFIX_SIZE  3
#endif
#define TOTAL_BUCKETS (1ULL << (PREFIX_SIZE * 8))  // 2^24 = 16,777,216

// Supported K range. The lower bound is set by the matching-factor table below:
// entries outside [MIN_K, MAX_K] are placeholders, so a run at such a K would
// silently use a bogus expected_distance. main.cpp rejects out-of-range K.
#ifndef MIN_K
#define MIN_K 27
#endif
#ifndef MAX_K
#define MAX_K 32
#endif

// Sort algorithm selection for the Table2 kernel (see sort_table2_*.{cu,cpp}).
//   0 = single-thread insertion sort moving whole records (original baseline)
//   1 = single-thread insertion sort over (key, index) pairs
//   2 = block-parallel bitonic sort over (key, index) pairs  [default]
#ifndef VAULTX_SORT
#define VAULTX_SORT 2
#endif

#define VAULTX_SORT_INSERTION_RECORDS 0
#define VAULTX_SORT_INSERTION_INDEX   1
#define VAULTX_SORT_BITONIC           2

// Name of the active sort, for the CSV/benchmark line.
#if   VAULTX_SORT == VAULTX_SORT_INSERTION_RECORDS
#define VAULTX_SORT_NAME "insertion_records"
#elif VAULTX_SORT == VAULTX_SORT_INSERTION_INDEX
#define VAULTX_SORT_NAME "insertion_index"
#elif VAULTX_SORT == VAULTX_SORT_BITONIC
#define VAULTX_SORT_NAME "bitonic"
#else
#error "VAULTX_SORT must be 0, 1 or 2"
#endif


// Data structures


// Table1 record: stores a single nonce
struct MemoRecord {
    uint8_t nonce[NONCE_SIZE];
};

// Table2 record: stores a matched nonce pair
struct MemoTable2Record {
    uint8_t nonce1[NONCE_SIZE];
    uint8_t nonce2[NONCE_SIZE];
};

// Matching factor table

inline double get_matching_factor(int K) {
    switch (K) {
        case 25: return 0.11680;
        case 26: return 0.00010;
        case 27: return 0.13639;
        case 28: return 0.33318;
        case 29: return 0.50763;
        case 30: return 0.62341;
        case 31: return 0.73366;
        case 32: return 0.83706;
        default: return 1.0;
    }
}


// Records per bucket for a given K. Always a power of two in [MIN_K, MAX_K]
// (8 at K=27 ... 256 at K=32), which the bitonic sort relies on.
inline uint32_t records_per_bucket_for(int K) {
    return static_cast<uint32_t>((1ULL << K) / TOTAL_BUCKETS);
}


// Utility: bucket index from hash prefix (big-endian)

inline uint32_t getBucketIndex(const uint8_t* hash) {
    uint32_t index = 0;
    for (size_t i = 0; i < PREFIX_SIZE && i < HASH_SIZE; i++) {
        index = (index << 8) | hash[i];
    }
    return index;
}


// Utility: convert byte array to big-endian uint64

inline uint64_t byteArrayToUint64(const uint8_t* arr, size_t len) {
    uint64_t result = 0;
    for (size_t i = 0; i < len && i < 8; i++) {
        result = (result << 8) | static_cast<uint64_t>(arr[i]);
    }
    return result;
}

#endif // VAULTXGPU_GLOBALS_H
