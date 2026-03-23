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
#define PREFIX_SIZE  3
#define TOTAL_BUCKETS (1ULL << (PREFIX_SIZE * 8))  // 2^24 = 16,777,216


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
