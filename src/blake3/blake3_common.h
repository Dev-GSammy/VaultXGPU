#ifndef VAULTXGPU_BLAKE3_COMMON_H
#define VAULTXGPU_BLAKE3_COMMON_H

//
// GPU-portable Blake3 compression.
// Ported from ~/vaultx/blake3/c/blake3_portable.c + blake3_impl.h.
//
// Compiles under both nvcc (__device__ __host__) and icpx (plain inline).
// Only the single-chunk, single-block keyed-hash path is needed:
//   input <= 64 bytes, counter = 0,
//   flags = CHUNK_START | CHUNK_END | KEYED_HASH | ROOT
//

#include <cstdint>
#include <cstring>

// Device/host annotation macros
#ifdef GPU_CUDA
  #define BLAKE3_DEV __device__ __host__
#else
  #define BLAKE3_DEV
#endif

// ──────────────────────────────────────────────
// Constants from blake3_impl.h
// ──────────────────────────────────────────────

// Blake3 initialization vector (same as SHA-256 initial values)
// Defined as constexpr -- compiler embeds into device code automatically.
BLAKE3_DEV inline void blake3_get_iv(uint32_t iv[8]) {
    iv[0] = 0x6A09E667UL; iv[1] = 0xBB67AE85UL;
    iv[2] = 0x3C6EF372UL; iv[3] = 0xA54FF53AUL;
    iv[4] = 0x510E527FUL; iv[5] = 0x9B05688CUL;
    iv[6] = 0x1F83D9ABUL; iv[7] = 0x5BE0CD19UL;
}

// Message schedule permutation per round (7 rounds total)
// Embedded as a lookup function to avoid __constant__ linkage issues.
BLAKE3_DEV inline uint8_t blake3_msg_schedule(int round, int idx) {
    // Flattened [7][16] schedule table
    static constexpr uint8_t TABLE[7 * 16] = {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
         2,  6,  3, 10,  7,  0,  4, 13,  1, 11, 12,  5,  9, 14, 15,  8,
         3,  4, 10, 12, 13,  2,  7, 14,  6,  5,  9,  0, 11, 15,  8,  1,
        10,  7, 12,  9, 14,  3, 13, 15,  4,  0, 11,  2,  5,  8,  1,  6,
        12, 13,  9, 11, 15, 10, 14,  8,  7,  2,  5,  3,  0,  1,  6,  4,
         9, 14, 11,  5,  8, 12, 15,  1, 13,  3,  0, 10,  2,  6,  4,  7,
        11, 15,  5,  0,  1,  9,  8,  6, 14, 10,  2, 12,  3,  4,  7, 13,
    };
    return TABLE[round * 16 + idx];
}

// Flag constants
enum Blake3Flags : uint8_t {
    BLAKE3_CHUNK_START = 1 << 0,  // 0x01
    BLAKE3_CHUNK_END   = 1 << 1,  // 0x02
    BLAKE3_ROOT        = 1 << 3,  // 0x08
    BLAKE3_KEYED_HASH  = 1 << 4,  // 0x10
};

// For single-block keyed hash: all flags combined
static constexpr uint8_t BLAKE3_KEYED_HASH_FLAGS =
    BLAKE3_CHUNK_START | BLAKE3_CHUNK_END | BLAKE3_KEYED_HASH | BLAKE3_ROOT;

// ──────────────────────────────────────────────
// Primitives
// ──────────────────────────────────────────────

BLAKE3_DEV inline uint32_t blake3_rotr32(uint32_t w, uint32_t c) {
    return (w >> c) | (w << (32 - c));
}

// Little-endian load of 4 bytes into uint32
BLAKE3_DEV inline uint32_t blake3_load32(const uint8_t* src) {
    return static_cast<uint32_t>(src[0])
         | (static_cast<uint32_t>(src[1]) << 8)
         | (static_cast<uint32_t>(src[2]) << 16)
         | (static_cast<uint32_t>(src[3]) << 24);
}

// Little-endian store of uint32 into 4 bytes
BLAKE3_DEV inline void blake3_store32(uint8_t* dst, uint32_t w) {
    dst[0] = static_cast<uint8_t>(w);
    dst[1] = static_cast<uint8_t>(w >> 8);
    dst[2] = static_cast<uint8_t>(w >> 16);
    dst[3] = static_cast<uint8_t>(w >> 24);
}

// ──────────────────────────────────────────────
// G mixing function
// ──────────────────────────────────────────────

BLAKE3_DEV inline void blake3_g(
    uint32_t* state, int a, int b, int c, int d, uint32_t x, uint32_t y
) {
    state[a] = state[a] + state[b] + x;
    state[d] = blake3_rotr32(state[d] ^ state[a], 16);
    state[c] = state[c] + state[d];
    state[b] = blake3_rotr32(state[b] ^ state[c], 12);
    state[a] = state[a] + state[b] + y;
    state[d] = blake3_rotr32(state[d] ^ state[a], 8);
    state[c] = state[c] + state[d];
    state[b] = blake3_rotr32(state[b] ^ state[c], 7);
}

// ──────────────────────────────────────────────
// One round: 8 G calls with message schedule
// ──────────────────────────────────────────────

BLAKE3_DEV inline void blake3_round_fn(
    uint32_t state[16], const uint32_t* msg, int round
) {
    // Mix columns
    blake3_g(state, 0, 4,  8, 12,
             msg[blake3_msg_schedule(round, 0)],  msg[blake3_msg_schedule(round, 1)]);
    blake3_g(state, 1, 5,  9, 13,
             msg[blake3_msg_schedule(round, 2)],  msg[blake3_msg_schedule(round, 3)]);
    blake3_g(state, 2, 6, 10, 14,
             msg[blake3_msg_schedule(round, 4)],  msg[blake3_msg_schedule(round, 5)]);
    blake3_g(state, 3, 7, 11, 15,
             msg[blake3_msg_schedule(round, 6)],  msg[blake3_msg_schedule(round, 7)]);

    // Mix diagonals
    blake3_g(state, 0, 5, 10, 15,
             msg[blake3_msg_schedule(round, 8)],  msg[blake3_msg_schedule(round, 9)]);
    blake3_g(state, 1, 6, 11, 12,
             msg[blake3_msg_schedule(round, 10)], msg[blake3_msg_schedule(round, 11)]);
    blake3_g(state, 2, 7,  8, 13,
             msg[blake3_msg_schedule(round, 12)], msg[blake3_msg_schedule(round, 13)]);
    blake3_g(state, 3, 4,  9, 14,
             msg[blake3_msg_schedule(round, 14)], msg[blake3_msg_schedule(round, 15)]);
}

// ──────────────────────────────────────────────
// Keyed hash: single block, single chunk
// ──────────────────────────────────────────────
//
// This is the GPU equivalent of:
//   blake3_hasher_init_keyed(&h, key);
//   blake3_hasher_update(&h, input, input_len);
//   blake3_hasher_finalize(&h, output, output_len);
//
// Valid for input_len <= 64 bytes (one block).
// key_words = 8 x uint32 derived from the 32-byte key via little-endian load.
//

BLAKE3_DEV inline void blake3_keyed_hash(
    const uint8_t* input,
    size_t         input_len,
    const uint32_t* key_words,  // 8 x uint32
    uint8_t*       output,
    size_t         output_len
) {
    // 1. Pad input into a 64-byte block (zero-padded)
    uint8_t block[64];
    for (int i = 0; i < 64; i++) {
        block[i] = (static_cast<size_t>(i) < input_len) ? input[i] : 0;
    }

    // 2. Load block words (little-endian)
    uint32_t block_words[16];
    for (int i = 0; i < 16; i++) {
        block_words[i] = blake3_load32(block + i * 4);
    }

    // 3. Initialize state
    //    state[0..7]  = cv = key_words (keyed mode)
    //    state[8..11] = IV[0..3]
    //    state[12]    = counter_low  = 0
    //    state[13]    = counter_high = 0
    //    state[14]    = block_len
    //    state[15]    = flags
    uint32_t state[16];
    state[0]  = key_words[0];
    state[1]  = key_words[1];
    state[2]  = key_words[2];
    state[3]  = key_words[3];
    state[4]  = key_words[4];
    state[5]  = key_words[5];
    state[6]  = key_words[6];
    state[7]  = key_words[7];
    uint32_t iv[8];
    blake3_get_iv(iv);
    state[8]  = iv[0];
    state[9]  = iv[1];
    state[10] = iv[2];
    state[11] = iv[3];
    state[12] = 0;  // counter low
    state[13] = 0;  // counter high
    state[14] = static_cast<uint32_t>(input_len);
    state[15] = static_cast<uint32_t>(BLAKE3_KEYED_HASH_FLAGS);

    // 4. Seven rounds of compression
    blake3_round_fn(state, block_words, 0);
    blake3_round_fn(state, block_words, 1);
    blake3_round_fn(state, block_words, 2);
    blake3_round_fn(state, block_words, 3);
    blake3_round_fn(state, block_words, 4);
    blake3_round_fn(state, block_words, 5);
    blake3_round_fn(state, block_words, 6);

    // 5. Finalize: XOR state halves, output first output_len bytes
    //    output_word[i] = state[i] ^ state[i+8]
    //    (This matches blake3_compress_xof_portable for the first 32 bytes)
    uint8_t full_output[32];
    for (int i = 0; i < 8; i++) {
        blake3_store32(full_output + i * 4, state[i] ^ state[i + 8]);
    }

    // Copy requested output length
    for (size_t i = 0; i < output_len && i < 32; i++) {
        output[i] = full_output[i];
    }
}

// ──────────────────────────────────────────────
// Helper: load 32-byte key into 8 x uint32 words
// ──────────────────────────────────────────────

BLAKE3_DEV inline void blake3_load_key_words(
    const uint8_t key[32], uint32_t key_words[8]
) {
    for (int i = 0; i < 8; i++) {
        key_words[i] = blake3_load32(key + i * 4);
    }
}

#endif // VAULTXGPU_BLAKE3_COMMON_H
