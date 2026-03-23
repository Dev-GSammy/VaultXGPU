#include "crypto_cpu.h"

#include <sodium.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Generate a random 32-byte plot ID.
// Matches CPU VaultX: generate 32 random bytes, SHA-256 hash them.
void generate_plot_id(uint8_t* plot_id_out) {
    uint8_t random_bytes[32];
    randombytes_buf(random_bytes, 32);
    crypto_hash_sha256(plot_id_out, random_bytes, sizeof(random_bytes));
}

// Derive key from plot_id and K value.
// key = SHA-256(plot_id || k_byte)
void derive_key(int k, const uint8_t* plot_id, uint8_t* key_out) {
    uint8_t temp_input[33];
    memcpy(temp_input, plot_id, 32);
    temp_input[32] = static_cast<uint8_t>(k);
    crypto_hash_sha256(key_out, temp_input, sizeof(temp_input));
}

// Convert 32-byte key to 8 x uint32 key words (little-endian load)
// Same logic as blake3_load_key_words but CPU-only, no device annotations.
void key_to_words(const uint8_t* key, uint32_t* key_words) {
    for (int i = 0; i < 8; i++) {
        const uint8_t* p = key + i * 4;
        key_words[i] = static_cast<uint32_t>(p[0])
                     | (static_cast<uint32_t>(p[1]) << 8)
                     | (static_cast<uint32_t>(p[2]) << 16)
                     | (static_cast<uint32_t>(p[3]) << 24);
    }
}

// Convert byte array to hex string
char* byteArrayToHexString(const uint8_t* bytes, size_t len) {
    char* out = static_cast<char*>(malloc(len * 2 + 1));
    if (!out) return nullptr;
    for (size_t i = 0; i < len; i++) {
        sprintf(out + i * 2, "%02x", bytes[i]);
    }
    out[len * 2] = '\0';
    return out;
}
