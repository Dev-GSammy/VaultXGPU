#ifndef VAULTXGPU_CRYPTO_CPU_H
#define VAULTXGPU_CRYPTO_CPU_H

#include <cstdint>
#include <cstddef>

// Generate a random 32-byte plot ID (uses libsodium)
void generate_plot_id(uint8_t* plot_id_out);

// Derive a 32-byte key from plot_id + K (SHA-256)
void derive_key(int k, const uint8_t* plot_id, uint8_t* key_out);

// Convert 32-byte key to 8 x uint32 key words (little-endian)
void key_to_words(const uint8_t* key, uint32_t* key_words);

// Convert byte array to hex string (caller must free)
char* byteArrayToHexString(const uint8_t* bytes, size_t len);

#endif // VAULTXGPU_CRYPTO_CPU_H
