#ifndef VAULTXGPU_MEMORY_H
#define VAULTXGPU_MEMORY_H

#include <cstdint>
#include <cstddef>

// Estimate peak GPU memory required for K value (in bytes)
size_t estimate_gpu_memory_required(int K);

// Check if K fits in available GPU memory
bool can_fit_in_memory(int K, size_t available_bytes);

// Find the largest K that fits in available memory
int largest_k_that_fits(size_t available_bytes, int max_k = 32);

// Print memory budget summary
void print_memory_budget(int K, size_t available_bytes);

#endif // VAULTXGPU_MEMORY_H
