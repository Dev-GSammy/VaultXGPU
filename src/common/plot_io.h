#ifndef VAULTXGPU_PLOT_IO_H
#define VAULTXGPU_PLOT_IO_H

#include "globals.h"
#include <cstdint>
#include <cstddef>

// Build the full plot file path into dest (dest_size bytes).
// Format: {dir}/k{K}-{hex_plot_id}.plot
void build_plot_path(char* dest, size_t dest_size,
                     const char* dir, int K, const uint8_t* plot_id);

// Write table2 data to a plot file.
// Data is written sequentially in 4MB chunks.
// Returns 0 on success, -1 on error.
int write_plot_file(
    const MemoTable2Record* table2_data,
    size_t total_nonces,
    int K,
    const uint8_t* plot_id,
    const char* output_dir
);

#endif // VAULTXGPU_PLOT_IO_H
