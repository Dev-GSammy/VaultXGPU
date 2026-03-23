#ifndef VAULTXGPU_PLOT_IO_H
#define VAULTXGPU_PLOT_IO_H

#include "globals.h"
#include <cstdint>

// Write table2 data to a plot file.
// Filename: K{K}_{hex_plot_id}.plot in the given output directory.
// Data is written sequentially in 4MB chunks.
//
// Returns 0 on success, -1 on error.
int write_plot_file(
    const MemoTable2Record* table2_data,
    size_t total_nonces,           // 2^K
    int K,
    const uint8_t* plot_id,
    const char* output_dir
);

#endif // VAULTXGPU_PLOT_IO_H
