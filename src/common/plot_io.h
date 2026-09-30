#ifndef VAULTXGPU_PLOT_IO_H
#define VAULTXGPU_PLOT_IO_H

#include "globals.h"
#include <cstdint>
#include <cstddef>

// Build the full plot file path into dest (dest_size bytes).
// Format: {dir}/k{K}-{hex_plot_id}.plot
void build_plot_path(char* dest, size_t dest_size,
                     const char* dir, int K, const uint8_t* plot_id);

// Table2 is written by PlotWriter (common/plot_writer.h), driven from each
// backend so the device-to-host copy and the disk write are timed separately.
// There is deliberately no second write path here.

#endif // VAULTXGPU_PLOT_IO_H
