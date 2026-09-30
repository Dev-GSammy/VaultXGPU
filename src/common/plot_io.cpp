#include "plot_io.h"
#include "crypto_cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Construct plot filename: k{K}-{hex_plot_id}.plot
// Matches CPU VaultX search format (lowercase k, dash separator)
void build_plot_path(char* dest, size_t dest_size,
                     const char* dir, int K, const uint8_t* plot_id) {
    char* hex = byteArrayToHexString(plot_id, 32);
    if (!hex) {
        snprintf(dest, dest_size, "%s/k%d-unknown.plot", dir, K);
        return;
    }
    size_t dir_len = strlen(dir);
    bool has_slash = (dir_len > 0 && dir[dir_len - 1] == '/');
    snprintf(dest, dest_size, "%s%sk%d-%s.plot",
             dir, has_slash ? "" : "/", K, hex);
    free(hex);
}
