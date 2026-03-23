#include "plot_io.h"
#include "crypto_cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

// Construct plot filename: k{K}-{hex_plot_id}.plot
// Matches CPU VaultX search format (lowercase k, dash separator)
static void build_plot_path(char* dest, size_t dest_size,
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

int write_plot_file(
    const MemoTable2Record* table2_data,
    size_t total_nonces,
    int K,
    const uint8_t* plot_id,
    const char* output_dir
) {
    char filepath[512];
    build_plot_path(filepath, sizeof(filepath), output_dir, K, plot_id);

    int fd = open(filepath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1) {
        fprintf(stderr, "Error opening file %s: %s\n", filepath, strerror(errno));
        return -1;
    }

    printf("Writing plot file: %s\n", filepath);

    size_t total_size = total_nonces * sizeof(MemoTable2Record);
    const char* data_ptr = reinterpret_cast<const char*>(table2_data);
    size_t total_written = 0;
    constexpr size_t chunk_size = 4 * 1024 * 1024; // 4 MB

    while (total_written < total_size) {
        size_t remaining = total_size - total_written;
        size_t to_write = remaining < chunk_size ? remaining : chunk_size;

        ssize_t bytes = write(fd, data_ptr + total_written, to_write);
        if (bytes < 0) {
            fprintf(stderr, "Error writing at offset %zu: %s\n",
                    total_written, strerror(errno));
            close(fd);
            return -1;
        }
        total_written += static_cast<size_t>(bytes);
    }

    close(fd);
    printf("Plot file written: %zu bytes\n", total_written);
    return 0;
}
