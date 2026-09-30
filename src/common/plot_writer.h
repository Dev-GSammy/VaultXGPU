#ifndef VAULTXGPU_PLOT_WRITER_H
#define VAULTXGPU_PLOT_WRITER_H

#include <cstdint>
#include <cstddef>

// ──────────────────────────────────────────────
// Output stage
//
// Owns the plot file so that device-to-host transfer and disk write are separate
// operations with separate timers. The backend copies a chunk into a staging
// buffer and hands it here; nothing in this file touches the GPU.
//
// Two things this exists to get right:
//
//  * write() returns when the data reaches the page cache, not the device. With
//    o_direct the page cache is bypassed and the timer measures the device. With
//    o_direct off, fsync_at_end at least charges the flush to the write stage
//    instead of leaving it to fall outside the measurement.
//
//  * With overlap on, the D2H copy of chunk i runs while chunk i-1 is being
//    written, so the stage costs max(pcie, disk) instead of pcie + disk. The
//    caller must alternate between two staging buffers; see submit().
// ──────────────────────────────────────────────

struct PlotWriterConfig {
    size_t chunk_bytes  = 256ULL * 1024 * 1024;
    bool   o_direct     = false;
    bool   overlap      = true;
    bool   fsync_at_end = true;
};

// Alignment required of staging buffers, chunk sizes and file offsets under
// O_DIRECT. cudaMallocHost and aligned_alloc(alignment, ...) both satisfy it.
size_t plot_writer_alignment();

class PlotWriter {
public:
    PlotWriter() = default;
    ~PlotWriter();

    PlotWriter(const PlotWriter&)            = delete;
    PlotWriter& operator=(const PlotWriter&) = delete;

    // Create/truncate `path`. Returns 0 on success.
    int open(const char* path, uint64_t total_bytes, const PlotWriterConfig& cfg);

    // Hand off a filled staging buffer of `bytes` bytes.
    //
    // With overlap on this returns as soon as the *previously* submitted buffer
    // has been fully written, so on return every buffer except `buf` is free to
    // refill. Alternate two buffers and the D2H copy overlaps the write.
    //
    // With overlap off it writes synchronously before returning.
    int submit(const void* buf, size_t bytes);

    // Drain the queue, fdatasync if configured, close. Safe to call twice.
    int finish();

    double   write_seconds() const { return write_seconds_; }
    double   fsync_seconds() const { return fsync_seconds_; }
    uint64_t bytes_written() const { return bytes_written_; }

    // True when O_DIRECT was requested and actually granted. Some filesystems
    // (tmpfs, and NFS depending on server and mount options) reject it; we fall
    // back to buffered I/O and report it rather than mislabelling the run.
    bool o_direct_effective() const { return o_direct_effective_; }

private:
    int  write_at(const void* buf, size_t bytes, uint64_t offset);
    void writer_loop();
    int  drain();

    struct Impl;
    Impl* impl_ = nullptr;

    int      fd_        = -1;
    int      fd_buffered_ = -1;    // opened lazily for a misaligned tail under O_DIRECT
    char     path_[1024] = {0};
    uint64_t offset_     = 0;
    uint64_t bytes_written_ = 0;
    double   write_seconds_ = 0.0;
    double   fsync_seconds_ = 0.0;
    bool     o_direct_effective_ = false;
    bool     overlap_   = false;
    bool     fsync_at_end_ = true;
    bool     finished_  = false;
    int      error_     = 0;
};

#endif // VAULTXGPU_PLOT_WRITER_H
