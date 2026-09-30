#include "plot_writer.h"
#include "metrics.h"

#include <cstdio>
#include <cstring>
#include <cerrno>
#include <condition_variable>
#include <mutex>
#include <thread>

#include <fcntl.h>
#include <unistd.h>

#ifndef O_DIRECT
#define O_DIRECT 0
#endif

namespace {
constexpr size_t kAlignment = 4096;
}

size_t plot_writer_alignment() { return kAlignment; }

// Queue slot shared with the writer thread. One slot deep: submit() waits for the
// previous buffer to drain, which is what makes caller-side double buffering safe.
struct PlotWriter::Impl {
    std::mutex              m;
    std::condition_variable cv;
    std::thread             thread;

    const void* buf     = nullptr;
    size_t      bytes   = 0;
    uint64_t    offset  = 0;
    bool        pending = false;
    bool        stop    = false;
    int         error   = 0;
};

PlotWriter::~PlotWriter() {
    finish();
    delete impl_;
    impl_ = nullptr;
}

int PlotWriter::open(const char* path, uint64_t total_bytes, const PlotWriterConfig& cfg) {
    snprintf(path_, sizeof(path_), "%s", path ? path : "");
    overlap_      = cfg.overlap;
    fsync_at_end_ = cfg.fsync_at_end;
    finished_     = false;

    int flags = O_WRONLY | O_CREAT | O_TRUNC;
    if (cfg.o_direct) {
        fd_ = ::open(path_, flags | O_DIRECT, 0644);
        if (fd_ >= 0) {
            o_direct_effective_ = true;
        } else {
            fprintf(stderr,
                "Warning: O_DIRECT not available on %s (%s); falling back to buffered I/O.\n",
                path_, strerror(errno));
        }
    }
    if (fd_ < 0) {
        fd_ = ::open(path_, flags, 0644);
        o_direct_effective_ = false;
    }
    if (fd_ < 0) {
        fprintf(stderr, "Error opening %s: %s\n", path_, strerror(errno));
        return -1;
    }

    // Preallocating keeps filesystem block allocation out of the timed writes.
    if (total_bytes > 0) {
        if (posix_fallocate(fd_, 0, static_cast<off_t>(total_bytes)) != 0) {
            // Not fatal: unsupported on some filesystems.
            if (ftruncate(fd_, static_cast<off_t>(total_bytes)) != 0) {
                fprintf(stderr, "Warning: could not preallocate %s: %s\n",
                        path_, strerror(errno));
            }
        }
    }

    if (overlap_) {
        impl_ = new Impl();
        impl_->thread = std::thread([this] { writer_loop(); });
    }
    return 0;
}

// One pwrite loop. Under O_DIRECT a trailing fragment shorter than the alignment
// cannot go through this fd, so it is routed to a buffered fd at the same offset.
int PlotWriter::write_at(const void* buf, size_t bytes, uint64_t offset) {
    const char* p = static_cast<const char*>(buf);

    size_t aligned = bytes;
    size_t tail    = 0;
    if (o_direct_effective_ && (bytes % kAlignment) != 0) {
        aligned = (bytes / kAlignment) * kAlignment;
        tail    = bytes - aligned;
    }

    size_t done = 0;
    while (done < aligned) {
        ssize_t n = pwrite(fd_, p + done, aligned - done, static_cast<off_t>(offset + done));
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            fprintf(stderr, "Error writing %s at offset %llu: %s\n",
                    path_, (unsigned long long)(offset + done), strerror(errno));
            return -1;
        }
        done += static_cast<size_t>(n);
    }

    if (tail > 0) {
        if (fd_buffered_ < 0) {
            fd_buffered_ = ::open(path_, O_WRONLY);
            if (fd_buffered_ < 0) {
                fprintf(stderr, "Error reopening %s for unaligned tail: %s\n",
                        path_, strerror(errno));
                return -1;
            }
        }
        size_t tdone = 0;
        while (tdone < tail) {
            ssize_t n = pwrite(fd_buffered_, p + aligned + tdone, tail - tdone,
                               static_cast<off_t>(offset + aligned + tdone));
            if (n <= 0) {
                if (n < 0 && errno == EINTR) continue;
                fprintf(stderr, "Error writing tail of %s: %s\n", path_, strerror(errno));
                return -1;
            }
            tdone += static_cast<size_t>(n);
        }
    }
    return 0;
}

void PlotWriter::writer_loop() {
    for (;;) {
        const void* buf;
        size_t      bytes;
        uint64_t    offset;
        {
            std::unique_lock<std::mutex> lk(impl_->m);
            impl_->cv.wait(lk, [this] { return impl_->pending || impl_->stop; });
            if (!impl_->pending && impl_->stop) return;
            buf    = impl_->buf;
            bytes  = impl_->bytes;
            offset = impl_->offset;
        }

        double t0 = now_seconds();
        int rc = write_at(buf, bytes, offset);
        double dt = now_seconds() - t0;

        {
            std::lock_guard<std::mutex> lk(impl_->m);
            if (rc != 0) impl_->error = -1;
            write_seconds_ += dt;
            if (rc == 0) bytes_written_ += bytes;
            impl_->pending = false;
            impl_->cv.notify_all();
        }
    }
}

int PlotWriter::submit(const void* buf, size_t bytes) {
    if (fd_ < 0 || bytes == 0) return (fd_ < 0) ? -1 : 0;

    if (!overlap_) {
        double t0 = now_seconds();
        int rc = write_at(buf, bytes, offset_);
        write_seconds_ += now_seconds() - t0;
        if (rc != 0) { error_ = -1; return -1; }
        bytes_written_ += bytes;
        offset_ += bytes;
        return 0;
    }

    std::unique_lock<std::mutex> lk(impl_->m);
    // Wait for the previously submitted buffer, so on return every buffer but
    // this one is free for the caller to refill.
    impl_->cv.wait(lk, [this] { return !impl_->pending; });
    if (impl_->error != 0) { error_ = -1; return -1; }

    impl_->buf     = buf;
    impl_->bytes   = bytes;
    impl_->offset  = offset_;
    impl_->pending = true;
    offset_ += bytes;
    impl_->cv.notify_all();
    return 0;
}

int PlotWriter::drain() {
    if (!impl_) return error_;
    std::unique_lock<std::mutex> lk(impl_->m);
    impl_->cv.wait(lk, [this] { return !impl_->pending; });
    if (impl_->error != 0) error_ = -1;
    return error_;
}

int PlotWriter::finish() {
    if (finished_) return error_;
    finished_ = true;

    if (impl_) {
        drain();
        {
            std::lock_guard<std::mutex> lk(impl_->m);
            impl_->stop = true;
            impl_->cv.notify_all();
        }
        if (impl_->thread.joinable()) impl_->thread.join();
    }

    if (fd_ >= 0) {
        if (fsync_at_end_) {
            double t0 = now_seconds();
            if (fdatasync(fd_) != 0)
                fprintf(stderr, "Warning: fdatasync(%s) failed: %s\n", path_, strerror(errno));
            if (fd_buffered_ >= 0 && fdatasync(fd_buffered_) != 0)
                fprintf(stderr, "Warning: fdatasync tail failed: %s\n", strerror(errno));
            fsync_seconds_ += now_seconds() - t0;
        }
        if (fd_buffered_ >= 0) { ::close(fd_buffered_); fd_buffered_ = -1; }
        ::close(fd_);
        fd_ = -1;
    }
    return error_;
}
