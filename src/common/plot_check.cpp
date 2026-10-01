#include "plot_check.h"
#include "crypto_cpu.h"
#include "../blake3/blake3_common.h"

#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cinttypes>
#include <random>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

namespace {

uint64_t hash_to_uint64(const uint8_t* hash, int len) {
    uint64_t r = 0;
    int n = len < 8 ? len : 8;
    for (int i = 0; i < n; i++) r = (r << 8) | hash[i];
    return r;
}

// Nonces are stored little-endian, matching the kernels.
uint64_t nonce_to_uint64(const uint8_t* nonce) {
    uint64_t v = 0;
    for (int i = NONCE_SIZE - 1; i >= 0; i--) v = (v << 8) | nonce[i];
    return v;
}

} // namespace

bool plot_check_parse_name(const char* path, int& K, uint8_t plot_id[32]) {
    const char* base = strrchr(path, '/');
    base = base ? base + 1 : path;

    int k = 0;
    if (sscanf(base, "k%d-", &k) != 1) return false;
    const char* dash = strchr(base, '-');
    if (!dash) return false;
    const char* hex = dash + 1;
    if (strlen(hex) < 64) return false;

    for (int i = 0; i < 32; i++) {
        unsigned byte = 0;
        if (sscanf(hex + 2 * i, "%2x", &byte) != 1) return false;
        plot_id[i] = static_cast<uint8_t>(byte);
    }
    K = k;
    return true;
}

int plot_check_run(const char* path, int K, const uint8_t* plot_id,
                   const PlotCheckConfig& cfg, PlotCheckResult& result,
                   FILE* out) {
    result = PlotCheckResult();

    const uint64_t N   = 1ULL << K;
    const uint32_t rpb = records_per_bucket_for(K);
    const double   mf  = get_matching_factor(K);
    const uint64_t expected_distance =
        static_cast<uint64_t>(static_cast<double>(1ULL << (64 - K)) * (1.0 / mf));
    result.expected_bytes = N * sizeof(MemoTable2Record);

    // The key is derived from (K, plot ID) exactly as the plotter derives it.
    uint8_t  key[32];
    uint32_t key_words[8];
    derive_key(K, plot_id, key);
    key_to_words(key, key_words);

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        if (out) fprintf(out, "Error opening %s: %s\n", path, strerror(errno));
        return 2;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        if (out) fprintf(out, "Error: fstat failed on %s: %s\n", path, strerror(errno));
        close(fd);
        return 2;
    }
    result.file_bytes = static_cast<uint64_t>(st.st_size);
    result.size_ok    = (result.file_bytes == result.expected_bytes);

    if (out) {
        fprintf(out, "  K=%d  N=%" PRIu64 "  RPB=%u  matching_factor=%.5f  "
                     "expected_distance=%" PRIu64 "\n", K, N, rpb, mf, expected_distance);
        fprintf(out, "  file size %" PRIu64 " bytes, expected %" PRIu64 "%s\n",
                result.file_bytes, result.expected_bytes,
                result.size_ok ? "" : "   <-- MISMATCH");
    }

    // Which buckets to check.
    std::vector<uint64_t> buckets;
    if (cfg.buckets && cfg.bucket_count > 0) {
        buckets.assign(cfg.buckets, cfg.buckets + cfg.bucket_count);
        if (out) fprintf(out, "  checking %zu named bucket(s)\n", buckets.size());
    } else if (cfg.sample > 0 && cfg.sample < TOTAL_BUCKETS) {
        std::mt19937_64 rng(cfg.seed);
        buckets.reserve(cfg.sample);
        for (uint64_t i = 0; i < cfg.sample; i++) buckets.push_back(rng() % TOTAL_BUCKETS);
        if (out) fprintf(out, "  checking %" PRIu64 " sampled buckets (seed %" PRIu64 ")\n",
                         cfg.sample, cfg.seed);
    } else {
        buckets.reserve(TOTAL_BUCKETS);
        for (uint64_t b = 0; b < TOTAL_BUCKETS; b++) buckets.push_back(b);
        if (out) fprintf(out, "  checking all %" PRIu64 " buckets\n", (uint64_t)TOTAL_BUCKETS);
    }

    std::vector<MemoTable2Record> buf(rpb);
    const size_t bucket_bytes = rpb * sizeof(MemoTable2Record);
    long printed = 0;

    for (size_t bi = 0; bi < buckets.size(); bi++) {
        const uint64_t b = buckets[bi];
        const off_t off = static_cast<off_t>(b * bucket_bytes);

        size_t got = 0;
        while (got < bucket_bytes) {
            ssize_t n = pread(fd, reinterpret_cast<char*>(buf.data()) + got,
                              bucket_bytes - got, off + static_cast<off_t>(got));
            if (n == 0) break;
            if (n < 0) {
                if (errno == EINTR) continue;
                if (out) fprintf(out, "Error reading bucket %" PRIu64 ": %s\n",
                                 b, strerror(errno));
                close(fd);
                return 2;
            }
            got += static_cast<size_t>(n);
        }
        if (got < bucket_bytes) {
            if (out) fprintf(out, "Error: short read at bucket %" PRIu64
                                  " (file truncated?)\n", b);
            close(fd);
            return 2;
        }

        for (uint32_t s = 0; s < rpb; s++) {
            result.slots_checked++;
            const MemoTable2Record& r = buf[s];

            // A real pair cannot be all zeros: the two nonces of a match are
            // always distinct, and only nonce 0 encodes as zero bytes.
            bool all_zero = true;
            for (int i = 0; i < NONCE_SIZE && all_zero; i++)
                if (r.nonce1[i] != 0 || r.nonce2[i] != 0) all_zero = false;
            if (all_zero) { result.empty++; continue; }
            result.records++;

            const char* fail = nullptr;

            uint64_t n1 = nonce_to_uint64(r.nonce1);
            uint64_t n2 = nonce_to_uint64(r.nonce2);
            if (n1 >= N || n2 >= N) { result.bad_range++; fail = "nonce out of range"; }

            uint8_t h1[HASH_SIZE], h2[HASH_SIZE], t2[HASH_SIZE];
            blake3_keyed_hash(r.nonce1, NONCE_SIZE, key_words, h1, HASH_SIZE);
            blake3_keyed_hash(r.nonce2, NONCE_SIZE, key_words, h2, HASH_SIZE);

            if (!fail && getBucketIndex(h1) != getBucketIndex(h2)) {
                result.bad_t1_bucket++;
                fail = "nonces are not from the same Table1 bucket";
            }

            if (!fail) {
                uint64_t k1 = hash_to_uint64(h1, HASH_SIZE);
                uint64_t k2 = hash_to_uint64(h2, HASH_SIZE);
                if (k2 < k1) {
                    result.bad_order++; fail = "pair stored out of hash order";
                } else if (k2 - k1 > expected_distance) {
                    result.bad_distance++; fail = "hash distance exceeds the match threshold";
                }
            }

            if (!fail) {
                uint8_t pair[NONCE_SIZE * 2];
                memcpy(pair, r.nonce1, NONCE_SIZE);
                memcpy(pair + NONCE_SIZE, r.nonce2, NONCE_SIZE);
                blake3_keyed_hash(pair, NONCE_SIZE * 2, key_words, t2, HASH_SIZE);
                if (getBucketIndex(t2) != b) {
                    result.bad_t2_bucket++;
                    fail = "pair is stored in the wrong Table2 bucket";
                }
            }

            if (fail && out && !cfg.quiet && printed < cfg.max_errors) {
                fprintf(out, "  FAIL bucket %" PRIu64 " slot %u: %s "
                             "(nonce1=%" PRIu64 ", nonce2=%" PRIu64 ")\n",
                        b, s, fail, n1, n2);
                printed++;
            }
        }
    }
    close(fd);

    result.invalid = result.bad_range + result.bad_t1_bucket + result.bad_order
                   + result.bad_distance + result.bad_t2_bucket;
    result.storage_efficiency = result.slots_checked
        ? (double)result.records / (double)result.slots_checked : 0.0;
    result.ok = (result.invalid == 0) && result.size_ok;

    if (out && !cfg.quiet && printed >= cfg.max_errors && result.invalid > (uint64_t)printed)
        fprintf(out, "(%" PRIu64 " further failures not printed)\n",
                result.invalid - (uint64_t)printed);

    return result.ok ? 0 : 1;
}

void plot_check_print_summary(FILE* out, const PlotCheckResult& r) {
    if (!out) return;
    fprintf(out, "\n=== Validation summary ===\n");
    fprintf(out, "slots checked        %" PRIu64 "\n", r.slots_checked);
    fprintf(out, "records present      %" PRIu64 "\n", r.records);
    fprintf(out, "empty slots          %" PRIu64 "\n", r.empty);
    fprintf(out, "storage efficiency   %.4f%%\n", 100.0 * r.storage_efficiency);
    fprintf(out, "invalid records      %" PRIu64 "\n", r.invalid);
    if (r.invalid) {
        fprintf(out, "  nonce out of range        %" PRIu64 "\n", r.bad_range);
        fprintf(out, "  wrong Table1 bucket       %" PRIu64 "\n", r.bad_t1_bucket);
        fprintf(out, "  out of hash order         %" PRIu64 "\n", r.bad_order);
        fprintf(out, "  distance over threshold   %" PRIu64 "\n", r.bad_distance);
        fprintf(out, "  wrong Table2 bucket       %" PRIu64 "\n", r.bad_t2_bucket);
    }
    fprintf(out, "\n%s\n", r.ok ? "PASS" : "FAIL");
}
