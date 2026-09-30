// ──────────────────────────────────────────────
// vaultx_validate -- structural validation of a finished plot file
//
// Checks a plot on disk against the format, using nothing but the plot itself and
// its plot ID. No GPU, no second plot, no reference implementation: for every
// stored pair it recomputes the hashes and asserts the four properties the
// construction guarantees.
//
//   1. Both nonces are in range, i.e. below 2^K.
//   2. Both nonces hash into the same Table1 bucket (they were matched there).
//   3. Their 64-bit hash keys are ordered and within expected_distance of each
//      other -- the match rule for this K.
//   4. The pair hashes into the Table2 bucket it is physically stored in, which
//      is what makes the O(1) bucket seek valid.
//
// It also counts empty slots, which gives storage efficiency directly. A slot is
// empty iff every byte is zero: a real pair cannot be all zeros, because the two
// nonces of a match are always distinct and only nonce 0 encodes as zero bytes.
//
// Record order within a bucket is not checked and must not be: slots are claimed
// with an atomic, so order varies between runs on the same device.
// ──────────────────────────────────────────────

#include "../common/globals.h"
#include "../common/crypto_cpu.h"
#include "../blake3/blake3_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cinttypes>
#include <random>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

namespace {

struct Failure {
    uint32_t bucket;
    uint32_t slot;
    const char* what;
};

uint64_t hash_to_uint64(const uint8_t* hash, int len) {
    uint64_t r = 0;
    int n = len < 8 ? len : 8;
    for (int i = 0; i < n; i++) r = (r << 8) | hash[i];
    return r;
}

uint64_t nonce_to_uint64(const uint8_t* nonce) {
    uint64_t v = 0;
    for (int i = NONCE_SIZE - 1; i >= 0; i--) v = (v << 8) | nonce[i];  // little-endian
    return v;
}

// Parse "k{K}-{64 hex chars}.plot" out of the path.
bool parse_plot_name(const char* path, int& K, uint8_t plot_id[32]) {
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

void usage(const char* prog) {
    printf("Usage: %s <plot-file> [options]\n", prog);
    printf("\nOptions:\n");
    printf("  --sample N     Validate N randomly chosen buckets instead of all\n");
    printf("  --bucket N     Validate only this bucket; repeatable (overrides --sample)\n");
    printf("  --seed N       Seed for --sample (default: 1)\n");
    printf("  --max-errors N Stop printing individual failures after N (default: 20)\n");
    printf("  --plot-id HEX  Override the plot ID parsed from the filename\n");
    printf("  --ksize N      Override K parsed from the filename\n");
    printf("  -h, --help     Show this help\n");
    printf("\nExit status is 0 only when every checked record passes.\n");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 2; }
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) { usage(argv[0]); return 0; }

    const char* path = argv[1];
    uint64_t sample = 0;
    uint64_t seed = 1;
    std::vector<uint64_t> only_buckets;
    long max_errors = 20;
    int K = 0;
    uint8_t plot_id[32] = {0};
    bool have_id = false;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--sample") == 0 && i + 1 < argc) {
            sample = strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--bucket") == 0 && i + 1 < argc) {
            uint64_t b = strtoull(argv[++i], nullptr, 10);
            if (b >= TOTAL_BUCKETS) {
                fprintf(stderr, "Error: --bucket %" PRIu64 " is out of range (max %" PRIu64 ")\n",
                        b, (uint64_t)TOTAL_BUCKETS - 1);
                return 2;
            }
            only_buckets.push_back(b);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--max-errors") == 0 && i + 1 < argc) {
            max_errors = strtol(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--ksize") == 0 && i + 1 < argc) {
            K = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--plot-id") == 0 && i + 1 < argc) {
            const char* hex = argv[++i];
            if (strlen(hex) < 64) { fprintf(stderr, "Error: --plot-id needs 64 hex chars\n"); return 2; }
            for (int b = 0; b < 32; b++) {
                unsigned v = 0;
                if (sscanf(hex + 2 * b, "%2x", &v) != 1) {
                    fprintf(stderr, "Error: bad hex in --plot-id\n"); return 2;
                }
                plot_id[b] = static_cast<uint8_t>(v);
            }
            have_id = true;
        } else {
            fprintf(stderr, "Error: unknown argument '%s'\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    int parsed_K = 0;
    uint8_t parsed_id[32];
    if (parse_plot_name(path, parsed_K, parsed_id)) {
        if (K == 0) K = parsed_K;
        if (!have_id) { memcpy(plot_id, parsed_id, 32); have_id = true; }
    }
    if (K == 0 || !have_id) {
        fprintf(stderr, "Error: could not determine K and plot ID from '%s'.\n"
                        "       Pass --ksize and --plot-id explicitly.\n", path);
        return 2;
    }
    if (K < MIN_K || K > MAX_K) {
        fprintf(stderr, "Error: K=%d is outside the supported range %d-%d\n", K, MIN_K, MAX_K);
        return 2;
    }

    const uint64_t N   = 1ULL << K;
    const uint32_t rpb = records_per_bucket_for(K);
    const double   mf  = get_matching_factor(K);
    const uint64_t expected_distance =
        static_cast<uint64_t>(static_cast<double>(1ULL << (64 - K)) * (1.0 / mf));
    const uint64_t expect_bytes = N * sizeof(MemoTable2Record);

    // Key derivation must match the plotter exactly.
    uint8_t  key[32];
    uint32_t key_words[8];
    derive_key(K, plot_id, key);
    key_to_words(key, key_words);

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Error opening %s: %s\n", path, strerror(errno));
        return 2;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "Error: fstat failed on %s: %s\n", path, strerror(errno));
        close(fd);
        return 2;
    }

    printf("Validating %s\n", path);
    printf("  K=%d  N=%" PRIu64 "  RPB=%u  matching_factor=%.5f  expected_distance=%" PRIu64 "\n",
           K, N, rpb, mf, expected_distance);
    printf("  file size %" PRIu64 " bytes, expected %" PRIu64 "%s\n",
           (uint64_t)st.st_size, expect_bytes,
           ((uint64_t)st.st_size == expect_bytes) ? "" : "   <-- MISMATCH");

    bool size_ok = ((uint64_t)st.st_size == expect_bytes);

    // Which buckets to check.
    std::vector<uint64_t> buckets;
    if (!only_buckets.empty()) {
        buckets = only_buckets;
        printf("  checking %zu explicitly named bucket(s)\n", buckets.size());
    } else if (sample > 0 && sample < TOTAL_BUCKETS) {
        std::mt19937_64 rng(seed);
        buckets.reserve(sample);
        for (uint64_t i = 0; i < sample; i++) buckets.push_back(rng() % TOTAL_BUCKETS);
        printf("  checking %" PRIu64 " sampled buckets (seed %" PRIu64 ")\n", sample, seed);
    } else {
        buckets.reserve(TOTAL_BUCKETS);
        for (uint64_t b = 0; b < TOTAL_BUCKETS; b++) buckets.push_back(b);
        printf("  checking all %" PRIu64 " buckets\n", (uint64_t)TOTAL_BUCKETS);
    }

    uint64_t slots = 0, filled = 0, empty = 0;
    uint64_t bad_range = 0, bad_t1_bucket = 0, bad_distance = 0,
             bad_order = 0, bad_t2_bucket = 0;
    long printed = 0;

    std::vector<MemoTable2Record> buf(rpb);
    const size_t bucket_bytes = rpb * sizeof(MemoTable2Record);

    for (uint64_t bi = 0; bi < buckets.size(); bi++) {
        const uint64_t b = buckets[bi];
        const off_t off = static_cast<off_t>(b * bucket_bytes);

        size_t got = 0;
        while (got < bucket_bytes) {
            ssize_t n = pread(fd, reinterpret_cast<char*>(buf.data()) + got,
                              bucket_bytes - got, off + static_cast<off_t>(got));
            if (n == 0) break;
            if (n < 0) {
                if (errno == EINTR) continue;
                fprintf(stderr, "Error reading bucket %" PRIu64 ": %s\n", b, strerror(errno));
                close(fd);
                return 2;
            }
            got += static_cast<size_t>(n);
        }
        if (got < bucket_bytes) {
            fprintf(stderr, "Error: short read at bucket %" PRIu64 " (file truncated?)\n", b);
            close(fd);
            return 2;
        }

        for (uint32_t s = 0; s < rpb; s++) {
            slots++;
            const MemoTable2Record& r = buf[s];

            bool all_zero = true;
            for (int i = 0; i < NONCE_SIZE && all_zero; i++) {
                if (r.nonce1[i] != 0 || r.nonce2[i] != 0) all_zero = false;
            }
            if (all_zero) { empty++; continue; }
            filled++;

            const char* fail = nullptr;

            uint64_t n1 = nonce_to_uint64(r.nonce1);
            uint64_t n2 = nonce_to_uint64(r.nonce2);
            if (n1 >= N || n2 >= N) { bad_range++; fail = "nonce out of range"; }

            uint8_t h1[HASH_SIZE], h2[HASH_SIZE], t2[HASH_SIZE];
            blake3_keyed_hash(r.nonce1, NONCE_SIZE, key_words, h1, HASH_SIZE);
            blake3_keyed_hash(r.nonce2, NONCE_SIZE, key_words, h2, HASH_SIZE);

            if (!fail && getBucketIndex(h1) != getBucketIndex(h2)) {
                bad_t1_bucket++; fail = "nonces are not from the same Table1 bucket";
            }

            if (!fail) {
                uint64_t k1 = hash_to_uint64(h1, HASH_SIZE);
                uint64_t k2 = hash_to_uint64(h2, HASH_SIZE);
                if (k2 < k1) {
                    bad_order++; fail = "pair stored out of hash order";
                } else if (k2 - k1 > expected_distance) {
                    bad_distance++; fail = "hash distance exceeds the match threshold";
                }
            }

            if (!fail) {
                uint8_t pair[NONCE_SIZE * 2];
                memcpy(pair, r.nonce1, NONCE_SIZE);
                memcpy(pair + NONCE_SIZE, r.nonce2, NONCE_SIZE);
                blake3_keyed_hash(pair, NONCE_SIZE * 2, key_words, t2, HASH_SIZE);
                if (getBucketIndex(t2) != b) {
                    bad_t2_bucket++; fail = "pair is stored in the wrong Table2 bucket";
                }
            }

            if (fail && printed < max_errors) {
                printf("  FAIL bucket %" PRIu64 " slot %u: %s (nonce1=%" PRIu64
                       ", nonce2=%" PRIu64 ")\n", b, s, fail, n1, n2);
                printed++;
            }
        }
    }
    close(fd);

    uint64_t bad = bad_range + bad_t1_bucket + bad_distance + bad_order + bad_t2_bucket;

    printf("\n=== Validation summary ===\n");
    printf("slots checked        %" PRIu64 "\n", slots);
    printf("records present      %" PRIu64 "\n", filled);
    printf("empty slots          %" PRIu64 "\n", empty);
    printf("storage efficiency   %.4f%%\n", slots ? 100.0 * (double)filled / (double)slots : 0.0);
    printf("invalid records      %" PRIu64 "\n", bad);
    if (bad) {
        printf("  nonce out of range        %" PRIu64 "\n", bad_range);
        printf("  wrong Table1 bucket       %" PRIu64 "\n", bad_t1_bucket);
        printf("  out of hash order         %" PRIu64 "\n", bad_order);
        printf("  distance over threshold   %" PRIu64 "\n", bad_distance);
        printf("  wrong Table2 bucket       %" PRIu64 "\n", bad_t2_bucket);
    }
    if (printed >= max_errors && bad > (uint64_t)printed)
        printf("(%" PRIu64 " further failures not printed)\n", bad - (uint64_t)printed);

    bool ok = (bad == 0) && size_ok;
    printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
