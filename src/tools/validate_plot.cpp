// ──────────────────────────────────────────────
// vaultx_validate -- command-line front end for plot_check.
//
// The checking itself lives in common/plot_check.cpp, shared with the plotter's
// -v and -V flags. This binary exists so plots can be validated on a machine
// with no CUDA or SYCL toolchain, where the plotter cannot be built at all.
// ──────────────────────────────────────────────

#include "../common/plot_check.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <vector>

static void usage(const char* prog) {
    printf("Usage: %s <plot-file> [options]\n", prog);
    printf("\nOptions:\n");
    printf("  --sample N     Validate N randomly chosen buckets instead of all\n");
    printf("  --seed N       Seed for --sample (default: 1)\n");
    printf("  --bucket N     Validate only this bucket; repeatable (overrides --sample)\n");
    printf("  --max-errors N Stop printing individual failures after N (default: 20)\n");
    printf("  --plot-id HEX  Override the plot ID parsed from the filename\n");
    printf("  --ksize N      Override K parsed from the filename\n");
    printf("  -h, --help     Show this help\n");
    printf("\nThe same check runs inside the plotter: 'vaultx_cuda -v' after a run,\n");
    printf("or 'vaultx_cuda -V <plot>' on an existing plot.\n");
    printf("\nExit status is 0 only when every checked record passes.\n");
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 2; }
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) { usage(argv[0]); return 0; }

    const char* path = argv[1];
    PlotCheckConfig cfg;
    std::vector<uint64_t> only_buckets;
    int K = 0;
    uint8_t plot_id[32] = {0};
    bool have_id = false;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--sample") == 0 && i + 1 < argc) {
            cfg.sample = strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg.seed = strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--max-errors") == 0 && i + 1 < argc) {
            cfg.max_errors = strtol(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--bucket") == 0 && i + 1 < argc) {
            uint64_t b = strtoull(argv[++i], nullptr, 10);
            if (b >= TOTAL_BUCKETS) {
                fprintf(stderr, "Error: --bucket %" PRIu64 " is out of range (max %" PRIu64 ")\n",
                        b, (uint64_t)TOTAL_BUCKETS - 1);
                return 2;
            }
            only_buckets.push_back(b);
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
    if (plot_check_parse_name(path, parsed_K, parsed_id)) {
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

    if (!only_buckets.empty()) {
        cfg.buckets      = only_buckets.data();
        cfg.bucket_count = only_buckets.size();
    }

    printf("Validating %s\n", path);
    PlotCheckResult r;
    int rc = plot_check_run(path, K, plot_id, cfg, r, stdout);
    if (rc == 2) return 2;
    plot_check_print_summary(stdout, r);
    return rc;
}
