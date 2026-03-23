#include "globals.h"
#include "crypto_cpu.h"
#include "memory.h"
#include "plot_io.h"
#include "../gpu_backend.h"

#include <sodium.h>
#include <getopt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

// ──────────────────────────────────────────────
// CLI option parsing
// ──────────────────────────────────────────────

struct Options {
    int  K          = 0;       // Required
    char file[512]  = {0};     // Required: output file path (directory)
    char tmpdir[512]  = {0};   // -g: accepted for compat, unused
    char tmpdir2[512] = {0};   // -j: accepted for compat, unused
    int  device     = 0;       // -d: GPU device index
    bool benchmark  = false;   // -b: benchmark mode
    bool verify     = false;   // -v: verify after generation
};

static void print_usage(const char* prog) {
    printf("Usage: %s -k <ksize> -f <output_dir> [options]\n", prog);
    printf("\nRequired:\n");
    printf("  -k, --ksize NUM       K value (exponent, e.g. 25-32)\n");
    printf("  -f, --file PATH       Output directory for plot file\n");
    printf("\nOptional:\n");
    printf("  -g, --tmpdir PATH     Temp dir (accepted for CLI compat, unused)\n");
    printf("  -j, --tmpdir2 PATH    Temp dir 2 (accepted for CLI compat, unused)\n");
    printf("  -d, --device NUM      GPU device index (default: 0)\n");
    printf("  -b, --benchmark       Benchmark mode (machine-readable output)\n");
    printf("  -v, --verify          Verify plot after generation\n");
    printf("  -h, --help            Show this help\n");
}

static int parse_args(int argc, char** argv, Options& opts) {
    static struct option long_options[] = {
        {"ksize",     required_argument, 0, 'k'},
        {"file",      required_argument, 0, 'f'},
        {"tmpdir",    required_argument, 0, 'g'},
        {"tmpdir2",   required_argument, 0, 'j'},
        {"device",    required_argument, 0, 'd'},
        {"benchmark", no_argument,       0, 'b'},
        {"verify",    no_argument,       0, 'v'},
        {"help",      no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "k:f:g:j:d:bvh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'k': opts.K = atoi(optarg); break;
            case 'f': strncpy(opts.file, optarg, sizeof(opts.file) - 1); break;
            case 'g': strncpy(opts.tmpdir, optarg, sizeof(opts.tmpdir) - 1); break;
            case 'j': strncpy(opts.tmpdir2, optarg, sizeof(opts.tmpdir2) - 1); break;
            case 'd': opts.device = atoi(optarg); break;
            case 'b': opts.benchmark = true; break;
            case 'v': opts.verify = true; break;
            case 'h': print_usage(argv[0]); return -1;
            default:  print_usage(argv[0]); return -1;
        }
    }

    if (opts.K <= 0) {
        fprintf(stderr, "Error: -k (ksize) is required and must be > 0\n");
        print_usage(argv[0]);
        return -1;
    }
    if (opts.file[0] == '\0') {
        fprintf(stderr, "Error: -f (output directory) is required\n");
        print_usage(argv[0]);
        return -1;
    }
    return 0;
}


// Main


int main(int argc, char** argv) {
    Options opts;
    if (parse_args(argc, argv, opts) != 0) {
        return 1;
    }

    // Initialize libsodium
    if (sodium_init() < 0) {
        fprintf(stderr, "Error: libsodium initialization failed\n");
        return 1;
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    // ── Phase 1: CPU staging ──

    // 1. Query GPU
    gpu_print_device_info(opts.device);
    size_t available_mem = gpu_query_device(opts.device);

    // 2. Memory check
    print_memory_budget(opts.K, available_mem);
    if (!can_fit_in_memory(opts.K, available_mem)) {
        fprintf(stderr, "\nError: K=%d does not fit in available GPU memory.\n", opts.K);
        return 1;
    }
    printf("\n");

    // 3. Generate plot ID and derive key
    uint8_t plot_id[32];
    uint8_t key[32];
    uint32_t key_words[8];

    generate_plot_id(plot_id);
    derive_key(opts.K, plot_id, key);
    key_to_words(key, key_words);

    char* hex_id = byteArrayToHexString(plot_id, 32);
    printf("Plot ID: %s\n", hex_id ? hex_id : "unknown");
    free(hex_id);

    // ── Phase 2 & 3: GPU computation ──

    GPUContext ctx;
    if (gpu_init(ctx, opts.K, key_words, opts.device) != 0) {
        fprintf(stderr, "Error: GPU initialization failed\n");
        return 1;
    }

    printf("\n--- Phase 2: Table1 Generation ---\n");
    auto t1_start = std::chrono::high_resolution_clock::now();
    gpu_generate_table1(ctx);
    auto t1_end = std::chrono::high_resolution_clock::now();
    double t1_sec = std::chrono::duration<double>(t1_end - t1_start).count();
    printf("Table1 done: %.3f seconds\n", t1_sec);

    printf("\n--- Phase 3: Sort + Table2 Generation ---\n");
    auto t2_start = std::chrono::high_resolution_clock::now();
    gpu_sort_and_match(ctx);
    auto t2_end = std::chrono::high_resolution_clock::now();
    double t2_sec = std::chrono::duration<double>(t2_end - t2_start).count();
    printf("Sort+Table2 done: %.3f seconds\n", t2_sec);

    // ── Phase 4: D2H transfer + write ──

    printf("\n--- Phase 4: Transfer + Write ---\n");
    auto w_start = std::chrono::high_resolution_clock::now();

    gpu_get_table2(ctx);

    uint64_t total_nonces = 1ULL << opts.K;
    int rc = write_plot_file(ctx.h_table2, total_nonces, opts.K, plot_id, opts.file);

    auto w_end = std::chrono::high_resolution_clock::now();
    double w_sec = std::chrono::duration<double>(w_end - w_start).count();
    printf("Transfer+Write done: %.3f seconds\n", w_sec);

    // Cleanup
    gpu_cleanup(ctx);

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(total_end - total_start).count();

    // ── Summary ──
    printf("\n=== Summary ===\n");
    printf("K=%d, Nonces=%llu\n", opts.K, (unsigned long long)total_nonces);
    printf("Table1:       %.3f s\n", t1_sec);
    printf("Sort+Table2:  %.3f s\n", t2_sec);
    printf("Write:        %.3f s\n", w_sec);
    printf("Total:        %.3f s\n", total_sec);

    if (opts.benchmark) {
        // Machine-readable single line
        printf("BENCHMARK: K=%d table1=%.3f sort_table2=%.3f write=%.3f total=%.3f\n",
               opts.K, t1_sec, t2_sec, w_sec, total_sec);
    }

    if (opts.verify) {
        printf("\nVerification: Use CPU VaultX to search the generated plot file.\n");
    }

    return rc;
}
