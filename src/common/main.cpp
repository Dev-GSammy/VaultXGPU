#include "globals.h"
#include "crypto_cpu.h"
#include "memory.h"
#include "metrics.h"
#include "plot_writer.h"
#include "plot_io.h"
#include "../gpu_backend.h"

#include <sodium.h>
#include <getopt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(GPU_CUDA)
static const char* kBackendName = "cuda";
#elif defined(GPU_SYCL)
static const char* kBackendName = "sycl";
#else
static const char* kBackendName = "unknown";
#endif

// ──────────────────────────────────────────────
// CLI option parsing
// ──────────────────────────────────────────────

// Long-only option ids.
enum {
    OPT_STATS = 1000,
    OPT_CSV,
    OPT_CSV_HEADER,
    OPT_CHUNK_MB,
    OPT_O_DIRECT,
    OPT_NO_OVERLAP,
    OPT_NO_FSYNC,
    OPT_REQUIRE_GPU,
    OPT_RUN
};

struct Options {
    int  K          = 0;       // Required
    char file[512]  = {0};     // Required: output file path (directory)
    char tmpdir[512]  = {0};   // -g: accepted for compat, unused
    char tmpdir2[512] = {0};   // -j: accepted for compat, unused
    int  device     = 0;       // -d: GPU device index
    bool benchmark  = false;   // -b: benchmark mode
    bool verify     = false;   // -v: verify after generation

    bool stats       = false;  // --stats: bucket counter readback
    char csv[512]    = {0};    // --csv PATH: append a CSV row
    bool csv_header  = false;  // --csv-header: print the schema and exit
    size_t chunk_mb  = 256;    // --chunk-mb: output staging chunk size
    bool o_direct    = false;  // --o-direct: bypass the page cache
    bool overlap     = true;   // --no-overlap: serialize transfer and write
    bool fsync_at_end = true;  // --no-fsync
    bool require_gpu = false;  // --require-gpu: fail instead of running on a CPU device
    int  run_index   = 0;      // --run N: repeat index, recorded in the CSV
};

static void print_usage(const char* prog) {
    printf("Usage: %s -k <ksize> -f <output_dir> [options]\n", prog);
    printf("\nRequired:\n");
    printf("  -k, --ksize NUM       K value (exponent, %d-%d)\n", MIN_K, MAX_K);
    printf("  -f, --file PATH       Output directory for plot file\n");
    printf("\nOptional:\n");
    printf("  -g, --tmpdir PATH     Temp dir (accepted for CLI compat, unused)\n");
    printf("  -j, --tmpdir2 PATH    Temp dir 2 (accepted for CLI compat, unused)\n");
    printf("  -d, --device NUM      GPU device index (default: 0)\n");
    printf("  -b, --benchmark       Benchmark mode (machine-readable summary line)\n");
    printf("  -v, --verify          Print how to verify the plot after generation\n");
    printf("  -h, --help            Show this help\n");
    printf("\nMeasurement:\n");
    printf("      --stats           Read bucket counters back and report occupancy,\n");
    printf("                        overflow and storage efficiency. Adds two 64 MB\n");
    printf("                        transfers, excluded from the reported total.\n");
    printf("      --csv PATH        Append one CSV row per run ('-' for stdout)\n");
    printf("      --csv-header      Print the CSV schema and exit\n");
    printf("      --run N           Repeat index recorded in the CSV row\n");
    printf("\nOutput pipeline:\n");
    printf("      --chunk-mb N      Staging chunk size in MB (default: 256)\n");
    printf("      --o-direct        Open the plot with O_DIRECT so write timing\n");
    printf("                        measures the device and not the page cache\n");
    printf("      --no-overlap      Serialize transfer and write (for the PCIe-vs-disk\n");
    printf("                        decomposition: d2h_s + write_s then equals wall)\n");
    printf("      --no-fsync        Skip the final fdatasync\n");
    printf("      --require-gpu     Fail if the backend selects a non-GPU device\n");
}

static int parse_args(int argc, char** argv, Options& opts) {
    static struct option long_options[] = {
        {"ksize",       required_argument, 0, 'k'},
        {"file",        required_argument, 0, 'f'},
        {"tmpdir",      required_argument, 0, 'g'},
        {"tmpdir2",     required_argument, 0, 'j'},
        {"device",      required_argument, 0, 'd'},
        {"benchmark",   no_argument,       0, 'b'},
        {"verify",      no_argument,       0, 'v'},
        {"help",        no_argument,       0, 'h'},
        {"stats",       no_argument,       0, OPT_STATS},
        {"csv",         required_argument, 0, OPT_CSV},
        {"csv-header",  no_argument,       0, OPT_CSV_HEADER},
        {"chunk-mb",    required_argument, 0, OPT_CHUNK_MB},
        {"o-direct",    no_argument,       0, OPT_O_DIRECT},
        {"no-overlap",  no_argument,       0, OPT_NO_OVERLAP},
        {"no-fsync",    no_argument,       0, OPT_NO_FSYNC},
        {"require-gpu", no_argument,       0, OPT_REQUIRE_GPU},
        {"run",         required_argument, 0, OPT_RUN},
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

            case OPT_STATS:       opts.stats = true; break;
            case OPT_CSV:         strncpy(opts.csv, optarg, sizeof(opts.csv) - 1); break;
            case OPT_CSV_HEADER:  opts.csv_header = true; break;
            case OPT_CHUNK_MB:    opts.chunk_mb = strtoull(optarg, nullptr, 10); break;
            case OPT_O_DIRECT:    opts.o_direct = true; break;
            case OPT_NO_OVERLAP:  opts.overlap = false; break;
            case OPT_NO_FSYNC:    opts.fsync_at_end = false; break;
            case OPT_REQUIRE_GPU: opts.require_gpu = true; break;
            case OPT_RUN:         opts.run_index = atoi(optarg); break;

            default:  print_usage(argv[0]); return -1;
        }
    }

    if (opts.csv_header) return 0;   // nothing else is required

    // K outside [MIN_K, MAX_K] has no valid matching factor, so a run would use a
    // placeholder value and silently produce a vault with the wrong match density.
    if (opts.K < MIN_K || opts.K > MAX_K) {
        if (opts.K <= 0)
            fprintf(stderr, "Error: -k (ksize) is required\n");
        else
            fprintf(stderr, "Error: K=%d is out of range; supported range is %d-%d "
                            "(the matching-factor table has no entry outside it)\n",
                    opts.K, MIN_K, MAX_K);
        print_usage(argv[0]);
        return -1;
    }
    if (opts.file[0] == '\0') {
        fprintf(stderr, "Error: -f (output directory) is required\n");
        print_usage(argv[0]);
        return -1;
    }
    if (opts.chunk_mb == 0) {
        fprintf(stderr, "Error: --chunk-mb must be > 0\n");
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

    if (opts.csv_header) {
        print_csv_header(stdout);
        return 0;
    }

    // Initialize libsodium
    if (sodium_init() < 0) {
        fprintf(stderr, "Error: libsodium initialization failed\n");
        return 1;
    }

    RunMetrics m;
    const double total_start = now_seconds();

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
    const char* plot_id_hex = hex_id ? hex_id : "unknown";
    printf("Plot ID: %s\n", plot_id_hex);

    // ── Phase 2 & 3: GPU computation ──

    GPUContext ctx;
    const double alloc_start = now_seconds();
    if (gpu_init(ctx, opts.K, key_words, opts.device) != 0) {
        fprintf(stderr, "Error: GPU initialization failed\n");
        free(hex_id);
        return 1;
    }
    m.alloc = now_seconds() - alloc_start;

#if defined(GPU_SYCL)
    // A SYCL run that silently landed on a CPU device must not be reported as a
    // GPU result; --require-gpu turns that into a hard failure.
    if (opts.require_gpu && !ctx.is_gpu) {
        fprintf(stderr, "Error: --require-gpu given but the SYCL runtime selected a "
                        "non-GPU device (%s).\n", ctx.device_name);
        gpu_cleanup(ctx);
        free(hex_id);
        return 1;
    }
    if (!ctx.is_gpu) {
        fprintf(stderr, "Warning: running on a non-GPU SYCL device (%s). Label any "
                        "result from this run as CPU-via-SYCL.\n", ctx.device_name);
    }
#endif

    printf("\n--- Phase 2: Table1 Generation ---\n");
    const double t1_start = now_seconds();
    gpu_generate_table1(ctx);
    m.table1 = now_seconds() - t1_start;
    m.table1_kernel = ctx.table1_kernel_seconds;
    printf("Table1 done: %.3f s (kernel %.3f s)\n", m.table1, m.table1_kernel);

    printf("\n--- Phase 3: Sort + Table2 Generation ---\n");
    const double t2_start = now_seconds();
    gpu_sort_and_match(ctx);
    m.sort = now_seconds() - t2_start;
    m.sort_kernel = ctx.sort_kernel_seconds;
    printf("Sort+Table2 done: %.3f s (kernel %.3f s)\n", m.sort, m.sort_kernel);

    // Counter readback, before Table1's counters are freed. Timed separately and
    // subtracted from the total so --stats does not inflate the pipeline cost.
    double stats_seconds = 0.0;
    if (opts.stats) {
        const double s_start = now_seconds();
        gpu_read_stats(ctx, m);
        stats_seconds = now_seconds() - s_start;
    }

    // Table1 is no longer needed -- free its device memory before the write phase
    gpu_free_table1(ctx);

    // ── Phase 4: Transfer + Write ──

    printf("\n--- Phase 4: Transfer + Write ---\n");

    PlotWriterConfig wcfg;
    wcfg.chunk_bytes  = opts.chunk_mb * 1024ULL * 1024ULL;
    wcfg.o_direct     = opts.o_direct;
    wcfg.overlap      = opts.overlap;
    wcfg.fsync_at_end = opts.fsync_at_end;

    // Kernel I/O counters around the output stage: the independent check on the
    // write timer, and the only way to see write amplification.
    StorageSnapshot io_before = storage_snapshot(opts.file);

    int rc = gpu_write_table2(ctx, opts.K, plot_id, opts.file, wcfg, m);

    StorageSnapshot io_after = storage_snapshot(opts.file);
    StorageDelta    io       = storage_delta(io_before, io_after);

    printf("Transfer+Write done: %.3f s wall (d2h %.3f s, write %.3f s, fsync %.3f s)\n",
           m.write_wall, m.d2h, m.write, m.fsync);

    // Cleanup
    const double td_start = now_seconds();
    gpu_cleanup(ctx);
    m.teardown = now_seconds() - td_start;

    m.total = (now_seconds() - total_start) - stats_seconds;

    // ── Summary ──
    uint64_t total_nonces = 1ULL << opts.K;
    printf("\n=== Summary ===\n");
    printf("K=%d, Nonces=%llu, sort=%s\n",
           opts.K, (unsigned long long)total_nonces, VAULTX_SORT_NAME);
    printf("Alloc:        %.3f s\n", m.alloc);
    printf("Table1:       %.3f s  (kernel %.3f s)\n", m.table1, m.table1_kernel);
    printf("Sort+Table2:  %.3f s  (kernel %.3f s)\n", m.sort, m.sort_kernel);
    printf("D2H:          %.3f s\n", m.d2h);
    printf("Write:        %.3f s\n", m.write);
    printf("Fsync:        %.3f s\n", m.fsync);
    printf("Output wall:  %.3f s%s\n", m.write_wall,
           opts.overlap ? "  (d2h and write overlap)" : "");
    printf("Teardown:     %.3f s\n", m.teardown);
    printf("Total:        %.3f s\n", m.total);

    if (io.valid) {
        double mb = (double)io.bytes_written / (1024.0 * 1024.0);
        printf("Device I/O:   %.1f MB written to %s (%s), amplification %.3fx\n",
               mb, io.label.c_str(), storage_kind_name(io.kind),
               m.bytes_written ? (double)io.bytes_written / (double)m.bytes_written : 0.0);
    } else {
        printf("Device I/O:   counters unavailable for %s (%s)\n",
               io.label.empty() ? "output path" : io.label.c_str(),
               storage_kind_name(io.kind));
    }

    print_stats_summary(stdout, opts.K, m);

    RunContext rctx;
    rctx.K           = opts.K;
    rctx.device      = opts.device;
    rctx.backend     = kBackendName;
    rctx.device_name = ctx.device_name;
    rctx.driver      = ctx.driver_version[0] ? ctx.driver_version : "unknown";
    rctx.plot_id_hex = plot_id_hex;
    rctx.sort_name   = VAULTX_SORT_NAME;
    rctx.chunk_bytes = wcfg.chunk_bytes;
    rctx.o_direct    = opts.o_direct;
    rctx.overlap     = opts.overlap;
    rctx.run_index   = opts.run_index;

    if (opts.benchmark) {
        printf("BENCHMARK: K=%d sort=%s table1=%.3f sort_table2=%.3f d2h=%.3f "
               "write=%.3f fsync=%.3f output_wall=%.3f total=%.3f\n",
               opts.K, VAULTX_SORT_NAME, m.table1, m.sort, m.d2h,
               m.write, m.fsync, m.write_wall, m.total);
    }

    if (opts.csv[0] != '\0') {
        bool to_stdout = (strcmp(opts.csv, "-") == 0);
        FILE* out = to_stdout ? stdout : nullptr;
        bool need_header = to_stdout;
        if (!to_stdout) {
            // Write the header only when creating the file, so repeated runs append.
            FILE* probe = fopen(opts.csv, "r");
            need_header = (probe == nullptr);
            if (probe) fclose(probe);
            out = fopen(opts.csv, "a");
            if (!out) {
                fprintf(stderr, "Warning: could not open %s for the CSV row\n", opts.csv);
            }
        }
        if (out) {
            if (need_header) print_csv_header(out);
            print_csv_row(out, rctx, m, io);
            if (!to_stdout) fclose(out);
        }
    }

    if (opts.verify) {
        char path[512];
        build_plot_path(path, sizeof(path), opts.file, opts.K, plot_id);
        printf("\nVerification:\n");
        printf("  ./vaultx_validate %s\n", path);
    }

    free(hex_id);
    return rc;
}
