#ifndef VAULTXGPU_PLOT_CHECK_H
#define VAULTXGPU_PLOT_CHECK_H

#include "globals.h"
#include <cstdint>
#include <cstddef>
#include <cstdio>

// ──────────────────────────────────────────────
// Structural validation of a finished plot
//
// Needs nothing but the plot file and its ID: no GPU, no second plot, no
// reference implementation. For every stored pair it recomputes the hashes and
// asserts the four properties the construction guarantees:
//
//   1. both nonces are below 2^K
//   2. both nonces hash into the same Table1 bucket -- they were matched there
//   3. their 64-bit keys are ordered and within expected_distance for this K
//   4. the pair hashes into the Table2 bucket it is physically stored in
//
// Checks 2 and 3 are the ones that test the sort and match logic. The CPU
// prover's own verify pass checks neither: it confirms storage efficiency and
// that Table2 prefixes are non-decreasing, both of which a vault full of invalid
// matches would still satisfy.
//
// Empty slots are counted along the way, which gives storage efficiency.
//
// Record order within a bucket is deliberately NOT checked. Slots are claimed
// with an atomic, so order varies between runs on the same device.
// ──────────────────────────────────────────────

struct PlotCheckConfig {
    uint64_t sample      = 0;      // 0 = every bucket; otherwise N random buckets
    uint64_t seed        = 1;      // seed for the sample
    long     max_errors  = 20;     // stop printing individual failures after this
    const uint64_t* buckets = nullptr;  // check only these buckets, if given
    size_t   bucket_count   = 0;
    bool     quiet       = false;  // suppress the per-record failure lines
};

struct PlotCheckResult {
    uint64_t slots_checked = 0;
    uint64_t records       = 0;
    uint64_t empty         = 0;
    double   storage_efficiency = 0.0;

    uint64_t bad_range     = 0;    // nonce >= 2^K
    uint64_t bad_t1_bucket = 0;    // nonces not from the same Table1 bucket
    uint64_t bad_order     = 0;    // pair stored out of hash order
    uint64_t bad_distance  = 0;    // hash distance over the match threshold
    uint64_t bad_t2_bucket = 0;    // pair in the wrong Table2 bucket
    uint64_t invalid       = 0;    // sum of the above

    uint64_t file_bytes     = 0;
    uint64_t expected_bytes = 0;
    bool     size_ok = false;
    bool     ok      = false;      // no invalid records and the size is right
};

// Validate `path` for the given K and plot ID. Progress and failures go to
// `out` (pass nullptr for silence). Returns 0 if everything checked passed,
// 1 if any record failed or the size is wrong, 2 if the file could not be read.
int plot_check_run(const char* path, int K, const uint8_t* plot_id,
                   const PlotCheckConfig& cfg, PlotCheckResult& result,
                   FILE* out);

// One-line summary, e.g.
//   validate: PASS  1536000 slots, 0 empty, SE 100.0000%, 0 invalid
void plot_check_print_summary(FILE* out, const PlotCheckResult& r);

// Parse "k{K}-{64 hex}.plot" out of a path. Returns false if it does not match.
bool plot_check_parse_name(const char* path, int& K, uint8_t plot_id[32]);

#endif // VAULTXGPU_PLOT_CHECK_H
