#include "gpu_context_sycl.h"
#include <cstdio>
#include <cstring>


// Get SYCL device by index
static sycl::device get_sycl_device(int device_id) {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) {
        fprintf(stderr, "No SYCL GPU devices found.\n");
        return sycl::device{sycl::default_selector_v};
    }
    if (device_id >= static_cast<int>(devices.size())) {
        fprintf(stderr, "Device %d not found, using device 0.\n", device_id);
        device_id = 0;
    }
    return devices[device_id];
}

size_t gpu_query_device(int device_id) {
    auto dev = get_sycl_device(device_id);
    return dev.get_info<sycl::info::device::global_mem_size>();
}

void gpu_print_device_info(int device_id) {
    auto dev = get_sycl_device(device_id);
    auto name = dev.get_info<sycl::info::device::name>();
    size_t total_mem = dev.get_info<sycl::info::device::global_mem_size>();
    printf("GPU: %s (%zu MB total)\n", name.c_str(), total_mem / (1024 * 1024));
}

int gpu_init(SyclGPUContext& ctx, int K, const uint32_t* key_words, int device_id) {
    ctx.device_id = device_id;
    ctx.K = K;
    ctx.N = 1ULL << K;
    ctx.records_per_bucket = static_cast<uint32_t>(ctx.N / TOTAL_BUCKETS);

    auto dev = get_sycl_device(device_id);
    auto name = dev.get_info<sycl::info::device::name>();
    strncpy(ctx.device_name, name.c_str(), sizeof(ctx.device_name) - 1);
    ctx.device_name[sizeof(ctx.device_name) - 1] = '\0';

    ctx.total_mem = dev.get_info<sycl::info::device::global_mem_size>();
    ctx.free_mem = ctx.total_mem; // SYCL doesn't have a direct "free mem" query

    memcpy(ctx.key_words, key_words, 8 * sizeof(uint32_t));

    // Create queue
    ctx.q = new sycl::queue(dev, sycl::property::queue::in_order{});

    // Allocate device memory (USM)
    ctx.d_table1 = sycl::malloc_device<MemoRecord>(ctx.N, *ctx.q);
    ctx.d_table1_counters = sycl::malloc_device<uint32_t>(TOTAL_BUCKETS, *ctx.q);
    ctx.d_table2 = sycl::malloc_device<MemoTable2Record>(ctx.N, *ctx.q);
    ctx.d_table2_counters = sycl::malloc_device<uint32_t>(TOTAL_BUCKETS, *ctx.q);
    ctx.d_key_words = sycl::malloc_device<uint32_t>(8, *ctx.q);

    if (!ctx.d_table1 || !ctx.d_table1_counters || !ctx.d_table2 ||
        !ctx.d_table2_counters || !ctx.d_key_words) {
        fprintf(stderr, "SYCL device memory allocation failed.\n");
        return -1;
    }

    // Initialize
    ctx.q->memset(ctx.d_table1_counters, 0, TOTAL_BUCKETS * sizeof(uint32_t));
    ctx.q->memset(ctx.d_table2, 0, ctx.N * sizeof(MemoTable2Record));
    ctx.q->memset(ctx.d_table2_counters, 0, TOTAL_BUCKETS * sizeof(uint32_t));
    ctx.q->memcpy(ctx.d_key_words, key_words, 8 * sizeof(uint32_t));
    ctx.q->wait();

    // Allocate host buffer
    ctx.h_table2 = new MemoTable2Record[ctx.N];
    memset(ctx.h_table2, 0, ctx.N * sizeof(MemoTable2Record));

    return 0;
}

// D2H Transfer

void gpu_get_table2(SyclGPUContext& ctx) {
    ctx.q->memcpy(ctx.h_table2, ctx.d_table2,
                  ctx.N * sizeof(MemoTable2Record));
    ctx.q->wait();
}

// Cleanup


void gpu_cleanup(SyclGPUContext& ctx) {
    if (ctx.d_table1)          sycl::free(ctx.d_table1, *ctx.q);
    if (ctx.d_table1_counters) sycl::free(ctx.d_table1_counters, *ctx.q);
    if (ctx.d_table2)          sycl::free(ctx.d_table2, *ctx.q);
    if (ctx.d_table2_counters) sycl::free(ctx.d_table2_counters, *ctx.q);
    if (ctx.d_key_words)       sycl::free(ctx.d_key_words, *ctx.q);
    if (ctx.h_table2)          delete[] ctx.h_table2;

    delete ctx.q;

    ctx.d_table1 = nullptr;
    ctx.d_table1_counters = nullptr;
    ctx.d_table2 = nullptr;
    ctx.d_table2_counters = nullptr;
    ctx.d_key_words = nullptr;
    ctx.h_table2 = nullptr;
    ctx.q = nullptr;
}
