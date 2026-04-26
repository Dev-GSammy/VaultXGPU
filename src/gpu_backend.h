#ifndef VAULTXGPU_GPU_BACKEND_H
#define VAULTXGPU_GPU_BACKEND_H

//
// Compile-time backend selection.
// main.cpp includes this file and gets the correct GPU context type + API.
//

#if defined(GPU_CUDA)
    #include "cuda/gpu_context_cuda.h"
    using GPUContext = CudaGPUContext;

#elif defined(GPU_SYCL)
    #include "sycl/gpu_context_sycl.h"
    using GPUContext = SyclGPUContext;

#else
    #error "Define GPU_CUDA or GPU_SYCL at compile time"
#endif

// Both backends expose the same API:
//   size_t gpu_query_device(int device_id);
//   void   gpu_print_device_info(int device_id);
//   int    gpu_init(GPUContext& ctx, int K, const uint32_t* key_words, int device_id);
//   void   gpu_generate_table1(GPUContext& ctx);
//   void   gpu_free_table1(GPUContext& ctx);   // free table1 VRAM after sort+match
//   void   gpu_sort_and_match(GPUContext& ctx);
//   int    gpu_write_table2(GPUContext& ctx, int K, const uint8_t* plot_id, const char* output_dir);
//   void   gpu_cleanup(GPUContext& ctx);

#endif // VAULTXGPU_GPU_BACKEND_H
