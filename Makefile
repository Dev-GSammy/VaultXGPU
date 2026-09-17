# VaultXGPU Makefile
# Builds vaultx_cuda (NVIDIA) and/or vaultx_sycl (Intel/AMD/NVIDIA)

# Compile-time configuration (override on command line if needed)
NONCE_SIZE  ?= 4
RECORD_SIZE ?= 12

# Common flags
COMMON_DEFS   = -DNONCE_SIZE=$(NONCE_SIZE) -DRECORD_SIZE=$(RECORD_SIZE)
COMMON_CFLAGS = -O3 -std=c++17 $(COMMON_DEFS)

# Sources
COMMON_SRCS = src/common/main.cpp src/common/crypto_cpu.cpp \
              src/common/memory.cpp src/common/plot_io.cpp

# Libraries
LIBS = -lsodium


# CUDA build

NVCC       ?= nvcc
CUDA_ARCH  ?= -gencode arch=compute_70,code=sm_70 \
              -gencode arch=compute_75,code=sm_75 \
              -gencode arch=compute_80,code=sm_80 \
              -gencode arch=compute_86,code=sm_86
CUDA_FLAGS  = $(COMMON_CFLAGS) -DGPU_CUDA=1 $(CUDA_ARCH) \
              --expt-relaxed-constexpr -rdc=true -Isrc

CUDA_SRCS   = src/cuda/gpu_context_cuda.cu src/cuda/table1_cuda.cu \
              src/cuda/sort_table2_cuda.cu

cuda: vaultx_cuda

# CUDA uses separate compilation (-dc) for device linking of __constant__ symbols
CUDA_OBJS = build/cuda/main.o build/cuda/crypto_cpu.o build/cuda/memory.o \
            build/cuda/plot_io.o build/cuda/gpu_context_cuda.o \
            build/cuda/table1_cuda.o build/cuda/sort_table2_cuda.o

vaultx_cuda: $(CUDA_OBJS)
	$(NVCC) $(CUDA_FLAGS) -dlink $(CUDA_OBJS) -o build/cuda/dlink.o
	$(NVCC) $(CUDA_FLAGS) $(CUDA_OBJS) build/cuda/dlink.o -o $@ $(LIBS)

build/cuda/main.o: src/common/main.cpp | build/cuda
	$(NVCC) $(CUDA_FLAGS) -dc -x cu $< -o $@

build/cuda/crypto_cpu.o: src/common/crypto_cpu.cpp | build/cuda
	$(NVCC) $(CUDA_FLAGS) -dc -x cu $< -o $@

build/cuda/memory.o: src/common/memory.cpp | build/cuda
	$(NVCC) $(CUDA_FLAGS) -dc -x cu $< -o $@

build/cuda/plot_io.o: src/common/plot_io.cpp | build/cuda
	$(NVCC) $(CUDA_FLAGS) -dc -x cu $< -o $@

build/cuda/gpu_context_cuda.o: src/cuda/gpu_context_cuda.cu | build/cuda
	$(NVCC) $(CUDA_FLAGS) -dc $< -o $@

build/cuda/table1_cuda.o: src/cuda/table1_cuda.cu | build/cuda
	$(NVCC) $(CUDA_FLAGS) -dc $< -o $@

build/cuda/sort_table2_cuda.o: src/cuda/sort_table2_cuda.cu | build/cuda
	$(NVCC) $(CUDA_FLAGS) -dc $< -o $@

build/cuda:
	mkdir -p build/cuda

# SYCL build
ICPX       ?= icpx
SYCL_FLAGS  = $(COMMON_CFLAGS) -DGPU_SYCL=1 -fsycl -Isrc

SYCL_SRCS   = src/sycl/gpu_context_sycl.cpp src/sycl/table1_sycl.cpp \
              src/sycl/sort_table2_sycl.cpp

sycl: vaultx_sycl

vaultx_sycl: $(COMMON_SRCS) $(SYCL_SRCS)
	$(ICPX) $(SYCL_FLAGS) $(COMMON_SRCS) $(SYCL_SRCS) -o $@ $(LIBS)

# Cleaning
all: cuda

clean:
	rm -f vaultx_cuda vaultx_sycl
	rm -rf build/

.PHONY: cuda sycl all clean
