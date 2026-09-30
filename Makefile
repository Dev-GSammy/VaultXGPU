# VaultXGPU Makefile
# Builds vaultx_cuda (NVIDIA), vaultx_sycl (Intel/AMD/NVIDIA) and vaultx_validate

# Compile-time configuration (override on command line if needed)
NONCE_SIZE  ?= 4
RECORD_SIZE ?= 12

# Table2 bucket sort: 0 = insertion over records (published baseline),
#                     1 = insertion over (key,index),
#                     2 = block-parallel bitonic  [default]
VAULTX_SORT ?= 2

# Common flags
COMMON_DEFS   = -DNONCE_SIZE=$(NONCE_SIZE) -DRECORD_SIZE=$(RECORD_SIZE) \
                -DVAULTX_SORT=$(VAULTX_SORT)
COMMON_CFLAGS = -O3 -std=c++17 $(COMMON_DEFS)

CXX ?= g++

# Sources
COMMON_SRCS = src/common/main.cpp src/common/crypto_cpu.cpp \
              src/common/memory.cpp src/common/plot_io.cpp \
              src/common/metrics.cpp src/common/plot_writer.cpp

# Host-only translation units: no device code, so they are compiled by the host
# compiler in both builds. plot_writer.cpp uses <thread>, which is cleaner to keep
# away from nvcc's -x cu front end.
HOST_ONLY_SRCS = src/common/metrics.cpp src/common/plot_writer.cpp

# Libraries
LIBS = -lsodium -lpthread

# Binary suffix so sort variants can coexist (empty for the default build)
ifeq ($(VAULTX_SORT),2)
  SORT_SUFFIX =
else
  SORT_SUFFIX = _sort$(VAULTX_SORT)
endif
BUILD_DIR = build/s$(VAULTX_SORT)


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

cuda: vaultx_cuda$(SORT_SUFFIX)

# Objects containing device code -- these take part in the device link step
CUDA_DEV_OBJS = $(BUILD_DIR)/cuda/main.o $(BUILD_DIR)/cuda/crypto_cpu.o \
                $(BUILD_DIR)/cuda/memory.o $(BUILD_DIR)/cuda/plot_io.o \
                $(BUILD_DIR)/cuda/gpu_context_cuda.o \
                $(BUILD_DIR)/cuda/table1_cuda.o $(BUILD_DIR)/cuda/sort_table2_cuda.o

# Host-only objects -- linked in, but kept out of -dlink
CUDA_HOST_OBJS = $(BUILD_DIR)/cuda/metrics.o $(BUILD_DIR)/cuda/plot_writer.o

# CUDA uses separate compilation (-dc) for device linking of __constant__ symbols
vaultx_cuda$(SORT_SUFFIX): $(CUDA_DEV_OBJS) $(CUDA_HOST_OBJS)
	$(NVCC) $(CUDA_FLAGS) -dlink $(CUDA_DEV_OBJS) -o $(BUILD_DIR)/cuda/dlink.o
	$(NVCC) $(CUDA_FLAGS) $(CUDA_DEV_OBJS) $(CUDA_HOST_OBJS) \
	    $(BUILD_DIR)/cuda/dlink.o -o $@ $(LIBS)

$(BUILD_DIR)/cuda/main.o: src/common/main.cpp | $(BUILD_DIR)/cuda
	$(NVCC) $(CUDA_FLAGS) -dc -x cu $< -o $@

$(BUILD_DIR)/cuda/crypto_cpu.o: src/common/crypto_cpu.cpp | $(BUILD_DIR)/cuda
	$(NVCC) $(CUDA_FLAGS) -dc -x cu $< -o $@

$(BUILD_DIR)/cuda/memory.o: src/common/memory.cpp | $(BUILD_DIR)/cuda
	$(NVCC) $(CUDA_FLAGS) -dc -x cu $< -o $@

$(BUILD_DIR)/cuda/plot_io.o: src/common/plot_io.cpp | $(BUILD_DIR)/cuda
	$(NVCC) $(CUDA_FLAGS) -dc -x cu $< -o $@

$(BUILD_DIR)/cuda/metrics.o: src/common/metrics.cpp | $(BUILD_DIR)/cuda
	$(CXX) $(COMMON_CFLAGS) -DGPU_CUDA=1 -Isrc -c $< -o $@

$(BUILD_DIR)/cuda/plot_writer.o: src/common/plot_writer.cpp | $(BUILD_DIR)/cuda
	$(CXX) $(COMMON_CFLAGS) -DGPU_CUDA=1 -Isrc -c $< -o $@

$(BUILD_DIR)/cuda/gpu_context_cuda.o: src/cuda/gpu_context_cuda.cu | $(BUILD_DIR)/cuda
	$(NVCC) $(CUDA_FLAGS) -dc $< -o $@

$(BUILD_DIR)/cuda/table1_cuda.o: src/cuda/table1_cuda.cu | $(BUILD_DIR)/cuda
	$(NVCC) $(CUDA_FLAGS) -dc $< -o $@

$(BUILD_DIR)/cuda/sort_table2_cuda.o: src/cuda/sort_table2_cuda.cu | $(BUILD_DIR)/cuda
	$(NVCC) $(CUDA_FLAGS) -dc $< -o $@

$(BUILD_DIR)/cuda:
	mkdir -p $(BUILD_DIR)/cuda

# SYCL build
ICPX       ?= icpx
SYCL_FLAGS  = $(COMMON_CFLAGS) -DGPU_SYCL=1 -fsycl -Isrc

SYCL_SRCS   = src/sycl/gpu_context_sycl.cpp src/sycl/table1_sycl.cpp \
              src/sycl/sort_table2_sycl.cpp

sycl: vaultx_sycl$(SORT_SUFFIX)

vaultx_sycl$(SORT_SUFFIX): $(COMMON_SRCS) $(SYCL_SRCS)
	$(ICPX) $(SYCL_FLAGS) $(COMMON_SRCS) $(SYCL_SRCS) -o $@ $(LIBS)

# Validator (host only, no GPU toolchain required)
VALIDATE_SRCS = src/tools/validate_plot.cpp src/common/crypto_cpu.cpp

validate: vaultx_validate

vaultx_validate: $(VALIDATE_SRCS)
	$(CXX) $(COMMON_CFLAGS) -Isrc $(VALIDATE_SRCS) -o $@ -lsodium

# Sort variants for the sort-comparison study: builds vaultx_cuda_sort0 and
# vaultx_cuda_sort1 alongside the default vaultx_cuda.
cuda-sort-variants:
	$(MAKE) VAULTX_SORT=0 cuda
	$(MAKE) VAULTX_SORT=1 cuda
	$(MAKE) VAULTX_SORT=2 cuda

sycl-sort-variants:
	$(MAKE) VAULTX_SORT=0 sycl
	$(MAKE) VAULTX_SORT=1 sycl
	$(MAKE) VAULTX_SORT=2 sycl

# Cleaning
all: cuda validate

clean:
	rm -f vaultx_cuda vaultx_cuda_sort0 vaultx_cuda_sort1 \
	      vaultx_sycl vaultx_sycl_sort0 vaultx_sycl_sort1 \
	      vaultx_validate
	rm -rf build/

.PHONY: cuda sycl all clean validate cuda-sort-variants sycl-sort-variants
