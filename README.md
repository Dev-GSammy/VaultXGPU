# VaultXGPU

GPU-accelerated plot generator for the VaultX PoS protocol. 

VaultXGPU runs the compute-intensive parts of plot generation on the GPU:

1. **Table1 generation**: Hash all 2^K nonces with Blake3 keyed hash, sort them into 2^24 buckets
2. **Sort + Table2 generation**: For each bucket, sort by hash, find close nonce pairs, hash pairs into Table2 buckets
3. **Write**: Transfer Table2 from GPU to host, write to disk as a `.plot` file

The output is a standard VaultX plot file (`k{K}-{hex_plot_id}.plot`) that CPU VaultX can search directly.

## Requirements

### All builds
- Linux (Ubuntu 22.04+ recommended)
- libsodium: `sudo apt install libsodium-dev`
- C++17 compiler

### CUDA build (NVIDIA GPUs)
- CUDA Toolkit >= 11.0 (12.0+ recommended)
- NVIDIA GPU with compute capability >= 7.0 (Volta or newer)
- Install: https://developer.nvidia.com/cuda-toolkit

### SYCL build (Intel/AMD/NVIDIA GPUs)
- Intel oneAPI Base Toolkit (2023.0+) which includes `icpx` and the SYCL runtime
- Install: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html
- For AMD GPUs: also install ROCm (https://rocm.docs.amd.com/) and the oneAPI AMD plugin
- For NVIDIA GPUs: also install the oneAPI NVIDIA plugin

### ROCm for AMD GPUs
1. Add the ROCm repo:
   ```
   wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
   echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.0.2 jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list
   sudo apt update && sudo apt install rocm-hip-runtime rocm-dev
   ```
2. Verify: `rocminfo` should list your GPU
3. Note: ROCm requires native Linux.

## Building

```bash
# NVIDIA CUDA
make cuda

# Intel/AMD/NVIDIA SYCL
make sycl

# Custom nonce/record sizes
make cuda NONCE_SIZE=5 RECORD_SIZE=13

# Clean
make clean
```

This produces:
- `vaultx_cuda` -- NVIDIA GPU binary
- `vaultx_sycl` -- SYCL GPU binary (Intel/AMD/NVIDIA)

## Usage

### Generate a plot

```bash
# Required: -k (K value) and -f (output directory)
./vaultx_cuda -k 27 -f /data/plots

# Specify which GPU (default: 0 = first GPU)
./vaultx_cuda -k 27 -f /data/plots -d 1

# SYCL backend
./vaultx_sycl -k 27 -f /data/plots
```

### Multi-GPU

If you have more than one GPU, use `-d` to select by index:
- `-d 0` -- first GPU (default)
- `-d 1` -- second GPU
- etc.

### CLI flags

| Flag | Long | Description | Default |
|------|------|-------------|---------|
| `-k` | `--ksize` | K value (exponent). Required. | -- |
| `-f` | `--file` | Output directory for plot file. Required. | -- |
| `-d` | `--device` | GPU device index | 0 |
| `-b` | `--benchmark` | Machine-readable timing output | false |
| `-v` | `--verify` | Verify plot after generation | false |
| `-g` | `--tmpdir` | Accepted for CLI compat, unused | -- |
| `-j` | `--tmpdir2` | Accepted for CLI compat, unused | -- |

### Output

Generates a file named `k{K}-{64_hex_chars_plot_id}.plot` in the output directory. Example:
```
k27-d273579a89d7ed3587c070e20fcfabe1a3ca88fea35d9cfca3c54a1bb7278503.plot
```

### Search GPU-generated plots

Use CPU VaultX to search. GPU-generated plots are byte-compatible:
```bash
# Single search
~/vaultx -s a1b2c3 -f /data/plots/k27-<id>.plot

# Batch search (1000 random lookups)
~/vaultx -S 1000 -D 1 -f /data/plots/k27-<id>.plot

# Verify plot integrity
~/vaultx -V true -f /data/plots/k27-<id>.plot -k 27
```

## GPU memory requirements

The entire Table1 + Table2 must fit in GPU VRAM. If it doesn't fit, the program prints the requirement and exits.

Formula: `Peak = N * 3 * NONCE_SIZE + 2 * 2^24 * 4` where N = 2^K

| K | Memory Required | Fits in |
|---|----------------|---------|
| 25 | 512 MB | 2 GB GPU |
| 26 | 896 MB | 2 GB GPU |
| 27 | 1,664 MB | 4 GB GPU |
| 28 | 3,200 MB | 4-6 GB GPU |
| 29 | 6,272 MB | 8 GB GPU |
| 30 | 12,416 MB | 16 GB GPU |
| 31 | 24,704 MB | 32 GB GPU |
| 32 | 49,280 MB | 80 GB GPU (A100) |

At startup the program queries the GPU, prints available memory, and refuses to proceed if K doesn't fit.

## Compile-time configuration

Set via `-D` flags in the Makefile or on the command line:

| Define | Default | Description |
|--------|---------|-------------|
| `NONCE_SIZE` | 4 | Bytes per nonce. 4 for K<=32, 5 for K>=33 |
| `RECORD_SIZE` | 12 | Total record size. HASH_SIZE = RECORD_SIZE - NONCE_SIZE |

These must match the CPU VaultX build for plot compatibility.

## Project structure

```
src/
  common/          Host-side code (shared by all backends)
    main.cpp       Entry point, CLI, orchestration
    globals.h      Types, constants, matching factors
    crypto_cpu.cpp  generate_plot_id(), derive_key() (libsodium)
    memory.cpp     GPU memory estimation
    plot_io.cpp    Write plot file to disk (4MB chunks)
  blake3/
    blake3_common.h  GPU-portable Blake3 keyed hash kernel
  cuda/            NVIDIA CUDA backend
    gpu_context_cuda.cu   Device init, memory, D2H, constant memory
    table1_cuda.cu        Table1 generation kernel
    sort_table2_cuda.cu   Per-bucket sort + match + Table2 kernel
  sycl/            Intel/AMD/NVIDIA SYCL backend
    gpu_context_sycl.cpp  Device init, USM memory, queue
    table1_sycl.cpp       Table1 generation kernel
    sort_table2_sycl.cpp  Per-bucket sort + match + Table2 kernel
  gpu_backend.h    Compile-time backend selection (#ifdef GPU_CUDA / GPU_SYCL)
```
