# VaultXGPU

GPU-accelerated plot generator for the VaultX PoS protocol. 

VaultXGPU runs the compute-intensive parts of plot generation on the GPU. The whole
working set stays in VRAM: no intermediate table ever crosses PCIe, and the only
host-bound transfer is the finished vault.

1. **Table1 generation**: hash all 2^K nonces with the Blake3 keyed hash and scatter
   them into 2^24 buckets by their 3-byte hash prefix (one thread per nonce)
2. **Sort + Table2 generation**: one block per bucket -- load the nonces, recompute
   their hashes, sort the bucket with a block-parallel bitonic network, scan the
   sorted keys for pairs within the matching distance, and emit each pair into its
   Table2 bucket
3. **Output**: copy Table2 to the host in chunks and write it to disk, with the
   transfer of one chunk overlapping the write of the previous one

The output is a standard VaultX plot file (`k{K}-{hex_plot_id}.plot`) that CPU VaultX
can search directly.

Supported K range is **27 to 32**. K outside that range is rejected at startup: the
matching-factor table has no entry for it, so a run would silently use a placeholder
value and produce a vault with the wrong match density.

## Requirements

### All builds
- Linux (Ubuntu 22.04+ recommended)
- libsodium: `sudo apt install libsodium-dev`
- C++17 compiler

### CUDA build (NVIDIA GPUs)
- CUDA Toolkit >= 11.0 (12.0+ recommended)
- NVIDIA GPU with compute capability >= 7.0 (Volta or newer)
- Install: https://developer.nvidia.com/cuda-toolkit

Terminal Commands: 
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

### SYCL build (Intel/AMD/NVIDIA GPUs)
- Intel oneAPI Base Toolkit (2023.0+) which includes `icpx` and the SYCL runtime
- Install: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html
- For AMD GPUs: also install ROCm (https://rocm.docs.amd.com/) and the oneAPI AMD plugin
- For NVIDIA GPUs: also install the oneAPI NVIDIA plugin\

Install: 
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-basekit

#You have to run this everytime you open a new terminal:
source /opt/intel/oneapi/setvars.sh

FOR WSL SYCL

wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install -y intel-oneapi-dpcpp-cpp
source /opt/intel/oneapi/setvars.sh


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

# Plot validator (host only -- no GPU toolchain needed)
make validate

# Custom nonce/record sizes
make cuda NONCE_SIZE=5 RECORD_SIZE=13

# Clean
make clean
```

This produces:
- `vaultx_cuda` -- NVIDIA GPU binary
- `vaultx_sycl` -- SYCL GPU binary (Intel/AMD/NVIDIA)
- `vaultx_validate` -- plot validator, no GPU required

### Sort variants

The Table2 bucket sort is selectable at compile time so the block-parallel sort can be
measured against the serial one it replaced.

| `VAULTX_SORT` | Sort | Binary suffix |
|---|---|---|
| 0 | Single-thread insertion sort, records move with the key (original baseline) | `_sort0` |
| 1 | Single-thread insertion sort over `(key, index)` pairs | `_sort1` |
| 2 | Block-parallel bitonic network over `(key, index)` pairs (**default**) | none |

```bash
make cuda VAULTX_SORT=0      # builds vaultx_cuda_sort0
make cuda-sort-variants      # builds all three at once
make sycl-sort-variants
```

Mode 1 exists to separate the two effects: mode 0 to 1 measures what it costs to move
whole records during the sort, and 1 to 2 measures what it costs to run on one thread.

## Usage

### Generate a plot

```bash
# Required: -k (K value) and -f (output directory)
./vaultx_cuda -k 27 -f /data/plots

# Specify which GPU (default: 0 = first GPU)
./vaultx_cuda -k 27 -f /data/plots -d 1

# SYCL backend
./vaultx_sycl -k 27 -f /data/plots --require-gpu
```

### Multi-GPU

If you have more than one GPU, use `-d` to select by index:
- `-d 0` -- first GPU (default)
- `-d 1` -- second GPU
- etc.

### CLI flags

| Flag | Long | Description | Default |
|------|------|-------------|---------|
| `-k` | `--ksize` | K value (exponent), 27-32. Required. | -- |
| `-f` | `--file` | Output directory for plot file. Required. | -- |
| `-d` | `--device` | GPU device index | 0 |
| `-b` | `--benchmark` | Machine-readable timing summary line | false |
| `-v` | `--verify` | Print the command to validate the plot afterwards | false |
| `-g` | `--tmpdir` | Accepted for CLI compat, unused | -- |
| `-j` | `--tmpdir2` | Accepted for CLI compat, unused | -- |

**Measurement**

| Long | Description | Default |
|------|-------------|---------|
| `--stats` | Read the bucket counters back and report occupancy, overflow and storage efficiency. Costs two 64 MB transfers, excluded from the reported total. | false |
| `--csv PATH` | Append one CSV row per run (`-` for stdout). Writes the header when creating the file. | -- |
| `--csv-header` | Print the CSV schema and exit | -- |
| `--run N` | Repeat index recorded in the CSV row | 0 |
| `--require-gpu` | Fail instead of running on a non-GPU device (SYCL) | false |

**Output pipeline**

| Long | Description | Default |
|------|-------------|---------|
| `--chunk-mb N` | Staging chunk size in MB | 256 |
| `--o-direct` | Open the plot with `O_DIRECT` so write timing measures the device, not the page cache | false |
| `--no-overlap` | Serialize transfer and write, so `d2h_s + write_s` equals the output stage's wall time | overlap on |
| `--no-fsync` | Skip the final `fdatasync` | fsync on |

### Output

Generates a file named `k{K}-{64_hex_chars_plot_id}.plot` in the output directory. Example:
```
k27-d273579a89d7ed3587c070e20fcfabe1a3ca88fea35d9cfca3c54a1bb7278503.plot
```

The plot ID is random per run, so plots are not reproducible by design.

## Measuring correctly

Three things will give you wrong numbers if you ignore them.

**`write()` does not measure the disk.** It returns when the data reaches the page
cache. On a host with more RAM than the plot, a plain timer around `write()` measures
memory bandwidth and the real cost surfaces later as a stall. Use `--o-direct` to
bypass the cache, and drop caches between runs:

```bash
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

**Kernel time and stage time are different numbers.** Each stage reports both: the
host-observed wall time (launch + synchronize) and the kernel-only time from CUDA
events or SYCL event profiling. Quote the kernel time when comparing kernels and the
wall time when comparing pipelines.

**Transfer and write overlap by default.** With overlap on, `d2h_s + write_s` exceeds
`write_wall_s`, and that gap is the point. For a clean PCIe-versus-disk decomposition,
run once with `--no-overlap`, where the two sum to the wall time.

Every run also reports the kernel's own I/O counters as an independent cross-check,
from `/proc/diskstats` for local block devices and `/proc/self/mountstats` for NFS
(which has no diskstats entry on the client at all). The reported amplification is
device bytes written divided by plot bytes; it should be very close to 1.0, since the
plot is written exactly once.

> NFS is not a local disk. Label it as a network filesystem in any figure rather than
> putting it in a row beside NVMe.

## Validating a plot

`vaultx_validate` checks a finished plot against the format using nothing but the plot
and its ID -- no GPU, no second plot, no reference implementation. For every stored
pair it recomputes the hashes and asserts that:

1. both nonces are below 2^K
2. both nonces hash into the same Table1 bucket
3. their 64-bit keys are ordered and within the matching distance for this K
4. the pair hashes into the Table2 bucket it is physically stored in

```bash
# full check (every bucket)
./vaultx_validate /data/plots/k27-<id>.plot

# sample 5000 random buckets -- much faster, same confidence per record
./vaultx_validate /data/plots/k27-<id>.plot --sample 5000 --seed 7

# drill into specific buckets
./vaultx_validate /data/plots/k27-<id>.plot --bucket 0 --bucket 12345
```

Exit status is 0 only if every checked record passes and the file is the expected size.
It also reports storage efficiency, since empty slots are counted along the way.

Record **order within a bucket is not checked, and must not be**: slots are claimed
with an atomic, so two runs on the same device produce the same records in a different
order. For the same reason two plots are never byte-identical, even with the same
inputs -- and where a bucket overflows, which records are dropped also varies. Compare
plots as per-bucket multisets, never with `cmp`.

You can also search a GPU-generated plot with CPU VaultX; the format is byte-compatible:
```bash
~/vaultx -S 1000 -D 1 -f /data/plots/k27-<id>.plot
```

## Statistics and storage efficiency

`--stats` reads both counter arrays back and reports true bucket occupancy. The
counters are deliberately left unclamped by the kernels, so a bucket that overflowed
reports how many records *wanted* in, not how many fit. This does not change what is
stored -- only the first `RPB` claimants are ever written -- but it makes overflow
loss observable.

```
=== Stats (K=27, RPB=8, N=134217728) ===
Table1  hashed=... stored=... dropped=... (13.96%)
Table1  buckets overflowed=... max occupancy=...
Table2  matched=... stored=... dropped=...
Storage efficiency = ...% of ... slots filled
```

Two things to expect. Table1 loses records to bucket overflow at a rate close to
`1/sqrt(2*pi*RPB)` -- about 14% at K=27 falling to 2.5% at K=32 -- because mean
occupancy equals RPB exactly, so roughly half of all buckets overflow by construction.
Table2, by contrast, is oversubscribed at every supported K (from about 3.5x at K=27
down to 1.2x at K=32), so storage efficiency should sit close to 100%.

## Reproducible benchmarking

```bash
# schema first, so the CSV is self-describing
./vaultx_cuda --csv-header > runs.csv

for k in 27 28 29 30 31; do
  for run in 1 2 3; do
    sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
    ./vaultx_cuda -k $k -f /mnt/nvme/plots --o-direct --csv runs.csv --run $run
    rm -f /mnt/nvme/plots/k$k-*.plot
  done
done
```

Discard the first run at each K as warm-up and report the median. The CSV carries
hostname, backend, device name, driver version, sort mode, chunk size, the O_DIRECT and
overlap settings and the plot ID, so rows from different machines and configurations can
be pooled without losing track of what produced them. `--csv-header` prints the full
field list.

## GPU memory requirements

The entire Table1 + Table2 must fit in GPU VRAM. If it doesn't fit, the program prints
the requirement and exits.

Formula: `Peak = N * 3 * NONCE_SIZE + 2 * 2^24 * 4` where N = 2^K

| K | Memory Required | Fits in |
|---|----------------|---------|
| 27 | 1,664 MB | 4 GB GPU |
| 28 | 3,200 MB | 4-6 GB GPU |
| 29 | 6,272 MB | 8 GB GPU |
| 30 | 12,416 MB | 16 GB GPU |
| 31 | 24,704 MB | 32 GB GPU |
| 32 | 49,280 MB | 80 GB GPU (A100) |

At startup the program queries the GPU, prints available memory, and refuses to proceed
if K doesn't fit. Table1 is freed after the sort+match stage, before the output stage,
so peak usage occurs during sort+match.

## Compile-time configuration

Set via `-D` flags in the Makefile or on the command line:

| Define | Default | Description |
|--------|---------|-------------|
| `NONCE_SIZE` | 4 | Bytes per nonce. 4 for K<=32, 5 for K>=33 |
| `RECORD_SIZE` | 12 | Total record size. HASH_SIZE = RECORD_SIZE - NONCE_SIZE |
| `VAULTX_SORT` | 2 | Table2 bucket sort: 0/1 serial, 2 block-parallel bitonic |
| `PREFIX_SIZE` | 3 | Bucket-index bytes; bucket count is 2^(8*PREFIX_SIZE). Only lower it for host-side tests. |
| `MIN_K` / `MAX_K` | 27 / 32 | Accepted K range |

`NONCE_SIZE` and `RECORD_SIZE` must match the CPU VaultX build for plot compatibility.

## Project structure

```
src/
  common/          Host-side code (shared by all backends)
    main.cpp       Entry point, CLI, stage timing, CSV emission
    globals.h      Types, constants, matching factors, K range, sort selection
    crypto_cpu.cpp generate_plot_id(), derive_key() (libsodium)
    memory.cpp     GPU memory estimation
    plot_io.cpp    Plot path construction
    plot_writer.cpp  Output stage: O_DIRECT, chunking, overlapped writer thread
    metrics.h/.cpp   Timers, /proc/diskstats + mountstats probes, CSV schema, stats
    sort_net.h     Bitonic comparator and network shared by both backends
  blake3/
    blake3_common.h  GPU-portable Blake3 keyed hash (also compiles on the host)
  cuda/            NVIDIA CUDA backend
    gpu_context_cuda.cu   Device init, VRAM, D2H + write stage, constant memory
    table1_cuda.cu        Table1 generation kernel
    sort_table2_cuda.cu   Per-bucket sort + match + Table2 kernel
  sycl/            Intel/AMD/NVIDIA SYCL backend
    gpu_context_sycl.cpp  Device init, USM memory, queue, D2H + write stage
    table1_sycl.cpp       Table1 generation kernel
    sort_table2_sycl.cpp  Per-bucket sort + match + Table2 kernel
  tools/
    validate_plot.cpp     vaultx_validate: structural plot validation, host only
  gpu_backend.h    Compile-time backend selection (#ifdef GPU_CUDA / GPU_SYCL)
```

The two backends are kept behaviourally identical on purpose: the same sort modes, the
same unclamped counters, the same output pipeline, and shared host code for statistics
and writing. Where they must diverge, the divergence is in the backend file and
commented there.
