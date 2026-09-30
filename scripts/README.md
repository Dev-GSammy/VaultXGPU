# Experiment scripts

Four runnable scripts plus a join helper. Every script takes `-h` for its full flag
list, `-dry-run` to see what it would do, and `-o` to choose where results go.

| Script | Produces | What it does |
|---|---|---|
| `run_bench.sh` | E1.3, E2.1, E2.3, E2.4, E3.2, E4.1–E4.4, E5.1–E5.3, E7.1, E7.3 | The main sweep: K × backend × sort × drive × chunk × O_DIRECT × overlap × power cap |
| `run_validate.sh` | E1.1 | Generates (or finds) plots, validates them structurally, optionally searches them with the CPU prover |
| `run_cpu_baseline.sh` | E2.2 | CPU VaultX on the same host, swept over thread count |
| `run_multigpu.sh` | E6.1, E6.3 | N concurrent plotters across GPUs; `-vram` for the VRAM ceiling |
| `join_runs.sh` | — | Merges a `bench_*.csv` with its `.runs.csv` sidecar |

`lib.sh` is shared code, not runnable on its own.

## Where results go

```
experiments/<hostname>/
    machine_info.txt          CPU, GPUs, driver, memory, mounts -- written by every script
    bench_<tag>.csv           the plotter's own CSV schema, one row per run
    bench_<tag>.runs.csv      sidecar: tag, drive, repeat, power (join on `run`)
    validate_<tag>.csv        one row per plot checked
    validate_<tag>.logs/      raw validator and prover output
    cpu_<tag>.csv             CPU baseline
    multigpu_<tag>.csv        scale-out
    vram_<tag>.csv            VRAM ceiling
```

Override the whole tree with `VAULTX_EXPERIMENTS_DIR=/path`, or a single file with
`-o /path/to/file.csv`. Passing a directory to `-o` keeps the default filename.

Because the directory is named after the host, results from several machines can be
committed side by side without colliding. `machine_info.txt` means a CSV is never
orphaned from the hardware that produced it.

## Backends

`-backend` accepts `cuda`, `sycl-gpu`, `sycl-cpu`, or a comma-separated list.

- **cuda** — `vaultx_cuda`.
- **sycl-gpu** — `vaultx_sycl` with `--require-gpu`, so a run that silently lands on a
  CPU device fails instead of being recorded as a GPU result.
- **sycl-cpu** — `vaultx_sycl` with `ONEAPI_DEVICE_SELECTOR=*:cpu`, which forces the
  CPU device even on a host that has a GPU. Label these CPU-via-SYCL in every figure.
  Override the selector with `VAULTX_SYCL_CPU_SELECTOR`.

Missing binaries are skipped with a warning, so the same command line works on a
machine that only has one toolchain installed.

## Drives

`-drives` takes a comma-separated list of output directories, one per storage medium.
Each is swept in turn and recorded by label, so one command covers E4.2:

```bash
./run_bench.sh -k 31 -drives /mnt/nvme,/mnt/ssd,/mnt/nfs,/dev/shm -tag media
```

In `run_multigpu.sh` the list is cycled across GPUs instead. With one drive and eight
GPUs the disk will cap the node — which is itself the result, but point `-drives` at
several disks to tell storage contention apart from PCIe contention.

## Recipes

```bash
# correctness first -- it licenses every number after it
./run_validate.sh -k 27-29 -backend cuda -drives /mnt/nvme -prover ~/vaultx/vaultx

# stage breakdown and scaling                                     (E2.1, E2.4)
./run_bench.sh -k 27-31 -drives /mnt/nvme -stats -runs 3 -tag stages

# the sort study -- needs 'make cuda-sort-variants' first         (E3.2)
./run_bench.sh -k 29-31 -sort 0,1,2 -drives /mnt/nvme -tag sortstudy

# PCIe vs disk, serialized so d2h + write equals wall             (E4.1)
./run_bench.sh -k 27-31 -overlap off -odirect on -tag decompose

# storage media, then overlap, then chunk size                    (E4.2-E4.4)
./run_bench.sh -k 31 -drives /mnt/nvme,/mnt/ssd,/mnt/nfs -tag media
./run_bench.sh -k 31 -overlap on,off -drives /mnt/nvme -tag overlap
./run_bench.sh -k 31 -chunk 16,64,256,512,1024 -odirect both -tag chunk

# CUDA vs SYCL on the same GPU, and the SYCL CPU device           (E5.1, E5.3)
./run_bench.sh -k 27-31 -backend cuda,sycl-gpu -tag cudavssycl
./run_bench.sh -k 27-29 -backend sycl-cpu -tag syclcpu

# CPU baseline on the same host                                   (E2.2)
./run_cpu_baseline.sh -bin ~/vaultx/vaultx -k 27-31 -drives /mnt/nvme

# scale-out and the VRAM ceiling                                  (E6.1, E6.3)
./run_multigpu.sh -k 31 -n 1,2,4,8 -drives /mnt/nvme0,/mnt/nvme1
./run_multigpu.sh -vram -k 27-32 -drives /mnt/nvme

# energy, and the energy-optimal power cap                        (E7.1, E7.3)
./run_bench.sh -k 31 -power -runs 3 -tag energy
./run_bench.sh -k 31 -power -powercap 150,200,250,none -tag powercap
```

## Analysis

```bash
./join_runs.sh ../experiments/$(hostname -s)/bench_stages.csv joined.csv
```

The joined CSV has one row per run with every timing, counter, tag, drive and energy
value in it. Take the **median** across repeats and discard the first run at each
configuration as warm-up.

## Things that will bite you

- **Page cache.** Without `-odirect on`, `write_s` can measure memory bandwidth rather
  than the disk. The scripts drop caches between runs when passwordless sudo is
  available and warn loudly when it is not.
- **Power sampling** needs `nvidia-smi`; on non-NVIDIA hosts `-power` falls back to
  RAPL package counters, which measure the CPU, not the GPU.
- **Power caps** need passwordless sudo. The script restores the device's default
  limit after a capped sweep, but check with `nvidia-smi -q -d POWER` if a run is
  interrupted.
- **Disk space.** A K=31 plot is 17 GB. Plots are deleted after each run unless you
  pass `-keep`.
- **NFS** has no `/proc/diskstats` entry on the client; the plotter reads
  `/proc/self/mountstats` instead. Keep NFS in its own row, never beside NVMe.
