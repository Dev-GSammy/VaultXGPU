# Experiment results

One directory per machine, named after its short hostname. Written by the scripts in
`../scripts/`; see `../scripts/README.md` for what produces what and
`../Paper/EXPERIMENTS.md` for what each experiment is for.

Every directory carries a `machine_info.txt` recording the CPU, GPUs, driver version,
memory and mounts at the time of the run, so no CSV is ever orphaned from the hardware
that produced it.

Commit the CSVs. Do not commit plot files.
