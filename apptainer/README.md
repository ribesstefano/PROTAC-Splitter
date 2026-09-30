# PROTAC-Splitter Apptainer container

An [Apptainer](https://apptainer.org/) (formerly Singularity) definition file that packages
PROTAC-Splitter's full feature set into a single `.sif` image:

- **All three splitting strategies**: heuristic, XGBoost, and the Transformer
  (`model="adaptive"`, `"xgboost"`, `"heuristic"`, `"transformer"`, or any `"->"`/`"+"`-chained
  combination — see the [main README](../README.md) for details).
- GPU support for the Transformer strategy (and optional GPU-accelerated XGBoost) via
  `apptainer run/exec --nv`.
- The `protac-splitter` CLI and the [Gradio web app](../scripts/protac_splitter_app.py).

The image installs everything under the `scripts` extra of [`pyproject.toml`](../pyproject.toml)
(the `transformer` extra — `torch`, `transformers`, `accelerate` — plus `gradio` and the plotting
stack), resolved exactly as pinned in [`uv.lock`](../uv.lock) via `uv sync --frozen`. The XGBoost
and Transformer model weights are **not** baked into the image — they're downloaded from the
HuggingFace Hub the first time `split_protac()` needs them (see
[Model cache](#model-cache-huggingface-downloads) below).

Nothing here is specific to any particular cluster: the same `.def` file, build command, and run
commands work on a laptop, a workstation, or any HPC login node with Apptainer/Singularity and
outbound internet access.

## Prerequisites

- [Apptainer](https://apptainer.org/docs/user/latest/quick_start.html#installation) or
  Singularity installed (`apptainer --version`).
- Outbound internet access **at build time** (to pull the `ubuntu:22.04` base image, install `uv`,
  and resolve/download Python packages). Compute nodes on many clusters have none — build on a
  login node or your own machine, not inside a batch job.
- For GPU use: an NVIDIA driver on the host and a real, usable GPU allocation (see
  [GPU notes](#gpu-notes) below) — the image itself needs no CUDA toolkit installed on the host,
  only the driver.

## Building the image

Run this from the **repository root** (the `%files` paths in the `.def` resolve relative to your
working directory, not the `.def` file's location):

```bash
apptainer build --fakeroot apptainer/protac-splitter.sif apptainer/protac-splitter.def
```

This takes several minutes (downloading `torch` and friends). The resulting `.sif` embeds
absolute paths baked in at build time — build once per machine/cluster, don't copy a `.sif` built
on one host to a different one.

If `--fakeroot` isn't available for your account (no `/etc/subuid`/`/etc/subgid` entry — check
with `grep "^$(whoami):" /etc/subuid`), you have two alternatives:

- **Build elsewhere and copy the `.sif` over**: build on any machine where you have root (a
  laptop, a VM, a Docker-enabled CI runner) — `apptainer build` alone (no `--fakeroot`) works
  there — then `scp`/copy `apptainer/protac-splitter.sif` to the target host.
- **Remote build service**: `apptainer remote login` (needs a
  [Sylabs Cloud](https://cloud.sylabs.io/) account and access token), then
  `apptainer build --remote apptainer/protac-splitter.sif apptainer/protac-splitter.def`. Note
  this uploads your `.def` file (and the packages it resolves) to a third-party build service —
  only do this if that's acceptable for your project.

## Running

### CLI — a single SMILES

```bash
apptainer run apptainer/protac-splitter.sif \
    --smiles "O=C(NCCCCCCCCCCCC(=O)N[C@@H]1C(=O)N2CC(...)...)" --model adaptive
```

`apptainer run` invokes the image's default `%runscript`, which is `protac-splitter "$@"` — so any
flag documented in `protac-splitter --help` works here (`--model`, `--input-csv`/`--output-csv`,
`--device`, ...).

### CLI — a CSV of SMILES

```bash
apptainer run apptainer/protac-splitter.sif \
    --input-csv my_protacs.csv --smiles-col SMILES \
    --output-csv my_protacs_split.csv --model adaptive
```

### Live code, no rebuild

The image's baked-in code snapshot lives at `/opt/repo`, but the installed Python environment
lives outside it at `/opt/venv`. `bind_live_repo.sh` mounts your live checkout over `/opt/repo`
(excluding `.git`/`.venv`/`__pycache__`) so edits to `protac_splitter/` or `scripts/` show up
immediately — no rebuild needed unless `pyproject.toml`/`uv.lock` changed:

```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) \
    apptainer/protac-splitter.sif protac-splitter --smiles "CC(C)..." --model heuristic
```

### Gradio app

```bash
apptainer run --app gradio apptainer/protac-splitter.sif
```

By default Gradio binds `127.0.0.1`; to reach it from outside the container (e.g. to forward the
port over SSH from a remote host), set `GRADIO_SERVER_NAME`:

```bash
apptainer exec --env GRADIO_SERVER_NAME=0.0.0.0 --app gradio apptainer/protac-splitter.sif
```

then browse to `http://localhost:7860` (add `--env GRADIO_SERVER_PORT=<port>` to change the
port, and forward it, e.g. `ssh -L 7860:localhost:7860 <host>`, if running remotely).

### GPU notes

GPU work needs **both** `--nv` *and* an actual usable GPU allocation — `--nv` only bind-mounts the
host's NVIDIA driver into the container; it does nothing if the host's GPU isn't actually
available to your process (e.g. many clusters leave login-node GPUs in `compute_mode=Prohibited`,
visible in `nvidia-smi` but unusable). Request a real GPU allocation from your scheduler first.

```bash
# Direct (already on a host with a usable GPU allocation)
apptainer run --nv apptainer/protac-splitter.sif \
    --smiles "CC(C)..." --model transformer --device cuda

# Via a scheduler that grants GPU allocations, e.g. Slurm
srun --gpus=1 apptainer run --nv apptainer/protac-splitter.sif \
    --smiles "CC(C)..." --model transformer --device cuda
```

If you hit an XGBoost error on a host with a GPU that's present but unusable (e.g.
`compute_mode=Prohibited`), even for plain CPU inference (`--model xgboost` or the default
`adaptive`): CUDA-enabled XGBoost wheels probe for a GPU regardless of the requested device. Get a
real GPU allocation and pass `--nv`, the same as for the Transformer strategy — it isn't a
CPU-vs-GPU choice you can dodge on affected hosts.

### Model cache (HuggingFace downloads)

The XGBoost edge classifier (via `GraphEdgeClassifier.from_pretrained()`) and the Transformer
model (via `transformers`) are both fetched from the HuggingFace Hub on first use, and both share
the standard `huggingface_hub` cache under `HF_HOME` (default `~/.cache/huggingface`, see
[`.env.example`](../.env.example)). Apptainer bind-mounts your home directory by default, so the
cache persists across runs without any extra flags. To point it elsewhere (e.g. a shared cache on
a cluster), pass `--env`:

```bash
apptainer run --env HF_HOME=/path/to/shared/cache \
    apptainer/protac-splitter.sif --smiles "CC(C)..." --model adaptive
```

The first run needs internet access to download the models; a compute node with none will fail
unless the cache directory was already warmed on a node that had internet.

## Rebuilding after a dependency change

Only needed when `pyproject.toml` or `uv.lock` changes (code edits are picked up live via the bind
mount above, no rebuild required):

```bash
apptainer build --fakeroot apptainer/protac-splitter.sif apptainer/protac-splitter.def
```
