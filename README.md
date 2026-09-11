# llm_perf

Benchmarking harness for running LLM inference performance tests against
[vLLM](https://github.com/vllm-project/vllm), originally built for the NVIDIA DGX Station.

Each model is brought up in its own container, verified, benchmarked with
[AIPerf](https://pypi.org/project/aiperf/) across a sweep of concurrency levels, and torn
down before the next one starts — so results are not contaminated by a previous model still
holding GPU memory.

---

## The pipeline

Three stages, each driven by the same JSON config so a run is reproducible:

| Stage | Script | What it does |
| --- | --- | --- |
| 1. Fetch | `bin/hfdl.sh` | Resolves model sizes and downloads weights to the local HF cache |
| 2. Verify | `bin/container_setup.sh` | Starts each model in vLLM, polls `/v1/models` until it serves, tears it down |
| 3. Benchmark | `bin/aiperf.sh` | Runs AIPerf against each model at every configured concurrency |

Run them in that order. Stage 2 is worth running on its own after any config change — it
catches a bad `vllm_args` in ~30s instead of failing an hour into a benchmark sweep.

---

## Requirements

- Linux with NVIDIA GPUs and a working `nvidia-container-toolkit`
- Docker with Compose v2
- Node.js (pinned — install with `bash bin/setup/node.sh`)
- Python 3 with `venv`
- Enough disk for model weights. **Check free space before a large run** — the HF cache
  reaches hundreds of GB quickly.

---

## Setup

Install Node via nvm (once per machine):

```bash
bash bin/setup/node.sh
```

Then initialize the environment. This creates and activates the Python venv, installs
AIPerf, and exports the Chromium paths AIPerf needs for plot rendering:

```bash
source bin/init.sh
```

`init.sh` **must be sourced**, not executed — it activates the venv in your current shell.
It already runs `source venv/bin/activate` for you.

Long runs take hours. Start a `tmux` session first so a dropped connection doesn't kill the
sweep:

```bash
tmux new -s perf
```

---

## Quick start

Using the two-model smoke-test config:

```bash
bin/hfdl.sh --models-conf conf/test.json --dl
```

```bash
bin/container_setup.sh --models-conf conf/test.json
```

```bash
bin/aiperf.sh --models-conf conf/test.json
```

Note `hfdl` only reports sizes unless you pass `--dl`. Without it, nothing downloads —
useful for checking what a config would cost in disk space first.

The three wrapper scripts `cd` to the repo root themselves, so you can invoke them from
anywhere and relative config paths still resolve.

### Testing a single model

`container_setup.sh` takes `--model` to run just one entry from a config, which is the
fastest way to check a new model's flags:

```bash
bin/container_setup.sh --models-conf conf/test.json --model "Qwen/Qwen3-0.6B"
```

---

## Configuration

Configs live in `conf/` and are JSON arrays, one object per model:

```json
[
  {
    "model": "Qwen/Qwen3-0.6B",
    "vllm_args": "--reasoning-parser qwen3",
    "aiperf_args": "--extra-inputs '{\"chat_template_kwargs\":{\"enable_thinking\":false}}'",
    "aiperf_concurrency": [1, 4, 16]
  }
]
```

| Field | Purpose |
| --- | --- |
| `model` | HuggingFace repo ID. Also derives the container name and cache path. |
| `vllm_args` | Extra flags appended to the vLLM server command |
| `aiperf_args` | Extra flags passed to AIPerf |
| `aiperf_concurrency` | Concurrency levels to sweep; each is a separate benchmark run |

Shipped configs:

- `conf/test.json` — two small Qwen models, for smoke tests
- `conf/core_models.json` — the main benchmark set
- `conf/big_models.json` — large models needing significant VRAM

### Per-model container overrides

`conf/model_compose_overrides/<container-name>.yaml` holds the generated Docker Compose
override for each model. `container_setup.sh` writes one automatically the first time it
sees a model, then **reuses it thereafter** — so later edits to `vllm_args` in the JSON will
not take effect until you delete the corresponding override file. Hand-tuned flags in these
files are preserved for the same reason.

Baseline flags come from `conf/base_compose.yaml`, which also pins the vLLM image version.
That pin is deliberate: a floating tag mid-campaign makes early and late results
incomparable.

---

## Before trusting a full run

Verify these by hand. They are the ones that silently produce meaningless numbers:

- **ISL / OSL** — input and output sequence lengths are what you intended
- **Request count** — enough requests per concurrency level to be statistically meaningful
- **Containers actually stopping** — a leftover container holding GPU memory will skew
  every subsequent model in the sweep

Confirm nothing is still running between stages:

```bash
docker ps --filter "name=vllm-"
```

---

## Output

Benchmark results land in `artifacts/<timestamp>_<model>_c<concurrency>/`. Each
`container_setup` run also writes a `<timestamp>_container_setup.log` to the repo root.

Both are gitignored — they are regenerated on every run.

---

## Troubleshooting

**AIPerf fails on import or at plot rendering.** Its wheel pulls headless-browser
dependencies that may not be present:

```bash
sudo apt install -y libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 libgbm1
```

**A model fails readiness.** `container_setup` interleaves the container's own logs into its
output and log file on failure — read those first. The usual causes are an unsupported
`vllm_args` flag, or the model not fitting at the configured `--gpu-memory-utilization` and
`--max-model-len`.

**Nothing downloads.** `hfdl.sh` requires `--dl` to actually fetch. Also note the vLLM
container runs with `HF_HUB_OFFLINE=1`, so weights must already be cached before stage 2.

**Recording the environment.** Worth capturing alongside a result set, since installed
package versions affect numbers:

```bash
apt list --installed > app_versions.txt
```
