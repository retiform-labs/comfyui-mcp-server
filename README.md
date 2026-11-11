# comfyui-mcp-server

An MCP server that gives an agent direct control of a running
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) instance: submit workflows,
run randomised batch discovery, shape LoRA stacks, and read back the provenance
of every image produced.

Built for **iterative aesthetic discovery** -- generate wide, curate narrow, and
keep a verifiable record of the parameters that produced the survivors.

> **Not affiliated with** `joenorton/comfyui-mcp-server` or any other project of
> a similar name. This is an independent implementation (`@retiform-labs`), and
> its tool surface is specific to the workflow templates it ships.

## What it does

Thirteen tools, split into four groups:

**Lifecycle** -- `system_check` · `start_comfyui` · `stop_comfyui` · `get_comfyui_status`

**Generation** -- `generate_image` · `generate_ipadapter` · `get_job_status`

**Batch** -- `batch_generate` · `batch_generate_ipadapter`

**Hardware** -- `list_models` · `download_model` · `get_vram_usage` · `emergency_cleanup`

### Randomised batch discovery

The point of the batch tools is *divergence*, not repetition. Pass a `randomize`
spec and each job in the batch draws its own parameters:

```jsonc
{
  "randomize": {
    "seed":      "random",            // fresh seed per job, deduplicated
    "sampler":   "random",            // from a curated safe pool
    "cfg_scale": [5.0, 9.0],          // uniform in range
    "loras":     {"mode": "auto", "count": [1, 3], "weight": [0.25, 0.7]}
  }
}
```

Ranges are **2-element lists** `[min, max]`. A dict like `{"min": 1, "max": 5}` is
rejected with an error rather than silently sent to the workflow as a literal --
that mistake is easy to make and produces a batch where nothing varies.

### LoRA stack shape matters more than the sum

Two LoRAs at equal weight average two aesthetics into a muddle. A stack
generally wants **one dominant adapter and one supporting it**:

- dominant: `0.55` -- `0.77`
- supporting: `0.25` -- `0.40`
- optional style pin: `~0.45`

A total weight around `1.3` is normal and is **not** a ceiling -- well-shaped
stacks run above it and produce the best results.


## Tool reference

Every tool returns JSON. `generate_*` and `batch_*` are **asynchronous**: they
queue work and return a job id immediately, because a diffusion run takes far
longer than any sane request timeout. Poll `get_job_status` for the result.

### Lifecycle

#### `system_check()`

Reports GPU name, driver version, total and free VRAM, and whether ComfyUI is
responding. Reads `nvidia-smi`, so it sees the whole GPU -- including memory
held by ComfyUI, which a torch call inside this process never could.

```jsonc
// no arguments
{"gpu": "NVIDIA GeForce RTX 4090", "vram_total_mb": 24564,
 "vram_free_mb": 3703, "comfyui_running": true, "comfyui_version": "0.x"}
```

#### `start_comfyui(lowvram=True, fp16=True, port=8188)`

Launches ComfyUI if it is not already running. `lowvram` is the safe default for
cards under ~8 GB.

```jsonc
{"success": true, "pid": 41233, "port": 8188, "already_running": false}
```

#### `stop_comfyui(force=False)`

Stops the ComfyUI process. `force=True` sends SIGKILL when a normal shutdown
hangs.

#### `get_comfyui_status()`

Lightweight liveness check -- cheaper than `system_check` because it never shells
out to the driver.

```jsonc
{"running": true, "port": 8188, "queue_remaining": 0}
```

### Generation

#### `generate_image(...)`

Text-to-image through the packaged SDXL workflow.

| Parameter | Default | Notes |
|---|---|---|
| `prompt` | *required* | What to generate |
| `negative_prompt` | `"text, watermark, blurry, low quality"` | What to avoid |
| `negative_embedding` | `None` | A textual-inversion embedding name |
| `width` / `height` | `1024` / `1024` | |
| `steps` | `20` | |
| `cfg_scale` | `7.0` | |
| `sampler` | `"euler_ancestral"` | |
| `scheduler` | `"normal"` | |
| `lora` | `None` | One LoRA filename from `models/loras/` |
| `lora_weight` | `0.8` | |
| `loras` | `None` | Several at once: `[{"name": ..., "weight": ...}]` |
| `seed` | `None` | Omit for a random seed |
| `output_format` | `"png"` | PNG carries provenance metadata |

```jsonc
{"prompt": "a geometric fox mark, single flat ink colour, high contrast",
 "width": 1024, "height": 1024, "steps": 30, "seed": 1777732588}
// -> {"job_id": "a3f9...", "status": "queued", "params": {...}}
```

#### `generate_ipadapter(...)`

Reference-guided generation. Feeds 1--4 images into the model to steer style,
composition or content.

| Parameter | Default | Notes |
|---|---|---|
| `prompt` | *required* | |
| `reference_images` | *required* | 1--4 filenames from `comfyui/input/` |
| `reference_weight` | `0.6` | How hard the references pull |
| `weight_type` | `"style transfer"` | Also `"linear"`, `"ease in"`, `"strong style transfer"` |
| `start_at` / `end_at` | `0.0` / `1.0` | Fraction of sampling the adapter applies over |
| `embeds_scaling` | `"V only"` | |
| `width` / `height` | `512` / `512` | |
| `steps` | `10` | |
| `cfg_scale` | `7.5` | |
| `sampler` / `scheduler` | `"dpmpp_2m"` / `"karras"` | |
| `clip_vision_model` | env default | Override the CLIP vision tower |
| `ipadapter_model` | env default | Override the IP-Adapter weight |

```jsonc
{"prompt": "the reference mark, redrawn as a woodcut",
 "reference_images": ["ref-a.png", "ref-b.png"],
 "reference_weight": 0.65, "weight_type": "style transfer"}
```

> **Reference images must already exist in ComfyUI's `input/` directory.** This
> tool does not upload files; it names them. A missing file is a raised error,
> not a silent no-op.

#### `get_job_status(job_id)`

Polls a queued job. Returns `queued`, `running`, `completed` or `failed`, and on
completion the output image filenames with their provenance payload.

```jsonc
{"job_id": "a3f9...", "status": "completed",
 "images": [{"filename": "image_output_00042_.png", "subfolder": "", "type": "output"}],
 "params": {"seed": 1777732588, "sampler": "euler_ancestral", "steps": 30, ...}}
```

### Batch

#### `batch_generate(prompts, shared_params=None, randomize=None, deduplicate_seeds=True, priority="normal", keep_loaded=False)`

Queues one job per prompt, optionally randomising parameters across the batch.

- `prompts` -- a list of positive prompt strings, one per job.
- `shared_params` -- static parameters applied to every job (accepts an object
  or a JSON string).
- `randomize` -- per-job draws; overrides `shared_params` where it names the same key.
- `deduplicate_seeds` -- guarantees no two jobs share a seed.
- `keep_loaded` -- leave the model resident after the batch. See *VRAM behaviour*.

```jsonc
{"prompts": ["a fox mark", "an owl mark", "a wave mark"],
 "shared_params": {"width": 1024, "height": 1024, "steps": 24},
 "randomize": {"seed": "random", "cfg_scale": [5.0, 9.0],
               "loras": {"mode": "auto", "count": [1, 2], "weight": [0.25, 0.7]}}}
```

Randomisation spec forms:

| Form | Meaning |
|---|---|
| `"random"` | Draw from a curated safe pool |
| `"random_unsafe"` | Draw from the full pool, including values that often break |
| `[min, max]` | Uniform draw in a numeric range |
| `{"mode": "auto", "count": [1,3], "weight": [0.25,0.7]}` | LoRA stack selection |
| scalar (e.g. `24`) | Fixed at that value |

#### `batch_generate_ipadapter(prompts, reference_images, ...)`

The reference-guided equivalent, taking the same batch arguments plus the
IP-Adapter parameters from `generate_ipadapter`.

### Hardware

#### `list_models(model_type="all")`

Lists what is actually on disk, with sizes. `model_type` filters to one of
`unet`, `clip`, `vae`, `loras`, `ipadapter`, `clip_vision`.

```jsonc
{"model_type": "loras"}
// -> [{"name": "copperplate_v2.safetensors", "type": "loras",
//      "size_mb": 172.4, "format": "safetensors"}]
```

#### `download_model(url, model_type, filename=None, expected_sha256=None, token=None)`

Downloads from a Civitai-style URL. Writes atomically -- a partial download never
lands under the final name -- and sanitises `filename` to a single path
component, so a crafted URL cannot write outside the model directory.

```jsonc
{"url": "https://civitai.com/api/download/models/12345",
 "model_type": "loras", "filename": "copperplate_v2.safetensors"}
```

#### `get_vram_usage()`

Current GPU memory as the driver reports it. Reads `nvidia-smi` rather than this
process's CUDA context, which never allocates and would always read zero.

```jsonc
{"used_mb": 9216, "total_mb": 24564, "free_mb": 15348}
```

#### `emergency_cleanup(restart_comfyui=False)`

Frees GPU memory by asking ComfyUI to unload its models, then re-reads the driver
to report what was **actually** released. Useful before handing the GPU to a
non-ComfyUI job.

```jsonc
{"models_unloaded": ["sdxl-base"], "vram_freed_mb": 9001,
 "vram_before_mb": 9216, "vram_after_mb": 215}
```

> `restart_comfyui=True` is the heavier option: it stops and restarts the server.
> Use it when a normal unload does not recover the memory.

## Requirements

- **Python 3.11+**
- **A running ComfyUI instance** with the nodes your workflows use
  (`UnetLoaderGGUF` and the IP-Adapter nodes, for the shipped templates)
- **`nvidia-smi`** on `PATH` for the VRAM tools (NVIDIA drivers)
- **Models, LoRAs, embeddings and CLIP/VAE weights are NOT included.** Source
  them yourself and place them where ComfyUI expects them. This repository ships
  code, workflow templates and instructions -- never another party's licensed
  artifacts.

The IP-Adapter templates reference these filenames; substitute your own:

| Node | Default filename | Override with |
|---|---|---|
| `CLIPVisionLoader` | `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` | `clip_vision_model=` or `COMFYUI_CLIP_VISION_MODEL` |
| `IPAdapterModelLoader` | `ip-adapter-plus_sdxl_vit-h.bin` | `ipadapter_model=` or `COMFYUI_IPADAPTER_MODEL` |

These are **defaults, not requirements.** If your install uses a different CLIP
vision tower or IP-Adapter weight, pass the filename to `generate_ipadapter` or
set the environment variable -- no code change, and no failure to diagnose
silently.

A negative embedding is optional; the templates accept one via the
`negative_embedding` argument (e.g. `DeepNegative_xl_v1`).

## Install

```bash
pip install comfyui-mcp-server
```

Or from source:

```bash
git clone https://github.com/retiform-labs/comfyui-mcp-server
cd comfyui-mcp-server
pip install -e ".[dev]"
```

> **`mcp` is pinned below 2.0** -- in the MCP SDK 2.x line,
> `mcp.server.fastmcp` is a compatibility stub that raises on import. Do not
> widen that pin without migrating the server to `MCPServer`.

## Configure

All configuration is environment-driven. Nothing is assumed about where ComfyUI
lives.

| Variable | Default | Meaning |
|---|---|---|
| `COMFYUI_HOST` | `127.0.0.1` | ComfyUI host |
| `COMFYUI_PORT` | `8188` | ComfyUI port |
| `COMFYUI_PATH` | `~/ComfyUI` | ComfyUI install root (the dir with `main.py`) |
| `COMFYUI_MODELS_PATH` | `$COMFYUI_PATH/models` | Models root |
| `COMFYUI_OUTPUT_PATH` | `$COMFYUI_PATH/output` | Where renders land |
| `COMFYUI_CLIP_VISION_MODEL` | `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` | CLIP vision tower for reference-guided generation |
| `COMFYUI_IPADAPTER_MODEL` | `ip-adapter-plus_sdxl_vit-h.bin` | IP-Adapter weight (plus variant) |
| `COMFYUI_IPADAPTER_BASIC_MODEL` | `ip-adapter_sdxl_vit-h.bin` | IP-Adapter weight used by the VRAM-saving fallback |
| `COMFYUI_DOWNLOAD_TOKEN` | *(none)* | Bearer token for gated model hosts, so it need not travel through a tool argument |

### MCP client

```jsonc
{
  "mcpServers": {
    "comfyui": {
      "command": "comfyui-mcp-server",
      "env": { "COMFYUI_PATH": "/opt/ComfyUI" }
    }
  }
}
```

## VRAM behaviour

The MCP server is a **separate process** from ComfyUI. It never allocates CUDA
tensors, so `torch.cuda.memory_allocated()` inside it is structurally zero no
matter what ComfyUI is holding. This server therefore reads VRAM from the
driver (`nvidia-smi`) and releases it by asking ComfyUI to unload --

- `get_vram_usage` -- live used/free/total from the driver
- `emergency_cleanup` -- `POST /free {"unload_models": true, "free_memory": true}`
  on ComfyUI, reporting real before/after MB and `freed_mb`

`torch` is deliberately **not** a dependency. It cost ~2GB of CUDA to produce a
number that measured the wrong process.

Unloading after every job is usually the wrong trade: the next job then pays a
full model-load penalty. If you release between jobs, release at **batch**
boundaries.

## Provenance

Generated PNGs carry an A1111-compatible `parameters` block plus a structured
`GenerationInfo` record. Values that were not supplied are written as `?` and are
**never** defaulted -- an invented parameter in an embedded block is a permanent,
machine-readable lie.

Images rendered before the rename carry the same payload under the key
`LogoGenInfo`. Both keys are read; only the current one is written.

## Development

```bash
pip install -e ".[dev]"
python -m unittest discover -s tests -v
ruff check src tests
```

Nothing under `src/` may contain an absolute filesystem path; a test enforces
this, because a package that only works on its author's machine is broken for
everyone else.

## License

MIT -- see [LICENSE](LICENSE).

Copyright (c) 2026 David Newman and Guan.

---

## Sponsors

If comfyui-mcp-server is useful to you, consider supporting its continued development:

- **[GitHub Sponsors](https://github.com/sponsors/guan-tends)**
- **Bitcoin:** `bc1q0gd3mwjg3zy9sghv22kmpg823vss4c0zzdzg24`
- **Solana:** `Eu8wQcW68TKMs1a6eqzZu8znzU52QLqQugAMG8uCD6y6`
- **Ethereum / EVM:** `0x2733ff7c865C56d565a99BE1DC11B81cc76850A5`
- **XRP Ledger:** `r4X6e7McAQj7e8vBCeued1RYu4mCJrREDG`

---

Crafted with ❤️ by [Retiform Labs](https://retiform.dev)
