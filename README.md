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

**Generation** -- `generate_logo` · `generate_ipadapter` · `get_job_status`

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

| Node | Default filename |
|---|---|
| `CLIPVisionLoader` | `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` |
| `IPAdapterModelLoader` | `ip-adapter-plus_sdxl_vit-h.bin` |

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
`LogoGenInfo` record. Values that were not supplied are written as `?` and are
**never** defaulted -- an invented parameter in an embedded block is a permanent,
machine-readable lie.

## Development

```bash
pip install -e ".[dev]"
python -m unittest discover -s tests -v   # 85 tests
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
