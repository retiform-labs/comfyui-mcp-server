#!/usr/bin/env python3
"""
ComfyUI MCP Server -- agentic image generation over the Model Context Protocol.

Thirteen tools covering server lifecycle, single and batch generation, LoRA
stack shaping, model management, and GPU telemetry.

The MCP server is a separate process from ComfyUI and holds no CUDA tensors, so
VRAM is read from the driver (nvidia-smi) and released through ComfyUI's own
POST /free. torch is deliberately not a dependency.
"""

import ast
import copy
import json
import os
import random
import secrets
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

import psutil
import requests

# MCP SDK
from mcp.server.fastmcp import FastMCP

# Local modules. Relative only: this package is entered through its console
# script or `python -m`, never as a flat script, so the sys.path fallback that
# used to sit here was unreachable code that also shadowed the names above
# (mypy flagged it as no-redef ten times over). A failure here is NOT
# survivable -- a silently-None logger degrades every tool's diagnostics
# without a word -- so it raises instead of warning.
from .logger import log_debug, log_error, log_info, log_warning
from .png_metadata import META_INFO_KEY, embed_metadata

# Persistent job params cache (survives process restarts via file backing)
_JOB_PARAMS_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".job_params_cache.json")


def _load_job_params() -> dict[str, dict[str, Any]]:
    """Load cached job params from disk."""
    if os.path.isfile(_JOB_PARAMS_CACHE_PATH):
        try:
            with open(_JOB_PARAMS_CACHE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_job_params(cache: dict[str, dict[str, Any]]) -> None:
    """Persist job params to disk."""
    try:
        with open(_JOB_PARAMS_CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


_JOB_PARAMS = _load_job_params()


def _trace_workflow(workflow: dict, label: str = "") -> None:
    """Debug helper: print key node connections to stderr. Persistent."""
    prefix = "[MCP-TRACE]"
    if label:
        prefix += f" [{label}]"
    log_debug(f"{prefix} === WORKFLOW TRACE ===")
    log_debug(f"{prefix} Total nodes: {len(workflow)}")

    for nid, node in workflow.items():
        if node.get("class_type") == "KSampler":
            log_debug(
                f"{prefix} KSampler({nid}): seed={node['inputs'].get('seed')}, cfg={node['inputs'].get('cfg')}, steps={node['inputs'].get('steps')}, sampler={node['inputs'].get('sampler_name')}, scheduler={node['inputs'].get('scheduler')}, model={node['inputs'].get('model')}"
            )

    for nid, node in workflow.items():
        if node.get("class_type") == "IPAdapterBatch":
            inp = node["inputs"]
            log_debug(
                f"{prefix} IPAdapterBatch({nid}): model={inp.get('model')}, image={inp.get('image')}, weight={inp.get('weight')}, weight_type={inp.get('weight_type')}"
            )

    lora_nodes = [(nid, node) for nid, node in workflow.items() if node.get("class_type") == "LoraLoader"]
    for nid, node in sorted(lora_nodes, key=lambda x: int(x[0])):
        inp = node["inputs"]
        log_debug(
            f"{prefix} LoraLoader({nid}): name={inp.get('lora_name')}, model_in={inp.get('model')}, clip_in={inp.get('clip')}, strength={inp.get('strength_model')}"
        )

    for nid, node in workflow.items():
        if node.get("class_type") == "LoadImage":
            log_debug(f"{prefix} LoadImage({nid}): image={node['inputs'].get('image')}")

    for nid, node in workflow.items():
        if node.get("class_type") == "CLIPTextEncode":
            text = node["inputs"].get("text", "MISSING")
            text_preview = repr(text[:60]) if isinstance(text, str) else str(text)
            log_debug(f"{prefix} CLIPTextEncode({nid}): text={text_preview}")

    log_debug(f"{prefix} === END TRACE ===")


def _validate_model_chain(workflow: dict) -> tuple[bool, str]:
    """Traverse model graph from KSampler backward to Base. Return (ok, error).

    Expected chain:
        KSampler → IPAdapterBatch → [LoraLoader...] → UnetLoaderGGUF (base)

    Validates:
    - KSampler.model points to a valid node
    - No orphan nodes in the chain
    - No CLIPVisionLoader or IPAdapterModelLoader in model path
    - Every link resolves to a node that exists in workflow
    """
    # Find KSampler
    ksampler = None
    for nid, node in workflow.items():
        if node.get("class_type") == "KSampler":
            ksampler = node
            break
    if ksampler is None:
        return False, "No KSampler node found in workflow"

    model_ref = ksampler.get("inputs", {}).get("model")
    if not model_ref:
        return False, "KSampler has no model input"
    if not isinstance(model_ref, list) or len(model_ref) < 2:
        return False, f"KSampler.model has invalid format: {model_ref}"

    current_id = str(model_ref[0])
    visited = set()
    chain = []

    while current_id in workflow:
        if current_id in visited:
            return False, f"Cycle detected at node {current_id}"
        visited.add(current_id)
        node = workflow[current_id]
        ctype = node.get("class_type", "unknown")
        chain.append(f"{current_id}({ctype})")

        # Guard against utility nodes sneaking into model path
        if ctype in ("CLIPVisionLoader", "IPAdapterModelLoader"):
            return (
                False,
                f"Utility node {ctype}({current_id}) in model chain — expected IPAdapterBatch or LoraLoader. Chain: {' -> '.join(chain)}",
            )

        # Base model reached
        if ctype in ("UnetLoaderGGUF", "CheckpointLoaderSimple"):
            return True, f"Valid chain: {' -> '.join(chain)}"

        # LoraLoader: follow model output to previous node
        if ctype == "LoraLoader":
            model_in = node.get("inputs", {}).get("model")
            if not model_in or not isinstance(model_in, list) or len(model_in) < 2:
                return False, f"LoraLoader({current_id}) has invalid model input"
            current_id = str(model_in[0])
            continue

        # IPAdapterBatch: follow model input to previous node
        if ctype == "IPAdapterBatch":
            model_in = node.get("inputs", {}).get("model")
            if not model_in or not isinstance(model_in, list) or len(model_in) < 2:
                return False, f"IPAdapterBatch({current_id}) has invalid model input"
            current_id = str(model_in[0])
            continue

        # Unknown node type in model path
        return False, f"Unexpected node type {ctype}({current_id}) in model chain. Chain: {' -> '.join(chain)}"

    return False, f"Orphan reference: node {current_id} not found in workflow. Chain: {' -> '.join(chain)}"


# Configuration
# Every filesystem root is environment-driven. A published package must not
# assume it lives on the machine it was developed on: hardcoded paths make
# every model operation silently fail for anyone else.
COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
COMFYUI_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"

# Root of the ComfyUI installation (the directory containing main.py).
COMFYUI_PATH = os.environ.get("COMFYUI_PATH", os.path.expanduser("~/ComfyUI"))

# Models root. Defaults to ComfyUI's own models/ directory.
MODELS_PATH = os.environ.get("COMFYUI_MODELS_PATH", os.path.join(COMFYUI_PATH, "models"))
OUTPUT_PATH = os.environ.get("COMFYUI_OUTPUT_PATH", os.path.join(COMFYUI_PATH, "output"))

# Initialize MCP Server

# ============================================================================
# CURATED SAMPLER POOLS (for randomized batch generation)
# ============================================================================
_SAFE_SAMPLERS = [
    "euler",
    "euler_ancestral",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "dpmpp_2m_cfg_pp",
    "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu",
    "heun",
    "heunpp2",
    "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "uni_pc",
    "uni_pc_bh2",
    "ddim",
]

_UNSAFE_SAMPLERS = [
    "lcm",
    "ddpm",
    "gradient_estimation",
    "gradient_estimation_cfg_pp",
    "seeds_2",
    "seeds_3",
    "sa_solver",
    "sa_solver_pece",
    "exp_heun_2_x0",
    "exp_heun_2_x0_sde",
    "er_sde",
    "ipndm",
    "ipndm_v",
    "deis",
    "res_multistep",
    "res_multistep_cfg_pp",
    "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp",
]

_ALL_SAMPLERS = _SAFE_SAMPLERS + _UNSAFE_SAMPLERS

_ALL_SCHEDULERS = [
    "simple",
    "sgm_uniform",
    "karras",
    "exponential",
    "ddim_uniform",
    "beta",
    "normal",
    "linear_quadratic",
    "kl_optimal",
]
try:
    from . import __version__ as _PKG_VERSION
except ImportError:  # executed as a flat script
    _PKG_VERSION = "0.0.0"

# FastMCP's constructor takes no `version` argument in mcp 1.x -- passing one
# raises TypeError at import, which no mock can catch (it shipped that way once).
# But it never forwards a version to the low-level server either, so the
# handshake advertises the SDK's own version instead of ours. Set it on the
# inner server after construction: that is the object the initialize response
# actually reads (lowlevel/server.py: `server_version=self.version or ...`).
mcp = FastMCP("comfyui-mcp-server")
mcp._mcp_server.version = _PKG_VERSION

# ============================================================================
# TOOL 1: system_check
# ============================================================================
# IP-Adapter loader defaults. These are DEFAULTS, not facts about any install:
# a different CLIP vision tower or IP-Adapter weight is a legitimate setup, so
# every one of them is overridable per call in generate_ipadapter() and, failing
# that, by environment variable. A hardcoded model name here silently pins every
# user to one specific download.
IPADAPTER_CLIP_VISION_MODEL = os.environ.get("COMFYUI_CLIP_VISION_MODEL", "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors")
IPADAPTER_PLUS_MODEL = os.environ.get("COMFYUI_IPADAPTER_MODEL", "ip-adapter-plus_sdxl_vit-h.bin")
IPADAPTER_BASIC_MODEL = os.environ.get("COMFYUI_IPADAPTER_BASIC_MODEL", "ip-adapter_sdxl_vit-h.bin")


@mcp.tool()
def system_check() -> dict[str, Any]:
    """Verify system readiness: GPU, driver, VRAM.

    Reads the driver, not torch. This process never allocates CUDA tensors, so
    torch.cuda would report 0 whether or not ComfyUI is holding every byte.
    """
    total = _nvidia_smi("memory.total")
    used = _nvidia_smi("memory.used")
    free = _nvidia_smi("memory.free")
    try:
        name_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        gpu_name, driver = (
            (name_out.stdout.strip().split(",") + ["", ""])[:2] if name_out.returncode == 0 else (None, None)
        )
        gpu_name = (gpu_name or "").strip() or None
        driver = (driver or "").strip() or None
    except (OSError, subprocess.SubprocessError):
        gpu_name = driver = None

    readable = total is not None
    return {
        "cuda_available": readable,
        "driver_version": driver,
        "gpu_name": gpu_name,
        "vram_total_mb": total,
        "vram_used_mb": used,
        "vram_free_mb": free,
        "utilization_percent": round(used / total * 100, 2) if (used is not None and total) else None,
        "timestamp": datetime.now().isoformat(),
        **({} if readable else {"error": "nvidia-smi unavailable or returned no data"}),
    }


# ============================================================================
# TOOL 2: start_comfyui
# ============================================================================
@mcp.tool()
def start_comfyui(lowvram: bool = True, fp16: bool = True, port: int = 8188) -> dict[str, Any]:
    """Launch ComfyUI with VRAM-optimized flags"""
    try:
        # Check if already running
        status = get_comfyui_status()
        if status.get("running"):
            return {
                "success": True,
                "pid": status.get("pid"),
                "port": status.get("port"),
                "message": "ComfyUI already running",
            }

        # Build command
        cmd = [sys.executable, os.path.join(COMFYUI_PATH, "main.py"), "--port", str(port)]
        if lowvram:
            cmd.append("--lowvram")
        if fp16:
            cmd.append("--fp16-unet")
        cmd.append("--preview-method")
        cmd.append("auto")

        # Launch process
        process = subprocess.Popen(
            cmd, cwd=COMFYUI_PATH, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
        )

        # Wait briefly and verify
        time.sleep(2)
        new_status = get_comfyui_status()

        return {
            "success": new_status.get("running", False),
            "pid": process.pid,
            "port": port,
            "message": "ComfyUI started" if new_status.get("running") else "Failed to start",
        }
    except Exception as e:
        return {"success": False, "pid": None, "port": None, "message": f"Error: {str(e)}"}


# ============================================================================
# TOOL 3: stop_comfyui
# ============================================================================
@mcp.tool()
def stop_comfyui(force: bool = False) -> dict[str, Any]:
    """Graceful shutdown of ComfyUI server"""
    try:
        status = get_comfyui_status()
        if not status.get("running"):
            return {"success": True, "message": "ComfyUI not running"}

        pid = status.get("pid")
        if pid:
            process = psutil.Process(pid)
            if force:
                process.kill()
                return {"success": True, "message": f"ComfyUI killed (PID {pid})"}
            else:
                process.terminate()
                try:
                    process.wait(timeout=5)
                    return {"success": True, "message": f"ComfyUI stopped (PID {pid})"}
                except psutil.TimeoutExpired:
                    process.kill()
                    return {"success": True, "message": f"ComfyUI force killed (PID {pid})"}
        return {"success": False, "message": "PID not found"}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


# ============================================================================
# TOOL 4: get_comfyui_status
# ============================================================================
@mcp.tool()
def get_comfyui_status() -> dict[str, Any]:
    """Check if ComfyUI is running"""
    try:
        # Try HTTP check first
        response = requests.get(f"{COMFYUI_URL}/system_stats", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return {
                "running": True,
                "pid": data.get("system", {}).get("process_id"),
                "port": COMFYUI_PORT,
                "uptime_sec": data.get("system", {}).get("uptime"),
                "comfyui_version": data.get("system", {}).get("comfyui_version"),
                "pytorch_version": data.get("system", {}).get("pytorch_version"),
            }
    except (requests.RequestException, ValueError, KeyError, TypeError):
        pass

    # Fallback: check for python process running main.py
    try:
        for proc in psutil.process_iter(["pid", "cmdline"]):
            cmdline = proc.info.get("cmdline", [])
            if cmdline and "main.py" in str(cmdline):
                return {
                    "running": True,
                    "pid": proc.info["pid"],
                    "port": COMFYUI_PORT,
                    "uptime_sec": None,
                    "comfyui_version": None,
                    "pytorch_version": None,
                }
    except (requests.RequestException, ValueError, KeyError, TypeError):
        pass

    return {
        "running": False,
        "pid": None,
        "port": None,
        "uptime_sec": None,
        "comfyui_version": None,
        "pytorch_version": None,
    }


# ============================================================================
# TOOL 5: generate_image - text-to-image
# ============================================================================
def _generator_tag(variant: str = "") -> str:
    """Provenance string stamped into generated images.

    Derived from the package version so it cannot drift from what was actually
    installed; the variant marks which code path produced the render.
    """
    suffix = f" ({variant})" if variant else ""
    return f"comfyui-mcp-server/{_PKG_VERSION}{suffix}"


def _workflow_path(filename: str) -> str:
    """Resolve a workflow template path relative to this package.

    Never hardcode an absolute path: the package must work from any checkout,
    any install prefix, and any working directory.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "workflows", filename))


def _nvidia_smi(field: str) -> int | None:
    """Query one numeric GPU field via nvidia-smi. None when unavailable.

    The MCP server runs in its own process and never allocates CUDA tensors, so
    torch.cuda.memory_allocated() is structurally 0 here. ComfyUI is a separate
    PID; only the driver can report what it actually holds.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def _free_comfyui_vram() -> dict[str, Any]:
    """Ask ComfyUI to unload models and release cached VRAM.

    ComfyUI owns the models. POST /free is its own endpoint and the only lever
    that reaches the process actually holding the memory.
    """
    before = _nvidia_smi("memory.used")
    try:
        response = requests.post(
            f"{COMFYUI_URL}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=30,
        )
        ok = response.status_code == 200
        error = None if ok else f"HTTP {response.status_code}"
    except requests.RequestException as exc:
        ok, error = False, str(exc)
    # ComfyUI unloads asynchronously: the endpoint returns 200 before the
    # memory is actually released, so a synchronous re-read reports a false
    # zero. Poll until the figure settles (or stops moving) before measuring.
    after = before
    if ok:
        after = _poll_until_released(before, timeout=20.0)
    else:
        after = _nvidia_smi("memory.used")

    freed = (before - after) if (before is not None and after is not None) else None
    return {
        "success": ok,
        "error": error,
        "used_before_mb": before,
        "used_after_mb": after,
        "freed_mb": freed,
    }


def _poll_until_released(
    before: int | None, timeout: float = 25.0, settle_reads: int = 2, interval: float = 0.5
) -> int | None:
    """Wait for driver VRAM to actually drop, then for the drop to settle.

    Two identical readings are NOT evidence of a finished release: before the
    unload begins the figure is also unchanging, so a naive settle-check returns
    immediately and reports a false zero. The discriminator has to be *a reading
    below the starting figure* first, and only then a settled reading.

    Returns the settled figure. If the release never starts, returns the last
    observation -- never a fabricated reduction.
    """
    if before is None:
        return _nvidia_smi("memory.used")

    deadline = time.monotonic() + timeout
    released = False
    last = before
    stable = 0

    while time.monotonic() < deadline:
        time.sleep(interval)
        current = _nvidia_smi("memory.used")
        if current is None:
            return last

        if current < before:
            released = True

        if released and current == last:
            stable += 1
            if stable >= settle_reads:
                return current
        else:
            stable = 0
        last = current

    return last


def _parse_param(val: Any) -> Any:
    """Coerce a params value that may arrive as a JSON-ish string.

    MCP clients sometimes serialise the whole dict; accept JSON, Python
    literals, and the legacy ``{key=value, ...}`` form.
    """
    if val is None or isinstance(val, dict):
        return val
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            s = s[1:-1].strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            pass
        if s.startswith("{") and s.endswith("}"):
            inner = s[1:-1].strip()
            import re

            parts = re.split(r", (?=\w+(?:\w|\s)*=)", inner)
            result = {}
            for part in parts:
                if "=" not in part:
                    continue
                key, v = part.split("=", 1)
                key = key.strip()
                v = v.strip()
                if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                    v = v[1:-1]
                elif v.lower() in ("true", "false"):
                    v = v.lower() == "true"
                elif v.lower() in ("null", "none"):
                    v = None
                elif v.startswith("[") and v.endswith("]"):
                    try:
                        v = ast.literal_eval(v)
                    except Exception:
                        pass
                elif v.startswith("{") and v.endswith("}"):
                    iv = _parse_param(v)
                    v = iv if iv is not None else v
                else:
                    try:
                        if "." in v or "e" in v.lower():
                            v = float(v)
                        else:
                            v = int(v)
                    except ValueError:
                        pass
                result[key] = v
            return result if result else None
    return None


def _release_after_batch(keep_loaded: bool, job_count: int) -> dict[str, Any] | None:
    """Release ComfyUI's VRAM once a batch is finished.

    Mirrors the transcription pipeline's doctrine: release BETWEEN STAGES, not
    between files. Unloading after every job would make each subsequent job pay
    a full model-load penalty, which is a worse trade than holding the models
    for the duration of a batch.

    Returns the release report, or None when the caller asked to keep them.
    """
    if keep_loaded:
        return None
    released = _free_comfyui_vram()
    released["jobs_in_batch"] = job_count
    return released


def _resolve_param(spec: Any, param_name: str) -> Any:
    """Resolve a parameter specification to a concrete value.

    Supports:
    - Scalar (int, float, str, bool): returned as-is
    - List with 2 numeric elements: uniform random in [min, max]
    - List with N>2 elements or string elements: random choice from pool
    - "random": random from safe pool (sampler/scheduler/seed)
    - "random_unsafe": random from all pools (including experimental samplers)

    Args:
        spec: The parameter specification (scalar, list, or string)
        param_name: Name of the parameter (for type-aware resolution)

    Returns:
        Resolved concrete value
    """
    if spec is None:
        return None

    if isinstance(spec, (int, float, bool)):
        return spec

    if isinstance(spec, str):
        if spec == "random":
            if param_name == "sampler":
                return random.choice(_SAFE_SAMPLERS)
            elif param_name == "scheduler":
                return random.choice(_ALL_SCHEDULERS)
            elif param_name == "seed":
                return secrets.randbits(32)
            return spec
        if spec == "random_unsafe":
            if param_name == "sampler":
                return random.choice(_ALL_SAMPLERS)
            elif param_name == "scheduler":
                return random.choice(_ALL_SCHEDULERS)
            return spec
        return spec

    if isinstance(spec, list):
        if len(spec) == 0:
            return None
        # Numeric 2-element range: [min, max]
        if len(spec) == 2 and all(isinstance(x, (int, float)) for x in spec):
            if param_name in ("steps", "width", "height"):
                return random.randint(int(spec[0]), int(spec[1]))
            return random.uniform(spec[0], spec[1])
        # Choice pool: random choice from N elements
        return random.choice(spec)

    if isinstance(spec, dict):
        if spec.get("mode") == "random":
            if param_name == "sampler":
                return random.choice(_SAFE_SAMPLERS)
            elif param_name == "scheduler":
                return random.choice(_ALL_SCHEDULERS)
            elif param_name == "seed":
                return secrets.randbits(32)
        # A {"min":..,"max":..} dict is NOT a range. Silently passing it through
        # ships the dict itself to the workflow as a literal value and the range
        # is never applied -- a no-op that looks like a successful batch. Refuse
        # loudly and point at the correct shape.
        if "min" in spec or "max" in spec:
            raise ValueError(
                f"Invalid spec for '{param_name}': got a dict {spec!r}. "
                "Ranges must be a 2-element LIST [min, max]; a dict is not "
                "resolved and would silently be sent to the workflow as-is."
            )
        return spec

    # Dict or other complex types: return as-is (handled by caller)
    return spec


def _resolve_lora_config(spec: Any) -> Any:
    """Resolve LoRA randomization spec to concrete LoRA list.

    Supports:
    - None: no LoRAs
    - List of {name, weight}: pass through unchanged (legacy / pre-resolved)
    - Dict with mode='fixed': specified LoRAs, optional randomized weights
    - Dict with mode='auto': system picks 0-N LoRAs (Task 5); optional
      spec["pinned"] = [{"name": str, "weight": float}] are ALWAYS included
      first in chain and excluded from the random pool (no duplicates).

    Args:
        spec: LoRA specification

    Returns:
        List of {name, weight} dicts compatible with _inject_lora_nodes, or None
    """
    if spec is None:
        return None
    if isinstance(spec, list):
        return spec
    if not isinstance(spec, dict):
        return None

    mode = spec.get("mode")

    # Mode: fixed — user specifies which LoRAs, weights may be randomized
    if mode == "fixed":
        result = []
        for lora_def in spec.get("loras", []):
            name = lora_def.get("name")
            if not name:
                continue
            weight_spec = lora_def.get("weight", 0.8)
            if isinstance(weight_spec, list) and len(weight_spec) == 2:
                weight = random.uniform(weight_spec[0], weight_spec[1])
            else:
                weight = float(weight_spec)
            result.append({"name": name, "weight": weight})
        return result if result else None

    # Mode: auto — system picks 0-N LoRAs from available pool
    if mode == "auto":
        pool = spec.get("pool", "all")
        count_spec = spec.get("count", [0, 1])
        weight_spec = spec.get("weight", [0.3, 1.2])
        # NEW (2026-09-21): optional pinned LoRAs — always included, first in
        # chain, and excluded from the random pool to avoid duplicates.
        # spec["pinned"] = [{"name": str, "weight": float}, ...]
        pinned = spec.get("pinned") or []

        # Resolve LoRA count
        if isinstance(count_spec, list) and len(count_spec) == 2:
            count = random.randint(int(count_spec[0]), int(count_spec[1]))
        else:
            count = int(count_spec)

        # Build available LoRA pool, excluding any pinned names
        available = _get_available_loras()
        if pool != "all" and isinstance(pool, list):
            available = [name for name in available if name in pool]
        pinned_names = {p.get("name") for p in pinned if isinstance(p, dict) and p.get("name")}
        available = [name for name in available if name not in pinned_names]

        if count == 0 or not available:
            # Even with zero random picks, pinned LoRAs still apply.
            if pinned:
                return [dict(p) for p in pinned]
            return None

        # Select random LoRAs without replacement
        selected = random.sample(available, min(count, len(available)))

        result = [{"name": p.get("name"), "weight": float(p.get("weight", 0.8))} for p in pinned]
        for name in selected:
            if isinstance(weight_spec, list) and len(weight_spec) == 2:
                weight = random.uniform(weight_spec[0], weight_spec[1])
            else:
                weight = float(weight_spec)
            result.append({"name": name, "weight": weight})
        return result if result else None

    return None


def _inject_lora_nodes(workflow: dict, loras: list[dict[str, str | float]]) -> dict:
    """Inject multiple chained LoraLoader nodes and reroute connections.

    Chained flow with N LoRAs:
        Node 2 (UnetLoaderGGUF) -> LoRA1 -> LoRA2 -> ... -> LoRA_N -> Node 6 (KSampler)
        Node 1 (DualCLIPLoader) -> LoRA1 -> LoRA2 -> ... -> LoRA_N -> Nodes 4,5 (CLIPTextEncode)

    Args:
        workflow: ComfyUI workflow dict
        loras: List of {"name": str, "weight": float} dicts, max 5 items

    Returns:
        Modified workflow with chained LoraLoader nodes

    Raises:
        ValueError: If loras list is empty or exceeds 5 items
    """
    MAX_LORAS = 5

    if not loras:
        raise ValueError("loras list cannot be empty")
    if len(loras) > MAX_LORAS:
        raise ValueError(f"Maximum {MAX_LORAS} LoRAs supported, got {len(loras)}")

    # Find the next available node ID
    max_id = max(int(k) for k in workflow if k.isdigit())

    # Create LoraLoader nodes sequentially
    lora_node_ids = []
    for i, lora_config in enumerate(loras):
        lora_node_id = str(max_id + i + 1)
        lora_node_ids.append(lora_node_id)

        # Determine inputs based on position in chain
        if i == 0:
            # First LoRA: inputs from base loaders (Node 1 and Node 2)
            model_input = ["2", 0]  # UnetLoaderGGUF MODEL output
            clip_input = ["1", 0]  # DualCLIPLoader CLIP output
        else:
            # Subsequent LoRAs: inputs from previous LoraLoader
            prev_node_id = lora_node_ids[i - 1]
            model_input = [prev_node_id, 0]  # Previous LoRA MODEL output
            clip_input = [prev_node_id, 1]  # Previous LoRA CLIP output

        # Create LoraLoader node
        workflow[lora_node_id] = {
            "inputs": {
                "model": model_input,
                "clip": clip_input,
                "lora_name": str(lora_config["name"]),
                "strength_model": lora_config.get("weight", 0.8),
                "strength_clip": lora_config.get("weight", 0.8),
            },
            "class_type": "LoraLoader",
            "_meta": {"title": f"Load LoRA {i + 1}: {str(lora_config['name'])[:20]}"},
        }

    # Final LoRA node ID for downstream connections
    final_lora_id = lora_node_ids[-1]

    # Update KSampler to use final LoraLoader model output [0] = MODEL
    for node_id, node in workflow.items():
        if node.get("class_type") == "KSampler":
            node["inputs"]["model"] = [final_lora_id, 0]

    # Update CLIPTextEncode nodes to use final LoraLoader clip output [1] = CLIP
    for node_id, node in workflow.items():
        if node.get("class_type") == "CLIPTextEncode":
            node["inputs"]["clip"] = [final_lora_id, 1]

    return workflow


def _inject_ipadapter_nodes(
    workflow: dict,
    ref_images: list[str],
    weight: float = 0.6,
    weight_type: str = "style transfer",
    start_at: float = 0.0,
    end_at: float = 1.0,
    embeds_scaling: str = "V only",
    encode_batch_size: int = 0,
    clip_vision_model: str | None = None,
    ipadapter_model: str | None = None,
) -> dict:
    """Inject IP-Adapter nodes for multi-image reference generation.

    For 1 reference image: direct LoadImage -> IPAdapterBatch
    For 2+ references: LoadImages -> ImageBatch chain -> IPAdapterBatch

    ``clip_vision_model`` and ``ipadapter_model`` name the weights the two
    loader nodes should use. They default to the module constants, which are
    themselves environment-overridable, so nothing here assumes a particular
    install's model filenames.
    """
    clip_vision_model = clip_vision_model or IPADAPTER_CLIP_VISION_MODEL
    ipadapter_model = ipadapter_model or IPADAPTER_PLUS_MODEL
    if not ref_images:
        return workflow
    if len(ref_images) > 4:
        raise ValueError(f"Maximum 4 reference images supported, got {len(ref_images)}")

    input_dir = os.path.join(COMFYUI_PATH, "input")
    for img_name in ref_images:
        if not os.path.isfile(os.path.join(input_dir, img_name)):
            raise FileNotFoundError(f"Reference image not found: {img_name}")

    # Cleanup: remove orphaned template placeholder nodes
    nodes_to_remove = []
    for node_id, node in list(workflow.items()):
        class_type = node.get("class_type", "")
        if (
            class_type == "LoadImage"
            and node.get("inputs", {}).get("image") == "REFERENCE_IMAGE_PLACEHOLDER"
            or class_type in ("CLIPVisionLoader", "IPAdapterModelLoader", "IPAdapterBatch")
        ):
            nodes_to_remove.append(node_id)
    for node_id in nodes_to_remove:
        del workflow[node_id]
    if nodes_to_remove:
        log_info(
            f"Removed {len(nodes_to_remove)} orphaned template nodes: {nodes_to_remove}",
            tool_name="_inject_ipadapter_nodes",
        )

    max_id = max(int(k) for k in workflow if k.isdigit())
    next_id = max_id + 1

    model_source = None
    for node_id, node in workflow.items():
        if node.get("class_type") == "LoraLoader":
            if model_source is None or int(node_id) > int(model_source[0]):
                model_source = [node_id, 0]
    if model_source is None:
        model_source = ["2", 0]

    # Create LoadImage nodes for each reference
    load_image_ids = []
    for i, img_name in enumerate(ref_images):
        load_id = str(next_id + i)
        load_image_ids.append(load_id)
        workflow[load_id] = {
            "inputs": {"image": img_name},
            "class_type": "LoadImage",
            "_meta": {"title": f"Load Ref {i + 1}: {img_name}"},
        }

    # Build ImageBatch chain for 2+ images
    if len(ref_images) == 1:
        image_output = [load_image_ids[0], 0]
    else:
        batch_nodes: list[str] = []
        img_idx = 0
        last_batch_id = None
        while img_idx < len(load_image_ids):
            if img_idx == 0 and len(load_image_ids) >= 2:
                # ImageBatch 1: load_img[0] + load_img[1]
                batch_id = str(next_id + len(load_image_ids) + len(batch_nodes))
                workflow[batch_id] = {
                    "inputs": {"image1": [load_image_ids[0], 0], "image2": [load_image_ids[1], 0]},
                    "class_type": "ImageBatch",
                    "_meta": {"title": "ImageBatch 1"},
                }
                batch_nodes.append(batch_id)
                last_batch_id = batch_id
                img_idx = 2
            elif img_idx < len(load_image_ids):
                # Chain: last_batch + next_img
                batch_id = str(next_id + len(load_image_ids) + len(batch_nodes))
                workflow[batch_id] = {
                    "inputs": {"image1": [last_batch_id, 0], "image2": [load_image_ids[img_idx], 0]},
                    "class_type": "ImageBatch",
                    "_meta": {"title": f"ImageBatch {len(batch_nodes) + 1}"},
                }
                batch_nodes.append(batch_id)
                last_batch_id = batch_id
                img_idx += 1
        image_output = [last_batch_id, 0]

    # CLIPVisionLoader
    offset = len(load_image_ids) + max(0, len(ref_images) - 1)
    clipvision_id = str(next_id + offset)
    workflow[clipvision_id] = {
        "inputs": {"clip_name": clip_vision_model},
        "class_type": "CLIPVisionLoader",
        "_meta": {"title": "Load CLIP Vision (IP-Adapter)"},
    }

    # IPAdapterModelLoader
    ipadapter_loader_id = str(int(clipvision_id) + 1)
    workflow[ipadapter_loader_id] = {
        "inputs": {"ipadapter_file": ipadapter_model},
        "class_type": "IPAdapterModelLoader",
        "_meta": {"title": "Load IPAdapter Model"},
    }

    # IPAdapterBatch node
    ipadapter_batch_id = str(int(ipadapter_loader_id) + 1)
    workflow[ipadapter_batch_id] = {
        "inputs": {
            "model": model_source,
            "ipadapter": [ipadapter_loader_id, 0],
            "image": image_output,
            "weight": weight,
            "weight_type": weight_type,
            "start_at": start_at,
            "end_at": end_at,
            "embeds_scaling": embeds_scaling,
            "encode_batch_size": encode_batch_size,
            "clip_vision": [clipvision_id, 0],
        },
        "class_type": "IPAdapterBatch",
        "_meta": {"title": f"IPAdapter Batch ({len(ref_images)} refs)"},
    }

    # Reroute KSampler to IPAdapterBatch output
    for node_id, node in workflow.items():
        if node.get("class_type") == "KSampler":
            node["inputs"]["model"] = [ipadapter_batch_id, 0]

    # Reroute KSampler to IPAdapterBatch output
    for node_id, node in workflow.items():
        if node.get("class_type") == "KSampler":
            node["inputs"]["model"] = [ipadapter_batch_id, 0]

    return workflow


def _validate_lora(lora_name: str) -> bool:
    """Validate that a LoRA file exists in the loras directory."""
    lora_path = os.path.join(COMFYUI_PATH, "models", "loras", lora_name)
    return os.path.isfile(lora_path)


def _get_available_loras() -> list[str]:
    """Get list of available LoRA filenames."""
    loras_dir = os.path.join(COMFYUI_PATH, "models", "loras")
    if not os.path.exists(loras_dir):
        return []
    return [f for f in os.listdir(loras_dir) if f.endswith((".safetensors", ".ckpt", ".pt"))]


@mcp.tool()
def generate_image(
    prompt: str,
    negative_embedding: str | None = None,
    negative_prompt: str = "text, watermark, blurry, low quality",
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    cfg_scale: float = 7.0,
    sampler: str = "euler_ancestral",
    scheduler: str = "normal",
    lora: str | None = None,
    lora_weight: float = 0.8,
    loras: list[dict[str, str | float]] | None = None,
    seed: int | None = None,
    output_format: str = "png",
) -> dict[str, Any]:
    """Submit an async text-to-image generation job.

    Runs the packaged SDXL workflow: loads the diffusion model, encodes the
    prompt pair, samples, and decodes. Returns immediately with a job id --
    poll ``get_job_status`` for the result.

    Args:
        prompt: What to generate.
        negative_embedding: Textual-inversion embedding appended to the negative prompt.
        negative_prompt: What to avoid.
        width: Output width in pixels.
        height: Output height in pixels.
        steps: Sampling steps. More is slower and usually better.
        cfg_scale: Classifier-free guidance. Higher follows the prompt harder.
        sampler: Sampler name, as ComfyUI exposes it.
        scheduler: Scheduler name, as ComfyUI exposes it.
        lora: Single LoRA filename from comfyui/models/loras/.
        lora_weight: Weight for ``lora``.
        loras: Several LoRAs at once, as ``[{"name": ..., "weight": ...}]``.
        seed: Fixed seed for reproducibility. Omitted means a random one.
        output_format: ``png`` keeps provenance metadata; other values do not.

    Returns:
        A job record with ``job_id``, the resolved parameters, and the workflow path.
    """
    # Entry-point parameters, logged at DEBUG so a normal run stays quiet.
    log_debug("[MCP-DEBUG] === generate_image called ===")
    log_debug(f"[MCP-DEBUG] prompt: {repr(prompt[:60])}...")
    log_debug(f"[MCP-DEBUG] negative_prompt: {repr(negative_prompt[:60])}...")
    log_debug(f"[MCP-DEBUG] sampler: {repr(sampler)}, scheduler: {repr(scheduler)}")
    log_debug(f"[MCP-DEBUG] lora: {repr(lora)}, lora_weight: {lora_weight}")
    log_debug(f"[MCP-DEBUG] loras: {repr(loras)}")
    try:
        # Ensure server is running
        status = get_comfyui_status()
        if not status.get("running"):
            return {"error": "ComfyUI not running", "job_id": None, "status": "failed"}

        # Inject textual inversion embedding into negative prompt if specified
        if negative_embedding:
            embedding_tag = f"(embedding:{negative_embedding}:1.0)"
            negative_prompt = f"{negative_prompt}, {embedding_tag}"

        # Validate LoRA if specified
        if lora and not _validate_lora(lora):
            available = _get_available_loras()[:5]
            return {
                "error": f"LoRA not found: {lora}",
                "available_loras": available,
                "job_id": None,
                "status": "failed",
            }

        # Load workflow template (resolved relative to this package, never absolute)
        workflow_path = _workflow_path("sdxl_gguf_template_api.json")
        with open(workflow_path) as f:
            workflow = json.load(f)

        # Build LoRAs list (backward compat: single lora -> list)
        loras_list = []
        if loras:
            loras_list = loras
        elif lora:
            loras_list = [{"name": lora, "weight": lora_weight}]

        # Validate all LoRAs
        if loras_list:
            for lora_config in loras_list:
                if not _validate_lora(str(lora_config["name"])):
                    available = _get_available_loras()[:5]
                    return {
                        "error": f"LoRA not found: {lora_config['name']}",
                        "available_loras": available,
                        "job_id": None,
                        "status": "failed",
                    }
            # Inject all LoRAs as chained nodes
            workflow = _inject_lora_nodes(workflow, loras_list)

        # Generate seed if not provided
        if seed is None:
            seed = secrets.randbits(32)

        # Update workflow with parameters - FIXED WIRING
        for node_id, node in workflow.items():
            # Positive prompt
            if node.get("class_type") == "CLIPTextEncode":
                text = node.get("inputs", {}).get("text", "")
                if text == "POSITIVE_PROMPT_PLACEHOLDER":
                    node["inputs"]["text"] = prompt
                elif text == "NEGATIVE_PROMPT_PLACEHOLDER":
                    node["inputs"]["text"] = negative_prompt
            # KSampler parameters
            elif node.get("class_type") == "KSampler":
                node["inputs"]["sampler_name"] = sampler
                node["inputs"]["scheduler"] = scheduler
                node["inputs"]["seed"] = seed
                node["inputs"]["steps"] = steps
                node["inputs"]["cfg"] = cfg_scale
            # Image dimensions
            elif node.get("class_type") == "EmptyLatentImage":
                node["inputs"]["width"] = width
                node["inputs"]["height"] = height

        # Build metadata payload for native PNG embedding alongside generation
        generation_info = {
            "positive_prompt": prompt,
            "negative_prompt": negative_prompt,
            "negative_embedding": negative_embedding,
            "seed": seed,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "sampler": sampler,
            "scheduler": scheduler,
            "width": width,
            "height": height,
            "model": "",
            "loras": [{"name": lo["name"], "weight": lo.get("weight", 0.8)} for lo in loras_list],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "generator": _generator_tag(),
        }

        # Persist debug trace
        _trace_workflow(workflow, "generate_image")

        # Submit to ComfyUI — extra_pnginfo lands in PNG via native SaveImage node
        response = requests.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow, "extra_data": {"extra_pnginfo": {META_INFO_KEY: generation_info}}},
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            job_id = result.get("prompt_id")
            if job_id:
                _JOB_PARAMS[job_id] = {
                    "positive": prompt,
                    "negative": negative_prompt,
                    "negative_embedding": negative_embedding,
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "cfg": cfg_scale,
                    "sampler": sampler,
                    "scheduler": scheduler,
                    "loras": [{k: v for k, v in lo.items() if k in ("name", "weight")} for lo in loras_list],
                }
                _save_job_params(_JOB_PARAMS)
            return {"job_id": job_id, "status": "pending", "seed": seed, "estimated_time_sec": 75}
        else:
            return {"error": f"ComfyUI error: {response.status_code}", "job_id": None, "status": "failed"}
    except Exception as e:
        return {"error": str(e), "job_id": None, "status": "failed"}


# ============================================================================
# TOOL 6: get_job_status
# ============================================================================
@mcp.tool()
def get_job_status(job_id: str) -> dict[str, Any]:
    """Poll for job completion"""
    try:
        # Check history
        response = requests.get(f"{COMFYUI_URL}/history/{job_id}", timeout=10)

        if response.status_code == 200:
            history = response.json()
            job_data = history.get(job_id, {})

            if job_data.get("outputs"):
                # Job completed
                images = []
                for node_id, output in job_data["outputs"].items():
                    if "images" in output:
                        for img in output["images"]:
                            images.append(
                                {
                                    "filename": img.get("filename"),
                                    "subfolder": img.get("subfolder"),
                                    "url": f"{COMFYUI_URL}/view?filename={img.get('filename')}&subfolder={img.get('subfolder')}",
                                }
                            )

                # Auto-embed metadata for completed jobs (best-effort)
                if job_id in _JOB_PARAMS:
                    params = _JOB_PARAMS[job_id]
                    for img_meta in images:
                        sub = img_meta.get("subfolder", "") or ""
                        img_path = os.path.join(OUTPUT_PATH, sub, img_meta["filename"])
                        if os.path.isfile(img_path):
                            try:
                                embed_metadata(img_path, params, overwrite=True)
                            except Exception:
                                pass  # best-effort: don't fail generation

                return {"status": "completed", "progress": 1.0, "images": images, "error": None}
            elif job_data.get("prompt"):
                # Job exists but may be running
                return {"status": "running", "progress": 0.5, "images": [], "error": None}

        # Job not found in history yet
        return {"status": "pending", "progress": 0.0, "images": [], "error": None}
    except Exception as e:
        return {"status": "error", "progress": 0.0, "images": [], "error": str(e)}


# ============================================================================
# TOOL 6b: generate_ipadapter (IP-Adapter multi-reference image generation)
# ============================================================================
@mcp.tool()
def generate_ipadapter(
    prompt: str,
    reference_images: list[str],
    negative_prompt: str = "text, watermark, blurry, low quality",
    negative_embedding: str | None = None,
    width: int = 512,
    height: int = 512,
    steps: int = 10,
    cfg_scale: float = 7.5,
    sampler: str = "dpmpp_2m",
    scheduler: str = "karras",
    seed: int | None = None,
    reference_weight: float = 0.6,
    weight_type: str = "style transfer",
    denoise: float = 1.0,
    start_at: float = 0.0,
    end_at: float = 1.0,
    embeds_scaling: str = "V only",
    loras: list[dict[str, str | float]] | None = None,
    output_format: str = "png",
    output_prefix: str = "ipadapter",
    clip_vision_model: str | None = None,
    ipadapter_model: str | None = None,
) -> dict[str, Any]:
    """Generate image with IP-Adapter multi-reference style blending.

    Feeds 1--4 reference images into the diffusion model to influence
    style, composition, and content of the generated output.

    Args:
        prompt: Text prompt describing desired output
        reference_images: List of 1-4 image filenames from comfyui/input/
        negative_prompt: What to avoid in the output
        negative_embedding: Textual inversion embedding name (e.g., "DeepNegative_xl_v1") to append to negative prompt
        width: Output width (512 recommended for 6GB VRAM)
        height: Output height
        steps: Sampling steps (10-20 for speed, 30+ for quality)
        cfg_scale: Classifier-free guidance scale
        sampler: Sampling algorithm
        scheduler: Noise schedule
        seed: Fixed seed for reproducibility (None = random)
        reference_weight: How strongly reference images influence output (0.0-1.0+)
        weight_type: "style transfer", "composition", "linear", etc.
        denoise: Denoising strength (1.0 = full generation)
        start_at: When to start applying IP-Adapter (0.0 = beginning)
        end_at: When to stop applying IP-Adapter (1.0 = end)
        embeds_scaling: Embedding scaling strategy
        loras: Optional list of LoRA configs [{"name": str, "weight": float}]
        output_format: Always "png" for now
        output_prefix: Filename prefix for saved images

    Returns:
        Dict with job_id, status, queue_position
    """
    try:
        # Ensure server is running
        status = get_comfyui_status()
        if not status.get("running"):
            return {"error": "ComfyUI not running", "job_id": None, "status": "failed"}

        # Inject textual inversion embedding into negative prompt if specified
        if negative_embedding:
            embedding_tag = f"(embedding:{negative_embedding}:1.0)"
            negative_prompt = f"{negative_prompt}, {embedding_tag}"

        # Validate reference images
        if not reference_images:
            return {"error": "At least 1 reference image required", "job_id": None, "status": "failed"}
        if len(reference_images) > 4:
            return {
                "error": f"Maximum 4 reference images supported, got {len(reference_images)}",
                "job_id": None,
                "status": "failed",
            }

        input_dir = os.path.join(COMFYUI_PATH, "input")
        for img_name in reference_images:
            if not os.path.isfile(os.path.join(input_dir, img_name)):
                return {"error": f"Reference image not found in input/: {img_name}", "job_id": None, "status": "failed"}

        # Load workflow template (resolved relative to this package, never absolute)
        wf_path = _workflow_path("sdxl_ipadapter_template_api.json")
        with open(wf_path) as f:
            template = json.load(f)
        workflow = copy.deepcopy(template)

        # Validate and inject LoRAs
        if loras:
            for lora_config in loras:
                if not _validate_lora(str(lora_config["name"])):
                    available = _get_available_loras()[:5]
                    return {
                        "error": f"LoRA not found: {lora_config['name']}",
                        "available_loras": available,
                        "job_id": None,
                        "status": "failed",
                    }
            workflow = _inject_lora_nodes(workflow, loras)

        # Update prompt texts on existing CLIPTextEncode nodes
        for node in workflow.values():
            if node.get("class_type") == "CLIPTextEncode":
                title = node.get("_meta", {}).get("title", "").lower()
                if "negative" in title:
                    node["inputs"]["text"] = negative_prompt
                else:
                    node["inputs"]["text"] = prompt
        # Inject IP-Adapter nodes
        workflow = _inject_ipadapter_nodes(
            workflow,
            reference_images,
            weight=reference_weight,
            weight_type=weight_type,
            start_at=start_at,
            end_at=end_at,
            embeds_scaling=embeds_scaling,
            clip_vision_model=clip_vision_model,
            ipadapter_model=ipadapter_model,
        )

        # Seed
        if seed is None:
            seed = secrets.randbits(32)

        # Replace template placeholders with actual values
        for node_id, node in workflow.items():
            if node.get("class_type") == "CLIPTextEncode":
                if "positive_prompt" in str(node.get("inputs", {}).get("text", "")).lower():
                    node["inputs"]["text"] = prompt
            if node.get("class_type") == "CLIPTextEncode":
                if "negative_prompt" in str(node.get("inputs", {}).get("text", "")).lower():
                    node["inputs"]["text"] = negative_prompt

        # Direct node updates (known IDs from template)
        workflow["8"]["inputs"]["text"] = prompt
        workflow["9"]["inputs"]["text"] = negative_prompt
        workflow["10"]["inputs"]["width"] = width
        workflow["10"]["inputs"]["height"] = height
        workflow["11"]["inputs"]["seed"] = seed
        workflow["11"]["inputs"]["steps"] = steps
        workflow["11"]["inputs"]["cfg"] = cfg_scale
        workflow["11"]["inputs"]["sampler_name"] = sampler
        workflow["11"]["inputs"]["scheduler"] = scheduler
        workflow["11"]["inputs"]["denoise"] = denoise
        workflow["13"]["inputs"]["filename_prefix"] = output_prefix

        # VALIDATE: traverse model chain from KSampler to Base
        ok, msg = _validate_model_chain(workflow)
        if not ok:
            log_error(f"[MCP-ERROR] Model chain validation failed: {msg}")
            return {"error": f"Pipeline validation failed: {msg}", "job_id": None, "status": "failed"}
        log_debug(f"[MCP-TRACE] Model chain OK: {msg}")

        # Build metadata payload FROM ACTUAL WORKFLOW NODES (post-mutation source of truth)
        ksampler_node = None
        for nid, node in workflow.items():
            if node.get("class_type") == "KSampler":
                ksampler_node = node
                break

        generation_info = {
            "positive_prompt": prompt,
            "negative_prompt": negative_prompt,
            "negative_embedding": negative_embedding,
            "seed": ksampler_node["inputs"]["seed"] if ksampler_node else seed,
            "steps": ksampler_node["inputs"]["steps"] if ksampler_node else steps,
            "cfg_scale": ksampler_node["inputs"]["cfg"] if ksampler_node else cfg_scale,
            "sampler": ksampler_node["inputs"]["sampler_name"] if ksampler_node else sampler,
            "scheduler": ksampler_node["inputs"]["scheduler"] if ksampler_node else scheduler,
            "width": workflow["10"]["inputs"]["width"],
            "height": workflow["10"]["inputs"]["height"],
            "model": "",
            "loras": [{"name": lo["name"], "weight": lo.get("weight", 0.8)} for lo in (loras or [])],
            "reference_images": reference_images,
            "reference_weight": reference_weight,
            "weight_type": weight_type,
            "start_at": start_at,
            "end_at": end_at,
            "embeds_scaling": embeds_scaling,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "generator": _generator_tag("ipadapter"),
        }

        _trace_workflow(workflow, "generate_ipadapter")

        # Submit to ComfyUI with extra_pnginfo
        res = requests.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow, "extra_data": {"extra_pnginfo": {META_INFO_KEY: generation_info}}},
            timeout=30,
        )
        data = res.json()

        if "prompt_id" not in data:
            return {"error": data.get("error", "Unknown error"), "job_id": None, "status": "failed"}

        job_id = data["prompt_id"]

        # Cache params for metadata embedding later
        params = {
            "prompt": prompt,
            "positive": prompt,
            "negative": negative_prompt,
            "negative_embedding": negative_embedding,
            "reference_images": reference_images,
            "reference_weight": reference_weight,
            "weight_type": weight_type,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "sampler": sampler,
            "scheduler": scheduler,
            "seed": seed,
            "loras": loras,
            "denoise": denoise,
            "start_at": start_at,
            "end_at": end_at,
            "embeds_scaling": embeds_scaling,
        }
        _JOB_PARAMS[job_id] = params
        _save_job_params(_JOB_PARAMS)

        return {"job_id": job_id, "status": "queued", "queue_position": 0, "prompt_id": job_id}
    except Exception as e:
        return {"error": str(e), "job_id": None, "status": "failed"}


# ============================================================================
# TOOL 6c: batch_generate_ipadapter (Batch with randomization per job)
# ============================================================================
@mcp.tool()
def batch_generate_ipadapter(
    prompts: list[str],
    reference_images: list[str],
    shared_params: dict[str, Any] | str | None = None,
    randomize: dict[str, Any] | str | None = None,
    deduplicate_seeds: bool = True,
    priority: str = "normal",
    keep_loaded: bool = False,
) -> dict[str, Any]:
    """Submit batch IP-Adapter jobs with optional per-generation randomization.

    All jobs in the batch use the SAME reference images, but each prompt
    creates a distinct output. Supports parameter randomization per job.

    Args:
        prompts: List of positive prompt strings (one per job)
        reference_images: Shared reference images for all jobs in batch
        shared_params: Static parameters applied to all generations
        randomize: Per-generation randomization specs (overrides shared_params)
        deduplicate_seeds: Ensure unique seeds across batch
        priority: Queue priority hint

    Returns:
        Batch metadata with batch_id, total_jobs, queued_jobs, job_ids
    """
    try:
        batch_id = f"batch_ipa_{int(time.time())}"
        jobs = []
        used_seeds = set()

        shared_params = _parse_param(shared_params)
        randomize = _parse_param(randomize)
        # shared_params arrives as Any from the tool boundary: a caller can pass
        # a JSON string instead of an object. Coerce rather than let .items()
        # raise deep inside the batch loop.
        params = shared_params or {}
        if isinstance(params, str):
            try:
                params = json.loads(params) if params.strip() else {}
            except json.JSONDecodeError:
                return {"error": "shared_params was a string but not valid JSON", "success": False}
        if not isinstance(params, dict):
            return {"error": f"shared_params must be an object, got {type(params).__name__}", "success": False}
        if randomize and not isinstance(randomize, dict):
            return {"error": f"randomize must be an object, got {type(randomize).__name__}", "success": False}

        for i, prompt in enumerate(prompts):
            resolved = dict(params)
            if randomize:
                for key, spec in randomize.items():
                    if key == "loras":
                        resolved["loras"] = _resolve_lora_config(spec)
                    else:
                        resolved[key] = _resolve_param(spec, key)

            # Seed deduplication
            if deduplicate_seeds:
                seed = resolved.get("seed")
                if seed is not None:
                    while seed in used_seeds:
                        seed = secrets.randbits(32)
                        resolved["seed"] = seed
                    used_seeds.add(seed)

            log_debug(f"[MCP-BATCH-TRACE] Submitting prompt {i + 1}/{len(prompts)}: {prompt[:40]}...")
            result = generate_ipadapter(
                prompt=prompt,
                reference_images=reference_images,
                negative_prompt=resolved.get("negative_prompt", "text, watermark, blurry, low quality"),
                negative_embedding=resolved.get("negative_embedding"),
                width=resolved.get("width", 512),
                height=resolved.get("height", 512),
                steps=resolved.get("steps", 10),
                cfg_scale=resolved.get("cfg_scale", 7.5),
                sampler=resolved.get("sampler", "dpmpp_2m"),
                scheduler=resolved.get("scheduler", "karras"),
                seed=resolved.get("seed"),
                reference_weight=resolved.get("reference_weight", 0.6),
                weight_type=resolved.get("weight_type", "style transfer"),
                denoise=resolved.get("denoise", 1.0),
                start_at=resolved.get("start_at", 0.0),
                end_at=resolved.get("end_at", 1.0),
                embeds_scaling=resolved.get("embeds_scaling", "V only"),
                loras=resolved.get("loras"),
                output_prefix=resolved.get("output_prefix", f"ipadapter_batch_{i:04d}"),
            )
            log_debug(
                f"[MCP-BATCH-TRACE] Prompt {i + 1} result: type={type(result).__name__}, has_job_id={result.get('job_id') is not None}, job_id={result.get('job_id')}, error={result.get('error')}"
            )
            if result.get("job_id"):
                jobs.append(result["job_id"])
            else:
                log_debug(
                    f"[MCP-BATCH-TRACE] WARNING: Prompt {i + 1} returned NO job_id. Full result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}"
                )

        # Models are held for the whole batch (correct trade) and released at
        # the end unless the caller asked to keep them warm for another run.
        vram = _release_after_batch(keep_loaded, len(jobs))

        return {
            "batch_id": batch_id,
            "total_jobs": len(prompts),
            "queued_jobs": len(jobs),
            "job_ids": jobs,
            "queue_position": 0 if priority == "high" else len(jobs),
            "vram_released": (vram or {}).get("freed_mb"),
            "models_unloaded": bool(vram and vram.get("success")),
        }
    except Exception as e:
        return {"batch_id": None, "error": str(e), "total_jobs": 0, "queued_jobs": 0}


# ============================================================================
# TOOL 7: list_models
# ============================================================================
@mcp.tool()
def list_models(model_type: str = "all") -> list[dict[str, Any]]:
    """Discover available models: unet, clip, vae, lora"""
    models = []

    model_paths = {
        "unet": f"{COMFYUI_PATH}/models/unet",
        "clip": f"{COMFYUI_PATH}/models/clip",
        "vae": f"{COMFYUI_PATH}/models/vae",
        "loras": f"{COMFYUI_PATH}/models/loras",
        "ipadapter": f"{COMFYUI_PATH}/models/ipadapter",
        "clip_vision": f"{COMFYUI_PATH}/models/clip_vision",
    }

    for mtype, path in model_paths.items():
        if model_type != "all" and model_type != mtype:
            continue

        if os.path.exists(path):
            for filename in os.listdir(path):
                filepath = os.path.join(path, filename)
                if os.path.isfile(filepath):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in [".gguf", ".safetensors", ".ckpt", ".pt"]:
                        size_mb = os.path.getsize(filepath) / (1024**2)
                        models.append(
                            {
                                "name": filename,
                                "type": mtype,
                                "path": filepath,
                                "size_mb": round(size_mb, 2),
                                "format": ext.lstrip("."),
                            }
                        )

    return models


# ============================================================================
# TOOL 8: download_model
# ============================================================================
@mcp.tool()
def download_model(
    url: str,
    filename: str,
    model_type: str,
    api_token: str | None = None,
) -> dict[str, Any]:
    """Download a model file into the right ComfyUI models subdirectory.

    The contract is deliberately tight, because this writes into the user's
    model tree:
      * ``filename`` must be a bare name -- no path separators, no ``..``.
      * The extension must be plausible for the model type, so a stray ``.sh``
        cannot be dropped into ``models/``.
      * Only ``http``/``https`` URLs are fetched, and redirects are not followed
        to non-HTTP schemes.
      * The write is atomic: a partial download lands on a ``.part`` file and is
        renamed only on success, so an interrupted transfer can never leave a
        corrupt ``.safetensors`` that ComfyUI will later try to load.

    ``api_token`` is sent as a bearer header for gated hosts and is never
    logged. It may also be supplied via ``COMFYUI_DOWNLOAD_TOKEN`` so it does
    not have to travel through a tool argument.
    """
    allowed_ext = {
        "unet": {".gguf", ".safetensors", ".ckpt", ".pt", ".bin"},
        "lora": {".safetensors", ".ckpt", ".pt"},
        "vae": {".safetensors", ".ckpt", ".pt"},
        "clip": {".safetensors", ".ckpt", ".pt"},
        "ipadapter": {".safetensors", ".bin"},
        "clip_vision": {".safetensors", ".bin"},
    }
    type_dirs = {
        "unet": os.path.join(MODELS_PATH, "unet"),
        "lora": os.path.join(MODELS_PATH, "loras"),
        "vae": os.path.join(MODELS_PATH, "vae"),
        "clip": os.path.join(MODELS_PATH, "clip"),
        "ipadapter": os.path.join(MODELS_PATH, "ipadapter"),
        "clip_vision": os.path.join(MODELS_PATH, "clip_vision"),
    }

    if model_type not in type_dirs:
        return {
            "success": False,
            "error": f"Unknown model type: {model_type!r}. Valid types: {sorted(type_dirs)}",
            "bytes_downloaded": 0,
        }

    # Reject traversal before touching the filesystem.
    if not filename or filename != os.path.basename(filename):
        return {
            "success": False,
            "error": f"filename must be a bare file name, got {filename!r}",
            "bytes_downloaded": 0,
        }
    if filename in (".", "..") or "\x00" in filename:
        return {"success": False, "error": "invalid filename", "bytes_downloaded": 0}

    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_ext[model_type]:
        return {
            "success": False,
            "error": f"extension {ext!r} is not allowed for model_type "
            f"{model_type!r} (allowed: {sorted(allowed_ext[model_type])})",
            "bytes_downloaded": 0,
        }

    scheme = (url.split(":", 1)[0] or "").lower()
    if scheme not in ("http", "https"):
        return {
            "success": False,
            "error": f"unsupported URL scheme {scheme!r}; http/https only",
            "bytes_downloaded": 0,
        }

    target_dir = type_dirs[model_type]
    os.makedirs(target_dir, exist_ok=True)
    output_path = os.path.join(target_dir, filename)
    part_path = output_path + ".part"

    token = api_token or os.environ.get("COMFYUI_DOWNLOAD_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    downloaded = 0
    try:
        with requests.get(url, stream=True, timeout=300, headers=headers, allow_redirects=True) as response:
            response.raise_for_status()
            expected = int(response.headers.get("content-length", 0) or 0)
            with open(part_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

        # A truncated transfer is the failure that matters: it produces a file
        # that looks like a model and is not one.
        if expected and downloaded != expected:
            os.remove(part_path)
            return {
                "success": False,
                "error": f"incomplete download: got {downloaded} of {expected} bytes",
                "bytes_downloaded": downloaded,
            }

        os.replace(part_path, output_path)
        return {
            "success": True,
            "path": output_path,
            "bytes_downloaded": downloaded,
            "filename": filename,
        }
    except Exception as e:
        if os.path.exists(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass
        return {"success": False, "error": str(e), "bytes_downloaded": downloaded}


# ============================================================================
# TOOL 9: batch_generate
# ============================================================================
@mcp.tool()
def batch_generate(
    prompts: list[str],
    shared_params: dict[str, Any] | str | None = None,
    randomize: dict[str, Any] | str | None = None,
    deduplicate_seeds: bool = True,
    priority: str = "normal",
    keep_loaded: bool = False,
) -> dict[str, Any]:
    """Submit batch jobs with optional per-generation randomization.

    Args:
        prompts: List of positive prompt strings (one per job)
        shared_params: Static parameters applied to all generations
        randomize: Per-generation randomization specs (override shared_params)
            Supports: scalar, [min, max] range, choice pool, "random"/"random_unsafe"
        deduplicate_seeds: Ensure unique seeds across all jobs in this batch
        priority: Queue priority hint ("high" | "normal")

    Returns:
        Batch metadata with batch_id, total_jobs, queued_jobs, job_ids
    """
    try:
        batch_id = f"batch_{int(time.time())}"
        jobs = []
        used_seeds = set()

        shared_params = _parse_param(shared_params)
        randomize = _parse_param(randomize)

        # shared_params arrives as Any from the tool boundary: a caller can pass
        # a JSON string instead of an object. Coerce rather than let .items()
        # raise deep inside the batch loop.
        params = shared_params or {}
        if isinstance(params, str):
            try:
                params = json.loads(params) if params.strip() else {}
            except json.JSONDecodeError:
                return {"error": "shared_params was a string but not valid JSON", "success": False}
        if not isinstance(params, dict):
            return {"error": f"shared_params must be an object, got {type(params).__name__}", "success": False}
        if randomize and not isinstance(randomize, dict):
            return {"error": f"randomize must be an object, got {type(randomize).__name__}", "success": False}

        for i, prompt in enumerate(prompts):
            # Resolve randomized parameters (randomize overrides shared_params)
            resolved = dict(params)
            if randomize:
                for key, spec in randomize.items():
                    if key == "loras":
                        resolved["loras"] = _resolve_lora_config(spec)
                    else:
                        resolved[key] = _resolve_param(spec, key)

            # Seed deduplication
            if deduplicate_seeds:
                seed = resolved.get("seed")
                if seed is not None:
                    while seed in used_seeds:
                        seed = secrets.randbits(32)
                        resolved["seed"] = seed
                    used_seeds.add(seed)

            # VRAM soft warning for multi-LoRA
            resolved_loras = resolved.get("loras")
            if resolved_loras and isinstance(resolved_loras, list) and len(resolved_loras) > 1:
                log_warning(
                    f"Multiple LoRAs ({len(resolved_loras)}) may exceed VRAM on 6GB GPU. User Beware — proceed at own risk."
                )

            # Submit generation with resolved concrete values
            log_debug(f"[MCP-BATCH-TRACE] Submitting prompt {i + 1}/{len(prompts)}: {prompt[:40]}...")
            result = generate_image(
                prompt=prompt,
                negative_prompt=resolved.get("negative_prompt", "text, watermark, blurry"),
                negative_embedding=resolved.get("negative_embedding"),
                width=resolved.get("width", 1024),
                height=resolved.get("height", 1024),
                steps=resolved.get("steps", 20),
                cfg_scale=resolved.get("cfg_scale", 7.0),
                sampler=resolved.get("sampler", "dpmpp_2m"),
                scheduler=resolved.get("scheduler", "karras"),
                loras=resolved.get("loras"),
                seed=resolved.get("seed"),
            )
            log_debug(
                f"[MCP-BATCH-TRACE] Prompt {i + 1} result: type={type(result).__name__}, has_job_id={result.get('job_id') is not None}, job_id={result.get('job_id')}, error={result.get('error')}"
            )
            if result.get("job_id"):
                jobs.append(result["job_id"])
            else:
                log_debug(
                    f"[MCP-BATCH-TRACE] WARNING: Prompt {i + 1} returned NO job_id. Keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}"
                )

        # Models are held for the whole batch (correct trade) and released at
        # the end unless the caller asked to keep them warm for another run.
        vram = _release_after_batch(keep_loaded, len(jobs))

        return {
            "batch_id": batch_id,
            "total_jobs": len(prompts),
            "queued_jobs": len(jobs),
            "job_ids": jobs,
            "queue_position": 0 if priority == "high" else len(jobs),
            "vram_released": (vram or {}).get("freed_mb"),
            "models_unloaded": bool(vram and vram.get("success")),
        }
    except Exception as e:
        return {"batch_id": None, "error": str(e), "total_jobs": 0, "queued_jobs": 0}


# ============================================================================
# TOOL 10: get_vram_usage
# ============================================================================
@mcp.tool()
def get_vram_usage() -> dict[str, Any]:
    """Monitor live GPU memory and utilisation.

    Reads the driver, not torch: this process holds no CUDA tensors, so torch
    would report 0 no matter what ComfyUI is actually holding.
    """
    total = _nvidia_smi("memory.total")
    used = _nvidia_smi("memory.used")
    free = _nvidia_smi("memory.free")
    out = {
        "total_mb": total,
        "used_mb": used,
        "free_mb": free,
        "utilization_percent": round(used / total * 100, 2) if (used is not None and total) else None,
        "timestamp": datetime.now().isoformat(),
    }
    if total is None:
        out["error"] = "nvidia-smi unavailable or returned no data"
    return out


# ============================================================================
# TOOL 11: emergency_cleanup
# ============================================================================
@mcp.tool()
def emergency_cleanup(restart_comfyui: bool = False) -> dict[str, Any]:
    """Recover GPU memory by unloading ComfyUI's models.

    ComfyUI is a separate process and holds the models, so torch.cuda in THIS
    process can never free them. POST /free on ComfyUI is the only real lever;
    nvidia-smi measures the result.
    """
    actions = []
    try:
        freed = _free_comfyui_vram()
        if freed.get("success"):
            actions.append(
                f"ComfyUI /free unloaded models (used {freed['used_before_mb']}MB -> {freed['used_after_mb']}MB)"
            )
        else:
            actions.append(f"ComfyUI /free FAILED: {freed.get('error')}")

        if restart_comfyui:
            stop_result = stop_comfyui()
            if stop_result.get("success"):
                actions.append("ComfyUI stopped")
                time.sleep(2)
                if start_comfyui().get("success"):
                    actions.append("ComfyUI restarted")

        return {
            "freed_mb": freed.get("freed_mb"),
            "used_before_mb": freed.get("used_before_mb"),
            "used_after_mb": freed.get("used_after_mb"),
            "actions_taken": actions,
            "success": bool(freed.get("success")),
        }
    except Exception as e:
        return {"freed_mb": None, "actions_taken": actions, "error": str(e), "success": False}


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main() -> None:
    """Console-script entry point (stdio transport)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
