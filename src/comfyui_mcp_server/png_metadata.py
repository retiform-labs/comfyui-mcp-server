"""PNG metadata embedding and extraction using Pillow.

Implements A1111-compatible tEXt chunk format for storing generation
parameters in output PNGs. Also supports reading metadata back out.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .logger import log_error, log_info, log_warning

# A1111-compatible parameter keys
META_PARAMETERS = "parameters"
META_NEGATIVE = "negative_prompt"
META_SEED = "seed"
META_STEPS = "steps"
META_CFG = "cfg_scale"
META_SAMPLER = "sampler"
META_SCHEDULER = "scheduler"
META_WIDTH = "width"
META_HEIGHT = "height"
META_MODEL = "model"
META_LORAS = "loras"
META_TIMESTAMP = "timestamp"
META_GENERATOR = "generator"

# Structured provenance key. Renamed from "LogoGenInfo" when the tool surface
# became generic; the old key is still READ so images rendered before the rename
# stay inspectable. New images are written with the current key only -- writing
# both would embed the same payload twice.
META_INFO_KEY = "GenerationInfo"
META_INFO_KEY_LEGACY = "LogoGenInfo"
try:
    from . import __version__ as _PKG_VERSION
except ImportError:  # executed as a flat script
    _PKG_VERSION = "0.0.0"

# Provenance string written into every image. Derived, never a literal: a
# hardcoded version here silently mislabels every render after the first bump.
META_GENERATOR_VALUE = f"comfyui-mcp-server/{_PKG_VERSION}"


def _sanitize_for_tex(text: str) -> str:
    """Sanitize text for PNG tEXt chunk (must be valid latin-1)."""
    # Replace invalid characters, keep basic ASCII + common latin-1
    return text.encode("latin-1", "replace").decode("latin-1")


def build_a1111_text(params: dict[str, Any]) -> str:
    """Build A1111-style parameters text block from a params dict.

    Accepts both structured-info-style keys (positive_prompt, negative_prompt, cfg_scale)
    and legacy short keys (positive, negative, cfg).
    """
    positive = params.get("positive_prompt", params.get("positive", ""))
    negative = params.get("negative_prompt", params.get("negative", ""))

    # Provenance rule: NEVER invent a value. A default written into an A1111
    # block is a permanent, embedded, machine-readable lie -- readers (and
    # future tooling) cannot distinguish "steps: 20" that ran from "steps: 20"
    # that was assumed. Omit what we were not told; render unknown as ?.
    steps = params.get("steps")
    sampler = params.get("sampler")
    cfg = params.get("cfg_scale", params.get("cfg"))
    seed = params.get("seed")
    scheduler = params.get("scheduler")
    width = params.get("width")
    height = params.get("height")

    def _v(value):
        return "?" if value is None else value

    line = (
        f"Steps: {_v(steps)}, Sampler: {_v(sampler)}, "
        f"Schedule type: {_v(scheduler)}, CFG scale: {_v(cfg)}, "
        f"Seed: {_v(seed)}, Size: {_v(width)}x{_v(height)}"
    )

    lines = [positive, f"Negative prompt: {negative}", line]
    # Minimal A1111 doesn't include model/lora in main params block
    return _sanitize_for_tex("\n".join(lines))


def embed_metadata(filepath: str, params: dict[str, Any], overwrite: bool = False) -> bool:
    """Embed generation parameters into a PNG file as A1111-compatible tEXt chunks."""
    if not os.path.isfile(filepath):
        log_error(f"File not found: {filepath}", tool_name="png_metadata")
        return False

    try:
        img = Image.open(filepath)
        if img.format != "PNG":
            log_warning(f"Skipping non-PNG: {filepath}", tool_name="png_metadata")
            return False
    except Exception as e:
        log_error(f"Open failed ({filepath}): {e}", tool_name="png_metadata")
        return False

    # Build metadata
    pnginfo = PngInfo()

    # A1111 main params block
    a1111_text = build_a1111_text(params)
    pnginfo.add_text(META_PARAMETERS, a1111_text)

    # Supplemental structured metadata
    structured = {
        "positive_prompt": params.get("positive_prompt", params.get("positive", "")),
        "negative_prompt": params.get("negative_prompt", params.get("negative", "")),
        "negative_embedding": params.get("negative_embedding"),
        "seed": params.get("seed", -1),
        "steps": params.get("steps", 20),
        "cfg_scale": params.get("cfg_scale", params.get("cfg", 7.0)),
        "sampler": params.get("sampler", "euler_ancestral"),
        "scheduler": params.get("scheduler", "normal"),
        "width": params.get("width", 1024),
        "height": params.get("height", 1024),
        "model": params.get("model", ""),
        "loras": params.get("loras", []),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generator": params.get("generator", META_GENERATOR_VALUE),
    }

    try:
        pnginfo.add_text(META_INFO_KEY, _sanitize_for_tex(json.dumps(structured, indent=None)))
    except Exception as e:
        log_warning(f"Structured metadata failed: {e}", tool_name="png_metadata")

    # Save with metadata
    try:
        output_path = filepath if overwrite else filepath + ".tmp"
        img.save(output_path, "PNG", pnginfo=pnginfo)
        if not overwrite:
            os.replace(output_path, filepath)
        log_info(f"Metadata embedded: {os.path.basename(filepath)}", tool_name="png_metadata")
        return True
    except Exception as e:
        log_error(f"Embed failed ({filepath}): {e}", tool_name="png_metadata")
        if os.path.exists(filepath + ".tmp"):
            os.unlink(filepath + ".tmp")
        return False


def extract_metadata(filepath: str) -> dict[str, Any]:
    """Extract generation parameters from a PNG's tEXt chunks."""
    if not os.path.isfile(filepath):
        log_error(f"File not found: {filepath}", tool_name="png_metadata")
        return {}

    try:
        img = Image.open(filepath)
        if img.format != "PNG":
            log_warning(f"Not a PNG: {filepath}", tool_name="png_metadata")
            return {}
    except Exception as e:
        log_error(f"Open failed ({filepath}): {e}", tool_name="png_metadata")
        return {}

    metadata = dict(img.info) if img.info else {}

    result = {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "has_parameters": META_PARAMETERS in metadata,
        "has_structured": META_INFO_KEY in metadata or META_INFO_KEY_LEGACY in metadata,
        "parameters_text": metadata.get(META_PARAMETERS, ""),
        "structured": {},
        "rawkeys": list(metadata.keys()),
    }

    structured_key = META_INFO_KEY if META_INFO_KEY in metadata else META_INFO_KEY_LEGACY
    if structured_key in metadata:
        try:
            result["structured"] = json.loads(metadata[structured_key])
        except json.JSONDecodeError:
            log_warning(f"Invalid structured metadata in {filepath}", tool_name="png_metadata")
            result["structured"] = {"raw": metadata[structured_key]}

    # Also try to parse A1111 parameters text into structured form
    if META_PARAMETERS in metadata and not result["structured"]:
        a1111_text = metadata[META_PARAMETERS]
        lines = a1111_text.split("\n")
        if lines:
            result["structured"]["positive_prompt"] = lines[0]
        if len(lines) > 1 and lines[1].startswith("Negative prompt:"):
            result["structured"]["negative_prompt"] = lines[1].split(": ", 1)[1] if ": " in lines[1] else ""
        # Steps, sampler, etc. from line 2 are harder to parse reliably

    log_info(f"Metadata extracted: {os.path.basename(filepath)} (keys: {len(metadata)})", tool_name="png_metadata")
    return result


def annotate_outputs(output_dir: str, params: dict[str, Any], pattern: str = "image_output*") -> list[str]:
    """Find matching PNGs in a directory and embed metadata in each."""
    import fnmatch

    annotated: list[str] = []
    if not os.path.isdir(output_dir):
        log_error(f"Output directory missing: {output_dir}", tool_name="png_metadata")
        return annotated

    for filename in sorted(os.listdir(output_dir)):
        if fnmatch.fnmatch(filename, pattern) and filename.lower().endswith(".png"):
            filepath = os.path.join(output_dir, filename)
            if embed_metadata(filepath, params, overwrite=True):
                annotated.append(filename)

    log_info(f"Annotated {len(annotated)} files in {output_dir}", tool_name="png_metadata")
    return annotated
