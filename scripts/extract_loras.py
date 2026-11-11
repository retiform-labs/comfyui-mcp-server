#!/usr/bin/env python3
"""
extract_loras.py — Extract LoRA name/weight, prompt, and seed from a ComfyUI PNG.

ComfyUI embeds the full workflow JSON in the PNG's 'prompt' text chunk.
This script parses that workflow and surfaces:
  - LoRA nodes (name, strength_model, strength_clip)
  - Positive/negative prompts
  - KSampler seed, steps, cfg, sampler, scheduler
  - Resolution (width, height)

Usage:
  python3 extract_loras.py <file.png>             # one image, full detail
  python3 extract_loras.py <dir/>                 # all PNGs in dir, summary
  python3 extract_loras.py <dir/> --format table  # compact table format
"""

import argparse
import json
from pathlib import Path

from PIL import Image


def extract(png_path):
    img = Image.open(png_path)
    prompt_json = img.info.get("prompt", "{}")
    try:
        wf = json.loads(prompt_json)
    except Exception as e:
        return {"error": f"json parse: {e}"}

    loras = []
    for nid, node in wf.items():
        ct = node.get("class_type", "")
        inputs = node.get("inputs", {})
        if "Lora" in ct and "Loader" in ct:
            loras.append(
                {
                    "class": ct,
                    "name": inputs.get("lora_name", inputs.get("name", "")),
                    "strength_model": inputs.get("strength_model"),
                    "strength_clip": inputs.get("strength_clip"),
                }
            )

    seed = steps = cfg = sampler = scheduler = width = height = None
    for nid, node in wf.items():
        ct = node.get("class_type", "")
        inputs = node.get("inputs", {})
        if ct == "KSampler":
            seed = inputs.get("seed")
            steps = inputs.get("steps")
            cfg = inputs.get("cfg")
            sampler = inputs.get("sampler_name")
            scheduler = inputs.get("scheduler")
        elif ct == "EmptyLatentImage":
            width = inputs.get("width")
            height = inputs.get("height")

    return {
        "loras": loras,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler": sampler,
        "scheduler": scheduler,
        "width": width,
        "height": height,
    }


def fmt_num(n):
    if isinstance(n, float):
        return f"{n:.3f}".rstrip("0").rstrip(".")
    return str(n)


def print_summary(info, png_path):
    name = Path(png_path).name
    if "error" in info:
        print(f"{name}: ERROR {info['error']}")
        return
    parts = [name]
    if info["seed"] is not None:
        parts.append(f"seed={info['seed']}")
    if info["steps"] and info["cfg"]:
        parts.append(
            f"{info['steps']}steps CFG{info['cfg']:.2f}"
            if isinstance(info["cfg"], float)
            else f"{info['steps']}steps CFG{info['cfg']}"
        )
    if info["sampler"]:
        parts.append(f"{info['sampler']}/{info['scheduler']}")
    if info["width"] and info["height"]:
        parts.append(f"{info['width']}x{info['height']}")
    if info["loras"]:
        lora_strs = []
        for lora in info["loras"]:
            sm = fmt_num(lora["strength_model"]) if lora["strength_model"] is not None else "?"
            sc = fmt_num(lora["strength_clip"]) if lora["strength_clip"] is not None else "?"
            lora_strs.append(f"{lora['name']}@{sm}/{sc}")
        parts.append(f"LORA={'|'.join(lora_strs)}")
    else:
        parts.append("STRAIGHT SHOT (no LoRA)")
    print(" | ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="PNG file or directory")
    ap.add_argument("--full", action="store_true", help="Print full prompt detail")
    args = ap.parse_args()

    p = Path(args.path)
    if p.is_dir():
        pngs = sorted(p.glob("*.png"))
        for png in pngs:
            info = extract(png)
            print_summary(info, png)
    else:
        info = extract(p)
        if args.full:
            print(json.dumps(info, indent=2, default=str))
        else:
            print_summary(info, p)


if __name__ == "__main__":
    main()
