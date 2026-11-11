"""ComfyUI MCP Server -- agentic image generation over the Model Context Protocol.

Exposes ComfyUI (workflow submission, batch generation, LoRA shaping, model
management) as MCP tools so an agent can drive image generation directly.

Design notes:
  * The MCP server is a SEPARATE PROCESS from ComfyUI. It holds no CUDA
    tensors, so VRAM is read via ``nvidia-smi`` and released via ComfyUI's own
    ``POST /free``. ``torch`` is deliberately not a dependency.
  * Workflow templates resolve relative to this package, never to an absolute
    path, so the package works from any checkout and any install prefix.
"""

__version__ = "2.0.0"

__all__ = ["__version__"]
