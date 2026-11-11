# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-11-10

First release under the `comfyui-mcp-server` name, rebuilt from an earlier
internal working repository that remains the historical record. This is a
**major** bump because the package name, distribution name, module path and
several tool contracts changed.

### Changed

- **Package renamed** from the internal working name to `comfyui-mcp-server`,
  with the module at `comfyui_mcp_server` and a `src/` layout.
- **Requires Python 3.11+**, raised from 3.10. This is forced by the Pillow pin
  below rather than chosen: Pillow 13 ships cp311 wheels and no 3.10 build. The
  cost is nil because 3.10 reached end-of-life on 2026-10-01, five weeks before
  this release.
- **Pillow floor raised to 13.0.0.** `pip-audit` reported 59 advisories against
  Pillow 12.3.0, five of them critical; the package's only Pillow call is
  `Image.open()` on files supplied by a client, which is exactly the decoder
  surface those advisories describe. The audit is now clean.
- **`_parse_param` de-duplicated.** It was defined twice, byte-identical, at
  roughly 40 lines each. Now a single definition.
- **`.gitignore` rewritten.** The previous file used an inline comment on a
  pattern (`package-lock.json  # ...`), which git reads as a filename containing
  the comment text — the pattern matched nothing and the file was not ignored.
  Every pattern is now verified with `git check-ignore`.
- **Logging routed through the logger module.** Twenty-two raw
  `print(..., file=sys.stderr)` calls bypassed the existing logger; they now go
  through `log_debug` / `log_error`, so a normal run is quiet and detail is
  opt-in via `set_log_level("debug")`.
- **Version is declared once.** `pyproject.toml` now uses
  `dynamic = ["version"]` with `[tool.hatch.version]` reading the package's
  `__version__`. Declaring it in both places previously allowed drift between
  the installed metadata and what the code reported.
- **Provenance strings derived, not literal.** The generator string stamped into
  every image was a hardcoded version; it now comes from the package version.
- **Every filesystem root is environment-driven** — `COMFYUI_PATH`,
  `COMFYUI_MODELS_PATH`, `COMFYUI_OUTPUT_PATH`. Paths were hardcoded to the
  machine the project was developed on, which broke every install elsewhere.

### Fixed

- **The MCP handshake advertised the SDK's version, not ours.** `FastMCP`
  accepts no `version` argument in mcp 1.x and does not forward one to the
  low-level server, so `initialize` reported the SDK's own `1.30.0` while the
  package declared `2.0.0`. The version is now set on the inner server after
  construction, which is the attribute the initialize response reads.
- **VRAM tools never measured the GPU.** `system_check`, `get_vram_usage` and
  `emergency_cleanup` all read `torch.cuda` inside the MCP server's own process.
  That process never allocates a tensor — ComfyUI is a separate PID — so the
  readings were structurally zero: `vram_used_mb: 0` and `reserved: 0` while
  ComfyUI held the memory, and `emergency_cleanup`'s `freed_mb` was always `0`
  by construction. All three now read the driver via `nvidia-smi`.
- **`emergency_cleanup` now actually frees memory**, by calling ComfyUI's
  `POST /free {"unload_models": true, "free_memory": true}` and reporting real
  before/after figures. An empty `torch.cuda.empty_cache()` in a process that
  holds nothing could never have reclaimed anything.
- **VRAM is released when a batch finishes** (new `keep_loaded` flag, default
  `false`). Models are held for the duration of a batch — unloading between
  jobs would make every subsequent job pay a full model-load penalty — and
  released at completion.
- **Workflow templates resolve relative to the package.** Two call sites
  hardcoded an absolute path into a development directory.
- **`download_model` had no filename sanitization**, so a `../` path could write
  outside the models tree. Now: bare filename only, extension allowlist per
  model type, `http`/`https` only, and an optional bearer token for gated hosts.
- **A truncated download could leave a corrupt model file** that looks loadable.
  Writes are now atomic via a `.part` file, with an explicit incomplete-download
  error and a size check against `Content-Length`.
- **`png_metadata` invented provenance.** A partial params dict wrote
  `steps 20` / `euler_ancestral` / `cfg 7.0` / `seed -1` into the embedded A1111
  block, permanently, inside a binary. Unknown values are now rendered as `?`
  and never defaulted.
- **A `{"min": .., "max": ..}` dict passed as a parameter spec was sent to the
  workflow as a literal**, so randomised batches could run with nothing
  randomised. This now raises with the correct shape named — ranges are
  two-element lists.
- **Two bare `except:` clauses narrowed** (they swallowed `KeyboardInterrupt`
  and `SystemExit`).
- **A failed local import degraded silently.** `log_info = None` let the server
  run with no diagnostics; the import now fails loudly.

### Removed

- `torch` is no longer a dependency. It was roughly 2GB of CUDA used only to
  produce a number about the wrong process.
- Client deliverables, a live client brief, a brand template directory, a
  downloader script carrying an API token, four unused evaluation modules, and
  six generations of superseded batch scripts — none of which belong in a
  published package.
- No model weights, LoRAs, embeddings, CLIP or VAE files are distributed. Users
  source those themselves.

### Added

- `CHANGELOG.md`, `pyproject.toml` with `src/` layout, and an MIT `LICENSE` with
  joint copyright.
- Test suite: 85 tests, up from 34. The previous suite covered none of the
  thirteen tools and passed vacuously, because it replaced the MCP SDK with a
  `MagicMock` whose `@tool()` decorator returned mocks instead of functions.
  Tests now use a real pass-through double that rejects unknown constructor
  keywords against the real SDK's signature, so an argument the SDK does not
  accept fails under test instead of only at import in a user's environment.
- Tests that fail the build rather than the user: version single-sourcing,
  filesystem-path absoluteness, parameter-spec rejection, download contract,
  and VRAM release semantics.
- Tests that run against the **real MCP SDK** rather than the pass-through
  test double: an import in a subprocess, a live stdio handshake, and the tool
  count a client actually receives. This is the instrument that catches a wrong
  assumption about the dependency, which no mock can.
- `.github/FUNDING.yml` and a Sponsors section in the README, carrying the
  GitHub Sponsors link and the self-custody rails. Both travel inside the
  wheel, since the dist-info metadata embeds the README, so the addresses reach
  anyone reading the package from a registry.

[2.0.0]: https://github.com/retiform-labs/comfyui-mcp-server/releases/tag/v2.0.0
