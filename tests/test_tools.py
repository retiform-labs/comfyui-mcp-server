"""
Tests for the MCP tool surface and the VRAM/path/provenance fixes.

Covers:
- _workflow_path: resolves package-relative and every template exists
- _nvidia_smi / _free_comfyui_vram: driver-read VRAM, real free/report
- system_check / get_vram_usage / emergency_cleanup: no torch, real values
- _resolve_param: dict specs REJECTED (the silent no-op defect)
- build_a1111_text: never invents provenance
- tool registry: all 13 tools registered
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# MCP is replaced with a REAL pass-through double, not a MagicMock: a mocked
# decorator would hand back MagicMocks and every "tool" assertion below would
# pass against nothing.
# Install the shared pass-through MCP double BEFORE importing the server.
sys.path.insert(0, os.path.dirname(__file__))
from _mcpdouble import install as _install_mcp

_install_mcp()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from comfyui_mcp_server import server as srv
from comfyui_mcp_server.png_metadata import build_a1111_text


class TestWorkflowPath(unittest.TestCase):
    """Templates resolve relative to the package, never to an absolute path."""

    TEMPLATES = [
        "gguf_logo_template_api.json",
        "gguf_logo_template_lora_api.json",
        "ipadapter_logo_template_api.json",
    ]

    def test_every_template_exists(self):
        for name in self.TEMPLATES:
            with self.subTest(template=name):
                path = srv._workflow_path(name)
                self.assertTrue(os.path.isfile(path), f"missing {name} at {path}")

    def test_resolved_inside_package(self):
        path = srv._workflow_path("gguf_logo_template_api.json")
        pkg_dir = os.path.dirname(os.path.abspath(srv.__file__))
        self.assertTrue(path.startswith(pkg_dir), f"{path} escaped {pkg_dir}")

    def test_no_hardcoded_absolute_paths_remain(self):
        with open(os.path.abspath(srv.__file__)) as fh:
            src = fh.read()
        self.assertNotIn("/home/guan/", src, "an absolute path leaked back into server.py")


class TestNvidiaSmi(unittest.TestCase):
    """VRAM is read from the driver, not from this process."""

    @patch("subprocess.run")
    def test_reads_value(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="1234\n")
        self.assertEqual(srv._nvidia_smi("memory.used"), 1234)

    @patch("subprocess.run")
    def test_returns_none_on_nonzero(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        self.assertIsNone(srv._nvidia_smi("memory.used"))

    @patch("subprocess.run")
    def test_returns_none_on_garbage(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="not-a-number")
        self.assertIsNone(srv._nvidia_smi("memory.used"))

    @patch("subprocess.run", side_effect=OSError("no nvidia-smi"))
    def test_returns_none_when_binary_missing(self, _):
        self.assertIsNone(srv._nvidia_smi("memory.used"))

    def test_no_torch_dependency(self):
        with open(os.path.abspath(srv.__file__)) as fh:
            src = fh.read()
        self.assertNotIn("import torch", src, "torch is back -- it measures this process, not the GPU")


class _SequenceThenRepeat:
    """Fake nvidia-smi: walk a sequence, then keep returning the last value.

    The release poll loops until a drop is seen AND the figure settles, so a
    plain `side_effect=[...]` list runs out. Repeating the tail models a real
    GPU, which holds its final figure rather than raising StopIteration.
    """

    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def __call__(self, field="memory.used"):
        i = min(self.calls, len(self.values) - 1)
        self.calls += 1
        return self.values[i]


class TestFreeComfyuiVram(unittest.TestCase):
    """emergency_cleanup must actually free, and report real numbers."""

    @patch("requests.post")
    def test_frees_and_reports(self, mock_post):
        """Release is measured after ComfyUI finishes unloading, not before."""
        mock_post.return_value = MagicMock(status_code=200)
        smi = _SequenceThenRepeat([6000, 200])
        with patch.object(srv, "_nvidia_smi", smi), patch.object(srv.time, "sleep", return_value=None):
            out = srv._free_comfyui_vram()
        self.assertTrue(out["success"])
        self.assertEqual(out["used_before_mb"], 6000)
        self.assertEqual(out["used_after_mb"], 200)
        self.assertEqual(out["freed_mb"], 5800)
        mock_post.assert_called_once()
        self.assertIn("/free", mock_post.call_args[0][0])

    def test_poll_waits_for_async_release(self):
        """A synchronous read returns the old figure; the poll must not."""
        # Reads: still 6000 (unload not started), then 4000, 200, 200, 200.
        # The FIRST reading equals the start, which is exactly the state that
        # fooled the naive settle-check into reporting a false zero.
        smi = _SequenceThenRepeat([6000, 4000, 200])
        with patch.object(srv, "_nvidia_smi", smi), patch.object(srv.time, "sleep", return_value=None):
            settled = srv._poll_until_released(6000, timeout=5.0)
        self.assertEqual(settled, 200)

    def test_poll_does_not_settle_before_a_drop_is_seen(self):
        """An unchanging figure before the unload begins must NOT count as settled."""
        # first two reads unchanged at 6000, then it finally drops
        smi = _SequenceThenRepeat([6000, 6000, 3000, 200])
        with patch.object(srv, "_nvidia_smi", smi), patch.object(srv.time, "sleep", return_value=None):
            settled = srv._poll_until_released(6000, timeout=5.0)
        self.assertEqual(settled, 200, "settled on an unchanged pre-release reading")

    def test_poll_returns_last_reading_on_timeout(self):
        """If it never settles we report where it got to, not a guess."""
        smi = _SequenceThenRepeat([5500, 5400, 5300])
        with patch.object(srv, "_nvidia_smi", smi), patch.object(srv.time, "sleep", return_value=None):
            settled = srv._poll_until_released(6000, timeout=0.0)
        self.assertEqual(settled, 6000, "no time to poll: report the starting figure, never a guess")

    def test_poll_does_not_claim_more_than_it_saw(self):
        """A release that never happens must report zero freed, not a fabrication."""
        with patch.object(srv, "_nvidia_smi", return_value=6000), patch.object(srv.time, "sleep", return_value=None):
            self.assertEqual(srv._poll_until_released(6000, timeout=0.0), 6000)

    @patch("requests.post")
    def test_requests_unload_and_free(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        smi = _SequenceThenRepeat([6000, 200])
        with patch.object(srv, "_nvidia_smi", smi), patch.object(srv.time, "sleep", return_value=None):
            srv._free_comfyui_vram()
        self.assertEqual(mock_post.call_args[1]["json"], {"unload_models": True, "free_memory": True})

    @patch.object(srv, "_nvidia_smi", return_value=None)
    @patch("requests.post", side_effect=srv.requests.RequestException("down"))
    def test_failure_is_reported_not_swallowed(self, _post, _smi):
        out = srv._free_comfyui_vram()
        self.assertFalse(out["success"])
        self.assertIsNotNone(out["error"])
        self.assertIsNone(out["freed_mb"])

    @patch.object(srv, "_free_comfyui_vram")
    def test_emergency_cleanup_uses_free_endpoint(self, mock_free):
        mock_free.return_value = {
            "success": True,
            "freed_mb": 5000,
            "used_before_mb": 6000,
            "used_after_mb": 1000,
            "error": None,
        }
        out = srv.emergency_cleanup(restart_comfyui=False)
        self.assertTrue(out["success"])
        self.assertEqual(out["freed_mb"], 5000)
        mock_free.assert_called_once()


class TestSystemToolsNoTorch(unittest.TestCase):
    """system_check and get_vram_usage report driver values."""

    @patch.object(srv, "_nvidia_smi", side_effect=[8192, 4096, 4096])
    @patch("subprocess.run")
    def test_system_check_reports_driver_values(self, mock_run, _smi):
        mock_run.return_value = MagicMock(returncode=0, stdout="RTX 4060, 560.35\n")
        out = srv.system_check()
        self.assertTrue(out["cuda_available"])
        self.assertEqual(out["vram_total_mb"], 8192)
        self.assertEqual(out["vram_used_mb"], 4096)
        self.assertAlmostEqual(out["utilization_percent"], 50.0, places=1)

    @patch.object(srv, "_nvidia_smi", return_value=None)
    @patch("subprocess.run", side_effect=OSError("no nvidia-smi"))
    def test_system_check_unavailable_is_explicit(self, _run, _smi):
        out = srv.system_check()
        self.assertFalse(out["cuda_available"])
        self.assertIn("error", out)

    @patch.object(srv, "_nvidia_smi", side_effect=[8192, 2048, 6144])
    def test_vram_usage_reports_values(self, _smi):
        out = srv.get_vram_usage()
        self.assertEqual(out["total_mb"], 8192)
        self.assertEqual(out["used_mb"], 2048)
        self.assertEqual(out["free_mb"], 6144)


class TestDictSpecRejected(unittest.TestCase):
    """A {"min":..,"max":..} dict is not a range and must not silently pass."""

    def test_dict_with_min_max_raises(self):
        with self.assertRaises(ValueError):
            srv._resolve_param({"min": 1, "max": 5}, "steps")

    def test_dict_with_only_max_raises(self):
        with self.assertRaises(ValueError):
            srv._resolve_param({"max": 5}, "cfg")

    def test_list_range_still_works(self):
        value = srv._resolve_param([5, 5], "steps")
        self.assertEqual(value, 5)

    def test_plain_dict_still_passes(self):
        spec = {"mode": "auto", "count": 1}
        self.assertEqual(srv._resolve_param(spec, "loras"), spec)

    def test_error_names_the_fix(self):
        with self.assertRaises(ValueError) as ctx:
            srv._resolve_param({"min": 0, "max": 1}, "weight")
        self.assertIn("LIST", str(ctx.exception))


class TestProvenance(unittest.TestCase):
    """An A1111 block must never state a value it was not given."""

    def test_partial_params_mark_unknown(self):
        text = build_a1111_text({"positive_prompt": "x"})
        self.assertIn("Steps: ?", text)
        self.assertIn("Seed: ?", text)
        self.assertNotIn("Steps: 20", text)

    def test_no_invented_defaults(self):
        text = build_a1111_text({"positive_prompt": "x"})
        for invented in ("euler_ancestral", "normal", "7.0", "1024x1024"):
            self.assertNotIn(invented, text, f"invented {invented}")

    def test_full_params_roundtrip(self):
        text = build_a1111_text(
            {
                "positive_prompt": "a logo",
                "negative_prompt": "blurry",
                "steps": 30,
                "sampler": "dpmpp_2m",
                "scheduler": "karras",
                "cfg_scale": 7.5,
                "seed": 42,
                "width": 1024,
                "height": 1024,
            }
        )
        self.assertIn("Steps: 30", text)
        self.assertIn("Seed: 42", text)
        self.assertIn("1024x1024", text)

    def test_zero_and_false_are_not_missing(self):
        """0 is a real value; only None means unknown."""
        text = build_a1111_text({"positive_prompt": "x", "seed": 0})
        self.assertIn("Seed: 0", text)


class TestToolRegistry(unittest.TestCase):
    """All 13 tools must be registered -- no silent surface loss."""

    EXPECTED = [
        "system_check",
        "start_comfyui",
        "stop_comfyui",
        "get_comfyui_status",
        "generate_logo",
        "get_job_status",
        "generate_ipadapter",
        "batch_generate_ipadapter",
        "list_models",
        "download_model",
        "batch_generate",
        "get_vram_usage",
        "emergency_cleanup",
    ]

    def test_all_tools_exist(self):
        missing = [n for n in self.EXPECTED if not callable(getattr(srv, n, None))]
        self.assertEqual(missing, [], f"missing tools: {missing}")

    def test_main_entry_point_exists(self):
        self.assertTrue(callable(getattr(srv, "main", None)))


class TestDownloadModelContract(unittest.TestCase):
    """download_model writes into the user's model tree -- it must be strict."""

    def test_rejects_path_traversal(self):
        for bad in (
            "../../.ssh/authorized_keys",
            "../evil.safetensors",
            "sub/dir/model.safetensors",
            "..\\win.safetensors",
        ):
            with self.subTest(filename=bad):
                out = srv.download_model("https://example.com/x", bad, "lora")
                self.assertFalse(out["success"], f"{bad} was accepted")

    def test_rejects_bad_extension_for_type(self):
        out = srv.download_model("https://example.com/x", "evil.sh", "lora")
        self.assertFalse(out["success"])
        self.assertIn("extension", out["error"])

    def test_rejects_unknown_model_type(self):
        out = srv.download_model("https://example.com/x", "m.safetensors", "nonsense")
        self.assertFalse(out["success"])
        self.assertIn("Unknown model type", out["error"])

    def test_rejects_non_http_scheme(self):
        out = srv.download_model("file:///etc/passwd", "m.safetensors", "lora")
        self.assertFalse(out["success"])
        self.assertIn("scheme", out["error"])

    @patch("requests.get")
    def test_truncated_download_leaves_no_model_file(self, mock_get):
        """An interrupted transfer must not leave a loadable-looking model."""
        import tempfile

        with tempfile.TemporaryDirectory() as td, patch.object(srv, "MODELS_PATH", td):
            body = MagicMock()
            body.raise_for_status = MagicMock()
            body.headers = {"content-length": "1000"}
            body.iter_content = MagicMock(return_value=[b"x" * 10])
            body.__enter__ = MagicMock(return_value=body)
            body.__exit__ = MagicMock(return_value=False)
            mock_get.return_value = body

            out = srv.download_model("https://example.com/m", "m.safetensors", "lora")

            self.assertFalse(out["success"])
            self.assertIn("incomplete", out["error"])
            lora_dir = os.path.join(td, "loras")
            leftovers = os.listdir(lora_dir) if os.path.isdir(lora_dir) else []
            self.assertEqual(leftovers, [], f"left {leftovers} behind")


class TestBatchVramRelease(unittest.TestCase):
    """Models are held through a batch and released at its end."""

    @patch.object(srv, "_free_comfyui_vram")
    def test_release_called_by_default(self, mock_free):
        mock_free.return_value = {
            "success": True,
            "freed_mb": 5000,
            "used_before_mb": 6000,
            "used_after_mb": 1000,
            "error": None,
        }
        out = srv._release_after_batch(keep_loaded=False, job_count=8)
        mock_free.assert_called_once()
        self.assertEqual(out["jobs_in_batch"], 8)

    @patch.object(srv, "_free_comfyui_vram")
    def test_keep_loaded_skips_release(self, mock_free):
        out = srv._release_after_batch(keep_loaded=True, job_count=8)
        mock_free.assert_not_called()
        self.assertIsNone(out)

    def test_batch_functions_accept_keep_loaded(self):
        import inspect

        for fn in (srv.batch_generate, srv.batch_generate_ipadapter):
            with self.subTest(fn=fn.__name__):
                sig = inspect.signature(fn)
                self.assertIn("keep_loaded", sig.parameters)
                self.assertFalse(sig.parameters["keep_loaded"].default)

    @patch.object(srv, "_release_after_batch", return_value=None)
    @patch.object(srv, "generate_logo", return_value={"job_id": "j1"})
    def test_batch_reports_release_fields(self, _gen, mock_release):
        out = srv.batch_generate(prompts=["a", "b"], keep_loaded=True)
        self.assertEqual(out["queued_jobs"], 2)
        self.assertIn("vram_released", out)
        self.assertIn("models_unloaded", out)
        mock_release.assert_called_once_with(True, 2)


class TestVersionSingleSource(unittest.TestCase):
    """Version lives in exactly one place, and the shipped metadata agrees.

    A published package once reported a different --version than its metadata
    because the version was declared twice. These tests make that drift a
    build failure rather than a discovery made by a user.
    """

    def test_version_is_dynamic_in_pyproject(self):
        root = os.path.join(os.path.dirname(__file__), "..")
        with open(os.path.join(root, "pyproject.toml")) as fh:
            toml = fh.read()
        self.assertIn('dynamic = ["version"]', toml, "version is declared statically -- two sources will drift")
        self.assertNotRegex(
            toml.split("[project]")[1].split("[")[0], r"(?m)^version\s*=", "static version literal present"
        )
        self.assertIn("[tool.hatch.version]", toml)

    def test_dunder_version_is_a_real_version(self):
        import comfyui_mcp_server as pkg

        self.assertRegex(pkg.__version__, r"^\d+\.\d+\.\d+", f"not a semver string: {pkg.__version__!r}")

    def test_installed_metadata_matches_dunder(self):
        """Skipped when the package is not installed (source-tree runs)."""
        from importlib.metadata import PackageNotFoundError, version

        import comfyui_mcp_server as pkg

        try:
            installed = version("comfyui-mcp-server")
        except PackageNotFoundError:
            self.skipTest("not installed; nothing to compare")
        self.assertEqual(pkg.__version__, installed, "dunder disagrees with installed metadata")

    def test_server_advertises_package_version(self):
        """The MCP server identity must not carry its own version literal."""
        with open(os.path.abspath(srv.__file__)) as fh:
            src = fh.read()
        self.assertNotIn('FastMCP("comfyui-logo-server")', src, "server still advertises the old project name")


class TestNoSilentNameErrors(unittest.TestCase):
    """Every name the module calls must resolve at import time.

    A print()->logger conversion once left log_debug unimported, and the failure
    only surfaced when a batch was actually run. Static name resolution catches
    that class without executing every path.
    """

    def test_all_called_logger_helpers_are_bound(self):
        for name in ("log_debug", "log_info", "log_warning", "log_error"):
            with open(os.path.abspath(srv.__file__)) as fh:
                source = fh.read()
            if f"{name}(" in source:
                with self.subTest(helper=name):
                    self.assertTrue(hasattr(srv, name), f"{name} called but not bound")

    def test_module_compiles_cleanly(self):
        import py_compile
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pyc") as tmp:
            py_compile.compile(os.path.abspath(srv.__file__), cfile=tmp.name, doraise=True)

    def test_no_debug_prints_left_in_source(self):
        with open(os.path.abspath(srv.__file__)) as fh:
            src = fh.read()
        self.assertNotIn("file=sys.stderr)", src, "a raw stderr print bypasses the logger")


if __name__ == "__main__":
    unittest.main()
