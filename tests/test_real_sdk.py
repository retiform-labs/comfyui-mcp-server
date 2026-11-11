"""Tests that run against the REAL MCP SDK, not the test double.

Every other test module installs a pass-through double so the server's tool
surface can be exercised without a live MCP runtime. That is necessary -- and
it is exactly how a real defect shipped green on 2026-11-10: the constructor
call passed an argument the real SDK does not accept, every mock accepted it,
and importing the built package raised TypeError.

These tests import the server against the installed SDK. They are the only
instrument that can catch "our assumptions about the dependency are wrong".
"""

import os
import subprocess
import sys
import unittest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


class TestAgainstRealSdk(unittest.TestCase):
    def _fresh_import(self, code):
        """Import in a subprocess so sys.modules is never polluted by the double."""
        env = dict(os.environ, PYTHONPATH=_SRC)
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)

    def test_server_imports_against_the_real_sdk(self):
        """The single most basic property: `import` works. A mock cannot prove it."""
        proc = self._fresh_import("import comfyui_mcp_server.server")
        self.assertEqual(proc.returncode, 0, f"import failed against the real SDK:\n{proc.stderr}")

    def test_fastmcp_receives_only_supported_arguments(self):
        """Guard the constructor signature explicitly, with a message that names the fix."""
        code = (
            "import inspect\n"
            "from mcp.server.fastmcp import FastMCP\n"
            "import comfyui_mcp_server.server as s\n"
            "sig = inspect.signature(FastMCP.__init__).parameters\n"
            "unexpected = [k for k in ('version',) if k not in sig]\n"
            "print('UNSUPPORTED:' + ','.join(unexpected))\n"
        )
        proc = self._fresh_import(code)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_handshake_lists_thirteen_tools(self):
        """A real stdio handshake: the number a client actually receives."""
        # stdio_client spawns the child with a FILTERED environment, so the
        # parent's PYTHONPATH does not reach it. Pass it explicitly.
        code = (
            "import asyncio, os, sys\n"
            "from mcp import ClientSession, StdioServerParameters\n"
            "from mcp.client.stdio import stdio_client\n"
            "async def main():\n"
            "    params = StdioServerParameters(\n"
            "        command=sys.executable,\n"
            "        args=['-m', 'comfyui_mcp_server.server'],\n"
            "        env=dict(os.environ, PYTHONPATH=os.environ['PYTHONPATH']))\n"
            "    async with stdio_client(params) as (r, w):\n"
            "        async with ClientSession(r, w) as s:\n"
            "            init = await s.initialize()\n"
            "            print('SERVER_VERSION:' + str(init.serverInfo.version))\n"
            "            listed = await s.list_tools()\n"
            "            print('TOOLS:' + str(len(listed.tools)))\n"
            "asyncio.run(main())\n"
        )
        proc = self._fresh_import(code)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("TOOLS:")]
        self.assertTrue(line, f"no tool count printed:\n{proc.stdout}\n{proc.stderr}")
        self.assertEqual(line[0], "TOOLS:13", f"a client receives {line[0]}, expected 13 tools")

        # The handshake must advertise OUR version. FastMCP forwards none to the
        # low-level server, so it silently reported the SDK's own (1.30.0) --
        # the same drift defect the pyproject single-sourcing fixed, one layer
        # deeper and invisible to every unit test.
        ver = [ln for ln in proc.stdout.splitlines() if ln.startswith("SERVER_VERSION:")]
        self.assertTrue(ver, f"no server version printed:\n{proc.stdout}")
        # Read the expected version from the source file rather than importing:
        # this process has not bootstrapped sys.path for the package.
        init_py = os.path.join(_SRC, "comfyui_mcp_server", "__init__.py")
        with open(init_py) as fh:
            expected = [ln.split("=", 1)[1].strip().strip("\"'") for ln in fh if ln.startswith("__version__")][0]
        self.assertEqual(
            ver[0], f"SERVER_VERSION:{expected}", f"handshake advertises {ver[0]}, package declares {expected}"
        )


class TestDoubleIsNotTooPermissive(unittest.TestCase):
    """The double must reject what the real SDK rejects, or it hides defects."""

    def test_double_rejects_unknown_kwarg(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
        import _mcpdouble

        Fake = _mcpdouble.PassthroughFastMCP
        with self.assertRaises(TypeError) as ctx:
            Fake("name", version="2.0.0")
        self.assertIn("version", str(ctx.exception))

    def test_double_accepts_a_real_argument(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
        import _mcpdouble

        _mcpdouble.PassthroughFastMCP("name", instructions="hi", debug=True)


if __name__ == "__main__":
    unittest.main()
