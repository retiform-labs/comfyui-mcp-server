"""A real, minimal stand-in for the MCP SDK, shared by every test module.

Why not MagicMock: the server registers its surface with ``@mcp.tool()``. A
mocked decorator returns MagicMocks, so every decorated function becomes
unfalsifiable and tool assertions pass against nothing. This double is a
pass-through -- the decorator returns the original function.

Import this BEFORE importing the server module, in every test file: the server
module is cached on first import, so whichever mock is installed first wins.
"""
import sys
import types


class PassthroughFastMCP:
    """Records registered tool names and returns functions unchanged."""

    def __init__(self, *_args, **_kwargs):
        self.tools = []

    def tool(self, *dargs, **dkwargs):
        def _wrap(fn):
            self.tools.append(fn.__name__)
            return fn

        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return _wrap(dargs[0])
        return _wrap

    def run(self, *_args, **_kwargs):
        return None


def install():
    """Install the double into sys.modules. Idempotent."""
    instance = PassthroughFastMCP()
    mcp_mod = types.ModuleType("mcp")
    server_mod = types.ModuleType("mcp.server")
    fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
    fastmcp_mod.FastMCP = lambda *a, **k: instance
    server_mod.fastmcp = fastmcp_mod
    mcp_mod.server = server_mod
    sys.modules["mcp"] = mcp_mod
    sys.modules["mcp.server"] = server_mod
    sys.modules["mcp.server.fastmcp"] = fastmcp_mod
    return instance
