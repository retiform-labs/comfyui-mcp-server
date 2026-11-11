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

# Parameters the REAL FastMCP accepts at mcp 1.x. Kept here so the double is
# not more permissive than the thing it stands in for: a double that accepts
# anything lets a TypeError-at-import ship green. This list was read from
# inspect.signature(FastMCP.__init__) on mcp 1.27.0 -- re-read it before
# trusting it against a different major.
_REAL_FASTMCP_PARAMS = frozenset(
    {
        "name",
        "instructions",
        "website_url",
        "icons",
        "auth_server_provider",
        "token_verifier",
        "event_store",
        "retry_interval",
        "tools",
        "debug",
        "log_level",
        "host",
        "port",
        "mount_path",
        "sse_path",
        "message_path",
        "streamable_http_path",
        "json_response",
        "stateless_http",
        "warn_on_duplicate_resources",
        "warn_on_duplicate_tools",
        "warn_on_duplicate_prompts",
        "dependencies",
        "lifespan",
        "auth",
        "transport_security",
    }
)


class PassthroughFastMCP:
    """Records registered tool names and returns functions unchanged.

    Rejects unknown keyword arguments the same way the real SDK does, so a
    bad constructor call fails here instead of only in production.
    """

    def __init__(self, *args, **kwargs):
        unexpected = set(kwargs) - _REAL_FASTMCP_PARAMS
        if unexpected:
            raise TypeError(f"FastMCP.__init__() got an unexpected keyword argument {sorted(unexpected)[0]!r}")
        self.tools = []
        # The server sets `mcp._mcp_server.version` after construction, because
        # FastMCP never forwards a version to the low-level server and the
        # handshake would otherwise advertise the SDK's own. The double must
        # expose the same attribute or the real call site breaks under test.
        self._mcp_server = types.SimpleNamespace(version=None)

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
