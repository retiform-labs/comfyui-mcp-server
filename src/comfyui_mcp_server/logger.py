"""Structured logging for ComfyUI MCP Server.

Provides consistent log output across all tools with timestamps,
level indicators, and tool context. No external dependencies.
"""

import logging
import sys

# Configure default logger
_logger: logging.Logger | None = None


def get_logger(name: str = "comfyui-mcp-server") -> logging.Logger:
    """Get or create the configured logger singleton."""
    global _logger
    if _logger is None:
        _logger = logging.getLogger(name)
        if not _logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            fmt = "%(asctime)s | %(levelname)-7s | %(tool_name)s | %(message)s"
            formatter = logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S")
            handler.setFormatter(formatter)
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)
    return _logger


def log_info(message: str, tool_name: str = "server") -> None:
    """Log an INFO level message."""
    extra = {"tool_name": tool_name}
    get_logger().info(message, extra=extra)


def log_warning(message: str, tool_name: str = "server") -> None:
    """Log a WARNING level message."""
    extra = {"tool_name": tool_name}
    get_logger().warning(message, extra=extra)


def log_error(message: str, tool_name: str = "server") -> None:
    """Log an ERROR level message."""
    extra = {"tool_name": tool_name}
    get_logger().error(message, extra=extra)


def log_debug(message: str, tool_name: str = "server") -> None:
    """Log a DEBUG level message."""
    extra = {"tool_name": tool_name}
    get_logger().debug(message, extra=extra)


def set_log_level(level: str) -> None:
    """Set the global log level. Options: DEBUG, INFO, WARNING, ERROR."""
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    numeric = level_map.get(level.lower(), logging.INFO)
    get_logger().setLevel(numeric)
    log_info(f"Log level set to {level.upper()}", tool_name="server")
