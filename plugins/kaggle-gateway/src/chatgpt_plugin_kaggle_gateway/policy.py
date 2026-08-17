from __future__ import annotations

import re
from enum import StrEnum

_TOOL = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,119}$")

# Authentication/configuration is owned by the gateway. Upstream tools must never switch
# credentials, start browser login, or mutate auth state on behalf of the model.
_BLOCKED_EXACT = {
    "authorize",
    "auth_login",
    "login",
    "logout",
    "config_set",
    "config_unset",
}
_BLOCKED_PARTS = ("credential", "oauth", "api_token", "access_token")
_WRITE_HINTS = (
    "cancel",
    "create",
    "delete",
    "execute",
    "push",
    "run",
    "save",
    "set",
    "start",
    "submit",
    "update",
    "upload",
)
_READ_HINTS = (
    "download",
    "fetch",
    "get",
    "list",
    "output",
    "pull",
    "search",
    "show",
    "status",
)


class ToolClass(StrEnum):
    READ = "read"
    WRITE = "write"
    UNKNOWN = "unknown"


def validate_tool_name(tool_name: str) -> str:
    name = tool_name.strip()
    if not _TOOL.fullmatch(name):
        raise ValueError("unsupported Kaggle MCP tool name")
    lower = name.casefold()
    if lower in _BLOCKED_EXACT or any(part in lower for part in _BLOCKED_PARTS):
        raise ValueError(f"Kaggle MCP tool is blocked by gateway policy: {name}")
    return name


def classify_tool(tool_name: str) -> ToolClass:
    name = validate_tool_name(tool_name).casefold()
    if any(hint in name for hint in _WRITE_HINTS):
        return ToolClass.WRITE
    if any(hint in name for hint in _READ_HINTS):
        return ToolClass.READ
    return ToolClass.UNKNOWN


def require_read_tool(tool_name: str) -> str:
    name = validate_tool_name(tool_name)
    kind = classify_tool(name)
    if kind is not ToolClass.READ:
        raise ValueError(
            f"tool {name!r} is not classified read-only; use kaggle_write for explicit approval"
        )
    return name


def require_write_tool(tool_name: str) -> str:
    name = validate_tool_name(tool_name)
    kind = classify_tool(name)
    if kind is ToolClass.READ:
        raise ValueError(f"tool {name!r} is read-only; use kaggle_read")
    return name
