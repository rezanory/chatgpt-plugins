from __future__ import annotations

import asyncio

from mcp import Client
from chatgpt_plugin_kaggle_gateway import server

_EXPECTED_TOOLS = {
    "kaggle_accounts",
    "kaggle_auth_check",
    "kaggle_auth_check_all",
    "kaggle_kernels_list",
    "kaggle_kernels_inventory_all",
    "kaggle_kernel_status",
    "kaggle_kernel_logs",
    "kaggle_kernel_output_manifest",
}


async def _tools():
    async with Client(server._mcp, raise_exceptions=True) as client:
        return (await client.list_tools()).tools


def test_mcp_exposes_only_expected_read_recovery_surface():
    tools = asyncio.run(_tools())
    by_name = {tool.name: tool for tool in tools}
    assert set(by_name) == _EXPECTED_TOOLS

    for tool in by_name.values():
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False
        assert tool.annotations.idempotentHint is True
