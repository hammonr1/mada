# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import pytest

from mada.core.config import AgentConfig
from mada.core.orchestration.agent_as_tool_strategy import (
    AgentAsToolOrchestrationStrategy,
)


@pytest.mark.asyncio
async def test_missing_named_mcp_definitions_fail_participant_initialization():
    strategy = AgentAsToolOrchestrationStrategy()
    orchestrator = SimpleNamespace(
        mcp_servers={},
        specialist_agents=[],
    )

    all_tools, failed_servers, failed_agents = await strategy._initialize_participants(
        orchestrator,
        [
            AgentConfig(
                agent_name="ToolAgent",
                description="Uses a named MCP server",
                instructions="Use tools.",
                mcp_servers=["missing_server"],
            )
        ],
    )

    assert all_tools == []
    assert failed_servers == []
    assert failed_agents == ["ToolAgent"]
    assert orchestrator.specialist_agents == []


@pytest.mark.asyncio
async def test_missing_named_mcp_definitions_preserve_legacy_server_path_fallback():
    strategy = AgentAsToolOrchestrationStrategy()
    orchestrator = SimpleNamespace(
        mcp_servers={},
        specialist_agents=[],
    )
    legacy_calls = []

    async def connect_legacy_agent(orchestrator, config, all_tools, failed_agents):
        legacy_calls.append(config.agent_name)
        all_tools.append(f"{config.agent_name}: {config.server_path}")

    strategy._connect_legacy_agent = connect_legacy_agent

    all_tools, failed_servers, failed_agents = await strategy._initialize_participants(
        orchestrator,
        [
            AgentConfig(
                agent_name="LegacyToolAgent",
                description="Uses a legacy MCP server path",
                instructions="Use tools.",
                mcp_servers=["missing_server"],
                server_path="/tmp/legacy_server.py",
            )
        ],
    )

    assert legacy_calls == ["LegacyToolAgent"]
    assert all_tools == ["LegacyToolAgent: /tmp/legacy_server.py"]
    assert failed_servers == []
    assert failed_agents == []
