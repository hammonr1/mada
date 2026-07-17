# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Base interface for orchestrator initialization strategies.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Tuple

from mada.core.config import AgentConfig, MCPServerConfig

if TYPE_CHECKING:
    from mada.core.orchestrator import MADAOrchestrator


class BaseOrchestrationStrategy(ABC):
    """
    Internal strategy boundary for orchestrator initialization patterns.
    """

    mode: str = ""

    @abstractmethod
    async def initialize(
        self,
        orchestrator: "MADAOrchestrator",
        agent_configs: List[AgentConfig],
        mcp_servers: Dict[str, MCPServerConfig] | None = None,
    ) -> Tuple[str, List[str]]:
        """Initialize the orchestrator for the strategy's orchestration mode."""
        pass
