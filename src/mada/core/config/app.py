# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Application configuration definitions and loading utilities.

This module defines the top-level configuration model for the MADA
multi-agent application. It provides `AppConfig`, which aggregates model,
agent, database, MCP server, and interface configuration objects, along with
`load_config_from_json` for loading a complete application configuration from
a JSON file.
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
from pathlib import Path

from mada.core.config.agents import AgentConfig
from mada.core.config.a2a import (
    A2AConfig,
    RemoteA2AAgentConfig,
    load_a2a_agents_config,
    load_a2a_config,
)
from mada.core.config.database import DatabaseConfig, load_database_config
from mada.core.config.interface import InterfaceConfig
from mada.core.config.mcp_servers import MCPServerConfig
from mada.core.config.models import ModelConfig, load_model_config
from mada.core.config.orchestration import (
    OrchestrationConfig,
    load_orchestration_config,
)

LOG = logging.getLogger("mada-interface")


@dataclass
class SkillRuntimeConfig:
    """
    Runtime policy configuration for manifest-based skills.

    Attributes:
        default_script_timeout_seconds: Default timeout for skill script execution.
        max_script_output_bytes: Maximum captured stdout/stderr size for a skill script.
        max_resource_bytes: Maximum readable size for a skill resource file.
        default_skill_script_approval_mode: Approval mode applied when no
            specific rule matches a script. One of "prompt", "approve", or "deny".
        skill_script_approval_modes: Per-skill and per-script approval overrides.
            Keys are matched most specific first: "skill_name:script_name", then
            "skill_name", then "*" as a catch-all. Values are "approve" or "deny".
    """

    default_script_timeout_seconds: int = 30
    max_script_output_bytes: int = 32 * 1024
    max_resource_bytes: int = 32 * 1024
    default_skill_script_approval_mode: str = "prompt"
    skill_script_approval_modes: Dict[str, str] = field(default_factory=dict)


@dataclass
class AppConfig:
    """
    Top-level configuration for the MADA multi-agent application.

    This configuration aggregates model settings, agent definitions, named MCP
    server definitions, database settings, orchestration settings, and optional
    interface layout settings. It can be loaded from a JSON file using the
    `from_dict` method.

    Attributes:
        model (ModelConfig): Provider-specific configuration for the model backend.
        agents (List[AgentConfig]): List of specialist agent definitions,
            including descriptions, optional named MCP server references, and
            legacy `server_path` entries.
        database (DatabaseConfig): Configuration for the database connection.
        mcp_servers (Dict[str, MCPServerConfig]): Named MCP server definitions.
        interface (InterfaceConfig): Configuration for the Gradio interface layout and options.
        orchestration (OrchestrationConfig): Mode and participant selection.
    """

    model: ModelConfig
    agents: List[AgentConfig]
    database: DatabaseConfig
    mcp_servers: Dict[str, MCPServerConfig] = None  # MCP server configurations
    interface: InterfaceConfig = None  # Optional, used only by the Gradio app
    skill_paths: List[Union[str, Path]] = field(default_factory=list)
    skill_runtime_config: SkillRuntimeConfig = field(default_factory=SkillRuntimeConfig)
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    a2a: A2AConfig = field(default_factory=A2AConfig)
    a2a_agents: Dict[str, RemoteA2AAgentConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        config_dict: Dict[str, Any],
        a2a_card_path_base: str | Path | None = None,
    ) -> "AppConfig":
        """
        Create an AppConfig instance from a dictionary.

        This method extracts and constructs the model, interface, and
        agent configurations from a flat configuration dictionary.

        Args:
            config_dict (Dict[str, Any]): Dictionary containing keys 'model', 'interface', and 'agents'.

        Returns:
            AppConfig: The fully populated application configuration.

        Raises:
            ValueError: If the 'agents' list is missing or empty.
        """
        app_conf = {}

        # Load model configuration
        model_settings = config_dict.get("model")
        if not model_settings:
            raise ValueError(
                "Please provide 'model' settings in the configuration for your app."
            )
        model_cfg = load_model_config(model_settings)
        app_conf["model"] = model_cfg

        # Load agents
        agent_entries = config_dict.get("agents")
        if not agent_entries:
            raise ValueError(
                "No agents were provided. Please define at least one agent."
            )
        agent_cfgs = [AgentConfig.from_dict(agent) for agent in agent_entries]
        app_conf["agents"] = agent_cfgs

        # Load database
        database_config = config_dict.get("database", {})
        app_conf["database"] = load_database_config(database_config)

        orchestration_cfg = load_orchestration_config(config_dict.get("orchestration"))
        orchestration_cfg.validate_participants(
            [agent.agent_name for agent in agent_cfgs]
        )
        app_conf["orchestration"] = orchestration_cfg

        a2a_self_config, a2a_agents_config = _get_a2a_config_blocks(config_dict)
        app_conf["a2a"] = load_a2a_config(
            a2a_self_config,
            card_path_base=a2a_card_path_base,
        )
        app_conf["a2a_agents"] = load_a2a_agents_config(a2a_agents_config)

        # Load MCP servers configuration (optional)
        python_exe = config_dict.get("python_executable", sys.executable)
        mcp_servers_entry = config_dict.get("mcp_servers")
        if mcp_servers_entry:
            mcp_servers_cfg = {}
            for name, server_config in mcp_servers_entry.items():
                if "python_executable" not in server_config:
                    server_config = {**server_config, "python_executable": python_exe}
                mcp_servers_cfg[name] = MCPServerConfig(**server_config)
            app_conf["mcp_servers"] = mcp_servers_cfg

        # Load skill paths (optional)
        skill_paths_entry, skill_runtime_entry = _get_skills_config_blocks(config_dict)
        if skill_paths_entry:
            if not isinstance(skill_paths_entry, list):
                raise ValueError("'skills.skill_paths' must be a list of paths.")
            skill_paths = []
            for raw_path in skill_paths_entry:
                if not isinstance(raw_path, str) or not raw_path.strip():
                    raise ValueError(
                        "Each entry in 'skills.skill_paths' must be a non-empty path."
                    )
                skill_paths.append(raw_path.strip())
            app_conf["skill_paths"] = skill_paths

        # Load skill runtime configuration (optional)
        if skill_runtime_entry:
            if not isinstance(skill_runtime_entry, dict):
                raise ValueError(
                    "'skills.skill_runtime' must be a mapping of runtime settings."
                )
            app_conf["skill_runtime_config"] = SkillRuntimeConfig(**skill_runtime_entry)

        # Load interface configuration (optional for multiagent app)
        interface_entry = config_dict.get("interface")
        if interface_entry:
            interface_cfg = InterfaceConfig(**config_dict["interface"])
            app_conf["interface"] = interface_cfg

        return cls(**app_conf)


def _resolve_skill_paths(skill_paths: List[str], config_file: str) -> List[Path]:
    """
    Resolve skill discovery roots relative to the configuration file.

    Relative entries are interpreted relative to the directory containing the
    configuration file so every interface receives absolute paths.

    Args:
        skill_paths: Raw skill path strings from the configuration.
        config_file: Path to the configuration file they came from.

    Returns:
        Absolute, resolved paths to each configured skill discovery root.
    """
    config_dir = Path(config_file).resolve().parent
    resolved_paths = []

    for raw_path in skill_paths:
        skill_path = Path(raw_path)
        if not skill_path.is_absolute():
            skill_path = config_dir / skill_path
        resolved_paths.append(skill_path.resolve())

    return resolved_paths


def _get_skills_config_blocks(
    config_dict: Dict[str, Any],
) -> tuple[list[Any] | None, dict[str, Any] | None]:
    """
    Return skill discovery paths and runtime settings from nested config.

    Args:
        config_dict: Raw application configuration mapping.

    Returns:
        Tuple of the raw `skill_paths` list and `skill_runtime` mapping, each
        None when not configured.

    Raises:
        ValueError: If the legacy top-level keys are used, or `skills` is not
            an object.
    """
    if "skill_paths" in config_dict or "skill_runtime" in config_dict:
        raise ValueError(
            "Use 'skills.skill_paths' and 'skills.skill_runtime' for skills configuration"
        )

    skills_section = config_dict.get("skills")
    if skills_section is None:
        return None, None

    if not isinstance(skills_section, dict):
        raise ValueError("'skills' must be an object")

    return skills_section.get("skill_paths"), skills_section.get("skill_runtime")


def _get_a2a_config_blocks(
    config_dict: Dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """
    Return server-side and remote-agent A2A blocks from nested config.
    """
    if "a2a_self" in config_dict or "a2a_agents" in config_dict:
        raise ValueError("Use 'a2a.self' and 'a2a.agents' for A2A configuration")

    a2a_section = config_dict.get("a2a")
    if a2a_section is None:
        return None, None

    if not isinstance(a2a_section, dict):
        raise ValueError("'a2a' must be an object")

    return a2a_section.get("self"), a2a_section.get("agents")


def load_config_from_json(path: str) -> AppConfig:
    """
    Load application configuration from a JSON file.

    Args:
        path (str): Path to the JSON configuration file.

    Returns:
        AppConfig: The parsed application configuration object.
    """
    config_path = Path(path)
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = AppConfig.from_dict(
        config_dict,
        a2a_card_path_base=config_path.parent,
    )
    config.skill_paths = _resolve_skill_paths(config.skill_paths, path)

    return config
