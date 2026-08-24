# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Multi-agent Gradio interface using the improved interface architecture.

This provides a configurable web interface that supports MCP server connections
and multi-agent interactions with a better UI design.
"""

import asyncio
import logging
import sys
import os
import importlib.resources

import click
from pathlib import Path
import gradio as gr

from mada.core.config import AppConfig, OrchestrationConfig, load_config_from_json
from mada.interfaces.gradio.interface import MADAMultiAgentGradioInterface
from mada.interfaces.gradio.mcp_client_wrapper import MCPGradioClientSession

from mada.core.skill.skill_registry import SkillRegistry
from mada.core.skill.skill_runtime import SkillRuntime
from mada.core.skill.skill_tools import (
    build_load_skill_tool,
    build_read_skill_resource_tool,
    build_run_skill_script_tool,
)
from mada.interfaces.gradio.skill_approval import GradioPolicySkillScriptApprover


def _load_asset_text(filename: str) -> str:
    package = "mada.interfaces.gradio.assets"
    return (
        importlib.resources.files(package)
        .joinpath(filename)
        .read_text(encoding="utf-8")
    )


_GRADIO_CSS = _load_asset_text("gradio.css")
_GRADIO_JS = _load_asset_text("gradio.js")


def _get_orchestration_config(config: AppConfig) -> OrchestrationConfig:
    return getattr(config, "orchestration", None) or OrchestrationConfig()


def setup_logging():
    """Configure logging for the Gradio interface."""
    # Get log level from environment variable, default to INFO
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=True,  # Force reconfiguration
    )

    # Set specific loggers
    logging.getLogger("mada").setLevel(logging.INFO)
    logging.getLogger("mada-gradio").setLevel(logging.INFO)

    # Reduce noise from other libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("gradio").setLevel(logging.WARNING)

    print(f"Logging configured at {log_level} level")


def _resolve_skill_paths(skill_paths: list[str], config_file: str) -> list[Path]:
    """
    Resolve configured skill discovery roots relative to the config file path.
    """
    config_dir = Path(config_file).resolve().parent
    resolved_paths = []

    for raw_path in skill_paths:
        skill_path = Path(raw_path)
        if not skill_path.is_absolute():
            skill_path = config_dir / skill_path
        resolved_paths.append(skill_path.resolve())

    return resolved_paths


def _initialize_skill_state(config: AppConfig, config_file: str):
    """
    Discover manifest-based skills and build runtime tools for the UI.
    """
    resolved_skill_paths = _resolve_skill_paths(config.skill_paths, config_file)
    skill_registry = SkillRegistry.discover(resolved_skill_paths)
    skill_runtime = SkillRuntime(
        skill_registry,
        config=config.skill_runtime_config,
        script_approver=GradioPolicySkillScriptApprover(),
    )

    skill_tools = []
    if skill_registry.has_skills_for_tool("load_skill"):
        skill_tools.append(build_load_skill_tool(skill_runtime))
    if skill_registry.has_resources_for_tool("read_skill_resource"):
        skill_tools.append(build_read_skill_resource_tool(skill_runtime))
    if skill_registry.has_scripts_for_tool("run_skill_script"):
        skill_tools.append(build_run_skill_script_tool(skill_runtime))

    return skill_registry, skill_tools


def run_gradio(
    config: AppConfig,
    skill_registry: SkillRegistry = None,
    skill_tools: list = None,
):
    """
    Launch the Gradio web interface using the provided configuration.

    Args:
        config: The full application configuration object
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Initialize MCP client session with model and agents configuration
    client = MCPGradioClientSession(
        model_config=config.model,
        agents=config.agents,
        database_config=config.database,
        mcp_servers=config.mcp_servers,
        skill_registry=skill_registry,
        skill_tools=skill_tools,
        orchestration_config=_get_orchestration_config(config),
    )
    gradio_interface = MADAMultiAgentGradioInterface(
        config.interface, config.agents, client
    )
    interface = gradio_interface.create_interface()
    interface.queue(max_size=20)

    # Get port and share settings from interface config if available
    port = 7860
    share = False

    if config.interface:
        port = getattr(config.interface, "port", 7860)
        share = getattr(config.interface, "share", False)

    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        debug=True,
        css=_GRADIO_CSS,
        js=_GRADIO_JS,
    )


def create_gradio_app(config_path: str) -> gr.Blocks:
    """
    Create a Gradio application from a configuration file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Gradio Blocks interface
    """
    config = load_config_from_json(config_path)
    skill_registry, skill_tools = _initialize_skill_state(config, config_path)

    client = MCPGradioClientSession(
        model_config=config.model,
        agents=config.agents,
        database_config=config.database,
        mcp_servers=config.mcp_servers,
        skill_registry=skill_registry,
        skill_tools=skill_tools,
        orchestration_config=_get_orchestration_config(config),
    )
    gradio_interface = MADAMultiAgentGradioInterface(
        config.interface, config.agents, client
    )
    return gradio_interface.create_interface()


def gradio_entrypoint(port: int | None, share: bool, config_file: str):
    """
    Entrypoint for the Gradio mode in MADA.

    This function loads in a configuration from the user, overrides
    settings given via the CLI, and starts MADA using Gradio.

    Args:
        port: Port for Gradio server. Overrides 'interface.port' setting
            in the configuration file.
        share: If True, enable Gradio sharing. Overrides 'interface.share'
            setting in the configuration file.
        config_file: Path to MADA configuration file.
    """
    # Set up logging first
    setup_logging()

    try:
        print(f"Loading configuration from {config_file}")
        config = load_config_from_json(config_file)
        skill_registry, skill_tools = _initialize_skill_state(config, config_file)

        if not config.interface:
            print(
                "No Gradio interface settings provided. Make sure your configuration file has an 'interface' section."
            )
            sys.exit(2)

        if port is not None:
            config.interface.port = port
        if share:
            config.interface.share = True

        print(f"Launching on port {config.interface.port}")
        run_gradio(config, skill_registry=skill_registry, skill_tools=skill_tools)

    except Exception as e:
        print(f"Error launching Gradio interface: {e}")
        sys.exit(1)


@click.command(
    name="mada-gradio",
    context_settings={
        "help_option_names": ["-h", "--help"],
    },
)
@click.option(
    "-p",
    "--port",
    type=int,
    help="Port for Gradio server. Overrides 'interface.port' setting in the configuration file.",
)
@click.option(
    "-s",
    "--share",
    is_flag=True,
    help="Enable Gradio sharing. Overrides 'interface.share' setting in the configuration file.",
)
@click.argument(
    "config_file",
    type=str,
)
def main(port: int | None, share: bool, config_file: str) -> None:
    """
    Run MADA in Gradio mode.

    CONFIG_FILE is the path to the MADA configuration file.
    """
    gradio_entrypoint(port, share, config_file)


if __name__ == "__main__":
    main()
