# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Agent-to-Agent HTTP interface for MADA Orchestrator.

This module exposes the configured MADA planning agent as an A2A-compatible
JSON-RPC service. The MADA agent card is available under the standard
`/.well-known/agent-card.json` path.

This is the server-side A2A entry point: use it when another A2A client or
agent needs to discover MADA and send work to MADA. The client-side support
for MADA calling other A2A agents lives in `mada.core.a2a_client` and is wired
through the `a2a_agents` configuration block.
"""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

import click

try:
    from fastapi import FastAPI, Header, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
except (
    ImportError
) as exc:  # pragma: no cover - exercised only in missing dependency environments
    FastAPI = None
    Header = None
    HTTPException = None
    JSONResponse = None
    StreamingResponse = None
    FASTAPI_IMPORT_ERROR = exc
else:
    FASTAPI_IMPORT_ERROR = None

try:
    import uvicorn
except (
    ImportError
) as exc:  # pragma: no cover - exercised only in missing dependency environments
    uvicorn = None
    UVICORN_IMPORT_ERROR = exc
else:
    UVICORN_IMPORT_ERROR = None

from mada.core import load_config_from_json
from mada.core.config import A2AConfig, AppConfig, OrchestrationConfig

if TYPE_CHECKING:
    from mada.core.orchestrator import MADAOrchestrator


def _get_orchestration_config(config: AppConfig) -> OrchestrationConfig:
    return getattr(config, "orchestration", None) or OrchestrationConfig()


def _get_a2a_config(config: AppConfig) -> A2AConfig:
    return getattr(config, "a2a", None) or A2AConfig()


class A2AStartupError(RuntimeError):
    """Raised when the orchestrator cannot be initialized for A2A requests."""


def _format_startup_error_message(exc: BaseException) -> str:
    details = str(exc).strip() or exc.__class__.__name__
    lowered = details.lower()

    if "connect" in lowered or "connection" in lowered or "cancellederror" in lowered:
        return (
            "MADA could not connect to one or more MCP servers. "
            "Check the MCP server processes and the URLs/commands in your config. "
            f"Details: {details}"
        )

    return f"MADA failed to initialize the configured agent team. Details: {details}"


def _require_fastapi() -> None:
    if FASTAPI_IMPORT_ERROR is not None or uvicorn is None:
        missing = []
        if FASTAPI_IMPORT_ERROR is not None:
            missing.append("fastapi")
        if UVICORN_IMPORT_ERROR is not None:
            missing.append("uvicorn")
        packages = ", ".join(missing) or "fastapi, uvicorn"
        raise RuntimeError(
            f"A2A mode requires {packages}. Install the project dependencies again."
        )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return slug or "mada-agent"


class MADAA2AService:
    """
    Manage the shared orchestrator instance used by the A2A API.
    """

    def __init__(
        self,
        config: AppConfig,
        public_url: str,
        api_key: Optional[str] = None,
        bearer_token: Optional[str] = None,
    ) -> None:
        self.config = config
        self.a2a_config = _get_a2a_config(config)
        self.public_url = self.a2a_config.url or public_url
        self.api_key = api_key
        self.bearer_token = bearer_token
        self.orchestrator: Optional[MADAOrchestrator] = None
        self._startup_lock = asyncio.Lock()

    async def startup(self) -> None:
        if self.orchestrator is not None:
            return

        from mada.core.orchestrator import MADAOrchestrator

        orchestrator = None
        try:
            orchestrator = MADAOrchestrator(
                model_config=self.config.model,
                database_config=self.config.database,
                orchestration_config=_get_orchestration_config(self.config),
                bearer_token=self.bearer_token,
            )
            await orchestrator.__aenter__()
            await orchestrator.initialize_orchestrator(
                self.config.agents,
                self.config.mcp_servers,
                getattr(self.config, "a2a_agents", {}),
            )
            self.orchestrator = orchestrator
        except BaseException as exc:
            if orchestrator is not None:
                await orchestrator.__aexit__(None, None, None)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise A2AStartupError(_format_startup_error_message(exc)) from exc

    async def ensure_started(self) -> None:
        if self.orchestrator is not None:
            return

        async with self._startup_lock:
            if self.orchestrator is None:
                await self.startup()

    async def shutdown(self) -> None:
        if self.orchestrator is None:
            return
        await self.orchestrator.__aexit__(None, None, None)
        self.orchestrator = None

    def validate_api_key(
        self, authorization: Optional[str], x_api_key: Optional[str]
    ) -> None:
        if not self.api_key:
            return

        provided_key = x_api_key
        if authorization and authorization.lower().startswith("bearer "):
            provided_key = authorization[7:].strip()

        if not secrets.compare_digest(provided_key or "", self.api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")

    def build_agent_card(self) -> Dict[str, Any]:
        if self.a2a_config.card_path:
            card = self._load_agent_card_file()
            card["url"] = self.public_url
            card.setdefault("protocolVersion", "0.3.0")
            card.setdefault("capabilities", {"streaming": True})
            card.setdefault("defaultInputModes", ["text/plain"])
            card.setdefault("defaultOutputModes", ["text/plain"])
            card["supportsAuthenticatedExtendedCard"] = bool(self.api_key)
            return card

        return {
            "protocolVersion": "0.3.0",
            "name": self.a2a_config.name,
            "description": self.a2a_config.description,
            "url": self.public_url,
            "version": self.a2a_config.version,
            "capabilities": {"streaming": True},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "skills": self._build_skills(),
            "supportsAuthenticatedExtendedCard": bool(self.api_key),
        }

    def _load_agent_card_file(self) -> Dict[str, Any]:
        card_path = Path(self.a2a_config.card_path)
        try:
            with card_path.open("r", encoding="utf-8") as card_file:
                card = json.load(card_file)
        except OSError as exc:
            raise RuntimeError(f"Could not read A2A agent card: {card_path}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"A2A agent card is not valid JSON: {card_path}"
            ) from exc

        if not isinstance(card, dict):
            raise RuntimeError(f"A2A agent card must be a JSON object: {card_path}")
        return card

    def _build_skills(self) -> list[dict[str, Any]]:
        if self.a2a_config.skills:
            return self.a2a_config.skills

        skills = []
        for agent in self.config.agents:
            if getattr(agent, "agent_name", "") == "PlanningAgent":
                continue
            name = getattr(agent, "agent_name", "") or "MADA Agent"
            description = getattr(agent, "description", "") or name
            skills.append(
                {
                    "id": _slugify(name),
                    "name": name,
                    "description": description,
                    "tags": [getattr(agent, "domain", "") or "mada"],
                }
            )

        if skills:
            return skills

        return [
            {
                "id": "mada-orchestration",
                "name": "MADA orchestration",
                "description": self.a2a_config.description,
                "tags": ["mada"],
            }
        ]

    async def collect_response(self, message: str) -> str:
        if self.orchestrator is None:
            raise RuntimeError("Orchestrator not initialized")
        return await self.orchestrator.collect_message_response(
            message,
            isolated_session=True,
        )

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        if self.orchestrator is None:
            raise RuntimeError("Orchestrator not initialized")

        async for chunk in self.orchestrator.process_message(
            message,
            isolated_session=True,
        ):
            yield chunk


def _json_rpc_error(
    request_id: Any,
    code: int,
    message: str,
    status_code: int = 200,
) -> JSONResponse:
    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        },
        status_code=status_code,
    )


def _extract_message_text(params: Dict[str, Any]) -> str:
    message = params.get("message", params)
    if isinstance(message, str):
        return message

    if not isinstance(message, dict):
        return ""

    parts = message.get("parts")
    if not isinstance(parts, list):
        return str(message.get("text", "") or "")

    text_parts = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("kind") == "text" or part.get("type") == "text":
            text = part.get("text")
            if text:
                text_parts.append(str(text))

    return "\n".join(text_parts)


def _build_task(
    task_id: str,
    context_id: str,
    text: str,
    state: str = "completed",
) -> Dict[str, Any]:
    message_id = f"msg-{uuid.uuid4().hex}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    response_message = {
        "kind": "message",
        "messageId": message_id,
        "role": "agent",
        "parts": [{"kind": "text", "text": text}],
        "taskId": task_id,
        "contextId": context_id,
    }
    return {
        "kind": "task",
        "id": task_id,
        "contextId": context_id,
        "status": {
            "state": state,
            "timestamp": now,
            "message": response_message,
        },
        "artifacts": [
            {
                "artifactId": f"artifact-{uuid.uuid4().hex}",
                "name": "response",
                "parts": [{"kind": "text", "text": text}],
            }
        ],
    }


def _ids_from_params(params: Dict[str, Any]) -> tuple[str, str]:
    message = params.get("message")
    task_id = params.get("id") or params.get("taskId")
    context_id = params.get("contextId")

    if isinstance(message, dict):
        task_id = task_id or message.get("taskId")
        context_id = context_id or message.get("contextId")

    return str(task_id or f"task-{uuid.uuid4().hex}"), str(
        context_id or f"context-{uuid.uuid4().hex}"
    )


def create_a2a_app(
    config: AppConfig,
    public_url: str,
    api_key: Optional[str] = None,
    bearer_token: Optional[str] = None,
) -> FastAPI:
    """
    Build and return a FastAPI app backed by the configured MADA orchestrator.
    """
    _require_fastapi()
    service = MADAA2AService(
        config=config,
        public_url=public_url,
        api_key=api_key,
        bearer_token=bearer_token,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.mada_a2a_service = service
        try:
            yield
        finally:
            await service.shutdown()

    app = FastAPI(title="MADA A2A API", lifespan=lifespan)

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {
            "status": "ok",
            "orchestrator_initialized": "true"
            if service.orchestrator is not None
            else "false",
        }

    async def get_agent_card() -> Dict[str, Any]:
        return service.build_agent_card()

    app.get("/.well-known/agent-card.json")(get_agent_card)
    app.get("/.well-known/agent.json")(get_agent_card)
    app.get("/agent-card.json")(get_agent_card)

    async def handle_rpc(
        body: Dict[str, Any],
        authorization: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None),
    ):
        service.validate_api_key(authorization, x_api_key)

        request_id = body.get("id")
        method = body.get("method")
        params = body.get("params") or {}
        if not isinstance(params, dict):
            return _json_rpc_error(request_id, -32602, "'params' must be an object")

        if method not in {"message/send", "message/stream"}:
            return _json_rpc_error(request_id, -32601, f"Unsupported method: {method}")

        message_text = _extract_message_text(params).strip()
        if not message_text:
            return _json_rpc_error(
                request_id,
                -32602,
                "A2A request must include a text message part",
            )

        try:
            await service.ensure_started()
        except A2AStartupError as exc:
            configured_servers = (
                ", ".join((service.config.mcp_servers or {}).keys()) or "none"
            )
            print(
                "No MCP servers connected; returning 503 for A2A request. "
                f"Configured MCP servers: {configured_servers}",
                file=sys.stderr,
                flush=True,
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        task_id, context_id = _ids_from_params(params)

        if method == "message/send":
            content = await service.collect_response(message_text)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": _build_task(task_id, context_id, content),
            }

        async def event_stream() -> AsyncGenerator[str, None]:
            collected = []
            async for chunk in service.stream_response(message_text):
                collected.append(chunk)
                task = _build_task(
                    task_id,
                    context_id,
                    "".join(collected),
                    state="working",
                )
                payload = {"jsonrpc": "2.0", "id": request_id, "result": task}
                yield f"data: {json.dumps(payload)}\n\n"

            final = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": _build_task(
                    task_id,
                    context_id,
                    "".join(collected),
                    state="completed",
                ),
            }
            yield f"data: {json.dumps(final)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    app.post("/")(handle_rpc)
    app.post("/a2a")(handle_rpc)

    return app


def run_a2a(
    config: AppConfig,
    host: str,
    port: int,
    public_url: Optional[str] = None,
    api_key: Optional[str] = None,
    bearer_token: Optional[str] = None,
) -> None:
    """
    Launch the A2A FastAPI server.
    """
    _require_fastapi()
    card_url = public_url or f"http://{host}:{port}"
    app = create_a2a_app(
        config=config,
        public_url=card_url,
        api_key=api_key,
        bearer_token=bearer_token,
    )
    uvicorn.run(app, host=host, port=port)


def a2a_entrypoint(
    host: str,
    port: int,
    public_url: Optional[str],
    api_key: Optional[str],
    bearer_token: Optional[str],
    config_file: str,
) -> None:
    """
    Load config and start the A2A API server.
    """
    try:
        print(f"Loading configuration from {config_file}")
        config = load_config_from_json(config_file)
        card_url = public_url or f"http://{host}:{port}"
        print(f"Serving A2A API on {card_url}")
        run_a2a(
            config=config,
            host=host,
            port=port,
            public_url=public_url,
            api_key=api_key,
            bearer_token=bearer_token,
        )
    except Exception as e:
        print(f"Error launching A2A interface: {e}")
        sys.exit(1)


@click.command(
    name="mada-a2a",
    context_settings={
        "help_option_names": ["-h", "--help"],
    },
)
@click.option(
    "--host",
    type=str,
    default="0.0.0.0",
    show_default=True,
    help="Host interface to bind.",
)
@click.option(
    "-p",
    "--port",
    type=int,
    default=8000,
    show_default=True,
    help="Port for the A2A API.",
)
@click.option(
    "--public-url",
    type=str,
    default=None,
    help="Externally reachable URL to publish in the A2A agent card.",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    help="Optional API key that incoming requests must provide.",
)
@click.option(
    "--bearer-token",
    type=str,
    default=None,
    help="Optional bearer token forwarded to streamable HTTP MCP servers as X-Token.",
)
@click.argument("config_file", type=str)
def main(
    host: str,
    port: int,
    public_url: Optional[str],
    api_key: Optional[str],
    bearer_token: Optional[str],
    config_file: str,
) -> None:
    """
    Run MADA Orchestrator as an A2A agent.
    """
    a2a_entrypoint(host, port, public_url, api_key, bearer_token, config_file)


if __name__ == "__main__":
    main()
