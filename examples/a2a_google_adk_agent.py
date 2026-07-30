# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Simple Google ADK-backed A2A agent for MADA.

Run:
    python examples/a2a_average_mcp_server.py --port 9102
    python examples/a2a_google_adk_agent.py --port 9002
    python examples/a2a_google_adk_agent.py --port 9002 --model gemini-2.5-pro

Install optional dependencies first:
    pip install google-adk fastapi uvicorn fastmcp

By default this example reads:
    MADA_MODEL or GOOGLE_MODEL

MADA config:
    {
      "a2a_agents": {
        "GoogleADKAgent": {
          "url": "http://localhost:9002/",
          "description": "Simple Google ADK remote agent"
        }
      }
    }

Smoke test prompt from MADA:
    What are the average values for the sample table columns?
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

try:
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
except ImportError:  # pragma: no cover - example dependency guard
    Agent = None
    Runner = None
    InMemorySessionService = None
    types = None


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_MCP_URL = "http://localhost:9102/mcp"
DEFAULT_AVERAGE_COLUMNS = "all"
DEFAULT_AGENT_CARD_PATH = (
    Path(__file__).parent / "agent_cards" / "google_adk_agent_card.json"
)
APP_NAME = "mada_google_adk_a2a_agent"


def should_run_average_tool(task: str) -> bool:
    lowered = task.lower()
    return (
        "average" in lowered
        or "mean" in lowered
        or "columns" in lowered
        or "column" in lowered
        or "numeric" in lowered
    )


def stringify_mcp_result(result: Any) -> str:
    if result is None:
        return ""
    content = getattr(result, "content", None)
    if content is not None:
        return stringify_mcp_result(content)
    if isinstance(result, list):
        parts = []
        for item in result:
            text = getattr(item, "text", None)
            parts.append(str(text if text is not None else item))
        return "\n".join(parts)
    return str(result)


class MCPExampleToolClient:
    def __init__(self, url: str) -> None:
        self.url = url

    async def calculate_column_averages(
        self, columns: str = DEFAULT_AVERAGE_COLUMNS
    ) -> str:
        try:
            from fastmcp import Client
        except ImportError as exc:  # pragma: no cover - example dependency guard
            raise RuntimeError(
                "This example requires fastmcp for MCP tool calls. Install it with "
                "`pip install fastmcp`."
            ) from exc

        async with Client(self.url) as client:
            try:
                result = await client.call_tool(
                    "calculate_column_averages",
                    arguments={"columns": columns},
                    timeout=30,
                )
            except Exception as exc:
                message = f"Average MCP tool call failed: {type(exc).__name__}: {exc}"
                return message
        text = stringify_mcp_result(result)
        return text


def extract_message_text(params: dict[str, Any]) -> str:
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


def build_task(task_id: str, context_id: str, text: str) -> dict[str, Any]:
    message_id = f"msg-{uuid.uuid4().hex}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    message = {
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
        "status": {"state": "completed", "timestamp": now, "message": message},
        "artifacts": [
            {
                "artifactId": f"artifact-{uuid.uuid4().hex}",
                "name": "response",
                "parts": [{"kind": "text", "text": text}],
            }
        ],
    }


def ids_from_params(params: dict[str, Any]) -> tuple[str, str]:
    message = params.get("message")
    task_id = params.get("id") or params.get("taskId")
    context_id = params.get("contextId")
    if isinstance(message, dict):
        task_id = task_id or message.get("taskId")
        context_id = context_id or message.get("contextId")
    return str(task_id or f"task-{uuid.uuid4().hex}"), str(
        context_id or f"context-{uuid.uuid4().hex}"
    )


class GoogleADKA2AAgent:
    def __init__(self, model: str, mcp_url: str = DEFAULT_MCP_URL) -> None:
        self.model = model
        self.mcp_tools = MCPExampleToolClient(mcp_url)
        self._session_service = None
        self._runner = None

    def _require_adk(self) -> None:
        if Agent is None or Runner is None or InMemorySessionService is None:
            raise RuntimeError(
                "This example requires Google ADK. Install it with "
                "`pip install google-adk`."
            )

    @property
    def runner(self):
        self._require_adk()
        if self._runner is None:
            agent = Agent(
                name="GoogleADKAgent",
                model=self.model,
                description="Simple Google ADK remote agent callable from MADA.",
                instruction=(
                    "You are a concise remote specialist called by MADA. "
                    "Complete the delegated task and return only the useful result."
                ),
            )
            self._session_service = InMemorySessionService()
            self._runner = Runner(
                agent=agent,
                app_name=APP_NAME,
                session_service=self._session_service,
            )
        return self._runner

    async def run(self, task: str) -> str:
        if should_run_average_tool(task):
            return await self.mcp_tools.calculate_column_averages()

        self._require_adk()
        runner = self.runner
        user_id = f"mada-user-{uuid.uuid4().hex}"
        session_id = f"mada-session-{uuid.uuid4().hex}"
        await self._session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )

        message = types.Content(
            role="user",
            parts=[types.Part(text=task)],
        )

        final_text = ""
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        ):
            if not event.is_final_response():
                continue
            if event.content and event.content.parts:
                final_text = "\n".join(
                    part.text
                    for part in event.content.parts
                    if getattr(part, "text", None)
                )

        return final_text


def create_app(agent: GoogleADKA2AAgent, public_url: str) -> FastAPI:
    app = FastAPI(title="Example Google ADK A2A Agent")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/.well-known/agent-card.json")
    async def agent_card() -> dict[str, Any]:
        card = json.loads(DEFAULT_AGENT_CARD_PATH.read_text(encoding="utf-8"))
        card["url"] = public_url
        return card

    @app.post("/")
    @app.post("/a2a")
    async def handle_rpc(body: dict[str, Any]):
        request_id = body.get("id")
        if body.get("method") != "message/send":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": "Only message/send is supported"},
            }

        params = body.get("params") or {}
        if not isinstance(params, dict):
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32602, "message": "'params' must be an object"},
            }

        task = extract_message_text(params).strip()
        if not task:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32602, "message": "Missing text message"},
            }

        try:
            text = await agent.run(task)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        task_id, context_id = ids_from_params(params)
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": build_task(task_id, context_id, text),
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple Google ADK A2A agent")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=9002, help="Port to bind")
    parser.add_argument(
        "--model",
        default=os.getenv("MADA_MODEL") or os.getenv("GOOGLE_MODEL") or DEFAULT_MODEL,
        help=(
            "Model to use. Defaults to MADA_MODEL, GOOGLE_MODEL, then gemini-2.5-flash."
        ),
    )
    parser.add_argument(
        "--mcp-url",
        default=os.getenv("A2A_AVERAGE_MCP_URL") or DEFAULT_MCP_URL,
        help="Column-average MCP server URL.",
    )
    parser.add_argument("--public-url", default=None)
    args = parser.parse_args()

    import uvicorn

    public_url = args.public_url or f"http://localhost:{args.port}"
    app = create_app(GoogleADKA2AAgent(args.model, args.mcp_url), public_url)
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
