# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Simple Google ADK-backed A2A agent for MADA."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from a2a_example_config import DEFAULT_CONFIG_PATH, load_model_settings
from fastapi import FastAPI, HTTPException
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams
from google.genai import types


DEFAULT_MCP_URL = "http://localhost:9102/mcp"
DEFAULT_AGENT_CARD_PATH = (
    Path(__file__).parent / "agent_cards" / "google_adk_agent_card.json"
)
APP_NAME = "mada_google_adk_a2a_agent"


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
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        mcp_url: str = DEFAULT_MCP_URL,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.mcp_url = mcp_url
        self._session_service = None
        self._runner = None

    @property
    def runner(self):
        if self._runner is None:
            agent = Agent(
                name="GoogleADKAgent",
                model=self._build_adk_model(),
                description="Simple Google ADK remote agent callable from MADA.",
                instruction=(
                    "You are a concise remote specialist called by MADA. "
                    "Complete the delegated task and return only the useful result. "
                    "Use your available MCP tools when they are relevant."
                ),
                tools=[
                    McpToolset(
                        connection_params=StreamableHTTPConnectionParams(
                            url=self.mcp_url,
                        )
                    )
                ],
            )
            self._session_service = InMemorySessionService()
            self._runner = Runner(
                agent=agent,
                app_name=APP_NAME,
                session_service=self._session_service,
            )
        return self._runner

    def _build_adk_model(self):
        provider = self.provider.lower()
        if provider in {"openai", "livai"}:
            if self.api_key:
                os.environ["OPENAI_API_KEY"] = self.api_key
            if self.base_url:
                os.environ["OPENAI_API_BASE"] = self.base_url
                os.environ["OPENAI_BASE_URL"] = self.base_url
            return LiteLlm(model=f"openai/{self.model}")

        return self.model

    async def run(self, task: str) -> str:
        return await self._run_adk_agent(task)

    async def _run_adk_agent(self, prompt: str) -> str:
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
            parts=[types.Part(text=prompt)],
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
        "--config",
        default=os.getenv("MADA_CONFIG") or str(DEFAULT_CONFIG_PATH),
        help="MADA config JSON to read default model settings from.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Provider override. Defaults to the MADA config provider.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model override. Defaults to the MADA config model.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key override. Defaults to the MADA config api_key.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL override for OpenAI-compatible ADK models.",
    )
    parser.add_argument(
        "--mcp-url",
        default=os.getenv("A2A_AVERAGE_MCP_URL") or DEFAULT_MCP_URL,
        help="Column-average MCP server URL.",
    )
    parser.add_argument("--public-url", default=None)
    args = parser.parse_args()

    import uvicorn

    model_settings = load_model_settings(args.config)
    provider = args.provider or model_settings.get("provider")
    model = args.model or model_settings.get("model")
    api_key = args.api_key or model_settings.get("api_key")
    base_url = args.base_url or model_settings.get("base_url")
    if not provider or not model:
        raise RuntimeError(
            "Google ADK A2A example requires provider and model from the MADA "
            "config or explicit --provider/--model overrides."
        )
    if provider.lower() in {"openai", "livai"} and (not api_key or not base_url):
        raise RuntimeError(
            "OpenAI-compatible ADK model providers require api_key and base_url "
            "from the MADA config or explicit --api-key/--base-url overrides."
        )

    public_url = args.public_url or f"http://localhost:{args.port}"
    app = create_app(
        GoogleADKA2AAgent(provider, model, api_key, base_url, args.mcp_url),
        public_url,
    )
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
