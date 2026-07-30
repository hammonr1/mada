# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Simple LangChain-backed A2A agent for MADA."""

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
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI


DEFAULT_MCP_URL = "http://localhost:9101/mcp"
DEFAULT_AGENT_CARD_PATH = (
    Path(__file__).parent / "agent_cards" / "langchain_agent_card.json"
)


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


class LangChainA2AAgent:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        mcp_url: str = DEFAULT_MCP_URL,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.mcp_url = mcp_url
        self._llm = None
        self._tools = None

    @property
    def llm(self):
        if self._llm is None:
            kwargs = {"model": self.model}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._llm = ChatOpenAI(**kwargs)
        return self._llm

    async def run(self, task: str) -> str:
        return await self._run_langchain_agent(task)

    async def _run_langchain_agent(self, prompt: str) -> str:
        tools = await self._get_tools()
        tools_by_name = {tool.name: tool for tool in tools}
        messages = [
            (
                "system",
                "You are a concise remote specialist called by MADA. "
                "Complete the delegated task and return only the useful result. "
                "Use your available MCP tools when they are relevant.",
            ),
            ("human", prompt),
        ]

        response = await self.llm.bind_tools(tools).ainvoke(messages)
        tool_calls = getattr(response, "tool_calls", []) or []
        if not tool_calls:
            return str(getattr(response, "content", response))

        messages.append(response)
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.get("name"))
            if tool is None:
                continue
            result = await tool.ainvoke(tool_call.get("args") or {})
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call.get("id") or f"tool-{uuid.uuid4().hex}",
                )
            )

        response = await self.llm.ainvoke(messages)
        return str(getattr(response, "content", response))

    async def _get_tools(self) -> list[Any]:
        if self._tools is None:
            client = MultiServerMCPClient(
                {
                    "example": {
                        "transport": "streamable_http",
                        "url": self.mcp_url,
                    }
                }
            )
            self._tools = await client.get_tools()
        return self._tools


def create_app(agent: LangChainA2AAgent, public_url: str) -> FastAPI:
    app = FastAPI(title="Example LangChain A2A Agent")

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
    parser = argparse.ArgumentParser(description="Run a simple LangChain A2A agent")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=9001, help="Port to bind")
    parser.add_argument(
        "--config",
        default=os.getenv("MADA_CONFIG") or str(DEFAULT_CONFIG_PATH),
        help="MADA config JSON to read default model settings from.",
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
        help="OpenAI-compatible base URL override. Defaults to the MADA config base_url.",
    )
    parser.add_argument(
        "--mcp-url",
        default=os.getenv("A2A_TABLE_MCP_URL") or DEFAULT_MCP_URL,
        help="Table-reader MCP server URL.",
    )
    parser.add_argument("--public-url", default=None)
    args = parser.parse_args()

    import uvicorn

    model_settings = load_model_settings(args.config)
    model = args.model or model_settings.get("model")
    api_key = args.api_key or model_settings.get("api_key")
    base_url = args.base_url or model_settings.get("base_url")
    if not model or not api_key or not base_url:
        raise RuntimeError(
            "LangChain A2A example requires model, api_key, and base_url from "
            "the MADA config or explicit --model/--api-key/--base-url overrides."
        )

    public_url = args.public_url or f"http://localhost:{args.port}"
    app = create_app(
        LangChainA2AAgent(model, api_key, base_url, args.mcp_url),
        public_url,
    )
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
