# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Simple LangChain-backed A2A agent for MADA.

Run:
    python examples/a2a_table_mcp_server.py --port 9101
    python examples/a2a_langchain_agent.py --port 9001
    python examples/a2a_langchain_agent.py --port 9001 --model gpt-5

Install optional dependencies first:
    pip install langchain-openai fastapi uvicorn fastmcp

By default this example reads the same common environment variables used by
MADA example configs:
    MADA_MODEL or OPENAI_MODEL
    API_KEY or OPENAI_API_KEY
    API_BASE_URL or OPENAI_BASE_URL

MADA config:
    {
      "a2a_agents": {
        "LangChainAgent": {
          "url": "http://localhost:9001/",
          "description": "Simple LangChain model-backed remote agent"
        }
      }
    }

Smoke test prompt from MADA:
    Read the sample CSV table.
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
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - example dependency guard
    ChatOpenAI = None


DEFAULT_MODEL = "gpt-5"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MCP_URL = "http://localhost:9101/mcp"
DEFAULT_ROW_LIMIT = 4
DEFAULT_AGENT_CARD_PATH = (
    Path(__file__).parent / "agent_cards" / "langchain_agent_card.json"
)


def should_run_table_tool(task: str) -> bool:
    lowered = task.lower()
    return (
        "table" in lowered or "csv" in lowered or "read" in lowered or "rows" in lowered
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

    async def read_sample_table(self, row_limit: int = DEFAULT_ROW_LIMIT) -> str:
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
                    "read_sample_table",
                    arguments={"row_limit": row_limit},
                    timeout=30,
                )
            except Exception as exc:
                message = f"Table MCP tool call failed: {type(exc).__name__}: {exc}"
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
        self.mcp_tools = MCPExampleToolClient(mcp_url)
        self._llm = None

    @property
    def llm(self):
        if ChatOpenAI is None:
            raise RuntimeError(
                "This example requires langchain-openai. Install it with "
                "`pip install langchain-openai`."
            )
        if self._llm is None:
            kwargs = {"model": self.model}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._llm = ChatOpenAI(**kwargs)
        return self._llm

    async def run(self, task: str) -> str:
        if should_run_table_tool(task):
            return await self.mcp_tools.read_sample_table()

        response = await self.llm.ainvoke(
            [
                (
                    "system",
                    "You are a concise remote specialist called by MADA. "
                    "Complete the delegated task and return only the useful result.",
                ),
                ("human", task),
            ]
        )
        return str(getattr(response, "content", response))


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
        "--model",
        default=os.getenv("MADA_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL,
        help="Model to use. Defaults to MADA_MODEL, OPENAI_MODEL, then gpt-5.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY"),
        help="API key. Defaults to API_KEY or OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        default=(
            os.getenv("API_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or DEFAULT_BASE_URL
        ),
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--mcp-url",
        default=os.getenv("A2A_TABLE_MCP_URL") or DEFAULT_MCP_URL,
        help="Table-reader MCP server URL.",
    )
    parser.add_argument("--public-url", default=None)
    args = parser.parse_args()

    import uvicorn

    public_url = args.public_url or f"http://localhost:{args.port}"
    app = create_app(
        LangChainA2AAgent(args.model, args.api_key, args.base_url, args.mcp_url),
        public_url,
    )
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
