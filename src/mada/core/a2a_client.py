# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Small A2A JSON-RPC client used by the MADA orchestrator.

This is the client-side A2A helper: the orchestrator uses it to call remote
A2A agents configured under `a2a.agents`. The server-side interface that
exposes MADA itself as an A2A agent lives in `mada.interfaces.a2a.main` and
uses the `a2a.self` configuration block.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx

from mada.core.config import RemoteA2AAgentConfig


class RemoteA2AClient:
    """
    Minimal client for delegating a text task to a remote A2A agent.
    """

    def __init__(self, name: str, config: RemoteA2AAgentConfig) -> None:
        """
        Initialize an HTTP client for a configured remote A2A agent.
        """
        self.name = name
        self.config = config
        headers = dict(config.headers)
        if config.api_key:
            headers["x-api-key"] = config.api_key
        self._client = httpx.AsyncClient(headers=headers, timeout=config.timeout)

    async def send_message(self, task: str) -> str:
        """
        Send a text task to the remote A2A agent and return its text response.
        """
        request_id = f"mada-{uuid.uuid4().hex}"
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "message/send",
            "params": {
                "message": {
                    "kind": "message",
                    "messageId": f"msg-{uuid.uuid4().hex}",
                    "role": "user",
                    "parts": [{"kind": "text", "text": task}],
                }
            },
        }

        response = await self._client.post(self.config.url, json=payload)
        response.raise_for_status()
        data = response.json()

        if data.get("error"):
            error = data["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise RuntimeError(f"A2A agent {self.name} returned an error: {message}")

        return self._extract_text(data.get("result"))

    async def get_agent_card(self) -> dict[str, Any]:
        """
        Fetch the remote agent card when the A2A server exposes one.
        """
        if self.config.card_url:
            try:
                response = await self._client.get(self.config.card_url, timeout=5.0)
                response.raise_for_status()
                data = response.json()
            except Exception:
                return {}
            return data if isinstance(data, dict) else {}

        base_url = self.config.url.rstrip("/")
        if base_url.endswith("/a2a"):
            base_url = base_url[: -len("/a2a")]
        for path in (
            "/.well-known/agent-card.json",
            "/.well-known/agent.json",
            "/agent-card.json",
        ):
            try:
                response = await self._client.get(f"{base_url}{path}", timeout=5.0)
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                data = response.json()
            except Exception:
                continue
            if isinstance(data, dict):
                return data
        return {}

    async def aclose(self) -> None:
        """
        Close the underlying async HTTP client.
        """
        await self._client.aclose()

    def _extract_text(self, result: Any) -> str:
        """
        Extract human-readable text from an A2A JSON-RPC result payload.
        """
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if not isinstance(result, dict):
            return str(result)

        texts = []
        self._collect_text_parts(result, texts)
        if texts:
            return "\n".join(texts)
        return str(result)

    def _collect_text_parts(self, value: Any, texts: list[str]) -> None:
        """
        Recursively collect text parts from an A2A result structure.
        """
        if isinstance(value, dict):
            parts = value.get("parts")
            if isinstance(parts, list):
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    if part.get("kind") == "text" or part.get("type") == "text":
                        text = part.get("text")
                        if text:
                            texts.append(str(text))
            for item in value.values():
                self._collect_text_parts(item, texts)
        elif isinstance(value, list):
            for item in value:
                self._collect_text_parts(item, texts)
