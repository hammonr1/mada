# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Magentic orchestration strategy implementation.
"""

import asyncio
import logging
import traceback
from collections.abc import AsyncIterable
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Tuple

from agent_framework import Agent, Message

from mada.core.config import AgentConfig, MCPServerConfig
from mada.core.orchestration.agent_as_tool_strategy import (
    AgentAsToolOrchestrationStrategy,
)

if TYPE_CHECKING:
    from mada.core.orchestrator import MADAOrchestrator

try:
    from agent_framework import MagenticBuilder
except ImportError:  # pragma: no cover - depends on installed agent framework version
    try:
        from agent_framework.orchestrations import (  # type: ignore[attr-defined]
            MagenticBuilder,
        )
    except ImportError:  # pragma: no cover
        MagenticBuilder = None


LOG = logging.getLogger(__name__)


class _InternalToolCallSignal(str):
    """
    Empty stream chunk that carries background-detach metadata.
    """

    def __new__(cls, tool_call_name: str):
        value = str.__new__(cls, "")
        value._mada_tool_call_name = tool_call_name
        return value


class MagenticOrchestrationStrategy(AgentAsToolOrchestrationStrategy):
    """
    Peer specialist group chat coordinated by a hidden manager agent.
    """

    mode = "magentic"

    def _create_manager_agent(
        self,
        orchestrator: "MADAOrchestrator",
        agent_configs: List[AgentConfig],
        participant_configs: List[AgentConfig],
    ) -> Agent:
        """
        Create the hidden manager agent used by Magentic orchestration.
        """
        team_description = orchestrator._generate_team_description(participant_configs)
        planning_cfg = orchestrator._get_planning_agent_config(agent_configs)

        if planning_cfg and planning_cfg.mcp_servers:
            LOG.warning(
                "PlanningAgent MCP server support is not implemented. "
                "MCP servers listed in PlanningAgent config will be ignored."
            )

        if planning_cfg and planning_cfg.instructions:
            base_instructions = planning_cfg.instructions.strip()
        else:
            base_instructions = """You are the hidden manager for MADA's Magentic orchestration mode.

Coordinate the specialist agents as peers, track plan and progress internally,
and produce the final response for the user."""

        instructions = f"""{base_instructions}

Specialist participants:
{team_description}

Guidelines:
- Coordinate the specialists as a peer conversation
- Re-plan when the current approach stalls or conflicts
- Keep internal planning and progress chatter out of the final user-facing answer
- Produce the final synthesized assistant response for the user
"""

        agent_kwargs = {}
        if planning_cfg:
            agent_kwargs.update(planning_cfg.extra)

        agent_name = planning_cfg.agent_name if planning_cfg else "PlanningAgent"
        return orchestrator.model_client.as_agent(
            name=agent_name,
            instructions=instructions,
            **agent_kwargs,
        )

    def _create_builder(self, orchestrator: "MADAOrchestrator"):
        """
        Create a fresh Magentic builder for a request.
        """
        if MagenticBuilder is None:
            raise RuntimeError(
                "Magentic orchestration requires agent_framework MagenticBuilder support"
            )

        return MagenticBuilder(
            participants=orchestrator.specialist_agents,
            manager_agent=orchestrator.manager_agent,
        )

    @staticmethod
    def _set_agent_metadata(agent: Agent, attribute: str, value: str) -> None:
        """
        Best-effort assignment for Agent Framework metadata attributes.
        """
        try:
            setattr(agent, attribute, value)
        except (AttributeError, TypeError):
            try:
                object.__setattr__(agent, attribute, value)
            except (AttributeError, TypeError):
                LOG.warning(
                    "Unable to set Magentic participant %s on agent %s",
                    attribute,
                    getattr(agent, "name", "<unknown>"),
                )

    def _preserve_participant_metadata(
        self,
        orchestrator: "MADAOrchestrator",
        participant_configs: List[AgentConfig],
    ) -> None:
        """
        Preserve configured participant IDs and descriptions for Magentic routing.
        """
        config_by_name = {config.agent_name: config for config in participant_configs}
        for agent in orchestrator.specialist_agents:
            config = config_by_name.get(getattr(agent, "name", ""))
            if not config:
                continue
            self._set_agent_metadata(agent, "id", config.agent_name)
            self._set_agent_metadata(agent, "description", config.description)

    def _build_runtime(self, orchestrator: "MADAOrchestrator"):
        """
        Build a runnable Magentic workflow instance.
        """
        builder = self._create_builder(orchestrator)

        for method_name in ("build", "create_workflow", "create"):
            method = getattr(builder, method_name, None)
            if callable(method):
                return method()

        return builder

    _IGNORED_EVENT_TYPES = frozenset(
        {
            "plan",
            "progress",
            "replan",
            "checkpoint",
            "function_call",
            "tool_call",
            "function_result",
            "tool_result",
        }
    )

    _TEXT_KEYS = (
        "final_output",
        "final_response",
        "assistant_response",
        "response",
        "output",
        "content",
        "text",
        "contents",
        "messages",
        "data",
    )

    def _extract_text(self, payload: Any) -> str:
        """
        Best-effort extraction of a final assistant reply from Magentic results.
        """
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        if isinstance(payload, (list, tuple)):
            return "".join(
                text for item in payload if (text := self._extract_text(item))
            )

        # Get event type (works for dict or object)
        event_type = ""
        if isinstance(payload, dict):
            event_type = str(payload.get("type") or payload.get("event") or "").lower()
        else:
            event_type = str(
                getattr(payload, "type", "") or getattr(payload, "event", "")
            ).lower()

        if event_type in self._IGNORED_EVENT_TYPES:
            return ""

        # Extract text from known keys/attributes
        for key in self._TEXT_KEYS:
            value = (
                payload.get(key)
                if isinstance(payload, dict)
                else getattr(payload, key, None)
            )
            if isinstance(value, str) and value.strip():
                return value
            if value is not None:
                text = self._extract_text(value)
                if text.strip():
                    return text

        # Fallback: try to_dict() for objects
        if hasattr(payload, "to_dict"):
            try:
                return self._extract_text(payload.to_dict())
            except (TypeError, ValueError):
                pass

        return ""

    @staticmethod
    def _event_type(payload: Any) -> str:
        """
        Return a normalized Magentic event type when one is available.
        """
        if isinstance(payload, dict):
            return str(payload.get("type") or payload.get("event") or "").lower()
        return str(
            getattr(payload, "type", "") or getattr(payload, "event", "")
        ).lower()

    def _is_terminal_output_event(self, event: Any) -> bool:
        """
        Return whether an event should be exposed as the user-facing answer.
        """
        event_type = self._event_type(event)
        if event_type in {
            "final",
            "final_output",
            "final_response",
            "output",
            "result",
        }:
            return True

        if event_type:
            return False

        if isinstance(event, str):
            return bool(event.strip())

        for key in (
            "final_output",
            "final_response",
            "assistant_response",
            "role",
            "content",
            "contents",
            "messages",
            "text",
        ):
            if isinstance(event, dict) and key in event:
                return True
            if hasattr(event, key):
                return True

        if hasattr(event, "to_dict"):
            try:
                return self._is_terminal_output_event(event.to_dict())
            except (TypeError, ValueError):
                return False

        return False

    def _is_final_result_event(self, event: Any) -> bool:
        """
        Return whether an event is an aggregate final result rather than a delta.
        """
        event_type = self._event_type(event)
        if event_type in {"final", "final_output", "final_response", "result"}:
            return True
        if event_type:
            return False
        return self._is_terminal_output_event(event)

    @staticmethod
    def _transcript_to_messages(
        transcript_messages: List[Dict[str, Any]],
    ) -> List[Message]:
        """
        Convert normalized transcript messages to Agent Framework messages.
        """
        messages = []
        for message in transcript_messages:
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            role = str(message.get("role") or "user").strip().lower() or "user"
            messages.append(Message(role=role, contents=[content]))
        return messages

    @staticmethod
    def _structured_history_unsupported(error: Exception) -> bool:
        """
        Return whether the installed Magentic runtime rejected multi-message input.
        """
        error_message = str(error)
        return (
            "Magentic only support a single task message" in error_message
            or "Magentic only supports a single task message" in error_message
        )

    @staticmethod
    def _conversation_history_messages(
        history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Remove interface background-task bookkeeping from persisted chat history.
        """
        messages = []
        for message in history:
            role = str(message.get("role") or "").strip().lower()
            content = str(message.get("content") or "").strip()
            header = content.split("\n", 1)[0]
            is_background_status = role == "assistant" and (
                (content.startswith("[task-") and "Started in background." in content)
                or (
                    content.startswith("[")
                    and "] Background tool `" in header
                    and header.endswith(":")
                )
            )
            if not is_background_status:
                messages.append(message)
        return messages

    @staticmethod
    def _call_notices_from_event(
        event: Any,
        tool_calls: List[Any],
    ) -> List[str]:
        """
        Return invisible handoff signals for Magentic group-chat events.
        """
        event_type = str(
            event.get("type") if isinstance(event, dict) else getattr(event, "type", "")
        ).lower()
        if event_type != "group_chat":
            return []

        data = (
            event.get("data")
            if isinstance(event, dict)
            else getattr(event, "data", None)
        )
        if data is None:
            return []

        data_type = data.get("type") if isinstance(data, dict) else type(data).__name__
        if data_type != "GroupChatRequestSentEvent":
            return []

        participant_name = (
            data.get("participant_name")
            if isinstance(data, dict)
            else getattr(data, "participant_name", None)
        )
        if not participant_name:
            return []

        round_index = (
            data.get("round_index")
            if isinstance(data, dict)
            else getattr(data, "round_index", None)
        )
        call_key = (round_index, participant_name)
        if call_key in tool_calls:
            return []

        tool_calls.append(call_key)
        return [_InternalToolCallSignal(participant_name)]

    async def _iter_result_events(
        self,
        result: Any,
    ) -> AsyncGenerator[Any, None]:
        """
        Iterate over Magentic workflow events or result payloads.
        """
        if asyncio.iscoroutine(result):
            result = await result

        if isinstance(result, AsyncIterable):
            async for event in result:
                yield event
            return

        if hasattr(result, "__iter__") and not isinstance(result, (str, dict)):
            for event in result:
                yield event
            return

        yield result

    def _start_runtime(
        self,
        runtime: Any,
        message_payload: Any,
    ) -> Any:
        """
        Start a Magentic runtime with the best supported streaming API.
        """
        run = getattr(runtime, "run", None)
        if callable(run):
            try:
                return run(message_payload, stream=True)
            except TypeError as e:
                if "unexpected keyword argument 'stream'" not in str(e):
                    raise
                return run(message_payload)

        for method_name in ("run_stream", "stream", "invoke"):
            method = getattr(runtime, method_name, None)
            if callable(method):
                return method(message_payload)

        raise RuntimeError(
            "Unable to execute Magentic workflow with the installed builder."
        )

    async def _iter_workflow_events(
        self,
        orchestrator: "MADAOrchestrator",
        transcript_messages: List[Dict[str, Any]],
    ) -> AsyncGenerator[Any, None]:
        """
        Run a Magentic workflow and yield its events.
        """
        if not orchestrator.manager_agent:
            raise RuntimeError("Magentic manager is not initialized.")

        structured_messages = self._transcript_to_messages(transcript_messages)
        if not structured_messages:
            structured_messages = [
                Message(role="user", contents=["Please introduce yourself."])
            ]

        try:
            runtime = self._build_runtime(orchestrator)
            result = self._start_runtime(runtime, structured_messages)
            async for event in self._iter_result_events(result):
                yield event
            return
        except (TypeError, ValueError) as e:
            if not (
                len(structured_messages) > 1 and self._structured_history_unsupported(e)
            ):
                raise

        runtime = self._build_runtime(orchestrator)
        fallback_message = Message(
            role="user",
            contents=[orchestrator.build_prompt_from_transcript(transcript_messages)],
        )
        result = self._start_runtime(
            runtime,
            [fallback_message],
        )
        async for event in self._iter_result_events(result):
            yield event

    async def _stream_workflow_response(
        self,
        orchestrator: "MADAOrchestrator",
        transcript_messages: List[Dict[str, Any]],
        *,
        include_tool_notices: bool,
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """
        Stream Magentic notices and return the final assistant reply as an event.
        """
        streamed_text_parts = []
        final_text = ""
        tool_calls = []
        async for event in self._iter_workflow_events(
            orchestrator, transcript_messages
        ):
            if include_tool_notices:
                for notice in self._call_notices_from_event(
                    event,
                    tool_calls,
                ):
                    yield "notice", notice

            if not self._is_terminal_output_event(event):
                continue

            event_text = self._extract_text(event)
            if not event_text:
                continue

            if self._is_final_result_event(event):
                final_text = event_text
                if not streamed_text_parts:
                    yield "chunk", event_text
                continue

            if event_text:
                streamed_text_parts.append(event_text)
                yield "chunk", event_text

        yield "final", final_text or "".join(streamed_text_parts)

    async def _run_workflow(
        self,
        orchestrator: "MADAOrchestrator",
        transcript_messages: List[Dict[str, Any]],
    ) -> str:
        """
        Run a fresh Magentic workflow from a rebuilt transcript.
        """
        final_text = ""
        async for kind, value in self._stream_workflow_response(
            orchestrator,
            transcript_messages,
            include_tool_notices=False,
        ):
            if kind == "final":
                final_text = value
        return final_text

    async def initialize(
        self,
        orchestrator: "MADAOrchestrator",
        agent_configs: List[AgentConfig],
        mcp_servers: Dict[str, MCPServerConfig] | None = None,
    ) -> Tuple[str, List[str]]:
        """
        Initialize the Magentic orchestration flow end to end.
        """
        if MagenticBuilder is None:
            raise RuntimeError(
                "Magentic orchestration requires agent_framework MagenticBuilder support"
            )

        orchestrator.specialist_agents = []
        orchestrator._mcp_tool_count = 0
        orchestrator._agent_descriptions = {}
        participant_configs = orchestrator.resolve_participant_configs(agent_configs)
        orchestrator.mcp_servers = mcp_servers or {}

        all_tools, failed_servers, failed_agents = await self._initialize_participants(
            orchestrator, participant_configs
        )
        active_participant_configs = self._resolve_active_participant_configs(
            orchestrator, participant_configs
        )
        self._preserve_participant_metadata(
            orchestrator,
            active_participant_configs,
        )

        if not active_participant_configs:
            raise RuntimeError(
                "Magentic orchestration requires at least one active specialist agent."
            )

        orchestrator.planning_agent = None
        orchestrator.session = None
        orchestrator.manager_agent = self._create_manager_agent(
            orchestrator,
            agent_configs=agent_configs,
            participant_configs=active_participant_configs,
        )

        status = self._build_status(orchestrator, failed_servers, failed_agents)
        LOG.info(status)

        return status, all_tools

    async def process_openai_messages(
        self,
        orchestrator: "MADAOrchestrator",
        messages: List[Dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """
        Process OpenAI-style chat messages through a fresh Magentic workflow.
        """
        if not orchestrator.manager_agent:
            yield "Error: Orchestrator not initialized."
            return

        transcript_messages = orchestrator._normalize_transcript_messages(messages)
        try:
            streamed = False
            final_text = ""
            async for kind, value in self._stream_workflow_response(
                orchestrator,
                transcript_messages,
                include_tool_notices=False,
            ):
                if kind == "chunk":
                    streamed = True
                    yield value
                elif kind == "final":
                    final_text = value
            if not streamed and final_text:
                yield final_text
            elif not final_text:
                LOG.warning("No final assistant text received from Magentic workflow")
        except Exception as e:
            error_msg = f"Error processing message: {e}"
            LOG.error(error_msg)
            traceback.print_exc()
            yield error_msg

    async def process_message(
        self,
        orchestrator: "MADAOrchestrator",
        message: str,
        isolated_session: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Process a user message through a fresh Magentic workflow.
        """
        if not orchestrator.manager_agent:
            yield "Error: Orchestrator not initialized. Call initialize_orchestrator() first."
            return

        try:
            streamed_response = False
            if isolated_session:
                transcript_messages = orchestrator._normalize_transcript_messages(
                    [{"role": "user", "content": message}]
                )
                aggregated_assistant_reply = ""
                async for kind, value in self._stream_workflow_response(
                    orchestrator,
                    transcript_messages,
                    include_tool_notices=True,
                ):
                    if kind == "notice":
                        yield value
                    elif kind == "chunk":
                        streamed_response = True
                        yield value
                    elif kind == "final":
                        aggregated_assistant_reply = value
                orchestrator._persist_isolated_response(
                    message,
                    aggregated_assistant_reply,
                )
            else:
                async with orchestrator._session_lock:
                    history = orchestrator.session_manager.load_history()
                    history = self._conversation_history_messages(history)
                    transcript_messages = orchestrator._normalize_transcript_messages(
                        [*history, {"role": "user", "content": message}]
                    )
                    aggregated_assistant_reply = ""
                    async for kind, value in self._stream_workflow_response(
                        orchestrator,
                        transcript_messages,
                        include_tool_notices=True,
                    ):
                        if kind == "notice":
                            yield value
                        elif kind == "chunk":
                            streamed_response = True
                            yield value
                        elif kind == "final":
                            aggregated_assistant_reply = value
                    orchestrator._persist_completed_turn(
                        {
                            "message": message,
                            "assistant_reply": aggregated_assistant_reply,
                        }
                    )

            if not aggregated_assistant_reply.strip():
                LOG.warning("No final assistant text received from Magentic workflow")
            elif not streamed_response:
                yield aggregated_assistant_reply
        except Exception as e:
            error_msg = f"Error processing message: {e}"
            LOG.error(error_msg)
            traceback.print_exc()
            yield error_msg
