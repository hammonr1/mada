# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Magentic orchestration strategy implementation.
"""

import asyncio
import json
import logging
import re
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

    _TOOL_CALL_KEYS = (
        "function_call",
        "tool_call",
        "function_calls",
        "tool_calls",
    )

    _TOOL_COLLECTION_KEYS = ("tool_calls", "tools", "function_calls", "functions")

    _FINAL_EVENT_TYPES = frozenset(
        {"final", "final_output", "final_response", "result"}
    )

    @staticmethod
    def _payload_value(payload: Any, key: str) -> Any:
        if isinstance(payload, dict):
            return payload.get(key)
        return getattr(payload, key, None)

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

        event_type = self._event_type(payload)
        if event_type in self._IGNORED_EVENT_TYPES or event_type in (
            "tool_result",
            "function_result",
        ):
            return ""

        # Extract text from known keys/attributes
        for key in self._TEXT_KEYS:
            value = self._payload_value(payload, key)
            if isinstance(value, str) and value.strip():
                return value
            if value is not None:
                text = self._extract_text(value)
                if text.strip():
                    return text

        if self._contains_tool_call(payload):
            # This is a tool invocation, not user-facing text
            return ""

        # Fallback: try to_dict() for objects (but not if it contains tool calls)
        if hasattr(payload, "to_dict"):
            try:
                return self._extract_text(payload.to_dict())
            except (TypeError, ValueError):
                pass

        # No text found - don't fall back to arbitrary attribute scanning
        # as that returns metadata strings like "agent_response_update"
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
        Return whether an event should be exposed as output or contains data to preserve.
        """
        event_type = self._event_type(event)
        if event_type in {
            "final",
            "final_output",
            "final_response",
            "output",
            "result",
            "assistant_response",
            "response",
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

    _BACKGROUND_TASK_STATUS_PATTERN = re.compile(
        r"^\[(?:task-|[^\]]+)\]\s*(?:Started|Running|Waiting)", re.IGNORECASE
    )

    @classmethod
    def _conversation_history_messages(
        cls,
        history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Remove transient background-task status messages but retain completed results.
        """
        messages = []
        for message in history:
            role = str(message.get("role") or "").strip().lower()
            content = str(message.get("content") or "").strip()
            # Only filter transient status updates, not completion messages with results
            is_status_only = role == "assistant" and bool(
                cls._BACKGROUND_TASK_STATUS_PATTERN.match(content)
            )
            if not is_status_only:
                messages.append(message)
        return messages

    @classmethod
    def _contains_tool_call(cls, payload: Any) -> bool:
        """
        Return whether payload or its contents describe a tool/function call.
        """
        event_type = cls._event_type(payload)
        if event_type in ("function_call", "tool_call"):
            return True

        if any(
            cls._payload_value(payload, key) is not None for key in cls._TOOL_CALL_KEYS
        ):
            return True

        contents = cls._payload_value(payload, "contents")
        if isinstance(contents, (list, tuple)):
            return any(cls._contains_tool_call(item) for item in contents)

        return False

    @staticmethod
    def _tool_call_signals(
        participant_name: Any,
    ) -> List[str]:
        if not participant_name:
            return []

        return [_InternalToolCallSignal(str(participant_name))]

    @classmethod
    def _call_notices_from_event(
        cls,
        event: Any,
        seen_executor_ids: set,
    ) -> List[str]:
        """
        Return invisible handoff signals when real MCP tool calls occur.

        In blocking=False mode, BackgroundTaskManager should detach only after
        a real tool execution. Actual Magentic tool invocations are surfaced as
        streamed output events carrying AgentResponseUpdate with function_call,
        not just executor_invoked events.

        Uses seen_executor_ids to deduplicate signals from the same tool execution
        (executor_invoked + output + tool_result all carry the same executor_id).
        """
        event_type = cls._event_type(event)

        if event_type == "output":
            data = cls._payload_value(event, "data")
            executor_id = cls._payload_value(data, "executor_id") or cls._payload_value(
                data, "agent_id"
            )
            if (
                executor_id
                and executor_id not in seen_executor_ids
                and cls._contains_tool_call(data)
            ):
                seen_executor_ids.add(executor_id)
                return cls._tool_call_signals(executor_id)
            return []

        if event_type not in {"executor_invoked", "tool_result", "function_result"}:
            return []

        data = cls._payload_value(event, "data")
        executor_id = cls._payload_value(data, "executor_id")

        if event_type == "executor_invoked":
            has_tools = any(
                cls._payload_value(data, key) is not None
                for key in cls._TOOL_COLLECTION_KEYS
            )
            if not has_tools or not executor_id:
                return []

        if executor_id and executor_id not in seen_executor_ids:
            seen_executor_ids.add(executor_id)
            return cls._tool_call_signals(executor_id)
        return []

    @classmethod
    def _background_task_descriptors_from_event(cls, event: Any) -> List[str]:
        """
        Return JSON descriptors for server-side background MCP tasks.

        When a Magentic specialist starts a server-side background MCP task,
        Agent Framework surfaces the descriptor inside executor_completed.data
        (AgentExecutorResponse / AgentResponseUpdate with function_result contents),
        not just in top-level tool_result events.
        """
        event_type = cls._event_type(event)
        if event_type not in {"tool_result", "function_result", "executor_completed"}:
            return []

        candidates = [
            value
            for key in ("data", "result", "content", "output")
            if (value := cls._payload_value(event, key)) is not None
        ]

        descriptors = []
        for candidate in candidates or [event]:
            descriptors.extend(cls._background_task_descriptors_from_value(candidate))
        return list(dict.fromkeys(descriptors))

    @classmethod
    def _background_task_descriptors_from_value(cls, value: Any) -> List[str]:
        """
        Extract parseable running background-task descriptors from a nested payload.

        In real Magentic worker responses, background-task JSON lives under
        structured messages/contents/items paths. The function_result wrapper
        itself usually has empty .text, so we must traverse the structured
        response (AgentResponse/AgentResponseUpdate) to find the task_id.
        """
        if value is None:
            return []

        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")

        if isinstance(value, str):
            parsed_values = cls._parse_json_candidates(value)
            descriptors = []
            for parsed in parsed_values:
                descriptors.extend(cls._background_task_descriptors_from_value(parsed))
            return descriptors

        if isinstance(value, (list, tuple)):
            descriptors = []
            for item in value:
                descriptors.extend(cls._background_task_descriptors_from_value(item))
            return descriptors

        if isinstance(value, dict):
            descriptor = cls._background_task_descriptor_json(value)
            if descriptor:
                return [descriptor]

            descriptors = []
            # Traverse structured response paths that contain background task descriptors
            for key in (
                "data",
                "result",
                "content",
                "output",
                "text",
                "message",
                "messages",
                "contents",
                "items",
                "function_result",
                "tool_result",
            ):
                if key in value:
                    descriptors.extend(
                        cls._background_task_descriptors_from_value(value[key])
                    )
            return descriptors

        # Try to extract from object attributes
        if hasattr(value, "to_dict"):
            try:
                return cls._background_task_descriptors_from_value(value.to_dict())
            except (TypeError, ValueError):
                pass

        # Check common attribute names on structured objects
        for attr in ("text", "content", "contents", "messages", "items", "data"):
            if hasattr(value, attr):
                attr_value = getattr(value, attr, None)
                if attr_value is not None:
                    descriptors = cls._background_task_descriptors_from_value(
                        attr_value
                    )
                    if descriptors:
                        return descriptors

        return []

    @staticmethod
    def _parse_json_candidates(value: str) -> List[Any]:
        """
        Parse a JSON value from a full string or from the widest JSON object within it.
        """
        value = value.strip()
        if not value:
            return []

        # Try full string first
        try:
            return [json.loads(value)]
        except json.JSONDecodeError:
            pass

        # Try extracting widest {...} substring
        start = value.find("{")
        end = value.rfind("}")
        if start != -1 and end > start:
            try:
                return [json.loads(value[start : end + 1])]
            except json.JSONDecodeError:
                pass

        return []

    @staticmethod
    def _background_task_descriptor_json(value: Dict[str, Any]) -> str:
        """
        Return a canonical JSON descriptor if the payload starts a background task.
        """
        task_id = value.get("task_id")
        if not task_id:
            return ""

        status = str(value.get("status") or "running").strip().lower()
        if status != "running":
            return ""

        descriptor = {
            "task_id": task_id,
            "status": status,
            "tool_name": value.get("tool_name", "background_tool"),
        }
        return json.dumps(descriptor, default=str)

    @staticmethod
    def _background_task_ack(descriptors: List[str]) -> str:
        """
        Build a concise user-facing acknowledgement when no final text is available.

        Format matches _BACKGROUND_TASK_STATUS_PATTERN so history filtering removes it.
        """
        if not descriptors:
            return ""

        try:
            descriptor = json.loads(descriptors[0])
        except json.JSONDecodeError:
            return "[background-task] Started"

        task_id = descriptor.get("task_id")
        tool_name = descriptor.get("tool_name", "background_tool")
        if task_id:
            return f"[{task_id}] Started background tool `{tool_name}`"
        return f"[background-task] Started tool `{tool_name}`"

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

        Magentic uses messages[0] as the task to plan against, so we must pass
        a single user message containing the latest request, not the full transcript.
        """
        if not orchestrator.manager_agent:
            raise RuntimeError("Magentic manager is not initialized.")

        # Build a single prompt from the full transcript for Magentic planning
        # Magentic uses messages[0] as the task, so passing multi-message history
        # would cause it to plan around the oldest message instead of latest request
        if not transcript_messages:
            task_message = Message(role="user", contents=["Please introduce yourself."])
        else:
            # Flatten transcript into single prompt that Magentic can plan against
            prompt = orchestrator.build_prompt_from_transcript(transcript_messages)
            task_message = Message(role="user", contents=[prompt])

        runtime = self._build_runtime(orchestrator)
        result = self._start_runtime(runtime, [task_message])
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
        background_task_descriptors = []
        seen_executor_ids = set()
        async for event in self._iter_workflow_events(
            orchestrator, transcript_messages
        ):
            if include_tool_notices:
                for notice in self._call_notices_from_event(event, seen_executor_ids):
                    yield "notice", notice

            event_type = self._event_type(event)

            if event_type in ("tool_result", "function_result", "executor_completed"):
                for descriptor in self._background_task_descriptors_from_event(event):
                    background_task_descriptors.append(descriptor)
                    yield "background_task", descriptor
                continue

            if not self._is_terminal_output_event(event):
                continue

            event_text = self._extract_text(event)
            if not event_text:
                continue

            if event_type in self._FINAL_EVENT_TYPES:
                # Overwrite (don't accumulate) - last final event is authoritative
                final_text = event_text
                continue

            streamed_text_parts.append(event_text)
            yield "chunk", event_text

        main_output = final_text or "".join(streamed_text_parts)
        if not main_output:
            main_output = self._background_task_ack(background_task_descriptors)

        # Stream the final/fallback text if it differs from what was already streamed
        streamed_text = "".join(streamed_text_parts)
        if main_output and main_output != streamed_text:
            yield "chunk", main_output
        yield "final", main_output

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
                elif kind == "background_task":
                    continue
            if final_text and not streamed:
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
            background_task_descriptors = []
            turn_id = None

            # Load history inside lock for non-isolated sessions to ensure atomicity
            # between turn_id reservation and history snapshot
            if isolated_session:
                history = orchestrator.session_manager.load_history()
            else:
                async with orchestrator._session_lock:
                    turn_id = orchestrator._next_turn_id
                    orchestrator._next_turn_id += 1
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
                elif kind == "background_task":
                    background_task_descriptors.append(value)

            if isolated_session:
                orchestrator._persist_isolated_response(
                    message,
                    aggregated_assistant_reply,
                    background_task_descriptors=background_task_descriptors,
                )
            else:
                await orchestrator._commit_completed_turn(
                    turn_id,
                    message,
                    aggregated_assistant_reply,
                    run_session=None,
                    history_lengths={},
                    background_task_descriptors=background_task_descriptors,
                )

            if aggregated_assistant_reply.strip() and not streamed_response:
                yield aggregated_assistant_reply
            elif not aggregated_assistant_reply.strip():
                LOG.warning("No final assistant text received from Magentic workflow")
        except Exception as e:
            error_msg = f"Error processing message: {e}"
            LOG.error(error_msg)
            traceback.print_exc()
            yield error_msg
