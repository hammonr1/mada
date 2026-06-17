# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
LivAI provider adapter implementation.

This module defines [`LivAIAdapter`][core.chat_clients.livai_adapter.LivAIAdapter],
an [`OpenAIAdapter`][core.chat_clients.openai_adapter.OpenAIAdapter] specialization
for LivAI-hosted models.
"""

from agent_framework.openai import OpenAIChatClient

from mada.core.chat_clients.openai_adapter import OpenAIAdapter


class LivAIAdapter(OpenAIAdapter):
    """
    Provider adapter for LivAI-hosted chat models.

    LivAI models are served through OpenAI's API, so we can re-use the
    same behavior as the
    [`OpenAIAdapter`][core.chat_clients.openai_adapter.OpenAIAdapter],
    while exposing a different provider name.

    Attributes:
        provider_name:
            Name of the provider handled by this adapter.
        chat_client:
            Chat client class used to create LivAI chat clients.
    """

    provider_name = "livai"
    chat_client = OpenAIChatClient
