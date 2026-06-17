# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Factory utilities for creating chat clients from provider-specific adapters.

This module exposes
[`MADAChatClientFactory`][core.chat_clients.chat_client_factory.MADAChatClientFactory],
which registers built-in provider adapters and creates
[`agent_framework.BaseChatClient`](https://learn.microsoft.com/en-us/python/api/agent-framework-core/agent_framework.basechatclient?view=agent-framework-python-latest)
instances from a [`BaseModelConfig`][core.config.models.BaseModelConfig].
"""

import logging
from typing import Dict, Optional

from agent_framework import BaseChatClient

from mada.core.config import BaseModelConfig
from mada.core.chat_clients.bedrock_adapter import BedrockAdapter
from mada.core.chat_clients.livai_adapter import LivAIAdapter
from mada.core.chat_clients.openai_adapter import OpenAIAdapter
from mada.core.chat_clients.provider_adapter import ProviderAdapter


LOG = logging.getLogger("mada-interface")


class MADAChatClientFactory:
    """
    Factory for registering provider adapters and creating chat clients.

    The factory maintains a mapping of provider names to `ProviderAdapter`
    instances. Each adapter is responsible for validating configuration and
    constructing the provider-specific client.

    Attributes:
        _adapters:
            Mapping of provider names to registered provider adapters.

    Methods:
        register_provider_adapter:
            Register a provider adapter with the factory.
        get_adapter:
            Return the adapter registered for a provider.
        create:
            Create a chat client from a model configuration.
    """

    def __init__(self):
        """
        Initialize the factory and register built-in provider adapters.
        """
        self._adapters: Dict[str, ProviderAdapter] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """
        Register the default provider adapters supported by the application.
        """
        self.register_provider_adapter(OpenAIAdapter())
        self.register_provider_adapter(LivAIAdapter())
        self.register_provider_adapter(BedrockAdapter())

    def register_provider_adapter(self, adapter: ProviderAdapter) -> None:
        """
        Register a provider adapter with the factory.

        Args:
            adapter:
                Provider adapter to register, keyed by its `provider_name`
                attribute.
        """
        self._adapters[adapter.provider_name] = adapter
        LOG.debug("Registered provider adapter '%s'", adapter.provider_name)

    def get_adapter(self, provider: str) -> Optional[ProviderAdapter]:
        """
        Return the registered adapter for a provider, if available.

        Args:
            provider:
                Provider name.

        Returns:
            The matching provider adapter, or `None` if the provider is not
            registered.
        """
        return self._adapters.get(provider)

    def create(self, model_config: BaseModelConfig) -> BaseChatClient:
        """
        Create a chat client for the provider and model in `model_config`.

        Args:
            model_config:
                Model configuration containing the target provider and model
                details.

        Returns:
            A provider-specific chat client instance.

        Raises:
            ValueError:
                If no adapter is registered for the requested provider, or if
                client creation fails.
        """
        adapter = self._adapters.get(model_config.provider)
        if adapter is None:
            raise ValueError(
                f"No provider adapter registered for provider '{model_config.provider}'"
            )

        try:
            return adapter.create_client(model_config)
        except Exception as e:
            raise ValueError(
                f"Failed to create chat client for provider='{model_config.provider}', "
                f"model='{model_config.model}': {e}"
            ) from e


chat_client_factory = MADAChatClientFactory()
