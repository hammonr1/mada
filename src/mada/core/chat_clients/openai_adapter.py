# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
OpenAI provider adapter implementation.

This module defines [`OpenAIAdapter`][core.chat_clients.openai_adapter.OpenAIAdapter],
a [`ProviderAdapter`][core.chat_clients.provider_adapter.ProviderAdapter] implementation
for OpenAI chat models. It validates that incoming model configuration objects are
instances of [`OpenAIModelConfig`][core.config.models.OpenAIModelConfig] before client
creation.
"""

from agent_framework.openai import OpenAIChatClient

from mada.core.config import BaseModelConfig, OpenAIModelConfig
from mada.core.chat_clients.provider_adapter import ProviderAdapter


class OpenAIAdapter(ProviderAdapter):
    """
    Provider adapter for OpenAI chat models.

    This adapter ensures that client creation uses `OpenAIModelConfig`
    instances.

    Attributes:
        provider_name:
            Name of the provider handled by this adapter.
        chat_client:
            Chat client class used to create OpenAI chat clients.

    Methods:
        validate_model_config:
            Validate that `model_config` is an `OpenAIModelConfig` instance.
    """

    provider_name = "openai"
    chat_client = OpenAIChatClient

    def validate_model_config(self, model_config: BaseModelConfig) -> None:
        """
        Validate that `model_config` is compatible with the OpenAI adapter.

        Args:
            model_config:
                Model configuration to validate.

        Raises:
            TypeError:
                Raised if `model_config` is not an `OpenAIModelConfig`
                instance.
        """
        if not isinstance(model_config, OpenAIModelConfig):
            raise TypeError(
                f"{self.provider_name} adapter requires OpenAIModelConfig, "
                f"got {type(model_config).__name__}"
            )
