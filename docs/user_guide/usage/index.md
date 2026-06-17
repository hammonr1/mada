# Using MADA

This page explains the different ways you can run MADA, and why these options exist.

There are three ways to run MADA:

1. [Using Gradio](./gradio.md)
2. [Using the CLI](./cli.md)
3. [Using the OpenAI API](./openai-api.md)

This page explains why those modes exist and gives an overview of each.

## Why Multiple Run Modes?

MADA supports three primary run modes: Gradio, CLI, and an OpenAI-compatible API. These modes determine how you interact with the agent group once MADA is running.

- [Gradio Mode](#gradio-mode-overview): Provides a user-friendly web interface for chat-based interactions.
- [CLI Mode](#cli-mode-overview): Offers a lightweight, text-based console for direct command-line communication.
- [OpenAI API Mode](#openai-api-mode-overview): Exposes the configured agent team through `/v1` endpoints for tools like Open WebUI.

This flexibility ensures MADA can be used in both graphical and headless environments, catering to a wide range of user preferences and deployment scenarios.

### Gradio Mode Overview

In Gradio mode, MADA launches a web-based UI powered by Gradio. This interface is intuitive and accessible from any browser, making it ideal for users who prefer graphical interaction or need to share access with others. Gradio mode can be started via both CLI and Python API, depending on your setup.

**Features:**

- Chat with multiple agents in a browser
- Customize interface elements via configuration
- Suitable for collaborative and demonstration purposes
- Easily record demonstrations of your agentic setup

For details, see [Gradio Mode](./gradio.md).

### CLI Mode Overview

CLI mode runs MADA entirely within the command-line interface. This is perfect for users who prefer keyboard-driven workflows or need to operate in environments without graphical support. CLI mode is lightweight and efficient, supporting all core MADA features.

**Features:**

- Direct interaction with agents via text prompts
- Minimal resource requirements
- Ideal for remote servers and automation

For details, see [CLI Mode](./cli.md).

### OpenAI API Mode Overview

OpenAI API mode starts a FastAPI server that exposes `/v1/models` and `/v1/chat/completions`. This is useful when you want another frontend, such as Open WebUI, to talk to the configured MADA agent team through the standard OpenAI chat API shape.

**Features:**

- OpenAI-compatible model listing and chat completion endpoints
- Streaming responses over server-sent events
- Optional API key validation for incoming requests
- Clean fit for Open WebUI or other OpenAI-compatible clients

For details, see [OpenAI API Mode](./openai-api.md).
