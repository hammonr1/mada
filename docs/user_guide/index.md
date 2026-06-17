# MADA User Guide

This user guide will cover [installation](./installation.md), [configuration](./configuration.md), and [usage](./usage/index.md) of MADA.

## What is MADA?

MADA is a framework designed to facilitate collaboration between multiple specialized agents within a unified system. Built on top of the autogen framework, MADA orchestrates a [planning agent](./configuration.md#the-planning-agent) that interprets user input and delegates tasks to relevant helper agents. These agents communicate within a group chat environment, enabling dynamic problem-solving and task execution. MADA provides an intuitive interface for users to interact with the agent group, either through the command-line interface or a Gradio-based web interface, streamlining complex workflows and enhancing automation capabilities.

## How Does MADA Work?

MADA begins by reading your [configuration](./configuration.md), which specifies the [model](./configuration.md#model-configuration) for the chat completion client and the [agents](./configuration.md#agent-configuration) to include in the group chat. If you are using the [Gradio run mode](./usage/index.md#gradio-mode-overview), you can also provide [interface](./configuration.md#optional-gradio-interface-configuration) settings for customizing the Gradio web application UI.

After the group chat is established, MADA waits for user input. When a prompt is received, a [planning agent](./configuration.md#the-planning-agent)—added to the group chat automatically—interprets the input and selects the appropriate helper agent to handle the request using an mcp tool call. The helper agent’s response is streamed back to the user through the chosen interface. This interactive process continues until the session ends, either after 10 messages or when the user enters "TERMINATE".
