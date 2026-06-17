# MADA Orchestrator Codebase Architecture

## Top-Level Repository Structure

At the top-level of the repository you'll find:

- Project configuration files (pyproject.toml, mkdocs.yaml, etc.)
- Directories to the rest of the project, including:
    - **Configuration:** Example configuration files
    - **Documentation:** Containing files for the documentation
    - **Examples:** Example agents for using the orchestrator
    - **Source Code:** All of the source code for the MADA Orchestrator project
    - **Tests:** All of the test files for testing the source code

Below is a visual representation of the top-level repository:

```bash
mada/
├── src/
├── examples/
├── configs/
├── docs/
├── .gitignore
├── pyproject.toml
├── mkdocs.yaml
└── README.md
```

## Source Code Structure

The source code for the MADA Orchestrator repository is designed with the goal of modularity and extensibility. There are multiple directories to keep components organized:

- **Core:** Implements the main orchestration logic, including the coordinator and orchestrator modules.
- **Common:** Contains shared utilities and exception handling.
- **Interfaces:** Provides CLI and Gradio interfaces for user interaction.
- **Config:** Manages configuration files and models.
- **Database:** Handles database interactions, including support for SQLite and PostgreSQL.

Below is a visual representation of the source code structure:

```bash
src/
└── mada/
    ├── core/                     # Core orchestration logic
    │   ├── config/                 # Configuration management
    │   ├── database/               # Database operations
    │   ├── coordinator.py          # Task coordination
    │   └── orchestrator.py         # Main orchestration logic
    ├── common/                   # Shared utilities
    ├── interfaces/               # User interfaces
    │   ├── cli/                    # Command-line interface
    │   └── gradio/                 # Web-based interface
```

## Test Code Structure

Tests should follow the same directory structure as the [source code](#source-code-structure). There should be organizational directories and appropriate test files underneath them. For example, tests for the coordinator module will live at `tests/core/coordinator/` just like its source code lives at `mada/core/coordinator.py`.

There can be a `conftest.py` file in each testing directory to help define shared fixtures for individual components. There will also be a `conftest.py` file at the top-level directory for fixtures that are shared across the entire test suite.

For more on testing, see the [MADA Orchestrator Testing Guide](./testing.md).
