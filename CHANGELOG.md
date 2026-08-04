# Changelog

## Unreleased

### Added
- Magentic Orchestration Functionality
- Safeguards for the orchestration switching and database consistency for async functionality
- Test coverage for Magentic orchestration

### Changed
-

### Fixed
- Wrong task planning from multi-turn conversations
- Tool call detection in nested Agent Framework event structures (including typed records)
- Incomplete streaming when final result differs from accumulated chunks

## 0.2.0 - 2026-07-07

### Added
- `bump_version.py` script for changing version
- `CHANGELOG` to track changes across releases
- workflows for running CI
- `pre-commit` and `ruff` for linting
- support for Windows OS
- workflows for publishing develop and stable versions of documentation
- Orchestration configuration layer and behavior selection through a mode-specific strategy, preserves existing CLI, Gradio, and OpenAI API interfaces.
- `orchestration.py` , `orchestrator.py` large updates to support the new pattern selection layer

### Changed
- re-architected the test suite into unit/integration/e2e tests
- converted interfaces to use async tools

### Fixed
- broken tests in the test suite
