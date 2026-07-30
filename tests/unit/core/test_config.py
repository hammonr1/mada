# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from mada.core.config import (
    A2AConfig,
    DEFAULT_ORCHESTRATION_MODE,
    PostgreSQLConfig,
    RemoteA2AAgentConfig,
    SQLiteConfig,
    load_a2a_agents_config,
    load_a2a_config,
    load_orchestration_config,
)


@pytest.mark.unit
class TestSQLiteConfig:
    def test_default_path(self):
        """Test that the default SQLite path exists and is expanded correctly."""
        config = SQLiteConfig()
        assert config.path.parent.exists(), "'~/.mada/' directory should exist."
        assert "~" not in str(config.path)
        assert str(config.path) == str(config.path.expanduser()), (
            "SQLite path should be expanded correctly."
        )


@pytest.mark.unit
class TestPostgreSQLConfig:
    def test_connection_string_initialization(self):
        """Test that PostgreSQLConfig initializes correctly with a connection string."""
        connection_string = "postgresql://user:password@localhost:5432/testdb"
        config = PostgreSQLConfig(connection_string=connection_string)
        assert config.get_connection_string() == connection_string, (
            "PostgreSQL connection string should match the initialized value."
        )

    def test_individual_fields_initialization(self):
        """Test that PostgreSQLConfig constructs a connection string from individual fields."""
        config = PostgreSQLConfig(
            host="localhost",
            port=5432,
            database="testdb",
            user="user",
            password="password",
        )
        expected_connection_string = "postgresql://user:password@localhost:5432/testdb"
        assert config.get_connection_string() == expected_connection_string, (
            "PostgreSQL connection string should be constructed from individual fields."
        )

    def test_missing_fields_validation(self):
        """Test that PostgreSQLConfig raises a ValueError when required fields are missing."""
        with pytest.raises(ValueError):
            PostgreSQLConfig(
                host="localhost",
                port=5432,
                database="testdb",
                user="user",
                # Missing password
            )

    def test_env_var_expansion(self, monkeypatch):
        """Test that PostgreSQLConfig expands environment variables correctly."""
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_USER", "user")
        monkeypatch.setenv("DB_PASSWORD", "password")
        monkeypatch.setenv("DB_DATABASE", "testdb")

        config = PostgreSQLConfig(
            host="${DB_HOST}",
            port=5432,
            database="${DB_DATABASE}",
            user="${DB_USER}",
            password="${DB_PASSWORD}",
        )
        expected_connection_string = "postgresql://user:password@localhost:5432/testdb"
        assert config.get_connection_string() == expected_connection_string, (
            "PostgreSQL connection string should expand environment variables correctly."
        )


@pytest.mark.unit
class TestOrchestrationConfig:
    def test_load_orchestration_config_defaults_when_omitted(self):
        config = load_orchestration_config(None)

        assert config.mode == DEFAULT_ORCHESTRATION_MODE
        assert config.participants is None

    def test_load_orchestration_config_defaults_for_empty_object(self):
        config = load_orchestration_config({})

        assert config.mode == DEFAULT_ORCHESTRATION_MODE
        assert config.participants is None

    @pytest.mark.parametrize("invalid_value", [False, []])
    def test_load_orchestration_config_rejects_non_object_blocks(self, invalid_value):
        with pytest.raises(ValueError, match="'orchestration' must be an object"):
            load_orchestration_config(invalid_value)


@pytest.mark.unit
class TestA2AConfig:
    def test_load_a2a_config_defaults_when_omitted(self):
        config = load_a2a_config(None)

        assert config.name == "MADA"
        assert config.description == "MADA multi-agent orchestration service"
        assert config.skills == []

    def test_load_a2a_config_accepts_metadata(self):
        config = load_a2a_config(
            {
                "name": "MADA A2A",
                "description": "Agent card description",
                "version": "1.2.3",
                "url": "https://mada.example/a2a",
                "skills": [{"id": "workflow", "name": "Workflow"}],
            }
        )

        assert config == A2AConfig(
            name="MADA A2A",
            description="Agent card description",
            version="1.2.3",
            url="https://mada.example/a2a",
            skills=[{"id": "workflow", "name": "Workflow"}],
        )

    @pytest.mark.parametrize("invalid_value", [False, []])
    def test_load_a2a_config_rejects_non_object_blocks(self, invalid_value):
        with pytest.raises(ValueError, match="'a2a' must be an object"):
            load_a2a_config(invalid_value)


@pytest.mark.unit
class TestRemoteA2AAgentConfig:
    def test_load_a2a_agents_config_defaults_when_omitted(self):
        assert load_a2a_agents_config(None) == {}

    def test_load_a2a_agents_config_accepts_remote_agents(self):
        config = load_a2a_agents_config(
            {
                "optimizer": {
                    "url": "https://optimizer.example/a2a",
                    "card_url": "https://optimizer.example/.well-known/agent-card.json",
                    "description": "Remote optimizer",
                    "timeout": 30,
                    "api_key": "secret",
                    "headers": {"x-trace": "enabled"},
                }
            }
        )

        assert config == {
            "optimizer": RemoteA2AAgentConfig(
                url="https://optimizer.example/a2a",
                card_url="https://optimizer.example/.well-known/agent-card.json",
                description="Remote optimizer",
                timeout=30,
                api_key="secret",
                headers={"x-trace": "enabled"},
            )
        }

    @pytest.mark.parametrize("invalid_value", [False, []])
    def test_load_a2a_agents_config_rejects_non_object_blocks(self, invalid_value):
        with pytest.raises(ValueError, match="'a2a_agents' must be an object"):
            load_a2a_agents_config(invalid_value)
