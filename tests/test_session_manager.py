# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for ChatSessionManager functionality.
"""

import tempfile
import pytest

from mada.core.config import SQLiteConfig
from mada.core.database import ChatSessionManager


@pytest.fixture
def temp_db_path():
    """Create a temporary database file."""
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()
    return temp_db.name


@pytest.fixture
def session_manager(temp_db_path):
    """Create a session manager with temporary database."""
    db_config = SQLiteConfig(path=temp_db_path)
    manager = ChatSessionManager(db_config)
    manager.chat_db.init_db()
    return manager


def test_create_and_list_sessions(session_manager):
    """Test creating and listing sessions."""
    # Initially no sessions
    sessions = session_manager.list_sessions()
    assert len(sessions) == 0

    # Create some sessions
    session_manager.create_new_session("session1")
    session_manager.create_new_session("session2")
    session_manager.create_new_session("session3")

    # Should have 3 sessions now
    sessions = session_manager.list_sessions()
    assert len(sessions) == 3


def test_delete_single_session(session_manager):
    """Test deleting a single session."""
    # Create sessions
    session_manager.create_new_session("session1")
    session_manager.create_new_session("session2")
    session_manager.create_new_session("session3")

    # Delete one
    session_manager.delete_session("session2")

    # Should have 2 sessions
    sessions = session_manager.list_sessions()
    assert len(sessions) == 2

    # Verify session2 is gone
    session_ids = [sid for sid, _ in sessions]
    assert "session2" not in session_ids
    assert "session1" in session_ids
    assert "session3" in session_ids


def test_delete_all_sessions(session_manager):
    """Test deleting all sessions at once."""
    # Create multiple sessions
    session_manager.create_new_session("session1")
    session_manager.create_new_session("session2")
    session_manager.create_new_session("session3")

    # Add some messages to verify complete cleanup
    session_manager.select_session("session1")
    session_manager.add_message("user", "Hello")
    session_manager.add_message("assistant", "Hi there")

    # Verify sessions exist
    sessions = session_manager.list_sessions()
    assert len(sessions) == 3

    # Delete all sessions (no confirmation prompt)
    session_manager.delete_all_sessions(confirm=False)

    # Should have no sessions
    sessions = session_manager.list_sessions()
    assert len(sessions) == 0


def test_delete_all_sessions_with_empty_database(session_manager):
    """Test that deleting all sessions works even when database is empty."""
    # Initially no sessions
    sessions = session_manager.list_sessions()
    assert len(sessions) == 0

    # Should not error when deleting from empty database
    session_manager.delete_all_sessions(confirm=False)

    # Still no sessions
    sessions = session_manager.list_sessions()
    assert len(sessions) == 0
