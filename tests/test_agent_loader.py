#!/usr/bin/env python3
"""
Tests for User Agent Loader
============================

Tests the core/agent_loader.py module functionality including:
- YAML frontmatter parsing
- Agent file loading from directories
- SDK format conversion
- Caching behavior
- Priority ordering (project > user)
"""

import os
from pathlib import Path

import pytest

from core.agent_loader import (
    _convert_to_sdk_format,
    _parse_yaml_frontmatter,
    invalidate_agent_cache,
    is_user_agents_enabled,
    load_agents_from_directory,
    load_user_agents,
)


class TestYamlFrontmatterParsing:
    """Tests for YAML frontmatter parsing."""

    def test_parse_simple_frontmatter(self):
        """Parse simple key-value frontmatter."""
        content = """---
name: test-agent
description: A test agent
model: sonnet
---

This is the body content.
"""
        frontmatter, body = _parse_yaml_frontmatter(content)

        assert frontmatter["name"] == "test-agent"
        assert frontmatter["description"] == "A test agent"
        assert frontmatter["model"] == "sonnet"
        assert body == "This is the body content."

    def test_parse_comma_separated_tools(self):
        """Parse tools as comma-separated values."""
        content = """---
name: reviewer
description: Reviews code
tools: Read, Grep, Glob, Bash
---

Body here.
"""
        frontmatter, body = _parse_yaml_frontmatter(content)

        assert frontmatter["tools"] == ["Read", "Grep", "Glob", "Bash"]

    def test_parse_yaml_list_tools(self):
        """Parse tools as YAML list."""
        content = """---
name: reviewer
description: Reviews code
tools:
  - Read
  - Grep
  - Glob
---

Body here.
"""
        frontmatter, body = _parse_yaml_frontmatter(content)

        assert frontmatter["tools"] == ["Read", "Grep", "Glob"]

    def test_parse_quoted_values(self):
        """Parse quoted string values."""
        content = """---
name: 'my-agent'
description: "An agent with 'quotes'"
---

Body.
"""
        frontmatter, body = _parse_yaml_frontmatter(content)

        assert frontmatter["name"] == "my-agent"
        assert frontmatter["description"] == "An agent with 'quotes'"

    def test_no_frontmatter(self):
        """Handle content without frontmatter."""
        content = "Just plain markdown content."
        frontmatter, body = _parse_yaml_frontmatter(content)

        assert frontmatter == {}
        assert body == content

    def test_incomplete_frontmatter(self):
        """Handle incomplete frontmatter (missing closing ---)."""
        content = """---
name: incomplete
description: Missing closing
"""
        frontmatter, body = _parse_yaml_frontmatter(content)

        assert frontmatter == {}


class TestSdkFormatConversion:
    """Tests for converting to SDK agent format."""

    def test_minimal_agent(self):
        """Convert agent with only required fields."""
        frontmatter = {
            "name": "minimal",
            "description": "A minimal agent",
        }
        body = "You are a helpful agent."

        result = _convert_to_sdk_format("minimal", frontmatter, body)

        assert result is not None
        assert result["description"] == "A minimal agent"
        assert result["prompt"] == "You are a helpful agent."
        assert "tools" not in result
        assert "model" not in result

    def test_full_agent(self):
        """Convert agent with all fields."""
        frontmatter = {
            "name": "full-agent",
            "description": "A fully configured agent",
            "tools": ["Read", "Write", "Edit"],
            "disallowedTools": ["Bash"],
            "model": "opus",
            "permissionMode": "acceptEdits",
            "skills": ["coding", "review"],
        }
        body = "You are an expert agent."

        result = _convert_to_sdk_format("full-agent", frontmatter, body)

        assert result is not None
        assert result["description"] == "A fully configured agent"
        assert result["prompt"] == "You are an expert agent."
        assert result["tools"] == ["Read", "Write", "Edit"]
        assert result["disallowedTools"] == ["Bash"]
        assert result["model"] == "opus"
        assert result["permissionMode"] == "acceptEdits"
        assert result["skills"] == ["coding", "review"]

    def test_missing_description_returns_none(self):
        """Agent without description should return None."""
        frontmatter = {"name": "no-desc"}
        body = "Some content"

        result = _convert_to_sdk_format("no-desc", frontmatter, body)

        assert result is None

    def test_model_normalization(self):
        """Model names should be normalized to lowercase."""
        frontmatter = {
            "description": "Test agent",
            "model": "SONNET",
        }

        result = _convert_to_sdk_format("test", frontmatter, "body")

        assert result["model"] == "sonnet"

    def test_tools_as_string(self):
        """Tools as comma-separated string should be split."""
        frontmatter = {
            "description": "Test agent",
            "tools": "Read, Write, Bash",
        }

        result = _convert_to_sdk_format("test", frontmatter, "body")

        assert result["tools"] == ["Read", "Write", "Bash"]


class TestLoadAgentsFromDirectory:
    """Tests for loading agents from a directory."""

    def test_load_single_agent(self, tmp_path):
        """Load a single agent file."""
        agent_file = tmp_path / "test-agent.md"
        agent_file.write_text("""---
name: test-agent
description: A test agent for unit testing
tools: Read, Grep
model: haiku
---

You are a test agent. Help with testing.
""")

        agents = load_agents_from_directory(tmp_path)

        assert "test-agent" in agents
        assert agents["test-agent"]["description"] == "A test agent for unit testing"
        assert agents["test-agent"]["tools"] == ["Read", "Grep"]
        assert agents["test-agent"]["model"] == "haiku"
        assert "You are a test agent" in agents["test-agent"]["prompt"]

    def test_load_multiple_agents(self, tmp_path):
        """Load multiple agent files."""
        (tmp_path / "agent-a.md").write_text("""---
name: agent-a
description: Agent A
---

Agent A prompt.
""")
        (tmp_path / "agent-b.md").write_text("""---
name: agent-b
description: Agent B
---

Agent B prompt.
""")

        agents = load_agents_from_directory(tmp_path)

        assert len(agents) == 2
        assert "agent-a" in agents
        assert "agent-b" in agents

    def test_skip_files_without_frontmatter(self, tmp_path):
        """Files without frontmatter should be skipped."""
        (tmp_path / "no-frontmatter.md").write_text("Just plain markdown content.")

        agents = load_agents_from_directory(tmp_path)

        assert len(agents) == 0

    def test_skip_invalid_agents(self, tmp_path):
        """Invalid agents (missing description) should be skipped."""
        (tmp_path / "invalid.md").write_text("""---
name: invalid
---

No description provided.
""")

        agents = load_agents_from_directory(tmp_path)

        assert len(agents) == 0

    def test_use_filename_as_fallback_name(self, tmp_path):
        """If name not in frontmatter, use filename."""
        (tmp_path / "my-custom-agent.md").write_text("""---
description: An agent without explicit name
---

The prompt.
""")

        agents = load_agents_from_directory(tmp_path)

        assert "my-custom-agent" in agents

    def test_nonexistent_directory(self, tmp_path):
        """Return empty dict for non-existent directory."""
        agents = load_agents_from_directory(tmp_path / "nonexistent")

        assert agents == {}

    def test_only_load_md_files(self, tmp_path):
        """Only .md files should be loaded."""
        (tmp_path / "agent.md").write_text("""---
name: agent
description: Valid agent
---

Prompt.
""")
        (tmp_path / "agent.txt").write_text("""---
name: txt-agent
description: Should be ignored
---

Prompt.
""")
        (tmp_path / "agent.yaml").write_text("name: yaml-agent")

        agents = load_agents_from_directory(tmp_path)

        assert len(agents) == 1
        assert "agent" in agents


class TestLoadUserAgents:
    """Tests for loading user agents with priority."""

    def test_load_from_user_directory(self, tmp_path, monkeypatch):
        """Load agents from ~/.claude/agents/."""
        user_agents_dir = tmp_path / ".claude" / "agents"
        user_agents_dir.mkdir(parents=True)
        (user_agents_dir / "user-agent.md").write_text("""---
name: user-agent
description: A user-level agent
---

User agent prompt.
""")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        invalidate_agent_cache()

        agents = load_user_agents(project_dir=None)

        assert "user-agent" in agents

    def test_project_overrides_user(self, tmp_path, monkeypatch):
        """Project-level agents should override user-level agents with same name."""
        # Create user agent
        user_agents_dir = tmp_path / ".claude" / "agents"
        user_agents_dir.mkdir(parents=True)
        (user_agents_dir / "shared-agent.md").write_text("""---
name: shared-agent
description: User version
---

User prompt.
""")

        # Create project agent with same name
        project_dir = tmp_path / "myproject"
        project_agents_dir = project_dir / ".claude" / "agents"
        project_agents_dir.mkdir(parents=True)
        (project_agents_dir / "shared-agent.md").write_text("""---
name: shared-agent
description: Project version
---

Project prompt.
""")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        invalidate_agent_cache()

        agents = load_user_agents(project_dir=project_dir)

        assert "shared-agent" in agents
        assert agents["shared-agent"]["description"] == "Project version"

    def test_merge_user_and_project_agents(self, tmp_path, monkeypatch):
        """Both user and project agents should be available."""
        # Create user agent
        user_agents_dir = tmp_path / ".claude" / "agents"
        user_agents_dir.mkdir(parents=True)
        (user_agents_dir / "user-only.md").write_text("""---
name: user-only
description: User only agent
---

User prompt.
""")

        # Create project agent
        project_dir = tmp_path / "myproject"
        project_agents_dir = project_dir / ".claude" / "agents"
        project_agents_dir.mkdir(parents=True)
        (project_agents_dir / "project-only.md").write_text("""---
name: project-only
description: Project only agent
---

Project prompt.
""")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        invalidate_agent_cache()

        agents = load_user_agents(project_dir=project_dir)

        assert "user-only" in agents
        assert "project-only" in agents


class TestAgentCaching:
    """Tests for agent caching behavior."""

    def test_cache_hit(self, tmp_path, monkeypatch):
        """Second call should use cached agents."""
        user_agents_dir = tmp_path / ".claude" / "agents"
        user_agents_dir.mkdir(parents=True)
        (user_agents_dir / "cached-agent.md").write_text("""---
name: cached-agent
description: A cached agent
---

Prompt.
""")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        invalidate_agent_cache()

        # First call loads from disk
        agents1 = load_user_agents(project_dir=None)

        # Remove the file
        (user_agents_dir / "cached-agent.md").unlink()

        # Second call should still return cached result
        agents2 = load_user_agents(project_dir=None)

        assert "cached-agent" in agents1
        assert "cached-agent" in agents2

    def test_cache_invalidation(self, tmp_path, monkeypatch):
        """Cache invalidation should force reload."""
        user_agents_dir = tmp_path / ".claude" / "agents"
        user_agents_dir.mkdir(parents=True)
        (user_agents_dir / "agent.md").write_text("""---
name: agent
description: Original
---

Original prompt.
""")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        invalidate_agent_cache()

        # First load
        agents1 = load_user_agents(project_dir=None)
        assert agents1["agent"]["description"] == "Original"

        # Update file
        (user_agents_dir / "agent.md").write_text("""---
name: agent
description: Updated
---

Updated prompt.
""")

        # Without invalidation, should still be cached
        agents2 = load_user_agents(project_dir=None)
        assert agents2["agent"]["description"] == "Original"

        # After invalidation, should reload
        invalidate_agent_cache()
        agents3 = load_user_agents(project_dir=None)
        assert agents3["agent"]["description"] == "Updated"


class TestUserAgentsEnabled:
    """Tests for the USER_AGENTS_ENABLED environment variable."""

    def test_enabled_by_default(self, monkeypatch):
        """User agents should be enabled by default."""
        monkeypatch.delenv("USER_AGENTS_ENABLED", raising=False)
        assert is_user_agents_enabled() is True

    def test_explicitly_enabled(self, monkeypatch):
        """USER_AGENTS_ENABLED=true should enable."""
        monkeypatch.setenv("USER_AGENTS_ENABLED", "true")
        assert is_user_agents_enabled() is True

    def test_disabled(self, monkeypatch):
        """USER_AGENTS_ENABLED=false should disable."""
        monkeypatch.setenv("USER_AGENTS_ENABLED", "false")
        assert is_user_agents_enabled() is False

    def test_case_insensitive(self, monkeypatch):
        """Environment variable should be case-insensitive."""
        monkeypatch.setenv("USER_AGENTS_ENABLED", "FALSE")
        assert is_user_agents_enabled() is False

        monkeypatch.setenv("USER_AGENTS_ENABLED", "False")
        assert is_user_agents_enabled() is False
