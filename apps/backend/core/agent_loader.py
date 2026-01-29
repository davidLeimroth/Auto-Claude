"""
User Agent Loader
=================

Loads custom subagent definitions from ~/.claude/agents/ directory.
These agents are made available to the Claude Agent SDK for use during sessions.

Agent files are Markdown with YAML frontmatter, matching Claude Code CLI format:

```markdown
---
name: code-reviewer
description: Reviews code for quality and best practices
tools: Read, Glob, Grep
model: sonnet
---

You are a code reviewer. When invoked, analyze the code...
```

Supported frontmatter fields:
- name (required): Unique identifier using lowercase letters and hyphens
- description (required): When Claude should delegate to this subagent
- tools: Comma-separated list or YAML list of allowed tools
- disallowedTools: Tools to deny from inherited list
- model: sonnet, opus, haiku, or inherit (default: inherit)
- permissionMode: default, acceptEdits, dontAsk, bypassPermissions, plan
"""

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# Agent Cache
# =============================================================================
# Caches loaded agents to avoid re-parsing on every create_client() call.

_AGENT_CACHE: dict[str, tuple[dict[str, dict[str, Any]], float]] = {}
_AGENT_CACHE_TTL_SECONDS = 60  # 1 minute TTL (agents change less frequently)
_AGENT_CACHE_LOCK = threading.Lock()


def _parse_yaml_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """
    Parse YAML frontmatter from markdown content.

    Args:
        content: Full markdown file content

    Returns:
        Tuple of (frontmatter dict, body content)
    """
    if not content.startswith("---"):
        return {}, content

    # Find the closing ---
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    frontmatter_str = parts[1].strip()
    body = parts[2].strip()

    # Simple YAML parsing (avoid heavy dependency)
    frontmatter: dict[str, Any] = {}
    current_key = None
    list_items: list[str] = []

    for line in frontmatter_str.split("\n"):
        line = line.rstrip()

        # Skip empty lines
        if not line:
            continue

        # Check for list item (starts with -)
        if line.startswith("  - ") or line.startswith("- "):
            item = line.lstrip(" -").strip()
            list_items.append(item)
            continue

        # If we have accumulated list items, save them
        if list_items and current_key:
            frontmatter[current_key] = list_items
            list_items = []

        # Parse key: value
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            # Remove quotes if present
            if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                value = value[1:-1]

            current_key = key

            if value:
                # Handle comma-separated values for tools
                if key in ("tools", "disallowedTools", "skills") and "," in value:
                    frontmatter[key] = [t.strip() for t in value.split(",")]
                else:
                    frontmatter[key] = value
            # If no value, might be start of a list
            elif key in ("tools", "disallowedTools", "skills"):
                list_items = []

    # Don't forget trailing list items
    if list_items and current_key:
        frontmatter[current_key] = list_items

    return frontmatter, body


def _convert_to_sdk_format(
    name: str, frontmatter: dict[str, Any], body: str
) -> dict[str, Any] | None:
    """
    Convert parsed agent file to Claude Agent SDK format.

    Args:
        name: Agent name (from filename or frontmatter)
        frontmatter: Parsed YAML frontmatter
        body: System prompt (markdown body)

    Returns:
        SDK agent definition dict or None if invalid
    """
    # Use name from frontmatter if available, otherwise use filename
    agent_name = frontmatter.get("name", name)

    # Description is required
    description = frontmatter.get("description")
    if not description:
        logger.warning(f"Agent '{agent_name}' missing required 'description' field")
        return None

    agent_def: dict[str, Any] = {
        "description": description,
        "prompt": body,
    }

    # Handle tools
    tools = frontmatter.get("tools")
    if tools:
        if isinstance(tools, str):
            tools = [t.strip() for t in tools.split(",")]
        agent_def["tools"] = tools

    # Handle disallowed tools
    disallowed = frontmatter.get("disallowedTools")
    if disallowed:
        if isinstance(disallowed, str):
            disallowed = [t.strip() for t in disallowed.split(",")]
        agent_def["disallowedTools"] = disallowed

    # Handle model
    model = frontmatter.get("model")
    if model:
        # Normalize model names
        model_map = {
            "sonnet": "sonnet",
            "opus": "opus",
            "haiku": "haiku",
            "inherit": "inherit",
        }
        agent_def["model"] = model_map.get(model.lower(), "inherit")

    # Handle permission mode
    permission_mode = frontmatter.get("permissionMode")
    if permission_mode:
        agent_def["permissionMode"] = permission_mode

    # Handle skills (preloaded into agent context)
    skills = frontmatter.get("skills")
    if skills:
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",")]
        agent_def["skills"] = skills

    return agent_def


def load_agents_from_directory(agents_dir: Path) -> dict[str, dict[str, Any]]:
    """
    Load all agent definitions from a directory.

    Args:
        agents_dir: Path to agents directory

    Returns:
        Dict mapping agent names to their SDK definitions
    """
    agents: dict[str, dict[str, Any]] = {}

    if not agents_dir.exists():
        return agents

    for md_file in agents_dir.glob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            frontmatter, body = _parse_yaml_frontmatter(content)

            if not frontmatter:
                logger.debug(f"Skipping {md_file.name}: no frontmatter")
                continue

            # Use filename (without extension) as fallback name
            filename_name = md_file.stem

            agent_def = _convert_to_sdk_format(filename_name, frontmatter, body)
            if agent_def:
                # Use name from frontmatter if available
                agent_name = frontmatter.get("name", filename_name)
                agents[agent_name] = agent_def
                logger.debug(f"Loaded agent: {agent_name} from {md_file.name}")

        except Exception as e:
            logger.warning(f"Failed to load agent from {md_file}: {e}")

    return agents


def load_user_agents(project_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """
    Load custom agents from user and project directories.

    Priority (higher priority overrides lower):
    1. Project-level: .claude/agents/ in project directory
    2. User-level: ~/.claude/agents/

    Args:
        project_dir: Optional project directory for project-level agents

    Returns:
        Dict mapping agent names to their SDK definitions
    """
    debug = os.environ.get("DEBUG", "").lower() in ("true", "1")

    # Check cache first
    cache_key = str(project_dir.resolve()) if project_dir else "user_only"
    now = time.time()

    with _AGENT_CACHE_LOCK:
        if cache_key in _AGENT_CACHE:
            cached_agents, cached_time = _AGENT_CACHE[cache_key]
            cache_age = now - cached_time
            if cache_age < _AGENT_CACHE_TTL_SECONDS:
                if debug:
                    print(
                        f"[AgentLoader] Cache HIT ({len(cached_agents)} agents, "
                        f"age: {cache_age:.1f}s / TTL: {_AGENT_CACHE_TTL_SECONDS}s)"
                    )
                return cached_agents.copy()

    # Load from directories (user-level first, then project overrides)
    agents: dict[str, dict[str, Any]] = {}

    # User-level agents (~/.claude/agents/)
    user_agents_dir = Path.home() / ".claude" / "agents"
    user_agents = load_agents_from_directory(user_agents_dir)
    agents.update(user_agents)

    if debug and user_agents:
        print(f"[AgentLoader] Loaded {len(user_agents)} user agents from {user_agents_dir}")

    # Project-level agents (.claude/agents/) - higher priority
    if project_dir:
        project_agents_dir = project_dir / ".claude" / "agents"
        project_agents = load_agents_from_directory(project_agents_dir)
        agents.update(project_agents)  # Override user-level with same names

        if debug and project_agents:
            print(
                f"[AgentLoader] Loaded {len(project_agents)} project agents "
                f"from {project_agents_dir}"
            )

    # Cache the result
    with _AGENT_CACHE_LOCK:
        _AGENT_CACHE[cache_key] = (agents.copy(), time.time())

    if debug:
        print(f"[AgentLoader] Total agents available: {len(agents)}")
        if agents:
            print(f"[AgentLoader] Agent names: {', '.join(sorted(agents.keys()))}")

    return agents


def invalidate_agent_cache(project_dir: Path | None = None) -> None:
    """
    Invalidate the agent cache.

    Args:
        project_dir: Specific project to invalidate, or None to clear all
    """
    with _AGENT_CACHE_LOCK:
        if project_dir is None:
            _AGENT_CACHE.clear()
            logger.debug("Cleared all agent cache entries")
        else:
            cache_key = str(project_dir.resolve())
            if cache_key in _AGENT_CACHE:
                del _AGENT_CACHE[cache_key]
                logger.debug(f"Invalidated agent cache for {project_dir}")


def is_user_agents_enabled() -> bool:
    """
    Check if user agents loading is enabled.

    Can be disabled via USER_AGENTS_ENABLED=false environment variable.

    Returns:
        True if user agents should be loaded
    """
    return os.environ.get("USER_AGENTS_ENABLED", "true").lower() != "false"
