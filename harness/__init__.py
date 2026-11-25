"""Minimal coding-agent harness: dynamic tool/skill loading + cache-aware context compaction."""
from .runtime.config import Config
from .runtime.loop import Agent, RunResult

__all__ = ["Agent", "Config", "RunResult"]
