"""Runtime configuration. Everything comes from the environment (or .env), nothing is hard-coded."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_MODES = ("eager", "append", "dispatch")


def load_dotenv(path: Path) -> None:
    """Tiny .env reader: KEY=VALUE lines, existing environment wins."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")


def _optional_float(raw: str | None, default: float | None) -> float | None:
    if raw is None:
        return default
    return None if raw.strip().lower() in ("", "none", "default") else float(raw)


@dataclass
class Config:
    # --- LLM endpoint ---
    base_url: str = "http://localhost:8000/v1"
    api_key: str = ""
    model: str = ""
    temperature: float | None = 0.1  # None = do not send (some models reject non-default values)
    max_output_tokens: int = 4096
    request_timeout: float = 300.0
    max_retries: int = 5
    extra_body: dict = field(default_factory=dict)
    # Provider-specific assistant-message fields that must be echoed back unchanged on the next
    # request (OpenRouter reasoning blocks, Gemini thought signatures). Others are dropped.
    passthrough_fields: tuple = ("reasoning_details", "extra_content")

    # --- loop ---
    max_turns: int = 200
    bash_timeout: int = 120
    max_api_attempts: int = 0  # total HTTP attempts, including retries/summaries; 0 disables
    max_wall_seconds: float = 0.0  # cooperative run deadline; 0 disables

    # --- context window ---
    context_window: int = 128_000
    enforce_window: bool = False  # refuse to send prompts that do not fit (simulates a small n_ctx)

    # --- dynamic loading ---
    # eager    : every tool schema in `tools`, every skill body in the system prompt (the "before")
    # append   : schemas are appended to `tools` when found (cache prefix breaks on every load)
    # dispatch : `tools` never changes; catalog tools are called through `use_tool` (cache-safe)
    tool_mode: str = "dispatch"

    # --- compaction ---
    compaction: bool = True
    compact_trigger: float = 0.75  # compact when the prompt reaches this share of the usable window
    compact_keep_tail: float = 0.25  # share of the usable window kept verbatim after compaction
    compact_elide: bool = True  # level 1: stub old large tool outputs first, without a model call
    compact_target: float = 0.5  # level 1 is enough if it brings the prompt under this share
    summary_max_tokens: int = 1200

    # --- tool output ---
    cap_tool_output: bool = True
    tool_output_chars: int = 8000

    # --- sandbox ---
    confine_file_tools: bool = False  # file tools may only touch workdir and run_dir (on inside Docker)

    # --- paths ---
    workdir: Path = field(default_factory=Path.cwd)
    run_dir: Path | None = None
    skills_dir: Path = REPO_ROOT / "skills"

    def __post_init__(self) -> None:
        if self.tool_mode not in TOOL_MODES:
            raise ValueError(f"tool_mode must be one of {TOOL_MODES}, got {self.tool_mode!r}")
        for name in ("max_output_tokens", "max_turns", "bash_timeout", "context_window",
                     "summary_max_tokens", "tool_output_chars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_retries", "max_api_attempts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("request_timeout", "max_wall_seconds"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.request_timeout == 0:
            raise ValueError("request_timeout must be positive")
        if self.context_window <= self.max_output_tokens:
            raise ValueError("context_window must exceed max_output_tokens")
        if not 0 < self.compact_target < self.compact_trigger <= 1:
            raise ValueError("compaction fractions must satisfy 0 < target < trigger <= 1")
        if not 0 <= self.compact_keep_tail < 1:
            raise ValueError("compact_keep_tail must be in [0, 1)")
        if not isinstance(self.extra_body, dict):
            raise ValueError("extra_body must be a JSON object")
        self.workdir = Path(self.workdir).resolve()
        if self.run_dir is not None:
            self.run_dir = Path(self.run_dir).resolve()

    @property
    def usable_window(self) -> int:
        """Tokens available for the prompt once room for the reply is reserved."""
        return self.context_window - self.max_output_tokens

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        load_dotenv(Path.cwd() / ".env")
        load_dotenv(REPO_ROOT / ".env")
        env = os.environ
        values: dict = dict(
            base_url=env.get("LLM_BASE_URL", cls.base_url).rstrip("/"),
            api_key=env.get("LLM_API_KEY", ""),
            model=env.get("LLM_MODEL", ""),
            temperature=_optional_float(env.get("LLM_TEMPERATURE"), cls.temperature),
            max_output_tokens=int(env.get("MAX_OUTPUT_TOKENS", cls.max_output_tokens)),
            request_timeout=float(env.get("LLM_TIMEOUT", cls.request_timeout)),
            max_retries=int(env.get("LLM_MAX_RETRIES", cls.max_retries)),
            extra_body=json.loads(env.get("LLM_EXTRA_BODY") or "{}"),
            max_turns=int(env.get("MAX_TURNS", cls.max_turns)),
            bash_timeout=int(env.get("BASH_TIMEOUT", cls.bash_timeout)),
            max_api_attempts=int(env.get("MAX_API_ATTEMPTS", cls.max_api_attempts)),
            max_wall_seconds=float(env.get("MAX_WALL_SECONDS", cls.max_wall_seconds)),
            context_window=int(env.get("CONTEXT_WINDOW", cls.context_window)),
            enforce_window=_flag("ENFORCE_WINDOW", cls.enforce_window),
            tool_mode=env.get("TOOL_MODE", cls.tool_mode),
            compaction=_flag("COMPACTION", cls.compaction),
            compact_elide=_flag("COMPACT_ELIDE", cls.compact_elide),
            cap_tool_output=_flag("CAP_TOOL_OUTPUT", cls.cap_tool_output),
            tool_output_chars=int(env.get("TOOL_OUTPUT_CHARS", cls.tool_output_chars)),
            confine_file_tools=_flag("HARNESS_SANDBOX", cls.confine_file_tools),
        )
        if env.get("RUN_DIR"):
            values["run_dir"] = Path(env["RUN_DIR"])
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)
