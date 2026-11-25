"""Source fingerprints and a local per-key benchmark lock; never records credentials."""
from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from contextlib import contextmanager
from pathlib import Path


class SourceChangedError(RuntimeError):
    pass


def source_manifest(root: Path) -> dict:
    root = Path(root)
    paths = [p for directory in ("harness", "baseline", "bench")
             for p in (root / directory).rglob("*.py") if "__pycache__" not in p.parts]
    paths += list((root / "skills").glob("*/SKILL.md"))
    paths += [root / name for name in ("pyproject.toml", "Dockerfile", ".dockerignore", "requirements.txt", "uv.lock")
              if (root / name).is_file()]
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(paths)}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"algorithm": "sha256", "digest": hashlib.sha256(encoded).hexdigest(), "files": files}


def verify_source(root: Path, expected: dict) -> dict:
    actual = source_manifest(root)
    if actual["digest"] != expected["digest"]:
        changed = sorted(name for name in expected["files"].keys() | actual["files"].keys()
                         if expected["files"].get(name) != actual["files"].get(name))
        raise SourceChangedError("Benchmark source changed; series stopped: " + ", ".join(changed))
    return actual


def runtime_metadata() -> dict:
    dependencies = {}
    for name in ("requests", "urllib3", "certifi", "charset-normalizer", "idna", "pytest", "setuptools"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "platform": platform.platform(), "dependencies": dependencies}


def safe_config(cfg) -> dict:
    names = ("model", "temperature", "max_output_tokens", "request_timeout", "max_retries", "bash_timeout",
             "max_api_attempts", "max_wall_seconds",
             "tool_mode", "compaction", "compact_trigger", "compact_keep_tail", "compact_elide", "compact_target",
             "summary_max_tokens", "cap_tool_output", "tool_output_chars", "confine_file_tools")
    config = {name: getattr(cfg, name) for name in names}
    # Endpoint URLs and provider extensions may contain credentials: fingerprint, never serialize them.
    config["endpoint_sha256"] = hashlib.sha256(cfg.base_url.encode()).hexdigest()
    config["extra_body_sha256"] = hashlib.sha256(
        json.dumps(cfg.extra_body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return config


@contextmanager
def series_lock(api_key: str, directory: Path | None = None):
    """Only one local series per credential. Kernel locks release even after a crash."""
    directory = Path(directory or Path(tempfile.gettempdir()) / f"loola-benchmark-locks-{os.getuid()}")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = hashlib.sha256(api_key.encode()).hexdigest() + ".lock"
    fd = os.open(directory / name, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another local benchmark series is using this API key; wait for it to finish.") from None
        yield
    finally:
        os.close(fd)
