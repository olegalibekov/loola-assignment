"""Archive creation and checked extraction."""
from __future__ import annotations

import shutil
import stat
import tarfile
import zipfile
from pathlib import Path

from ..base import ToolContext, ToolError
from ..core import _existing
from .filesystem import _check_tree


def archive_create(ctx: ToolContext, source: str, destination: str) -> str:
    src, dst = _existing(ctx, source), ctx.path(destination)
    fmt = "zip" if dst.suffix == ".zip" else "gztar"
    base = str(dst)[:-len(".zip" if fmt == "zip" else ".tar.gz")] if dst.name.endswith((".zip", ".tar.gz")) else str(dst)
    ctx.path(base + (".zip" if fmt == "zip" else ".tar.gz"))
    _check_tree(ctx, src)
    return f"Created {shutil.make_archive(base, fmt, root_dir=src.parent, base_dir=src.name)}"


def _archive_target(ctx: ToolContext, destination: Path, name: str) -> Path:
    relative = Path(name)
    target = destination / relative
    root = destination.resolve()
    resolved = target.resolve()
    if relative.is_absolute() or ".." in relative.parts or not (resolved == root or root in resolved.parents):
        raise ToolError("policy_denied", f"archive member escapes its destination: {name}",
                        hint="Use an archive containing only relative paths under its destination.")
    return ctx.path(str(target))


def archive_extract(ctx: ToolContext, archive: str, destination: str = ".") -> str:
    src, dst = _existing(ctx, archive), ctx.path(destination)
    try:
        # Extract only ordinary data, with the same checks on every supported Python version.
        # shutil's tar-only `filter` keyword breaks ZIP and is absent from older Python 3.10.
        if zipfile.is_zipfile(src):
            with zipfile.ZipFile(src) as packed:
                members = packed.infolist()
                for member in members:
                    kind = stat.S_IFMT(member.external_attr >> 16)
                    if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                        raise ToolError("policy_denied", "archive links and special files are not supported",
                                        hint="Use an archive containing only regular files and directories.")
                    _archive_target(ctx, dst, member.filename)
                for member in members:
                    target = _archive_target(ctx, dst, member.filename)
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with packed.open(member) as source, target.open("wb") as output:
                            shutil.copyfileobj(source, output)
        else:
            with tarfile.open(src) as packed:
                members = packed.getmembers()
                for member in members:
                    if not (member.isfile() or member.isdir()):
                        raise ToolError("policy_denied", "archive links and special files are not supported",
                                        hint="Use an archive containing only regular files and directories.")
                    _archive_target(ctx, dst, member.name)
                for member in members:
                    target = _archive_target(ctx, dst, member.name)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with packed.extractfile(member) as source, target.open("wb") as output:
                            shutil.copyfileobj(source, output)
                        # Keep ordinary permissions (including executable scripts), never the
                        # archive's setuid, setgid, or sticky bits or ownership information.
                        target.chmod(member.mode & 0o777)
    except (tarfile.TarError, zipfile.BadZipFile, ValueError) as exc:
        raise ToolError("execution_error", f"cannot extract {archive}: {exc}") from None
    return f"Extracted {archive} -> {destination}"
