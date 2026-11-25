import io
import stat
import tarfile
import zipfile

import pytest

from harness.tools.registry import ToolRegistry
from harness.context.skills import SkillStore


@pytest.fixture
def confined(make_cfg):
    cfg = make_cfg(confine_file_tools=True, tool_mode="eager")
    return ToolRegistry(cfg, SkillStore(None))


def test_recursive_reads_skip_links_outside_workspace(confined, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.txt").write_text("OUTSIDE_MARKER")
    work = confined.ctx.workdir
    (work / "file-link").symlink_to(outside / "private.txt")
    (work / "dir-link").symlink_to(outside, target_is_directory=True)
    (work / "local.txt").write_text("LOCAL_MARKER")
    grep = confined.execute("grep", {"pattern": "MARKER"})
    assert grep.ok and "LOCAL_MARKER" in grep.text and "OUTSIDE_MARKER" not in grep.text
    listing = confined.execute("list_dir", {"depth": 3})
    assert listing.ok and "local.txt" in listing.text and "link" not in listing.text
    usage = confined.execute("disk_usage", {})
    assert usage.ok and "total 12 B" in usage.text and "link" not in usage.text


@pytest.mark.parametrize("tool,destination", [("copy_path", "copy"), ("move_path", "moved"),
                                              ("archive_create", "files.zip")])
def test_recursive_mutations_reject_nested_external_links(confined, tmp_path, tool, destination):
    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE_MARKER")
    source = confined.ctx.workdir / "tree"
    source.mkdir()
    (source / "file-link").symlink_to(outside)
    result = confined.execute(tool, {"source": "tree", "destination": destination})
    assert result.kind == "policy_denied" and not (confined.ctx.workdir / destination).exists()
    assert outside.read_text() == "OUTSIDE_MARKER"


@pytest.mark.parametrize("tool", ["copy_path", "move_path"])
def test_copy_and_move_check_implicit_destination_filename(confined, tmp_path, tool):
    outside = tmp_path / "outside.txt"
    outside.write_text("KEEP")
    work = confined.ctx.workdir
    (work / "source.txt").write_text("replacement")
    (work / "target").mkdir()
    (work / "target" / "source.txt").symlink_to(outside)
    result = confined.execute(tool, {"source": "source.txt", "destination": "target"})
    assert result.kind == "policy_denied"
    assert outside.read_text() == "KEEP" and (work / "source.txt").read_text() == "replacement"


def test_archive_checks_automatically_added_extension(confined, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("KEEP")
    work = confined.ctx.workdir
    (work / "source.txt").write_text("hello")
    (work / "archive.tar.gz").symlink_to(outside)
    result = confined.execute("archive_create", {"source": "source.txt", "destination": "archive"})
    assert result.kind == "policy_denied" and outside.read_text() == "KEEP"


@pytest.mark.parametrize("extension", ["zip", "tar.gz"])
def test_copy_and_archive_roundtrip(confined, extension):
    work = confined.ctx.workdir
    (work / "tree" / "nested").mkdir(parents=True)
    (work / "tree" / "nested" / "hello.txt").write_text("hello")
    assert confined.execute("copy_path", {"source": "tree", "destination": "copy"}).ok
    assert (work / "copy" / "nested" / "hello.txt").read_text() == "hello"
    assert confined.execute("move_path", {"source": "copy", "destination": "moved"}).ok
    archive = "files." + extension
    assert confined.execute("archive_create", {"source": "tree", "destination": archive}).ok
    assert confined.execute("archive_extract", {"archive": archive, "destination": "output"}).ok
    assert (work / "output" / "tree" / "nested" / "hello.txt").read_text() == "hello"


def test_tar_preserves_executable_permissions_without_special_bits(confined):
    work = confined.ctx.workdir
    with tarfile.open(work / "scripts.tar", "w") as packed:
        member = tarfile.TarInfo("run.sh")
        member.mode = 0o7755
        body = b"#!/bin/sh\necho hello\n"
        member.size = len(body)
        packed.addfile(member, io.BytesIO(body))
    result = confined.execute("archive_extract", {"archive": "scripts.tar", "destination": "output"})
    assert result.ok, result.text
    extracted = work / "output" / "run.sh"
    assert extracted.read_bytes() == body
    assert stat.S_IMODE(extracted.stat().st_mode) == 0o755


@pytest.mark.parametrize("extension", ["zip", "tar"])
def test_corrupt_archive_returns_tool_error(confined, extension):
    path = confined.ctx.workdir / ("corrupt." + extension)
    path.write_bytes(b"not a valid archive")
    result = confined.execute("archive_extract", {"archive": path.name, "destination": "output"})
    assert result.kind == "execution_error" and "cannot extract" in result.text
    assert not (confined.ctx.workdir / "output").exists()


@pytest.mark.parametrize("format", ["zip", "tar"])
@pytest.mark.parametrize("attack", ["parent", "absolute", "existing_link", "archive_link"])
def test_archive_rejects_escaping_paths_and_links(confined, tmp_path, format, attack):
    work = confined.ctx.workdir
    outside = tmp_path / "outside.txt"
    outside.write_text("KEEP")
    (work / "output").mkdir()
    if attack == "existing_link":
        (work / "output" / "link").symlink_to(outside)
    name = {"parent": "../outside.txt", "absolute": str(outside), "existing_link": "link",
            "archive_link": "new-link"}[attack]
    path = work / ("malicious." + format)
    if format == "zip":
        with zipfile.ZipFile(path, "w") as packed:
            member = zipfile.ZipInfo(name)
            if attack == "archive_link":
                member.external_attr = (stat.S_IFLNK | 0o777) << 16
            packed.writestr(member, "replacement")
    else:
        with tarfile.open(path, "w") as packed:
            member = tarfile.TarInfo(name)
            if attack == "archive_link":
                member.type = tarfile.SYMTYPE
                member.linkname = str(outside)
                packed.addfile(member)
            else:
                member.size = len(b"replacement")
                packed.addfile(member, io.BytesIO(b"replacement"))
    result = confined.execute("archive_extract", {"archive": path.name, "destination": "output"})
    assert result.kind == "policy_denied"
    assert outside.read_text() == "KEEP" and not (work / "outside.txt").exists()


@pytest.mark.parametrize("operation", ["attach", "vacuum"])
def test_sql_cannot_open_secondary_paths_in_sandbox(confined, tmp_path, operation):
    outside = tmp_path / "outside.db"
    target = str(outside).replace("'", "''")
    sql = f"ATTACH DATABASE '{target}' AS outside" if operation == "attach" else f"VACUUM INTO '{target}'"
    result = confined.execute("sqlite_query", {"database": "local.db", "sql": sql})
    assert result.kind == "policy_denied" and not outside.exists()


def test_sql_ordinary_queries_and_vacuum_still_work(confined):
    def execute(sql):
        result = confined.execute("sqlite_query", {"database": "local.db", "sql": sql})
        assert result.ok, result.text
        return result.text

    execute("CREATE TABLE notes (text TEXT)")
    execute("INSERT INTO notes VALUES ('hello')")
    assert "hello" in execute("SELECT * FROM notes")
    execute("VACUUM")
