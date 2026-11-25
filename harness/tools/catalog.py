"""Catalog tools: real, working, but too many to describe in every prompt.

In `eager` mode all of them sit in the `tools` array (the baseline we measure against);
in `append` / `dispatch` mode the model discovers them through `tool_search`.
"""
from __future__ import annotations

from .base import Tool, params
from .implementations.archives import archive_create, archive_extract
from .implementations.data import csv_aggregate, csv_summary, json_query, log_levels, sqlite_query
from .implementations.filesystem import copy_path, delete_path, file_info, glob_files, grep, list_dir, make_dir, move_path
from .implementations.git import git_commit, git_diff, git_log, git_show, git_status
from .implementations.python import py_outline, py_syntax_check, run_pytest, run_python
from .implementations.system import calc, current_time, disk_usage, env_get, http_request, which
from .implementations.text import count_lines, diff_files, hash_file, head_tail, regex_replace


def _t(category: str, name: str, description: str, func, spec: dict | None = None) -> Tool:
    return Tool(name, description, params(spec), func, category)


PATH = "string: Path relative to the working directory or absolute."

CATALOG_TOOLS = [
    # fs
    _t("fs", "list_dir", "List a directory as an indented tree with file sizes.", list_dir,
       {"path?": PATH, "depth?": "integer: How many levels to descend (default 1)."}),
    _t("fs", "glob_files", "Find files by glob pattern (supports ** for recursion).", glob_files,
       {"pattern": "string: Glob such as **/*.py.", "path?": "string: Directory to search from (default .)."}),
    _t("fs", "grep", "Search file contents with a regular expression; returns path:line: text.", grep,
       {"pattern": "string: Python regular expression.", "path?": "string: File or directory to search (default .).",
        "include?": "string: Only search file names matching this glob, e.g. *.py.",
        "ignore_case?": "boolean: Case-insensitive match.", "max_results?": "integer: Maximum matches to return (default 50)."}),
    _t("fs", "file_info", "Show type, size, permissions and modification time of a path.", file_info, {"path": PATH}),
    _t("fs", "move_path", "Move or rename a file or directory.", move_path,
       {"source": PATH, "destination": "string: New path."}),
    _t("fs", "copy_path", "Copy a file or a directory tree.", copy_path,
       {"source": PATH, "destination": "string: Target path."}),
    _t("fs", "delete_path", "Delete a file or directory.", delete_path,
       {"path": PATH, "recursive?": "boolean: Required to delete a non-empty directory."}),
    _t("fs", "make_dir", "Create a directory including parents.", make_dir, {"path": PATH}),
    # text
    _t("text", "head_tail", "Show the first and last lines of a text file, omitting the middle.", head_tail,
       {"path": PATH, "head?": "integer: Lines from the start (default 20).", "tail?": "integer: Lines from the end (default 20)."}),
    _t("text", "count_lines", "Count lines, words and characters of a text file.", count_lines, {"path": PATH}),
    _t("text", "diff_files", "Unified diff between two text files.", diff_files, {"a": "string: First file.", "b": "string: Second file."}),
    _t("text", "regex_replace", "Regex search-and-replace inside one file (multiline mode).", regex_replace,
       {"path": PATH, "pattern": "string: Python regular expression.", "replacement": "string: Replacement, may use \\1 groups.",
        "count?": "integer: Maximum replacements, 0 = all."}),
    _t("text", "hash_file", "Compute a checksum of a file.", hash_file,
       {"path": PATH, "algorithm?": "enum[sha256|sha1|md5]: Hash algorithm (default sha256)."}),
    # python
    _t("python", "run_pytest", "Run pytest in the working directory with short tracebacks.", run_pytest,
       {"target?": "string: Test file, directory or node id (default: whole project).",
        "keyword?": "string: Only run tests matching this -k expression.",
        "maxfail?": "integer: Stop after this many failures.", "verbose?": "boolean: One line per test."}),
    _t("python", "run_python", "Run a Python snippet in a fresh interpreter and return its output.", run_python,
       {"code": "string: Python source code.", "timeout?": "integer: Seconds before it is killed (default 60)."}),
    _t("python", "py_syntax_check", "Check a Python file for syntax errors without running it.", py_syntax_check, {"path": PATH}),
    _t("python", "py_outline", "List classes and functions of a Python file with line numbers and signatures.", py_outline, {"path": PATH}),
    # git
    _t("git", "git_status", "Short git status of the working directory including the branch.", git_status),
    _t("git", "git_diff", "Show unstaged (or staged) changes.", git_diff,
       {"path?": "string: Limit the diff to this path.", "staged?": "boolean: Show staged changes instead."}),
    _t("git", "git_log", "Show recent commits, one line each.", git_log, {"limit?": "integer: Number of commits (default 10)."}),
    _t("git", "git_show", "Show a commit's message and changed-file statistics.", git_show, {"ref?": "string: Commit reference (default HEAD)."}),
    _t("git", "git_commit", "Stage changes and create a commit.", git_commit,
       {"message": "string: Commit message.", "add_all?": "boolean: Run `git add -A` first (default true)."}),
    # data
    _t("data", "csv_summary", "Profile a CSV file: row count and per-column type, range, distinct and top values.", csv_summary, {"path": PATH}),
    _t("data", "csv_aggregate", "Group a CSV by one column and aggregate another (raw values, no cleaning).", csv_aggregate,
       {"path": PATH, "group_by": "string: Column to group by.", "value?": "string: Numeric column to aggregate.",
        "agg?": "enum[count|sum|mean|min|max]: Aggregation (default count)."}),
    _t("data", "json_query", "Read a JSON file and return the node at a dotted path such as items.0.name.", json_query,
       {"path": PATH, "pointer?": "string: Dotted path; empty returns the whole document."}),
    _t("data", "sqlite_query", "Run one SQL statement against a SQLite database file.", sqlite_query,
       {"database": "string: Path to the .db file.", "sql": "string: SQL statement."}),
    _t("data", "log_levels", "Count log lines per severity level and show the first lines of the highest severity.", log_levels,
       {"path": PATH, "show?": "integer: How many top-severity lines to print (default 5)."}),
    # net
    _t("net", "http_request", "Perform an HTTP request and return status and body.", http_request,
       {"url": "string: http(s) URL.", "method?": "enum[GET|POST|PUT|PATCH|DELETE|HEAD]: HTTP method (default GET).",
        "body?": "string: Request body.", "timeout?": "integer: Seconds (default 30)."}),
    # sys
    _t("sys", "which", "Locate an executable on PATH.", which, {"command": "string: Program name."}),
    _t("sys", "env_get", "Read an environment variable (secret-looking names are redacted).", env_get, {"name": "string: Variable name."}),
    _t("sys", "disk_usage", "Total size of a directory broken down by top-level entry.", disk_usage, {"path?": PATH}),
    _t("sys", "archive_create", "Pack a file or directory into a .zip or .tar.gz archive.", archive_create,
       {"source": PATH, "destination": "string: Archive path ending in .zip or .tar.gz."}),
    _t("sys", "archive_extract", "Unpack a .zip or .tar archive.", archive_extract,
       {"archive": PATH, "destination?": "string: Target directory (default .)."}),
    _t("sys", "current_time", "Current local date and time in ISO 8601.", current_time),
    _t("sys", "calc", "Evaluate an arithmetic expression exactly.", calc, {"expression": "string: e.g. (12.5 * 3) / 4."}),
]
