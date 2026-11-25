"""A fake OpenAI-compatible server that plays the reference solution over HTTP.

Purpose: smoke-test the whole benchmark pipeline (subprocesses, env, metrics, checker) without
a model or a key. It is not a benchmark - the "model" is the scripted solution from replay.py.

    python -m bench.runners.mock_server --port 8099 &
    LLM_BASE_URL=http://127.0.0.1:8099/v1 LLM_API_KEY=x LLM_MODEL=mock python -m bench.run --runs 1
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # started as a file (IDE "Run"), not with `-m`: make the repo root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from harness.llm.client import serialize_prompt
from harness.llm.scripted import call, say

from bench.runners.replay import reference_steps, script_for

BASH_ONLY = [call("bash", command="ls && head -c 300 TASK.md"), say("I only looked around.")]


def detect_mode(tools: list[dict]) -> str:
    names = {t["function"]["name"] for t in tools}
    if "use_tool" in names:
        return "dispatch"
    if "tool_search" in names:
        return "append"
    return "eager" if len(names) > 10 else "bash-only"


class Handler(BaseHTTPRequestHandler):
    cursors: dict[str, int] = {}  # conversation key -> next script index
    counter = 0

    def log_message(self, *args) -> None:  # keep the terminal quiet
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        messages, tools = body["messages"], body.get("tools") or []
        mode = detect_mode(tools)
        key = mode + "|" + str(messages[1]["content"])[:200]
        if body.get("tool_choice") == "none":
            message = {"role": "assistant", "content": "## Done\nSteps so far are complete.\n## Next\nContinue with the next step."}
            finish = "stop"
        else:
            if len(messages) == 2:
                Handler.cursors[key] = 0
            script = BASH_ONLY if mode == "bash-only" else script_for(mode, reference_steps(), lookup=mode == "append")
            index = Handler.cursors.get(key, 0)
            Handler.cursors[key] = index + 1
            resp = script[min(index, len(script) - 1)]
            for tool_call in resp.tool_calls:
                Handler.counter += 1
                tool_call["id"] = f"call_{Handler.counter}"
            message = {"role": "assistant", "content": resp.content or None, "tool_calls": resp.tool_calls or None}
            finish = resp.finish_reason
        prompt = len(serialize_prompt(tools, messages)) // 4
        payload = json.dumps({"choices": [{"index": 0, "message": message, "finish_reason": finish}],
                              "usage": {"prompt_tokens": prompt, "completion_tokens": len(json.dumps(message)) // 4,
                                        "total_tokens": prompt}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()
    print(f"mock LLM on http://127.0.0.1:{args.port}/v1")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
