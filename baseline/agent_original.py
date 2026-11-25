"""Minimal coding agent loop — one tool: bash.

This is the script from the assignment, kept as the "before" reference. The agent logic is
untouched (including its bugs, see README). Lines marked `# [bench]` are the only additions:
configuration from the environment and token accounting, so that it can be measured.
"""
import json, os, sys, subprocess, time  # [bench] os, time
import requests

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")  # [bench] was hard-coded
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")  # [bench]
LLM_MODEL = os.environ.get("LLM_MODEL", "")  # [bench]
LLM_HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {LLM_API_KEY}"}
MAX_TURNS = int(os.environ.get("MAX_TURNS", "1000"))  # [bench] was 1000
MAX_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "4096"))  # [bench] was 4096
EXTRA_BODY = json.loads(os.environ.get("LLM_EXTRA_BODY") or "{}")  # [bench]
# [bench] 0 = the original behaviour (no retries: the first 429/5xx kills the run). The `original-retry`
# benchmark variant sets it so that the loop itself can be measured on a flaky endpoint.
HTTP_RETRIES = int(os.environ.get("BENCH_HTTP_RETRIES", "0"))
# [bench] optional simulated window, to compare with the harness under the same limit
WINDOW = int(os.environ.get("CONTEXT_WINDOW", "0")) if os.environ.get("ENFORCE_WINDOW") in ("1", "true") else 0
METRICS = {"status": "running", "error": "", "turns": 0, "requests": 0, "prompt_tokens": 0,  # [bench]
           "completion_tokens": 0, "cached_tokens": 0, "peak_prompt_tokens": 0, "tool_calls": 0,
           "compactions": [], "started": time.time()}

SYSTEM_PROMPT = """\
You are a coding agent. Your job is to help the user with programming tasks.

You have access to ONE tool: `bash` — which executes shell commands and returns stdout/stderr.

Workflow:
1. Plan what needs to be done.
2. Use `bash` to read files, run commands, write code, etc.
3. After gathering enough information or completing the task, give your final answer in natural language.
4. To finish, reply with a regular message (no tool call).

Be concise. Explain what you're doing before each command."""

LLM_TOOLS = [
    {"type": "function",
     "function": {"name": "bash",
                  "description": "Execute a shell command and return the output.",
                  "parameters": {"type": "object",
                                 "properties": {
                                     "command": {"type": "string", "description": "The bash command to execute."}
                                 },
                                 "required": ["command"]}
                 }
            }]

def run_bash(command: str) -> str:
    try:
        command_result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        out = command_result.stdout + (f"\nSTDERR:\n{command_result.stderr}" if command_result.stderr else "")
        return f"Exit code: {command_result.returncode}\n{out}"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120s"

def call_tool(name: str, arguments: dict) -> str:
    func = {"bash": run_bash}.get(name)
    if not func:
        return f"Error: unknown tool '{name}'"
    try:
        return func(**arguments)
    except Exception as e:
        return f"Error calling {name}: {e}"

def call_llm(messages):
    payload = {"model": LLM_MODEL, "messages": messages, "tools": LLM_TOOLS, "tool_choice": "auto",
               "temperature": 0.1, "max_tokens": MAX_TOKENS}
    payload.update(EXTRA_BODY)  # [bench]
    for attempt in range(HTTP_RETRIES + 1):  # [bench] a single pass when HTTP_RETRIES == 0
        llm_http_response = requests.post(f"{LLM_BASE_URL}/chat/completions", json=payload, headers=LLM_HEADERS)
        if llm_http_response.status_code not in (429, 500, 502, 503, 504) or attempt == HTTP_RETRIES:  # [bench]
            break  # [bench]
        METRICS["http_retries"] = METRICS.get("http_retries", 0) + 1  # [bench]
        time.sleep(min(60, 1.5 * 2 ** attempt))  # [bench]
    llm_http_response.raise_for_status()
    data = llm_http_response.json()  # [bench] keep the body to read `usage`
    usage = data.get("usage") or {}  # [bench]
    METRICS["requests"] += 1  # [bench]
    METRICS["prompt_tokens"] += usage.get("prompt_tokens") or 0  # [bench]
    METRICS["completion_tokens"] += usage.get("completion_tokens") or 0  # [bench]
    METRICS["cached_tokens"] += (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0  # [bench]
    METRICS["peak_prompt_tokens"] = max(METRICS["peak_prompt_tokens"], usage.get("prompt_tokens") or 0)  # [bench]
    if WINDOW and (usage.get("prompt_tokens") or 0) > WINDOW - MAX_TOKENS:  # [bench] a real server would return 400
        raise OverflowError(f"prompt of {usage.get('prompt_tokens')} tokens exceeds the simulated window {WINDOW}")
    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    tool_calls = msg.get("tool_calls") or []
    return content, tool_calls

def agent_loop(user_message: str) -> None:
    messages: list[dict[str, object]] = [
        {"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}
    ]
    for turn in range(1, MAX_TURNS + 1):
        METRICS["turns"] = turn  # [bench]
        print(f"\n{'='*60}\n🔄 Turn {turn}\n{'='*60}")
        content, tool_calls = call_llm(messages)
        if content:
            print(f"\n🤖 {content}")
        if not tool_calls:
            print("(no text output)" if not content else "")
            print("✅ Agent finished")
            METRICS["status"] = "finished"  # [bench]
            return
        prefix = "\n" if content else ""
        for tool_call in tool_calls:
            function = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            tool_call_id = tool_call["id"]
            print(f"{prefix}🔧 Tool: {function}({json.dumps(arguments, ensure_ascii=False)})")
            result = call_tool(function, arguments)
            METRICS["tool_calls"] += 1  # [bench]
            print(f"   → {result[:500]}{'...' if len(result)>500 else ''}")
            messages.append({"role": "assistant", "content": content or None, "tool_calls": [tool_call]})
            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result})
    print(f"\n⚠️  Max turns ({MAX_TURNS}) reached. Stopping.")
    METRICS["status"] = "max_turns"  # [bench]

def write_metrics() -> None:  # [bench]
    METRICS["wall_seconds"] = round(time.time() - METRICS.pop("started"), 2)
    METRICS["total_tokens"] = METRICS["prompt_tokens"] + METRICS["completion_tokens"]
    run_dir = os.environ.get("RUN_DIR")
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
            json.dump(METRICS, fh, indent=2)

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    if not prompt.strip():
        print("No task provided. Exiting.")
        sys.exit(1)
    try:  # [bench] record how the run ended instead of losing the numbers in a traceback
        agent_loop(prompt)
    except OverflowError as exc:
        METRICS["status"], METRICS["error"] = "context_overflow", str(exc)
    except Exception as exc:
        METRICS["status"], METRICS["error"] = "crashed", f"{type(exc).__name__}: {exc}"[:300]
        print(f"\n💥 {METRICS['error']}")
    write_metrics()  # [bench]
    sys.exit(0 if METRICS["status"] == "finished" else 1)  # [bench]
