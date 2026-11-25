# Sandbox image for agent runs (see bench/sandbox/docker.py and README "Песочница").
#
# Inside the container the harness runs as root and is the only process that has the API key
# (in its own environment; there is no .env file in the image or the mounts). Everything the
# model does is fenced off from it:
#   * shell commands run as an unprivileged uid with a scrubbed environment, so neither `env`
#     nor /proc/<harness pid>/environ gives the key away;
#   * in-process file tools are confined to /workspace and /run (HARNESS_SANDBOX=1).
FROM python:3.12-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "requests>=2.31" "pytest>=8" \
    && useradd --uid 1000 --create-home --shell /bin/bash agent \
    # tool commands may run under the host uid, which has no passwd entry here
    && chmod 1777 /home/agent \
    # bind-mounted workspaces are owned by another uid
    && git config --system --add safe.directory '*'

WORKDIR /app
COPY harness ./harness
COPY skills ./skills
COPY baseline ./baseline

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    HARNESS_SANDBOX=1 \
    HARNESS_TOOL_UID=1000 \
    HARNESS_TOOL_GID=1000

WORKDIR /workspace

# Only the checker image contains task generators, reference answers and hidden checks.
FROM runtime AS checker
COPY bench /app/bench
ENTRYPOINT ["python", "-m", "bench.task.check"]

FROM runtime AS agent
COPY bench/__init__.py /app/bench/
COPY bench/sandbox/__init__.py bench/sandbox/baseline.py /app/bench/sandbox/
ENTRYPOINT ["python", "-m", "harness", "--workdir", "/workspace", "--run-dir", "/run"]
