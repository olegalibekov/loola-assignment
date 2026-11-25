import pytest

from harness.cli import build_parser, main
from harness.runtime.config import Config


@pytest.mark.parametrize("values", [
    {"max_turns": 0}, {"max_output_tokens": -1}, {"max_retries": -1},
    {"max_api_attempts": -1}, {"max_wall_seconds": -1}, {"max_wall_seconds": float("nan")},
    {"request_timeout": 0}, {"request_timeout": float("inf")}, {"bash_timeout": 0},
    {"tool_output_chars": 0}, {"summary_max_tokens": 0},
    {"context_window": 4096, "max_output_tokens": 4096},
    {"compact_target": 0.8}, {"compact_trigger": 1.1}, {"compact_keep_tail": 1},
    {"extra_body": []},
])
def test_invalid_limits_fail_fast(values):
    with pytest.raises(ValueError):
        Config(**values)


def test_budget_environment_and_cli(monkeypatch):
    monkeypatch.setenv("MAX_API_ATTEMPTS", "7")
    monkeypatch.setenv("MAX_WALL_SECONDS", "12.5")
    cfg = Config.from_env()
    assert cfg.max_api_attempts == 7 and cfg.max_wall_seconds == 12.5
    args = build_parser().parse_args(["task", "--max-api-attempts", "3", "--max-wall-seconds", "8"])
    assert args.max_api_attempts == 3 and args.max_wall_seconds == 8


def test_invalid_cli_configuration_is_a_clean_error(capsys):
    assert main(["task", "--max-api-attempts", "-1"]) == 2
    assert "Invalid configuration" in capsys.readouterr().err


def test_usable_window_never_exceeds_actual_prompt_capacity():
    assert Config(context_window=4097, max_output_tokens=4096).usable_window == 1
