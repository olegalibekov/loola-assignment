import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.runtime.config import Config  # noqa: E402


@pytest.fixture
def make_cfg(tmp_path):
    def factory(**overrides) -> Config:
        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        values = dict(workdir=work, run_dir=tmp_path / "run", model="scripted", max_turns=50)
        values.update(overrides)
        return Config(**values)

    return factory
