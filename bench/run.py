"""Compatibility command for the model benchmark: python -m bench.run."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.runners.model import main


if __name__ == "__main__":
    raise SystemExit(main())
