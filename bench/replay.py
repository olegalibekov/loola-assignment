"""Compatibility command for the offline replay: python -m bench.replay."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.runners.replay import main


if __name__ == "__main__":
    main()
