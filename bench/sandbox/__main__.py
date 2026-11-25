"""Run the Docker sandbox selfcheck with python -m bench.sandbox."""
from .docker import selfcheck


if __name__ == "__main__":
    raise SystemExit(selfcheck())
