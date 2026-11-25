---
name: cli-design
description: Add or change argparse-based command line subcommands. Load before adding or changing a CLI command.
---
- One subparser per command; shared options go on the parent parser.
- A command function takes parsed args and returns an exit code; `main(argv=None)` returns it too, and
  the module ends with `raise SystemExit(main())` so `python -m package.cli` works.
- Data goes to stdout, diagnostics to stderr. Machine-readable output must be the only thing on stdout.
- Unknown input files should produce a one-line error on stderr and exit code 2, not a traceback.
- Test the command the way a user runs it (`python -m package.cli sub args`), not only via functions.
