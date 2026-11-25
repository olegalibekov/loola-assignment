---
name: python-bugfix
description: Fix failing Python tests methodically: reproduce, localise, minimal patch, re-run. Load when tests fail or code contradicts its spec.
---
1. Reproduce: run the test suite once with short tracebacks (`python -m pytest -q --tb=short`, or the
   `run_pytest` catalog tool). Note how many tests fail and in which files.
2. One failure at a time. Read the assertion: expected versus actual tells you the direction of the bug.
3. Read the function under test fully before editing, and the spec or docstring it is meant to follow.
4. Make the smallest change that makes the code match the spec. Do not edit a test to make it pass
   unless the test itself contradicts the written spec.
5. Re-run only the affected test file, then the full suite. A fix is done when the whole suite is green.
6. Keep a running list of what you changed and why; you will need it for the hand-off.
