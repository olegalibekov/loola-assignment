---
name: test-writing
description: Write focused pytest tests for new behaviour. Load before writing new tests.
---
- Name files `test_<module>.py` and functions `test_<behaviour>`; one behaviour per test.
- Arrange / act / assert, with literal expected values. Do not compute the expectation with the code
  under test.
- Cover the normal case, an edge case, and an input that must be ignored or rejected.
- Build inputs with small helpers or fixtures rather than reading large data files.
- Tests must be deterministic: no network, no clock, no reliance on unordered iteration.
- Run the new file alone first, then the whole suite.
