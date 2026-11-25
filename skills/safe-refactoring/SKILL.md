---
name: safe-refactoring
description: Restructure code without changing behaviour. Load before moving, renaming or extracting code.
---
- Tests green before you start; if there are none for the code you touch, add characterisation tests.
- One mechanical step at a time (rename, extract, move), running tests after each.
- Do not mix refactoring with behaviour changes in one commit.
- Search for all usages before renaming (`grep -rn name .`), including strings and docs.
- Keep public names working or update every caller in the same step.
