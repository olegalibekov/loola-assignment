---
name: sqlite-recipes
description: Inspect and query SQLite databases safely. Load before querying a SQLite database.
---
- List tables: `SELECT name FROM sqlite_master WHERE type='table'`. Schema: `PRAGMA table_info(t)`.
- Always add `LIMIT` when exploring. Count first: `SELECT COUNT(*) FROM t`.
- Use parameters, never string formatting, when values come from data.
- Wrap multi-statement changes in a transaction; take a file copy of the database before a migration.
- `EXPLAIN QUERY PLAN` shows whether an index is used.
