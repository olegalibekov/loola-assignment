---
name: log-triage
description: Find the important events in a large log file without reading all of it. Load before opening any log file.
---
Logs are the fastest way to fill the context window. Never print a whole log.

1. Size it first: `wc -l file` and `ls -la file`.
2. Get the shape before the detail: count lines per severity level (`grep -c`, or the `log_levels`
   catalog tool, which prints a per-level histogram).
3. Go from rare to common: start with the highest severity present and work down.
4. Read around a hit by line number (`grep -n ... | head`, then `sed -n 'A,Bp' file`), not the whole file.
5. Copy identifiers from the log line exactly as written; record them as soon as you find them.
6. When you report a value, quote the line it came from.
