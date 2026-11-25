---
name: csv-data-analysis
description: Compute aggregates from CSV data exactly and reproducibly. Load before computing numbers from tabular data.
---
1. Profile before computing: header, row count and the distinct values of the columns you will group
   by (the `csv_summary` catalog tool, or `head` plus `cut | sort | uniq -c`).
2. If the project has its own loader, compute through it instead of re-implementing the parsing in an
   ad-hoc script: results must match what the project code produces.
3. Follow the project's specification for numeric types, precision and rounding.
4. Write the computation as a small script, run it, and paste its output into the deliverable. Do not
   do arithmetic in your head and do not copy numbers by eye from a long listing.
5. Cross-check one group independently.
6. Re-read any file you wrote to confirm it parses.
