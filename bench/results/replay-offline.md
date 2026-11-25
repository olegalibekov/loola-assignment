Tokens are estimated as chars/4 (no tokenizer); the trajectory is identical in every row.

### unlimited window

| variant | result | checks | requests | prompt tokens | reusable prefix | uncached prompt | peak prompt | compactions |
|---|---|---|---|---|---|---|---|---|
| eager, no cap, no compaction (before) | finished | 12/12 | 49 | 806,127 | 97% | 23,577 | 23,571 | 0 |
| eager + cap + compaction | finished | 12/12 | 49 | 738,995 | 97% | 22,051 | 22,045 | 0 |
| append + cap + compaction | finished | 12/12 | 65 | 825,697 | 87% | 110,937 | 21,677 | 0 |
| dispatch + cap + compaction, schema lookups | finished | 12/12 | 65 | 830,757 | 97% | 21,759 | 21,747 | 0 |
| dispatch + cap + compaction (after) | finished | 12/12 | 56 | 670,927 | 97% | 19,847 | 19,839 | 0 |

### window 24,000 tokens (enforced)

| variant | result | checks | requests | prompt tokens | reusable prefix | uncached prompt | peak prompt | compactions |
|---|---|---|---|---|---|---|---|---|
| eager, no cap, no compaction (before) | context_overflow | 11/12 | 45 | 712,367 | 97% | 21,974 | 21,969 | 0 |
| eager + cap + compaction | finished | 12/12 | 50 | 639,807 | 96% | 26,046 | 16,324 | 1 |
| append + cap + compaction | finished | 12/12 | 66 | 658,777 | 86% | 93,554 | 16,364 | 1 |
| dispatch + cap + compaction, schema lookups | finished | 12/12 | 66 | 645,059 | 96% | 27,487 | 16,443 | 1 |
| dispatch + cap + compaction (after) | finished | 12/12 | 57 | 551,567 | 95% | 25,170 | 16,200 | 1 |

### window 12,000 tokens (enforced)

| variant | result | checks | requests | prompt tokens | reusable prefix | uncached prompt | peak prompt | compactions |
|---|---|---|---|---|---|---|---|---|
| eager, no cap, no compaction (before) | context_overflow | 0/12 | 5 | 34,643 | 79% | 7,359 | 7,359 | 0 |
| eager + cap + compaction | finished | 12/12 | 64 | 523,712 | 91% | 46,995 | 9,777 | 15 |
| append + cap + compaction | finished | 12/12 | 71 | 421,936 | 83% | 71,464 | 7,394 | 6 |
| dispatch + cap + compaction, schema lookups | finished | 12/12 | 71 | 409,502 | 92% | 34,631 | 7,474 | 6 |
| dispatch + cap + compaction (after) | finished | 12/12 | 60 | 353,170 | 92% | 27,051 | 7,457 | 4 |

