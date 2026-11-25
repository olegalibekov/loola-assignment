---
name: http-api-debugging
description: Debug HTTP APIs: reproduce with a minimal request and read status codes correctly. Load when an HTTP call fails or behaves unexpectedly.
---
- Reproduce with the smallest request that shows the problem (`curl -sS -i` or the `http_request` tool).
- 4xx means the request is wrong - read the response body, it usually names the field. 5xx / timeouts are
  the server's problem: retry with backoff, do not change the request.
- 401 vs 403: missing/invalid credentials vs valid credentials without permission.
- 429: honour `Retry-After`.
- Never print secrets; refer to tokens by variable name.
