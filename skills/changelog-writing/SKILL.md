---
name: changelog-writing
description: Write CHANGELOG.md entries in Keep-a-Changelog style. Load before editing a changelog.
---
- Entries go under the existing `## [Unreleased]` heading; never create a second one.
- Group by `### Added`, `### Changed`, `### Fixed`, `### Removed`, in that order; omit empty groups.
- One bullet per user-visible change. Start with the affected module or command in backticks, then
  what changed, in the past tense: "- `auth`: expired sessions were not refreshed; they are now renewed
  automatically."
- A fix bullet names the symptom, not just "fixed bug".
- Do not reflow or delete existing entries. Keep the file's trailing newline.
