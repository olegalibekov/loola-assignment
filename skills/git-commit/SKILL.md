---
name: git-commit
description: Create clean git commits: check state, stage deliberately, conventional message. Load before committing.
---
1. `git status --short` first. Make sure generated files are not about to be committed; add a
   `.gitignore` entry if needed.
2. Stage what belongs to the change (`git add -A` is fine when the tree is clean of generated files).
3. Message format: `<type>: <summary>` with type one of fix, feat, docs, test, refactor, chore. Summary
   in the imperative, under 72 characters. Add a body only when the why is not obvious.
4. Never use interactive commands (`git commit` without `-m`, `git rebase -i`, `git add -p`): there is
   no terminal. Never rewrite history that already exists unless asked.
5. After committing, `git status --short` must be empty and `git log --oneline -3` must show the commit.
