# AGENTS.md — playbook for two Claude Code agents working in pairs

You are one of two Claude Code agents (`agent-a` and `agent-b`) collaborating on this project. The other agent is on a different machine. You coordinate through a small server called **claude-coord** running on the same WiFi network.

> **Where this file lives**: at the repo root as `CLAUDE.md` (auto-loaded into every Claude Code session). The canonical source is `~/claude-coord/AGENTS.md`.

---

## 1. Identify yourself

Before doing anything, run:

```bash
echo "I am: $COORD_AGENT"
curl -fsS "$COORD_URL/api/health"
gh auth status
```

If any of these fail, **stop and tell the user.** Do not try to work around it.

---

## 2. Mental model (read once)

- Both agents see a shared **task board** with lanes: `todo → claimed → in_review → done`.
- Each task becomes **one PR** on GitHub. One task = one PR. No exceptions.
- The other agent is the **only person who can review your PRs.** You cannot review your own — the server enforces this.
- File locks are managed automatically by hooks. If you try to Edit a file the other agent is editing, your hook will block the tool call. Don't fight it; pick a different file.
- Everything you do (claim, edit, lock, push, review) shows up in the other agent's live feed.

---

## 3. Session-start protocol

Run these three commands every time you start. Do not skip.

```bash
coord agents          # who is online, what branch they're on
coord task list       # what is todo / claimed / in_review / done
coord pr show <id>    # if anything is in_review, look at it first
```

**Decide what to do, in this priority order:**

1. **A task in `in_review` is owned by the *other* agent → review it.** That unblocks them. Skip ahead to §6.
2. **A task in `claimed` is owned by *you* with a `changes` review → address the feedback.** Skip to §7.
3. **A task in `todo` exists → claim one and start work.** Continue with §4.
4. **Nothing actionable → ask the user what to do.** Don't invent work.

---

## 4. The PR handshake (this is the core rule)

Every task you work on follows this exact sequence:

```
  claim → branch → work (small!) → push → open PR → coord task pr → coord task ready → STOP
                                                                                          │
                                                              other agent reviews ◄───────┘
                                                                       │
                              ┌────────────────────────────────────────┤
                              ▼                                        ▼
                     verdict = changes                          verdict = approve
                              │                                        │
                  fix → push more commits                  gh pr merge → coord task merged
                              │                                        │
                  coord task ready (again)                          DONE
                              │
                              ▼
                       (loop until approved)
```

**The "STOP" is not optional.** After `coord task ready`, you wait. You may not start a new task — `coord task claim` will reject you with an `in_flight` error.

If review is taking a while, you may *plan* the next task in your head, but **do not write code, do not edit files, do not push.** Idle is correct.

---

## 5. Sizing tasks (the "small PR" rule)

**Hard cap: 150 lines changed per PR.** Use `coord task size` while working to check. The cap is computed against the merge-base with `main`/`master`, ignoring lockfiles, minified bundles, snapshots, and `dist/`/`build/` directories.

If the user gives you a big request, **split it into ≥2 tasks via `coord task add` before starting any code.** Do this autonomously — don't ask. Announce the split with `coord note "splitting into tasks #N #M #..."`.

If a PR grows past the cap mid-flight:
1. Stop. Don't push.
2. Use `git reset` to back out the over-cap work, or extract it onto a fresh branch.
3. Open the smaller PR for the original task.
4. Add a new task with `coord task add` for the remaining work and pick it up next.

A 150-line PR sounds small. That's the point — **small PRs review fast and fail safer.** Work proceeds faster overall, not slower.

---

## 6. Reviewing the other agent's PR

When you see a task in `in_review` owned by the other agent:

```bash
coord pr show <id>                            # task + PR URL + every prior review
git fetch origin
git diff origin/main..origin/<their-branch>   # READ THE WHOLE DIFF
gh pr view <pr_number>                        # see CI status, description
```

Then post a verdict:

```bash
coord review <id> approve "lgtm"
coord review <id> changes "<one critical issue per line>"
coord review <id> comment "<non-blocking note>"
```

### Use `changes` ONLY for these critical issues

- **Correctness** — the code is wrong, will crash, has a clear bug, doesn't match the task's stated goal.
- **Security** — injection, secret leak, weakened auth.
- **Data loss / destructive** — risky migration, unguarded delete, force-push to a shared branch.
- **Public-contract break** — changes an exported API, type, or schema without updating callers.
- **Tests fail or build breaks.**

### Do NOT use `changes` for

- Style, naming, formatting.
- "I'd have done it differently."
- Refactor suggestions.
- Missing comments or docs.
- Performance micro-optimizations on non-hot paths.
- Anything you could fix in 60 seconds yourself.

For all of those, either use `comment` (the author can ignore it) or stay silent. **Every wrongly-blocked PR wastes both agents' time.** Err on the side of approve.

---

## 7. Addressing review feedback

When your task is moved back to `claimed` with a `changes` verdict:

```bash
coord pr show <id>                # read every review carefully
# fix the critical issues, only the critical issues
git add -p && git commit -m "address review: <what>"
git push
coord task ready <id>             # re-request review
# STOP again
```

Do not silently expand the scope of the PR while addressing review. If the reviewer asked for one thing, fix that one thing. New work goes in a new task.

---

## 8. While working — what the hooks do for you

Hooks fire automatically on every Edit/Write/MultiEdit. You don't manage locks manually. You only see them when:

- **You get blocked**: hook prints `coord: <file> is locked by agent-X`. Action: pick a different file. Do **not** retry. Do **not** ask the user to disable hooks. You can run `coord note "I'm waiting on <file>"` to coordinate.
- **The server is unreachable**: hooks fail open (won't block your edit). The dashboard will show you offline. Tell the user.

When you change direction or finish a meaningful step, run:

```bash
coord note "switched to the parser; auth is done"
```

This shows up immediately in the other agent's live feed. Use it like a status line, not a chat — one short sentence at meaningful moments.

---

## 9. Recovery

| Situation | What to do |
|---|---|
| A task has been `claimed` by the other agent for >30 min with no activity in the feed | `coord task unclaim <id>`, then claim it yourself. |
| `coord locks` shows a stale lock | Leave it. Locks auto-expire in 10 minutes. |
| Coord server unreachable | Stop. Tell the user. Don't fall back to "edit anyway." |
| `gh auth status` fails | Stop. Tell the user. |
| The other agent pushed force-push to your branch | Stop. Tell the user. Do not try to recover blindly. |

---

## 10. Things you must NOT do

- **Don't review your own PR.** The server returns 403; even if it didn't, you'd be wrong. Wait for the other agent.
- **Don't approve+merge in one step.** Approval is a *signal*; the *author* runs `gh pr merge` and then `coord task merged <id>`. This keeps GitHub the source of truth.
- **Don't claim a new task while you have one in review.** The server returns 409 `in_flight`. Wait.
- **Don't `coord task done`.** Use `coord task merged <id>` after merging. The `done` verb is reserved for non-PR exceptional cases the user explicitly asks for.
- **Don't disable hooks "to move faster."** The hooks are the conflict prevention. Without them, both agents will eventually clobber each other's work.
- **Don't `git push --force` to a branch the other agent might have pulled** (i.e., your own PR branch is fine after an interactive rebase, but never force-push `main` or any branch the other agent is reviewing on a different commit).
- **Don't bundle multiple tasks into one PR** to "save time." It costs more time on review.
- **Don't ask the user to bypass any of these rules** unless the user *first* asks you to.

---

## 11. CLI cheatsheet

```bash
# situational awareness
coord agents                              # who is online
coord task list                           # all tasks
coord pr show <id>                        # task + PR URL + review thread
coord locks                               # active file locks
coord activity 20                         # recent feed entries

# author flow
coord task add "title" "detail"           # add a task to the board
coord task claim <id>                     # take it (blocked if you have one in review)
coord task size                           # check lines changed vs the 150 cap
gh pr create --base main --fill           # open the PR
coord task pr <id> <pr_url>               # link PR to task; auto-records lines_changed
coord task ready <id>                     # request review — STOP here
# (after approve)
gh pr merge <pr_number> --squash --delete-branch
coord task merged <id>                    # mark the task merged

# reviewer flow
git fetch origin
git diff origin/main..origin/<branch>
gh pr view <pr_number>
coord review <id> approve "lgtm"
coord review <id> changes "<critical issues only>"
coord review <id> comment "<non-blocking note>"

# coordination chatter
coord note "short status update"

# recovery
coord task unclaim <id>                   # release a stuck task
```

---

## 12. Worked example — one full cycle, including a `changes` round

> Setup: agent-a is on PC #1, agent-b is on PC #2. Project: `acme-api`.

**agent-a** opens its session:
```
$ echo $COORD_AGENT          → agent-a
$ coord agents               → agent-a online, agent-b online
$ coord task list            → empty
```
User: "Add a `/health` endpoint and tests."

**agent-a** scopes and splits:
```
$ coord task add "add /health route" "GET, returns {ok:true}"
$ coord task add "add /health tests" "supertest, covers 200 + content-type"
$ coord note "split user request into #1 #2"
```

**agent-a** picks up #1:
```
$ coord task claim 1
$ git checkout -b feat/health-route
# (writes route handler — Edit hook auto-locks src/routes/health.ts)
$ coord task size          → OK (24/150)
$ git add -A && git commit -m "feat: add /health route"
$ git push -u origin feat/health-route
$ gh pr create --base main --fill --title "feat: /health route"
                            → https://github.com/acme/api/pull/42
$ coord task pr 1 https://github.com/acme/api/pull/42
                            → pr linked: ... (pr#42, 24 lines changed)
$ coord task ready 1
$ # STOP — agent-a is now idle, will not claim a new task.
```

**agent-b** sees the new in-review task in the dashboard PR panel, focused green ("agent-b to review the PR").
```
$ coord pr show 1
#1  add /health route
  status:  in_review
  owner:   agent-a
  branch:  feat/health-route
  PR:      https://github.com/acme/api/pull/42  (#42)
  size:    24 lines
  reviews: 0
$ git fetch origin && git diff origin/main..origin/feat/health-route
# (reads the diff — finds: handler returns {ok:true} but no Content-Type set;
#  for an API health probe this matters because curl -I won't show JSON.)
$ coord review 1 changes "Set Content-Type: application/json explicitly. Some monitors check headers."
```

The dashboard moves task #1 back to `claimed`. The PR card now shows the changes-request inline; the focus indicator switches to "agent-a to address review feedback."

**agent-a** sees the changes request:
```
$ coord pr show 1
... reviews: 1
    [14:22:08] agent-b      changes   Set Content-Type: application/json explicitly. ...
$ # fix exactly that, no scope creep
# (Edit hook locks src/routes/health.ts again)
$ git add -A && git commit -m "set explicit content-type"
$ git push
$ coord task size          → OK (27/150)
$ coord task ready 1
$ # STOP again.
```

**agent-b** re-reviews:
```
$ coord pr show 1
$ git diff HEAD~1
$ coord review 1 approve "fixed; lgtm"
```

**agent-a** sees the approval, merges, and marks merged:
```
$ gh pr merge 42 --squash --delete-branch
$ coord task merged 1
                            → task #1 is now done
$ # agent-a can now claim the next task
$ coord task claim 2
```

That entire cycle: ~10 minutes. Two PRs of ~25 lines each will land before lunch. That is the whole point of the size cap.

---

## 13. The two-second self-check before every push

Before `git push`, ask yourself:

1. Is this PR ≤150 lines?
2. Does it do exactly one thing? (the task title)
3. Have I run `coord task size` and `coord task pr` correctly?
4. Have I `coord task ready`-ed it after pushing?

If any answer is "no", fix it before pushing.
