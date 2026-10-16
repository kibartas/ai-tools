---
name: claude-friction
description: Find where the user had to step in during their Claude Code sessions (corrections, redirects, repeated rules, interrupts, rejected tool calls) and rank the causes, each with a proposed fix. Use when the user asks for a friction report, "where do I keep correcting the agent", "what should go in my CLAUDE.md", "why do my sessions go sideways", or wants to improve their agent setup from evidence.
---

# claude-friction

`friction.py` sits in this skill's base directory. Run it with `python3 <base directory>/friction.py`.
It needs Python 3.10+ and a signed-in `claude` CLI, and reads transcripts from `~/.claude/projects/`.

## Run it

1. Count first, which is free: `python3 <base>/friction.py --no-llm`. It prints how many typed turns
   and sessions fall in the window.
2. Run the report: `python3 <base>/friction.py` (last 7 days). Use `--since N` for another window,
   and `--limit N` to label only the N newest unlabelled turns when the count is large. A week of
   about 770 turns costs roughly $0.30. Ask before a window over 30 days or a count over 3,000 turns.
3. If the user keeps standing rules outside `~/.claude/CLAUDE.md` (a team conventions file, a
   project `CLAUDE.md`, `AGENTS.md`), pass each one with `--rules FILE` so causes are checked
   against it.

The run prints the path to the report as its last line (`report: file://…/friction-<date>.html`).
Read the `.json` file with the same name in the same folder; the HTML is for the user to open.

## Read the JSON

- `summary`: `window_days`, `sessions`, `turns`, `flagged`, `unverified`, `spend_usd`.
- `causes[]`, ranked by `score`: `title`, `rule_status` (`rule_exists_not_followed`, `no_rule`,
  `not_actionable`), `existing_rule`, `fix_type` (`claude_md`, `hook`, `skill`, `memory`,
  `repo_config`, `none`), `proposal`, `why`, `turn_ids`.
- `friction_turns[]`: each turn's `text`, `assistant_before`, `tools_before`, `signals`, and its
  `label` (`type`, `severity`, `agent_mistake`, `rule`), keyed by `key`, which `turn_ids` reference.

## Report back

Lead with the top three to five causes: what kept going wrong, how many turns it cost, and the fix.
Quote one turn per cause as evidence. Skip `not_actionable` causes unless the user asks for them.

- `rule_exists_not_followed`: the rule is already there and did not work. Suggest a stronger
  mechanism (a hook that enforces it, or moving it to where it is read) rather than repeating it.
- `no_rule`: draft the rule, hook or skill the `proposal` describes.

Propose fixes; do not apply them. Edit `CLAUDE.md`, settings or hooks only after the user approves
a specific change.

## Known noise

- Messages that an orchestrator or scheduler sends as user turns (PR-watch updates, scheduled
  follow-ups, coordinator relays) count as friction. If a cause is made of these, tell the user to
  add their fixed opening to `CLAUDE_FRICTION_AUTOMATED_PREFIXES` (`|`-separated) and rerun.
- An unpriced model's spend is not counted. Add a row to `RATES` in `pricing.py`.
