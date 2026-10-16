# claude-friction

Finds the moments in your Claude Code sessions where you had to step in (a correction, a
redirect, a rule you had to repeat, an interrupt, a rejected tool call) and groups them into
ranked causes, each with a suggested fix: a CLAUDE.md rule, a hook, a skill, a memory note or
repo config.

It reads the transcripts Claude Code already keeps in `~/.claude/projects/`. Nothing leaves your
machine except the headless `claude -p` calls it makes with your own login.

## Requirements

- Python 3.10+, standard library only
- The `claude` CLI on your `PATH`, signed in

## Use it from an agent

This folder is also a Claude Code skill. Link it into your skills folder and Claude Code offers it
when you ask for a friction report or where you keep correcting the agent:

```
git clone https://github.com/kibartas/ai-tools.git ~/src/ai-tools
ln -s ~/src/ai-tools/claude-friction ~/.claude/skills/claude-friction
```

`SKILL.md` tells the agent how to run it, read the JSON report, and propose fixes without applying
them. Other agents can run `friction.py` directly and read the `.json` report next to the HTML one.

## Usage

```
./friction.py                         last 7 days
./friction.py --since 14              last 14 days
./friction.py --no-llm                count the typed turns only, free
./friction.py --limit 40              label only the 40 newest unlabelled turns
./friction.py --rules ~/notes/team-conventions.md   extra rule file to cluster against
```

It writes `friction-<date>.html` and `.json` to `~/.claude-friction/reports/` and prints the top
causes. A 7-day run over about 770 turns costs roughly $0.30 at API rates; labels are cached, so
a weekly rerun only pays for new turns.

## How it works

1. **Extract** (free). Every message you typed after a session's opening prompt, with the
   agent's last words and tool calls before it, plus interrupts and rejected tool calls.
2. **Label**. Haiku labels each turn. It over-flags, roughly a quarter of its flags are plain
   pushback, so Sonnet re-judges only the flagged turns.
3. **Cluster**. Sonnet groups the confirmed friction into causes, checks each against your
   standing rules (`~/.claude/CLAUDE.md` plus any `--rules` files), and says whether the rule
   was missing or existed and was not followed.

The model calls run with `--setting-sources ""` (no hooks, CLAUDE.md or MCP servers), which cuts
the per-call baseline from about 6.8k tokens to 0.6k, and `--no-session-persistence`, so labelling
runs don't show up as sessions next week. `DISABLE_PROMPT_CACHING=1` is set because each prompt is
sent once and a cache write would cost 2x input for nothing.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `CLAUDE_FRICTION_HOME` | `~/.claude-friction` | cache and reports |
| `CLAUDE_FRICTION_RULES` | none | extra rule files, `:`-separated |
| `CLAUDE_FRICTION_AUTOMATED_PREFIXES` | none | `\|`-separated message prefixes to ignore |

Orchestrators and schedulers (T3 Code, cron follow-ups, coordinator agents) often deliver their
messages as ordinary user turns. Add their fixed openings to `CLAUDE_FRICTION_AUTOMATED_PREFIXES`
or they will be counted as friction.

Costs come from the rate table in `pricing.py`. Add a row when you use a model it doesn't list;
an unlisted model's spend is not counted.

## Tests

```
python3 tests/test_friction.py
```
