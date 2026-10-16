#!/usr/bin/env python3
"""Find where you had to step in during Claude Code sessions, and why.

This measures how sessions went rather than what they cost. A
friction turn is a typed message that corrects, redirects, interrupts or
re-states a rule to the agent. Three stages, cheapest first:

  1. extract   free: typed human turns after the kickoff, with interrupts and
               rejected tool calls attached as signals, and the agent's last
               words and tool calls as context
  2. label     claude-haiku-5-5 labels each turn, cached per turn, so a weekly
               rerun only pays for turns it has not seen. It over-flags, so
               claude-sonnet-5-5 re-judges only the flagged turns
  3. cluster   claude-sonnet-5-5 groups the friction into causes, each tied to
               an existing rule that was not followed or to a gap, with a fix

The headless calls run with an empty setting source: no hooks, CLAUDE.md or MCP
servers, which takes the per-call baseline from ~6.8k tokens to ~0.6k. They
also run without session persistence, or next week's run would mine this one.

  ./friction.py                 last 7 days
  ./friction.py --since 14
  ./friction.py --no-llm        extraction counts only, free
  ./friction.py --rules ~/notes/conventions.md   extra standing rules to cluster against
"""

import argparse
import collections
import concurrent.futures
import dataclasses
import datetime
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pricing

PROJECTS_ROOT = Path.home() / ".claude" / "projects"
HOME = Path(os.environ.get("CLAUDE_FRICTION_HOME", Path.home() / ".claude-friction"))
CACHE = HOME / "friction-cache.jsonl"
REPORTS = HOME / "reports"
RULE_SOURCES = [Path.home() / ".claude" / "CLAUDE.md"] + [
    Path(path).expanduser() for path in os.environ.get("CLAUDE_FRICTION_RULES", "").split(os.pathsep) if path]

LABEL_MODEL = "claude-haiku-5-5"
CLUSTER_MODEL = "claude-sonnet-5-5"
# Bump when the labelling prompt or schema changes; cached labels from another
# version are relabelled.
LABEL_VERSION = 2
VERIFY_VERSION = 1003
VERIFY_MODEL = "claude-sonnet-5-5"
VERIFY_BATCH_SIZE = 30
VERIFY_TEXT_CHARS = 800
VERIFY_CONTEXT_CHARS = 400
VERIFY_TOOLS = 2
BATCH_SIZE = 20
WORKERS = 4
CLUSTER_LIMIT = 400

TURN_CHARS = 1500
CONTEXT_CHARS = 800
TOOL_INPUT_CHARS = 160
RECENT_TOOLS = 4

INTERRUPT_PREFIX = "[Request interrupted by user"
REJECTION_MARKER = "The user doesn't want to proceed with this tool use"
AUTOMATED_PREFIXES = (
    "<task-notification>",
    "<command-message>",
    "<command-name>",
    "<local-command",
    "[SYSTEM NOTIFICATION",
    "The coordinator sent a message",
    "This session is being continued",
    "Caveat:",
    # Orchestrators such as T3 Code send these as ordinary sdk turns, indistinguishable by metadata.
    "Update on pull request #",
) + tuple(prefix for prefix in os.environ.get("CLAUDE_FRICTION_AUTOMATED_PREFIXES", "").split("|") if prefix)
# Messages one agent relays to another, e.g. a QA coordinator to its workers.
RELAYED_PATTERN = re.compile(r"^(Note|Correction|Update|Message|Instruction) from the TASK-[\w.]+ (coordinator|thread)")

FRICTION_TYPES = ("correction", "repeated_rule", "redirect", "frustration", "rejection", "interruption")
LABEL_TYPES = FRICTION_TYPES + ("none", "automated")
FIX_TYPES = ("claude_md", "hook", "skill", "memory", "repo_config", "none")


@dataclasses.dataclass
class Turn:
    key: str
    project: str
    session: str
    timestamp: str
    text: str
    signals: list = dataclasses.field(default_factory=list)
    assistant_before: str = ""
    tools_before: list = dataclasses.field(default_factory=list)


class ClaudeError(RuntimeError):
    pass


def record_time(stamp):
    if not stamp:
        return None
    try:
        return datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def tail(text, limit):
    return text if len(text) <= limit else "…" + text[-limit:]


def message_text(content):
    if isinstance(content, str):
        return content.strip()
    parts = [block.get("text", "") for block in content
             if isinstance(block, dict) and block.get("type") == "text"]
    return "\n".join(parts).strip()


def tool_result_text(block):
    content = block.get("content")
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content if isinstance(content, str) else ""


def is_automated(record, text):
    origin = record.get("origin")
    if isinstance(origin, dict) and origin.get("kind") not in (None, "human"):
        return True
    if record.get("promptSource") == "system":
        return True
    return text.startswith(AUTOMATED_PREFIXES) or bool(RELAYED_PATTERN.match(text))


def extract_turns(lines, project="", session="", cutoff=None):
    """Typed human turns after the kickoff, each with the context that preceded it."""
    turns = []
    seen_kickoff = False
    pending_signals = []
    assistant_text = ""
    recent_tools = collections.deque(maxlen=RECENT_TOOLS)
    tool_names = {}

    def reset_context():
        nonlocal assistant_text
        assistant_text = ""
        recent_tools.clear()
        pending_signals.clear()

    for line in lines:
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if record.get("isSidechain"):
            continue
        content = (record.get("message") or {}).get("content")
        if content is None:
            continue

        if record.get("type") == "assistant":
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and block.get("text", "").strip():
                    assistant_text = block["text"].strip()
                elif block.get("type") == "tool_use":
                    tool_names[block.get("id")] = block.get("name", "?")
                    recent_tools.append("{}: {}".format(
                        block.get("name", "?"), json.dumps(block.get("input", {}))[:TOOL_INPUT_CHARS]))
            continue

        if record.get("type") != "user":
            continue
        if isinstance(content, list) and any(
                isinstance(block, dict) and block.get("type") == "tool_result" for block in content):
            for block in content:
                if isinstance(block, dict) and REJECTION_MARKER in tool_result_text(block):
                    pending_signals.append("rejected:" + tool_names.get(block.get("tool_use_id"), "?"))
            continue
        if record.get("isMeta") or record.get("isCompactSummary") or record.get("isVisibleInTranscriptOnly"):
            continue

        text = message_text(content)
        if text.startswith(INTERRUPT_PREFIX):
            pending_signals.append("interrupt")
            continue
        if not text or is_automated(record, text):
            continue
        if not seen_kickoff:
            seen_kickoff = True
            reset_context()
            continue
        when = record_time(record.get("timestamp"))
        if cutoff is not None and when is not None and when < cutoff:
            reset_context()
            continue

        turns.append(Turn(
            key="{}:{}".format(session, record.get("uuid", len(turns))),
            project=project,
            session=session,
            timestamp=record.get("timestamp", ""),
            text=text[:TURN_CHARS],
            signals=list(pending_signals),
            assistant_before=tail(assistant_text, CONTEXT_CHARS),
            tools_before=list(recent_tools),
        ))
        reset_context()
    return turns


def load_turns(since_days):
    cutoff = time.time() - since_days * 86400
    turns, sessions = [], 0
    for path in PROJECTS_ROOT.rglob("*.jsonl"):
        if "subagents" in path.parts:
            continue
        try:
            if path.stat().st_mtime < cutoff:
                continue
            with path.open(errors="ignore") as handle:
                found = extract_turns(handle, project=path.parent.name, session=path.stem, cutoff=cutoff)
        except OSError:
            continue
        if found:
            sessions += 1
            turns.extend(found)
    turns.sort(key=lambda turn: turn.timestamp)
    return turns, sessions


def batches(items, size):
    return [items[start:start + size] for start in range(0, len(items), size)]


def load_labels(path, version=LABEL_VERSION):
    labels = {}
    if not path.exists():
        return labels
    with path.open() as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("version") == version:
                labels[entry["key"]] = entry["label"]
    return labels


def save_labels(path, labels, version=LABEL_VERSION):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for key, label in labels.items():
            handle.write(json.dumps({"key": key, "version": version, "label": label}) + "\n")


def claude_command(model, system_prompt, schema, effort=None):
    command = [
        "claude", "-p",
        "--model", model,
        "--tools", "",
        "--setting-sources", "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--system-prompt", system_prompt,
        "--json-schema", json.dumps(schema),
        "--output-format", "json",
    ]
    if effort:
        command += ["--effort", effort]
    return command


def subprocess_runner(command, prompt):
    # Each prompt is sent once, so the CLI's default 1h cache write (2x input) is pure
    # overhead: disabling it halved the cost of a verify batch.
    environment = dict(os.environ, DISABLE_PROMPT_CACHING="1")
    with tempfile.TemporaryDirectory() as neutral_directory:
        completed = subprocess.run(command, input=prompt, capture_output=True, text=True,
                                   cwd=neutral_directory, timeout=900, env=environment)
    if completed.returncode != 0 and not completed.stdout.strip():
        raise ClaudeError("claude exited {}: {}".format(completed.returncode, completed.stderr.strip()[:300]))
    return completed.stdout


def run_claude(model, system_prompt, schema, prompt, runner=None, effort=None):
    """One headless call; returns (structured output, dollars spent)."""
    output = (runner or subprocess_runner)(claude_command(model, system_prompt, schema, effort), prompt)
    try:
        envelope = json.loads(output)
    except json.JSONDecodeError as error:
        raise ClaudeError("unparseable claude output: {}".format(output[:300])) from error
    if envelope.get("is_error") or envelope.get("structured_output") is None:
        raise ClaudeError("claude returned an error: {}".format(str(envelope.get("result"))[:300]))
    return envelope["structured_output"], envelope_cost(envelope)


def envelope_cost(envelope):
    """Price each model's tokens from our rate table; the CLI's own estimate is wrong for models it has no price for."""
    model_usage = envelope.get("modelUsage")
    if not model_usage:
        return float(envelope.get("total_cost_usd") or 0.0)
    total = 0.0
    for model, usage in model_usage.items():
        model = re.sub(r"-\d{8}$", "", model)
        if pricing.is_priced(model):
            total += pricing.cost_usd(
                model, cache_read=usage.get("cacheReadInputTokens") or 0,
                cache_write=usage.get("cacheCreationInputTokens") or 0,
                input_tokens=usage.get("inputTokens") or 0, output=usage.get("outputTokens") or 0)
        else:
            total += float(usage.get("costUSD") or 0.0)
    return total


LABEL_SYSTEM = """You label messages a software engineer typed to an AI coding agent.
Each item has the agent's last words and tool calls before the message, any signals
(the user interrupted the agent, or rejected a tool call), and the message itself.

Pick one type per item:
- correction: the agent did something wrong or not what was asked, and the user says so
- repeated_rule: the user restates a standing preference or rule the agent should already
  follow ("I told you", "again", "always", "never", "draft only")
- redirect: the user abandons the agent's direction for a different one, because the
  agent's approach was unwanted (not just the next step of the plan)
- frustration: the user expresses annoyance without a specific correction
- rejection: the user refused a tool call because the call itself was unwanted
- interruption: the user stopped the agent because what it was doing was unwanted
- none: a new request, an answer to the agent's question, approval, extra information,
  a follow-up, or thanks. Most messages are none. Do not stretch.
- automated: written by a program or another agent, not a person

severity: 0 for none/automated; 1 mild nudge; 2 clear correction; 3 the agent wasted
real work, broke something, or ignored an explicit rule.
agent_mistake: one short sentence on what the agent did that caused it, or "".
rule: if the message states or implies a general rule for future work, phrase it as
that rule in one line; otherwise "".
The interrupt and rejected signals are recorded whenever the user sends a message while
the agent is working, so they often mean nothing more than "here is my next message".
Judge by the message: a new request, extra information or a follow-up question after an
interrupt is none; only use interruption or rejection when the message shows the agent
was doing something unwanted.
Return one label per item, using the item's id."""

LABEL_SCHEMA = {
    "type": "object",
    "properties": {"labels": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "type": {"type": "string", "enum": list(LABEL_TYPES)},
            "severity": {"type": "integer", "minimum": 0, "maximum": 3},
            "agent_mistake": {"type": "string"},
            "rule": {"type": "string"},
        },
        "required": ["id", "type", "severity", "agent_mistake", "rule"],
        "additionalProperties": False,
    }}},
    "required": ["labels"],
    "additionalProperties": False,
}


def label_prompt(batch):
    items = [{
        "id": str(index),
        "signals": turn.signals,
        "agent_said": turn.assistant_before,
        "agent_tool_calls": turn.tools_before,
        "user_message": turn.text,
    } for index, turn in enumerate(batch)]
    return json.dumps({"items": items}, ensure_ascii=False)


def label_batch(batch):
    result, cost = run_claude(LABEL_MODEL, LABEL_SYSTEM, LABEL_SCHEMA, label_prompt(batch))
    labels = {}
    for label in result.get("labels", []):
        try:
            turn = batch[int(label.pop("id"))]
        except (ValueError, IndexError, KeyError):
            continue
        labels[turn.key] = label
    return labels, cost


def cached_batches(items, key_of, batch_function, batch_size, version, description, cache_path=CACHE):
    """Run batch_function over the items with no cached result; returns (results, dollars, failed batches)."""
    results = load_labels(cache_path, version)
    todo = [item for item in items if key_of(item) not in results]
    spend, failures = 0.0, 0
    if not todo:
        return results, spend, failures
    work = batches(todo, batch_size)
    print("{} {} new turns in {} batches".format(description, len(todo), len(work)), file=sys.stderr)
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(batch_function, batch) for batch in work]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                batch_labels, cost = future.result()
            except (ClaudeError, subprocess.TimeoutExpired) as error:
                failures += 1
                print("  batch failed: {}".format(error), file=sys.stderr)
                continue
            spend += cost
            results.update(batch_labels)
            save_labels(cache_path, batch_labels, version)
            print("  {}/{} batches, ${:.3f}".format(done, len(work), spend), file=sys.stderr)
    return results, spend, failures


def label_turns(turns, cache_path=CACHE):
    return cached_batches(turns, lambda turn: turn.key, label_batch, BATCH_SIZE, LABEL_VERSION,
                          "labelling with " + LABEL_MODEL, cache_path)


VERIFY_SYSTEM = """A first-pass classifier flagged these messages, which a software engineer
typed to an AI coding agent, as friction. Many of its flags are wrong. Re-judge each one.

The test: would a better agent have made this message unnecessary? Friction means the
engineer had to step in because of the agent's behaviour: it did something unwanted, did it
wrong, ignored an instruction or standing rule, went down a path the engineer had to stop,
or missed something it should have caught.

Not friction (is_friction false, type none):
- questions, discussion or pushback on the substance (design, API, product, architecture)
  where the agent had not done anything wrong yet; engineers think out loud
- the engineer changing their mind, adding a requirement, or giving information the agent
  could not have known
- choosing between options the agent offered, approving, or asking for the next step
Messages written by a program or another agent rather than a person: is_friction false,
type automated.

When it is friction, give the right type (correction, repeated_rule, redirect, frustration,
rejection, interruption) and severity (1 mild nudge, 2 clear correction, 3 wasted real work,
broke something, or ignored an explicit rule)."""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "is_friction": {"type": "boolean"},
            "type": {"type": "string", "enum": list(LABEL_TYPES)},
            "severity": {"type": "integer", "minimum": 0, "maximum": 3},
        },
        "required": ["id", "is_friction", "type", "severity"],
        "additionalProperties": False,
    }}},
    "required": ["verdicts"],
    "additionalProperties": False,
}


def verify_prompt(batch):
    items = [{
        "id": str(index),
        "signals": turn.signals,
        "agent_said": tail(turn.assistant_before, VERIFY_CONTEXT_CHARS),
        "agent_tool_calls": turn.tools_before[-VERIFY_TOOLS:],
        "user_message": turn.text[:VERIFY_TEXT_CHARS],
        "first_label": {"type": label["type"], "severity": label["severity"],
                        "agent_mistake": label["agent_mistake"]},
    } for index, (turn, label) in enumerate(batch)]
    return json.dumps({"items": items}, ensure_ascii=False)


def verify_batch(batch):
    result, cost = run_claude(VERIFY_MODEL, VERIFY_SYSTEM, VERIFY_SCHEMA, verify_prompt(batch))
    verdicts = {}
    for verdict in result.get("verdicts", []):
        try:
            turn, _ = batch[int(verdict.pop("id"))]
        except (ValueError, IndexError, KeyError):
            continue
        verdicts[turn.key] = verdict
    return verdicts, cost


def verify_turns(friction_turns, cache_path=CACHE):
    return cached_batches(friction_turns, lambda pair: pair[0].key, verify_batch, VERIFY_BATCH_SIZE,
                          VERIFY_VERSION, "verifying with " + VERIFY_MODEL, cache_path)


def apply_verdicts(friction_turns, verdicts):
    """Keep confirmed turns with the verified type and severity; returns (kept, unverified count)."""
    kept, unverified = [], 0
    for turn, label in friction_turns:
        verdict = verdicts.get(turn.key)
        if verdict is None:
            unverified += 1
            kept.append((turn, label))
            continue
        if not verdict.get("is_friction") or verdict.get("type") not in FRICTION_TYPES:
            continue
        kept.append((turn, dict(label, type=verdict["type"], severity=verdict["severity"])))
    return kept, unverified


CLUSTER_SYSTEM = """You review moments where a software engineer had to correct or redirect
their AI coding agents. You get the engineer's standing rules (their global CLAUDE.md and
any extra rule files they keep) and a list of labelled friction turns.

Group the turns into causes: one cause is one underlying agent behaviour, not one topic.
For each cause:
- rule_status: rule_exists_not_followed if a standing rule already covers it (quote the
  rule in existing_rule), no_rule if nothing covers it, not_actionable if it is one-off
  noise or a matter of taste that no rule would fix
- fix_type: claude_md (add or reword a rule), hook (enforce it mechanically, best for a
  rule that exists and keeps failing), skill (a missing procedure), memory (a missing
  fact), repo_config (belongs in one repository's config), none
- proposal: the concrete change, written so it could be applied as is: the exact rule
  text, or what the hook checks and blocks
- why: one sentence on why this fix would stop the behaviour when the current setup does not
A rule that exists and is still broken repeatedly will not be fixed by restating it more
loudly; prefer a hook or a structural change there.
Skip causes with fewer than two turns unless the turn has severity 3. Use the turn ids given."""

CLUSTER_SCHEMA = {
    "type": "object",
    "properties": {"causes": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "turn_ids": {"type": "array", "items": {"type": "string"}},
            "rule_status": {"type": "string",
                            "enum": ["rule_exists_not_followed", "no_rule", "not_actionable"]},
            "existing_rule": {"type": "string"},
            "fix_type": {"type": "string", "enum": list(FIX_TYPES)},
            "proposal": {"type": "string"},
            "why": {"type": "string"},
        },
        "required": ["title", "turn_ids", "rule_status", "existing_rule", "fix_type", "proposal", "why"],
        "additionalProperties": False,
    }}},
    "required": ["causes"],
    "additionalProperties": False,
}


def rules_text():
    sections = []
    for path in RULE_SOURCES:
        if path.exists():
            sections.append("## {}\n\n{}".format(path, path.read_text(errors="ignore")))
    return "\n\n".join(sections)


def cluster(friction_turns):
    """friction_turns: [(Turn, label)]. Returns (causes keyed by turn key, dollars)."""
    chosen = sorted(friction_turns, key=lambda pair: -pair[1]["severity"])[:CLUSTER_LIMIT]
    short_ids = {"t{}".format(index): turn.key for index, (turn, _) in enumerate(chosen)}
    items = [{
        "id": short_id,
        "project": turn.project,
        "type": label["type"],
        "severity": label["severity"],
        "agent_mistake": label["agent_mistake"],
        "rule": label["rule"],
        "user_message": turn.text[:400],
    } for short_id, (turn, label) in zip(short_ids, chosen)]
    prompt = "# Standing rules\n\n{}\n\n# Friction turns\n\n{}".format(
        rules_text(), json.dumps(items, ensure_ascii=False))
    result, cost = run_claude(CLUSTER_MODEL, CLUSTER_SYSTEM, CLUSTER_SCHEMA, prompt)
    causes = result.get("causes", [])
    for cause in causes:
        cause["turn_ids"] = [short_ids[short_id] for short_id in cause.get("turn_ids", [])
                             if short_id in short_ids]
    return causes, cost


def rank_causes(causes, severity_by_key):
    ranked = []
    for cause in causes:
        known = [key for key in cause.get("turn_ids", []) if key in severity_by_key]
        if not known:
            continue
        ranked.append(dict(cause, turn_ids=known, score=sum(severity_by_key[key] for key in known)))
    ranked.sort(key=lambda cause: (-cause["score"], -len(cause["turn_ids"])))
    return ranked


STYLE = """
:root { color-scheme: light dark; --fg:#1f2328; --muted:#656d76; --bg:#ffffff; --card:#f6f8fa;
  --line:#d0d7de; --accent:#0969da; --warn:#9a6700; --bad:#cf222e; }
@media (prefers-color-scheme: dark) { :root { --fg:#e6edf3; --muted:#8d96a0; --bg:#0d1117;
  --card:#161b22; --line:#30363d; --accent:#4493f8; --warn:#d29922; --bad:#f85149; } }
body { font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--fg);
  background: var(--bg); max-width: 980px; margin: 2rem auto; padding: 0 1rem; }
h1 { font-size: 1.5rem; margin-bottom: .25rem; } h2 { margin-top: 2.5rem; font-size: 1.15rem; }
.muted { color: var(--muted); }
.stats { display: flex; gap: 2rem; flex-wrap: wrap; margin: 1rem 0 0; }
.stat b { display: block; font-size: 1.6rem; font-variant-numeric: tabular-nums; }
.cause { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 1rem 1.25rem;
  margin: 1rem 0; }
.cause h3 { margin: 0 0 .4rem; font-size: 1.05rem; }
.badge { display: inline-block; font-size: .75rem; padding: .05rem .5rem; border-radius: 999px;
  border: 1px solid var(--line); margin-right: .35rem; color: var(--muted); }
.badge.exists { color: var(--bad); border-color: var(--bad); }
.badge.gap { color: var(--warn); border-color: var(--warn); }
.proposal { border-left: 3px solid var(--accent); padding-left: .75rem; margin: .6rem 0; white-space: pre-wrap; }
blockquote { margin: .4rem 0; padding: .3rem .75rem; border-left: 3px solid var(--line); white-space: pre-wrap; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
td, th { text-align: left; padding: .35rem .5rem; border-bottom: 1px solid var(--line); vertical-align: top; }
td.number { text-align: right; font-variant-numeric: tabular-nums; }
details { margin-top: .5rem; } summary { cursor: pointer; color: var(--accent); }
"""


def render_html(summary, causes, friction_turns):
    escape = html.escape
    turns_by_key = {turn.key: (turn, label) for turn, label in friction_turns}

    def quote(turn, label):
        return "<blockquote>{}<div class='muted'>{} · {} · {} {}{}</div></blockquote>".format(
            escape(turn.text[:600]), escape(turn.project[-48:]), escape(turn.timestamp[:16]),
            escape(label["type"]), label["severity"],
            " · " + escape(", ".join(turn.signals)) if turn.signals else "")

    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Friction report</title>",
             "<style>{}</style></head><body>".format(STYLE),
             "<h1>Where you had to step in</h1>",
             "<div class='muted'>last {} days · generated {}</div>".format(
                 summary["window_days"], time.strftime("%Y-%m-%d %H:%M"))]
    rate = 100 * len(friction_turns) / summary["turns"] if summary["turns"] else 0
    parts.append("<div class='stats'>" + "".join(
        "<div class='stat'><b>{}</b><span class='muted'>{}</span></div>".format(value, name) for value, name in (
            (summary["sessions"], "sessions"), (summary["turns"], "typed turns"),
            (summary.get("flagged", "-"), "flagged by Haiku"), (len(friction_turns), "confirmed by Sonnet"),
            ("{:.0f}%".format(rate), "friction rate"),
            ("${:.2f}".format(summary["spend_usd"]), "LLM spend"))) + "</div>")

    parts.append("<h2>Causes, ranked by total severity</h2>")
    if not causes:
        parts.append("<p class='muted'>No causes (clustering skipped or nothing to group).</p>")
    for cause in causes:
        status = cause.get("rule_status", "")
        badge_class = {"rule_exists_not_followed": "exists", "no_rule": "gap"}.get(status, "")
        members = [turns_by_key[key] for key in cause["turn_ids"] if key in turns_by_key]
        parts.append("<div class='cause'><h3>{}</h3>".format(escape(cause.get("title", ""))))
        parts.append("<span class='badge'>score {}</span><span class='badge'>{} turns</span>"
                     "<span class='badge {}'>{}</span><span class='badge'>fix: {}</span>".format(
                         cause["score"], len(members), badge_class, escape(status.replace("_", " ")),
                         escape(cause.get("fix_type", ""))))
        if cause.get("existing_rule"):
            parts.append("<p class='muted'>Existing rule: {}</p>".format(escape(cause["existing_rule"])))
        parts.append("<div class='proposal'>{}</div>".format(escape(cause.get("proposal", ""))))
        parts.append("<p class='muted'>{}</p>".format(escape(cause.get("why", ""))))
        parts.extend(quote(turn, label) for turn, label in members[:3])
        if len(members) > 3:
            parts.append("<details><summary>{} more</summary>{}</details>".format(
                len(members) - 3, "".join(quote(turn, label) for turn, label in members[3:])))
        parts.append("</div>")

    type_counts = collections.Counter(label["type"] for _, label in friction_turns)
    parts.append("<h2>Friction by type</h2><table><tr><th>type</th><th>turns</th></tr>")
    parts.extend("<tr><td>{}</td><td class='number'>{}</td></tr>".format(escape(kind), count)
                 for kind, count in type_counts.most_common())
    parts.append("</table>")

    project_counts = collections.Counter(turn.project for turn, _ in friction_turns)
    parts.append("<h2>Friction by project</h2><table><tr><th>project</th><th>turns</th></tr>")
    parts.extend("<tr><td>{}</td><td class='number'>{}</td></tr>".format(escape(project), count)
                 for project, count in project_counts.most_common(15))
    parts.append("</table>")

    parts.append("<h2>All friction turns</h2><details><summary>show {}</summary>".format(len(friction_turns)))
    for turn, label in sorted(friction_turns, key=lambda pair: pair[0].timestamp, reverse=True):
        parts.append(quote(turn, label))
        if label.get("agent_mistake"):
            parts.append("<div class='muted' style='margin:-.2rem 0 .8rem .9rem'>agent: {}</div>".format(
                escape(label["agent_mistake"])))
    parts.append("</details></body></html>")
    return "\n".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="friction.py", description=__doc__.split("\n")[0])
    parser.add_argument("--since", type=int, default=7, metavar="DAYS")
    parser.add_argument("--no-llm", action="store_true", help="extraction counts only, no model calls")
    parser.add_argument("--limit", type=int, metavar="N", help="label at most the N newest uncached turns")
    parser.add_argument("--out", type=Path, default=REPORTS, help="report directory")
    parser.add_argument("--rules", type=Path, action="append", default=[], metavar="FILE",
                        help="extra standing-rules file to cluster against (repeatable)")
    args = parser.parse_args(argv)
    RULE_SOURCES.extend(path.expanduser() for path in args.rules)

    turns, sessions = load_turns(args.since)
    signal_counts = collections.Counter(signal.split(":")[0] for turn in turns for signal in turn.signals)
    print("{} typed turns after kickoff across {} sessions (last {} days); signals: {}".format(
        len(turns), sessions, args.since, dict(signal_counts) or "none"))
    if args.no_llm:
        return 0

    if args.limit:
        cached = load_labels(CACHE)
        uncached = [turn for turn in turns if turn.key not in cached]
        keep = {turn.key for turn in uncached[-args.limit:]} | {turn.key for turn in turns if turn.key in cached}
        turns = [turn for turn in turns if turn.key in keep]

    labels, spend, failures = label_turns(turns)
    labelled = [(turn, labels[turn.key]) for turn in turns if turn.key in labels]
    flagged = [(turn, label) for turn, label in labelled if label["type"] in FRICTION_TYPES]
    print("{} labelled, {} flagged by {}, {} failed batches, labelling spend ${:.3f}".format(
        len(labelled), len(flagged), LABEL_MODEL, failures, spend))

    verdicts, verify_cost, verify_failures = verify_turns(flagged)
    spend += verify_cost
    friction_turns, unverified = apply_verdicts(flagged, verdicts)
    print("{} confirmed as friction ({:.0f}% of typed turns), {} unverified, verify spend ${:.3f}".format(
        len(friction_turns), 100 * len(friction_turns) / (len(labelled) or 1), unverified, verify_cost))

    causes = []
    if friction_turns:
        print("clustering {} friction turns with {}".format(min(len(friction_turns), CLUSTER_LIMIT), CLUSTER_MODEL),
              file=sys.stderr)
        try:
            raw_causes, cluster_cost = cluster(friction_turns)
            spend += cluster_cost
            causes = rank_causes(raw_causes, {turn.key: label["severity"] for turn, label in friction_turns})
        except (ClaudeError, subprocess.TimeoutExpired) as error:
            print("clustering failed: {}".format(error), file=sys.stderr)

    summary = {"window_days": args.since, "sessions": sessions, "turns": len(labelled), "spend_usd": spend,
               "flagged": len(flagged), "unverified": unverified}
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / "friction-{}".format(time.strftime("%Y-%m-%d"))
    stem.with_suffix(".html").write_text(render_html(summary, causes, friction_turns))
    stem.with_suffix(".json").write_text(json.dumps({
        "summary": summary,
        "causes": causes,
        "friction_turns": [dict(dataclasses.asdict(turn), label=label) for turn, label in friction_turns],
    }, indent=2, ensure_ascii=False))

    print("\nTOP CAUSES")
    for cause in causes[:8]:
        print("  {:>4}  {:>3} turns  {:<26} {}".format(
            cause["score"], len(cause["turn_ids"]), cause["rule_status"][:26], cause["title"][:70]))
    print("\ntotal LLM spend ${:.3f}".format(spend))
    print("report: file://{}".format(stem.with_suffix(".html")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
