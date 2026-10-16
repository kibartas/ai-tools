import datetime
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "friction.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("friction", MODULE_PATH)
friction = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(friction)


def stamp(minutes):
    when = datetime.datetime(2026, 10, 7, 9, 0, tzinfo=datetime.timezone.utc) + datetime.timedelta(minutes=minutes)
    return when.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def typed(text, uuid, minutes=0, **extra):
    record = {"type": "user", "uuid": uuid, "timestamp": stamp(minutes),
              "message": {"role": "user", "content": text}}
    record.update(extra)
    return json.dumps(record)


def said(text, minutes=0):
    return json.dumps({"type": "assistant", "timestamp": stamp(minutes),
                       "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}})


def tool_call(name, tool_input, tool_id="t1", minutes=0):
    return json.dumps({"type": "assistant", "timestamp": stamp(minutes),
                       "message": {"role": "assistant", "content": [
                           {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}]}})


def tool_output(content, tool_id="t1", is_error=False, minutes=0):
    return json.dumps({"type": "user", "uuid": "r-" + tool_id, "timestamp": stamp(minutes),
                       "message": {"role": "user", "content": [
                           {"type": "tool_result", "tool_use_id": tool_id, "content": content,
                            "is_error": is_error}]}})


KICKOFF = typed("Work on TASK-1", "u0")


def extract(lines, cutoff=None):
    return friction.extract_turns(lines, project="proj", session="sess", cutoff=cutoff)


class HumanTurnExtractionTests(unittest.TestCase):
    """Only what the user typed after the kickoff can be friction; everything the
    harness injects as a user record has to be filtered out first."""

    def test_a_typed_follow_up_is_a_candidate(self):
        turns = extract([KICKOFF, said("done"), typed("no, use a draft PR", "u1", 1)])

        self.assertEqual([turn.text for turn in turns], ["no, use a draft PR"])

    def test_the_kickoff_message_is_not_a_candidate(self):
        self.assertEqual(extract([KICKOFF]), [])

    def test_task_notifications_are_dropped(self):
        notification = typed("<task-notification> <task-id>x</task-id>", "u1", 1,
                             origin={"kind": "task-notification"}, promptSource="system")

        self.assertEqual(extract([KICKOFF, notification]), [])

    def test_a_notification_without_origin_metadata_is_dropped_by_its_text(self):
        self.assertEqual(extract([KICKOFF, typed("<task-notification> done", "u1", 1)]), [])

    def test_t3_and_coordinator_templates_are_dropped(self):
        # Orchestrators send these as ordinary sdk turns, so only the text tells them apart.
        templates = [
            "Update on pull request #560 (https://github.com/x/y/pull/560), which T3 Code is watching",
            "Note from the TASK-269 coordinator: do NOT recompose any page",
            "Correction from the TASK-269 coordinator: do not run sync.sh either",
        ]
        lines = [KICKOFF] + [typed(template, "u{}".format(index), index + 1)
                             for index, template in enumerate(templates)]

        self.assertEqual(extract(lines), [])

    def test_meta_sidechain_and_compaction_records_are_dropped(self):
        lines = [
            KICKOFF,
            typed("Base directory for this skill: /x", "u1", 1, isMeta=True),
            typed("subagent prompt", "u2", 2, isSidechain=True),
            typed("This session is being continued from a previous conversation", "u3", 3,
                  isCompactSummary=True),
        ]

        self.assertEqual(extract(lines), [])

    def test_slash_command_invocations_are_dropped(self):
        command = typed("<command-message>merge</command-message> <command-name>/merge</command-name>", "u1", 1)

        self.assertEqual(extract([KICKOFF, command]), [])

    def test_text_blocks_are_joined_and_images_ignored(self):
        blocks = [{"type": "image", "source": {}}, {"type": "text", "text": "this looks wrong"}]
        record = json.dumps({"type": "user", "uuid": "u1", "timestamp": stamp(1),
                             "message": {"role": "user", "content": blocks}})

        self.assertEqual([turn.text for turn in extract([KICKOFF, record])], ["this looks wrong"])

    def test_turns_before_the_cutoff_are_dropped_but_still_count_as_the_kickoff(self):
        cutoff = datetime.datetime(2026, 10, 7, 9, 30, tzinfo=datetime.timezone.utc).timestamp()
        lines = [KICKOFF, typed("old correction", "u1", 10), typed("new correction", "u2", 40)]

        self.assertEqual([turn.text for turn in extract(lines, cutoff)], ["new correction"])


class SignalTests(unittest.TestCase):
    def test_an_interrupt_marks_the_next_typed_turn(self):
        lines = [KICKOFF, said("running"),
                 typed([{"type": "text", "text": "[Request interrupted by user for tool use]"}], "u1", 1),
                 typed("stop, wrong repo", "u2", 2)]
        turns = extract(lines)

        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].signals, ["interrupt"])

    def test_a_rejected_tool_call_marks_the_next_typed_turn_with_the_tool(self):
        lines = [KICKOFF, tool_call("Bash", {"command": "gh pr create"}),
                 tool_output("The user doesn't want to proceed with this tool use.", is_error=True),
                 typed("as a draft please", "u2", 2)]
        turns = extract(lines)

        self.assertEqual(turns[0].signals, ["rejected:Bash"])

    def test_signals_do_not_leak_past_the_turn_they_mark(self):
        lines = [KICKOFF,
                 typed([{"type": "text", "text": "[Request interrupted by user]"}], "u1", 1),
                 typed("first", "u2", 2), typed("second", "u3", 3)]
        turns = extract(lines)

        self.assertEqual([turn.signals for turn in turns], [["interrupt"], []])


class ContextTests(unittest.TestCase):
    def test_the_last_assistant_text_and_recent_tool_calls_are_attached(self):
        lines = [KICKOFF, said("I'll open the PR"),
                 tool_call("Bash", {"command": "gh pr create --fill"}),
                 tool_output("https://github.com/x/y/pull/1"),
                 typed("I said draft", "u1", 1)]
        turn = extract(lines)[0]

        self.assertEqual(turn.assistant_before, "I'll open the PR")
        self.assertEqual(turn.tools_before, ['Bash: {"command": "gh pr create --fill"}'])

    def test_context_resets_after_each_typed_turn(self):
        lines = [KICKOFF, said("first answer"), typed("ok", "u1", 1), typed("and?", "u2", 2)]
        turns = extract(lines)

        self.assertEqual(turns[1].assistant_before, "")

    def test_long_text_is_truncated_from_the_front_for_context(self):
        lines = [KICKOFF, said("x" * 5000 + "TAIL"), typed("hm", "u1", 1)]

        self.assertTrue(extract(lines)[0].assistant_before.endswith("TAIL"))
        self.assertLessEqual(len(extract(lines)[0].assistant_before), friction.CONTEXT_CHARS + 1)


class BatchTests(unittest.TestCase):
    def test_items_are_chunked_into_batches_of_the_given_size(self):
        self.assertEqual([len(batch) for batch in friction.batches(list(range(45)), 20)], [20, 20, 5])


class CacheTests(unittest.TestCase):
    """Labels are cached per turn so a weekly rerun only pays for new turns."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "cache.jsonl"

    def tearDown(self):
        self.directory.cleanup()

    def test_saved_labels_load_back_by_key(self):
        friction.save_labels(self.path, {"sess:u1": {"type": "correction", "severity": 2}})

        self.assertEqual(friction.load_labels(self.path)["sess:u1"]["type"], "correction")

    def test_first_save_creates_the_cache_directory(self):
        path = Path(self.directory.name) / "fresh" / "cache.jsonl"

        friction.save_labels(path, {"sess:u1": {"type": "none"}})

        self.assertEqual(friction.load_labels(path)["sess:u1"]["type"], "none")

    def test_labels_from_an_older_prompt_version_are_ignored(self):
        self.path.write_text(json.dumps({"key": "sess:u1", "version": -1, "label": {"type": "none"}}) + "\n")

        self.assertEqual(friction.load_labels(self.path), {})

    def test_a_missing_cache_is_empty(self):
        self.assertEqual(friction.load_labels(self.path), {})


class VersionedCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "cache.jsonl"

    def tearDown(self):
        self.directory.cleanup()

    def test_verdicts_and_labels_share_a_file_without_colliding(self):
        friction.save_labels(self.path, {"sess:u1": {"type": "correction"}})
        friction.save_labels(self.path, {"sess:u1": {"is_friction": False}}, version=friction.VERIFY_VERSION)

        self.assertEqual(friction.load_labels(self.path)["sess:u1"], {"type": "correction"})
        self.assertEqual(friction.load_labels(self.path, version=friction.VERIFY_VERSION)["sess:u1"],
                         {"is_friction": False})


def labelled(uuid, kind="correction", severity=2):
    turn = friction.Turn(key="sess:" + uuid, project="proj", session="sess", timestamp=stamp(0),
                         text="message " + uuid, assistant_before="agent said", tools_before=["Bash: {}"])
    return turn, {"type": kind, "severity": severity, "agent_mistake": "did x", "rule": ""}


class VerifyTests(unittest.TestCase):
    """Haiku is the recall filter; Sonnet re-judges only what Haiku flagged, so the
    headline friction rate is not inflated by pushback labelled as correction."""

    def test_the_prompt_carries_the_context_and_the_first_label(self):
        prompt = json.loads(friction.verify_prompt([labelled("u1")]))
        item = prompt["items"][0]

        self.assertEqual(item["user_message"], "message u1")
        self.assertEqual(item["agent_said"], "agent said")
        self.assertEqual(item["first_label"], {"type": "correction", "severity": 2, "agent_mistake": "did x"})

    def test_the_verify_prompt_carries_less_context_than_labelling(self):
        turn, label = labelled("u1")
        turn.text = "m" * 5000
        turn.assistant_before = "a" * 5000
        turn.tools_before = ["Bash: 1", "Bash: 2", "Bash: 3", "Bash: 4"]
        item = json.loads(friction.verify_prompt([(turn, label)]))["items"][0]

        self.assertEqual(len(item["user_message"]), friction.VERIFY_TEXT_CHARS)
        self.assertLessEqual(len(item["agent_said"]), friction.VERIFY_CONTEXT_CHARS + 1)
        self.assertEqual(item["agent_tool_calls"], ["Bash: 3", "Bash: 4"])

    def test_a_rejected_turn_is_dropped(self):
        kept, unverified = friction.apply_verdicts(
            [labelled("u1"), labelled("u2")],
            {"sess:u1": {"is_friction": False, "type": "none", "severity": 0, "reason": "pushback"},
             "sess:u2": {"is_friction": True, "type": "correction", "severity": 2, "reason": "ok"}})

        self.assertEqual([turn.key for turn, _ in kept], ["sess:u2"])
        self.assertEqual(unverified, 0)

    def test_a_confirmed_turn_takes_the_verified_type_and_severity(self):
        kept, _ = friction.apply_verdicts(
            [labelled("u1", "correction", 1)],
            {"sess:u1": {"is_friction": True, "type": "repeated_rule", "severity": 3, "reason": "r"}})

        self.assertEqual(kept[0][1]["type"], "repeated_rule")
        self.assertEqual(kept[0][1]["severity"], 3)
        self.assertEqual(kept[0][1]["agent_mistake"], "did x")

    def test_a_turn_without_a_verdict_keeps_its_first_label_and_is_counted(self):
        kept, unverified = friction.apply_verdicts([labelled("u1")], {})

        self.assertEqual(kept[0][1]["type"], "correction")
        self.assertEqual(unverified, 1)

    def test_a_verdict_of_a_non_friction_type_drops_the_turn(self):
        kept, _ = friction.apply_verdicts(
            [labelled("u1")], {"sess:u1": {"is_friction": True, "type": "automated", "severity": 0, "reason": ""}})

        self.assertEqual(kept, [])


class ClaudeCallTests(unittest.TestCase):
    """The headless call must not load the user's settings, hooks, CLAUDE.md or MCP
    servers (6.8k -> 0.6k token baseline) and must not write a transcript, or the
    miner would read its own labelling runs next week."""

    def test_the_command_strips_config_and_persistence(self):
        command = friction.claude_command("claude-haiku-5-5", "system", {"type": "object"})

        self.assertEqual(command[:2], ["claude", "-p"])
        for flag, value in (("--model", "claude-haiku-5-5"), ("--setting-sources", ""), ("--tools", "")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--no-session-persistence", command)
        self.assertIn("--strict-mcp-config", command)

    def test_effort_is_passed_only_when_given(self):
        self.assertNotIn("--effort", friction.claude_command("m", "s", {}))
        command = friction.claude_command("m", "s", {}, effort="low")

        self.assertEqual(command[command.index("--effort") + 1], "low")

    def test_structured_output_and_cost_are_returned(self):
        envelope = {"is_error": False, "structured_output": {"labels": []}, "total_cost_usd": 0.002}

        def fake_runner(command, prompt):
            return json.dumps(envelope)

        result, cost = friction.run_claude("claude-haiku-5-5", "system", {}, "prompt", runner=fake_runner)

        self.assertEqual(result, {"labels": []})
        self.assertEqual(cost, 0.002)

    def test_cost_is_priced_from_token_usage_not_the_cli_estimate(self):
        envelope = {"is_error": False, "structured_output": {"labels": []}, "total_cost_usd": 0.0915,
                    "modelUsage": {"claude-haiku-5-5": {
                        "inputTokens": 11318, "outputTokens": 1898, "cacheReadInputTokens": 0,
                        "cacheCreationInputTokens": 0, "costUSD": 0.0832, "costBasis": "unknown"}}}

        def fake_runner(command, prompt):
            return json.dumps(envelope)

        _, cost = friction.run_claude("claude-haiku-5-5", "system", {}, "prompt", runner=fake_runner)

        self.assertAlmostEqual(cost, 11318 * 0.10 / 1e6 + 1898 * 0.50 / 1e6)

    def test_a_model_missing_from_the_rate_table_keeps_the_cli_cost(self):
        envelope = {"is_error": False, "structured_output": {"labels": []}, "total_cost_usd": 0.05,
                    "modelUsage": {
                        "claude-haiku-4-5-20251001": {"inputTokens": 8215, "outputTokens": 17,
                                                      "costUSD": 0.0083},
                        "claude-new-model": {"inputTokens": 100, "outputTokens": 10, "costUSD": 0.04}}}

        def fake_runner(command, prompt):
            return json.dumps(envelope)

        _, cost = friction.run_claude("claude-haiku-5-5", "system", {}, "prompt", runner=fake_runner)

        self.assertAlmostEqual(cost, 8215 * 1.0 / 1e6 + 17 * 5.0 / 1e6 + 0.04)

    def test_an_error_envelope_raises(self):
        def fake_runner(command, prompt):
            return json.dumps({"is_error": True, "result": "rate limited"})

        with self.assertRaises(friction.ClaudeError):
            friction.run_claude("claude-haiku-5-5", "system", {}, "prompt", runner=fake_runner)


class RankingTests(unittest.TestCase):
    def test_causes_rank_by_total_severity_of_their_turns(self):
        severity = {"a": 3, "b": 3, "c": 1, "d": 1, "e": 1}
        causes = [{"title": "many small", "turn_ids": ["c", "d", "e"]},
                  {"title": "few severe", "turn_ids": ["a", "b"]}]

        ranked = friction.rank_causes(causes, severity)

        self.assertEqual([cause["title"] for cause in ranked], ["few severe", "many small"])
        self.assertEqual(ranked[0]["score"], 6)

    def test_unknown_turn_ids_from_the_model_are_dropped(self):
        ranked = friction.rank_causes([{"title": "x", "turn_ids": ["a", "ghost"]}], {"a": 2})

        self.assertEqual(ranked[0]["turn_ids"], ["a"])


class ReportTests(unittest.TestCase):
    def test_transcript_text_is_html_escaped(self):
        turn = friction.Turn(key="sess:u1", project="proj", session="sess", timestamp=stamp(0),
                             text="<script>alert(1)</script>")
        label = {"type": "correction", "severity": 2, "agent_mistake": "x", "rule": ""}
        html = friction.render_html({"window_days": 7, "sessions": 1, "turns": 1, "spend_usd": 0.0},
                                    [], [(turn, label)])

        self.assertNotIn("<script>alert(1)", html)
        self.assertIn("&lt;script&gt;", html)


if __name__ == "__main__":
    unittest.main()
