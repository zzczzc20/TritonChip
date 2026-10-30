"""Each CLI's own usage report, read into one token schema: nothing counted twice, nothing invented."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))

from tritonchip_agent_harness.adapters import parse, tokens  # noqa: E402
from tritonchip_agent_harness.siteconfig import Account  # noqa: E402


def account(cli: str) -> Account:
    return Account(label=cli, cli=cli, auth="subscription", credential=Path("/nonexistent"), hosts=("h.example",),
                   models={"main": "m"}, options={})


def lines(*events) -> str:
    return "\n".join(json.dumps(e) for e in events) + "\n"


# Claude Code's stream-json: the final `result` event. `usage` is the top-level loop's (the pilot-2 numbers
# of README section 19); `modelUsage` adds what a background model or subagent spent.
CLAUDE_RESULT = {
    "type": "result", "subtype": "success", "is_error": False, "num_turns": 60, "result": "Done.",
    "total_cost_usd": 1.36, "session_id": "s1",
    "usage": {"input_tokens": 120, "cache_creation_input_tokens": 91519, "cache_read_input_tokens": 3868404,
              "output_tokens": 22455, "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
              "service_tier": "standard",
              "cache_creation": {"ephemeral_1h_input_tokens": 91519, "ephemeral_5m_input_tokens": 0},
              "output_tokens_details": {"thinking_tokens": 6120}},
    "modelUsage": {
        "claude-sonnet-5-5": {"inputTokens": 120, "outputTokens": 22455, "thinkingTokens": 6120,
                              "cacheReadInputTokens": 3868404, "cacheCreationInputTokens": 91519,
                              "webSearchRequests": 0, "costUSD": 1.31, "contextWindow": 200000,
                              "maxOutputTokens": 32000},
        "claude-haiku-5-5": {"inputTokens": 2210, "outputTokens": 380, "cacheReadInputTokens": 0,
                             "cacheCreationInputTokens": 0, "webSearchRequests": 0, "costUSD": 0.05,
                             "contextWindow": 200000, "maxOutputTokens": 8192}},
}


class ClaudeCode(unittest.TestCase):
    def test_totals_are_the_models_sum_subagents_included(self):
        r = parse(account("claude-code"), lines({"type": "system", "subtype": "init", "model": "claude-sonnet-5-5"},
                                                CLAUDE_RESULT), Path(tempfile.mkdtemp()))
        t = r["tokens"]
        self.assertEqual(r["usage"], CLAUDE_RESULT["usage"], "the raw usage is kept as reported")
        self.assertEqual((t["input_uncached"], t["cache_write"], t["cache_read"], t["output"], t["reasoning"]),
                         (120 + 2210, 91519, 3868404, 22455 + 380, 6120))
        self.assertEqual(t["total"], 2330 + 91519 + 3868404 + 22835, "thinking is inside output, not added again")
        self.assertEqual(t["by_model"]["claude-sonnet-5-5"]["total"], 120 + 91519 + 3868404 + 22455)
        # The background model reports no thinking count: None, not 0.
        self.assertIsNone(t["by_model"]["claude-haiku-5-5"]["reasoning"])

    def test_without_model_usage_the_result_usage_is_read(self):
        final = {k: v for k, v in CLAUDE_RESULT.items() if k != "modelUsage"}
        final["usage"] = {k: v for k, v in final["usage"].items() if k != "output_tokens_details"}
        t = parse(account("claude-code"), lines(final), Path(tempfile.mkdtemp()))["tokens"]
        self.assertEqual(t, {"input_uncached": 120, "cache_write": 91519, "cache_read": 3868404, "output": 22455,
                             "reasoning": None, "total": 3982498, "by_model": None})

    def test_no_result_reports_nothing(self):
        t = parse(account("claude-code"), "not json\n", Path(tempfile.mkdtemp()))["tokens"]
        self.assertEqual(set(t.values()), {None})


class Codex(unittest.TestCase):
    def setUp(self):
        # Where `codex exec` keeps its rollout, which names the model that served the run.
        self.run_dir = Path(tempfile.mkdtemp())
        sessions = self.run_dir / "codex-home" / "sessions" / "2026" / "10" / "16"
        sessions.mkdir(parents=True)
        (sessions / "rollout-2026-10-16T10-00-00-1.jsonl").write_text(
            lines({"type": "turn_context", "payload": {"model": "gpt-5.6-luna", "cwd": "/work"}}))

    def events(self, usage) -> str:
        return lines({"type": "thread.started", "thread_id": "t1"}, {"type": "turn.started"},
                     {"type": "item.completed", "item": {"id": "item_0", "type": "reasoning", "text": "Plan."}},
                     {"type": "item.completed", "item": {"id": "item_1", "type": "agent_message", "text": "Done."}},
                     *([{"type": "turn.completed", "usage": usage}] if usage else []))

    def test_input_holds_the_cached_tokens_and_output_the_reasoning(self):
        # Pilot 2's Codex run (README section 19): 1,488,541 input, 1,418,496 of them cached; 7,800 output,
        # 2,071 of them reasoning.
        usage = {"input_tokens": 1488541, "cached_input_tokens": 1418496, "cache_write_input_tokens": 0,
                 "output_tokens": 7800, "reasoning_output_tokens": 2071}
        r = parse(account("codex"), self.events(usage), self.run_dir)
        self.assertEqual((r["text"], r["served_models"], r["usage"]), ("Done.", ["gpt-5.6-luna"], usage))
        self.assertEqual(r["tokens"], {"input_uncached": 70045, "cache_write": 0, "cache_read": 1418496,
                                       "output": 7800, "reasoning": 2071, "total": 1488541 + 7800,
                                       "by_model": None})

    def test_cache_writes_come_out_of_the_input_too(self):
        # codex-api's own fixture: 100 input = 40 read from the cache + 60 written to it.
        t = tokens("codex", {"input_tokens": 100, "cached_input_tokens": 40, "cache_write_input_tokens": 60,
                             "output_tokens": 10, "reasoning_output_tokens": 5})
        self.assertEqual((t["input_uncached"], t["cache_write"], t["cache_read"], t["total"]), (0, 60, 40, 110))

    def test_fields_an_older_exec_did_not_print_are_none(self):
        t = parse(account("codex"), self.events({"input_tokens": 1000, "cached_input_tokens": 600,
                                                  "output_tokens": 50}), self.run_dir)["tokens"]
        self.assertEqual((t["input_uncached"], t["cache_write"], t["reasoning"], t["total"]), (400, None, None, 1050))

    def test_a_run_stopped_before_its_turn_completed_reports_nothing(self):
        t = parse(account("codex"), self.events(None), self.run_dir)["tokens"]
        self.assertEqual(set(t.values()), {None})


def step_finish(n: int, **tokens_) -> dict:
    part = {"id": f"prt_{n}", "sessionID": "ses_1", "messageID": f"msg_{n}", "type": "step-finish",
            "reason": "tool-calls", "tokens": tokens_, "cost": 0}
    return {"type": "step_finish", "timestamp": 1761000000000 + n, "sessionID": "ses_1", "part": part}


class Opencode(unittest.TestCase):
    def test_steps_add_up_and_reasoning_goes_back_into_output(self):
        # opencode's `output` leaves the reasoning out and its `input` the cache; `total` is the provider's own.
        out = lines({"type": "step_start", "timestamp": 1761000000000, "sessionID": "ses_1",
                     "part": {"id": "prt_0", "type": "step-start"}},
                    step_finish(1, total=12550, input=12000, output=300, reasoning=250, cache={"read": 0, "write": 0}),
                    {"type": "text", "timestamp": 1761000000002, "sessionID": "ses_1",
                     "part": {"id": "prt_2", "type": "text", "text": "Done.", "time": {"start": 1, "end": 2}}},
                    step_finish(3, total=14600, input=500, output=120, reasoning=0, cache={"read": 13980, "write": 0}))
        r = parse(account("opencode"), out, Path(tempfile.mkdtemp()))
        self.assertEqual(r["usage"], {"total": 27150, "input": 12500, "output": 420, "reasoning": 250,
                                      "cache.read": 13980, "cache.write": 0})
        self.assertEqual(r["tokens"], {"input_uncached": 12500, "cache_write": 0, "cache_read": 13980, "output": 670,
                                       "reasoning": 250, "total": 27150, "by_model": None})

    def test_a_run_stopped_before_any_step_finished_reports_nothing(self):
        out = lines({"type": "step_start", "timestamp": 1761000000000, "sessionID": "ses_1",
                     "part": {"id": "prt_0", "type": "step-start"}})
        t = parse(account("opencode"), out, Path(tempfile.mkdtemp()))["tokens"]
        self.assertEqual(set(t.values()), {None})


if __name__ == "__main__":
    unittest.main()
