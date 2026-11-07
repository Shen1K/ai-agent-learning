"""Agent控制流测试。全部使用离线后端或假网络响应，绝不真实调用DeepSeek。"""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from creative_agent.contracts import InvalidInput, decode, example_task, validate_task, validate_action
from creative_agent.evaluation import hard_checks
from creative_agent.prompts import judge_messages
from creative_agent.providers import DeepSeekProvider, MockProvider, ProviderFailure, action
from creative_agent.runtime import AgentRuntime
from creative_agent.store import Store


class Scripted(MockProvider):
    """测试可以换决策顺序，用来证明runtime不是写死的业务流水线。"""
    def __init__(self, actions):
        self.actions = iter(actions)

    def decide(self, messages, tools):
        return next(self.actions), {}


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "runs.sqlite3"
        self.store = Store(self.path)
        self.runtime = AgentRuntime(self.store, MockProvider())

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def create(self, **kwargs):
        return self.runtime.create(example_task(), **kwargs)[0]["id"]

    def test_demo_revises_from_feedback_then_waits_for_human(self):
        state = self.runtime.run(self.create())
        self.assertEqual(state["status"], "awaiting_review")
        self.assertEqual(len(state["candidates"]), 2)
        self.assertEqual(state["candidates"][1]["parent_id"], "c1")
        self.assertFalse(state["evaluations"]["c1"]["eligible"])
        self.assertTrue(state["evaluations"]["c2"]["eligible"])
        self.assertTrue(state["evaluations"]["c2"]["simulated"])
        self.assertEqual(state["model_calls"], 7)
        self.assertFalse(state["publication_authorized"])

    def test_submit_without_review_is_rejected(self):
        self.runtime.provider = Scripted([
            action("get_materials"),
            action("save_candidate", headline="咖啡杯", body="给自己片刻休息。", direction="示例", hypothesis="示例", parent_id=""),
            action("submit_candidate", candidate_id="c1"),
            action("request_clarification", question="需要补充评审资料。")])
        state = self.runtime.run(self.create())
        self.assertEqual(state["status"], "needs_input")
        self.assertIsNone(state["selected_candidate_id"])
        self.assertTrue(any(not json.loads(m["content"])["ok"] for m in state["messages"] if m["role"] == "tool"))

    def test_unknown_tool_never_executes(self):
        self.runtime.provider = Scripted([action("publish_ad", budget="100000"),
                                          action("request_clarification", question="请确认任务。")])
        state = self.runtime.run(self.create())
        self.assertEqual(state["status"], "needs_input")
        self.assertIn("工具不在允许名单", state["messages"][3]["content"])

    def test_model_can_pause_before_creating_any_candidate(self):
        self.runtime.provider = Scripted([action("request_clarification", question="需要哪种受众？")])
        state = self.runtime.run(self.create())
        self.assertEqual(state["status"], "needs_input")
        self.assertEqual(state["candidates"], [])
        answered = self.runtime.review(state["id"], "answer", "办公人群。")
        self.assertEqual(answered["status"], "ready")

    def test_model_call_budget_includes_failures(self):
        class Failing(MockProvider):
            def decide(self, messages, tools):
                raise ProviderFailure("网络超时，处理状态未知。")
        self.runtime.provider = Failing()
        run_id = self.create(limits={"max_steps": 5, "max_model_calls": 1, "max_candidates": 2})
        state = self.runtime.run(run_id)
        self.assertEqual(state["status"], "error")
        self.assertEqual(state["model_calls"], 1)
        self.runtime.resume(run_id)
        self.assertEqual(self.runtime.run(run_id)["status"], "budget_exhausted")

    def test_judge_failure_resume_preserves_candidate_and_pending_tool(self):
        class FailOnce(MockProvider):
            def __init__(self):
                self.failed = False
            def judge(self, messages):
                if not self.failed:
                    self.failed = True
                    raise ProviderFailure("模拟评审超时")
                return super().judge(messages)
        self.runtime.provider = FailOnce()
        run_id = self.create()
        first = self.runtime.run(run_id)
        self.assertEqual(first["status"], "error")
        self.assertEqual(first["pending_action"]["name"], "evaluate_candidate")
        self.assertNotIn("c2", first["evaluations"])
        before = deepcopy(first["candidates"])
        self.runtime.resume(run_id)
        second = self.runtime.run(run_id)
        self.assertEqual(second["status"], "awaiting_review")
        self.assertEqual(second["candidates"], before)
        self.assertEqual(second["model_calls"], 8)

    def test_false_quote_is_error_not_low_score(self):
        class BadJudge(MockProvider):
            def judge(self, messages):
                value, stats = super().judge(messages)
                value["checks"]["brief_fit"]["evidence"][0]["quote"] = "伪造的原文"
                return value, stats
        self.runtime.provider = BadJudge()
        state = self.runtime.run(self.create())
        self.assertEqual(state["status"], "error")
        self.assertNotIn("c2", state["evaluations"])

    def test_unknown_judgment_cannot_submit(self):
        class Uncertain(MockProvider):
            def judge(self, messages):
                result, stats = super().judge(messages)
                result["checks"]["facts_supported"]["passed"] = None
                return result, stats
        self.runtime.provider = Uncertain()
        state = self.runtime.run(self.create(limits={"max_steps": 16, "max_model_calls": 20, "max_candidates": 2}))
        self.assertEqual(state["status"], "budget_exhausted")
        self.assertFalse(state["evaluations"]["c2"]["eligible"])

    def test_creation_idempotency_and_conflict(self):
        a, created = self.runtime.create(example_task(), request_key="request-1")
        b, created_again = self.runtime.create(example_task(), request_key="request-1")
        self.assertTrue(created)
        self.assertFalse(created_again)
        self.assertEqual(a["id"], b["id"])
        task = example_task()
        task["brief"]["goal"] = "不同任务"
        with self.assertRaises(InvalidInput):
            self.runtime.create(task, request_key="request-1")

    def test_judge_context_excludes_author_hypothesis_and_previous_scores(self):
        candidate = {"headline": "咖啡杯", "body": "给自己一杯的时间。",
                     "hypothesis": "秘密生成理由", "direction": "作者自评满分", "id": "c1"}
        messages = judge_messages(example_task(), candidate)
        self.assertNotIn("秘密生成理由", json.dumps(messages, ensure_ascii=False))
        self.assertNotIn("作者自评满分", json.dumps(messages, ensure_ascii=False))
        data = json.loads(messages[1]["content"])
        self.assertEqual(set(data["candidate"]), {"headline", "body"})

    def test_judge_receives_user_feedback_but_not_generation_history(self):
        messages = judge_messages(example_task(), {"headline": "咖啡杯", "body": "正文"}, ["不要出现命令语气"])
        self.assertEqual(json.loads(messages[1]["content"])["user_feedback"], ["不要出现命令语气"])

    def test_same_round_duplicate_draft_does_not_get_new_id(self):
        candidate = {"headline": "咖啡杯", "body": "给自己片刻休息。", "direction": "示例",
                     "hypothesis": "示例", "parent_id": ""}
        self.runtime.provider = Scripted([action("get_materials"), action("save_candidate", **candidate),
                                          action("save_candidate", **candidate),
                                          action("request_clarification", question="请补充新方向。")])
        state = self.runtime.run(self.create())
        self.assertEqual(len(state["candidates"]), 1)
        self.assertEqual(state["status"], "needs_input")

    def test_approval_only_after_submission_and_never_publishes(self):
        run_id = self.create()
        with self.assertRaises(InvalidInput):
            self.runtime.review(run_id, "approve")
        self.runtime.run(run_id)
        state = self.runtime.review(run_id, "approve")
        self.assertEqual(state["status"], "approved")
        self.assertFalse(state["publication_authorized"])

    def test_human_revision_requires_new_candidate(self):
        run_id = self.create()
        self.runtime.run(run_id)
        self.runtime.review(run_id, "revise", "请再给一个更自然的版本。")
        state = self.runtime.run(run_id)
        self.assertEqual(state["status"], "awaiting_review")
        self.assertEqual(state["selected_candidate_id"], "c3")
        self.assertEqual(state["candidates"][2]["feedback_round"], 1)

    def test_cancelled_run_makes_no_calls(self):
        run_id = self.create()
        self.runtime.cancel(run_id)
        state = self.runtime.run(run_id)
        self.assertEqual(state["status"], "cancelled")
        self.assertEqual(state["model_calls"], 0)
        with self.assertRaises(InvalidInput):
            self.runtime.resume(run_id)

    def test_cancel_during_judge_does_not_submit(self):
        runtime = self.runtime
        class CancelDuringJudge(MockProvider):
            def judge(self, messages):
                runtime.cancel(run_id)
                return super().judge(messages)
        self.runtime.provider = CancelDuringJudge()
        run_id = self.create()
        state = self.runtime.run(run_id)
        self.assertEqual(state["status"], "cancelled")
        self.assertIsNone(state["selected_candidate_id"])

    def test_restart_preserves_pending_action_and_requires_manual_resume(self):
        run_id = self.create()
        state = self.store.get(run_id)
        state.update(status="running", pending_action=action("get_materials"))
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": [{
            "id": state["pending_action"]["id"], "type": "function", "function": {
                "name": "get_materials", "arguments": "{}"}}]})
        self.store.save(state, "test_interruption")
        self.store.close()
        self.store = Store(self.path)
        self.runtime = AgentRuntime(self.store, MockProvider())
        self.assertEqual(self.store.get(run_id)["status"], "error")
        self.runtime.resume(run_id)
        self.assertEqual(self.runtime.run(run_id)["status"], "awaiting_review")

    def test_only_one_process_can_own_database(self):
        with self.assertRaises(InvalidInput):
            Store(self.path)

    def test_mutated_source_task_does_not_change_frozen_run(self):
        task = example_task()
        state, _ = self.runtime.create(task)
        task["product"]["facts"]["F1"] = "已被外部修改"
        self.assertNotEqual(self.store.get(state["id"])["task"]["product"]["facts"]["F1"], "已被外部修改")

    def test_invalid_input_types_duplicate_keys_and_extra_tool_args(self):
        task = example_task()
        task["constraints"]["headline_max_chars"] = True
        with self.assertRaises(InvalidInput):
            validate_task(task)
        for raw in ('{"x":1,"x":2}', '{"x":NaN}'):
            with self.assertRaises(InvalidInput):
                decode(raw)
        with self.assertRaises(InvalidInput):
            validate_action(action("get_materials", override_rule="approve_all"))

    def test_hard_checks_count_characters_and_report_hits(self):
        task = example_task()
        c = {"headline": "咖啡杯！", "body": "保温十二小时"}
        result = hard_checks(task, c)
        self.assertEqual(result["headline_chars"], 3)
        self.assertFalse(result["passed"])


class ProviderTests(unittest.TestCase):
    def test_real_adapter_tool_protocol_is_mocked_no_network(self):
        envelope = {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [{
            "id": "call_test", "type": "function", "function": {"name": "get_materials", "arguments": "{}"}}]}}],
            "usage": {"total_tokens": 12}}
        with patch("creative_agent.providers.call_deepseek", return_value=(envelope, 0.1)) as transport:
            chosen, stats = DeepSeekProvider().decide([], [])
        self.assertEqual(chosen["name"], "get_materials")
        payload = transport.call_args.args[0]
        self.assertEqual(payload["tool_choice"], "required")
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(stats["usage"]["total_tokens"], 12)

    def test_invalid_tool_response_does_not_execute(self):
        for envelope in ({}, {"choices": [{"finish_reason": "length", "message": {}}]},
                         {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": []}}]}):
            with self.subTest(envelope=envelope), patch("creative_agent.providers.call_deepseek", return_value=(envelope, 0.1)):
                with self.assertRaises(ProviderFailure):
                    DeepSeekProvider().decide([], [])

    def test_missing_real_key_is_safe_and_offline(self):
        with patch.dict("os.environ", {}, clear=True), patch("lessons.creative_studio.deepseek_review.request.build_opener") as network:
            with self.assertRaises(ProviderFailure):
                DeepSeekProvider().decide([], [])
        network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
