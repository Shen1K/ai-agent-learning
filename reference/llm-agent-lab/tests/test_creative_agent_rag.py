"""RAG检索是真实本地计算；Agent决策与评审使用脚本，不发网络请求。"""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from creative_agent.contracts import InvalidInput, example_task
from creative_agent.evaluation import validate_judgment
from creative_agent.prompts import judge_messages
from creative_agent.providers import MockProvider, RagMockProvider, ProviderFailure, action
from creative_agent.rag import (KnowledgeBase, candidate_context, chunks, required_context,
                                search, sections, MAX_SEARCHES)
from creative_agent.rag_eval import evaluate
from creative_agent.runtime import AgentRuntime
from creative_agent.store import Store


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase.bundled()
        self.snapshot = self.kb.snapshot("cup-001")

    def test_scope_and_active_version_filtered_before_indexing(self):
        self.assertTrue(all(d["product_id"] == "cup-001" and d["status"] == "active"
                            for d in self.snapshot["documents"]))
        for query in ("紫电独家", "狂欢冲销量"):
            self.assertEqual(search(self.snapshot, query)["hits"], [])

    def test_no_match_does_not_fabricate_evidence(self):
        result = search(self.snapshot, "核聚变发动机")
        self.assertEqual(result["status"], "no_match")
        self.assertEqual(result["hits"], [])

    def test_parent_restores_exception_missing_from_small_chunk(self):
        query = "幽默 俏皮表达"
        small = search(self.snapshot, query, strategy="small_chunks", top_k=1)["hits"][0]
        parent = search(self.snapshot, query, strategy="parent_context", top_k=1)["hits"][0]
        self.assertEqual(small["source_id"], parent["source_id"])
        self.assertNotIn("不得暗示", small["text"])
        self.assertIn("不得暗示", parent["text"])
        self.assertGreater(len(parent["text"]), len(small["text"]))

    def test_source_offsets_point_to_same_version_original(self):
        originals = {s["source_id"]: s["text"] for s in sections(self.snapshot)}
        for item in chunks(self.snapshot):
            self.assertEqual(item["text"], item["parent"]["text"][item["start"]:item["end"]])
        for strategy in ("small_chunks", "parent_context"):
            for hit in search(self.snapshot, "办公 捧杯 休息", strategy=strategy)["hits"]:
                self.assertEqual(hit["text"], originals[hit["source_id"]][hit["start"]:hit["end"]])

    def test_return_budget_skips_whole_evidence_instead_of_cutting_condition(self):
        result = search(self.snapshot, "幽默", max_chars=1)
        self.assertEqual(result["hits"], [])
        self.assertGreater(result["budget_skipped"], 0)

    def test_required_policies_present_even_without_search(self):
        context = candidate_context({"snapshot": self.snapshot, "retrievals": []})
        self.assertEqual(len(context), len(required_context(self.snapshot)))
        self.assertTrue(all(x["kind"] == "policy" for x in context.values()))

    def test_duplicate_active_version_rejected_not_guessed(self):
        docs = deepcopy(self.kb.documents)
        docs[1]["status"] = "active"
        with self.assertRaises(InvalidInput):
            KnowledgeBase(docs)

    def test_snapshot_is_not_changed_by_external_mutation(self):
        self.kb.documents[0]["sections"][0]["text"] = "新规范"
        self.assertNotIn("新规范", json.dumps(self.snapshot, ensure_ascii=False))
        self.assertNotEqual(self.kb.snapshot("cup-001")["snapshot_id"], self.snapshot["snapshot_id"])

    def test_invalid_search_options_and_unknown_product(self):
        for options in ({"top_k": True}, {"top_k": 100}, {"max_chars": 0}, {"strategy": "vector_fake"}):
            with self.assertRaises(InvalidInput):
                search(self.snapshot, "办公", **options)
        with self.assertRaises(InvalidInput):
            search(self.snapshot, "x" * 241)
        with self.assertRaises(InvalidInput):
            self.kb.snapshot("unknown")

    def test_eval_preserves_semantic_failure_and_measures_coverage_not_creativity(self):
        reports = evaluate()["reports"]
        self.assertGreater(reports[1]["case_passes"], reports[0]["case_passes"])
        self.assertGreater(reports[1]["total_returned_chars"], reports[0]["total_returned_chars"])
        for report in reports:
            gap = next(x for x in report["results"] if x["case_id"] == "semantic_gap")
            self.assertFalse(gap["passed"])


class RagRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "runs.sqlite3"
        self.store = Store(self.path)
        self.kb = KnowledgeBase.bundled()
        self.runtime = AgentRuntime(self.store, RagMockProvider(), self.kb)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_agent_search_result_saved_and_frozen_into_candidate_and_judge(self):
        state, _ = self.runtime.create(example_task())
        state = self.runtime.run(state["id"])
        self.assertEqual(state["status"], "awaiting_review")
        self.assertEqual(len(state["rag"]["retrievals"]), 1)
        candidate = state["candidates"][0]
        context = candidate["knowledge_context"]
        self.assertTrue(any(x["document_id"] == "case-pause" for x in context.values()))
        raw = json.dumps(judge_messages(state["task"], candidate), ensure_ascii=False)
        self.assertNotIn("matched_chunk_id", raw)
        self.assertNotIn("hypothesis", raw)
        self.assertNotIn("其他商品的模拟隔离资料", raw)
        self.assertIn("knowledge_context", raw)
        self.assertTrue(state["evaluations"]["c1"]["simulated"])
        self.assertFalse(state["publication_authorized"])
        events = self.store.events(state["id"])
        self.assertTrue(any(e["kind"] == "tool_finished" and e["payload"]["tool"] == "search_knowledge"
                            for e in events))

    def test_no_rag_tool_advertised_in_original_mode(self):
        class Inspect(MockProvider):
            def decide(inner, messages, tools):
                self.assertNotIn("search_knowledge", [x["function"]["name"] for x in tools])
                return super().decide(messages, tools)
        runtime = AgentRuntime(self.store, Inspect())
        state, _ = runtime.create(example_task())
        state = runtime.run(state["id"])
        self.assertNotIn("rag", state)
        self.assertEqual(state["status"], "awaiting_review")

    def test_search_cannot_override_scope_or_run_when_disabled(self):
        state, _ = self.runtime.create(example_task())
        self.runtime._execute(state, action("get_materials"))
        with self.assertRaises(InvalidInput):
            self.runtime._execute(state, action("search_knowledge", query="办公", product_id="cup-999"))
        plain, _ = AgentRuntime(self.store, MockProvider()).create(example_task())
        plain["materials_read"] = True
        with self.assertRaises(InvalidInput):
            self.runtime._execute(plain, action("search_knowledge", query="办公"))

    def test_retrieval_limit_is_enforced(self):
        state, _ = self.runtime.create(example_task())
        self.runtime._execute(state, action("get_materials"))
        for _ in range(MAX_SEARCHES):
            self.runtime._execute(state, action("search_knowledge", query="办公"))
        with self.assertRaises(InvalidInput):
            self.runtime._execute(state, action("search_knowledge", query="办公"))

    def test_idempotency_includes_knowledge_snapshot(self):
        self.runtime.create(example_task(), request_key="rag-key")
        self.kb.documents[0]["sections"][0]["text"] = "新的品牌规范"
        with self.assertRaises(InvalidInput):
            self.runtime.create(example_task(), request_key="rag-key")

    def test_restart_uses_persisted_corpus_not_mutated_catalog(self):
        state, _ = self.runtime.create(example_task())
        original_id = state["rag"]["snapshot"]["snapshot_id"]
        self.store.close()
        self.store = Store(self.path)
        self.kb.documents[0]["sections"][0]["text"] = "外部修改不得进入旧任务"
        runtime = AgentRuntime(self.store, RagMockProvider(), self.kb)
        state = runtime.run(state["id"])
        self.assertEqual(state["status"], "awaiting_review")
        self.assertEqual(state["rag"]["snapshot"]["snapshot_id"], original_id)
        self.assertNotIn("外部修改不得进入旧任务", json.dumps(state, ensure_ascii=False))

    def test_knowledge_quote_is_checked_and_cannot_replace_product_fact(self):
        state, _ = self.runtime.create(example_task())
        state = self.runtime.run(state["id"])
        candidate = state["candidates"][0]
        judgment, _ = RagMockProvider().judge(judge_messages(state["task"], candidate))
        validate_judgment(judgment, state["task"], candidate)
        tampered = deepcopy(judgment)
        tampered["checks"]["tone_fit"]["evidence"][-1]["quote"] = "编造的资料引用"
        with self.assertRaises(InvalidInput):
            validate_judgment(tampered, state["task"], candidate)
        tampered = deepcopy(judgment)
        tampered["checks"]["facts_supported"]["evidence"] = deepcopy(judgment["checks"]["tone_fit"]["evidence"])
        with self.assertRaises(InvalidInput):
            validate_judgment(tampered, state["task"], candidate)

    def test_failed_judge_resume_does_not_repeat_search_or_change_evidence(self):
        class FailOnce(RagMockProvider):
            failed = False
            def judge(inner, messages):
                if not inner.failed:
                    inner.failed = True
                    raise ProviderFailure("模拟评审超时")
                return super().judge(messages)
        self.runtime.provider = FailOnce()
        state, _ = self.runtime.create(example_task())
        first = self.runtime.run(state["id"])
        self.assertEqual(first["status"], "error")
        before = deepcopy(first["candidates"])
        self.runtime.resume(state["id"])
        final = self.runtime.run(state["id"])
        self.assertEqual(final["status"], "awaiting_review")
        self.assertEqual(final["candidates"], before)
        self.assertEqual(len(final["rag"]["retrievals"]), 1)


if __name__ == "__main__":
    unittest.main()
