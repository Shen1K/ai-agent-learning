"""验证 C11 读取、字段优先及受限文字提取。

用模拟文本替代磁盘内容，测试不会修改真实清单。
"""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from lessons.c11_evidence_flow import (
    answer_from_candidate, answer_question, extract_duration, read_evidence,
)


class ReadEvidenceTests(unittest.TestCase):
    def test_numeric_value(self):
        with patch.object(Path, "read_text", return_value='{"count": 895}'):
            result = read_evidence(Path("example.json"), "count")
        self.assertTrue(result["tool_ok"])
        self.assertEqual(result["value"], 895)
        self.assertEqual(result["field"], "count")

    def test_missing_field_is_not_read_failure(self):
        with patch.object(Path, "read_text", return_value='{"count": 895}'):
            result = read_evidence(Path("example.json"), "duration")
        self.assertTrue(result["tool_ok"])
        self.assertIsNone(result["value"])

    def test_zero_is_preserved(self):
        with patch.object(Path, "read_text", return_value='{"count": 0}'):
            result = read_evidence(Path("example.json"), "count")
        self.assertEqual(result["value"], 0)
        self.assertIsNotNone(result["value"])

    def test_missing_file_preserves_error(self):
        with patch.object(Path, "read_text", side_effect=FileNotFoundError("文件不存在")):
            result = read_evidence(Path("example.json"), "count")
        self.assertFalse(result["tool_ok"])
        self.assertEqual(result["error"], "文件不存在")

    def test_bad_json_and_wrong_root(self):
        for text in ("invalid json", "[]"):
            with self.subTest(text=text):
                with patch.object(Path, "read_text", return_value=text):
                    result = read_evidence(Path("example.json"), "count")
                self.assertFalse(result["tool_ok"])
                self.assertIsNotNone(result["error"])


class AnswerQuestionTests(unittest.TestCase):
    def ask_duration(self, data):
        # json.dumps 把课堂字典变成模拟文件内容；不产生真实文件。
        with patch.object(Path, "read_text", return_value=json.dumps(data)) as read:
            state = answer_question("训练耗时？", "training_duration_seconds", "秒",
                                    Path("教学示例.json"))
        read.assert_called_once()
        return state

    def test_field_answer(self):
        state = self.ask_duration({"training_duration_seconds": 90})
        self.assertEqual(state["status"], "answered")
        self.assertEqual(state["evidence"]["method"], "field")
        self.assertEqual(state["evidence"]["value"], 90)

    def test_text_answer_keeps_source_quote(self):
        state = self.ask_duration({"notes": "本次训练持续了90秒。"})
        self.assertEqual(state["status"], "answered")
        self.assertEqual(state["evidence"]["method"], "text_rule")
        self.assertEqual(state["evidence"]["value"], 90)
        self.assertEqual(state["evidence"]["quote"], "本次训练持续了90秒。")
        self.assertEqual(state["evidence"]["source"], "教学示例.json")
        self.assertEqual(len(state["history"]), 2)

    def test_no_evidence_does_not_invent_answer(self):
        state = self.ask_duration({"training_document_count": 895})
        self.assertEqual(state["status"], "insufficient_evidence")
        self.assertIsNone(state["evidence"])

    def test_zero_field_does_not_fall_through(self):
        state = self.ask_duration({"training_duration_seconds": 0})
        self.assertEqual(state["status"], "answered")
        self.assertEqual(state["evidence"]["value"], 0)
        self.assertEqual(len(state["history"]), 1)

    def test_null_field_allows_text_fallback(self):
        state = self.ask_duration({"training_duration_seconds": None,
                                   "notes": "本次训练持续了0秒。"})
        self.assertEqual(state["evidence"]["method"], "text_rule")
        self.assertEqual(state["evidence"]["value"], 0)

    def test_other_question_cannot_use_duration(self):
        with patch.object(Path, "read_text", return_value='{"notes":"本次训练持续了90秒。"}'):
            state = answer_question("文章数量？", "training_document_count", "篇")
        self.assertEqual(state["status"], "insufficient_evidence")

    def test_read_failure_stops_without_extraction(self):
        with patch.object(Path, "read_text", side_effect=FileNotFoundError("文件不存在")):
            state = answer_question("训练耗时？", "training_duration_seconds", "秒")
        self.assertEqual(state["status"], "tool_error")
        self.assertEqual(len(state["history"]), 1)
        self.assertIsNone(state["evidence"])

    def test_rule_does_not_search_inside_arbitrary_text(self):
        self.assertIsNone(extract_duration("举例：本次训练持续了90秒。"))

    def test_vocabulary_supports_old_and_new_phrase(self):
        for text in ("本次训练持续了90秒。", "本次训练耗时90秒。"):
            with self.subTest(text=text):
                state = self.ask_duration({"notes": text})
                self.assertEqual(state["status"], "answered")
                self.assertEqual(state["evidence"]["value"], 90)
                self.assertEqual(state["evidence"]["quote"], text)

    def test_keyword_alone_is_not_evidence(self):
        texts = ("尚未记录本次训练耗时。", "本次训练预计耗时90秒。",
                 "数据下载耗时90秒。", "本次训练耗时不超过90秒。")
        for text in texts:
            with self.subTest(text=text):
                state = self.ask_duration({"notes": text})
                self.assertEqual(state["status"], "insufficient_evidence")
                self.assertIsNone(state["evidence"])

    def test_new_phrase_preserves_zero(self):
        self.assertEqual(extract_duration("本次训练耗时0秒")["value"], 0)


class CandidateValidationTests(unittest.TestCase):
    def setUp(self):
        # 完全虚构的结构化来源，每个测试重新创建，不修改真实数据。
        self.records = {
            "a": {"run_id": "A", "phase": "training", "kind": "actual",
                  "unit": "seconds", "duration_seconds": 120,
                  "text": "任务A实际训练用了120秒。"},
            "b": {"run_id": "B", "phase": "training", "kind": "actual",
                  "unit": "seconds", "duration_seconds": 80,
                  "text": "任务B训练用了80秒。"},
        }
        self.candidate = {"run_id": "A", "record_id": "a",
                          "duration_seconds": 120, "quote": self.records["a"]["text"]}

    def assert_rejected(self, candidate, reason):
        state = answer_from_candidate("A", candidate, self.records)
        self.assertEqual(state["status"], "rejected")
        self.assertEqual(state["history"][-1]["result"]["reason"], reason)
        self.assertIsNone(state["evidence"])

    def test_correct_candidate_is_accepted(self):
        state = answer_from_candidate("A", self.candidate, self.records)
        self.assertEqual(state["status"], "answered")
        self.assertIn("120秒", state["answer"])

    def test_claiming_a_cannot_override_b_source(self):
        self.candidate.update(record_id="b", duration_seconds=80,
                              quote=self.records["b"]["text"])
        self.assert_rejected(self.candidate, "来源记录属于其他任务")

    def test_wrong_stage_or_estimate_is_rejected(self):
        for phase, kind in (("download", "actual"), ("training", "estimated")):
            with self.subTest(phase=phase, kind=kind):
                self.records["a"].update(phase=phase, kind=kind)
                self.assert_rejected(self.candidate, "来源不是实际训练阶段的记录")

    def test_wrong_unit_is_rejected(self):
        self.records["a"]["unit"] = "minutes"
        self.assert_rejected(self.candidate, "来源单位不是秒，需另行转换")

    def test_missing_source_is_rejected(self):
        self.candidate["record_id"] = "unknown"
        self.assert_rejected(self.candidate, "来源记录不存在")

    def test_wrong_quote_is_rejected(self):
        self.candidate["quote"] = self.records["b"]["text"]
        self.assert_rejected(self.candidate, "引用与来源原文不一致")

    def test_wrong_value_is_rejected(self):
        self.candidate["duration_seconds"] = 80
        self.assert_rejected(self.candidate, "耗时与来源记录不一致")

    def test_invalid_value_types_are_rejected(self):
        for value in (True, None, "120", -1, 120.0):
            with self.subTest(value=value):
                self.candidate["duration_seconds"] = value
                self.assert_rejected(self.candidate, "耗时必须是非负整数秒")

    def test_wrong_candidate_shape_is_rejected(self):
        self.assert_rejected(None, "候选结果必须是字典")
        self.assert_rejected({"record_id": []}, "缺少有效的来源记录编号")

    def test_wrong_claimed_task_is_rejected(self):
        self.candidate["run_id"] = "B"
        self.assert_rejected(self.candidate, "候选声明的任务编号不符")


if __name__ == "__main__":
    unittest.main()
