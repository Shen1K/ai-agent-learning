"""只检验程序协议，不把合成裁判返回值当成模型实测或创意金标准。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from lessons.creative_studio.creative_quality_eval import (
    BASE_DIR, build_messages, input_digest, load_task, read_json, validate_and_score,
)


class CreativeQualityProtocolTests(unittest.TestCase):
    def setUp(self):
        self.rubric = read_json(BASE_DIR / "creative_rubric_v01.json")
        self.task = load_task("B")
        # 全通过只是测试计分的假数据，不是对B的实际评价。
        evidence = [
            {"field": "candidate.headline", "quote": self.task["candidate"]["headline"]},
            {"field": "references.R3", "quote": self.task["references"]["R3"]},
            {"field": "facts.F1", "quote": self.task["facts"]["F1"]},
        ]
        self.response = {
            "rubric_version": self.rubric["version"],
            "input_sha256": input_digest(self.task, self.rubric), "candidate_id": "B",
            "checks": {check[0]: {"passed": True, "reason": "测试占位，非语义判断",
                                   "evidence": copy.deepcopy(evidence)}
                       for dim in self.rubric["dimensions"] for check in dim["checks"]},
        }

    def score(self):
        return validate_and_score(self.response, self.task, self.rubric)

    def test_complete_total_and_no_publication(self):
        result = self.score()
        self.assertEqual(result["rubric_score_out_of_100"], 100)
        self.assertFalse(result["publication_authorized"])
        self.assertEqual(result["facts_and_compliance"], "not_run")

    def test_false_reduces_total(self):
        self.response["checks"]["E3"]["passed"] = False
        self.assertEqual(self.score()["rubric_score_out_of_100"], 95)

    def test_unknown_does_not_become_zero_or_renormalized_score(self):
        self.response["checks"]["E3"]["passed"] = None
        self.assertIsNone(self.score()["rubric_score_out_of_100"])

    def test_numeric_boolean_rejected(self):
        self.response["checks"]["E3"]["passed"] = 1
        with self.assertRaises(ValueError):
            self.score()

    def test_fake_quote_rejected(self):
        self.response["checks"]["E3"]["evidence"][0]["quote"] = "原文并不存在这个片段"
        with self.assertRaises(ValueError):
            self.score()

    def test_wrong_source_rejected(self):
        self.response["checks"]["E3"]["evidence"][0]["field"] = "candidate.body"
        with self.assertRaises(ValueError):
            self.score()

    def test_missing_or_extra_check_rejected(self):
        for mode in ("missing", "extra"):
            with self.subTest(mode=mode):
                value = copy.deepcopy(self.response)
                if mode == "missing":
                    del value["checks"]["I1"]
                else:
                    value["checks"]["X1"] = value["checks"]["I1"]
                with self.assertRaises(ValueError):
                    validate_and_score(value, self.task, self.rubric)

    def test_no_reference_cannot_claim_novelty(self):
        self.task["references"] = {}
        self.response["input_sha256"] = input_digest(self.task, self.rubric)
        for item in self.response["checks"].values():
            item["evidence"] = [e for e in item["evidence"] if not e["field"].startswith("references.")]
        with self.assertRaises(ValueError):
            self.score()
        for key in ("N1", "N2", "N3", "N4"):
            self.response["checks"][key]["passed"] = None
        self.assertEqual(self.score()["unknown_checks"], ["N1", "N2", "N3", "N4"])

    def test_reference_evidence_required(self):
        self.response["checks"]["N1"]["evidence"] = self.response["checks"]["N1"]["evidence"][:1]
        with self.assertRaises(ValueError):
            self.score()

    def test_product_feature_requires_fact_evidence(self):
        self.response["checks"]["P4"]["evidence"] = self.response["checks"]["P4"]["evidence"][:1]
        with self.assertRaises(ValueError):
            self.score()

    def test_dependency_conflict_rejected(self):
        self.response["checks"]["N3"]["passed"] = False
        with self.assertRaises(ValueError):
            self.score()
        self.response["checks"]["N4"]["passed"] = False
        self.assertEqual(self.score()["rubric_score_out_of_100"], 90)

    def test_unknown_dependency_propagates(self):
        self.response["checks"]["N3"]["passed"] = None
        with self.assertRaises(ValueError):
            self.score()
        self.response["checks"]["N4"]["passed"] = None
        self.assertIsNone(self.score()["rubric_score_out_of_100"])

    def test_stale_input_or_version_rejected(self):
        self.task["candidate"]["body"] += "改稿"
        with self.assertRaises(ValueError):
            self.score()
        self.response["input_sha256"] = input_digest(self.task, self.rubric)
        self.response["rubric_version"] = "old"
        with self.assertRaises(ValueError):
            self.score()

    def test_expected_labels_not_in_messages(self):
        messages = build_messages(self.task, self.rubric)
        payload = json.loads(messages[1]["content"])
        self.assertEqual(set(payload["task"]), {"brief", "facts", "references", "candidate"})
        self.assertNotIn("development_hypotheses", messages[1]["content"])
        self.assertEqual(len(messages), 2)

    def test_empty_reason_and_evidence_rejected(self):
        for field, value in (("reason", " "), ("evidence", [])):
            with self.subTest(field=field):
                response = copy.deepcopy(self.response)
                response["checks"]["E3"][field] = value
                with self.assertRaises(ValueError):
                    validate_and_score(response, self.task, self.rubric)

    def test_duplicate_json_keys_rejected(self):
        # 测试临时文件由测试用例管理，不改动任何已有资料。
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"score": 0, "score": 100}', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_json(path)


if __name__ == "__main__":
    unittest.main()
