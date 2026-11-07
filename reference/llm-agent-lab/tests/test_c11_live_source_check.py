"""真实资料核对器的边界测试；所有篡改都仅在内存中进行。"""

import hashlib
import json
import unittest

from lessons.c11_live_source_check import SOURCE_RELATIVE, verify_claim, verify_snapshot


class LiveSourceCheckTests(unittest.TestCase):
    def setUp(self):
        self.document = {"training_document_count": 895}
        self.text = json.dumps(self.document, indent=2)
        self.raw = self.text.encode("utf-8")
        self.found = {"question_id": "training_count", "field": "training_document_count",
                      "status": "found", "value": 895,
                      "quote": '"training_document_count": 895', "line": 2}
        self.absent = {"question_id": "training_duration", "field": "training_duration_seconds",
                       "status": "not_found", "value": None, "quote": None, "line": None}
        self.snapshot = {"source": SOURCE_RELATIVE,
                         "source_sha256": hashlib.sha256(self.raw).hexdigest(),
                         "claims": [self.found, self.absent]}

    def check_found(self):
        return verify_claim(self.document, self.text, self.found)

    def test_correct_value_and_absent_field(self):
        report = verify_snapshot(self.snapshot, self.raw)
        self.assertEqual([item["outcome"] for item in report["results"]],
                         ["verified_value", "confirmed_field_absent"])

    def test_wrong_value_rejected(self):
        self.found["value"] = 995
        self.assertEqual(self.check_found()["reason"], "候选数值与原文件不一致")

    def test_wrong_field_rejected(self):
        self.found["field"] = "accepted_document_count"
        self.assertEqual(self.check_found()["outcome"], "rejected")

    def test_wrong_quote_or_line_rejected(self):
        self.found["quote"] = "895"
        self.assertEqual(self.check_found()["outcome"], "rejected")
        self.found["quote"] = '"training_document_count": 895'
        self.found["line"] = 1
        self.assertEqual(self.check_found()["outcome"], "rejected")

    def test_false_absence_rejected(self):
        self.absent.update(question_id="training_count", field="training_document_count")
        result = verify_claim(self.document, self.text, self.absent)
        self.assertEqual(result["outcome"], "rejected")

    def test_missing_field_is_not_claim_about_all_text(self):
        # 即使文字记录有耗时，程序也只断言“指定字段缺失”。
        data = {"notes": "本次训练实际耗时90秒。"}
        result = verify_claim(data, json.dumps(data), self.absent)
        self.assertEqual(result["outcome"], "confirmed_field_absent")

    def test_source_change_stops_verification(self):
        with self.assertRaisesRegex(ValueError, "版本不同"):
            verify_snapshot(self.snapshot, self.raw + b"\n")

    def test_source_substitution_stops_verification(self):
        self.snapshot["source"] = "another.json"
        with self.assertRaisesRegex(ValueError, "来源"):
            verify_snapshot(self.snapshot, self.raw)

    def test_duplicate_questions_rejected(self):
        self.snapshot["claims"] = [self.found, self.found]
        with self.assertRaisesRegex(ValueError, "各出现一次"):
            verify_snapshot(self.snapshot, self.raw)

    def test_bool_is_not_integer_count(self):
        self.found["value"] = True
        self.assertEqual(self.check_found()["outcome"], "rejected")


if __name__ == "__main__":
    unittest.main()
