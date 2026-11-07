"""规则逻辑测试，不是模型理解准确率评测；标准答案是教学约定的手算值。"""

from copy import deepcopy
import unittest

from lessons.creative_studio.promotion_review import (
    calculate_order, coupon_discount, load_materials, review_case,
)


class PromotionReviewTests(unittest.TestCase):
    def setUp(self):
        self.rules, bundle = load_materials()
        self.cases = bundle["cases"]

    def test_threshold_boundaries(self):
        # 门槛函数的单元测试，不假装69元杯子能组成98.99元订单。
        for eligible, expected in ((9899, 0), (9900, 1000), (9901, 1000)):
            with self.subTest(eligible=eligible):
                self.assertEqual(coupon_discount(eligible, 9900, 1000), expected)

    def test_expected_order_amounts(self):
        # 独立写出教学手算结果，不调用同一函数生成expected自我比较。
        for index, expected in ((0, (6900, 0, 6900)), (1, (13800, 1000, 12800)),
                                (2, (6900, 0, 9900))):
            result = calculate_order(self.cases[index]["order"], self.rules)
            self.assertEqual((result["eligible_cents"], result["discount_cents"],
                              result["total_cents"]), expected)

    def test_case_routes(self):
        expected = ["conflict_given_interpretation", "consistent_given_interpretation",
                    "conflict_given_interpretation", "needs_confirmation"]
        self.assertEqual([review_case(case, self.rules)["status"] for case in self.cases], expected)
        self.assertTrue(all(not review_case(case, self.rules)["can_publish"] for case in self.cases))

    def test_missing_shipping_rule_does_not_default(self):
        self.rules["shipping_counts_toward_threshold"] = None
        result = review_case(self.cases[2], self.rules)
        self.assertEqual(result["status"], "needs_information")
        self.assertNotIn("calculation", result)

    def test_shipping_policy_changes_eligibility(self):
        # 假设另一版规则将运费计入，只测试公式分支，不改正式课堂规则。
        self.rules["shipping_counts_toward_threshold"] = True
        result = calculate_order(self.cases[2]["order"], self.rules)
        self.assertEqual(result["total_cents"], 8900)

    def test_rule_version_or_activity_mismatch(self):
        for field in ("rule_version", "activity_id"):
            case = deepcopy(self.cases[0])
            case[field] = "unknown"
            self.assertEqual(review_case(case, self.rules)["status"], "needs_confirmation")

    def test_quote_must_match(self):
        self.cases[0]["interpretation"]["quote"] = "伪造的引用"
        self.assertEqual(review_case(self.cases[0], self.rules)["status"], "needs_confirmation")

    def test_claim_quantity_mismatch(self):
        self.cases[0]["interpretation"]["quantity"] = 2
        self.assertEqual(review_case(self.cases[0], self.rules)["status"], "needs_confirmation")

    def test_invalid_amounts(self):
        for value in (True, -1, "6900", 6900.0, None):
            with self.subTest(value=value):
                self.cases[0]["interpretation"]["claimed_total_cents"] = value
                self.assertEqual(review_case(self.cases[0], self.rules)["status"], "needs_confirmation")

    def test_unsupported_stacking_and_wrong_product(self):
        for field, value in (("other_discounts_present", True), ("product_id", "other")):
            case = deepcopy(self.cases[0])
            case["order"][field] = value
            self.assertEqual(review_case(case, self.rules)["status"], "needs_confirmation")

    def test_discount_cannot_exceed_threshold(self):
        self.rules["discount_cents"] = 10000
        self.assertEqual(review_case(self.cases[0], self.rules)["status"], "needs_confirmation")

    def test_inputs_unchanged(self):
        before = deepcopy((self.rules, self.cases))
        for case in self.cases:
            review_case(case, self.rules)
        self.assertEqual((self.rules, self.cases), before)

    def test_quote_existence_does_not_prove_interpretation(self):
        # 刻意保存错误理解：标题说一个，候选却说两个，并配上两杯订单。
        # 程序检查会一致；这正是缺少独立语义核对的已知限制，不能叫海报通过。
        case = deepcopy(self.cases[1])
        case["headline"] = "买一个杯子，领券后共59元（免运费）"
        case["interpretation"]["quote"] = case["headline"]
        result = review_case(case, self.rules)
        self.assertEqual(result["status"], "consistent_given_interpretation")
        self.assertFalse(result["can_publish"])


if __name__ == "__main__":
    unittest.main()
