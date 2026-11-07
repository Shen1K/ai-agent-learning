"""用固定案例检查任务组装；不调用模型，不生成任何营销素材。"""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from lessons.creative_studio.task_contract import (
    DEFAULT_FIXTURE, assemble_task, load_fixture,
)


class CreativeTaskContractTests(unittest.TestCase):
    def setUp(self):
        # 每个测试重新读取：前一个测试修改价格，不影响后一个。
        self.data = load_fixture(DEFAULT_FIXTURE)

    def assemble(self):
        return assemble_task(self.data, DEFAULT_FIXTURE.parent)

    def codes(self, result):
        return [issue["code"] for issue in result["blocking_issues"]]

    def test_price_present_but_asset_missing(self):
        result = self.assemble()
        self.assertEqual(result["overlay_text"]["price_text"], "活动价 ¥69")
        self.assertEqual(self.codes(result), ["missing_product_asset"])
        self.assertEqual(result["status"], "needs_input")
        self.assertFalse(result["can_render"])

    def test_missing_price_and_asset_are_both_reported(self):
        self.data["product"]["activity_price_cny"] = None
        result = self.assemble()
        self.assertIsNone(result["overlay_text"]["price_text"])
        self.assertEqual(self.codes(result), ["missing_activity_price", "missing_product_asset"])
        self.assertEqual(result["creative_candidate"]["headline"],
                         self.data["creative_proposal"]["headline"])

    def test_zero_is_not_missing(self):
        self.data["product"]["activity_price_cny"] = 0
        result = self.assemble()
        self.assertEqual(result["overlay_text"]["price_text"], "活动价 ¥0")
        self.assertNotIn("missing_activity_price", self.codes(result))

    def test_decimal_price_preserved(self):
        self.data["product"]["activity_price_cny"] = 69.99
        self.assertEqual(self.assemble()["overlay_text"]["price_text"], "活动价 ¥69.99")

    def test_invalid_prices_rejected(self):
        for price in ("69", True, -1, float("nan"), float("inf"), 69.999):
            with self.subTest(price=price):
                self.data["product"]["activity_price_cny"] = price
                with self.assertRaises(ValueError):
                    self.assemble()

    def test_missing_key_is_contract_error(self):
        del self.data["product"]["activity_price_cny"]
        with self.assertRaisesRegex(ValueError, "缺少"):
            self.assemble()

    def test_other_product_rejected(self):
        self.data["request"]["product_id"] = "cup-002"
        with self.assertRaisesRegex(ValueError, "编号"):
            self.assemble()

    def test_candidate_cannot_override_price_or_name(self):
        self.data["creative_proposal"].update(price_text="活动价 ¥59", name="保温杯")
        result = self.assemble()
        self.assertEqual(result["overlay_text"]["price_text"], "活动价 ¥69")
        self.assertEqual(result["overlay_text"]["product_name"], "暖序陶瓷咖啡杯")

    def test_old_plan_is_not_reused(self):
        self.data["execution_plan"]["overlay_text"]["price_text"] = "活动价 ¥1"
        self.assertEqual(self.assemble()["overlay_text"]["price_text"], "活动价 ¥69")

    def test_inputs_and_nested_candidates_are_not_mutated(self):
        before = deepcopy(self.data)
        result = self.assemble()
        result["creative_candidate"]["product_claims"].append("测试用修改")
        self.assertEqual(self.data, before)

    def test_missing_path_and_directory_are_not_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in (directory, str(Path(directory) / "absent.png")):
                with self.subTest(path=path):
                    self.data["product"]["reference_image_path"] = path
                    self.assertIn("product_asset_not_found", self.codes(self.assemble()))

    def test_existing_file_is_not_approved_image(self):
        # 临时占位文件只验证“存在”分支，故意不冒充可用图片。
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "placeholder.txt").touch()
            self.data["product"]["reference_image_path"] = "placeholder.txt"
            result = assemble_task(self.data, directory)
            self.assertEqual(result["blocking_issues"], [])
            self.assertEqual(result["status"], "needs_review")
            self.assertTrue(result["review_required"])
            self.assertFalse(result["can_render"])
            self.assertFalse(result["image_generation_executed"])

    def test_unknown_claim_is_not_automatically_approved(self):
        self.data["creative_proposal"]["headline"] = "保温24小时"
        result = self.assemble()
        self.assertTrue(result["review_required"])
        self.assertFalse(result["can_render"])

    def test_wrong_section_and_canvas_rejected(self):
        for section in ("product", "request", "brand_rules", "creative_proposal"):
            bad = deepcopy(self.data)
            bad[section] = []
            with self.subTest(section=section), self.assertRaises(ValueError):
                assemble_task(bad, DEFAULT_FIXTURE.parent)
        self.data["request"]["width_px"] = 512
        with self.assertRaises(ValueError):
            self.assemble()

    def test_unsupported_policy_rejected(self):
        self.data["brand_rules"]["publication_policy"] = "publish_now"
        with self.assertRaisesRegex(ValueError, "不支持"):
            self.assemble()

    def test_load_errors_are_not_missing_price(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_fixture(Path(directory) / "absent.json")
            empty = Path(directory) / "empty.json"
            empty.touch()
            with self.assertRaises(ValueError):
                load_fixture(empty)


if __name__ == "__main__":
    unittest.main()
