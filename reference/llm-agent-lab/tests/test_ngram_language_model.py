"""使用 Python 标准库验证 N-gram 的关键行为。

测试不是为了再次实现算法，而是给算法写可以自动检查的“小例题”。这里的
语料故意极小，因此每个计数都能用纸笔验证。修改模型后运行测试，可以快速
知道原来的关键行为是否被意外破坏，这称为回归测试。
"""

import unittest

from ngram_language_model import (
    interpolated_distribution,
    rank_with_interpolation,
    select_backoff_counts,
    train_single_character_model,
    train_two_character_model,
    train_unigram_model,
)


class NgramLanguageModelTests(unittest.TestCase):
    def setUp(self):
        """每个 test_ 方法运行前都会重新创建一套干净的小模型。"""
        # 使用非常小的固定语料，让每个预期计数都可以人工验证。
        self.documents = ["中国人", "中国家", "中华人"]
        self.models = {
            0: train_unigram_model(self.documents),
            1: train_single_character_model(self.documents),
            2: train_two_character_model(self.documents),
        }

    def test_single_character_model_keeps_original_counting_logic(self):
        """确认 Bigram 只观察前一个字符，而且计数正确。"""
        # “中”后面两次是“国”，一次是“华”。
        self.assertEqual(self.models[1]["中"]["国"], 2)
        self.assertEqual(self.models[1]["中"]["华"], 1)

    def test_two_character_model_uses_two_character_context(self):
        """确认 Trigram 把两个字符共同作为外层字典键。"""
        # “中国”后面分别出现过“人”和“家”。
        self.assertEqual(self.models[2]["中国"]["人"], 1)
        self.assertEqual(self.models[2]["中国"]["家"], 1)

    def test_documents_do_not_create_false_cross_document_transition(self):
        """确认两篇文章的边界不会被错误地当作相邻字符。"""
        model = train_single_character_model(["甲乙", "丙丁"])
        # 如果错误地拼接文章，就会学到“乙→丙”；正确实现不应该有。
        self.assertNotIn("乙", model)

    def test_backoff_uses_available_level(self):
        """确认有高阶用高阶，没有时依次退到低阶。"""
        # 返回值第一项是 Counter，本测试只关心第二项层级，所以用 `_` 忽略。
        _, exact_level = select_backoff_counts("中国", self.models)
        _, fallback_level = select_backoff_counts("外国", self.models)
        _, final_level = select_backoff_counts("未知", self.models)

        self.assertEqual(exact_level, 2)
        self.assertEqual(fallback_level, 1)
        self.assertEqual(final_level, 0)

    def test_interpolated_distribution_sums_to_one(self):
        """概率分布的基本不变量：所有候选概率相加必须等于1。"""
        distribution = interpolated_distribution(
            "中国",
            self.models,
            strength=1,
        )
        self.assertAlmostEqual(sum(distribution.values()), 1.0)

    def test_fast_ranking_matches_full_distribution(self):
        """确认优化后的快速 Top-k 与完整计算结果完全一致。"""
        distribution = interpolated_distribution(
            "中国",
            self.models,
            strength=1,
        )
        # lambda 告诉 sorted：先按负概率（即概率降序），再按字符稳定排序。
        expected = sorted(
            distribution.items(),
            key=lambda item: (-item[1], item[0]),
        )[:3]
        actual = rank_with_interpolation(
            "中国",
            self.models,
            strength=1,
            top_k=3,
        )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
