"""神经语言模型的最小行为测试。

神经模型内部有随机参数，直接断言“必须预测某个汉字”通常不稳定。因此这里
检查更基础的不变量：概率和为1、数组形状正确、训练后损失下降，以及未见
组合仍有输出。这些条件失败时，后面的真实数据成绩也不再可信。
"""

import unittest

import numpy as np

from neural_language_model import (
    FeedForwardLanguageModel,
    UNKNOWN_TOKEN,
    build_training_examples,
    build_vocabulary,
    softmax,
)


class NeuralLanguageModelTests(unittest.TestCase):
    def test_softmax_rows_are_probability_distributions(self):
        """每道题所有候选的概率之和都必须约等于1。"""
        # 两行代表两道题，每行三个数字代表三个候选字符的原始分数。
        logits = np.asarray([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]])
        probabilities = softmax(logits)

        # 浮点运算可能有极小误差，所以不用 assertEqual 精确比较。
        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            np.ones(2),
        )
        self.assertTrue(np.all(probabilities >= 0))

    def test_vocabulary_is_built_from_frequency(self):
        """词表容量有限时，保留最高频字符并把编号0留给 <UNK>。"""
        token_to_id, id_to_token, _ = build_vocabulary(
            ["甲甲甲乙乙丙"],
            maximum_size=3,
        )

        self.assertEqual(id_to_token, [UNKNOWN_TOKEN, "甲", "乙"])
        self.assertNotIn("丙", token_to_id)

    def test_training_examples_use_two_character_context(self):
        """确认字符串“中国人”被制作成输入“中国”、答案“人”。"""
        token_to_id, _, _ = build_vocabulary(
            ["中国人"],
            maximum_size=10,
        )
        x, y = build_training_examples(
            ["中国人"],
            token_to_id,
            context_length=2,
        )

        # 一道题、每题两个上下文字符，所以二维形状是 (1,2)。
        self.assertEqual(x.shape, (1, 2))
        self.assertEqual(x[0].tolist(), [token_to_id["中"], token_to_id["国"]])
        self.assertEqual(y[0], token_to_id["人"])

    def test_gradient_descent_reduces_loss(self):
        """在一个极小重复语料上，训练后损失应该明显降低。"""
        documents = ["甲乙丙" * 20]
        token_to_id, id_to_token, _ = build_vocabulary(
            documents,
            maximum_size=10,
        )
        x, y = build_training_examples(
            documents,
            token_to_id,
            context_length=2,
        )
        model = FeedForwardLanguageModel(
            vocabulary_size=len(id_to_token),
            context_length=2,
            embedding_size=4,
            hidden_size=6,
            seed=7,
        )

        initial_loss, _ = model.loss_and_gradients(x, y)
        # `_` 表示循环编号不重要，只需要重复执行100次更新。
        for _ in range(100):
            _, gradients = model.loss_and_gradients(x, y)
            model.apply_gradients(gradients, learning_rate=0.3)
        final_loss, _ = model.loss_and_gradients(x, y)

        # 不只要求下降，还要求降到初始值一半以下，避免微小误差蒙混过关。
        self.assertLess(final_loss, initial_loss * 0.5)

    def test_unseen_known_pair_still_has_a_distribution(self):
        """神经模型不需要为每个字符对单独建一行查找表。"""
        documents = ["甲乙丙", "丁戊己"]
        token_to_id, id_to_token, _ = build_vocabulary(
            documents,
            maximum_size=10,
        )
        model = FeedForwardLanguageModel(
            vocabulary_size=len(id_to_token),
            context_length=2,
            seed=7,
        )

        # “甲戊”从未在 documents 中相邻出现，但两个字符各自都见过。
        unseen_pair = np.asarray(
            [[token_to_id["甲"], token_to_id["戊"]]],
            dtype=np.int64,
        )
        probabilities = model.predict_probabilities(unseen_pair)

        # 一道题必须为词表中的每个候选都输出一个概率。
        self.assertEqual(probabilities.shape, (1, len(id_to_token)))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
