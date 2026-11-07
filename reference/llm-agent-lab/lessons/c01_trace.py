"""C01 微型演示：追踪一段文本如何变成训练题、参数和预测。

运行方式：
    python3 lessons/c01_trace.py

这个文件不下载数据、不读取正式语料、也不覆盖已有模型。它只使用三句
可以手算的文本调用现有 N-gram 函数，让学习者先看清数据流。C02 才会
详细计算条件概率；C03、C04 再进入向量和参数更新。
"""

import sys
from pathlib import Path


# 直接运行 lessons/c01_trace.py 时，Python 默认只在 lessons/ 中查找模块。
# 将项目根目录加入导入路径后，才能复用根目录的 ngram_language_model.py。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ngram_language_model import (  # noqa: E402
    probabilities_from_counts,
    train_two_character_model,
)


def build_visible_training_examples(documents, context_length=2):
    """把正文拆成适合课堂展示的“上下文 → 正确答案”。

    参数：
        documents：字符串列表，每个字符串是一篇独立的微型文章。
        context_length：每道题向左观察几个字符，本课固定为2。

    返回值：
        元组列表，例如 [("中国", "人"), ("中国", "家")]。

    这里重复写出训练题拆分过程，是为了课堂上能逐步打印；正式训练逻辑
    仍然使用 ngram_language_model.py 中经过测试的实现。
    """
    examples = []

    for document in documents:
        # index 指向当前题目的正确答案。index 从2开始，是因为答案左边
        # 至少要有两个字符，才能组成长度为2的上下文。
        for index in range(context_length, len(document)):
            context = document[index - context_length : index]
            correct_answer = document[index]
            examples.append((context, correct_answer))

    return examples


def main():
    """依次打印原始数据、训练题、模型参数和一次预测。"""
    documents = ["中国人", "中国家", "中华人"]

    print("第1步：原始训练数据")
    for document_number, document in enumerate(documents, start=1):
        print(f"  文章{document_number}：{document}")

    print("\n第2步：把文本变成监督学习题目")
    examples = build_visible_training_examples(documents)
    for context, correct_answer in examples:
        print(f"  输入上下文 {context!r} → 正确答案 {correct_answer!r}")

    print("\n第3步：训练形成参数（这里的参数就是次数）")
    model = train_two_character_model(documents)
    for context, next_character_counts in sorted(model.items()):
        print(f"  {context!r} → {dict(next_character_counts)}")

    print("\n第4步：使用参数进行一次预测")
    prediction_context = "中国"
    counts = model[prediction_context]
    probabilities = probabilities_from_counts(counts)
    for character, probability in sorted(
        probabilities.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(
            f"  P({character!r} | {prediction_context!r}) "
            f"= {probability:.0%}"
        )

    print("\n观察：训练结束后，预测阶段只读取参数；不会再修改计数。")


if __name__ == "__main__":
    main()
