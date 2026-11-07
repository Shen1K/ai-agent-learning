"""交互式推理入口：查看一个双字符上下文的神经模型预测。

示例：
    python3 predict_neural.py 种基

“种基”来自当前专项报告：两个字符在训练中都见过，但这个相邻组合
没有出现在 Trigram 参数表中。你也可以换成任意两个字符亲自实验。

这个脚本不训练模型，只加载已经保存的权重进行一次前向传播。
"""

import argparse

import numpy as np

from neural_language_model import UNKNOWN_TOKEN, encode_text, load_neural_model
from ngram_language_model import TRIGRAM_MODEL_PATH, load_model


def parse_arguments():
    """读取两个字符的上下文，以及可选的候选数量。"""
    parser = argparse.ArgumentParser(
        description="让神经语言模型根据两个字符预测下一个字符"
    )
    parser.add_argument("context", help="恰好两个字符，例如：种基")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main():
    """验证输入、编码、执行预测，并把字符概率按从高到低打印。"""
    arguments = parse_arguments()
    if len(arguments.context) != 2:
        raise ValueError("当前模型固定观察两个字符，请输入恰好两个字符")

    # metadata 在本脚本中不用，因此用 `_` 接住第一个返回值。
    _, model, token_to_id, id_to_token = load_neural_model()
    _, trigram_model = load_model(TRIGRAM_MODEL_PATH)
    unknown_id = token_to_id[UNKNOWN_TOKEN]
    # 例如“种基”会变成两个整数 ID。模型只计算数字，不直接计算字符串。
    context_ids = encode_text(arguments.context, token_to_id)

    if unknown_id in context_ids:
        unknown_characters = [
            character
            for character in arguments.context
            if character not in token_to_id
        ]
        raise ValueError(
            f"字符 {'、'.join(unknown_characters)} 不在神经模型的512字符词表中"
        )

    # 模型需要二维输入：(题目数量, 每题上下文长度)。这里题目数量是1。
    # 外层 [] 表示“只有一道题”，所以 x.shape 是 (1,2)，而不是 (2,)。
    x = np.asarray([context_ids], dtype=np.int64)
    probabilities = model.predict_probabilities(x)[0]
    top_k = min(arguments.top_k, len(id_to_token))
    # 负号把降序问题变成 NumPy 默认的升序排序问题。
    candidate_ids = np.argsort(-probabilities)[:top_k]

    if arguments.context in trigram_model:
        count = sum(trigram_model[arguments.context].values())
        print(f"Trigram 训练中见过上下文“{arguments.context}”{count}次。")
    else:
        print(
            f"Trigram 训练中没见过“{arguments.context}”，"
            "但神经模型仍能共享字符向量和网络参数进行计算。"
        )

    print("神经模型候选：")
    for rank, token_id in enumerate(candidate_ids, start=1):
        token = id_to_token[token_id]
        print(f"  {rank}. {token!r}：{probabilities[token_id]:.2%}")


if __name__ == "__main__":
    main()
