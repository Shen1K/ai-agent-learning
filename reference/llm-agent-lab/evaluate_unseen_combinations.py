"""专项评测：组合没见过，但组成它的字符都见过时，模型表现如何。

这里的“未见组合”定义得非常严格：

1. 测试题的前两个字符 A、B 都在神经模型的训练词表中；
2. 正确答案 C 也在训练词表中，避免把“无法表达的生僻字”混进问题；
3. 训练 Trigram 的参数表中从未出现过上下文 AB。

因此这份评测专门隔离用户提出的情况，而不是笼统评估所有测试字符。

本文件仍然遵守“测试集不调参”：神经模型轮次由验证损失选择，N-gram
插值强度也来自上一阶段验证集。本文件只用锁定参数计算最终成绩。
"""

import json
from datetime import datetime, timezone

import numpy as np

from data_pipeline import PROJECT_ROOT, TEST_DATA_PATH, load_documents
from evaluate_models import load_all_models
from neural_language_model import (
    UNKNOWN_TOKEN,
    encode_text,
    load_neural_model,
)
from ngram_language_model import (
    rank_pure_model,
    rank_with_backoff,
    rank_with_interpolation,
)


REPORT_DIRECTORY = PROJECT_ROOT / "reports"
JSON_REPORT_PATH = REPORT_DIRECTORY / "unseen_combination_evaluation.json"
MARKDOWN_REPORT_PATH = REPORT_DIRECTORY / "unseen_combination_evaluation.md"

# 这个值已经由上一阶段的验证集选定，绝不能再根据本测试集调参。
INTERPOLATION_STRENGTH = 0.5


def collect_unseen_combination_examples(
    documents,
    trigram_model,
    token_to_id,
):
    """收集满足专项定义的 (上下文, 正确答案) 测试题。

    返回 list[dict]；每个字典同时保留人能读的字符和神经模型需要的 ID。
    这样报告阶段不用把 ID 再猜回字符，推理阶段也不用重复编码。
    """
    examples = []
    unknown_id = token_to_id[UNKNOWN_TOKEN]

    for document in documents:
        for index in range(2, len(document)):
            # index 是正确答案位置，左侧切片取两个字符且不包含 index。
            context = document[index - 2 : index]
            actual = document[index]

            # 三个字符都必须拥有训练阶段学到的独立向量。
            context_ids = encode_text(context, token_to_id)
            actual_id = token_to_id.get(actual, unknown_id)
            all_tokens_known = (
                unknown_id not in context_ids and actual_id != unknown_id
            )
            # trigram_model 外层键就是所有训练时见过的双字符上下文。
            context_was_unseen = context not in trigram_model

            if all_tokens_known and context_was_unseen:
                examples.append(
                    {
                        "context": context,
                        "actual": actual,
                        "context_ids": context_ids,
                        "actual_id": actual_id,
                    }
                )
    return examples


def calculate_top_k_metrics(expected_ids, probability_matrix):
    """从神经网络的完整概率矩阵计算 Top-1/3/5 准确率。"""
    # expected_ids shape=(N,)，probability_matrix shape=(N,V)。N是专项题目数。
    # argpartition 不会完整排序 512 个候选，只找出概率最大的 k 个。
    metrics = {}
    for k in (1, 3, 5):
        top_ids = np.argpartition(
            probability_matrix,
            -k,
            axis=1,
        )[:, -k:]
        # expected_ids[:, None] 把 (N,) 变成 (N,1)，NumPy 会把它广播到
        # (N,k)，逐行比较正确答案是否出现在 k 个候选中。
        correct = np.any(top_ids == expected_ids[:, None], axis=1)
        metrics[f"top_{k}_accuracy"] = float(correct.mean())
    return metrics


def evaluate_neural_model(examples, model):
    """批量执行神经模型，避免逐题重复矩阵运算。"""
    # contexts shape=(N,2)，expected shape=(N,)。
    contexts = np.asarray(
        [example["context_ids"] for example in examples],
        dtype=np.int64,
    )
    expected = np.asarray(
        [example["actual_id"] for example in examples],
        dtype=np.int64,
    )
    probabilities = model.predict_probabilities(contexts)
    return calculate_top_k_metrics(expected, probabilities), probabilities


def evaluate_ngram_strategy(examples, ranker):
    """用同一批题目评估一个 N-gram 策略。

    神经模型和 N-gram 内部不同，但最后都转换成 Top-k 命中率，确保比较
    的题目、正确答案与分母完全相同。
    """
    correct = {1: 0, 3: 0, 5: 0}
    for example in examples:
        ranked = [
            character
            for character, _ in ranker(example["context"], 5)
        ]
        for k in correct:
            if example["actual"] in ranked[:k]:
                correct[k] += 1
    return {
        f"top_{k}_accuracy": correct[k] / len(examples)
        for k in correct
    }


def build_examples_for_report(examples, probabilities, id_to_token, limit=10):
    """保存少量肉眼可检查的例子，展示神经模型确实能给未见组合排序。"""
    output = []
    # argsort 默认从小到大；加负号后，原概率最大的元素排在最前面。
    top_ids = np.argsort(-probabilities, axis=1)[:, :5]

    # zip 把第1道题与第1行候选配对，并依次向后，不会改变顺序。
    for example, candidate_ids in zip(examples, top_ids):
        # 为了可读性，报告只展示由基本汉字组成的例子。
        if not all(
            "\u4e00" <= character <= "\u9fff"
            for character in example["context"] + example["actual"]
        ):
            continue
        predicted = [id_to_token[token_id] for token_id in candidate_ids]
        output.append(
            {
                "unseen_context": example["context"],
                "actual": example["actual"],
                "neural_top_5": predicted,
                "top_1_correct": predicted[0] == example["actual"],
            }
        )
        if len(output) >= limit:
            break
    return output


def percentage(value):
    return f"{value:.2%}"


def build_markdown_report(report):
    """把机器可读 report 字典转换成便于学习检查的 Markdown。"""
    rows = []
    for result in report["results"].values():
        rows.append(
            f"| {result['name']} | "
            f"{percentage(result['top_1_accuracy'])} | "
            f"{percentage(result['top_3_accuracy'])} | "
            f"{percentage(result['top_5_accuracy'])} |"
        )

    lines = [
        "# 未见字符组合专项评测",
        "",
        "## 这次到底测什么",
        "",
        "测试题中的前两个字符和正确答案都在训练词表中，但这两个前文字符组成的上下文从未在训练集中出现。换句话说：零件见过，组合没见过。",
        "",
        f"符合条件的测试题共有 **{report['eligible_example_count']:,}** 道。",
        "",
        "## 结果",
        "",
        "| 方法 | Top-1 | Top-3 | Top-5 |",
        "|---|---:|---:|---:|",
        *rows,
        "",
        "## 如何理解",
        "",
        "- 纯 Trigram 对这些题必然没有参数，因此没有列入比较。",
        "- 回退和插值通过减少上下文要求来给出答案，但会丢失一部分双字符信息。",
        "- 神经模型为每个字符复用同一组向量参数，所以即使某个二字符组合没出现，也仍能执行矩阵计算并输出完整概率分布。这叫参数共享带来的组合泛化。",
        "- 能输出不等于一定答对；Top-k 才衡量这种泛化是否真的有用。",
        "",
        "## 神经模型示例",
        "",
    ]
    for example in report["examples"]:
        lines.append(
            f"- 未见上下文 `{example['unseen_context']}`："
            f"预测前五 `{'、'.join(example['neural_top_5'])}`，"
            f"实际 `{example['actual']}`"
        )
    return "\n".join(lines)


def main():
    """检查训练源一致性，筛题，评估四种方法并保存两种报告。"""
    ngram_metadata, ngram_models = load_all_models()
    # 用括号换行只是排版，不会创建额外算法步骤。
    neural_metadata, neural_model, token_to_id, id_to_token = (
        load_neural_model()
    )

    ngram_hash = ngram_metadata[0]["training_data_sha256"]
    neural_hash = neural_metadata["training"]["training_data_sha256"]
    if ngram_hash != neural_hash:
        raise ValueError("N-gram 与神经模型使用的训练数据不同，不能公平比较")

    test_documents = load_documents(TEST_DATA_PATH)
    examples = collect_unseen_combination_examples(
        test_documents,
        ngram_models[2],
        token_to_id,
    )
    if not examples:
        raise ValueError("测试集中没有符合条件的未见字符组合")

    neural_metrics, neural_probabilities = evaluate_neural_model(
        examples,
        neural_model,
    )
    rankers = {
        "bigram": (
            "单字符 Bigram",
            lambda history, top_k: rank_pure_model(
                history,
                ngram_models[1],
                context_length=1,
                top_k=top_k,
            ),
        ),
        "hard_backoff": (
            "Trigram→Bigram→Unigram 硬回退",
            lambda history, top_k: rank_with_backoff(
                history,
                ngram_models,
                top_k,
            ),
        ),
        "adaptive_interpolation": (
            "三级自适应插值",
            lambda history, top_k: rank_with_interpolation(
                history,
                ngram_models,
                INTERPOLATION_STRENGTH,
                top_k,
            ),
        ),
    }
    # 先统一评测三种 N-gram 策略，再加入神经模型的批量计算结果。
    results = {
        key: {
            "name": name,
            **evaluate_ngram_strategy(examples, ranker),
        }
        for key, (name, ranker) in rankers.items()
    }
    results["neural"] = {
        "name": "Embedding + 前馈神经网络",
        **neural_metrics,
    }

    report = {
        "evaluated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "definition": (
            "two-character context absent from training trigram table; "
            "both context characters and target exist in training vocabulary"
        ),
        "eligible_example_count": len(examples),
        "neural_vocabulary_size": len(id_to_token),
        "interpolation_strength_selected_on_validation": (
            INTERPOLATION_STRENGTH
        ),
        "training_data_sha256": ngram_hash,
        "results": results,
        "examples": build_examples_for_report(
            examples,
            neural_probabilities,
            id_to_token,
        ),
    }

    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    JSON_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    MARKDOWN_REPORT_PATH.write_text(
        build_markdown_report(report) + "\n",
        encoding="utf-8",
    )

    print(f"未见组合测试题：{len(examples):,} 道")
    for result in results.values():
        print(
            f"{result['name']}：Top-1 {percentage(result['top_1_accuracy'])}，"
            f"Top-3 {percentage(result['top_3_accuracy'])}，"
            f"Top-5 {percentage(result['top_5_accuracy'])}"
        )
    print(f"详细报告：{MARKDOWN_REPORT_PATH}")


if __name__ == "__main__":
    main()
