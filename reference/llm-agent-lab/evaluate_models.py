"""评测能力：用验证集选择参数，再在测试集比较四种预测策略。

这个文件不会训练模型。请先运行 python3 main.py 生成模型文件。

评测过程可以类比学校考试：

- 训练集相当于教材，用来计算模型参数；
- 验证集相当于模拟考试，用来选择插值强度；
- 测试集相当于最终考试，只能在所有选择完成后使用一次。

最重要的输出指标：coverage 表示能否给答案，Top-k 表示正确字符是否出现
在前 k 个候选中。所有模型必须面对完全相同的题目，比较才公平。
"""

import json
from datetime import datetime, timezone

from data_pipeline import (
    PROJECT_ROOT,
    TEST_DATA_PATH,
    TRAIN_SPLIT_PATH,
    VALIDATION_SPLIT_PATH,
    load_documents,
    load_jsonl_records,
)
from ngram_language_model import (
    BIGRAM_MODEL_PATH,
    TRIGRAM_MODEL_PATH,
    UNIGRAM_MODEL_PATH,
    load_model,
    rank_pure_model,
    rank_with_backoff,
    rank_with_interpolation,
)


REPORT_DIRECTORY = PROJECT_ROOT / "reports"
JSON_REPORT_PATH = REPORT_DIRECTORY / "model_evaluation.json"
MARKDOWN_REPORT_PATH = REPORT_DIRECTORY / "model_evaluation.md"

# 插值强度越大，越不轻易相信只出现过少数几次的高阶上下文。
# 我们让验证集选一个值，而不是观察测试成绩后手工挑选。
INTERPOLATION_STRENGTH_CANDIDATES = (0.5, 1, 2, 5, 10, 20, 50)


def load_all_models():
    """加载三层模型，并确认它们来自同一份训练语料。"""
    # 三个字典都使用 0/1/2 作为键，便于确认路径、元数据和模型一一对应。
    paths = {
        0: UNIGRAM_MODEL_PATH,
        1: BIGRAM_MODEL_PATH,
        2: TRIGRAM_MODEL_PATH,
    }
    models = {}
    metadata = {}

    for context_length, path in paths.items():
        model_metadata, model = load_model(path)
        metadata[context_length] = model_metadata
        models[context_length] = model

    # 集合会自动去重。若集合长度为1，说明三个哈希完全相同。
    training_hashes = {
        item["training_data_sha256"] for item in metadata.values()
    }
    if len(training_hashes) != 1:
        raise ValueError("三个模型使用的训练数据不同，无法公平比较")
    return metadata, models


def is_chinese_example(context, actual):
    """示例只展示基本汉字，让新手更容易观察预测关系。

    all(...) 要求生成器中的每一个条件都为 True；只要有标点就返回 False。
    这只控制报告展示，不会删掉正式评测题目。
    """
    return all(
        "\u4e00" <= character <= "\u9fff"
        for character in context + actual
    )


def add_example(examples, example, limit=5):
    """每类最多保存5个上下文不重复的例子。

    any(...) 检查是否已经保存过同一上下文，避免报告被重复句子占满。
    """
    if len(examples) >= limit:
        return
    if any(item["context"] == example["context"] for item in examples):
        return
    examples.append(example)


def evaluate_strategy(name, documents, ranker, minimum_history=2):
    """逐字遮住正确答案，检查它是否位于前1、3、5名。

    所有策略都从每篇文章的第三个字符开始预测。这样单字符模型和
    双字符模型面对完全相同的题目，分母也完全相同。

    ranker 是一个函数，调用形式固定为 ranker(history, top_k)。函数返回
    [(候选字符, 概率), ...]。统一接口让评测逻辑不必知道内部使用哪种模型。
    """
    total_predictions = 0
    covered_predictions = 0
    # 三个计数器共用一个字典，后面可用同一循环同时更新 Top-1/3/5。
    top_k_correct = {1: 0, 3: 0, 5: 0}
    correct_examples = []
    wrong_examples = []
    uncovered_examples = []
    # 同一个上下文可能在语料中出现很多次。缓存可避免重复排序。
    ranked_cache = {}

    for document in documents:
        for index in range(minimum_history, len(document)):
            total_predictions += 1
            # 当前所有模型最多只看前两个字符，因此无须复制整段历史。
            history = document[index - minimum_history : index]
            actual = document[index]
            # 同一个上下文可能出现很多次，预测排名只需计算一次。
            if history not in ranked_cache:
                ranked_cache[history] = ranker(history, 5)
            ranked = ranked_cache[history]

            if not ranked:
                if is_chinese_example(history[-minimum_history:], actual):
                    add_example(
                        uncovered_examples,
                        {
                            "context": history[-minimum_history:],
                            "actual": actual,
                        },
                    )
                continue

            covered_predictions += 1
            # 列表推导式只取元组中的字符；概率在判断是否命中时不需要。
            ranked_characters = [character for character, _ in ranked]
            for k in top_k_correct:
                if actual in ranked_characters[:k]:
                    top_k_correct[k] += 1

            context = history[-minimum_history:]
            example = {
                "context": context,
                "predicted": ranked_characters[0],
                "actual": actual,
                "top_5": ranked_characters,
            }
            if is_chinese_example(context, actual):
                if ranked_characters[0] == actual:
                    add_example(correct_examples, example)
                else:
                    add_example(wrong_examples, example)

    # 嵌套函数能直接读取外层 total_predictions，不必反复传入分母。
    def rate(count):
        return count / total_predictions if total_predictions else 0

    # “已覆盖准确率”排除了模型完全答不出的题，用来区分覆盖与判断能力。
    def covered_rate(count):
        return count / covered_predictions if covered_predictions else 0

    return {
        "name": name,
        "total_predictions": total_predictions,
        "covered_predictions": covered_predictions,
        "coverage": rate(covered_predictions),
        "top_1_correct": top_k_correct[1],
        "top_1_accuracy": rate(top_k_correct[1]),
        "top_3_accuracy": rate(top_k_correct[3]),
        "top_5_accuracy": rate(top_k_correct[5]),
        "top_1_accuracy_when_covered": covered_rate(top_k_correct[1]),
        "examples": {
            "correct": correct_examples,
            "wrong": wrong_examples,
            "uncovered": uncovered_examples,
        },
    }


def choose_interpolation_strength(validation_documents, models):
    """只看验证集成绩选择插值强度。

    lambda 的 `value=strength` 会把当前循环值固定下来。如果省略默认参数，
    Python 闭包可能在真正调用时统一读取到循环结束后的最后一个 strength。
    """
    trials = []

    for strength in INTERPOLATION_STRENGTH_CANDIDATES:
        result = evaluate_strategy(
            f"插值强度 {strength}",
            validation_documents,
            lambda history, top_k, value=strength: rank_with_interpolation(
                history,
                models,
                value,
                top_k,
            ),
        )
        trials.append(
            {
                "strength": strength,
                "top_1_accuracy": result["top_1_accuracy"],
                "top_3_accuracy": result["top_3_accuracy"],
                "top_5_accuracy": result["top_5_accuracy"],
            }
        )

    # max 在相同成绩时会保留候选列表中更靠前的值，结果可重复。
    selected = max(trials, key=lambda trial: trial["top_1_accuracy"])
    return selected["strength"], trials


def build_strategies(models, interpolation_strength):
    """把不同预测方法包装成统一接口，交给同一个评测函数。"""
    # 每个值都是 (人类可读名称, 统一签名的排名函数)。
    return {
        "single_character": (
            "单字符 Bigram",
            lambda history, top_k: rank_pure_model(
                history,
                models[1],
                context_length=1,
                top_k=top_k,
            ),
        ),
        "two_character": (
            "双字符 Trigram",
            lambda history, top_k: rank_pure_model(
                history,
                models[2],
                context_length=2,
                top_k=top_k,
            ),
        ),
        "hard_backoff": (
            "Trigram→Bigram→Unigram 硬回退",
            lambda history, top_k: rank_with_backoff(
                history,
                models,
                top_k,
            ),
        ),
        "adaptive_interpolation": (
            "Trigram+Bigram+Unigram 自适应插值",
            lambda history, top_k: rank_with_interpolation(
                history,
                models,
                interpolation_strength,
                top_k,
            ),
        ),
    }


def verify_document_splits():
    """确认训练、验证、测试三部分没有文章 ID 重叠。"""
    split_paths = {
        "train": TRAIN_SPLIT_PATH,
        "validation": VALIDATION_SPLIT_PATH,
        "test": TEST_DATA_PATH,
    }
    # 三份数据都转换成 ID 集合；`&` 是集合交集运算符。
    id_sets = {
        name: {record["id"] for record in load_jsonl_records(path)}
        for name, path in split_paths.items()
    }
    overlaps = {
        "train_validation": len(id_sets["train"] & id_sets["validation"]),
        "train_test": len(id_sets["train"] & id_sets["test"]),
        "validation_test": len(
            id_sets["validation"] & id_sets["test"]
        ),
    }
    if any(overlaps.values()):
        raise ValueError(f"数据集之间存在文章重叠：{overlaps}")
    return overlaps


def percentage(value):
    """把 0~1 的小数格式化成保留两位小数的百分比字符串。"""
    return f"{value:.2%}"


def build_markdown_report(report):
    """生成面向学习者的 Markdown 评测报告。

    计算过程保存到 JSON，展示过程保存到 Markdown。把两者分开后，程序
    可以读取 JSON 做二次分析，人可以直接阅读 Markdown。
    """
    results = report["test_results"]
    rows = []
    for result in results.values():
        rows.append(
            (
                result["name"],
                percentage(result["coverage"]),
                percentage(result["top_1_accuracy"]),
                percentage(result["top_3_accuracy"]),
                percentage(result["top_5_accuracy"]),
                percentage(result["top_1_accuracy_when_covered"]),
            )
        )

    # max 的 key 参数不是字典键，而是“按什么分数比较”的函数。
    best_key = max(
        results,
        key=lambda key: results[key]["top_1_accuracy"],
    )
    best = results[best_key]
    lines = [
        "# N-gram 回退与插值策略评测",
        "",
        f"验证集用于选择插值强度，最终选中 `{report['selected_interpolation_strength']}`。测试集只用于最终比较。",
        "",
        "## 数据隔离",
        "",
        f"- 训练文章：{report['document_counts']['train']} 篇",
        f"- 验证文章：{report['document_counts']['validation']} 篇",
        f"- 测试文章：{report['document_counts']['test']} 篇",
        "- 三部分文章 ID 的交集均为 0。",
        "",
        "## 测试集结果",
        "",
        "| 策略 | 覆盖率 | Top-1 | Top-3 | Top-5 | 已覆盖 Top-1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {name} | {coverage} | {top_1} | {top_3} | {top_5} | {covered_top_1} |"
        for name, coverage, top_1, top_3, top_5, covered_top_1 in rows
    )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"本次 Top-1 整体准确率最高的是 **{best['name']}**，达到 **{percentage(best['top_1_accuracy'])}**。硬回退解决了高阶上下文缺失问题；自适应插值进一步避免模型过度相信只出现过少数几次的双字符上下文。",
            "",
            "## 验证集参数选择",
            "",
            "| 插值强度 | Top-1 | Top-3 | Top-5 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for trial in report["validation_trials"]:
        lines.append(
            f"| {trial['strength']} | {percentage(trial['top_1_accuracy'])} | "
            f"{percentage(trial['top_3_accuracy'])} | "
            f"{percentage(trial['top_5_accuracy'])} |"
        )
    lines.extend(["", "## 部分测试示例", ""])
    for result in results.values():
        lines.extend([f"### {result['name']}", "", "正确示例：", ""])
        for example in result["examples"]["correct"][:3]:
            lines.append(
                f"- `{example['context']}` → 预测 `{example['predicted']}`，"
                f"实际 `{example['actual']}`"
            )
        lines.extend(["", "错误示例：", ""])
        for example in result["examples"]["wrong"][:3]:
            lines.append(
                f"- `{example['context']}` → 预测 `{example['predicted']}`，"
                f"实际 `{example['actual']}`"
            )
        lines.append("")
    return "\n".join(lines)


def print_result(result):
    """把一个策略的核心指标打印到终端。"""
    print(f"{result['name']}：")
    print(f"  覆盖率：{percentage(result['coverage'])}")
    print(f"  Top-1：{percentage(result['top_1_accuracy'])}")
    print(f"  Top-3：{percentage(result['top_3_accuracy'])}")
    print(f"  Top-5：{percentage(result['top_5_accuracy'])}")


def main():
    """执行 数据隔离检查→验证集选参→测试集评测→保存报告。"""
    overlaps = verify_document_splits()
    metadata, models = load_all_models()
    validation_documents = load_documents(VALIDATION_SPLIT_PATH)
    test_documents = load_documents(TEST_DATA_PATH)

    selected_strength, validation_trials = choose_interpolation_strength(
        validation_documents,
        models,
    )
    strategies = build_strategies(models, selected_strength)
    # 字典推导式对所有策略调用同一个 evaluate_strategy。
    test_results = {
        key: evaluate_strategy(name, test_documents, ranker)
        for key, (name, ranker) in strategies.items()
    }

    report = {
        "evaluated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "document_counts": {
            "train": len(load_jsonl_records(TRAIN_SPLIT_PATH)),
            "validation": len(load_jsonl_records(VALIDATION_SPLIT_PATH)),
            "test": len(load_jsonl_records(TEST_DATA_PATH)),
        },
        "document_id_overlaps": overlaps,
        "training_data_sha256": metadata[0]["training_data_sha256"],
        "selected_interpolation_strength": selected_strength,
        "validation_trials": validation_trials,
        "test_results": test_results,
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

    print(f"验证集选择的插值强度：{selected_strength}")
    print("\n测试集最终结果：")
    for result in test_results.values():
        print_result(result)
    print(f"\n详细报告：{MARKDOWN_REPORT_PATH}")


if __name__ == "__main__":
    main()
