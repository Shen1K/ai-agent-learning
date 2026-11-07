"""模型能力：训练、保存、加载和组合字符级 N-gram 语言模型。

本文件只关心“模型如何学习和预测”，不读取具体数据集，也不负责评测。
这种拆分让每个文件只有一个主要职责，初学时更容易定位问题。

先记住本文件最重要的数据结构：

    {
        "中国": Counter({"人": 10, "的": 6}),
        "语言": Counter({"模": 8, "学": 2}),
    }

外层 dict 的键是上下文，内层 Counter 保存“下一个字符→出现次数”。
训练就是增加这些整数；预测就是把整数除以总数，再排序或采样。
"""

import json
import random
from collections import Counter, defaultdict

from data_pipeline import PROJECT_ROOT


MODEL_DIRECTORY = PROJECT_ROOT / "models"
UNIGRAM_MODEL_PATH = MODEL_DIRECTORY / "unigram_model.json"
BIGRAM_MODEL_PATH = MODEL_DIRECTORY / "character_model.json"
TRIGRAM_MODEL_PATH = MODEL_DIRECTORY / "two_character_model.json"


# 用整数表示上下文长度，字符串表示保存到文件中的可读模型名称。
MODEL_TYPE_NAMES = {
    0: "character_unigram",
    1: "character_bigram",
    2: "character_trigram",
}


def train_ngram_model(documents, context_length):
    """训练指定上下文长度的字符模型。

    context_length=0：不看前文，统计所有字符，称为 Unigram。
    context_length=1：看前一个字符，称为 Bigram。
    context_length=2：看前两个字符，称为 Trigram。

    注意：每篇文章单独遍历。我们不会把上一篇文章的结尾与下一篇文章
    的开头连接起来，因此不会产生虚假的跨文章字符关系。

    documents 是 list[str]；返回值是 dict[str, Counter]。
    例如 context_length=2 时，“中国人”产生 context="中国"、答案="人"。
    """
    if context_length not in MODEL_TYPE_NAMES:
        raise ValueError("本课程目前只支持 0、1、2 三种上下文长度")

    # defaultdict(Counter) 的作用：第一次访问新上下文时，自动创建空 Counter。
    # 因此不需要每次先写 `if context not in counts_by_context`。
    counts_by_context = defaultdict(Counter)

    for document in documents:
        if context_length == 0:
            # 空字符串表示“不使用任何上下文”。
            counts_by_context[""].update(document)
            continue

        # index 指向“要预测的字符”。从 context_length 开始，保证左侧
        # 已经拥有足够多的上下文字符。
        for index in range(context_length, len(document)):
            # 切片左边包含、右边不包含。例如 index=2、长度=2 时取 [0:2]。
            context = document[index - context_length : index]
            next_character = document[index]
            counts_by_context[context][next_character] += 1

    # 转成普通 dict 后，模型在预测阶段不会因误访问而悄悄增加新键。
    return dict(counts_by_context)


def train_unigram_model(documents):
    """训练全局字符模型，作为最后一层保底。"""
    return train_ngram_model(documents, context_length=0)


def train_single_character_model(documents):
    """保留原来的单字符预测逻辑。"""
    return train_ngram_model(documents, context_length=1)


def train_two_character_model(documents):
    """训练观察前两个字符的模型。"""
    return train_ngram_model(documents, context_length=2)


def save_model(model, path, context_length, training_information):
    """把一个 N-gram 模型保存为适合学习查看的 JSON。

    Counter 是 dict 的子类，虽然也能序列化，这里仍显式转换成普通 dict，
    让保存格式不依赖 Python 专用类型，其他语言也能读取。
    """
    model_file = {
        "format_version": 2,
        "model_type": MODEL_TYPE_NAMES[context_length],
        "context_length": context_length,
        "training_data_file": training_information["training_data_file"],
        "training_data_sha256": training_information["training_data_sha256"],
        "training_document_count": training_information["document_count"],
        "training_character_count": training_information["character_count"],
        # Python 的 `a if condition else b` 是条件表达式。
        # Unigram 的全部字符位于空上下文；其他模型沿用训练信息里的词表数。
        "vocabulary_size": len(model.get("", {}))
        if context_length == 0
        else training_information["vocabulary_size"],
        "stored_context_count": len(model),
        # 这是字典推导式：遍历旧字典并生成一个新字典。
        "parameters": {
            context: dict(next_character_counts)
            for context, next_character_counts in model.items()
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model_file, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_model(path):
    """从 JSON 恢复模型说明和 Counter 参数。

    返回 (metadata, model) 二元组：前者是说明信息，后者才是预测参数。
    JSON 只能保存普通字典，所以加载后重新包装成 Counter 以使用 most_common。
    """
    metadata = json.loads(path.read_text(encoding="utf-8"))
    supported_types = set(MODEL_TYPE_NAMES.values())
    if metadata["model_type"] not in supported_types:
        raise ValueError(f"不支持的模型类型：{metadata['model_type']}")

    model = {
        context: Counter(next_character_counts)
        for context, next_character_counts
        in metadata["parameters"].items()
    }
    return metadata, model


def counts_for_history(history, model, context_length):
    """根据历史文字找到对应的计数表。

    找到时返回 Counter；没有该上下文或历史太短时返回 None。
    history[-2:] 表示字符串最后两个字符，负数下标从末尾开始计算。
    """
    if context_length == 0:
        return model.get("")
    if len(history) < context_length:
        return None
    return model.get(history[-context_length:])


def probabilities_from_counts(counts):
    """把出现次数转换为总和为1的概率分布。

    例如 {"人": 3, "家": 1} 会变成 {"人": 0.75, "家": 0.25}。
    空计数返回空字典，而不是除以零。
    """
    if not counts:
        return {}
    total = sum(counts.values())
    return {
        character: count / total for character, count in counts.items()
    }


def rank_counts(counts, top_k=5):
    """返回概率最高的前 k 个候选字符。

    返回值如 [("人", 0.75), ("家", 0.25)]。元组第一项是字符，
    第二项是概率。Counter.most_common 已经按次数从高到低排列。
    """
    if not counts:
        return []
    total = sum(counts.values())
    return [
        (character, count / total)
        for character, count in counts.most_common(top_k)
    ]


def rank_pure_model(history, model, context_length, top_k=5):
    """只使用指定层级的模型，不进行回退。

    这是最严格的基线：Trigram 没见过某个双字符组合时直接返回空列表。
    """
    counts = counts_for_history(history, model, context_length)
    return rank_counts(counts, top_k)


def select_backoff_counts(history, models):
    """Trigram 找不到时退回 Bigram，最后退回 Unigram。

    返回的第二个值表示最终使用了几字符上下文，便于调试和解释。
    """
    # tuple (2, 1, 0) 明确规定尝试顺序：先精确，再逐级放宽条件。
    for context_length in (2, 1, 0):
        counts = counts_for_history(
            history,
            models[context_length],
            context_length,
        )
        if counts:
            return counts, context_length
    return None, None


def rank_with_backoff(history, models, top_k=5):
    """先选择可用层级，再把该层计数转换为 Top-k 概率。"""
    # `_` 是合法变量名，惯例表示“这个返回值本处不需要使用”。
    counts, _ = select_backoff_counts(history, models)
    return rank_counts(counts, top_k)


def interpolation_weights(history, models, strength):
    """根据上下文出现次数计算三层模型的自适应权重。

    strength 越大，模型越保守，越依赖低阶模型。
    某段上下文出现次数越多，对对应高阶模型的信任越高。

    返回 {0: Unigram权重, 1: Bigram权重, 2: Trigram权重}，三项相加为1。
    置信度公式 count/(count+strength)：count=0 时为0，次数很大时接近1。
    """
    if strength <= 0:
        raise ValueError("插值强度必须大于0")

    bigram_counts = counts_for_history(history, models[1], 1)
    trigram_counts = counts_for_history(history, models[2], 2)
    # 条件表达式避免在 None 上调用 .values()。
    bigram_total = sum(bigram_counts.values()) if bigram_counts else 0
    trigram_total = sum(trigram_counts.values()) if trigram_counts else 0

    bigram_confidence = bigram_total / (bigram_total + strength)
    trigram_confidence = trigram_total / (trigram_total + strength)

    # 先混合 Unigram 和 Bigram，再让 Trigram 与这个低阶结果混合。
    return {
        0: (1 - trigram_confidence) * (1 - bigram_confidence),
        1: (1 - trigram_confidence) * bigram_confidence,
        2: trigram_confidence,
    }


def probability_in_model(character, history, model, context_length):
    """查询某个字符在某一层模型中的概率；未见上下文或字符都返回0。"""
    counts = counts_for_history(history, model, context_length)
    if not counts:
        return 0
    return counts.get(character, 0) / sum(counts.values())


def rank_with_interpolation(history, models, strength, top_k=5):
    """综合三层模型，返回概率最高的候选字符。

    为了避免每次都遍历全部四千多个字符，只需考虑：
    - 高阶模型当前上下文中出现过的字符；
    - Unigram 中最常见的前 k 个字符。

    没有高阶加分的其他字符不可能超过 Unigram 的前 k 名。
    """
    unigram_counts = counts_for_history(history, models[0], 0)
    bigram_counts = counts_for_history(history, models[1], 1)
    trigram_counts = counts_for_history(history, models[2], 2)

    # 花括号推导式生成 set。集合自动去重，适合合并三层候选。
    candidates = {
        character for character, _ in unigram_counts.most_common(top_k)
    }
    if bigram_counts:
        candidates.update(bigram_counts)
    if trigram_counts:
        candidates.update(trigram_counts)

    weights = interpolation_weights(history, models, strength)
    counts_by_length = {
        0: unigram_counts,
        1: bigram_counts,
        2: trigram_counts,
    }
    # 每层分母在本次预测中不会变化，只计算一次即可。
    totals = {
        context_length: sum(counts.values()) if counts else 0
        for context_length, counts in counts_by_length.items()
    }
    # 最终列表中的每项形如 (字符, 三层加权概率)。
    scored_candidates = []
    for character in candidates:
        # 这就是插值公式：P = w0*P0 + w1*P1 + w2*P2。
        score = sum(
            weights[context_length]
            * (
                counts_by_length[context_length].get(character, 0)
                / totals[context_length]
                if totals[context_length]
                else 0
            )
            for context_length in (0, 1, 2)
        )
        scored_candidates.append((character, score))

    # 字符作为第二排序条件，让概率相同时的结果保持稳定。
    scored_candidates.sort(key=lambda item: (-item[1], item[0]))
    return scored_candidates[:top_k]


def interpolated_distribution(history, models, strength):
    """计算完整插值概率分布，供随机采样使用。

    与 rank_with_interpolation 的区别：这里为整个 Unigram 词表计算概率，
    速度稍慢但概率总和为1，适合作为 random.choices 的采样权重。
    """
    unigram_counts = counts_for_history(history, models[0], 0)
    bigram_counts = counts_for_history(history, models[1], 1)
    trigram_counts = counts_for_history(history, models[2], 2)
    candidates = set(unigram_counts)
    weights = interpolation_weights(history, models, strength)
    counts_by_length = {
        0: unigram_counts,
        1: bigram_counts,
        2: trigram_counts,
    }
    totals = {
        context_length: sum(counts.values()) if counts else 0
        for context_length, counts in counts_by_length.items()
    }
    distribution = {}

    for character in candidates:
        distribution[character] = sum(
            weights[context_length]
            * (
                counts_by_length[context_length].get(character, 0)
                / totals[context_length]
                if totals[context_length]
                else 0
            )
            for context_length in (0, 1, 2)
        )
    return distribution


def choose_from_counts(counts, random_generator=None):
    """不传随机数生成器时选第一名；传入时按次数随机采样。

    `is None` 用来准确判断没有传生成器；不能写 `if not random_generator`，
    因为这里关心的是 None 这个特殊值，而不是一般真假值。
    """
    if not counts:
        return None
    if random_generator is None:
        return counts.most_common(1)[0][0]
    return random_generator.choices(
        list(counts.keys()),
        weights=list(counts.values()),
        k=1,
    )[0]


def choose_from_distribution(distribution, random_generator=None):
    """从概率分布中选择一个字符；支持最高概率和随机采样两种模式。"""
    if not distribution:
        return None
    if random_generator is None:
        return max(distribution, key=distribution.get)
    return random_generator.choices(
        list(distribution.keys()),
        weights=list(distribution.values()),
        k=1,
    )[0]


def choose_pure_next(
    history,
    model,
    context_length,
    random_generator=None,
):
    """为 generate_text 适配纯 N-gram 的统一调用接口。"""
    counts = counts_for_history(history, model, context_length)
    return choose_from_counts(counts, random_generator)


def choose_backoff_next(history, models, random_generator=None):
    """为 generate_text 适配硬回退策略。"""
    counts, _ = select_backoff_counts(history, models)
    return choose_from_counts(counts, random_generator)


def choose_interpolated_next(
    history,
    models,
    strength,
    random_generator=None,
):
    """为 generate_text 适配概率插值策略。"""
    distribution = interpolated_distribution(history, models, strength)
    return choose_from_distribution(distribution, random_generator)


def generate_text(start_text, choose_next, length=80):
    """反复调用某种选择策略，直到达到目标长度或无法继续。

    choose_next 是函数，不是字符。把函数作为参数传进来后，同一个生成循环
    可以替换纯模型、回退或插值策略，这在 Python 中称为函数是一等对象。
    """
    if not start_text:
        raise ValueError("起始文字不能为空")

    result = start_text
    while len(result) < length:
        next_character = choose_next(result)
        if next_character is None:
            break
        # 字符串不可原地修改；+= 实际会创建连接后的新字符串再赋值。
        result += next_character
    return result


def make_random_generator(seed=7):
    """创建独立随机数生成器，保证相同 seed 得到相同采样序列。"""
    return random.Random(seed)
