"""课程入口：准备数据、训练三层模型，并展示四种生成策略。

运行方式：
    python3 main.py

为了让每个能力边界清楚，本文件只负责编排流程：
- data_pipeline.py 负责数据清洗与拆分；
- ngram_language_model.py 负责训练、保存、预测和生成；
- evaluate_models.py 负责验证参数与最终测试。

这里没有复杂算法，主要展示工程中的“编排层”：数据函数和模型函数已经
分别写好，main() 只决定调用顺序并打印阶段结果。阅读时建议先从 main()
开始，再顺着函数名跳到对应文件。
"""

from data_pipeline import (
    PROJECT_ROOT,
    TRAIN_SPLIT_PATH,
    load_documents,
    prepare_dataset_splits,
)
from ngram_language_model import (
    BIGRAM_MODEL_PATH,
    TRIGRAM_MODEL_PATH,
    UNIGRAM_MODEL_PATH,
    choose_backoff_next,
    choose_interpolated_next,
    choose_pure_next,
    generate_text,
    load_model,
    make_random_generator,
    save_model,
    train_single_character_model,
    train_two_character_model,
    train_unigram_model,
)


# 这个值不是最终调好的参数，只用于训练后的生成演示。
# 正式参数由 evaluate_models.py 在验证集上选择。
DEMO_INTERPOLATION_STRENGTH = 0.5


def build_training_information(documents, split_manifest):
    """整理需要写入模型文件的数据来源说明。

    documents 是正文字符串列表；split_manifest 是数据处理阶段生成的字典。
    返回的新字典会复制进每个模型文件，用来证明三个模型来自同一训练集。
    """
    # `*(...)` 把多组 set 作为多个位置参数传给 union。
    # 最终 all_characters 是训练语料中所有不重复字符的集合。
    all_characters = set().union(*(set(document) for document in documents))
    return {
        "training_data_file": TRAIN_SPLIT_PATH.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "training_data_sha256": split_manifest["train_sha256"],
        "document_count": len(documents),
        "character_count": sum(len(document) for document in documents),
        "vocabulary_size": len(all_characters),
    }


def train_and_save_models(documents, training_information):
    """分别训练三层模型，并保存为三个独立文件。

    三个模型使用完全相同的训练文章，区别只是观察的上下文长度。
    独立保存让我们可以清楚比较模型大小和参数结构。
    """
    # 三个数字键就是上下文长度。放在同一个字典中，后续可以循环处理，
    # 避免为三个模型重复写几乎相同的保存代码。
    models = {
        0: train_unigram_model(documents),
        1: train_single_character_model(documents),
        2: train_two_character_model(documents),
    }
    paths = {
        0: UNIGRAM_MODEL_PATH,
        1: BIGRAM_MODEL_PATH,
        2: TRIGRAM_MODEL_PATH,
    }

    # dict.items() 每轮返回 (键, 值)，这里分别解包成两个变量。
    for context_length, model in models.items():
        save_model(
            model,
            paths[context_length],
            context_length,
            training_information,
        )
    return paths


def load_saved_models(paths):
    """从硬盘重新加载，保证后面的生成使用的确实是模型文件。

    这一步看似多余，却能立即发现保存格式和加载逻辑是否互相兼容。
    返回的 metadata 放说明信息，models 放真正的 Counter 参数。
    """
    models = {}
    metadata = {}
    for context_length, path in paths.items():
        model_metadata, model = load_model(path)
        metadata[context_length] = model_metadata
        models[context_length] = model
    return metadata, models


def generation_strategies(models, random_seed=None):
    """创建四种可互换的“选择下一个字符”能力。

    random_seed=None 表示永远选择概率最高的字符。
    传入数字表示按概率采样，并保证相同数字可以复现实验。
    每种策略使用独立随机数生成器，避免前一种策略消耗随机数后
    影响后一种策略的结果。

    返回值是 {显示名称: 函数}。lambda 是没有名字的短函数；每个 lambda
    都接收 history，再把模型和配置补齐后调用对应预测函数。
    """
    # 下面是 Python 条件表达式：条件成立取前值，否则取 None。
    single_random = (
        make_random_generator(random_seed) if random_seed is not None else None
    )
    double_random = (
        make_random_generator(random_seed) if random_seed is not None else None
    )
    backoff_random = (
        make_random_generator(random_seed) if random_seed is not None else None
    )
    interpolation_random = (
        make_random_generator(random_seed) if random_seed is not None else None
    )

    return {
        "单字符模型": lambda history: choose_pure_next(
            history,
            models[1],
            context_length=1,
            random_generator=single_random,
        ),
        "双字符模型": lambda history: choose_pure_next(
            history,
            models[2],
            context_length=2,
            random_generator=double_random,
        ),
        "三级硬回退": lambda history: choose_backoff_next(
            history,
            models,
            random_generator=backoff_random,
        ),
        "三级自适应插值": lambda history: choose_interpolated_next(
            history,
            models,
            strength=DEMO_INTERPOLATION_STRENGTH,
            random_generator=interpolation_random,
        ),
    }


def show_generation_comparison(models, start_text="中国", length=80):
    """使用相同开头和长度，并排比较四种策略。

    第一组总选第一名，容易陷入重复循环；第二组按概率采样，文本更多样。
    两组不能直接比较“准确率”，这里只用于观察生成行为。
    """
    print("\n总是选择概率最高的字符：")
    for name, choose_next in generation_strategies(models).items():
        print(f"{name}：{generate_text(start_text, choose_next, length)}")

    print("\n按照概率随机生成（固定随机种子7）：")
    for name, choose_next in generation_strategies(
        models,
        random_seed=7,
    ).items():
        print(f"{name}：{generate_text(start_text, choose_next, length)}")


def main():
    """按 数据准备→训练保存→重新加载→生成演示 的顺序执行。"""
    # 第一步：重新执行稳定的数据拆分。相同原始数据会得到相同结果。
    split_manifest = prepare_dataset_splits()
    documents = load_documents(TRAIN_SPLIT_PATH)
    training_information = build_training_information(
        documents,
        split_manifest,
    )

    print(f"训练文章：{training_information['document_count']} 篇")
    print(f"训练字符：{training_information['character_count']:,}")
    print(f"字符种类：{training_information['vocabulary_size']:,}")

    # 第二步：训练并保存 Unigram、Bigram、Trigram。
    paths = train_and_save_models(documents, training_information)
    metadata, models = load_saved_models(paths)

    print("\n模型已保存并重新加载：")
    for context_length in (0, 1, 2):
        print(
            f"上下文长度 {context_length}："
            f"{metadata[context_length]['stored_context_count']:,} 个上下文，"
            f"{paths[context_length].stat().st_size:,} 字节"
        )

    # 第三步：展示纯模型、硬回退和插值生成的区别。
    show_generation_comparison(models)

    print("\n训练与生成完成。运行 python3 evaluate_models.py 进行正式评测。")


# 被测试代码 import 时不会训练；直接执行 python3 main.py 时才训练。
if __name__ == "__main__":
    main()
