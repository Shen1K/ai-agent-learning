"""训练入口：用字符向量和神经网络学习“根据前两个字符预测下一个字符”。

运行方式：
    python3 train_neural_model.py

这个文件只负责组织训练流程。真正的矩阵计算、反向传播和模型保存都在
neural_language_model.py 中，方便初学者把“流程代码”和“算法代码”分开看。

默认执行流程：读取训练/验证文章 → 只用训练集建立词表 → 制作 X/y 题目
→ 创建随机参数 → 多轮训练 → 恢复验证集最好的一轮 → 保存模型。
"""

import argparse
import json

from data_pipeline import (
    SPLIT_MANIFEST_PATH,
    TRAIN_SPLIT_PATH,
    VALIDATION_SPLIT_PATH,
    load_documents,
)
from neural_language_model import (
    FeedForwardLanguageModel,
    NEURAL_METADATA_PATH,
    NEURAL_WEIGHTS_PATH,
    build_training_examples,
    build_vocabulary,
    save_neural_model,
    train_model,
)


def parse_arguments():
    """集中声明可调参数；第一次运行时可以完全不传参数。

    argparse 会把终端文本转换成 Python 值。例如 `--epochs 5` 会得到
    arguments.epochs == 5（int），不用我们手工解析 sys.argv。
    """
    parser = argparse.ArgumentParser(
        description="训练一个只依赖 NumPy 的字符级神经语言模型"
    )
    # default 是省略参数时使用的值，type=int 负责检查并转换输入。
    parser.add_argument("--vocabulary-size", type=int, default=512)
    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument(
        "--optimizer",
        choices=("adam", "sgd"),
        default="adam",
        help="默认 Adam；可传 sgd 复现最朴素梯度下降",
    )
    parser.add_argument(
        "--maximum-training-examples",
        type=int,
        # 0 在 main() 中会被转换为 None，表示不限制、使用全部题目。
        default=0,
        help="限制练习项目的训练时间；传入 0 表示使用全部样本",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def known_character_rate(documents, token_to_id):
    """统计文本中有多少字符拥有自己的词表编号，而不是 <UNK>。

    `character in token_to_id` 返回 bool，sum 会把 True 当1、False当0。
    返回 0~1 的比例，只是数据诊断指标，不参与梯度训练。
    """
    total = sum(len(document) for document in documents)
    known = sum(
        character in token_to_id
        for document in documents
        for character in document
    )
    return known / total if total else 0


def main():
    """执行神经模型的完整训练和保存流程。"""
    arguments = parse_arguments()
    training_documents = load_documents(TRAIN_SPLIT_PATH)
    validation_documents = load_documents(VALIDATION_SPLIT_PATH)

    # 词表必须只根据训练集建立。验证集只负责检查模型是否能泛化。
    # 一个函数返回三个值，Python 会按位置把它们解包到三个变量。
    token_to_id, id_to_token, character_counts = build_vocabulary(
        training_documents,
        maximum_size=arguments.vocabulary_size,
    )

    # Python 中数字0是假值，因此 `0 or None` 得到 None；正数则保留自身。
    maximum_examples = arguments.maximum_training_examples or None
    # train_x 是二维整数数组，train_y 是一维正确答案数组。
    train_x, train_y = build_training_examples(
        training_documents,
        token_to_id,
        context_length=2,
        maximum_examples=maximum_examples,
    )
    validation_x, validation_y = build_training_examples(
        validation_documents,
        token_to_id,
        context_length=2,
    )

    print(f"训练文章：{len(training_documents)} 篇")
    print(f"验证文章：{len(validation_documents)} 篇")
    print(f"词表大小：{len(id_to_token)}（其中 1 个是 <UNK>）")
    print(f"训练集字符覆盖率：{known_character_rate(training_documents, token_to_id):.2%}")
    print(f"验证集字符覆盖率：{known_character_rate(validation_documents, token_to_id):.2%}")
    print(f"训练题目：{len(train_x):,} 道")
    print(f"验证题目：{len(validation_x):,} 道\n")

    # 此时模型只有随机初始化参数，还没有学到任何语言规律。
    model = FeedForwardLanguageModel(
        vocabulary_size=len(id_to_token),
        context_length=2,
        embedding_size=arguments.embedding_size,
        hidden_size=arguments.hidden_size,
        seed=arguments.seed,
    )
    # train_model 会原地修改 model 中的数组，并返回每轮指标列表。
    history = train_model(
        model,
        train_x,
        train_y,
        validation_x,
        validation_y,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
        optimizer_name=arguments.optimizer,
    )

    # 读取数据指纹，让模型文件能追溯自己使用的准确训练集版本。
    split_manifest = json.loads(
        SPLIT_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    training_metadata = {
        "training_data_file": "data/processed/train.jsonl",
        "validation_data_file": "data/processed/validation.jsonl",
        "training_data_sha256": split_manifest["train_sha256"],
        "validation_data_sha256": split_manifest["validation_sha256"],
        "training_document_count": len(training_documents),
        "validation_document_count": len(validation_documents),
        "training_example_count": len(train_x),
        "validation_example_count": len(validation_x),
        "original_character_vocabulary_size": len(character_counts),
        "training_known_character_rate": known_character_rate(
            training_documents,
            token_to_id,
        ),
        "validation_known_character_rate": known_character_rate(
            validation_documents,
            token_to_id,
        ),
        "epochs": arguments.epochs,
        "batch_size": arguments.batch_size,
        "initial_learning_rate": arguments.learning_rate,
        "optimizer": arguments.optimizer,
        "maximum_training_examples": maximum_examples,
        "random_seed": arguments.seed,
        # min(..., key=...) 按验证损失而不是按字典默认规则找最小项。
        "selected_checkpoint_epoch": min(
            history,
            key=lambda item: item["validation_loss"],
        )["epoch"],
    }
    save_neural_model(
        model,
        id_to_token,
        training_metadata,
        history,
    )

    print("\n神经模型训练完成：")
    print(f"权重（二进制矩阵）：{NEURAL_WEIGHTS_PATH}")
    print(f"结构与训练记录（JSON）：{NEURAL_METADATA_PATH}")


if __name__ == "__main__":
    main()
