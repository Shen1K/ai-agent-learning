"""数据能力：读取百科记录、规范化正文，并按文章拆分训练集和验证集。

为什么按“文章”拆分，而不是把所有文字混在一起再切一刀？
因为同一篇文章的相邻句子非常相似。如果一半进入训练集、另一半进入
验证集，验证成绩会虚高。按文章拆分可以让验证集真正包含模型没见过的文章。

数据流可以反复对照：

    data/train/chinese_train.jsonl（原始语料池）
        → 清洗和质量过滤
        → 按文章 ID 稳定拆分
        → data/processed/train.jsonl
        → data/processed/validation.jsonl

本文件只处理数据，不训练模型。这样发现数据问题时不用进入算法代码排查。
"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path


# Path 对象比普通路径字符串更方便拼接。`/` 在这里是 Path 重载的运算符，
# 不是数学除法，例如 PROJECT_ROOT / "data" 会得到 data 目录路径。
PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_POOL_PATH = PROJECT_ROOT / "data" / "train" / "chinese_train.jsonl"
PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"
TRAIN_SPLIT_PATH = PROCESSED_DIRECTORY / "train.jsonl"
VALIDATION_SPLIT_PATH = PROCESSED_DIRECTORY / "validation.jsonl"
SPLIT_MANIFEST_PATH = PROCESSED_DIRECTORY / "manifest.json"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "test" / "chinese_test.jsonl"


def normalize_text(text):
    """把不同文章转换成一致的文本形式。

    这里只做三种容易理解且可逆风险较低的处理：

    1. NFKC：把全角英文字母、全角数字等兼容字符统一。
    2. lower：把英文大写统一成小写，减少无意义的字符种类。
    3. 空白规范化：连续空格和换行统一成一个普通空格。

    暂时不做繁体转简体，因为 Python 标准库没有可靠的繁简转换器。
    """
    # 每一步都把处理结果重新赋值给 text，最后返回统一后的字符串。
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chinese_character_ratio(text):
    """计算基本汉字在非空白字符中的比例。

    列表推导式相当于一个紧凑 for 循环：只把非空白字符加入列表。
    返回值是 0~1 的 float，例如 0.8 表示约 80%。
    """
    characters = [character for character in text if not character.isspace()]
    if not characters:
        return 0

    # 在 Python 中 True 可当作 1、False 可当作 0，因此可以直接 sum。
    chinese_count = sum(
        "\u4e00" <= character <= "\u9fff" for character in characters
    )
    return chinese_count / len(characters)


def load_jsonl_records(path):
    """读取 JSONL：文件中的每一行都是一篇独立文章。

    参数 path 是 Path；返回 list[dict]。enumerate(..., start=1) 同时提供
    行号和行内容，报错时可以告诉学习者具体是哪一行损坏。
    """
    records = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path} 第 {line_number} 行不是有效 JSON"
                ) from error
    return records


def prepare_records(records, minimum_length=20, minimum_chinese_ratio=0.25):
    """规范化正文，并过滤过短或中文比例过低的文章。

    返回 (prepared, rejected) 两个列表。prepared 保存可训练的完整记录，
    rejected 只保存被拒绝文章的 ID 和原因，方便检查过滤规则。
    """
    prepared = []
    rejected = []

    for record in records:
        normalized_text = normalize_text(record["text"])
        chinese_ratio = chinese_character_ratio(normalized_text)

        reason = None
        if len(normalized_text) < minimum_length:
            reason = "正文过短"
        elif chinese_ratio < minimum_chinese_ratio:
            reason = "中文比例过低"

        if reason:
            rejected.append({"id": record["id"], "reason": reason})
            continue

        # dict(record) 创建浅拷贝，避免直接修改调用方传进来的原始字典。
        prepared_record = dict(record)
        prepared_record["text"] = normalized_text
        prepared_record["character_count"] = len(normalized_text)
        prepared_record["chinese_character_ratio"] = round(chinese_ratio, 6)
        prepared.append(prepared_record)

    return prepared, rejected


def deterministic_split(records, validation_ratio=0.10):
    """用文章 ID 的哈希值稳定拆分数据。

    相同输入每次都会得到相同结果，同时哈希排序又不会按照文章标题、
    页面编号等明显规律分组。验证集数量固定为全部文章的约 10%。
    """
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio 必须在 0 和 1 之间")

    # 不能使用 Python 内置 hash()，因为它可能在不同进程中变化。
    # SHA-256 对同一文章 ID 始终给出相同字符串，所以拆分可以复现。
    ordered_records = sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"llm-agent-lab-v1:{record['id']}".encode("utf-8")
        ).hexdigest(),
    )
    validation_count = max(1, round(len(ordered_records) * validation_ratio))
    # [:validation_count] 取开头部分；[validation_count:] 取剩余部分。
    validation_records = ordered_records[:validation_count]
    training_records = ordered_records[validation_count:]
    return training_records, validation_records


def jsonl_content(records):
    """把记录列表序列化为 JSONL 字符串，每条记录结尾增加换行。"""
    return "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )


def write_split(path, records):
    """写入一个数据分片，并返回内容的 SHA-256 指纹。"""
    content = jsonl_content(records)
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def prepare_dataset_splits(
    source_path=SOURCE_POOL_PATH,
    validation_ratio=0.10,
    minimum_length=20,
    minimum_chinese_ratio=0.25,
):
    """执行完整的数据准备流程并返回清单。

    这是本文件的“编排函数”：它不实现新的数学算法，而是按顺序调用
    读取、去重检查、清洗、拆分、写文件，并汇总 manifest。
    """
    source_records = load_jsonl_records(source_path)

    # 列表长度与集合长度不同，说明至少有一个 ID 重复。
    source_ids = [record["id"] for record in source_records]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("训练语料池中存在重复文章 ID")

    prepared_records, rejected_records = prepare_records(
        source_records,
        minimum_length,
        minimum_chinese_ratio,
    )
    training_records, validation_records = deterministic_split(
        prepared_records,
        validation_ratio,
    )

    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    train_sha256 = write_split(TRAIN_SPLIT_PATH, training_records)
    validation_sha256 = write_split(
        VALIDATION_SPLIT_PATH,
        validation_records,
    )

    # manifest 是数据说明书。模型训练时会保存这里的哈希，确保可追溯。
    manifest = {
        "source_file": source_path.relative_to(PROJECT_ROOT).as_posix(),
        "source_document_count": len(source_records),
        "accepted_document_count": len(prepared_records),
        "rejected_document_count": len(rejected_records),
        "training_document_count": len(training_records),
        "validation_document_count": len(validation_records),
        "training_character_count": sum(
            len(record["text"]) for record in training_records
        ),
        "validation_character_count": sum(
            len(record["text"]) for record in validation_records
        ),
        "validation_ratio": validation_ratio,
        "minimum_length": minimum_length,
        "minimum_chinese_ratio": minimum_chinese_ratio,
        "normalization": ["NFKC", "lowercase Latin", "collapse whitespace"],
        "split_method": "SHA-256 ordering by document id",
        "train_sha256": train_sha256,
        "validation_sha256": validation_sha256,
        "rejected_examples": rejected_records[:10],
    }
    SPLIT_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_documents(path):
    """只返回正文字符串列表，并继续保持每篇文章相互独立。

    模型代码不关心标题和网址，所以这里把 list[dict] 转为 list[str]。
    仍然不把多篇文章拼接，避免创造“上一篇结尾→下一篇开头”的假关系。
    """
    documents = []
    for record in load_jsonl_records(path):
        normalized_text = normalize_text(record["text"])
        if normalized_text:
            documents.append(normalized_text)
    return documents


def main():
    """单独运行本文件时，重新生成稳定的数据分片并打印摘要。"""
    manifest = prepare_dataset_splits()
    print(f"原始文章：{manifest['source_document_count']} 篇")
    print(f"过滤文章：{manifest['rejected_document_count']} 篇")
    print(f"训练集：{manifest['training_document_count']} 篇")
    print(f"验证集：{manifest['validation_document_count']} 篇")
    print(f"训练字符：{manifest['training_character_count']:,}")
    print(f"验证字符：{manifest['validation_character_count']:,}")
    print(f"位置：{PROCESSED_DIRECTORY}")


# import data_pipeline 时只获得函数；python3 data_pipeline.py 时才执行 main。
if __name__ == "__main__":
    main()
