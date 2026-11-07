"""可读、可替换的本地RAG检索层（Python标准库，无模型下载和网络）。

先学习这条链：原文段落 → 小片段 → BM25排序 → 取回父段 → 带来源返回。
BM25是真正的词项检索，不是Embedding，也不是用假向量伪装语义理解。
中文使用相邻双字作为词项，只是免依赖教学选择，不等于成熟中文分词。

本层不写文案；Agent拿到检索结果后才调用生成模型。snapshot把资料与配置
冻结进任务，保证恢复时不会悄悄读取另一版资料。大语料生产系统应保存版本化
索引引用，而不是像本教学例一样把小资料集复制进SQLite状态。
"""

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

from .contracts import decode, exact, require, string


ENGINE_VERSION = "bm25-char-bigram-v1"
DATA_DIR = Path(__file__).with_name("knowledge")
MAX_SEARCHES = 4
TOP_K = 3
MAX_CONTEXT_CHARS = 1600  # 字符预算，不冒充token或人民币预算。


def terms(text):
    """将文本变成可计数词项：英文/型号保留整词，中文取相邻双字。

    例如‘办公间隙’→‘办公’、‘公间’、‘间隙’。它不会知道‘喘息’与
    ‘片刻休息’含义接近；评测中专门保留这个失败以引出Embedding。
    """
    result = []
    for part in re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part) and len(part) > 1:
            result.extend(part[i:i + 2] for i in range(len(part) - 1))
        else:
            result.append(part)
    return result


def fingerprint(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class KnowledgeBase:
    """输入必须来自程序配置的资料，不允许模型提供任意文件路径或URL。"""

    def __init__(self, documents):
        require(type(documents) is list and len(documents) <= 100, "教学资料最多100份")
        seen, active = set(), set()
        for doc in documents:
            exact(doc, ("id", "version", "product_id", "status", "kind", "title", "required", "sections"), "知识文档")
            for key in ("id", "version", "product_id"):
                string(doc[key], key, 80)
                require(re.fullmatch(r"[a-zA-Z0-9_-]+", doc[key]) is not None, "知识编号仅允许英文数字及-_")
            string(doc["title"], "文档标题", 120)
            require(doc["status"] in ("active", "archived"), "资料状态无效")
            require(doc["kind"] in ("policy", "case", "feedback"), "资料类型无效")
            require(type(doc["required"]) is bool, "required必须是布尔值")
            require(not doc["required"] or doc["kind"] == "policy", "只有规范可标为必需")
            identity = (doc["product_id"], doc["id"], doc["version"])
            require(identity not in seen, "文档版本重复")
            seen.add(identity)
            if doc["status"] == "active":
                current = (doc["product_id"], doc["id"])
                require(current not in active, "同一文档存在多个active版本，请资料维护者先裁决")
                active.add(current)
            require(type(doc["sections"]) is list and 1 <= len(doc["sections"]) <= 12, "文档须有1至12节")
            sections = set()
            for section in doc["sections"]:
                exact(section, ("id", "heading", "text"), "段落")
                string(section["id"], "段落编号", 60)
                require(re.fullmatch(r"[a-zA-Z0-9_-]+", section["id"]) is not None, "段落编号无效")
                require(section["id"] not in sections, "段落编号重复")
                sections.add(section["id"])
                string(section["heading"], "段落标题", 120)
                string(section["text"], "段落原文", 1400)
        self.documents = deepcopy(documents)

    @classmethod
    def bundled(cls):
        # 只加载包内已知的虚构资料。没有用户上传或扫描任意目录的入口。
        return cls(decode((DATA_DIR / "documents.json").read_text(encoding="utf-8"))["documents"])

    def snapshot(self, product_id):
        # 先限定资料范围再建索引，避免其他商品内容或旧版影响检索和IDF。
        # 这只是本地教学的商品范围隔离，不是多租户账号授权。
        documents = [d for d in self.documents
                     if d["product_id"] == product_id and d["status"] == "active"]
        require(bool(documents), "该商品没有已配置的有效知识资料；请关闭RAG或配置资料")
        require(len(json.dumps(documents, ensure_ascii=False)) <= 32000, "任务知识快照超过教学上限")
        snapshot = {"engine_version": ENGINE_VERSION, "strategy": "parent_context",
                    "product_id": product_id, "documents": deepcopy(documents)}
        snapshot["snapshot_id"] = fingerprint(snapshot)
        require(sum(len(x["text"]) for x in required_context(snapshot)) <= MAX_CONTEXT_CHARS,
                "必需规范超过预算，不能静默丢弃，请先缩小任务范围")
        return snapshot


def sections(snapshot):
    """把嵌套文档展开为段落记录；source_id带版本，不能只用相同文件名。"""
    result = []
    for doc in snapshot["documents"]:
        for section in doc["sections"]:
            result.append({"source_id": f"{doc['id']}@{doc['version']}:{section['id']}",
                           "document_id": doc["id"], "version": doc["version"],
                           "section_id": section["id"], "title": doc["title"],
                           "heading": section["heading"], "kind": doc["kind"],
                           "required": doc["required"], "text": section["text"],
                           "start": 0, "end": len(section["text"])})
    return result


def required_context(snapshot):
    # 必需规范不靠‘搜索到了才遵守’。它们在get_materials时始终提供。
    return [x for x in sections(snapshot) if x["required"]]


def chunks(snapshot):
    """段落按换行拆小块，保留字符区间，可回到同一版本原文。

    数据文件已人工整理好章节。这一步不包含PDF/OCR解析；不能把准备好的
    JSON叫作已解决所有文档解析。本例换行恰好分开规则和例外以观察丢信息。
    """
    result = []
    for parent in sections(snapshot):
        for match in re.finditer(r"[^\n]+", parent["text"]):
            result.append({"parent": parent, "start": match.start(), "end": match.end(),
                           "text": match.group(), "chunk_id": f"{parent['source_id']}:{match.start()}"})
    return result


def search(snapshot, query, strategy=None, top_k=TOP_K, max_chars=MAX_CONTEXT_CHARS):
    """对同一候选排序比较small_chunks与parent_context，只改变返回范围。

    BM25分数由词项稀有程度、出现次数和长度构成。它是排序依据，不是答案
    正确概率。空结果仅表示这次检索没找到，不能断言整个业务没有相关资料。
    """
    require(snapshot["engine_version"] == ENGINE_VERSION, "检索实现版本变化，请新建任务")
    string(query, "检索问题", 240)
    strategy = strategy or snapshot["strategy"]
    require(strategy in ("small_chunks", "parent_context"), "未知检索方案")
    require(type(top_k) is int and 1 <= top_k <= 10, "top_k须为1至10")
    require(type(max_chars) is int and 1 <= max_chars <= 10000, "上下文字符预算无效")
    indexed = chunks(snapshot)
    counts = [Counter(terms(x["text"])) for x in indexed]
    query_terms = set(terms(query))
    n = len(counts)
    mean_length = sum(sum(c.values()) for c in counts) / max(n, 1)
    frequencies = Counter(t for c in counts for t in c)
    ranked = []
    for chunk, count in zip(indexed, counts):
        score = 0.0
        for token in query_terms:
            frequency = count[token]
            if not frequency:
                continue
            # k1=1.2、b=0.75是本教学基线配置，不是本业务已调优参数。
            idf = math.log(1 + (n - frequencies[token] + 0.5) / (frequencies[token] + 0.5))
            normalization = 1.2 * (1 - 0.75 + 0.75 * sum(count.values()) / max(mean_length, 1))
            score += idf * frequency * 2.2 / (frequency + normalization)
        if score > 0:
            ranked.append((score, chunk))
    ranked.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))
    hits, seen, used, skipped = [], set(), 0, 0
    for score, chunk in ranked:
        parent = chunk["parent"]
        key = parent["source_id"] if strategy == "parent_context" else chunk["chunk_id"]
        if key in seen:
            continue
        hit = deepcopy(parent)
        if strategy == "small_chunks":
            hit.update(text=chunk["text"], start=chunk["start"], end=chunk["end"])
        if used + len(hit["text"]) > max_chars:
            # 不直接截断半个条件；报告预算跳过，调用者可缩小问题或调整预算。
            skipped += 1
            continue
        hit.update(score=round(score, 6), matched_chunk_id=chunk["chunk_id"])
        hits.append(hit)
        used += len(hit["text"])
        seen.add(key)
        if len(hits) == top_k:
            break
    return {"query": query, "strategy": strategy, "snapshot_id": snapshot["snapshot_id"],
            "engine_version": ENGINE_VERSION, "hits": hits, "returned_chars": used,
            "budget_skipped": skipped, "status": "found" if hits else "no_match",
            "notice": "相关性不等于正确性；未命中不等于资料不存在；案例不能作为商品事实。"}


def candidate_context(rag_state):
    """候选保存时冻结‘当时可用的证据’，不是宣称模型实际引用了每一项。

    必需规范优先；其他片段取最近检索结果，按来源去重且限制总长度。
    评审不接收检索问题、排序分数或生成者理由，只得到原文与来源。
    """
    selected = deepcopy(required_context(rag_state["snapshot"]))
    seen = {x["source_id"] for x in selected}
    used = sum(len(x["text"]) for x in selected)
    for retrieval in reversed(rag_state["retrievals"]):
        for hit in retrieval["hits"]:
            if hit["source_id"] in seen or used + len(hit["text"]) > MAX_CONTEXT_CHARS:
                continue
            item = {k: v for k, v in hit.items() if k not in ("score", "matched_chunk_id")}
            selected.append(item)
            seen.add(hit["source_id"])
            used += len(hit["text"])
    # 简短字段E1用于评审引文，真正来源仍保留source_id／版本／区间。
    return {"E" + str(i): item for i, item in enumerate(selected, 1)}
