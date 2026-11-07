"""C11：重读真实原文件，核对当前对话模型给出的候选快照。

运行：python3 lessons/c11_live_source_check.py
第一次提取发生在2026-10-04的当前对话，不是这里的程序生成的。
这个脚本只重放保存的候选并重新核对，不调用模型、不训练、不写资料。
核对文件版本、字段、值、逐字引用及行号；不执行通用文本语义判断。
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_RELATIVE = "data/processed/manifest.json"
SOURCE = ROOT / SOURCE_RELATIVE
CANDIDATES = Path(__file__).with_name("c11_live_candidates.json")

# 本课问题与字段的对应由任务要求预先确定，不听候选自己重新定义。
# 这里只验证当前数据清单，不能推广为所有文件都应使用这些字段名。
QUESTION_FIELDS = {
    "training_count": "training_document_count",
    "training_duration": "training_duration_seconds",
}


def verify_claim(document, original_text, claim):
    """输入独立读取的JSON、原文、候选；输出明确的检查记录。

    document 是由原文件解析的字典，不是模型返回的“来源副本”。
    checks保存已经通过的检查；拒绝时保留原因，不输出未经核对的答案。
    """
    result = {"outcome": "rejected", "checks": [], "reason": ""}
    if not isinstance(claim, dict):
        result["reason"] = "候选必须是字典"
        return result
    question_id = claim.get("question_id")
    if not isinstance(question_id, str) or question_id not in QUESTION_FIELDS:
        result["reason"] = "未知问题编号"
        return result
    field = QUESTION_FIELDS[question_id]
    if claim.get("field") != field:
        result["reason"] = "候选字段与问题要求不符"
        return result
    result["checks"].append("字段与预先确定的问题要求相符")

    if claim.get("status") == "not_found":
        # null是JSON写法，json.loads读取后会变成Python的None。
        if any(key not in claim or claim[key] is not None
               for key in ("value", "quote", "line")):
            result["reason"] = "未找到的结果不应夹带数值或伪造引用"
            return result
        if field in document:
            result["reason"] = "原文件实际存在该字段，不能声称字段不存在"
            return result
        result["checks"].append("重新解析原文件，确认指定字段不存在")
        result["outcome"] = "confirmed_field_absent"
        result["reason"] = "仅确认指定字段缺失；不证明任意原文或其他文件无此信息"
        return result

    if claim.get("status") != "found" or field not in document:
        result["reason"] = "候选状态不合法或原文件不存在所声明的字段"
        return result
    value = claim.get("value")
    # 本课两个查询均限定非负整数；明确拒绝布尔值及数字字符串。
    if type(value) is not int or value < 0 or type(document[field]) is not int:
        result["reason"] = "候选或来源不是本课允许的非负整数记录"
        return result
    if value != document[field]:
        result["reason"] = "候选数值与原文件不一致"
        return result
    result["checks"].append(f"独立读取的字段值与候选一致：{value}")

    # 引用须包含完整字段与数值，不能只引用一个碰巧相同的数字。
    expected_quote = f'"{field}": {value}'
    if claim.get("quote") != expected_quote:
        result["reason"] = "引用不是所需字段和值"
        return result
    line_number = claim.get("line")
    lines = original_text.splitlines()
    if type(line_number) is not int or not 1 <= line_number <= len(lines):
        result["reason"] = "行号无效"
        return result
    # 行号从1开始，列表下标从0开始，所以这里减1。
    # strip去掉缩进；removesuffix仅去掉JSON行末的一个逗号。
    source_line = lines[line_number - 1].strip().removesuffix(",")
    if source_line != expected_quote:
        result["reason"] = "该行原文与引用不一致"
        return result
    result["checks"].append(f"第{line_number}行的原文与引用逐字相符")
    result["outcome"] = "verified_value"
    result["reason"] = "候选通过本课的文件一致性检查"
    return result


def verify_snapshot(snapshot, raw):
    """先固定来源与版本，再逐项验证，防止核对时偷换文件。"""
    actual_hash = hashlib.sha256(raw).hexdigest()
    if snapshot.get("source") != SOURCE_RELATIVE:
        raise ValueError("候选快照声明的来源不是本课指定文件")
    if snapshot.get("source_sha256") != actual_hash:
        raise ValueError("原文件与提取时版本不同：停止核对，请重新读取和提取")
    text = raw.decode("utf-8")
    document = json.loads(text)
    if not isinstance(document, dict):
        raise ValueError("原文件必须是JSON对象")
    claims = snapshot.get("claims")
    if not isinstance(claims, list) or len(claims) != 2:
        raise ValueError("本课快照必须恰好包含两个问题")
    ids = [claim.get("question_id") if isinstance(claim, dict) else None
           for claim in claims]
    if any(ids.count(question_id) != 1 for question_id in QUESTION_FIELDS):
        raise ValueError("两个问题必须各出现一次")
    return {"source": str(SOURCE), "source_sha256": actual_hash,
            "source_version_matches": True,
            "results": [{"question_id": claim["question_id"],
                         **verify_claim(document, text, claim)} for claim in claims]}


def main():
    print("阶段1：读取当前对话留下的候选快照，不重新调用模型。")
    snapshot = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    print("阶段2：程序重新读取原文件；不使用候选内附的所谓原文副本。")
    raw = SOURCE.read_bytes()
    print("阶段3：核对文件指纹、字段、数值、引用及位置。")
    report = verify_snapshot(snapshot, raw)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if any(item["outcome"] == "rejected" for item in report["results"]):
        raise SystemExit("存在被拒绝的候选，停止采用其数值。")
    print("阶段4：核对完成。字段缺失不是0秒，也不是对整个项目的穷尽搜索。")


if __name__ == "__main__":
    main()
