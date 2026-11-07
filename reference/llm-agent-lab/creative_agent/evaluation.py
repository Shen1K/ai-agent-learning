"""程序检查与语义判断分开：协议有效不等于创意已经有效。

本服务暂不采用旧20项等权总分。仅使用四项明确的草稿检查，
全部满足才可提交给用户；仍不表示完成法律合规审查或达到线上效果。
"""

import unicodedata

from .contracts import CHECKS, exact, require, string


def hard_checks(task, candidate):
    title = candidate["headline"]
    text = title + "\n" + candidate["body"]
    # 字符数不计空白和Unicode标点；数字、拉丁字母也逐字符计数。
    count = sum(not c.isspace() and not unicodedata.category(c).startswith("P") for c in title)
    constraints = task["constraints"]
    missing = [x for x in constraints["required_terms"] if x not in text]
    forbidden = [x for x in constraints["forbidden_terms"] if x in text]
    return {"passed": count <= constraints["headline_max_chars"] and not missing and not forbidden,
            "headline_chars": count, "missing_required_terms": missing, "forbidden_hits": forbidden}


def validate_judgment(raw, task, candidate):
    exact(raw, ("checks",), "评审")
    exact(raw["checks"], CHECKS, "语义检查")
    sources = {key: candidate[key] for key in ("headline", "body")}
    sources.update({"facts." + key: value for key, value in task["product"]["facts"].items()})
    sources.update({"knowledge." + key: value["text"]
                    for key, value in candidate.get("knowledge_context", {}).items()})
    for name, item in raw["checks"].items():
        exact(item, ("passed", "reason", "evidence"), "单项评审")
        require(item["passed"] is None or type(item["passed"]) is bool, "评审必须返回布尔值或null")
        string(item["reason"], "评审理由", 1500)
        require(type(item["evidence"]) is list and len(item["evidence"]) <= 12, "评审证据必须是有界数组")
        fields = []
        for evidence in item["evidence"]:
            exact(evidence, ("field", "quote"), "引文")
            string(evidence["field"], "证据字段", 100)
            string(evidence["quote"], "证据原文", 2000)
            require(evidence["field"] in sources
                    and evidence["quote"] in sources[evidence["field"]], "引文不在指定原文中")
            fields.append(evidence["field"])
        if item["passed"] is not None:
            require(any(f in ("headline", "body") for f in fields), "非null评审须引用候选原文")
        if name == "facts_supported" and item["passed"] is True:
            require(any(f.startswith("facts.") for f in fields), "事实检查通过时须引用资料")
    return raw["checks"]
