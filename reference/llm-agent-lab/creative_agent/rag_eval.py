"""运行：python3 -S -B -m creative_agent.rag_eval

只测真实的本地检索，不调用文案模型。把预期证据放在评分端，绝不把答案
拼进搜索问题。比较两个方案时固定语料、问题、BM25和预算，只改变返回
小片段还是父段。这是刻意构造的教学开发集，不能推断线上准确率。
"""

import argparse
import json

from .contracts import decode
from .rag import DATA_DIR, KnowledgeBase, search


def evaluate():
    snapshot = KnowledgeBase.bundled().snapshot("cup-001")
    cases = decode((DATA_DIR / "cases.json").read_text(encoding="utf-8"))["cases"]
    reports = []
    for strategy in ("small_chunks", "parent_context"):
        results = []
        for case in cases:
            # top_k=1让‘命中片段但丢掉条件’更清楚；是教学配置，不是推荐默认值。
            response = search(snapshot, case["question"], strategy=strategy, top_k=1)
            matched = [h for h in response["hits"] if h["source_id"] == case["source_id"]]
            evidence_text = "\n".join(h["text"] for h in matched)
            passed = (not response["hits"] if case["expect_empty"] else
                      bool(matched) and all(q in evidence_text for q in case["required_quotes"]))
            results.append({"case_id": case["id"], "question": case["question"],
                            "lesson": case["lesson"], "passed": passed,
                            "evidence_hit": bool(matched) if not case["expect_empty"] else None,
                            "returned_chars": response["returned_chars"],
                            "sources": [h["source_id"] for h in response["hits"]]})
        answerable = [r for r in results if r["evidence_hit"] is not None]
        reports.append({"strategy": strategy, "snapshot_id": snapshot["snapshot_id"],
                        "case_passes": sum(r["passed"] for r in results), "case_count": len(results),
                        "evidence_hits": sum(r["evidence_hit"] for r in answerable),
                        "answerable_count": len(answerable),
                        "total_returned_chars": sum(r["returned_chars"] for r in results),
                        "results": results})
    return {"scope": "教学开发集；无生成模型、无真实创意评分；不是线上准确率", "reports": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="可选：只观察自己的问题如何检索，不运行文案模型")
    parser.add_argument("--strategy", choices=("small_chunks", "parent_context"), default="parent_context")
    parser.add_argument("--top-k", type=int, choices=range(1, 11), default=1)
    args = parser.parse_args()
    result = (search(KnowledgeBase.bundled().snapshot("cup-001"), args.query,
                     strategy=args.strategy, top_k=args.top_k) if args.query is not None else evaluate())
    print(json.dumps(result, ensure_ascii=False, indent=2))
