"""从虚构原始记录构建评测教学样本，仅使用Python标准库，不调用模型。

本脚本演示数据职责分离，不是生产清洗管道，也不是完整创意评审器。
注意：输出目录里的文件是可重新生成的教学产物；每次运行写入新目录。
"""

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path


def fingerprint(value):
    """字典先稳定排序再计算摘要：用于识别相同输入，不是权限凭证。"""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hard_checks(task, candidate):
    """只检查四项明确规则；返回True不代表创意或全部事实合格。

    len在这里采用Python字符串长度，包含标点；真实业务需明确计数口径。
    必需词只查标题，禁用词查标题和正文；不自动修补被测文案。
    """
    valid = (
        isinstance(candidate, dict)
        and set(candidate) == {"headline", "body"}
        and all(isinstance(candidate[key], str) and candidate[key].strip()
                for key in ("headline", "body"))
    )
    if not valid:
        # None表示后续检查无法执行，不把结构异常默默修成合格数据。
        return {"format": False, "headline_length": None,
                "required_headline_term": None, "forbidden_exact_terms": None}
    rules = task["constraints"]
    headline = candidate["headline"]
    fields = (headline, candidate["body"])
    return {
        "format": True,
        "headline_length": len(headline) <= rules["headline_max_chars"],
        "required_headline_term": all(term in headline for term in rules["required_terms"]),
        "forbidden_exact_terms": not any(term in field for term in rules["forbidden_terms"]
                                         for field in fields),
    }


def build(raw):
    """把原始记录变成一条生成评测题、四条评审器校准题和独立标签。

    全部记录只有一个brief组，故不能随机分出可信训练／测试集。
    去重针对同一任务快照、同一brief组和完全相同的文案；语义近重复及
    相互矛盾的标注在生产场景还需要专门审查，此示例不宣称解决了它们。
    """
    task = deepcopy(raw["task_snapshot"])
    accepted, duplicates, quarantine, seen = [], [], [], {}
    for record in raw["records"]:
        if record["task_snapshot_id"] != raw["task_snapshot_id"] or record["teaching_annotation"] is None:
            quarantine.append({"record_id": record["record_id"],
                               "reason": "缺少可确认的任务快照或判据标注；暂不计分，保留待补资料"})
            continue
        key = fingerprint([record["brief_group_id"], record["task_snapshot_id"], record["candidate"]])
        if key in seen:
            duplicates.append({"record_id": record["record_id"], "duplicate_of": seen[key]})
            continue
        seen[key] = record["record_id"]
        accepted.append(deepcopy(record))

    criteria = {
        "hard": "输出仅headline和body两个非空文本字段；标题不超过18个Python字符且含咖啡杯；标题及正文不得含约定禁用词。",
        "supported_claims": "所有客观商品主张需有本任务product.facts支持；不得添加功效、价格、销量或售后承诺。",
        "non_preachy": "不得要求用户必须接受某种生活态度，或因不休息而指责用户；场景叙述、可选择的动作邀请不自动判说教。",
        "product_integration": "杯子通过使用动作或类别用途参与情境；只署产品名不算。该项不衡量独创性。",
        "judgment": "语义结论为pass／fail／unknown，引用候选原文并解释；资料不足用unknown，不强行二分。",
        "scope": "仅为本次规则符合度；不是完整创意质量、真实性保证或营销收益评测。"
    }
    case_id = "cup-case-001"
    generation = [{"case_id": case_id, "input": task}]
    specification = [{"case_id": case_id, "brief_group_id": "office-pause-001",
                      "split": "development_validation_demo", "criteria": criteria,
                      "reference_output": accepted[0]["candidate"],
                      "reference_note": "非唯一正确答案，不要求逐字匹配。不得把本参考作为本题生成输入。"}]
    judge_inputs, labels, checks = [], [], []
    for index, record in enumerate(accepted, 1):
        item_id = f"judge-{index:03d}"
        # 给评审器的是任务、待评文案和通用标准；不包含参考标签或客户反馈。
        judge_inputs.append({"item_id": item_id, "case_id": case_id,
                             "input": {"task": task, "candidate": record["candidate"], "criteria": criteria}})
        labels.append({"item_id": item_id, "source_record_id": record["record_id"],
                       "brief_group_id": record["brief_group_id"],
                       "expected": record["teaching_annotation"]})
        actual = hard_checks(task, record["candidate"])
        checks.append({"item_id": item_id, "actual_hard_checks": actual,
                       "matches_teaching_hard_labels": actual == record["teaching_annotation"]["hard"],
                       "semantic_execution": "not_run", "overall_machine_verdict": "not_evaluated"})
    return {
        "generation_inputs.json": generation,
        "evaluation_spec.json": specification,
        "judge_inputs.json": judge_inputs,
        "judge_labels.json": labels,
        "cleaning_report.json": {
            "dataset_version": raw["dataset_version"], "raw_sha256": fingerprint(raw),
            "raw_records": len(raw["records"]), "accepted_unique_candidates": len(accepted),
            "unique_generation_cases": len(generation), "brief_groups": 1,
            "duplicates": duplicates, "quarantine": quarantine,
            "cleaning_boundary": "原文保留；不删除坏文案、不纠正缺词、不补写事实、不把缺资料当成模型错误。",
            "split_warning": "只有一个brief组，不能据此构造独立训练／验证／测试切分。所有公开样例只用于教学开发与校准。"
        },
        "offline_hard_check_report.json": {
            "scope": "只检查固定候选的四项确定性规则；没有执行Agent生成或模型语义评审。",
            "results": checks,
            "matching_candidates": sum(row["matches_teaching_hard_labels"] for row in checks),
            "candidate_count": len(checks),
            "metric_warning": "标签一致仅验证本演示规则，不是Agent准确率或模型审核准确率。"
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="新的输出目录；已经存在则拒绝覆盖")
    args = parser.parse_args()
    source = Path(__file__).with_name("raw_records.json")
    raw = json.loads(source.read_text(encoding="utf-8"))
    outputs = build(raw)
    directory = args.output or source.parent / ("run-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    directory.mkdir(parents=True, exist_ok=False)
    for name, value in outputs.items():
        (directory / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("原始6条：保留4条，合并重复1条，隔离待补资料1条。")
    print("形成1条生成评测题、4条评审器校准题；全部属于同一brief组。")
    report = outputs["offline_hard_check_report.json"]
    print(f"程序规则与教学标签一致：{report['matching_candidates']}/{report['candidate_count']}；语义评审未运行。")
    print(f"输出目录：{directory}")


if __name__ == "__main__":
    main()
