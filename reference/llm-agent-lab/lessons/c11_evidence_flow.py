"""C11：字段优先、文字提取兜底的只读流程，尚未接入大模型。

运行：python3 lessons/c11_evidence_flow.py
本文件不下载、不训练、不写入资料；只读取现有数据清单。
先由程序明确指定问题对应的字段，排除自然语言理解这个额外难点。
分支方案由学习者提出，Python 实现由助手补充。
文字提取支持小词表里的两种教学表达，不是通用日志理解或模型调用。
"""

import json
import re
from pathlib import Path


# 用脚本自身的位置找项目，不依赖终端当前在哪个文件夹。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT_ROOT / "data" / "processed" / "manifest.json"

# 按学习者方案维护“耗时表达词表”。这里只需要列出词，不需要键值映射，
# 所以用元组保存；不是 Python 的 dict，也不是学习得到的词向量。
DURATION_PHRASES = ("持续了", "耗时")


def read_evidence(source, field):
    """读取并提取字段，返回一个字典，保留来源及执行结果。

    source 是 Path 文件路径；field 是字符串形式的字段名。
    tool_ok 仅代表成功读取一个 JSON 对象，不代表已找到所需字段。
    value 是字段值；本课把不存在或值为 null 的字段当作没有记录。
    """
    result = {"source": str(source), "field": field,
              "tool_ok": False, "value": None, "text": "", "error": None}
    try:
        # read_text 读取文本；json.loads 将 JSON 文本转换为 Python 对象。
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("资料顶层必须是 JSON 对象")
    except (OSError, ValueError) as error:
        # 文件不存在、无权限、编码或 JSON 格式错误：保存原因并结束。
        # 不反复读取同一个失败文件，也不伪造一个默认答案。
        result["error"] = str(error)
        return result

    result["tool_ok"] = True
    # get 找不到字段时返回 None，不抛出“键不存在”的异常。
    # 注意：None 不是数字 0，也不是文本 "None"。
    result["value"] = data.get(field)
    # notes 是教学约定的文字记录字段，真实清单目前没有这个字段。
    # 保留原文，后续步骤才能提取，而不是只拿到一个空的数值。
    text = data.get("notes", "")
    if isinstance(text, str):
        result["text"] = text
    return result


def has_evidence(value):
    """检查字段是否非空，不把数值0当作缺失。

    本课限定字段是数值或 None；这里只检查缺失，不做通用事实核验。
    数字 0 也是一条记录，不能仅因为它为零就认定资料缺失。
    is not None 表示“不是没有值”；这一行语法由助手补充。
    """
    return value is not None


def extract_duration(text):
    """从词表允许的完整句式提取整数秒数，同时保留原文依据。

    接受“本次训练持续了90秒。”和“本次训练耗时90秒。”这类句子。
    不能仅因出现“耗时”二字就作答，还要匹配对象、数字和单位。
    [0-9]+ 表示一个或多个数字；括号把数字单独捕获出来。
    fullmatch 要求整段文字匹配，避免从更复杂的句子中贸然取数。
    匹配失败只表示这条规则没识别到，不证明原文没有耗时。
    """
    for phrase in DURATION_PHRASES:
        # 每次取一种表达，检查完整句子；不是把新词覆盖到旧词上。
        # re.escape 使词表项按普通文字匹配，防止标点被当成正则指令。
        # rf 字符串同时支持插入变量和保留正则写法。
        pattern = rf"本次训练{re.escape(phrase)}([0-9]+)秒。?"
        match = re.fullmatch(pattern, text.strip())
        if match is not None:
            # group(1) 取数字文本；int 将“90”变成数字90。
            return {"value": int(match.group(1)), "quote": match.group(0)}

    # 所有允许的句式都试过仍没匹配，不等于原文一定没有耗时信息。
    return None


def answer_question(question, field, unit, source=SOURCE):
    """连接读取、状态、证据判断和回复；每道题只读取一次文件。

    课堂假设数值字段有效，notes 是同一次训练的记录，二者没有冲突。
    真实系统还需检查类型、单位、训练标识和资料冲突；本例不代替这些检查。
    """
    state = {"question": question, "history": [],
             "status": "running", "answer": None, "evidence": None}
    result = read_evidence(source, field)
    # append 把一条结果加入列表；请求字段、来源和失败原因都一起保留。
    state["history"].append(result)

    if not result["tool_ok"]:
        state["status"] = "tool_error"
        state["answer"] = "读取失败，无法根据这份资料回答。"
        return state

    if has_evidence(result["value"]):
        state["status"] = "answered"
        state["evidence"] = {"method": "field", "source": result["source"],
                             "field": field, "value": result["value"]}
        # 这是固定模板回复，不是模型生成；限定为当前文件记录的值。
        state["answer"] = f"根据这份清单，记录值为 {result['value']} {unit}。"
        return state

    # 这条文字规则只会提取秒数，不能拿来回答文章数量或其他问题。
    if field == "training_duration_seconds" and unit == "秒":
        extracted = extract_duration(result["text"])
        state["history"].append({"action": "extract_duration_by_rule",
                                 "source": result["source"], "result": extracted})
        if extracted is not None:
            state["status"] = "answered"
            state["evidence"] = {"method": "text_rule", "source": result["source"],
                                 "field": "notes", **extracted}
            # **extracted 把提取结果的 value、quote 两项放进同一个字典。
            state["answer"] = f"根据该文件的文字记录，训练耗时为 {extracted['value']} 秒。"
            return state

    state["status"] = "insufficient_evidence"
    state["answer"] = "未从指定字段或当前提取规则找到可用依据，无法据此确认。"
    return state


def main():
    # 问题到字段的对应关系由人指定，不声称程序能理解任意提问。
    # training_duration_seconds 是教学约定的查询字段，现有清单没有它。
    questions = [
        ("这份清单的训练集有多少篇文章？", "training_document_count", "篇"),
        ("这份清单是否记录了训练耗时？", "training_duration_seconds", "秒"),
    ]
    for question, field, unit in questions:
        state = answer_question(question, field, unit)
        # ensure_ascii=False 保留中文；indent=2 让层级容易阅读。
        print(json.dumps(state, ensure_ascii=False, indent=2))

    print("\n以下为人工构造的候选结果校验演示，不是模型调用或真实训练日志：")
    # 来源记录由程序保存，不能用候选输出里的字段来覆盖。
    # 为了单独练习校验，本例把元数据和值都预先结构化了。
    # 现实中若已经有这些字段，直接查询即可，无需再让模型提取一遍。
    records = {
        "record-A": {"run_id": "A", "phase": "training", "kind": "actual",
                     "unit": "seconds", "duration_seconds": 120,
                     "text": "任务A实际训练用了120秒。"},
        "record-B": {"run_id": "B", "phase": "training", "kind": "actual",
                     "unit": "seconds", "duration_seconds": 80,
                     "text": "任务B训练用了80秒。"},
    }
    for record_id in ("record-A", "record-B"):
        record = records[record_id]
        candidate = {"run_id": "A", "record_id": record_id,
                     "duration_seconds": record["duration_seconds"],
                     "quote": record["text"]}
        state = answer_from_candidate("A", candidate, records)
        print(json.dumps(state, ensure_ascii=False, indent=2))


def validate_candidate(target_run_id, candidate, records):
    """独立检查候选结果，返回 accepted 和 reason，不负责生成答案。

    target_run_id：程序根据任务确定的目标，不由候选答案重新定义。
    candidate：不可信的提取结果；本课用测试样例代替模型输出。
    records：程序持有的来源记录表，假设已验证格式且内部一致。
    这里检查结构化来源的一致性，不会自动理解任意自由文本。
    """
    # 先检查形状，避免缺字段或错误类型导致后续查表崩溃。
    if not isinstance(candidate, dict):
        return {"accepted": False, "reason": "候选结果必须是字典"}
    if not isinstance(candidate.get("record_id"), str):
        return {"accepted": False, "reason": "缺少有效的来源记录编号"}
    record = records.get(candidate["record_id"])
    if record is None:
        return {"accepted": False, "reason": "来源记录不存在"}

    # 关键：先核对来源所属任务，不是只看候选自称属于哪个任务。
    if record["run_id"] != target_run_id:
        return {"accepted": False, "reason": "来源记录属于其他任务"}
    if record["phase"] != "training" or record["kind"] != "actual":
        return {"accepted": False, "reason": "来源不是实际训练阶段的记录"}
    if record["unit"] != "seconds":
        return {"accepted": False, "reason": "来源单位不是秒，需另行转换"}
    if candidate.get("run_id") != target_run_id:
        return {"accepted": False, "reason": "候选声明的任务编号不符"}

    # 教学记录只有一句话，要求引用完整且逐字一致。
    # 真实长文常用片段位置；字符串存在仍不等于语义正确。
    if candidate.get("quote") != record["text"]:
        return {"accepted": False, "reason": "引用与来源原文不一致"}
    value = candidate.get("duration_seconds")
    # 本例只接受非负整数秒。type(... ) is int 排除 Python 中也能
    # 当作整数参与运算的 True/False，不将 True 错认成1秒。
    if type(value) is not int or value < 0:
        return {"accepted": False, "reason": "耗时必须是非负整数秒"}
    if value != record["duration_seconds"]:
        return {"accepted": False, "reason": "耗时与来源记录不一致"}
    return {"accepted": True, "reason": "通过本例的来源一致性检查"}


def answer_from_candidate(target_run_id, candidate, records):
    """候选结果进入校验，再决定回答或拒绝；不自动调用模型或重试。

    这是新增的校验分支，原有 answer_question 的规则读取分支仍保留。
    拒绝某个候选不等于整个资料库没有答案，也不等于任务已完成。
    """
    state = {"target_run_id": target_run_id, "status": "running",
             "history": [], "answer": None, "evidence": None}
    state["history"].append({"action": "receive_candidate", "result": candidate})
    check = validate_candidate(target_run_id, candidate, records)
    state["history"].append({"action": "validate_candidate", "result": check})
    if not check["accepted"]:
        state["status"] = "rejected"
        state["answer"] = "候选结果未通过校验，暂不据此给出耗时。"
        return state

    state["status"] = "answered"
    state["evidence"] = {"record_id": candidate["record_id"],
                         "quote": candidate["quote"]}
    state["answer"] = f"任务{target_run_id}实际训练耗时为{candidate['duration_seconds']}秒。"
    return state


if __name__ == "__main__":
    # 直接运行才进入演示；测试导入此文件时不会自动执行演示。
    main()
