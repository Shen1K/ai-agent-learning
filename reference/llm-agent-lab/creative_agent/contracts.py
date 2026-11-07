"""输入契约：JSON 是数据，不是可以直接执行的 Python 代码。

模型给出的工具名、参数和评审结论都需要再检查。这里不用额外库，
便于先理解 Schema 的职责；不是通用 JSON Schema 实现。
"""

import json
from pathlib import Path

from lessons.creative_studio.deepseek_review import unique_object


RULE_VERSION = "creative-agent-gate-v1"
CHECKS = ("brief_fit", "product_link", "tone_fit", "facts_supported")
DEFAULT_LIMITS = {"max_steps": 16, "max_model_calls": 20, "max_candidates": 4}
CEILINGS = {"max_steps": 40, "max_model_calls": 40, "max_candidates": 8}


class InvalidInput(ValueError):
    """只放可以公开给用户看的错误，不包含密钥或原始网络异常。"""


def require(condition, message):
    if not condition:
        raise InvalidInput(message)


def _constant(value):
    raise InvalidInput("JSON 不接受 NaN 或 Infinity")


def decode(raw):
    try:
        return json.loads(raw, object_pairs_hook=unique_object, parse_constant=_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise InvalidInput("JSON 无效、过深或存在重复字段") from None


def exact(value, fields, label):
    require(type(value) is dict and set(value) == set(fields), f"{label}字段不匹配")


def string(value, label, maximum=2000, empty=False):
    require(type(value) is str and len(value) <= maximum
            and (empty or bool(value.strip())), f"{label}必须是长度受限的文本")


def string_map(value, label, allow_empty=False):
    require(type(value) is dict and len(value) <= 20 and (allow_empty or bool(value)),
            f"{label}必须是最多20项的文本字典")
    for key, text in value.items():
        string(key, label + "编号", 80)
        string(text, label + "内容", 2000)


def validate_task(task):
    exact(task, ("brief", "product", "references", "constraints"), "任务")
    exact(task["brief"], ("audience", "goal", "tone"), "brief")
    for key, value in task["brief"].items():
        string(value, "brief." + key)
    product = task["product"]
    exact(product, ("id", "name", "facts"), "product")
    string(product["id"], "product.id", 80)
    string(product["name"], "product.name", 120)
    string_map(product["facts"], "事实")
    string_map(task["references"], "参考", allow_empty=True)
    constraints = task["constraints"]
    exact(constraints, ("headline_max_chars", "required_terms", "forbidden_terms"), "约束")
    require(type(constraints["headline_max_chars"]) is int
            and 1 <= constraints["headline_max_chars"] <= 80, "标题长度限额须为1至80的整数")
    for key in ("required_terms", "forbidden_terms"):
        items = constraints[key]
        require(type(items) is list and len(items) <= 20, "词表必须是最多20项的数组")
        for item in items:
            string(item, "词表项", 80)
    require(not (set(constraints["required_terms"]) & set(constraints["forbidden_terms"])),
            "必含词与禁用词直接冲突，请先修正任务")
    require(len(json.dumps(task, ensure_ascii=False)) <= 16000, "任务资料过大")


def limits_or_default(limits=None):
    limits = dict(DEFAULT_LIMITS) if limits is None else limits
    exact(limits, DEFAULT_LIMITS, "预算")
    for key, value in limits.items():
        require(type(value) is int and 1 <= value <= CEILINGS[key], "预算超出本地服务上限")
    return dict(limits)


def example_task():
    return decode(Path(__file__).with_name("example_task.json").read_text(encoding="utf-8"))


def tool_schema(name, description, fields):
    # 所有参数都显式必填；没有父稿时传空字符串，避免隐含默认值。
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {
            key: {"type": "string", "description": meaning} for key, meaning in fields.items()},
            "required": list(fields), "additionalProperties": False}}}


TOOLS = [
    tool_schema("get_materials", "读取当前任务的商品事实、品牌要求、参考；首次创作前使用。", {}),
    tool_schema("search_knowledge", "检索本任务适用的品牌规范、历史案例和反馈，返回原文及版本。仅RAG启用时可用；不能用参考案例新增商品事实。", {
        "query": "具体的问题或需要参考的创意情境，最多240字；不要提供文件路径或URL"}),
    tool_schema("save_candidate", "保存你创作的新稿或改稿，不能覆盖历史候选。", {
        "headline": "标题", "body": "正文", "direction": "简短创意方向",
        "hypothesis": "本稿试图改善什么，写简短理由，不输出长篇思维过程",
        "parent_id": "改稿的父候选编号；首稿用空字符串"}),
    tool_schema("evaluate_candidate", "调用隔离上下文的评审器，返回证据与改进反馈。", {
        "candidate_id": "已保存的候选编号"}),
    tool_schema("submit_candidate", "提交已通过本轮检查的候选，进入用户确认；不能发布。", {
        "candidate_id": "已评审的本轮候选编号"}),
    tool_schema("request_clarification", "资料有歧义或任务无法继续时暂停并提问。", {
        "question": "需要用户回答的一个具体问题"}),
]


def validate_action(action):
    exact(action, ("id", "name", "arguments"), "动作")
    string(action["id"], "调用编号", 128)
    names = {item["function"]["name"]: item["function"] for item in TOOLS}
    require(type(action["name"]) is str and action["name"] in names, "工具不在允许名单中")
    exact(action["arguments"], names[action["name"]]["parameters"]["properties"], "工具参数")
    for key, value in action["arguments"].items():
        string(value, key, 1200 if key == "body" else 500, empty=(key == "parent_id"))
