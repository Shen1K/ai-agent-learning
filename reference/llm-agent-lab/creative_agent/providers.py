"""同一个接口，两种后端：离线脚本模拟 / DeepSeek真实工具调用。

MockProvider没有语言理解能力，只测试程序的控制流；不能把它的结果当作模型效果。
DeepSeekProvider复用旧课的HTTPS传输，不重复实现密钥、超时和安全错误处理。
"""

import json
import os
import uuid

from lessons.creative_studio.deepseek_review import call_deepseek, CallFailure, DEFAULT_MODEL
from .contracts import CHECKS, InvalidInput, decode, require, string


class ProviderFailure(RuntimeError):
    pass


def action(name, **arguments):
    return {"id": "call_" + uuid.uuid4().hex, "name": name, "arguments": arguments}


class DeepSeekProvider:
    mode = "deepseek"

    def __init__(self, model=DEFAULT_MODEL):
        string(model, "模型名", 100)
        self.model = model

    def _call(self, payload):
        try:
            # 不把key存在任务state、日志或CLI参数里，只在真正调用时读取环境变量。
            envelope, elapsed = call_deepseek(payload, os.environ.get("DEEPSEEK_API_KEY"))
            choice = envelope["choices"][0]
            usage = envelope.get("usage") or {}
            stats = {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                     if type(usage) is dict and type(usage.get(key)) is int and usage[key] >= 0}
            return choice, {"elapsed_seconds": elapsed, "usage": stats}
        except CallFailure as exc:
            raise ProviderFailure(str(exc)) from None
        except (KeyError, IndexError, TypeError):
            raise ProviderFailure("服务响应结构不完整；本次调用已计入预算。") from None

    def _payload(self, messages):
        return {"model": self.model, "messages": messages, "stream": False,
                "thinking": {"type": "disabled"}, "temperature": 0.3, "max_tokens": 1800}

    def decide(self, messages, tools):
        payload = self._payload(messages)
        payload.update(tools=tools, tool_choice="required")
        choice, stats = self._call(payload)
        try:
            require(choice["finish_reason"] == "tool_calls", "模型未完成工具决策")
            calls = choice["message"]["tool_calls"]
            require(type(calls) is list and len(calls) == 1, "教学版每轮只接受一个工具调用")
            call = calls[0]
            require(call["type"] == "function", "仅支持function工具")
            string(call["id"], "工具调用ID", 128)
            string(call["function"]["name"], "工具名", 100)
            result = {"id": call["id"], "name": call["function"]["name"],
                      "arguments": decode(call["function"]["arguments"])}
            return result, stats
        except (KeyError, TypeError, InvalidInput):
            raise ProviderFailure("模型工具调用无效或不止一个；未执行任何工具。") from None

    def judge(self, messages):
        payload = self._payload(messages)
        payload.update(response_format={"type": "json_object"}, temperature=0)
        choice, stats = self._call(payload)
        try:
            require(choice["finish_reason"] == "stop", "评审输出未正常结束")
            return decode(choice["message"]["content"]), stats
        except (KeyError, TypeError, InvalidInput):
            raise ProviderFailure("评审输出无效或截断；保留候选，等待显式恢复。") from None


class MockProvider:
    mode = "mock"
    model = "scripted-cup-demo-v1"

    def decide(self, messages, tools):
        # 用实际工具反馈驱动分支，而不是让runtime写死每一步。
        observations = [decode(m["content"]) for m in messages if m["role"] == "tool"]
        if not observations:
            return action("get_materials"), {}
        last = observations[-1]
        user_feedback = messages[-1]["role"] == "user"
        if last["tool"] == "get_materials":
            return action("save_candidate", headline="消息可以等，这一口先给自己。",
                          body="把视线从屏幕挪开，喝一口，再继续。",
                          direction="工作间隙的自我照顾", hypothesis="先观察产品指向是否足够明确",
                          parent_id=""), {}
        if last["tool"] == "save_candidate" and last["ok"]:
            return action("evaluate_candidate", candidate_id=last["result"]["candidate_id"]), {}
        if last["tool"] == "evaluate_candidate" and last["ok"] and last["result"]["eligible"]:
            return action("submit_candidate", candidate_id=last["result"]["candidate_id"]), {}
        if ((last["tool"] == "evaluate_candidate" and last["ok"]) or user_feedback):
            candidates = [x["result"]["candidate_id"] for x in observations
                          if x["tool"] == "save_candidate" and x["ok"]]
            return action("save_candidate", headline="消息可以等，捧起咖啡杯，先照顾自己。",
                          body="把视线从屏幕挪开，给自己留一杯的时间，再继续。",
                          direction="明确咖啡杯类别", hypothesis="补充容器线索，不把产品误写成咖啡饮品",
                          parent_id=candidates[-1] if candidates else ""), {}
        return action("request_clarification", question="离线模拟遇到未覆盖分支，请检查任务或使用真实模型模式。"), {}

    def judge(self, messages):
        data = decode(messages[-1]["content"])
        headline = data["candidate"]["headline"]
        evidence = [{"field": "headline", "quote": headline}]
        fact_id = next(iter(data["facts"]))
        checks = {}
        for key in CHECKS:
            checks[key] = {"passed": ("咖啡杯" in headline) if key == "product_link" else True,
                           "reason": "离线脚本假反馈，仅验证流程，不是语义评测。",
                           "evidence": list(evidence)}
        checks["facts_supported"]["evidence"].append(
            {"field": "facts." + fact_id, "quote": data["facts"][fact_id]})
        return {"checks": checks}, {}


class RagMockProvider(MockProvider):
    """教学用脚本替身：搜索是真搜索；选工具、写稿和评审依然是脚本。

    与旧MockProvider使用不同model标识，避免恢复时静默换行为。
    这个演示只证明‘检索→工具反馈→候选证据→评审’接通，不能证明真实
    模型会自主搜对、会采纳资料或创意质量提高。
    """
    model = "scripted-cup-rag-demo-v1"

    def decide(self, messages, tools):
        observations = [decode(m["content"]) for m in messages if m["role"] == "tool"]
        if not observations:
            return action("get_materials"), {}
        last = observations[-1]
        if last["tool"] == "get_materials" and last["ok"]:
            return action("search_knowledge", query="办公场景 离开屏幕 捧起咖啡杯 片刻休息"), {}
        if last["tool"] == "search_knowledge":
            if not last["ok"] or not last["result"]["hits"]:
                return action("request_clarification", question="未取得相关参考，是否补充办公场景资料？"), {}
            return action("save_candidate", headline="捧起咖啡杯，给自己片刻。",
                          body="视线暂时离开屏幕，手里的咖啡杯，为忙碌留一小段空白。",
                          direction="捧杯动作承接休息", hypothesis="脚本示例，仅观察检索证据如何传递",
                          parent_id=""), {}
        return super().decide(messages, tools)

    def judge(self, messages):
        result, stats = super().judge(messages)
        # 仍是假评审，但实际引用传入的规范原文，能测试引文校验链路。
        data = decode(messages[-1]["content"])
        for key, item in data.get("knowledge_context", {}).items():
            if item["kind"] == "policy":
                result["checks"]["tone_fit"]["evidence"].append(
                    {"field": "knowledge." + key, "quote": item["text"]})
                break
        return result, stats
