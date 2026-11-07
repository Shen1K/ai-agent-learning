"""Agent核心：模型选工具，程序验证与执行，工具反馈回到下一轮。

阅读顺序：create → run → _decide → _execute。
run不写死“生成→评审→修改”的业务顺序；只维护统一的决策/执行循环。
"""

from copy import deepcopy
import json
import threading

from .contracts import (RULE_VERSION, TOOLS, InvalidInput, limits_or_default, require,
                        string, validate_action, validate_task)
from .evaluation import hard_checks, validate_judgment
from .prompts import initial_messages, judge_messages
from .providers import ProviderFailure
from .rag import (ENGINE_VERSION, MAX_SEARCHES, candidate_context, required_context,
                  search)


class BudgetExhausted(Exception):
    pass


class Cancelled(Exception):
    pass


class AgentRuntime:
    def __init__(self, store, provider, knowledge_base=None):
        self.store = store
        self.provider = provider
        # 可选依赖注入：原来的无RAG流程不变；资料由程序配置，不由模型选路径。
        self.knowledge_base = knowledge_base
        # 第一版一次只执行一个任务。HTTP查询不拿此锁，因此仍可查看状态和取消。
        self.execution_lock = threading.Lock()

    @property
    def backend(self):
        return {"mode": self.provider.mode, "model": self.provider.model}

    def create(self, task, limits=None, request_key=None):
        validate_task(task)
        state = {"task": deepcopy(task), "limits": limits_or_default(limits),
                 "backend": self.backend, "rule_version": RULE_VERSION,
                 "status": "ready", "messages": initial_messages(task),
                 "materials_read": False, "steps": 0, "model_calls": 0,
                 "candidates": [], "evaluations": {}, "pending_action": None,
                 "selected_candidate_id": None, "feedback_round": 0,
                 "user_feedback": [],
                 "question": None, "error": None, "cancel_requested": False,
                 "publication_authorized": False, "billing": "not_measured"}
        if self.knowledge_base is not None:
            state["rag"] = {"snapshot": self.knowledge_base.snapshot(task["product"]["id"]),
                            "retrievals": []}
        return self.store.create(state, request_key)

    def _check_cancel(self, state):
        if self.store.get(state["id"])["cancel_requested"]:
            raise Cancelled()

    def _model_call(self, state, purpose, invoke):
        self._check_cancel(state)
        if state["model_calls"] >= state["limits"]["max_model_calls"]:
            raise BudgetExhausted("模型调用次数达到上限")
        # 在网络请求之前扣次数。即使超时，供应商仍可能已经处理和计费。
        state["model_calls"] += 1
        self.store.save(state, "model_call_started", {"purpose": purpose, "number": state["model_calls"]})
        result, stats = invoke()
        self._check_cancel(state)
        self.store.save(state, "model_call_finished", {"purpose": purpose, **stats})
        return result

    def _decide(self, state):
        if state["steps"] >= state["limits"]["max_steps"]:
            raise BudgetExhausted("决策轮次达到上限")
        # 上下文有硬上限；不偷偷丢弃约束和历史，避免截断造成错误行为。
        if len(json.dumps(state["messages"], ensure_ascii=False)) > 100000:
            raise BudgetExhausted("上下文达到教学版上限，需要新建范围更小的任务")
        state["steps"] += 1
        available_tools = [tool for tool in TOOLS if state.get("rag") is not None
                           or tool["function"]["name"] != "search_knowledge"]
        chosen = self._model_call(state, "decision",
                                  lambda: self.provider.decide(deepcopy(state["messages"]), deepcopy(available_tools)))
        # Provider返回不是可信程序。先检查基本形状；工具白名单在执行前再次验证。
        require(type(chosen) is dict and set(chosen) == {"id", "name", "arguments"}, "模型决策结构错误")
        string(chosen["id"], "调用编号", 128)
        string(chosen["name"], "工具名", 100)
        previous_ids = [call["id"] for msg in state["messages"]
                        for call in msg.get("tool_calls", [])]
        require(chosen["id"] not in previous_ids, "模型重复了工具调用编号")
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": [{
            "id": chosen["id"], "type": "function", "function": {
                "name": chosen["name"], "arguments": json.dumps(chosen["arguments"], ensure_ascii=False)}}]})
        state["pending_action"] = chosen
        # 保存动作后才执行。评审失败时恢复这个动作，不重新生成候选。
        self.store.save(state, "tool_requested", chosen)

    def _candidate(self, state, candidate_id):
        for candidate in state["candidates"]:
            if candidate["id"] == candidate_id:
                return candidate
        raise InvalidInput("候选不存在于当前任务")

    def _execute(self, state, chosen):
        validate_action(chosen)
        name, args = chosen["name"], chosen["arguments"]
        if name == "get_materials":
            state["materials_read"] = True
            result = {"task": state["task"], "rule_version": state["rule_version"]}
            if state.get("rag"):
                snapshot = state["rag"]["snapshot"]
                result["knowledge"] = {"snapshot_id": snapshot["snapshot_id"],
                                       "required_policies": required_context(snapshot),
                                       "tool": "search_knowledge", "max_searches": MAX_SEARCHES}
            return result
        if name == "request_clarification":
            state.update(status="needs_input", question=args["question"])
            return {"question": args["question"]}
        require(state["materials_read"], "请先get_materials读取事实与约束")
        if name == "search_knowledge":
            require(state.get("rag") is not None, "本任务没有启用RAG")
            rag_state = state["rag"]
            require(len(rag_state["retrievals"]) < MAX_SEARCHES, "检索次数达到上限，请利用已有资料或请求澄清")
            result = search(rag_state["snapshot"], args["query"])
            # 与tool_finished一起提交到SQLite。恢复使用冻结资料，不重读最新文件。
            rag_state["retrievals"].append(deepcopy(result))
            return result
        if name == "save_candidate":
            if len(state["candidates"]) >= state["limits"]["max_candidates"]:
                raise BudgetExhausted("候选数量达到上限")
            if args["parent_id"]:
                self._candidate(state, args["parent_id"])
            require(not any(c["headline"].strip() == args["headline"].strip()
                            and c["body"].strip() == args["body"].strip()
                            and c["feedback_round"] == state["feedback_round"]
                            for c in state["candidates"]), "本轮已有相同文案，请复用其评审或做实质修改")
            candidate = {"id": "c" + str(len(state["candidates"]) + 1),
                         "feedback_round": state["feedback_round"], **deepcopy(args)}
            if state.get("rag"):
                # 不是让模型随意声明证据ID。程序只带入本任务当时可见的原文。
                candidate["knowledge_context"] = candidate_context(state["rag"])
                candidate["knowledge_snapshot_id"] = state["rag"]["snapshot"]["snapshot_id"]
            state["candidates"].append(candidate)
            return {"candidate_id": candidate["id"], "candidate": candidate}
        candidate = self._candidate(state, args["candidate_id"])
        if name == "evaluate_candidate":
            if candidate["id"] in state["evaluations"]:
                # 候选和规则冻结，已有评审可复用，避免反复抽取直到偶然高分。
                return state["evaluations"][candidate["id"]]
            hard = hard_checks(state["task"], candidate)
            semantic = None
            if hard["passed"]:
                response = self._model_call(state, "judge", lambda: self.provider.judge(
                    judge_messages(state["task"], candidate, state.get("user_feedback", []))))
                try:
                    semantic = validate_judgment(response, state["task"], candidate)
                except InvalidInput:
                    # 裁判坏返回不是文案低分，也不能给它存一个伪造的0分。
                    raise ProviderFailure("评审协议或原文证据校验失败；未保存评分，可显式恢复。") from None
            result = {"candidate_id": candidate["id"], "rule_version": RULE_VERSION,
                      "hard": hard, "semantic": semantic,
                      "eligible": hard["passed"] and semantic is not None
                      and all(item["passed"] is True for item in semantic.values()),
                      "simulated": self.provider.mode == "mock",
                      "meaning": "仅表示可提交用户审核的草稿，不证明创意或投放效果"}
            state["evaluations"][candidate["id"]] = result
            return result
        if name == "submit_candidate":
            require(candidate["feedback_round"] == state["feedback_round"], "用户已给新反馈，请先保存本轮改稿")
            evaluation = state["evaluations"].get(candidate["id"])
            require(evaluation is not None and evaluation["eligible"], "候选没有通过本轮检查，不能提交")
            state.update(status="awaiting_review", selected_candidate_id=candidate["id"])
            return {"candidate_id": candidate["id"], "status": "awaiting_review", "published": False}
        raise InvalidInput("未知工具")

    def run(self, run_id):
        # 不阻塞等待同一runtime的另一执行者；服务层用一个worker安排任务。
        require(self.execution_lock.acquire(blocking=False), "已有任务正在执行，请稍后再试")
        try:
            state = self.store.get(run_id)
            require(state["backend"] == self.backend, "恢复时不能切换模型后端，请使用原配置")
            require(state["rule_version"] == RULE_VERSION, "规则版本已变更，不能直接恢复")
            if state.get("rag"):
                require(state["rag"]["snapshot"]["engine_version"] == ENGINE_VERSION,
                        "检索实现版本已变更，请新建任务")
            if state["status"] != "ready":
                return state
            state.update(status="running", error=None)
            self.store.save(state, "started")
            try:
                while state["status"] == "running":
                    self._check_cancel(state)
                    if state["pending_action"] is None:
                        self._decide(state)
                    self._check_cancel(state)
                    chosen = state["pending_action"]
                    try:
                        result = self._execute(state, chosen)
                        observation = {"tool": chosen["name"], "ok": True, "result": result}
                    except InvalidInput as exc:
                        # 未知工具/参数错误作为反馈交还模型，但不会执行任意函数或代码。
                        observation = {"tool": chosen["name"], "ok": False, "error": str(exc)}
                    state["messages"].append({"role": "tool", "tool_call_id": chosen["id"],
                                              "content": json.dumps(observation, ensure_ascii=False)})
                    state["pending_action"] = None
                    # 候选变更、工具结果与动作完成标记一次提交，避免重启重复保存候选。
                    self.store.save(state, "tool_finished", observation)
            except Cancelled:
                state.update(status="cancelled", cancel_requested=True)
                self.store.save(state, "cancelled")
            except BudgetExhausted as exc:
                state.update(status="budget_exhausted", error=str(exc))
                self.store.save(state, "budget_exhausted", {"reason": str(exc)})
            except (ProviderFailure, InvalidInput) as exc:
                state.update(status="error", error=str(exc))
                self.store.save(state, "error", {"reason": str(exc)})
            except Exception:
                # 不把未知异常正文/堆栈送入HTTP响应，避免泄露资料或认证信息。
                state.update(status="error", error="内部执行异常；保留任务，请检查程序。")
                self.store.save(state, "error", {"reason": state["error"]})
            return self.store.get(run_id)
        finally:
            self.execution_lock.release()

    def resume(self, run_id):
        def change(state):
            require(state["status"] in ("error", "ready"), "只有error或待执行任务能显式恢复")
            require(state["backend"] == self.backend, "请使用原模型后端恢复")
            require(not state["cancel_requested"], "已取消的任务不能恢复")
            state.update(status="ready", error=None)
        return self.store.mutate(run_id, change, "resume_requested")

    def cancel(self, run_id):
        def change(state):
            require(state["status"] not in ("approved", "cancelled"), "任务已确认或取消")
            state.update(cancel_requested=True, status="cancelled")
        return self.store.mutate(run_id, change, "cancel_requested")

    def review(self, run_id, decision, feedback=""):
        require(decision in ("approve", "revise", "answer"), "不支持的确认动作")
        string(feedback, "反馈", 2000, empty=decision == "approve")
        def change(state):
            allowed = "needs_input" if decision == "answer" else "awaiting_review"
            require(state["status"] == allowed, "当前状态不接受此确认动作")
            if decision == "approve":
                state["status"] = "approved"  # 仅确认本地草稿，没有发布工具。
            else:
                state["feedback_round"] += 1
                state.setdefault("user_feedback", []).append(feedback)
                state.update(status="ready", selected_candidate_id=None, question=None)
                state["messages"].append({"role": "user", "content": json.dumps({
                    "feedback": feedback, "instruction": "在冻结事实和约束内回应；修改需新建候选。"}, ensure_ascii=False)})
        return self.store.mutate(run_id, change, "human_" + decision)
