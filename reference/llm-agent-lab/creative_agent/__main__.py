"""入口：python3 -S -B -m creative_agent demo。默认离线，--live才联网。"""

import argparse
import json
import os
from pathlib import Path

from .contracts import InvalidInput, decode, example_task, require
from .providers import DeepSeekProvider, MockProvider, RagMockProvider
from .rag import KnowledgeBase
from .runtime import AgentRuntime
from .service import make_server
from .store import Store


DEFAULT_DB = Path(__file__).resolve().parent.parent / ".creative-agent" / "runs.sqlite3"


def print_result(state, store):
    print(f"任务：{state['id']}  模式：{state['backend']['mode']}  状态：{state['status']}")
    for event in store.events(state["id"]):
        payload = event["payload"]
        if event["kind"] == "tool_requested":
            print(f"  请求工具：{payload['name']}")
        elif event["kind"] == "tool_finished":
            print(f"  工具结果：{payload['tool']} / {'成功' if payload['ok'] else '拒绝'}")
            if payload["tool"] == "search_knowledge" and payload["ok"]:
                for hit in payload["result"]["hits"]:
                    print(f"    来源：{hit['source_id']}；匹配分数：{hit['score']}（不是正确概率）")
                    print("    原文：" + hit["text"].replace("\n", " / "))
    selected = next((c for c in state["candidates"] if c["id"] == state["selected_candidate_id"]), None)
    if selected:
        print("\n标题：" + selected["headline"] + "\n正文：" + selected["body"])
        if "knowledge_context" in selected:
            print("本稿可用证据：" + "、".join(x["source_id"] for x in selected["knowledge_context"].values()))
    if state["question"]:
        print("待澄清：" + state["question"])
    if state["error"]:
        print("停止原因：" + state["error"])
    print(f"决策轮数：{state['steps']}；后端调用次数：{state['model_calls']}；未发布。")
    if state["backend"]["mode"] == "mock":
        print("注意：离线脚本模拟，无真实大模型判断，不代表创意效果。")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "run", "serve", "show", "events", "resume", "review", "cancel"):
        child = sub.add_parser(name)
        child.add_argument("--db", type=Path, default=DEFAULT_DB)
        child.add_argument("--live", action="store_true", help="显式使用DeepSeek，可能多次计费")
        child.add_argument("--rag", action="store_true", help="启用内置虚构知识库；默认不联网")
        if name in ("show", "events", "resume", "review", "cancel"):
            child.add_argument("run_id")
        if name == "run":
            child.add_argument("--task", type=Path, help="任务JSON，省略时使用虚构杯子资料")
            child.add_argument("--request-key", help="可选创建幂等键")
        if name == "serve":
            child.add_argument("--port", type=int, default=8765)
        if name == "review":
            child.add_argument("--decision", choices=("approve", "revise", "answer"), required=True)
            child.add_argument("--feedback", default="")
    args = parser.parse_args(argv)
    require(not (args.command == "demo" and args.live), "demo只允许离线；真实调用请用run --live")
    provider = DeepSeekProvider() if args.live else (RagMockProvider() if args.rag else MockProvider())
    store = Store(args.db)
    try:
        runtime = AgentRuntime(store, provider, KnowledgeBase.bundled() if args.rag else None)
        if args.command == "serve":
            server = make_server(runtime, os.environ.get("CREATIVE_AGENT_SERVICE_TOKEN", ""), args.port)
            print(f"本地Agent服务：http://127.0.0.1:{server.server_port}，模式={provider.mode}。Ctrl+C停止。", flush=True)
            print("API需要环境变量中的本地服务令牌；不输出令牌或DeepSeek密钥。", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("停止接收请求，等待当前调用结束……")
            finally:
                server.server_close()
                server.agent_service.close()
            return 0
        if args.command in ("demo", "run"):
            task_path = getattr(args, "task", None)
            task = decode(task_path.read_text(encoding="utf-8")) if task_path else example_task()
            state, _ = runtime.create(task, request_key=getattr(args, "request_key", None))
            state = runtime.run(state["id"])
        elif args.command == "resume":
            runtime.resume(args.run_id)
            state = runtime.run(args.run_id)
        elif args.command == "review":
            state = runtime.review(args.run_id, args.decision, args.feedback)
            if state["status"] == "ready":
                state = runtime.run(args.run_id)
        elif args.command == "cancel":
            state = runtime.cancel(args.run_id)
        elif args.command == "events":
            print(json.dumps(store.events(args.run_id), ensure_ascii=False, indent=2))
            return 0
        else:
            state = store.get(args.run_id)
        print_result(state, store)
        return 1 if state["status"] in ("error", "budget_exhausted") else 0
    finally:
        store.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (InvalidInput, OSError, KeyError) as exc:
        print(str(exc) if isinstance(exc, InvalidInput) else "本地文件或任务不可用，请检查路径／任务编号。")
        raise SystemExit(1)
