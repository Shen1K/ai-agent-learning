"""真实本地HTTP集成测试：只有127.0.0.1通信，不连接外部模型。"""

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib import error, request

from creative_agent.contracts import example_task
from creative_agent.providers import MockProvider, RagMockProvider
from creative_agent.rag import KnowledgeBase
from creative_agent.runtime import AgentRuntime
from creative_agent.service import make_server
from creative_agent.store import Store


TOKEN = "local-test-token-not-a-real-secret"


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "runs.sqlite3")
        self.runtime = AgentRuntime(self.store, MockProvider())
        self.server = make_server(self.runtime, TOKEN, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:" + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.agent_service.close()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.store.close()
        self.temp.cleanup()

    def call(self, path, data=None, headers=None, auth=True):
        all_headers = {"Content-Type": "application/json"}
        if auth:
            all_headers["Authorization"] = "Bearer " + TOKEN
        all_headers.update(headers or {})
        req = request.Request(self.base + path, headers=all_headers,
                              data=None if data is None else json.dumps(data).encode())
        try:
            with request.urlopen(req, timeout=3) as response:
                return response.status, json.load(response)
        except error.HTTPError as exc:
            with exc:
                return exc.code, json.load(exc)

    def wait_state(self, run_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            code, state = self.call("/runs/" + run_id)
            self.assertEqual(code, 200)
            if state["status"] not in ("ready", "running"):
                return state
            time.sleep(0.01)
        self.fail("本地模拟任务未及时完成")

    def test_http_creation_trace_idempotency_and_human_approval(self):
        payload = {"task": example_task()}
        code, first = self.call("/runs", payload, {"Idempotency-Key": "http-1"})
        self.assertEqual(code, 202)
        code, again = self.call("/runs", payload, {"Idempotency-Key": "http-1"})
        self.assertEqual(first["id"], again["id"])
        state = self.wait_state(first["id"])
        self.assertEqual(state["status"], "awaiting_review")
        self.assertEqual(len(state["candidates"]), 2)
        _, result = self.call("/runs/" + first["id"] + "/events")
        tools = [e["payload"]["name"] for e in result["events"] if e["kind"] == "tool_requested"]
        self.assertEqual(tools, ["get_materials", "save_candidate", "evaluate_candidate",
                                 "save_candidate", "evaluate_candidate", "submit_candidate"])
        code, result = self.call("/runs/" + first["id"] + "/review", {"decision": "approve", "feedback": ""})
        self.assertEqual(code, 202)
        self.assertEqual(result["status"], "approved")
        self.assertFalse(self.store.get(first["id"])["publication_authorized"])

    def test_auth_and_cross_origin_rejection(self):
        self.assertEqual(self.call("/health", auth=False)[0], 401)
        self.assertEqual(self.call("/health", headers={"Origin": "https://untrusted.example"})[0], 403)
        code, result = self.call("/health")
        self.assertEqual(code, 200)
        self.assertEqual(result["backend"]["mode"], "mock")

    def test_invalid_input_and_unknown_routes(self):
        self.assertEqual(self.call("/runs", {"task": {}, "api_key": "not-accepted"})[0], 400)
        self.assertEqual(self.call("/not-found")[0], 404)
        self.assertEqual(self.call("/runs/" + "0" * 32)[0], 404)

    def test_revision_is_enqueued_and_saved_as_new_version(self):
        _, created = self.call("/runs", {"task": example_task()})
        self.wait_state(created["id"])
        code, result = self.call("/runs/" + created["id"] + "/review",
                                 {"decision": "revise", "feedback": "请改得更自然，保留产品线索。"})
        self.assertEqual(code, 202)
        state = self.wait_state(created["id"])
        self.assertEqual(state["status"], "awaiting_review")
        self.assertEqual(state["selected_candidate_id"], "c3")

    def test_conflicting_idempotency_key(self):
        _, created = self.call("/runs", {"task": example_task()}, {"Idempotency-Key": "same"})
        task = example_task()
        task["brief"]["tone"] = "不同语气"
        self.assertEqual(self.call("/runs", {"task": task}, {"Idempotency-Key": "same"})[0], 400)
        self.wait_state(created["id"])

    def test_rag_runs_through_same_http_api_and_persists_sources(self):
        # 等价于serve --rag：只在测试开始、尚无任务时设置服务器依赖。
        self.runtime.provider = RagMockProvider()
        self.runtime.knowledge_base = KnowledgeBase.bundled()
        code, created = self.call("/runs", {"task": example_task()})
        self.assertEqual(code, 202)
        state = self.wait_state(created["id"])
        self.assertEqual(state["status"], "awaiting_review")
        self.assertEqual(len(state["rag"]["retrievals"]), 1)
        self.assertTrue(state["candidates"][0]["knowledge_context"])
        _, result = self.call("/runs/" + created["id"] + "/events")
        searches = [e for e in result["events"] if e["kind"] == "tool_finished"
                    and e["payload"]["tool"] == "search_knowledge"]
        self.assertEqual(len(searches), 1)
        self.assertTrue(searches[0]["payload"]["result"]["hits"])


if __name__ == "__main__":
    unittest.main()
