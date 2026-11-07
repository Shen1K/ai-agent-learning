"""本机HTTP服务：只负责接收请求、安排执行、返回持久状态。

HTTP线程不直接等待模型完成。一个后台worker执行Agent，客户端轮询状态。
仅监听127.0.0.1，要求Bearer令牌；这是单用户教学服务，不是生产网关。
"""

from concurrent.futures import ThreadPoolExecutor
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import threading
from urllib.parse import urlsplit

from .contracts import InvalidInput, decode, exact, require, string


class AgentService:
    def __init__(self, runtime):
        self.runtime = runtime
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="creative-agent")
        self.lock = threading.RLock()
        self.scheduled = set()
        self.closing = False

    def enqueue(self, run_id):
        with self.lock:
            require(not self.closing, "服务正在关闭")
            if run_id in self.scheduled:
                return
            require(len(self.scheduled) < 16, "本地执行队列已满，请稍后再试")
            self.scheduled.add(run_id)
            future = self.worker.submit(self.runtime.run, run_id)
            future.add_done_callback(lambda f: self._done(run_id, f))

    def _done(self, run_id, future):
        # 用户可能在worker刚提交待确认结果时就反馈。不能因旧worker尚未清理
        # scheduled标记而丢掉这次继续执行请求。
        with self.lock:
            self.scheduled.discard(run_id)
            if future.exception() is not None:
                def mark_error(state):
                    if state["status"] == "ready":
                        state.update(status="error", error="执行配置不匹配或worker异常，请检查服务配置。")
                self.runtime.store.mutate(run_id, mark_error, "worker_error")
            elif not self.closing and self.runtime.store.get(run_id)["status"] == "ready":
                self.enqueue(run_id)

    def close(self):
        with self.lock:
            self.closing = True
        self.worker.shutdown(wait=True)


def make_server(runtime, token, port=8765):
    string(token, "服务令牌", 256)
    require(len(token) >= 16 and token.isascii() and not any(c.isspace() for c in token),
            "CREATIVE_AGENT_SERVICE_TOKEN至少16位ASCII字符且不能有空白")
    service = AgentService(runtime)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # 默认HTTP日志可能包含路径和资料，不记录Authorization、请求体或原始异常。
            pass

        def _reply(self, status, body):
            raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(raw)

        def _body(self):
            require(self.headers.get_content_type() == "application/json", "请求必须为application/json")
            require(self.headers.get("Transfer-Encoding") is None, "不支持分块上传")
            value = self.headers.get("Content-Length", "")
            require(value.isdigit() and 0 < int(value) <= 32768, "请求体须为1至32768字节")
            raw = self.rfile.read(int(value))
            require(len(raw) == int(value), "请求体不完整")
            return decode(raw)

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def _dispatch(self, method):
            self.connection.settimeout(5)
            try:
                # 不开放跨域访问；防止网页静默驱动本地、可能计费的模型服务。
                if self.headers.get("Origin"):
                    return self._reply(403, {"error": "本地教学API不接受浏览器跨域请求"})
                host = self.headers.get("Host", "")
                if host not in (f"127.0.0.1:{server.server_port}", f"localhost:{server.server_port}"):
                    return self._reply(403, {"error": "Host不匹配"})
                auth = self.headers.get("Authorization", "")
                if not hmac.compare_digest(auth.encode("utf-8"), ("Bearer " + token).encode("utf-8")):
                    return self._reply(401, {"error": "需要有效的本地服务令牌"})
                path = urlsplit(self.path).path
                if method == "GET" and path == "/health":
                    return self._reply(200, {"status": "ok", "backend": runtime.backend,
                                             "publication_enabled": False})
                if method == "GET" and path == "/runs":
                    return self._reply(200, {"runs": runtime.store.list_runs()})
                if method == "POST" and path == "/runs":
                    data = self._body()
                    require(type(data) is dict and "task" in data and set(data) <= {"task", "limits"}, "创建请求字段不匹配")
                    state, created = runtime.create(data["task"], data.get("limits"),
                                                    self.headers.get("Idempotency-Key"))
                    if state["status"] == "ready":
                        service.enqueue(state["id"])
                    return self._reply(202, {"id": state["id"], "created": created, "status": state["status"]})
                match = re.fullmatch(r"/runs/([0-9a-f]{32})(?:/(events|resume|cancel|review))?", path)
                if not match:
                    return self._reply(404, {"error": "接口不存在"})
                run_id, operation = match.groups()
                if method == "GET" and operation is None:
                    return self._reply(200, runtime.store.get(run_id))
                if method == "GET" and operation == "events":
                    return self._reply(200, {"events": runtime.store.events(run_id)})
                if method == "POST" and operation in ("resume", "cancel", "review"):
                    data = self._body()
                    if operation == "review":
                        exact(data, ("decision", "feedback"), "人工确认")
                        state = runtime.review(run_id, data["decision"], data["feedback"])
                    else:
                        exact(data, (), "控制请求")
                        state = runtime.resume(run_id) if operation == "resume" else runtime.cancel(run_id)
                    if state["status"] == "ready":
                        service.enqueue(run_id)
                    return self._reply(202, {"id": run_id, "status": state["status"]})
                return self._reply(405, {"error": "方法不支持"})
            except KeyError:
                self._reply(404, {"error": "任务不存在"})
            except InvalidInput as exc:
                self._reply(400, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                return
            except Exception:
                self._reply(500, {"error": "内部服务错误；请查看任务状态"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.agent_service = service
    return server
