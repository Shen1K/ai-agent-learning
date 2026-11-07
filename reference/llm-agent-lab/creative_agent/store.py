"""SQLite保存任务快照与事件；两者在同一事务中提交。

第一版采用单进程所有权锁，进程内允许HTTP读取与一个后台worker。
它不是分布式队列。进程退出后状态还在；恢复不会自动重发可能计费的请求。
"""

from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import uuid

from .contracts import InvalidInput, require, string


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class Store:
    def __init__(self, path):
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(str(path) + ".lock", "a+")
        try:
            fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._file.close()
            raise InvalidInput("该任务库已由另一个进程使用；请先关闭原服务。") from None
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, request_key TEXT UNIQUE, request_hash TEXT NOT NULL,
                state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                time TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
        """)
        # 取得独占进程锁后，才可判定旧running属于中断，不会抢另一个进程的任务。
        with self.lock, self.db:
            for run_id, raw in self.db.execute("SELECT id,state FROM runs").fetchall():
                state = json.loads(raw)
                if state["status"] == "running":
                    state.update(status="error", error="上次进程中断；待人工确认恢复，网络调用可能已计费。")
                    self._write(state, "recovered_interruption", {})

    def close(self):
        self.db.close()
        self._file.close()

    def _event(self, run_id, kind, payload):
        self.db.execute("INSERT INTO events(run_id,time,kind,payload) VALUES(?,?,?,?)",
                        (run_id, now(), kind, dump(payload)))

    def _write(self, state, kind, payload):
        state["updated_at"] = now()
        self.db.execute("UPDATE runs SET state=? WHERE id=?", (dump(state), state["id"]))
        self._event(state["id"], kind, payload)

    def create(self, state, request_key=None):
        if request_key is not None:
            string(request_key, "幂等键", 128)
        request_spec = {key: state[key] for key in ("task", "limits", "backend", "rule_version")}
        # 同一创建键不能把无RAG任务悄悄变成RAG任务，或切换冻结的知识版本。
        # 无RAG请求保持原指纹结构，兼容旧任务的幂等重试。
        if state.get("rag"):
            request_spec["knowledge_snapshot_id"] = state["rag"]["snapshot"]["snapshot_id"]
        fingerprint = hashlib.sha256(dump(request_spec).encode()).hexdigest()
        with self.lock, self.db:
            if request_key:
                row = self.db.execute("SELECT id,request_hash FROM runs WHERE request_key=?", (request_key,)).fetchone()
                if row:
                    require(row[1] == fingerprint, "同一幂等键不能提交不同任务或预算")
                    return self.get(row[0]), False
            state = deepcopy(state)
            state.update(id=uuid.uuid4().hex, created_at=now(), updated_at=now())
            self.db.execute("INSERT INTO runs VALUES(?,?,?,?)", (state["id"], request_key, fingerprint, dump(state)))
            self._event(state["id"], "created", {"backend": state["backend"], "rule_version": state["rule_version"]})
            return state, True

    def get(self, run_id):
        with self.lock:
            row = self.db.execute("SELECT state FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError("任务不存在")
            return json.loads(row[0])

    def save(self, state, kind, payload=None):
        with self.lock, self.db:
            current = self.get(state["id"])
            # 取消请求可能在模型网络请求期间进入，不能被旧内存快照覆盖。
            if current["cancel_requested"]:
                state["cancel_requested"] = True
                state["status"] = "cancelled"
            self._write(state, kind, payload or {})

    def mutate(self, run_id, change, kind):
        with self.lock, self.db:
            state = self.get(run_id)
            change(state)
            self._write(state, kind, {})
            return state

    def events(self, run_id):
        self.get(run_id)
        with self.lock:
            rows = self.db.execute("SELECT seq,time,kind,payload FROM events WHERE run_id=? ORDER BY seq", (run_id,)).fetchall()
            return [{"seq": a, "time": b, "kind": c, "payload": json.loads(d)} for a, b, c, d in rows]

    def list_runs(self):
        with self.lock:
            rows = self.db.execute("SELECT state FROM runs ORDER BY rowid DESC LIMIT 100").fetchall()
            return [{key: json.loads(row[0])[key] for key in ("id", "status", "created_at", "backend")}
                    for row in rows]
