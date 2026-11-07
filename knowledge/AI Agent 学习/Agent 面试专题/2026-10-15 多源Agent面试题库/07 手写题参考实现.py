"""Agent 面试手写题参考，Python 3.9+，仅使用标准库。

运行本文件会执行八组本地示例，不访问模型、网络或付费服务。
这是机制示范：没有实现生产鉴权、分布式锁、完整持久 Runtime 或外部事务。
P03 的同步超时只在调用边界检查，不能强制杀死阻塞调用。
P04 要求工具遵守 asyncio 的协作取消；取消不能撤回已发生的副作用。
P08 只实现持久受理去重与状态转换，不自动重试未知外部动作。
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Awaitable, Callable, Iterable, Mapping


# P01：输入已按调用整理。一行表示一次尝试的终态或 pending，不做事件流去重。
def tool_statistics(records: Iterable[Mapping[str, Any]]) -> dict:
    stats: dict[str, dict] = {}
    for record in records:
        name, status = record['tool'], record['status']
        if not isinstance(name, str) or not name:
            raise ValueError('工具名必须非空')
        if status not in {'success', 'failed', 'pending'}:
            raise ValueError('未知状态')
        item = stats.setdefault(name, {'success': 0, 'failed': 0, 'pending': 0})
        item[status] += 1
    for item in stats.values():
        item['total'] = item['success'] + item['failed']
        item['success_rate'] = item['success'] / item['total'] if item['total'] else None
    max_failed = max((item['failed'] for item in stats.values()), default=0)
    tied = [name for name, item in stats.items()
            if max_failed > 0 and item['failed'] == max_failed]
    return {'tools': stats, 'most_failed': tied}


# P02：每一路先去重，再计算排名；排序同分时按文档 ID，便于复现。
def rrf_fuse(rankings: Iterable[Iterable[str]], c: float = 60.0,
             limit: int | None = None) -> list[tuple[str, float]]:
    if not math.isfinite(c) or c < 0:
        raise ValueError('c 必须是有限非负数')
    if limit is not None and (not isinstance(limit, int) or limit < 0):
        raise ValueError('limit 必须为非负整数或 None')
    scores: dict[str, float] = {}
    for ranking in rankings:
        unique = dict.fromkeys(ranking)
        for rank, doc_id in enumerate(unique, start=1):
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError('文档 ID 必须为非空字符串')
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (c + rank)
    result = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return result if limit is None else result[:limit]


@dataclass
class ToolSpec:
    validate: Callable[[dict], bool]
    execute: Callable[[dict], Any]


# P03：注入模型、工具、授权及验证，避免将模型文本直接当代码执行。
def run_agent(task: str, decide: Callable[[dict], dict], tools: Mapping[str, ToolSpec],
              authorize: Callable[[str, dict], bool], verify: Callable[[str], bool],
              max_steps: int = 6, max_tool_calls: int = 3,
              deadline_s: float = 5.0) -> dict:
    if max_steps < 1 or max_tool_calls < 0 or not math.isfinite(deadline_s) or deadline_s <= 0:
        raise ValueError('预算参数非法')
    state = {'task': task, 'observations': []}
    deadline = time.monotonic() + deadline_s
    calls = 0
    seen_ids: set[str] = set()

    def finish(status: str, **extra: Any) -> dict:
        return {'status': status, 'tool_calls': calls, 'state': state, **extra}

    for _ in range(max_steps):
        if time.monotonic() >= deadline:
            return finish('deadline_exhausted')
        try:
            action = decide(copy.deepcopy(state))
        except Exception as error:
            return finish('model_error', error_type=type(error).__name__)
        if time.monotonic() >= deadline:
            return finish('deadline_exhausted')
        if not isinstance(action, dict):
            state['observations'].append({'error': 'invalid_action'})
            continue
        if action.get('type') == 'wait':
            return finish('waiting', reason=str(action.get('reason', '')))
        if action.get('type') == 'finish':
            answer = action.get('answer')
            try:
                verified = isinstance(answer, str) and verify(answer)
            except Exception as error:
                return finish('verification_error', error_type=type(error).__name__)
            if verified:
                return finish('succeeded', answer=answer)
            state['observations'].append({'error': 'verification_failed'})
            continue
        if action.get('type') != 'tool':
            state['observations'].append({'error': 'unknown_action'})
            continue
        if calls >= max_tool_calls:
            return finish('tool_budget_exhausted')
        name, args, call_id = action.get('name'), action.get('args'), action.get('call_id')
        if not isinstance(name, str) or name not in tools or not isinstance(args, dict):
            state['observations'].append({'error': 'invalid_tool_or_args'})
            continue
        if not isinstance(call_id, str) or not call_id or call_id in seen_ids:
            state['observations'].append({'error': 'invalid_or_repeated_call_id'})
            continue
        try:
            valid = tools[name].validate(args)
        except Exception:
            valid = False
        if not valid:
            state['observations'].append({'call_id': call_id, 'error': 'invalid_args'})
            continue
        try:
            permitted = authorize(name, args)
        except Exception:
            permitted = False
        if not permitted:
            return finish('blocked')
        seen_ids.add(call_id)
        calls += 1
        try:
            value = tools[name].execute(args)
            observation = {'call_id': call_id, 'status': 'success', 'value': value}
        except Exception as error:
            observation = {'call_id': call_id, 'status': 'error',
                           'error_type': type(error).__name__}
        state['observations'].append(observation)
        if time.monotonic() >= deadline:
            return finish('deadline_exhausted')
    return finish('step_budget_exhausted')


# P04：总时限包含排队；局部时限从获得信号量、真正开始调用时计算。
async def execute_tools(calls: Mapping[str, Callable[[], Awaitable[Any]]],
                        max_concurrency: int = 2, tool_timeout_s: float = 1.0,
                        total_timeout_s: float = 2.0) -> dict:
    if max_concurrency < 1 or not all(math.isfinite(t) and t > 0
                                    for t in (tool_timeout_s, total_timeout_s)):
        raise ValueError('并发和时限参数非法')
    if not calls:
        return {}
    semaphore = asyncio.Semaphore(max_concurrency)

    async def worker(factory: Callable[[], Awaitable[Any]]) -> dict:
        async with semaphore:
            try:
                value = await asyncio.wait_for(factory(), timeout=tool_timeout_s)
                return {'status': 'success', 'value': value}
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                return {'status': 'tool_timeout'}
            except Exception as error:
                return {'status': 'error', 'error_type': type(error).__name__}

    tasks = {key: asyncio.create_task(worker(factory)) for key, factory in calls.items()}
    try:
        _, pending = await asyncio.wait(tasks.values(), timeout=total_timeout_s)
    except BaseException:
        for future in tasks.values():
            future.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        raise
    timed_out = {key for key, future in tasks.items() if future in pending}
    for future in pending:
        future.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    result = {}
    for key, future in tasks.items():
        if key in timed_out:
            result[key] = {'status': 'total_timeout'}
        elif future.cancelled():
            result[key] = {'status': 'cancelled'}
        else:
            result[key] = future.result()
    return result


# P05：单进程并发保护；惰性过期不主动遍历全部键。
class TTLLRU:
    def __init__(self, capacity: int, clock: Callable[[], float] = time.monotonic):
        if not isinstance(capacity, int) or capacity < 0:
            raise ValueError('capacity 必须为非负整数')
        self.capacity, self.clock = capacity, clock
        self.data: OrderedDict[Any, tuple[Any, float | None]] = OrderedDict()
        self.lock = threading.RLock()

    def get(self, key: Any, default: Any = None) -> Any:
        with self.lock:
            if key not in self.data:
                return default
            value, expires = self.data[key]
            if expires is not None and self.clock() >= expires:
                del self.data[key]
                return default
            self.data.move_to_end(key)
            return value

    def put(self, key: Any, value: Any, ttl_s: float | None = None) -> None:
        if ttl_s is not None and (not math.isfinite(ttl_s) or ttl_s < 0):
            raise ValueError('TTL 必须为有限非负数或 None')
        with self.lock:
            if self.capacity == 0 or ttl_s == 0:
                self.data.pop(key, None)
                return
            expires = None if ttl_s is None else self.clock() + ttl_s
            self.data[key] = (value, expires)
            self.data.move_to_end(key)
            while len(self.data) > self.capacity:
                self.data.popitem(last=False)

    def prune(self) -> int:
        # 显式清理是 O(n)，不能算入基础 LRU 的 O(1) 承诺。
        with self.lock:
            now = self.clock()
            expired = [key for key, (_, end) in self.data.items()
                       if end is not None and now >= end]
            for key in expired:
                del self.data[key]
            return len(expired)


# P06：锁保护补充和扣减；这是单机算法，不提供跨进程全局额度。
class TokenBucket:
    def __init__(self, capacity: float, rate: float,
                 clock: Callable[[], float] = time.monotonic):
        if not all(math.isfinite(x) and x > 0 for x in (capacity, rate)):
            raise ValueError('容量与速率必须为有限正数')
        self.capacity, self.rate, self.clock = capacity, rate, clock
        self.tokens, self.updated = capacity, clock()
        self.lock = threading.Lock()

    def consume(self, amount: float = 1.0) -> dict:
        if not math.isfinite(amount) or amount <= 0 or amount > self.capacity:
            raise ValueError('消费量必须为正且不超过桶容量')
        with self.lock:
            now = self.clock()
            if now < self.updated:
                raise RuntimeError('需要单调时钟')
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            allowed = self.tokens >= amount
            if allowed:
                self.tokens -= amount
            wait = 0.0 if allowed else (amount - self.tokens) / self.rate
            return {'allowed': allowed, 'remaining': self.tokens, 'retry_after_s': wait}


# P07：沿输入顺序稳定处理，不额外排序，维持 O(V+E)。
def topological_order(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> list[str]:
    ordered_nodes = list(nodes)
    if len(set(ordered_nodes)) != len(ordered_nodes):
        raise ValueError('节点重复')
    graph = {node: {} for node in ordered_nodes}
    indegree = {node: 0 for node in ordered_nodes}
    for left, right in edges:
        if left not in graph or right not in graph:
            raise ValueError('边包含未声明节点')
        if right not in graph[left]:
            graph[left][right] = None
            indegree[right] += 1
    queue = deque(node for node in ordered_nodes if indegree[node] == 0)
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for successor in graph[node]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)
    if len(result) != len(ordered_nodes):
        raise ValueError('依赖图存在环')
    return result


# P08：可信 tenant 必须由上层认证提供；每个线程或执行实例使用自己的连接。
class TaskStore:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, timeout=5.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('''CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, business_key TEXT NOT NULL,
            params_hash TEXT NOT NULL, params_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending','succeeded','failed','unknown')),
            result_json TEXT, UNIQUE(tenant, business_key))''')
        self.conn.commit()

    @staticmethod
    def encode(value: Any) -> str:
        # 业务若认为 1 和 1.0 等价，需要在进入此处前明确规范化。
        return json.dumps(value, sort_keys=True, ensure_ascii=False,
                          allow_nan=False, separators=(',', ':'))

    def submit(self, tenant: str, business_key: str, params: dict) -> dict:
        if not isinstance(tenant, str) or not tenant or not isinstance(business_key, str) or not business_key:
            raise ValueError('租户和业务键必须非空')
        encoded = self.encode(params)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with self.conn:
            self.conn.execute('''INSERT INTO tasks
                (task_id,tenant,business_key,params_hash,params_json,status)
                VALUES(?,?,?,?,?,'pending')
                ON CONFLICT(tenant,business_key) DO NOTHING''',
                (str(uuid.uuid4()), tenant, business_key, digest, encoded))
            row = self.conn.execute('SELECT * FROM tasks WHERE tenant=? AND business_key=?',
                                    (tenant, business_key)).fetchone()
            if row['params_hash'] != digest or row['params_json'] != encoded:
                raise ValueError('同一业务键的参数冲突')
        return self.get(tenant, row['task_id'])

    def get(self, tenant: str, task_id: str) -> dict:
        row = self.conn.execute('SELECT task_id,status,result_json FROM tasks WHERE tenant=? AND task_id=?',
                                (tenant, task_id)).fetchone()
        if row is None:
            raise KeyError('未找到当前租户任务')
        return {'task_id': row['task_id'], 'status': row['status'],
                'result': json.loads(row['result_json']) if row['result_json'] is not None else None}

    def set_status(self, tenant: str, task_id: str, status: str, result: Any = None) -> dict:
        if status not in {'succeeded', 'failed', 'unknown'}:
            raise ValueError('非法目标状态')
        encoded = self.encode(result)
        # 提前取得写事务，确保读状态与更新属于同一临界区。
        self.conn.execute('BEGIN IMMEDIATE')
        try:
            old = self.get(tenant, task_id)
            if old['status'] == status and self.encode(old['result']) == encoded:
                self.conn.commit()
                return old
            allowed = {'pending': {'succeeded', 'failed', 'unknown'},
                       'unknown': {'succeeded', 'failed'}, 'succeeded': set(), 'failed': set()}
            if status not in allowed[old['status']]:
                raise ValueError('不允许的状态转换')
            self.conn.execute('UPDATE tasks SET status=?,result_json=? WHERE tenant=? AND task_id=?',
                              (status, encoded, tenant, task_id))
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise
        return self.get(tenant, task_id)

    def close(self) -> None:
        self.conn.close()


def run_examples() -> None:
    # 这八组是局部参考实现验证，不是 Agent 线上质量评测。
    stats = tool_statistics([{'tool': 'search', 'status': 'success'},
                             {'tool': 'search', 'status': 'failed'},
                             {'tool': 'draw', 'status': 'pending'}])
    assert stats['tools']['search']['success_rate'] == 0.5
    assert stats['tools']['draw']['success_rate'] is None
    assert stats['most_failed'] == ['search']
    fused = rrf_fuse([['A', 'A', 'B'], ['B', 'C']], c=10)
    assert [doc for doc, _ in fused] == ['B', 'A', 'C']
    assert rrf_fuse([]) == []

    def decide(state: dict) -> dict:
        if not state['observations']:
            return {'type': 'tool', 'name': 'double', 'args': {'x': 2}, 'call_id': 'c1'}
        return {'type': 'finish', 'answer': '4'}
    tools = {'double': ToolSpec(lambda a: set(a) == {'x'} and type(a['x']) is int,
                                lambda a: a['x'] * 2)}
    assert run_agent('翻倍', decide, tools, lambda n, a: True,
                     lambda answer: answer == '4')['status'] == 'succeeded'
    assert run_agent('翻倍', decide, tools, lambda n, a: False,
                     lambda answer: True)['status'] == 'blocked'
    invalid = lambda state: {'type': 'not_supported'}
    assert run_agent('非法行动', invalid, {}, lambda n, a: False,
                     lambda answer: False, max_steps=2)['status'] == 'step_budget_exhausted'

    async def async_case() -> None:
        async def fast() -> int:
            await asyncio.sleep(0)
            return 7
        async def slow() -> int:
            await asyncio.sleep(1)
            return 9
        result = await execute_tools({'fast': fast, 'slow': slow}, tool_timeout_s=0.01,
                                     total_timeout_s=0.1)
        assert result['fast']['value'] == 7 and result['slow']['status'] == 'tool_timeout'
        result = await execute_tools({'slow': slow}, tool_timeout_s=1, total_timeout_s=0.01)
        assert result['slow']['status'] == 'total_timeout'
        assert await execute_tools({}) == {}
    asyncio.run(async_case())

    now = [0.0]
    cache = TTLLRU(2, clock=lambda: now[0])
    cache.put('A', 1, ttl_s=1)
    cache.put('B', 2)
    assert cache.get('A') == 1
    cache.put('C', 3)
    assert cache.get('B') is None
    now[0] = 1.0
    assert cache.get('A') is None
    bucket = TokenBucket(2, 1, clock=lambda: now[0])
    assert bucket.consume(2)['allowed']
    assert not bucket.consume()['allowed']
    now[0] = 2.0
    assert bucket.consume()['allowed']

    order = topological_order(['A', 'B', 'C', 'D'], [('A', 'C'), ('B', 'C'), ('A', 'C')])
    assert set(order) == {'A', 'B', 'C', 'D'}
    assert order.index('A') < order.index('C') and order.index('B') < order.index('C')
    try:
        topological_order(['A', 'B'], [('A', 'B'), ('B', 'A')])
    except ValueError:
        pass
    else:
        raise AssertionError('未检测到环')

    with TemporaryDirectory() as directory:
        path = str(Path(directory) / 'tasks.sqlite3')
        store = TaskStore(path)
        first = store.submit('tenantA', 'draft1', {'version': 1})
        assert store.submit('tenantA', 'draft1', {'version': 1}) == first
        try:
            store.submit('tenantA', 'draft1', {'version': 2})
        except ValueError:
            pass
        else:
            raise AssertionError('未检测到参数冲突')
        store.set_status('tenantA', first['task_id'], 'unknown')
        done = store.set_status('tenantA', first['task_id'], 'succeeded', {'id': 'artifact1'})
        assert store.set_status('tenantA', first['task_id'], 'succeeded', {'id': 'artifact1'}) == done
        try:
            store.get('tenantB', first['task_id'])
        except KeyError:
            pass
        else:
            raise AssertionError('租户隔离失败')
        store.close()
        reopened = TaskStore(path)
        assert reopened.submit('tenantA', 'draft1', {'version': 1})['status'] == 'succeeded'
        reopened.close()
    print('八组参考实现示例通过；未调用模型、网络或外部业务服务。')


if __name__ == '__main__':
    run_examples()
