# 广告创意 Agent 服务

本项目用广告创意业务学习Agent开发与服务化。模型依据工具反馈决定下一步；程序执行白名单工具、保存状态、限制预算，并把交付停在用户确认处。

第一版范围是中文标题和正文，不包含图片生成、广告投放或真实AB实验。默认是离线脚本模拟；真实模式提供DeepSeek工具调用适配，尚未进行付费实测。模拟模式的“评审通过”只用于验证程序分支。

2026-11-01新增可选的本地RAG教学基线，见下文“RAG教学：按失败场景迭代”。本地检索是真实BM25计算；演示中的模型选工具、创作文案和评审仍为脚本替身。无RAG原流程保留。

## 先运行一次

从本公开仓库根目录执行以下命令，无需安装新依赖：

```bash
cd reference/llm-agent-lab
python3 -S -B -m creative_agent demo
```

演示会先生成产品指向不清的文案，再根据工具反馈保存改稿，最后进入`awaiting_review`。输出包含任务ID、工具顺序和最终候选。默认状态库是项目内`.creative-agent/runs.sqlite3`，已加入Git忽略，不覆盖已有数据、模型或报告。

`-S`跳过本机额外包初始化，适用于这个仅使用标准库的示范；`-B`不生成字节码缓存。当前验证环境为macOS Python 3.9。进程所有权锁使用`fcntl`，也适用于常见Linux；Windows需要替换该锁实现。

## 为什么这是 Agent 而不是固定流水线

[runtime.py](<runtime.py>)的`run()`只固定通用循环：获得模型动作、校验、执行工具、保存结果、把反馈交给下一轮。它没有写死每次都必须生成两稿或评审后必定改稿。

真实模式由DeepSeek返回工具名与参数。模型可以先请求澄清，也可以提交已通过检查的候选，或根据失败反馈继续修改。每轮最多接受一个工具调用，避免并发写入和恢复语义在教学初版中混在一起。

离线`MockProvider`则是明确标注的脚本替身，不是大模型，也不是效果评测。测试通过替换不同动作序列验证：同一个runtime既能走改稿分支，也能在没有生成候选时暂停询问。

## 模块与阅读顺序

| 文件 | 职责 | 先看什么 |
|---|---|---|
| [runtime.py](<runtime.py>) | Agent循环、执行边界、暂停恢复与确认 | `run`、`_execute` |
| [contracts.py](<contracts.py>) | 任务约定、工具Schema与参数验证 | `TOOLS`、`validate_action` |
| [providers.py](<providers.py>) | 模型适配；真实Function Calling／离线替身 | `decide`与`judge`的区别 |
| [prompts.py](<prompts.py>) | 生成上下文与独立评审上下文 | `initial_messages`、`judge_messages` |
| [evaluation.py](<evaluation.py>) | 程序检查、评审协议和原文引用核对 | `hard_checks`、`validate_judgment` |
| [store.py](<store.py>) | SQLite快照与事件事务、创建幂等、进程锁 | `create`、`save` |
| [service.py](<service.py>) | HTTP接口与单后台worker | `enqueue`、接口分派 |
| [__main__.py](<__main__.py>) | 启动演示、命令行管理、HTTP服务 | `main` |
| [rag.py](<rag.py>) | 资料快照、分块、BM25、父段返回与证据预算 | `snapshot`、`search`、`candidate_context` |
| [rag_eval.py](<rag_eval.py>) | 固定语料／查询比较两种返回策略 | `evaluate`，不要把预期证据传给搜索端 |
| [knowledge/documents.json](<knowledge/documents.json>) | 虚构品牌规范、案例、反馈、旧版及其他商品干扰资料 | `status`、`required`、`sections` |
| [knowledge/cases.json](<knowledge/cases.json>) | 8个可复现的教学开发案例 | `required_quotes`只用于评分，不是检索输入 |

先学习模型、工具、状态、停止条件这四件事，再把框架映射进来。此版未安装LangGraph、FastAPI或Redis：使用标准库是为了直接看见机制，不表示这些库不适合生产服务。HTTP实现仅作本地教学；后续可以保留runtime和工具层，替换为正式API框架与持久任务队列。

## 五个基础工具与可选检索工具

| 工具 | 输入 | 程序负责的边界 |
|---|---|---|
| `get_materials` | 空对象 | 只返回本任务冻结的资料，不读任意路径；不是RAG |
| `save_candidate` | 标题、正文、方向、假设、父候选ID | 新建版本，不覆盖原稿；拒绝同轮完全相同文案 |
| `evaluate_candidate` | 本任务候选ID | 先运行硬检查，满足后才调用隔离的语义评审 |
| `submit_candidate` | 本轮已通过检查的候选ID | 仅转为待用户确认；不能跳过评审或复用反馈前的旧稿 |
| `request_clarification` | 一个具体问题 | 转为待输入并停止循环，不继续猜事实 |
| `search_knowledge`（仅启用RAG时） | `query`，最多240字符 | 只查本任务冻结的有效商品范围资料；不能传路径、URL或扩大范围 |

没有任意代码执行、网页访问、文件读取或发布广告的工具。模型提出不在名单内的工具时，程序返回拒绝结果，不根据模型字符串动态执行函数。

RAG模式下`get_materials`还提供所有必需品牌规范，不让模型通过不检索来漏掉这些要求；参考案例则由检索工具按需取得。当前商品范围过滤不是用户身份认证，不能冒充生产多租户ACL。

## 本版评审与隔离方式

本版不把旧20项等权总分接成上线门槛。程序检查标题长度、整段标题加正文的必含词、禁用词；标题计数不计空白及Unicode标点，其余字符逐个计数。默认案例刻意要求出现“咖啡杯”，便于观察首稿失败与修改，不代表所有广告必须直述品类。

通过硬检查后，模型检查`brief_fit`、`product_link`、`tone_fit`、`facts_supported`，逐项返回true/false/null、理由和原文证据。任何false/null都不能提交。证据必须在候选／事实原文中；存在引文仍不保证语义判断正确，四项也不构成全面创意价值或合规审查。

生成侧能够看见任务历史和工具反馈。评审侧只接收本稿文字、必要brief、商品事实和用户补充要求，不接收生成假设、作者自评、父稿、历史分数。同一底层模型可以承担两个调用；上下文隔离不保证偏好独立。

启用RAG时，保存候选会冻结`knowledge_context`：必需规范加预算内的已检索原文及来源。它表示当时可用证据，不表示模型实际引用了每条资料。评审可以引用`knowledge.E1`等字段，程序核对逐字引文；检索问题、排序分数和生成理由不发给评审。`facts_supported=true`仍须引用冻结的`product.facts`，历史案例不能替代当前商品事实。引用存在不等于语义正确，品牌规范的遵循仍需真实评审校准。

已经完成的同一候选评审会复用，不自动反复抽样直到得到高分。工具失败或裁判返回无效JSON／假引文时，不记录假低分，而是保留待执行动作、进入`error`。

## 状态和失败恢复

| 状态 | 含义 | 下一步 |
|---|---|---|
| `ready` | 已创建或获得继续指令 | 调度执行 |
| `running` | 正在决策或执行工具 | 可查询或取消 |
| `needs_input` | Agent请求补充 | 用户用`answer`回答 |
| `awaiting_review` | 候选通过本轮检查 | 用户`approve`或`revise` |
| `approved` | 用户确认本地草稿 | 结束；仍未发布 |
| `error` | 调用／协议／内部执行异常 | 检查后显式`resume` |
| `budget_exhausted` | 轮数、调用数、候选数或上下文到上限 | 停止；不自动加预算 |
| `cancelled` | 用户取消 | 结束，不恢复 |

任务保存冻结资料、规则版本、模型配置、候选父子关系、评审、待执行动作、对话、用户反馈与预算计数。事件记录模型调用开始／结束及每个工具的参数和结果。

评审超时时，候选已经保存，恢复会重试待执行的评审，而不是重新生成文案。每次模型请求发出前即计入预算，包括超时／失败请求；默认最多16轮决策、20次后端调用、4稿，恢复和用户改稿共享这些上限。每次网络请求30秒是网络操作超时，不是整个任务的绝对墙钟上限。调用数预算也不是精确人民币预算；token用量在供应商提供时记录，失败请求可能没有用量。

取消是协作式的：已发出的HTTP模型请求不能保证撤销，返回后不再执行后续动作，也不能保证退款。进程崩溃后旧`running`转为`error`，不会自动重发请求。若供应商已经执行而本地尚未保存响应，恢复可能再次计费；本版不承诺外部调用恰好一次。

本地快照、工具结果、候选与事件在SQLite事务中提交。一个数据库只允许一个进程拥有；服务运行时请通过HTTP管理，不能同时用CLI打开同一数据库。该限制防止两个教学worker重复推进任务，不是分布式锁方案。

## 命令行管理

下列`任务ID`替换为演示实际输出。服务运行时不要同时用这些CLI打开同一任务库。

```bash
python3 -S -B -m creative_agent show 任务ID
python3 -S -B -m creative_agent events 任务ID
python3 -S -B -m creative_agent review 任务ID --decision approve
python3 -S -B -m creative_agent review 任务ID --decision revise --feedback '保留杯子线索，让语气更自然。'
python3 -S -B -m creative_agent review 任务ID --decision answer --feedback '面向办公人群。'
python3 -S -B -m creative_agent resume 任务ID
python3 -S -B -m creative_agent cancel 任务ID
```

`approve`仅接受待确认任务，`revise`进入新反馈轮，`answer`仅接受待输入任务。离线替身不会真正理解任意反馈，只验证新版本和状态流转。真实模式任务恢复／改稿时也要传`--live`，不能静默换成模拟模式。事实或冻结约束需要变化时，应创建新任务，而不是在聊天反馈中改写规则。

## 启动本地 HTTP 服务

先在准备启动服务的终端生成本地令牌。这不是DeepSeek密钥，也不要发送给聊天：

```bash
export CREATIVE_AGENT_SERVICE_TOKEN="$(python3 -S -c 'import secrets; print(secrets.token_urlsafe(24))')"
python3 -S -B -m creative_agent serve
```

默认仅监听`127.0.0.1:8765`。其他终端调用时须自行安全地配置同一个令牌；令牌不写入任务库、不由服务打印。所有接口要求`Authorization: Bearer <本地令牌>`，不接受浏览器Origin请求。没有对公网开放或多租户隔离能力。

| 方法与路径 | 请求体 | 用途 |
|---|---|---|
| `GET /health` | 无 | 查看模式，不证明供应商可用 |
| `POST /runs` | `{"task": {...}}`，可选`limits` | 创建并排队；返回任务ID |
| `GET /runs` | 无 | 最近100个任务摘要 |
| `GET /runs/{id}` | 无 | 完整状态与候选 |
| `GET /runs/{id}/events` | 无 | 逐步执行记录 |
| `POST /runs/{id}/resume` | `{}` | 显式恢复错误或待执行任务 |
| `POST /runs/{id}/cancel` | `{}` | 取消 |
| `POST /runs/{id}/review` | `{"decision":"approve","feedback":""}` | 确认；也支持`revise`／`answer` |

POST须为`application/json`，请求体上限32KB。创建可传`Idempotency-Key`；相同键、相同任务与预算返回原任务ID，不重复创建。不同内容复用同一个键会被拒绝。第一版参数／状态冲突统一返回400，缺令牌401，跨域／Host异常403，不存在404；HTTP错误体不给出原始堆栈。

在已配置同一令牌的客户端终端，提交内置案例：

```bash
python3 -S -c 'import json; from creative_agent.contracts import example_task; print(json.dumps({"task": example_task()}, ensure_ascii=False))' | curl --fail-with-body http://127.0.0.1:8765/runs \
  -H "Authorization: Bearer $CREATIVE_AGENT_SERVICE_TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: cup-demo-001' --data-binary @-
```

服务返回202只表示已受理，不表示完成。随后查询任务与事件；`awaiting_review`才表示本轮候选待确认。服务使用一个后台worker，最多安排16个待执行／执行中的任务；队列满时请求可能已创建为`ready`但未安排执行，可在负载下降后用同一幂等键重试或调用resume。重启后不自动恢复旧任务。

## 真实 DeepSeek 模式

确认当前终端已经配置`DEEPSEEK_API_KEY`，不要将密钥写入JSON、代码、提交信息或聊天。以下命令会发送任务资料、候选和评审请求，可能多次计费：

```bash
python3 -S -B -m creative_agent run --live
# 或者为自己的任务提供文件：
python3 -S -B -m creative_agent run --live --task creative_agent/example_task.json
# 启动真实模型服务仍须配置上述本地服务令牌：
python3 -S -B -m creative_agent serve --live
```

接口格式依据[DeepSeek Chat Completions官方文档](https://api-docs.deepseek.com/api/create-chat-completion/)核对：提供`tools`、使用`tool_choice=required`并关闭thinking，回传assistant工具请求与匹配的tool结果。服务只接受一次一个工具调用，遇到多工具或截断响应会暂停，不猜测执行。非标准strict模式未启用，参数仍由本地校验。

传输复用[已有DeepSeek入口](<../lessons/creative_studio/deepseek_review.py>)的HTTPS、环境变量认证、禁止重定向、安全错误处理。模型默认使用该入口的配置。旧价格核算／提取模块保留，未被新Agent自动调用；需要结构化订单时再把它接为业务工具，不能拿价格规则替代创意评审。

## 验证与已知边界

运行新Agent测试：

```bash
python3 -S -B -m unittest discover -s tests -p 'test_creative_agent*.py' -v
```

其中HTTP测试会短暂监听127.0.0.1随机端口并自动关闭，某些沙箱需要允许本地监听。测试包含决策分支、非法工具、提前提交、重复候选、预算、假引文、独立上下文、创建幂等、用户改稿、取消、进程恢复与HTTP鉴权；外部模型网络调用全部替换为假响应。

运行数据包含brief、文案和工具结果，虽然不含认证头，仍可能涉及业务保密信息。不要未经授权把真实客户资料送入API或提交状态库。

尚未实现：真实模型质量基准、严格费用账单预算、完整语义安全防护、分布式执行、Web界面、图片生成、语义Embedding／混合检索／模型重排、生产检索服务、线上AB平台。提示中的不可信数据隔离不是已证明的提示注入防御；安全边界首先来自有限工具和程序权限检查。当前未运行真实DeepSeek闭环，不能把离线模拟包装为模型成功案例。

## RAG教学：按失败场景迭代

### 先跑接通后的Agent，不用看代码

在项目根目录执行。独立状态库与旧演示分开；重复运行会新增任务，不覆盖旧任务：

```bash
python3 -S -B -m creative_agent demo --rag --db .creative-agent/rag-teaching.sqlite3
```

会展示读取资料、检索原文、保存候选、评审、提交待确认的工具过程。`search_knowledge`的来源ID、版本、文本与BM25分数都会显示，模型仍是脚本。原来的`demo`不加`--rag`时保持原流程。

本地HTTP可用`serve --rag`启用同一能力；仍需要本地服务令牌。创建请求结构不变，由服务端配置资料库，调用者不能在工具参数里选择任意资料路径。`run --rag --live`可使用已有真实适配器，但会向外发送任务和检索到的资料并可能计费；本轮没有执行。恢复或改稿要保持原后端，离线RAG任务使用`resume/review --rag`；已存在任务始终使用自己的冻结知识快照，不通过重启服务自动迁移。

### 每一层谁负责

1. 程序根据任务商品ID筛选`active`资料，校验同一文档没有两个有效版本，并冻结快照；本地商品范围过滤不是账户授权。
2. `get_materials`提供不可省略的规范；模型可按需调用`search_knowledge`查案例或反馈。
3. 检索程序将预整理段落按换行拆成小片段，使用中文双字／英文词项的BM25排序；不需要联网。
4. 默认命中小片段后返回所在完整段落；按段落来源去重，保留版本和字符区间。返回字符预算不足时跳过整段并报告，不静默截掉限制条件。
5. 原文作为工具结果进入生成上下文；每次查询和结果随任务事件保存，最多4次检索，每次最多3个段落、1600原文字符。
6. 保存候选时另冻结可用证据，必需规范优先，其他参考取最近检索并去重；评审只看这份证据而不是生成历史。上下文超过预算会省略部分参考，因此生成端看到过不等于评审端一定保留，不能据此放宽事实门槛。
7. 商品事实仍来自任务`product.facts`。知识库用于品牌要求与创意参考，不自动覆盖事实、规则或发布权限。

**教学简化**：文档已经人工整理成JSON，未实现PDF/PPT/OCR解析；索引统计每次在内存计算，适用于这个小语料，不是生产索引服务。BM25的`k1=1.2,b=0.75`未做业务调优，中文双字不是成熟分词。只要有共同词项可能返回结果，尚无校准的相关性拒绝阈值。关键词检索不会自动理解同义改写。

### 第一个可复现实验：找到规范，却丢了例外

原文第一行允许“幽默和俏皮表达”，第二行说“不得暗示咖啡杯可以提高工作效率”。同一个问题，保持检索排序不变，只改返回范围：

```bash
python3 -S -B -m creative_agent.rag_eval --query '幽默 俏皮表达' --strategy small_chunks
python3 -S -B -m creative_agent.rag_eval --query '幽默 俏皮表达' --strategy parent_context
```

第一条只拿到允许条款，第二条同时拿到例外。这里验证的是**检索返回内容完整性**，不是模型生成质量；Agent里必需规范本身已经固定提供，不故意删除安全上下文来展示失败。

运行全部教学案例：

```bash
python3 -S -B -m creative_agent.rag_eval
```

2026-11-01的真实本地检索结果（8个公开、刻意构造的开发案例，固定`top_k=1`）：

| 策略 | 案例判据通过 | 5个有预期证据问题的来源命中 | 合计返回原文字符 |
|---|---|---|---|
| 只返回命中小片段 | 4/8 | 4/5 | 117 |
| 命中后返回完整父段 | 7/8 | 4/5 | 377 |

案例判据包括目标来源与必要原文是否齐全，或预期无可用证据时是否返回空。父段方案改善了例外、否决原因、无效果数据说明的覆盖，但**没有提高来源命中**；代价是更多上下文。两者对“喘息”的查询都未找到“片刻休息”的案例，这是保留的词汇漏检，不硬编码答案修复。不能把7/8说成创意准确率、线上收益或独立测试成绩。

### 后续怎样从项目问题推导技术（不是本轮全部已实现）

| 项目问题／课堂场景 | 对应技术与题库 | 对照与验收 | 当前边界 |
|---|---|---|---|
| 品牌手册扫描件没有正确文字 | OCR、版面／表格解析，D01 | 核对原文覆盖、条款与来源位置后再建索引 | 当前只有预整理JSON，解析待实现 |
| 找到幽默条款却漏掉例外；找到被否决句却没拿到原因 | 父子分块，D02 | 相同问题与排序，比证据完整性和输入长度 | 两种返回策略已有 |
| 旧品牌口号、其他商品参考混入 | 元数据过滤、版本治理，E08 | 查归档／其他商品专属词，确认内容不进入结果；旧任务保持版本 | 商品范围／active过滤、快照已有；账号ACL未实现 |
| 用户说“喘息”，资料写“片刻休息” | 真实Embedding、BM25+向量、RRF，D06/E01/E02 | 保留语义改写与型号精确词两类测试，比较漏检、误检、时延 | 已有失败案例；向量、融合待选模型和实现 |
| 候选主题相近但不满足当前brief | Cross Encoder重排，E03/E04 | 固定初召回候选，对比排序和上下文证据；记录额外时延 | 待实现；不会拿BM25二次排序冒充模型重排 |
| 一句要求同时包含品牌规范、历史案例和否决原因 | 查询拆解、多路／多跳、Agentic RAG，E06 | 比单次查询与多次检索的证据覆盖、调用数及停止行为 | 单查询工具可多次调用；自主规划效果未实测 |
| 问跨品牌创意主题与关联，而非某段事实 | GraphRAG、关系与社区摘要，E06 | 与普通检索对比覆盖、关系错误和建库成本；精确计数仍用结构化数据 | 仅专题入口，没有建图，先证明确有问题 |
| 改完检索后答案仍错 | 分层评测、正确证据对照、引用校验，E05/E08/K02 | 先查资料／检索／上下文，再查生成；开发与保留测试分开 | 引文与开发案例已有，真实模型评测待授权 |

每次只选一个问题，保留旧策略、只改主要因素，再解释收益与代价。新技术不以“接上了”作为学习结论，也不为了消灭教学失败把预期答案传给检索端。

### 文件与验证入口

- [RAG实现及新手注释](<rag.py>)：从`snapshot`理解版本与范围，从`search`理解BM25和父段返回；不要一次通读所有函数。
- [开发案例](<knowledge/cases.json>)：课堂修改查询，先观察实际返回，不先改预期答案。
- [离线测试](<../tests/test_creative_agent_rag.py>)：覆盖来源区间、例外覆盖、版本、商品范围、预算、幂等、恢复、评审隔离及假引用。
- 官方机制参考：[召回与重排](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)、[Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval)。本轮未安装这些框架或模型；父段回取与为片段补背景是不同机制。

这份基线的目的，是把面试题变成可观察的项目问题。具体下次课与本人掌握证据只在学习进度（本分享版未收录）维护。

## 下一课看什么

打开`runtime.py`的`run()`：观察`pending_action`为什么在执行前保存，模型请求失败为什么保留它，工具完成为什么清空它。再看`_execute()`，区分模型决定“选哪个工具”与程序决定“这次请求能不能执行”。动态学习断点仍以学习进度（本分享版未收录）为准。
