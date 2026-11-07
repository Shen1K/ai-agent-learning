---
id: AK-S01
type: source
schema_version: "1.0"
status: hypothesis
title: "来源与证据登记"
updated: "2026-11-05"
tags:
  - ai-agent-map
---

# 来源与证据登记

本试点沉淀广告创意Agent会话的工程知识，不迁移整份聊天，也不把自动整理当作独立验证。知识页的证据由本页统一登记，新增案例保留自己的具体上下文。

## 本地项目材料

下列实现与虚构样例以相对路径链接到公开版 reference/ 目录中的快照；无需原作者的电脑路径。个人学习进度与教学方案没有收录。

- 学习进度（本分享版未收录）：个人学习证据与下一步的唯一入口，不在此复制动态完成状态。
- 现行教学方案（本分享版未收录）：课程范围和深度。
- [架构地图](<../../../reference/llm-agent-lab/docs/advertising-ai-maps.md>)：项目能力与候选架构。
- [Agent实现说明](<../../../reference/llm-agent-lab/creative_agent/README.md>)：运行时、工具、状态、模拟模式与真实适配器的边界。
- [杯子任务](<../../../reference/llm-agent-lab/creative_agent/example_task.json>)：brief、商品事实和约束。
- [RAG教学资料](<../../../reference/llm-agent-lab/creative_agent/knowledge/documents.json>)：虚构品牌规范、案例和干扰资料。
- [评测原始记录](<../../../reference/llm-agent-lab/lessons/creative_studio/evaluation_dataset_demo/raw_records.json>)：六条助手编写的模拟记录。
- [实际硬规则报告](<../../../reference/llm-agent-lab/lessons/creative_studio/evaluation_dataset_demo/example_output/offline_hard_check_report.json>)：固定候选的程序检查结果，不是模型准确率。

## 证据强度与适用范围

| 证据 | 能支持什么 | 不能支持什么 |
|---|---|---|
| 会话讨论与教学示范 | 本轮形成的问题、概念区分和设计动机 | 生产效果、学习者独立实现 |
| 本地代码与离线运行 | 已实现的程序机制、特定样例路径 | 真实模型决策或语义质量 |
| 官方文档 | 组件的定位和机制 | 本项目已部署、性能提升数值 |
| 虚构候选与教学标签 | 演示规则、校准流程 | 真实客户偏好、投放收益 |

项目运行时当前的必需词检查覆盖标题加正文；2026-11-05的独立数据教学示例则明确要求标题含词，计数口径也不同。案例卡描述的是独立教学示例，不能冒充主运行时已采用该规则。需要合并时，应另做规则统一与回归验证。

## 官方资料

以下是本会话核对过的定位材料。外部文档会更新，实际实现前应再次锁定版本与兼容性。

- Agent状态持久化：[LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- 后台任务：[Celery介绍](https://docs.celeryq.dev/en/stable/getting-started/introduction.html)
- 模型运行：[vLLM](https://docs.vllm.ai/en/stable/)
- 聊天输入格式：[Chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
- 推理缓存：[KV cache](https://huggingface.co/docs/transformers/en/cache_explanation)
- 前缀复用：[vLLM prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)
- 微调目标：[TRL SFT](https://huggingface.co/docs/trl/en/sft_trainer)
- 参数适配：[PEFT LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
- 偏好优化：[TRL DPO](https://huggingface.co/docs/trl/en/dpo_trainer)
- 链路关联：[OpenTelemetry traces](https://opentelemetry.io/docs/concepts/signals/traces/)
- 评测闭环：[Langfuse evaluation](https://langfuse.com/docs/evaluation/overview)
- 原生引用：[Obsidian embeds](https://obsidian.md/help/embeds)
- 图卡引用：[Obsidian Canvas](https://obsidian.md/help/plugins/canvas)

## 历史材料的定位

已有 AI Agent 学习总索引（本分享版未收录） 是另一时期的课程入口，保留其内容，不将其中“已完成”自动合并为本项目能力证据。本试点提供机制知识入口；个人实际掌握情况仍回到项目学习进度核对。

## 变更历史

- 2026-11-05：登记本轮来源，区分教学规则、当前实现与拟议扩展；不覆盖旧课程或面试题库。

