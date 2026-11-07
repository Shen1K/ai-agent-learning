# AI Agent 学习知识库

一份中文 AI 与 Agent 学习资料，围绕模型基础、Agent 工程、RAG、评测与面试表达，连接机制、架构、案例和参考实现。

[GitHub 仓库](https://github.com/Shen1K/ai-agent-learning) · [下载完整知识库 ZIP](https://github.com/Shen1K/ai-agent-learning/archive/refs/heads/main.zip)

**推荐下载整个仓库，用 Obsidian 打开阅读。** 知识页之间有双向链接和段落引用，两张 Canvas 图直接引用原笔记；从图谱出发，可以逐层进入概念、案例与参考答案。

## 用 Obsidian 阅读（推荐）

1. 安装 [Obsidian](https://obsidian.md/download)。
2. 在本仓库页面点击绿色 **Code → Download ZIP**，把下载的压缩包完整解压。
3. 打开 Obsidian，在仓库选择界面选择 **Open folder as vault / 打开本地仓库**。
4. 选择解压后的 `ai-agent-learning-main` 文件夹；这一层同时包含 `README.md`、`knowledge/` 和 `docs/`。**不要只选择 `knowledge/` 子文件夹**，否则以仓库根目录为起点的链接和 Canvas 引用会失效。
5. 打开 **[AI 与 Agent 知识首页](knowledge/00-index/home.md)**。也可以按 `Ctrl+O`（macOS 为 `⌘O`），输入 `home` 找到首页。

只需 Obsidian 自带功能，无需社区插件、Obsidian Sync 或付费发布服务。打开后若停在源码视图，可切换到阅读视图查看嵌入内容。

详细说明：[Obsidian 阅读指南](docs/obsidian-guide.md)。

## 从哪里开始

| 想看什么 | 阅读入口 |
|---|---|
| 整体知识路线 | [知识首页](knowledge/00-index/home.md) |
| 模型怎样工作 | [语言模型知识体系](<knowledge/AI Agent 学习/2026-10-01 语言模型知识体系/学习总结.md>) |
| Agent 的架构与模块 | [工程知识图谱](<knowledge/AI Agent 学习/广告创意Agent知识图谱/AK-00 知识图谱入口.md>) |
| 一次任务如何执行 | [杯子创意任务视角](<knowledge/AI Agent 学习/广告创意Agent知识图谱/AK-V02 杯子任务执行视角.md>) |
| 面试准备 | [144 题入口](<knowledge/AI Agent 学习/Agent 面试专题/2026-10-15 多源Agent面试题库/00 题库入口与面试理解.md>)、[高价值题精选](<knowledge/AI Agent 学习/Agent 面试专题/2026-10-15 多源Agent面试题库/08 高价值面试题精选与理由.md>) |
| 对照代码与离线示例 | [教学参考实现](reference/llm-agent-lab/README.md) |

工程图谱包含 10 个主题：编排与工作流、上下文与状态、工具与执行边界、RAG、创意生成与多模态、服务调度与可靠性、部署与推理优化、SFT 与 LoRA、评测数据与反馈、观测版本与发布。

## 怎么看图谱

- **关系图谱**：在 Obsidian 打开全局关系图，筛选框输入 `tag:#ai-main`，只看学习主线；清空筛选可以浏览更多关联页。
- **系统架构图**：打开 `AK-V01 系统架构.canvas`，查看各技术模块及其关系。
- **任务执行图**：打开 `AK-V02 杯子任务执行.canvas`，沿需求、生成、检查、改稿和交付阅读。
- **局部关联**：打开某一知识页后查看局部关系图或反向链接，连接概念、案例与面试题。

GitHub 可以直接阅读 Markdown 正文，但不会把 Obsidian 的 `[[双向链接]]`、`![[段落嵌入]]` 和 `.canvas` 显示为相同的交互体验，因此推荐使用 Obsidian。

## 内容与使用边界

这是持续整理的学习资料，部分内容由 AI 辅助归纳。面试回答属于参考表达，不代表作者或读者已完成相关项目；题目来源、改写和工程补充分开标注。来源频次只覆盖题库说明中列出的样本，不是全网面试概率。

广告创意案例使用虚构资料。代码区包含离线替身和真实接口适配器，默认阅读或离线演示不需要 API 密钥；离线流程通过不等于真实模型质量、生产性能或广告投放效果已验证。详见[材料范围与证据说明](docs/material-status.md)。

本分享版不包含个人学习进度、聊天记录、个人简历、训练语料和模型权重。引用资料的作者与链接保留；第三方内容的权利归原作者。这里没有为第三方材料授予额外许可。

快照整理日期：2026-11-07。技术实现与外部文档可能变化，实际使用时请核对当前版本。
