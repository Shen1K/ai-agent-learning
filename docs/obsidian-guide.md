# Obsidian 阅读指南

## 首次打开

从 GitHub 的 Code → Download ZIP 下载并完整解压，然后在 Obsidian 的仓库选择界面选择“打开本地仓库 / Open folder as vault”，选中最外层的 `ai-agent-learning-main` 文件夹。如果通过 Git 克隆，选中 `ai-agent-learning` 文件夹。

选择的文件夹必须直接包含 `README.md`、`knowledge/` 和 `docs/`。Obsidian 中的 vault（仓库）就是本地文件夹，不需要导入数据库。

打开 [[knowledge/00-index/home|AI 与 Agent 知识首页]]。通过顶部路径或快速切换器（Windows/Linux：Ctrl+O；macOS：⌘O）输入 `home` 即可定位。

官方说明：[管理仓库](https://help.obsidian.md/manage-vaults)、[关系图谱](https://help.obsidian.md/plugins/graph)、[Canvas](https://help.obsidian.md/plugins/canvas)。

## 建议阅读顺序

1. 在首页先看模型基础，再进入 Agent 工程地图。
2. 打开系统架构 Canvas，看各模块输入输出与依赖。
3. 选择一个知识页，阅读机制、适用条件、反例和验证办法。
4. 沿杯子任务案例观察机制怎样用于任务。
5. 在面试题库中用同一机制组织回答，回到来源页核查依据。

## 保持关系图清楚

全局关系图筛选使用 `tag:#ai-main`。默认配置只提供这个展示筛选，不携带原作者的工作区布局、账户或社区插件设置。筛选若未自动生效，可手动输入。

两张 `.canvas` 是不同视角：系统图解释模块关系，任务图解释一次执行过程。它们引用原笔记与段落，正文修改后无需另维护一套相同内容。

## 常见问题

- **双向链接点不开**：确认打开的是完整解压后的仓库根目录；不要单独搬走某篇笔记或重命名文件夹。
- **嵌入内容只显示符号**：切换阅读视图；同时确认目标文件已完整下载。
- **图谱内容太多**：输入 `tag:#ai-main`，或只看当前笔记的局部关系图。
- **GitHub 上 Canvas 看不懂**：下载后由 Obsidian 打开，不需要安装额外插件。
- **更新时怕覆盖自己的笔记**：把新版本下载到新的文件夹；确认差异后再合并。用 Git 更新时先保存自己的修改。

通过微信分享时，发送本 GitHub 仓库链接即可；接收者按 README 下载，再用自己的 Obsidian 打开。无需分享个人 iCloud 文件夹或同步账户。
