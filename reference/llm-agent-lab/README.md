# 教学参考实现

这里保存知识库引用的教学代码、虚构输入、离线评测样例与历史报告。代码默认使用离线替身，不代表真实模型表现。

在本目录中使用 Python 3.9 或更高版本运行（macOS / Linux）：

```bash
python3 -S -B -m creative_agent demo
python3 -S -B -m creative_agent demo --rag
python3 -S -B -m creative_agent.rag_eval
```

详细说明见 [Agent README](creative_agent/README.md)，图示见[项目架构](docs/advertising-ai-maps.md)。默认演示会在本地创建任务数据库；不要把含个人输入的运行目录提交到 GitHub。

进程锁使用 `fcntl`，Windows 原生运行需要替换锁实现或使用兼容的 Linux 环境；这不影响 Windows 上用 Obsidian 阅读笔记。

模型基础的源码、评测报告保留用于阅读；原训练语料、模型权重和个人学习进度未打包。重现训练需要依据[数据说明](data/README.md)另行准备数据，不能把历史报告当作本次重新训练所得。

只有明确选择真实调用模式并配置自己的密钥后，模型适配器才会连接服务并可能产生费用。仅阅读知识库不需要配置模型服务。
