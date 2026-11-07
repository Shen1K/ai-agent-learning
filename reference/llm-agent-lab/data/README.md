# 中文训练集与测试集

这里分别保存训练语料和评测语料。两者严格分开，避免模型在训练时提前见到测试文章。

## 文件

- `test/chinese_test.txt`：适合直接读取的纯文本。
- `test/chinese_test.jsonl`：每行一篇文章，同时保留标题、来源、版本、授权和正文。
- `test/manifest.json`：数据规模和文件校验值，用来确认文件是否意外变化。
- `train/chinese_train.txt`：模型实际读取的训练纯文本。
- `train/chinese_train.jsonl`：训练文章及其标题、来源、版本和授权信息。
- `train/manifest.json`：训练集规模、选择条件和文件校验值。
- `processed/train.jsonl`：经过规范化并用于模型训练的文章。
- `processed/validation.jsonl`：用于选择插值参数的独立验证文章。
- `processed/manifest.json`：过滤条件、拆分方法和校验值。

## 来源与授权

正文取自中文维基百科。每条 JSONL 记录均包含原页面、精确修订版本、编辑历史和获取时间。

维基百科文本采用 [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/) 授权。使用或再发布这份数据时，应保留署名和来源链接，并遵守相同方式共享（ShareAlike）的要求。

数据经过以下处理：通过 MediaWiki TextExtracts API 提取文章导言的纯文本、请求 `zh-cn` 语言变体、规范化空白字符。数据内容不代表 Wikimedia Foundation 对本项目的认可。

## 重新获取

在项目目录运行：

```bash
python3 scripts/fetch_wikipedia_dataset.py
python3 scripts/fetch_wikipedia_training_dataset.py
python3 data_pipeline.py
```

维基百科内容会持续更新，因此重新获取后的文本和校验值可能不同。需要完全复现实验时，应保留当前数据快照，并使用 JSONL 中记录的 `revision_url`。

## 使用约定

`test/` 只用于评测，不放入训练数据。`train/` 默认由 1,000 篇随机百科文章导言组成，构建时会根据页面编号排除测试集的文章。后续课程将用两者比较不同模型的覆盖率、损失和预测能力。

原始训练语料还会按文章稳定拆分为训练集和验证集。当前规则是约 90% 用于训练、10% 用于验证；不会把同一篇文章拆到两边。文本采用 NFKC 规范化、英文小写化和空白合并，并过滤正文过短或中文比例过低的页面。
