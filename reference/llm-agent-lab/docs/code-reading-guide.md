# 代码阅读指南：给会编程、但不熟 Python 和算法的学习者

本页用于查阅具体代码和语法。教学顺序与验收见 6 周核心方案（本分享版未收录），
当前起点见 学习进度（本分享版未收录）。每课只选完成当前因果解释所需的代码。
下方阅读顺序是代码索引，不是第一阶段全部必读；其他主题的位置和用途见 [全景导航](<ai-landscape.md>)。

## 不要从最长的文件第一行硬读

广告创意Agent主线先看[服务说明](<../creative_agent/README.md>)，再依次看`creative_agent/runtime.py`的`run()`、`_execute()`，以及`contracts.py`的工具Schema、`prompts.py`的上下文隔离。模型选工具与程序验证工具请求是不同职责；离线替身仅用于控制流测试。下列N-gram／神经模型入口继续保留作为模型基础查阅。

推荐顺序：

1. `main.py`：先看整个 N-gram 流程如何串起来；
2. `data_pipeline.py`：理解训练集、验证集和文章边界；
3. `ngram_language_model.py`：理解计数、概率、回退和插值；
4. `evaluate_models.py`：理解覆盖率、Top-k 和公平评测；
5. `train_neural_model.py`：先看神经训练的大步骤；
6. `neural_language_model.py`：最后进入矩阵、损失和反向传播；
7. `evaluate_unseen_combinations.py`：看新模型究竟有没有解决指定问题；
8. `tests/`：用可纸笔验证的小例题复习所有关键行为。

## 项目反复出现的 Python 写法

### 字典与 Counter

```python
model["中国"]["人"] += 1
```

可以把它理解为二维映射：先找到上下文“中国”，再把答案“人”的次数加一。`Counter` 本质上是专门计数的字典。

### 字符串切片

```python
context = document[index - 2:index]
actual = document[index]
```

切片包含左边界、不包含右边界。如果 `index` 指向“人”，切片就取它左边的两个字符“中国”。

### 推导式

```python
characters = [character for character in text if not character.isspace()]
```

它等价于创建空列表、执行 `for`、判断 `if`、再 `append`。项目保留推导式，因为真实 Python 工程中很常见，但旁边会注明数据含义。

### 元组解包

```python
metadata, model = load_model(path)
```

函数返回两个值，Python 按位置分别放进两个变量。`_` 通常表示“这个位置的值我不需要”。

### lambda

```python
lambda history: choose_backoff_next(history, models)
```

它是一个没有名字的短函数。这里用于把不同预测算法包装成相同调用方式，不是新的模型公式。

### `if __name__ == "__main__"`

直接执行文件时条件成立，开始运行 `main()`；其他文件导入它时条件不成立，只加载函数，不会意外开始下载或训练。

## NumPy 形状速查

神经模型使用以下符号：

| 符号 | 含义 | 当前默认值 |
|---|---|---:|
| B | 一批训练题的数量 | 512 |
| C | 每题上下文字符数 | 2 |
| D | 每个字符向量维度 | 32 |
| H | 隐藏层维度 | 96 |
| V | 词表大小 | 512 |

主要数组：

| 数组 | shape | 含义 |
|---|---|---|
| `x` | `(B, C)` | 一批上下文字符 ID |
| `embedding` | `(V, D)` | 每个字符对应一个可学习向量 |
| `embedded` | `(B, C, D)` | 根据 x 查出的这一批字符向量 |
| `flattened` | `(B, C*D)` | 把每题的两个向量连接起来 |
| `w1` | `(C*D, H)` | 输入到隐藏层的参数 |
| `hidden` | `(B, H)` | 每道题提取出的隐藏特征 |
| `w2` | `(H, V)` | 隐藏层到候选字符的参数 |
| `probabilities` | `(B, V)` | 每题对全部字符的预测概率 |

看到矩阵乘法时，只检查中间维度能否相消：

```text
(B, C*D) @ (C*D, H) = (B, H)
(B, H)   @ (H, V)   = (B, V)
```

## 怎样判断自己真的理解了

每读完一个函数，回答四个问题：

1. 输入数据的类型和一个具体例子是什么？
2. 返回值的类型和一个具体例子是什么？
3. 它修改了文件或模型参数吗？
4. 如果训练数据中不存在某个组合，它会返回空结果、进行回退，还是依然计算概率？

暂时答不上来时，先运行 `tests/` 中的小例子，不需要直接调试完整的 1,000 篇语料。
