# 从“没见过这个组合”到神经语言模型

## 先把问题说精确

假设训练集见过字符“机”和“器”，却从未把“机器”作为相邻上下文见过。现在要预测：

```text
机器 → 下一个字符是什么？
```

纯 Trigram 保存的是一张精确查找表：

```text
P(人 | 机器) = count(机器→人) / count(机器→任意字符)
```

如果 `count(机器→任意字符) = 0`，整行参数都不存在。这不是代码报错，而是统计模型的数据稀疏问题。

这个问题不是突然被某个算法一次解决的。学术界和工程界大致经历了以下几层推演。

## 第一层：先把语言写成概率问题

1948 年，Claude Shannon 用逐级逼近的随机过程说明：比起完全随机地产生字母，利用一阶、二阶等上下文统计会越来越像语言。这个阶段确立了核心任务：根据已有序列估计下一个符号的概率。

项目对应：`ngram_language_model.py` 中的 Unigram、Bigram 和 Trigram。

原始资料：[Shannon, A Mathematical Theory of Communication](https://www.mpi.nl/publications/item2383162/mathematical-theory-communication)

## 第二层：发现最大似然估计会把未见事件判为零

N-gram 的最大似然估计只相信计数。见过 10 次就使用 10 次，没见过就是 0。上下文越长，可能的组合数增长越快，而训练语料不可能覆盖所有组合。这通常被称为数据稀疏；更广义的高维组合爆炸也常被称为维数灾难。

工程问题是严重的：整句话的概率通常由多个条件概率相乘，只要其中一项为 0，整句话概率就变成 0。

## 第三层：平滑、折扣、回退和插值

统计语言模型首先没有抛弃计数，而是修正计数。

- 平滑：不要把所有概率质量都分给已见事件，给未见事件留一点空间。
- 折扣：把已见事件的计数略微减小，腾出概率质量。
- 回退：高阶上下文没见过时，退回较短上下文。例如 `机器` 没见过，就改看最后一个字符 `器`。
- 插值：不只选一层，而是把 Trigram、Bigram、Unigram 的概率加权混合。

Katz 回退把“高阶有可靠计数就使用，否则回到低阶”系统化。Kneser–Ney 则进一步改进低阶概率：一个词是否适合作为续接，不只看它出现得多不多，还看它能跟多少种不同上下文搭配。Chen 与 Goodman 的大规模实验系统比较了多种平滑方法，并展示了改进 Kneser–Ney 的强表现。

项目对应：

- `rank_with_backoff`：硬回退；
- `rank_with_interpolation`：根据上下文计数自适应插值；
- `evaluate_models.py`：只能用验证集选参数，不能偷看测试集。

原始资料：

- [Katz, Estimation of Probabilities from Sparse Data for the Language Model Component of a Speech Recognizer](https://www.cis.upenn.edu/~danroth/Teaching/CS598-05/Papers/Katz87.pdf)
- [Kneser & Ney, Improved backing-off for m-gram language modeling](https://cir.nii.ac.jp/crid/1360855571158097152)
- [Chen & Goodman, An Empirical Study of Smoothing Techniques for Language Modeling](https://aclanthology.org/P96-1041/)

局限：这一步解决了“没有概率”的问题，却主要通过舍弃部分上下文来完成。它没有真正理解“机”和“器”组合后可能表达什么。

## 第四层：让符号共享连续参数

查表模型给每个精确组合单独存一格参数。神经模型换了一种表示方式：每个字符先获得一个短向量，也就是 Embedding。

```text
“机” → [ 0.12, -0.31, ...]
“器” → [-0.08,  0.44, ...]
```

这两个向量经过同一个神经网络：

```text
字符 ID → Embedding → 拼接 → 隐藏层 → Logits → Softmax 概率
```

关键变化不是“向量看起来高级”，而是参数共享：

- “机”的向量会从所有包含“机”的训练题中更新；
- “器”的向量会从所有包含“器”的训练题中更新；
- 同一个隐藏层和输出层会从所有上下文中更新；
- 所以即使“机器”这对输入从未共同出现，矩阵运算仍然有完整定义。

2003 年 Bengio 等人的神经概率语言模型直接针对维数灾难提出这条路线：同时学习分布式词表示和条件概率函数，让相似词组成的未见序列也能获得合理概率。

项目对应：`neural_language_model.py`。它没有调用深度学习框架，而是用 NumPy 展开 Embedding、Softmax、交叉熵、反向传播和梯度下降，便于逐行学习。

原始资料：[Bengio et al., A Neural Probabilistic Language Model](https://jmlr.csail.mit.edu/papers/v3/bengio03a.html)

## 第五层：Word2Vec 把“学向量”做得高效，但它不是完整语言模型

2013 年的 Word2Vec（CBOW、Skip-gram）用很高效的浅层目标学习词向量，使语义或句法相近的词在向量空间中靠近。

它与当前任务的关系是：

- 共同点：都用上下文训练连续向量，靠共享表示改善泛化；
- 不同点：Word2Vec 的主要产物是“词向量”，通常不直接提供完整的 `P(下一个 token | 前文)`；
- 当前神经语言模型：向量和下一个字符预测器联合训练，最终产品就是概率分布。

因此 Word2Vec 可以给语言模型提供预训练向量，但单独拥有 Word2Vec 并不等于拥有一个生成式语言模型。

原始资料：[Mikolov et al., Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)

## 一个容易漏掉的工程层：优化器

“网络结构可以表达答案”不等于“训练程序能找到好参数”。最朴素梯度下降给所有参数使用同一步长，但高频字符和低频字符收到梯度的次数、大小都不相同。实践中会使用动量、自适应学习率、初始化、梯度裁剪等手段改善优化。

本项目故意保留 `sgd`，同时默认使用 Adam。第一次可以分别运行：

```bash
python3 train_neural_model.py --optimizer sgd --learning-rate 0.15 --epochs 8
python3 train_neural_model.py --optimizer adam --learning-rate 0.005 --epochs 15
```

比较验证损失和困惑度，就能看到“同一模型结构，仅训练方法不同”也会产生很大差异。

模型还会在每一轮后检查验证集，并在训练结束时恢复验证损失最低的那一轮。因为训练损失继续下降而验证损失上升意味着开始过拟合；如果根据测试集选择轮次，测试答案就泄漏进训练决策了。

原始资料：[Kingma & Ba, Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

## 第六层：固定窗口仍然太短，于是引入可变历史

前馈模型仍然只看固定的两个字符。Elman 早期循环网络通过隐藏状态把过去的信息逐步传下去，使模型能表示可变长度历史。后续 LSTM、GRU 又缓解了普通循环网络难以保留长距离信息的问题。

原始资料：[Elman, Finding Structure in Time](https://doi.org/10.1207/S15516709COG1402_1)

## 第七层：Transformer 用注意力直接选择相关历史

循环网络必须一步一步传递状态。2017 年 Transformer 使用自注意力，让当前位置直接计算它与上下文中各位置的关系，并且更适合并行训练。现代 GPT 类模型依然执行“预测下一个 token”，变化在于表示能力、上下文长度、参数规模和训练数据量。

原始资料：[Vaswani et al., Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 两条路线不是互相替代，而是层层补问题

| 阶段 | 未见组合时怎么办 | 保留两个字符的信息吗 | 能利用字符相似性吗 |
|---|---|---:|---:|
| 纯 Trigram | 无法预测 | 是 | 否 |
| 平滑/硬回退 | 给未见事件概率或退到 Bigram | 通常部分丢失 | 否 |
| 插值 | 混合高阶与低阶 | 部分保留 | 否 |
| Embedding + 神经网络 | 共享字符向量和网络参数后直接计算 | 是 | 可以学习 |
| RNN / Transformer | 再扩大可利用的上下文 | 是 | 可以学习 |

## 工程上应当如何验证

不要只挑一句“机器”看输出。专项测试应当自动筛出所有满足以下条件的位置：组成上下文的字符见过、答案字符见过、完整上下文没见过。然后比较各模型的 Top-1、Top-3 和 Top-5。

本项目执行：

```bash
python3 train_neural_model.py
python3 evaluate_unseen_combinations.py
python3 predict_neural.py 种基
```

评测报告会写入 `reports/unseen_combination_evaluation.md`。如果神经模型可以输出、但准确率仍低，说明“组合泛化的机制已经建立”，而数据量、词表、模型容量或优化过程仍需要改进。机制和效果必须分开判断。
