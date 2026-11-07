"""神经模型能力：只用 NumPy 实现 Embedding + MLP 字符语言模型。

这个模型对应 Bengio 等人 2003 年神经概率语言模型的核心思想：
先把离散字符转换成可以学习的连续向量，再使用神经网络预测下一个字符。

它不是现代大模型，但已经包含现代语言模型的基本训练部件：
Token ID、Embedding、Logits、Softmax、交叉熵、反向传播和梯度下降。

阅读 NumPy 代码时先记住四个字母：

    B = batch size，一批有多少道题
    C = context length，每道题使用几个上下文字符
    D = embedding size，每个字符向量有多少个数字
    V = vocabulary size，模型一共能预测多少种字符

本项目默认 C=2、D=32、V=512。数组 shape 不是额外数据，而是描述数组
每个维度的长度。例如 (512, 2) 表示 512 道题，每题两个字符编号。
"""

import json
import math
from collections import Counter

import numpy as np

from data_pipeline import PROJECT_ROOT


NEURAL_MODEL_DIRECTORY = PROJECT_ROOT / "models" / "neural"
NEURAL_WEIGHTS_PATH = NEURAL_MODEL_DIRECTORY / "weights.npz"
NEURAL_METADATA_PATH = NEURAL_MODEL_DIRECTORY / "metadata.json"
UNKNOWN_TOKEN = "<UNK>"


def build_vocabulary(documents, maximum_size=512):
    """根据训练集建立固定词表。

    本课程仍然把一个字符当成一个 token。频率最高的字符获得独立编号，
    其他低频字符统一映射为 <UNK>。词表只能使用训练集创建，不能偷看
    验证集或测试集，否则会发生数据泄漏。

    返回三个值：
    - token_to_id：字符查编号，例如 {"中": 22}；
    - id_to_token：编号查字符，例如 id_to_token[22] == "中"；
    - character_counts：完整字符频次，便于写训练统计。
    """
    if maximum_size < 2:
        raise ValueError("词表至少需要容纳 <UNK> 和一个普通字符")

    character_counts = Counter()
    for document in documents:
        character_counts.update(document)

    # most_common(n) 返回 [(字符, 次数), ...]；这里只保留字符，丢弃次数。
    common_characters = [
        character
        for character, _ in character_counts.most_common(maximum_size - 1)
    ]
    # 固定让 <UNK> 占编号0。未知字符就映射到0，不会让数组索引越界。
    id_to_token = [UNKNOWN_TOKEN] + common_characters
    token_to_id = {
        token: token_id for token_id, token in enumerate(id_to_token)
    }
    return token_to_id, id_to_token, character_counts


def encode_text(text, token_to_id):
    """把字符序列转换成整数序列。

    dict.get(character, unknown_id) 表示：有这个键就返回编号，没有就返回
    unknown_id。比如“中国”可能被转换为 [22, 17]。
    """
    unknown_id = token_to_id[UNKNOWN_TOKEN]
    return [token_to_id.get(character, unknown_id) for character in text]


def build_training_examples(
    documents,
    token_to_id,
    context_length=2,
    maximum_examples=None,
):
    """把文章转换成监督学习题目 X 和正确答案 y。

    例如文章包含“中国人”，上下文长度为2时会产生：
        X = [“中”的编号, “国”的编号]
        y = “人”的编号

    正确答案不在词表中的题目会被过滤。否则模型可能通过频繁预测
    <UNK> 获得看似不错、实际没有语言意义的成绩。

    返回两个 NumPy 数组：
    - x.shape == (题目数, context_length)，保存输入字符 ID；
    - y.shape == (题目数,)，保存每道题唯一的正确答案 ID。
    """
    contexts = []
    targets = []
    unknown_id = token_to_id[UNKNOWN_TOKEN]

    for document in documents:
        encoded = encode_text(document, token_to_id)
        for index in range(context_length, len(encoded)):
            target = encoded[index]
            if target == unknown_id:
                continue
            contexts.append(encoded[index - context_length : index])
            targets.append(target)

    # 神经网络使用整数 ID 访问 Embedding 行，所以 dtype 必须是整数。
    x = np.asarray(contexts, dtype=np.int64)
    y = np.asarray(targets, dtype=np.int64)

    # 固定取均匀分布在完整语料中的样本，而不是只取文件开头。
    if maximum_examples and len(x) > maximum_examples:
        selected = np.linspace(
            0,
            len(x) - 1,
            maximum_examples,
            dtype=np.int64,
        )
        x = x[selected]
        y = y[selected]
    return x, y


def softmax(logits):
    """把任意实数分数转换成每行总和为1的概率。

    先减去每行最大值是数值稳定技巧。它不会改变 Softmax 结果，
    却能避免 exp(很大的数) 产生浮点数溢出。

    输入和输出 shape 都是 (B, V)。输入是任意分数，输出每行都在 0~1
    之间且总和为1，因此可以解释成 V 个候选字符的概率。
    """
    # axis=1 表示沿每一行寻找最大值；keepdims=True 保留 (B, 1) 形状，
    # 这样 NumPy 能把每行最大值广播减到该行的 V 个元素上。
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


class FeedForwardLanguageModel:
    """观察固定长度上下文的前馈神经语言模型。

    类把参数矩阵和使用这些参数的方法放在一起。实例 model 训练前后仍是
    同一个对象，只是 model.embedding、model.w1 等数组中的数字改变了。
    """

    def __init__(
        self,
        vocabulary_size,
        context_length=2,
        embedding_size=12,
        hidden_size=24,
        seed=7,
    ):
        self.vocabulary_size = vocabulary_size
        self.context_length = context_length
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        random_generator = np.random.default_rng(seed)
        input_size = context_length * embedding_size

        # Embedding shape = (V, D)：每一行对应一个字符，每一列是一项
        # 可学习特征。它不是人工填写的汉字含义，而是由训练自动调整。
        self.embedding = random_generator.normal(
            0,
            0.05,
            size=(vocabulary_size, embedding_size),
        ).astype(np.float32)

        # w1 shape = (C*D, H)，负责把拼接后的上下文映射到隐藏层。
        # Xavier 风格初始化让前向信号不会在一开始过大或过小。
        self.w1 = random_generator.normal(
            0,
            math.sqrt(2 / (input_size + hidden_size)),
            size=(input_size, hidden_size),
        ).astype(np.float32)
        # b1 shape = (H,)。一维数组会自动广播到批次中的每道题。
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        # w2 shape = (H, V)，把隐藏特征转换成每个词表字符的分数。
        self.w2 = random_generator.normal(
            0,
            math.sqrt(2 / (hidden_size + vocabulary_size)),
            size=(hidden_size, vocabulary_size),
        ).astype(np.float32)
        # b2 shape = (V,)，每个可预测字符都有一个偏置参数。
        self.b2 = np.zeros(vocabulary_size, dtype=np.float32)

    def forward(self, x):
        """前向传播：Token ID → Embedding → 隐藏层 → 概率。

        假设批量大小 B=256、上下文 C=2、向量维度 D=12：
        embedded 的形状是 (B, C, D)，展平后是 (B, C×D)。
        """
        # x shape: (B, C)。NumPy 高级索引用每个 ID 取 Embedding 对应行。
        # embedded shape: (B, C, D)。
        embedded = self.embedding[x]
        # reshape 把每题 C 个向量连接起来：(B, C, D) → (B, C*D)。
        flattened = embedded.reshape(len(x), -1)
        # @ 是矩阵乘法，不是逐元素乘法：(B,C*D)@(C*D,H) → (B,H)。
        hidden_linear = flattened @ self.w1 + self.b1
        # tanh 加入非线性；没有它，多层线性变换仍等价于单层线性变换。
        hidden = np.tanh(hidden_linear)
        # (B,H)@(H,V) → (B,V)，为每道题产生 V 个 logits。
        logits = hidden @ self.w2 + self.b2
        probabilities = softmax(logits)
        # cache 保存反向传播会再次用到的中间量，避免重新计算。
        cache = {
            "x": x,
            "flattened": flattened,
            "hidden": hidden,
            "probabilities": probabilities,
        }
        return probabilities, cache

    def loss_and_gradients(self, x, y):
        """计算交叉熵损失，并手写反向传播。

        交叉熵只关心模型给正确答案分配了多少概率：
            loss = -log(P(正确字符))

        正确字符概率越接近1，损失越接近0。

        返回 (loss, gradients)：loss 是普通 float；gradients 是字典，
        每个梯度数组都与它对应的参数数组 shape 完全相同。
        """
        probabilities, cache = self.forward(x)
        # row_indices=[0,1,...,B-1]，配合 y 可从每行取出正确答案那一列。
        row_indices = np.arange(len(x))
        correct_probabilities = probabilities[row_indices, y]
        loss = -np.log(correct_probabilities + 1e-12).mean()

        # Softmax + 交叉熵组合后的梯度可以化简为 p - one_hot(y)。
        d_logits = probabilities.copy()
        d_logits[row_indices, y] -= 1
        d_logits /= len(x)

        # `.T` 转置二维矩阵：(B,H) → (H,B)。
        # 所以 d_w2: (H,B)@(B,V) → (H,V)，与 w2 形状一致。
        d_w2 = cache["hidden"].T @ d_logits
        d_b2 = d_logits.sum(axis=0)

        d_hidden = d_logits @ self.w2.T
        # tanh(x) 的导数是 1 - tanh(x)^2。
        d_hidden_linear = d_hidden * (1 - cache["hidden"] ** 2)
        d_w1 = cache["flattened"].T @ d_hidden_linear
        d_b1 = d_hidden_linear.sum(axis=0)

        # 梯度按照前向传播的反方向逐层传回 Embedding。
        d_flattened = d_hidden_linear @ self.w1.T
        d_embedded = d_flattened.reshape(
            len(x),
            self.context_length,
            self.embedding_size,
        )
        # zeros_like 创建形状和 dtype 都与 Embedding 一样的全零数组。
        d_embedding = np.zeros_like(self.embedding)

        # 同一个字符可能在批次中出现多次，np.add.at 会累加所有梯度。
        for position in range(self.context_length):
            np.add.at(
                d_embedding,
                x[:, position],
                d_embedded[:, position, :],
            )

        gradients = {
            "embedding": d_embedding,
            "w1": d_w1,
            "b1": d_b1,
            "w2": d_w2,
            "b2": d_b2,
        }
        return float(loss), gradients

    def apply_gradients(self, gradients, learning_rate, maximum_norm=5.0):
        """使用最朴素的梯度下降更新参数，并进行全局梯度裁剪。

        这个方法保留给教学和单元测试使用。真实语料中，不同参数收到的
        梯度尺度差异很大，训练入口默认会改用下面的 AdamOptimizer。
        """
        # 把所有参数梯度看成一个长向量，计算整体长度（L2 norm）。
        squared_norm = sum(
            float(np.sum(gradient * gradient))
            for gradient in gradients.values()
        )
        gradient_norm = math.sqrt(squared_norm)
        scale = min(1.0, maximum_norm / (gradient_norm + 1e-12))

        self.embedding -= learning_rate * gradients["embedding"] * scale
        self.w1 -= learning_rate * gradients["w1"] * scale
        self.b1 -= learning_rate * gradients["b1"] * scale
        self.w2 -= learning_rate * gradients["w2"] * scale
        self.b2 -= learning_rate * gradients["b2"] * scale

    def predict_probabilities(self, x, batch_size=1024):
        """分批预测，避免一次把全部样本放入内存。

        x.shape=(题目数,C)，返回 shape=(题目数,V)。np.concatenate 会沿
        第0维把多个小批次重新接回原来的题目顺序。
        """
        batches = []
        for start in range(0, len(x), batch_size):
            probabilities, _ = self.forward(x[start : start + batch_size])
            batches.append(probabilities)
        if not batches:
            return np.empty((0, self.vocabulary_size), dtype=np.float32)
        return np.concatenate(batches)


class AdamOptimizer:
    """一个最小但完整的 Adam 优化器。

    普通梯度下降给所有参数使用同一个步长。Adam 为每个参数分别记录：

    - 一阶矩 m：梯度近期的平均方向；
    - 二阶矩 v：梯度平方的近期平均，用来估计尺度。

    更新时使用 m / sqrt(v)。经常收到大梯度的参数会自动缩小步长，
    梯度较小但方向稳定的参数可以走得更快。这对稀疏出现的字符向量
    尤其重要，也是神经网络工程从“公式正确”走向“能够训练”的一环。
    """

    def __init__(
        self,
        model,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.step = 0
        self.first_moments = {
            name: np.zeros_like(getattr(model, name))
            for name in ("embedding", "w1", "b1", "w2", "b2")
        }
        self.second_moments = {
            name: np.zeros_like(getattr(model, name))
            for name in ("embedding", "w1", "b1", "w2", "b2")
        }

    def apply(self, model, gradients, learning_rate, maximum_norm=5.0):
        """根据当前批次梯度更新全部参数。"""
        squared_norm = sum(
            float(np.sum(gradient * gradient))
            for gradient in gradients.values()
        )
        gradient_norm = math.sqrt(squared_norm)
        clip_scale = min(
            1.0,
            maximum_norm / (gradient_norm + 1e-12),
        )
        self.step += 1

        # items() 每轮给出参数名和对应梯度，例如 ("w1", d_w1)。
        for name, raw_gradient in gradients.items():
            gradient = raw_gradient * clip_scale
            first = self.first_moments[name]
            second = self.second_moments[name]

            first *= self.beta1
            first += (1 - self.beta1) * gradient
            second *= self.beta2
            second += (1 - self.beta2) * (gradient * gradient)

            # 刚开始时移动平均会天然偏小，下面两项做偏差修正。
            corrected_first = first / (1 - self.beta1 ** self.step)
            corrected_second = second / (1 - self.beta2 ** self.step)
            # getattr(model, "w1") 等价于 model.w1，让五组参数共用更新代码。
            parameter = getattr(model, name)
            parameter -= (
                learning_rate
                * corrected_first
                / (np.sqrt(corrected_second) + self.epsilon)
            )


def calculate_loss(model, x, y, batch_size=1024):
    """不更新参数，只计算数据集的平均交叉熵。

    训练集和验证集都能调用它；因为函数不执行参数更新，所以验证集信息
    不会通过梯度进入模型。
    """
    total_loss = 0.0
    total_examples = 0

    for start in range(0, len(x), batch_size):
        batch_x = x[start : start + batch_size]
        batch_y = y[start : start + batch_size]
        probabilities, _ = model.forward(batch_x)
        correct = probabilities[np.arange(len(batch_x)), batch_y]
        total_loss += float(-np.log(correct + 1e-12).sum())
        total_examples += len(batch_x)
    return total_loss / total_examples if total_examples else 0


def train_model(
    model,
    train_x,
    train_y,
    validation_x,
    validation_y,
    epochs=12,
    batch_size=512,
    learning_rate=0.005,
    seed=7,
    optimizer_name="adam",
):
    """使用小批量训练，记录损失，最后恢复验证集最好的参数。

    optimizer_name="sgd" 使用最朴素梯度下降；默认 "adam" 更适合真实
    稀疏语料。保留两个选项可以让学习者亲自复现优化器带来的差异。
    """
    if optimizer_name not in {"sgd", "adam"}:
        raise ValueError("optimizer_name 只能是 'sgd' 或 'adam'")

    # 独立随机生成器只负责打乱题目；固定 seed 让实验顺序可复现。
    random_generator = np.random.default_rng(seed)
    history = []
    adam = AdamOptimizer(model) if optimizer_name == "adam" else None
    best_validation_loss = float("inf")
    best_parameters = None
    parameter_names = ("embedding", "w1", "b1", "w2", "b2")

    for epoch in range(1, epochs + 1):
        # order 是打乱后的行号，例如 [2,0,1]；x 和 y 必须使用同一个
        # order，才能保证题目和正确答案仍然一一对应。
        order = random_generator.permutation(len(train_x))
        shuffled_x = train_x[order]
        shuffled_y = train_y[order]
        loss_sum = 0.0

        # 后期稍微降低学习率，让参数在较好位置附近稳定下来。
        decay = 0.95 if optimizer_name == "adam" else 0.85
        current_learning_rate = learning_rate * (decay ** (epoch - 1))
        # 最后一批不足 batch_size 也没关系，Python 切片会安全地停在末尾。
        for start in range(0, len(shuffled_x), batch_size):
            batch_x = shuffled_x[start : start + batch_size]
            batch_y = shuffled_y[start : start + batch_size]
            loss, gradients = model.loss_and_gradients(batch_x, batch_y)
            if adam:
                adam.apply(model, gradients, current_learning_rate)
            else:
                model.apply_gradients(gradients, current_learning_rate)
            loss_sum += loss * len(batch_x)

        train_loss = loss_sum / len(train_x)
        validation_loss = calculate_loss(
            model,
            validation_x,
            validation_y,
            batch_size,
        )
        epoch_result = {
            "epoch": epoch,
            "learning_rate": current_learning_rate,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_perplexity": math.exp(min(validation_loss, 20)),
            "is_best_so_far": validation_loss < best_validation_loss,
        }
        history.append(epoch_result)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            # copy 很重要：如果只保存引用，下一轮更新会连“最佳参数”一起改掉。
            best_parameters = {
                name: getattr(model, name).copy()
                for name in parameter_names
            }
        print(
            f"第 {epoch}/{epochs} 轮："
            f"训练损失 {train_loss:.4f}，"
            f"验证损失 {validation_loss:.4f}，"
            f"验证困惑度 {epoch_result['validation_perplexity']:.2f}",
            flush=True,
        )

    # 最后一轮不一定最好。模型选择只能看验证集，绝不能看测试集。
    if best_parameters:
        for name, value in best_parameters.items():
            setattr(model, name, value)
        best_epoch = min(history, key=lambda item: item["validation_loss"])
        print(
            f"已恢复验证集最优的第 {best_epoch['epoch']} 轮参数："
            f"验证损失 {best_epoch['validation_loss']:.4f}",
            flush=True,
        )
    return history


def save_neural_model(model, id_to_token, metadata, training_history):
    """权重使用二进制 NPZ，说明信息使用可读 JSON。

    大矩阵存 JSON 会很大而且读写慢，因此 NumPy 参数放 weights.npz；
    结构、词表和训练历史需要人工检查，放 metadata.json。两者共同组成模型。
    """
    NEURAL_MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        NEURAL_WEIGHTS_PATH,
        embedding=model.embedding,
        w1=model.w1,
        b1=model.b1,
        w2=model.w2,
        b2=model.b2,
    )
    model_metadata = {
        "format_version": 1,
        "model_type": "character_feed_forward_neural_language_model",
        "architecture": "Embedding -> Concatenate -> Tanh -> Linear -> Softmax",
        "vocabulary": id_to_token,
        "vocabulary_size": model.vocabulary_size,
        "context_length": model.context_length,
        "embedding_size": model.embedding_size,
        "hidden_size": model.hidden_size,
        "weights_file": NEURAL_WEIGHTS_PATH.name,
        "training": metadata,
        "training_history": training_history,
    }
    NEURAL_METADATA_PATH.write_text(
        json.dumps(model_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_neural_model():
    """加载元数据和二进制权重，恢复可推理模型。

    先按元数据创建同结构的空模型，再用文件中的数组替换随机初始化参数。
    返回 metadata、model、token_to_id、id_to_token 四项供不同调用方使用。
    """
    metadata = json.loads(NEURAL_METADATA_PATH.read_text(encoding="utf-8"))
    model = FeedForwardLanguageModel(
        vocabulary_size=metadata["vocabulary_size"],
        context_length=metadata["context_length"],
        embedding_size=metadata["embedding_size"],
        hidden_size=metadata["hidden_size"],
    )
    weights = np.load(NEURAL_WEIGHTS_PATH)
    model.embedding = weights["embedding"]
    model.w1 = weights["w1"]
    model.b1 = weights["b1"]
    model.w2 = weights["w2"]
    model.b2 = weights["b2"]

    id_to_token = metadata["vocabulary"]
    token_to_id = {
        token: token_id for token_id, token in enumerate(id_to_token)
    }
    return metadata, model, token_to_id, id_to_token
