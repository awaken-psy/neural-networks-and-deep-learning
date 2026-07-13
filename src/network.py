"""
network.py
~~~~~~~~~~

实现随机梯度下降学习算法的前馈神经网络模块。梯度通过反向传播计算。
注意，我主要关注代码的简洁性、可读性和易修改性。代码未经过优化，并且省略了许多理想的功能。

"""

#### 库
import random
import time
import numpy as np

import mnist_loader


class Network(object):

    # sizes 是一个列表（list），类似 C 的数组但可变长，如 [784, 30, 10]
    def __init__(self, sizes):
        """列表 ``sizes`` 包含网络各层的神经元数量。例如，如果列表
        为 [2, 3, 1]，则这是一个三层网络，第一层有 2 个神经元，
        第二层有 3 个神经元，第三层有 1 个神经元。网络的偏置和权重
        使用均值为 0、方差为 1 的高斯分布随机初始化。注意，第一层
        被假定为输入层，按惯例不为这些神经元设置偏置，因为偏置只
        在计算后续层的输出时使用。"""

        self.num_layers = len(sizes)   
        self.sizes = sizes          

        # np.random.randn(y, 1) 生成 y×1 的矩阵，元素服从标准正态分布 N(0,1)
        # 类似 C 中生成一个 y 行 1 列的二维数组，每个元素是随机的正态分布值
        # sizes[1:] 是 Python 的"切片"语法，表示从索引 1 开始取到末尾
        #   例如 sizes = [784, 30, 10] → sizes[1:] = [30, 10]
        #   即跳过第一层（输入层），因为输入层不需要偏置
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # 结果：biases[0] 是 30×1 矩阵（第一隐藏层的偏置）
        #       biases[1] 是 10×1 矩阵（输出层的偏置）

        # weights 是每层之间的权重矩阵列表
        # zip(a, b) 将两个列表"拉链式"配对，类似：
        #   zip([784,30], [30,10]) → [(784,30), (30,10)]
        # sizes[:-1] 表示从开头取到倒数第一个（不含），如 [784, 30]
        # sizes[1:]  表示从索引 1 取到末尾，如 [30, 10]
        # 所以 zip(sizes[:-1], sizes[1:]) → [(784,30), (30,10)]
        #   每对 (x, y) 表示：前一层有 x 个神经元，后一层有 y 个神经元
        # np.random.randn(y, x) 生成 y×x 的矩阵：
        #   weights[0] = 30×784 矩阵（输入层→隐藏层的权重）
        #   weights[1] = 10×30  矩阵（隐藏层→输出层的权重）
        # 权重矩阵的行数 = 后一层神经元数，列数 = 前一层神经元数
        # 这样 w·a （矩阵乘法）就能把前一层的激活值映射到后一层
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    # feedforward：前向传播，给定输入 a，计算网络输出
    def feedforward(self, a):
        """如果输入为 ``a``，返回网络的输出。"""

        # zip(self.biases, self.weights) 把每层的偏置和权重配对
        for b, w in zip(self.biases, self.weights):
            # np.dot(w, a) 是矩阵乘法，+b 是矩阵加法
            # sigmoid(...) 对结果逐元素应用激活函数
            # 然后用新的 a 覆盖旧值，传入下一层继续计算
            a = sigmoid(np.dot(w, a)+b)
        return a
        # 最终 a 是最后一层的输出，如 10×1 向量，表示对每个数字的预测概率

    # SGD：随机梯度下降，训练网络的主函数
    # 参数：
    #   training_data     - 训练数据，列表 of (x, y) 元组
    #   epochs            - 训练轮数（遍历全部训练数据的次数）
    #   mini_batch_size   - 每个小批量的样本数
    #   eta               - 学习率（learning rate）
    #   test_data=None    - 可选的测试数据，=None 表示参数有默认值
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """使用小批量随机梯度下降训练神经网络。``training_data`` 是
        一个包含元组 ``(x, y)`` 的列表，表示训练输入和期望输出。
        其他非可选参数的含义不言自明。如果提供了 ``test_data``，
        则在每个 epoch 后用测试数据评估网络，并打印部分进度。这有
        助于跟踪进度，但会显著降低速度。"""

        # 类似 C 的 if (test_data != NULL)
        if test_data: n_test = len(test_data)
        n = len(training_data)   # 训练样本总数

        for j in range(epochs):
            time1 = time.time()  # 记录开始时间，time.time() 返回当前时间戳（秒）
            random.shuffle(training_data)

            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

            # 遍历每个 mini-batch，用梯度下降更新权重和偏置
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            time2 = time.time()  # 记录结束时间

            # 如果有测试数据，每个 epoch 后评估并打印准确率
            if test_data:
                # {3:.2f} 表示保留 2 位小数
                print("Epoch {0}: {1} / {2}, 耗时 {3:.2f} 秒".format(j, self.evaluate(test_data), n_test, time2-time1))
            else:
                print("Epoch {0} 完成，耗时 {1:.2f} 秒".format(j, time2-time1))

    # update_mini_batch：用一个小批量数据更新权重和偏置
    def update_mini_batch(self, mini_batch, eta):
        """通过使用反向传播对一个迷你批次应用梯度下降来更新网络
        的权重和偏置。``mini_batch`` 是一个包含元组 ``(x, y)`` 的列表，
        ``eta`` 是学习率。"""

        # 初始化梯度累加器为全零矩阵
        # np.zeros(b.shape) 创建一个和 b 形状相同的全零矩阵
        nabla_b = [np.zeros(b.shape) for b in self.biases]   # 偏置梯度累加器
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # 权重梯度累加器

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            # 累加梯度
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # 用累加后的平均梯度更新权重和偏置
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    # backprop：反向传播算法，计算单个样本 (x, y) 的梯度
    # 这是整个神经网络最核心的函数
    def backprop(self, x, y):
        """返回一个元组 ``(nabla_b, nabla_w)``，表示代价函数 C_x 的梯度。``nabla_b`` 和 ``nabla_w`` 是逐层的 numpy 数组列表，
        类似于 ``self.biases`` 和 ``self.weights``。"""

        # 初始化梯度为全零
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # ========== 前向传播 ========
        activation = x         # 当前层的激活值，初始为输入 x
        activations = [x]      # 存储各层激活值的列表: [a0, a1, a2, ...]
        zs = []                # 存储各层 z 向量的列表: [z1, z2, ...]

        # 遍历每一层的权重和偏置，做前向计算
        for b, w in zip(self.biases, self.weights):
            # z = w·a + b：矩阵乘法 + 加偏置
            z = np.dot(w, activation)+b
            zs.append(z)                # 保存 z 供反向传播使用
            activation = sigmoid(z)
            activations.append(activation)  # 保存 a 供反向传播使用

        # ========== 反向传播 ==========
        # 目的：从最后一层开始，反向逐层计算梯度

        # 计算输出层的误差 delta = C'(a) * σ'(z)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        # nabla_b[-1] = delta：偏置的梯度 = 误差
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 从倒数第二层开始，往前逐层计算梯度
        # range(2, self.num_layers) 生成 [2, 3, ..., num_layers-1]
        # l=2 表示倒数第二层，l=3 表示倒数第三层，以此类推
        for l in range(2, self.num_layers):
            # zs[-l]：倒数第 l 层的 z 值
            z = zs[-l]

            # sigmoid 的导数 σ'(z)
            sp = sigmoid_prime(z)

            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            # 当前层的偏置梯度 = 误差
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        # 返回这个样本的梯度（元组，包含两个列表）
        return (nabla_b, nabla_w)

    # evaluate：在测试数据上评估网络准确率
    def evaluate(self, test_data):
        """返回神经网络输出正确结果的测试输入数量。注意，神经网络
        的输出被假定为最后一层中激活值最高的神经元的索引。"""

        # 列表推导式：对每个测试样本 (x, y) 计算预测结果
        # np.argmax(self.feedforward(x))：
        #   - feedforward(x) 返回 10×1 向量（各数字的概率）
        #   - np.argmax 返回最大值的索引（即网络认为最可能的数字 0-9）
        # y 是测试数据的真实标签（0-9 的整数）
        # 结果是一个列表，每个元素是 (预测值, 真实值) 元组
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]

        # 统计预测正确的数量
        return sum(int(x == y) for (x, y) in test_results)

    # cost_derivative：代价函数对输出激活值的导数
    # 这里使用二次代价函数 C = 0.5 * ||a - y||^2
    # 所以 dC/da = a - y
    def cost_derivative(self, output_activations, y):
        """返回偏导数 \partial C_x / \partial a 关于输出激活值的向量。"""

        # output_activations - y 是 NumPy 逐元素减法
        # output_activations 是 10×1 向量（网络输出）
        # y 是 10×1 向量（真实标签的 one-hot 编码）
        # 返回 10×1 向量（每个输出神经元的误差信号）
        return (output_activations-y)

#### 杂项函数

def sigmoid(z):
    """sigmoid 函数。"""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """sigmoid 函数的导数。"""
    return sigmoid(z)*(1-sigmoid(z))


if __name__ == "__main__":
    # 加载 MNIST 数据
    # training_data:   50000 个 (x, y) 元组，x 是 784×1 图像向量，y 是 10×1 one-hot 标签
    # validation_data: 10000 个 (x, y) 元组，y 是整数标签
    # test_data:        10000 个 (x, y) 元组，y 是整数标签
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # 创建网络: 784 个输入 → 30 个隐藏神经元 → 10 个输出
    # 784 = 28×28，对应 MNIST 图像的像素数
    # 10 对应数字 0~9
    net = Network([784,  10])

    # 用随机梯度下降训练:
    #   training_data    - 训练数据
    #   30               - 训练 30 个 epoch
    #   10               - 每个mini-batch 10 个样本
    #   3.0              - 学习率 η = 3.0
    #   test_data        - 每轮结束后用测试数据评估准确率
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
