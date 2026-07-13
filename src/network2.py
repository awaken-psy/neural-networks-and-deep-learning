"""network2.py
~~~~~~~~~~~~~~

network.py 的改进版本，实现了随机梯度下降学习算法的前馈神经网络。
改进包括添加了交叉熵代价函数、正则化以及更好的权重初始化。注意，
我主要关注代码的简洁性、可读性和易修改性。代码未经过优化，
并且省略了许多理想的功能。

"""

#### 库
# 标准库
import json
import random
import sys

# 第三方库
import numpy as np


#### 定义二次代价函数和交叉熵代价函数

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """返回输出 ``a`` 和期望输出 ``y`` 之间的代价。

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """返回输出层的误差 delta。"""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """返回输出 ``a`` 和期望输出 ``y`` 之间的代价。注意，
        使用 np.nan_to_num 来确保数值稳定性。特别是，如果 ``a`` 和
        ``y`` 在同一位置都为 1.0，则表达式 (1-y)*np.log(1-a) 会返回
        nan。np.nan_to_num 确保将其转换为正确的值（0.0）。

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """返回输出层的误差 delta。注意，参数 ``z`` 未被该方法使用。
        将其包含在方法参数中是为了使接口与其他代价类的 delta 方法
        保持一致。

        """
        return (a-y)


#### 主 Network 类
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """列表 ``sizes`` 包含网络各层的神经元数量。例如，如果列表
        为 [2, 3, 1]，则这是一个三层网络，第一层有 2 个神经元，
        第二层有 3 个神经元，第三层有 1 个神经元。网络的偏置和权重
        使用 ``self.default_weight_initializer`` 随机初始化（参见该
        方法的文档字符串）。

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        """使用均值为 0、标准差为 1 除以连接到同一神经元的权重数
        的平方根的高斯分布来初始化每个权重。使用均值为 0、标准差
        为 1 的高斯分布来初始化偏置。

        注意，第一层被假定为输入层，按惯例不为这些神经元设置偏置，
        因为偏置只在计算后续层的输出时使用。

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """使用均值为 0、标准差为 1 的高斯分布来初始化权重。
        使用均值为 0、标准差为 1 的高斯分布来初始化偏置。

        注意，第一层被假定为输入层，按惯例不为这些神经元设置偏置，
        因为偏置只在计算后续层的输出时使用。

        这种权重和偏置初始化方法使用了与第一章相同的方法，包含此
        方法是为了便于比较。通常使用默认权重初始化器会更好。

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """如果输入为 ``a``，返回网络的输出。"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """使用小批量随机梯度下降训练神经网络。``training_data`` 是
        一个包含元组 ``(x, y)`` 的列表，表示训练输入和期望输出。
        其他非可选参数的含义不言自明，正则化参数 ``lmbda`` 也是如此。
        该方法还接受 ``evaluation_data``，通常是验证数据或测试数据。
        我们可以通过设置相应标志来监控评估数据或训练数据上的代价和
        准确率。该方法返回一个包含四个列表的元组：评估数据上的（每
        epoch 的）代价、评估数据上的准确率、训练数据上的代价、以及
        训练数据上的准确率。所有值都在每个训练 epoch 结束时评估。因此，
        例如我们训练 30 个 epoch，则元组的第一个元素将是一个包含 30
        个元素的列表，表示每个 epoch 结束时评估数据上的代价。注意，
        如果相应标志未设置，则列表为空。

        """
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s 训练完成" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("训练数据上的代价: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("训练数据上的准确率: {} / {}".format(
                    accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("评估数据上的代价: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("评估数据上的准确率: {} / {}".format(
                    self.accuracy(evaluation_data), n_data))
            print
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """通过使用反向传播对一个迷你批次应用梯度下降来更新网络
        的权重和偏置。``mini_batch`` 是一个包含元组 ``(x, y)`` 的列表，
        ``eta`` 是学习率，``lmbda`` 是正则化参数，``n`` 是训练数据
        集的总大小。

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """返回一个元组 ``(nabla_b, nabla_w)``，表示代价函数 C_x 的
        梯度。``nabla_b`` 和 ``nabla_w`` 是逐层的 numpy 数组列表，
        类似于 ``self.biases`` 和 ``self.weights``。"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 前向传播
        activation = x
        activations = [x] # 存储各层激活值的列表
        zs = [] # 存储各层 z 向量的列表
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # 反向传播
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # 注意，下面循环中的变量 l 的使用方式与书中第二章的记号
        # 略有不同。这里 l = 1 表示最后一层神经元，l = 2 表示倒数
        # 第二层，以此类推。这是对书中编号方案的重新编号，利用了
        # Python 可以使用负索引的特性。
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """返回 ``data`` 中神经网络输出正确结果的输入数量。神经网络
        的输出被假定为最后一层中激活值最高的神经元的索引。

        标志 ``convert`` 在数据集是验证或测试数据（通常情况）时应设为
        False，在数据集是训练数据时应设为 True。需要此标志的原因是
        不同数据集中结果 ``y`` 的表示方式不同。特别是，它标志是否需要
        在不同表示之间转换。对不同数据集使用不同表示可能看起来很奇怪。
        为什么不对所有三个数据集使用相同的表示？这是出于效率原因 ——
        程序通常在训练数据上评估代价，在其他数据集上评估准确率。这些
        是不同类型的计算，使用不同的表示可以加快速度。有关表示的更多
        细节可以在 mnist_loader.load_data_wrapper 中找到。

        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """返回数据集 ``data`` 的总代价。标志 ``convert`` 在数据集
        是训练数据（通常情况）时应设为 False，在数据集是验证或测试
        数据时应设为 True。参见上面 ``accuracy`` 方法中类似但相反的
        约定说明。
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """将神经网络保存到文件 ``filename`` 中。"""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### 加载网络
def load(filename):
    """从文件 ``filename`` 加载神经网络。返回一个 Network 实例。

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### 杂项函数
def vectorized_result(j):
    """返回一个 10 维的单位向量，在第 j 个位置为 1.0，其余为零。
    用于将一个数字（0...9）转换为神经网络对应的期望输出。

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """sigmoid 函数。"""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """sigmoid 函数的导数。"""
    return sigmoid(z)*(1-sigmoid(z))
