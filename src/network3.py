"""
代码来自 https://github.com/MichalDanielDobrzanski/DeepLearningPython/pull/14/
"""

"""network3.py
~~~~~~~~~~~~~~
基于 Theano 的程序，用于训练和运行简单的神经网络。
支持多种层类型（全连接层、卷积层、最大池化层、softmax 层）和
激活函数（sigmoid、tanh 和修正线性单元，且易于添加更多）。
在 CPU 上运行时，此程序比 network.py 和 network2.py 快得多。
但与 network.py 和 network2.py 不同，它还可以在 GPU 上运行，
从而更快。
因为代码基于 Theano，所以在许多方面与 network.py 和 network2.py
不同。不过，在可能的地方我尽量与之前的程序保持一致。特别是，
API 与 network2.py 类似。注意，我主要关注代码的简洁性、可读性
和易修改性。代码未经过优化，并且省略了许多理想的功能。
此程序融合了 Theano 卷积神经网络文档（特别是
http://deeplearning.net/tutorial/lenet.html ）、Misha Denil 的
dropout 实现（https://github.com/mdenil/dropout ）以及
Chris Olah（http://colah.github.io ）的思想。
"""

#### 库
# 标准库
import pickle
import gzip

import os

# 第三方库
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal.pool import pool_2d

# 神经元的激活函数
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


#### 常量
GPU = True
if GPU:
    print("尝试在 GPU 上运行。如果不希望如此，请修改 "+\
        "network3.py\n将 GPU 标志设为 False。")
    try: theano.config.device = 'gpu'
    except: pass # 已经设置过了
    theano.config.floatX = 'float32'
else:
    print("在 CPU 上运行。如果不希望如此，请修改 "+\
        "network3.py\n将 GPU 标志设为 True。")

#### 加载 MNIST 数据
def load_data_shared(filename=None):
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__), '..', 'data', 'mnist.pkl.gz')
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    def shared(data):
        """将数据放入共享变量中。这允许 Theano 在有 GPU 时
        将数据复制到 GPU 上。
        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]

#### 用于构建和训练网络的主类
class Network(object):

    def __init__(self, layers, mini_batch_size):
        """接受一个 ``layers`` 列表，描述网络架构，以及一个用于训练时
        随机梯度下降的 ``mini_batch_size`` 值。
        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)): # xrange() 在 Python 3 中被重命名为 range()
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """使用小批量随机梯度下降训练网络。"""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # 计算训练、验证和测试的迷你批次数量
        num_training_batches = int(size(training_data)/mini_batch_size)
        num_validation_batches = int(size(validation_data)/mini_batch_size)
        num_test_batches = int(size(test_data)/mini_batch_size)

        # 定义（正则化的）代价函数、符号梯度和更新规则
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]

        # 定义训练迷你批次的函数，以及计算验证和测试迷你批次准确率的函数
        i = T.lscalar() # 迷你批次索引
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        # 实际训练过程
        best_validation_accuracy = 0.0
        for epoch in range(epochs):
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0:
                    print("正在训练第 {0} 个迷你批次".format(iteration))
                cost_ij = train_mb(minibatch_index)
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in range(num_validation_batches)])
                    print("Epoch {0}: 验证准确率 {1:.2%}".format(
                        epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("这是迄今为最好的验证准确率。")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in range(num_test_batches)])
                            print('对应的测试准确率为 {0:.2%}'.format(
                                test_accuracy))
        print("网络训练完成。")
        print("在迭代 {1} 处获得最佳验证准确率 {0:.2%}".format(
            best_validation_accuracy, best_iteration))
        print("对应的测试准确率为 {0:.2%}".format(test_accuracy))

#### 定义层类型

class ConvPoolLayer(object):
    """用于创建卷积和最大池化层的组合。更复杂的实现会将两者分开，
    但在我们的应用中总是将它们一起使用，且合并可以简化代码，
    所以将它们合并是有意义的。
    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):
        """``filter_shape`` 是长度为 4 的元组，其元素分别为滤波器数量、
        输入特征图数量、滤波器高度和滤波器宽度。
        ``image_shape`` 是长度为 4 的元组，其元素分别为迷你批次大小、
        输入特征图数量、图像高度和图像宽度。
        ``poolsize`` 是长度为 2 的元组，其元素分别为 y 和 x 方向的
        池化大小。
        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        # 初始化权重和偏置
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape)
        pooled_out = pool_2d(
            input=conv_out, ws=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # 卷积层中没有 dropout

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # 初始化权重和偏置
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "返回迷你批次的准确率。"
        return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # 初始化权重和偏置
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "返回对数似然代价。"
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "返回迷你批次的准确率。"
        return T.mean(T.eq(y, self.y_out))


#### 杂项
def size(data):
    "返回数据集 ``data`` 的大小。"
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
