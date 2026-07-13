"""
mnist_loader
~~~~~~~~~~~~

一个用于加载 MNIST 图像数据的库。有关返回数据结构的详细信息，
请参见 ``load_data`` 和 ``load_data_wrapper`` 的文档字符串。
在实际使用中，``load_data_wrapper`` 是我们的神经网络代码通常
调用的函数。

"""

#### 库
# 标准库
import pickle
import gzip
import os

# 获取本文件所在目录，再拼接 ../data/mnist.pkl.gz
# 这样无论从哪个目录运行都能找到数据文件
_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'mnist.pkl.gz')

# 第三方库
import numpy as np

def load_data():
    """以元组形式返回 MNIST 数据，包含训练数据、验证数据和测试数据。


    print(load_data_wrapper()[0])
    ``training_data`` 以包含两个条目的元组形式返回。
    第一个条目包含实际的训练图像。这是一个有 50,000 个条目的
    numpy ndarray。每个条目又是一个有 784 个值的 numpy ndarray，
    表示单张 MNIST 图像中 28 * 28 = 784 个像素。

    ``training_data`` 元组的第二个条目是一个包含 50,000 个条目的
    numpy ndarray。这些条目就是对应第一个条目中所包含图像的
    数字值（0...9）。

    ``validation_data`` 和 ``test_data`` 类似，只是各只包含
    10,000 张图像。

    这是一个不错的数据格式，但在神经网络中使用时，稍微修改一下
    ``training_data`` 的格式会更有帮助。这由包装函数
    ``load_data_wrapper()`` 完成，见下文。
    """
    f = gzip.open(_DATA_PATH, 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """返回一个包含 ``(training_data, validation_data, test_data)``
    的元组。基于 ``load_data``，但格式更适合我们的神经网络实现。

    特别地，``training_data`` 是一个包含 50,000 个二元组 ``(x, y)``
    的列表。``x`` 是一个 784 维的 numpy.ndarray，包含输入图像。
    ``y`` 是一个 10 维的 numpy.ndarray，表示对应 ``x`` 的正确数字
    的 one-hot 向量。

    ``validation_data`` 和 ``test_data`` 是包含 10,000 个二元组
    ``(x, y)`` 的列表。在每个二元组中，``x`` 是一个 784 维的
    numpy.ndarray，包含输入图像，``y`` 是对应的分类，即对应 ``x``
    的数字值（整数）。

    显然，这意味着训练数据与验证/测试数据使用了略有不同的格式。
    这些格式被证明是我们的神经网络代码中最方便使用的。"""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """返回一个 10 维的单位向量，在第 j 个位置为 1.0，其余为零。
    用于将一个数字（0...9）转换为神经网络对应的期望输出。"""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
