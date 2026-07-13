"""expand_mnist.py
~~~~~~~~~~~~~~~~~~

取 50,000 张 MNIST 训练图像，通过将每张训练图像上、下、左、右各
平移一个像素，创建一个包含 250,000 张图像的扩展数据集。将结果
保存到 ../data/mnist_expanded.pkl.gz。

注意，此程序内存消耗较大，在小内存系统上可能无法运行。

"""

from __future__ import print_function

#### 库

# 标准库
import pickle
import gzip
import os
import os.path
import random

# 第三方库
import numpy as np

# 获取本文件所在目录，拼接数据路径
_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_DIR, '..', 'data')
_MNIST_PATH = os.path.join(_DATA_DIR, 'mnist.pkl.gz')
_EXPANDED_PATH = os.path.join(_DATA_DIR, 'mnist_expanded.pkl.gz')

print("正在扩展 MNIST 训练集")

if os.path.exists(_EXPANDED_PATH):
    print("扩展训练集已存在。退出。")
else:
    f = gzip.open(_MNIST_PATH, 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    expanded_training_pairs = []
    j = 0 # 计数器
    for x, y in zip(training_data[0], training_data[1]):
        expanded_training_pairs.append((x, y))
        image = np.reshape(x, (-1, 28))
        j += 1
        if j % 1000 == 0: print("正在扩展第", j, "张图像")
        # 遍历位移的详细信息
        for d, axis, index_position, index in [
                (1,  0, "first", 0),
                (-1, 0, "first", 27),
                (1,  1, "last",  0),
                (-1, 1, "last",  27)]:
            new_img = np.roll(image, d, axis)
            if index_position == "first": 
                new_img[index, :] = np.zeros(28)
            else: 
                new_img[:, index] = np.zeros(28)
            expanded_training_pairs.append((np.reshape(new_img, 784), y))
    random.shuffle(expanded_training_pairs)
    expanded_training_data = [list(d) for d in zip(*expanded_training_pairs)]
    print("正在保存扩展数据。这可能需要几分钟。")
    f = gzip.open(_EXPANDED_PATH, "w")
    pickle.dump((expanded_training_data, validation_data, test_data), f)
    f.close()
