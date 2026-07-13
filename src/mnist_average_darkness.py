"""
mnist_average_darkness
~~~~~~~~~~~~~~~~~~~~~~

一个朴素分类器，用于识别 MNIST 数据集中的手写数字。该程序根据图像的
平均暗度来分类数字 —— 思路是像 "1" 这样的数字通常比 "8" 这样的数字
更不暗，因为后者的形状更复杂。当给定一张图像时，分类器返回训练数据
中平均暗度最接近该图像的数字。

程序分两步工作：首先训练分类器，然后将分类器应用于 MNIST 测试数据，
查看有多少数字被正确分类。

不用说，这不是识别手写数字的好方法！不过它有助于展示朴素方法
能达到什么样的性能。"""

#### 库
# 标准库
from collections import defaultdict

# 本地库
import mnist_loader

def main():
    training_data, validation_data, test_data = mnist_loader.load_data()
    # 训练阶段：基于训练数据计算每个数字的平均暗度
    avgs = avg_darknesses(training_data)
    # 测试阶段：查看有多少测试图像被正确分类
    num_correct = sum(int(guess_digit(image, avgs) == digit)
                      for image, digit in zip(test_data[0], test_data[1]))
    print("使用图像平均暗度的基准分类器。")
    print("{0} / {1} 个正确。".format(num_correct, len(test_data[1])))

def avg_darknesses(training_data):
    """返回一个 defaultdict，键为 0 到 9 的数字。
    对每个数字，计算包含该数字的训练图像的平均暗度。
    单张图像的暗度就是所有像素暗度之和。"""
    digit_counts = defaultdict(int)
    darknesses = defaultdict(float)
    for image, digit in zip(training_data[0], training_data[1]):
        digit_counts[digit] += 1
        darknesses[digit] += sum(image)
    avgs = defaultdict(float)
    for digit, n in digit_counts.items():
        avgs[digit] = darknesses[digit] / n
    return avgs

def guess_digit(image, avgs):
    """返回在训练数据中平均暗度与 ``image`` 的暗度最接近的数字。
    注意，``avgs`` 是一个 defaultdict，键为 0...9，值为训练数据中
    对应数字的平均暗度。"""
    darkness = sum(image)
    distances = {k: abs(v-darkness) for k, v in avgs.items()}
    return min(distances, key=distances.get)

if __name__ == "__main__":
    main()
