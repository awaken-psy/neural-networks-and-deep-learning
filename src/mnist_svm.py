"""
mnist_svm
~~~~~~~~~

一个使用 SVM 分类器识别 MNIST 手写数字的分类程序。
"""

#### 库
# 本地库
import mnist_loader 

# 第三方库
from sklearn import svm

def svm_baseline():
    training_data, validation_data, test_data = mnist_loader.load_data()
    # 训练
    clf = svm.SVC()
    clf.fit(training_data[0], training_data[1])
    # 测试
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print("使用 SVM 的基准分类器。")
    print("{0} / {1} 个正确。".format(num_correct, len(test_data[1])))

if __name__ == "__main__":
    svm_baseline()
