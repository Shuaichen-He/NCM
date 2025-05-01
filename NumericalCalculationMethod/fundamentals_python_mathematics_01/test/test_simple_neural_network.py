# -*- coding: UTF-8 -*-
"""
@file_name: test_simple_neural_network.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from fundamentals_python_mathematics_01.simple_neural_network import SimpleNeuralNetwork
import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(14, 9))
plt.subplot(221)
iris = load_iris()  # 鸢尾花为三分类数据，共150个样本，每类样本50个，四个特征
X_iris, y_iris = iris.data[:100], iris.target[:100]  # 选取前100个二分类的数据训练模型
X_iris = StandardScaler().fit_transform(X_iris)  # 数据的标准化处理
y_iris = LabelEncoder().fit_transform(y_iris)  # 类别数据的编码，针对文本标签类别
# 划分70%样本作为训练集，30%的样本作为测试集，且采用分层抽样
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.3, random_state=0, stratify=y_iris)
snn = SimpleNeuralNetwork(alpha=0.05, epochs=10000, eps=1e-3)  # 构建神经网络
snn.fit_net(X_train, y_train)  # 神经网络训练
print(snn.nn_weight)
y_pred = snn.predict(X_test)  # 神经网络预测
snn.plt_loss_curve(is_show=False, title_txt="Iris")  # 可视化平方和损失曲线
print(classification_report(y_test, y_pred))  # 打印测试样本的分类报告

plt.subplot(222)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
ticks = ["Setosa", "Versicolor"]
sns.heatmap(cm, annot=True, cmap="GnBu", xticklabels=ticks, yticklabels=ticks,
            cbar=False, annot_kws={"fontsize": 18})
plt.title("$Iris$混淆矩阵（$Acc=%.5f$）" % acc, fontsize=18)
plt.xticks(font="Times New Roman")
plt.yticks(font="Times New Roman")
plt.tick_params(labelsize=20)

plt.subplot(223)
cancer = load_breast_cancer()  #  乳腺癌二分类数据，569个样本，30个特征
X, y = cancer.data, cancer.target
X = StandardScaler().fit_transform(X)  # 数据的标准化处理
y = LabelEncoder().fit_transform(y)  # 类别数据的编码，针对文本标签类别
# 划分70%样本作为训练集，30%的样本作为测试集，且采用分层抽样
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
snn = SimpleNeuralNetwork(alpha=0.05, epochs=10000, eps=1e-3)  # 构建神经网络
snn.fit_net(X_train, y_train)  # 神经网络训练
print(snn.nn_weight)
y_pred = snn.predict(X_test)  # 神经网络预测
snn.plt_loss_curve(is_show=False, title_txt="Breast \ Cancer")  # 可视化平方和损失曲线
print(classification_report(y_test, y_pred))  # 打印测试样本的分类报告

plt.subplot(224)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
ticks = ["malignant", "benign"]
sns.heatmap(cm, annot=True, cmap="GnBu", xticklabels=ticks, yticklabels=ticks,
            cbar=False, annot_kws={"fontsize": 18})
plt.title("$Breast \  Cancer$混淆矩阵（$Acc=%.5f$）" % acc, fontsize=18)
plt.xticks(font="Times New Roman")
plt.yticks(font="Times New Roman")
plt.tick_params(labelsize=20)

plt.show()